"""Phase 2/D3-D5: secret redaction, failure classification, bounded retry, and
failure-always provenance. Network-free; no pydantic_ai needed (only pydantic for records).
Skips when the optional ``pydantic`` extra is absent.
"""
from __future__ import annotations

import datetime as dt
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _spec():
    return types.SimpleNamespace(name="judge", prompt="You are a judge.", result_contract="JudgeVote")


class _Clock:
    def __init__(self):
        self.t = dt.datetime(2026, 8, 7, tzinfo=dt.timezone.utc)

    def __call__(self):
        self.t = self.t + dt.timedelta(seconds=1)
        return self.t


class _Caller:
    """Replays a scripted sequence of behaviors: an Exception is raised, a tuple is returned."""
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def __call__(self):
        b = self.behaviors[self.calls]
        self.calls += 1
        if isinstance(b, BaseException):
            raise b
        return b


def _rate_limit_exc():
    exc = Exception("rate limited")
    exc.status_code = 429
    return exc


def _server_exc():
    exc = Exception("service unavailable")
    exc.status_code = 503
    return exc


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class RedactionTests(unittest.TestCase):
    def test_masks_common_secret_shapes(self):
        from runtimes.pydantic_ai.redaction import redact
        s = "key=sk-abcdef0123456789ABCDEF here; Authorization: Bearer abcdefgh12345678"
        out = redact(s)
        self.assertNotIn("sk-abcdef0123456789ABCDEF", out)
        self.assertNotIn("Bearer abcdefgh12345678", out)
        self.assertIn("[REDACTED]", out)

    def test_masks_exact_extra_secret(self):
        from runtimes.pydantic_ai.redaction import redact
        out = redact("failed with token MY-SUPER-SECRET-VALUE", extra_secrets=["MY-SUPER-SECRET-VALUE"])
        self.assertNotIn("MY-SUPER-SECRET-VALUE", out)

    def test_leaves_ordinary_text(self):
        from runtimes.pydantic_ai.redaction import redact
        self.assertEqual(redact("student-teacher force MAE 0.18 eV/A"),
                         "student-teacher force MAE 0.18 eV/A")

    def test_secrets_from_env_collects_credentialish(self):
        from runtimes.pydantic_ai.redaction import secrets_from_env
        vals = secrets_from_env({"ANTHROPIC_API_KEY": "sk-x", "MY_TOKEN": "abc", "PATH": "/usr/bin"})
        self.assertIn("sk-x", vals)
        self.assertIn("abc", vals)
        self.assertNotIn("/usr/bin", vals)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class FailureClassificationTests(unittest.TestCase):
    def test_categories_and_retryability(self):
        from runtimes.pydantic_ai.failures import classify_failure
        cases = [
            (Exception("invalid api key"), "authentication_failure", False),
            (TimeoutError("timed out"), "timeout", True),
            (_rate_limit_exc(), "rate_limit", True),
            (_server_exc(), "provider_internal_failure", True),
            (ConnectionError("connection refused"), "provider_network_failure", True),
        ]
        for exc, cat, retry in cases:
            gc, gr = classify_failure(exc)
            self.assertEqual(gc, cat, exc)
            self.assertEqual(gr, retry, exc)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class RetryAndProvenanceTests(unittest.TestCase):
    def _ctx(self, **kw):
        from runtimes.pydantic_ai.models import RuntimeContext
        base = dict(exchange_dir=tempfile.mkdtemp(), repo_root=str(ROOT),
                    provider="anthropic", model_id="anthropic:claude-x",
                    backoff_base_s=0.01, correlation_id="corr-1")
        base.update(kw)
        return RuntimeContext(**base)

    def _run(self, behaviors, ctx):
        from runtimes.pydantic_ai.interface import execute_with_retry, RUNTIME_VERSION
        from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset
        sleeps = []
        caller = _Caller(behaviors)
        inv = execute_with_retry(
            call=caller, task={"task_id": "t1", "agent": "judge", "inputs": []},
            spec=_spec(), context=ctx, toolset=ReadOnlyToolset([]),
            usage_source="provider", runtime_version=RUNTIME_VERSION,
            clock=_Clock(), sleep=lambda s: sleeps.append(s))
        return inv, caller, sleeps

    def test_success_first_try(self):
        inv, caller, sleeps = self._run([('{"ok": true}', 5, 3)], self._ctx())
        self.assertEqual(caller.calls, 1)
        self.assertEqual(inv.candidate, {"ok": True})
        self.assertEqual(inv.prior_attempts, [])
        self.assertEqual(inv.provenance.failure_category, "")
        self.assertGreater(inv.provenance.latency_s, 0)
        self.assertEqual(inv.provenance.correlation_id, "corr-1")

    def test_retryable_then_success(self):
        ctx = self._ctx(provider_retries=2, max_total_calls=3)
        inv, caller, sleeps = self._run([_rate_limit_exc(), ('{"ok": 1}', 1, 1)], ctx)
        self.assertEqual(caller.calls, 2)
        self.assertEqual(len(inv.prior_attempts), 1)
        self.assertEqual(inv.prior_attempts[0].failure_category, "rate_limit")
        # retry lineage: the success attempt links to the failed one
        self.assertEqual(inv.provenance.parent_attempt_id, inv.prior_attempts[0].attempt_id)
        self.assertEqual(len(sleeps), 1)  # backed off once

    def test_non_retryable_is_terminal_first_attempt(self):
        ctx = self._ctx(provider_retries=3, max_total_calls=4)
        auth = Exception("Authentication failed: invalid x-api-key sk-abcdef0123456789ABCDEF")
        inv, caller, sleeps = self._run([auth, ('{"ok": 1}', 1, 1)], ctx)
        self.assertEqual(caller.calls, 1)  # NOT retried
        self.assertEqual(inv.candidate, {})
        self.assertEqual(inv.provenance.failure_category, "authentication_failure")
        self.assertFalse(inv.provenance.retryable)
        self.assertEqual(sleeps, [])
        # the secret in the exception text is redacted in provenance
        self.assertNotIn("sk-abcdef0123456789ABCDEF", inv.provenance.exception_message)
        self.assertIn("[REDACTED]", inv.provenance.exception_message)

    def test_retryable_exhausts_and_is_preserved(self):
        ctx = self._ctx(provider_retries=2, max_total_calls=5)
        inv, caller, sleeps = self._run([_server_exc(), _server_exc(), _server_exc()], ctx)
        self.assertEqual(caller.calls, 3)             # provider_retries+1
        self.assertEqual(len(inv.prior_attempts), 2)  # first two preserved as priors
        self.assertEqual(inv.provenance.failure_category, "provider_internal_failure")
        self.assertEqual(inv.candidate, {})

    def test_max_total_calls_is_a_hard_cost_cap(self):
        ctx = self._ctx(provider_retries=5, max_total_calls=2)  # cap below retries
        inv, caller, sleeps = self._run([_server_exc()] * 6, ctx)
        self.assertEqual(caller.calls, 2)  # capped at max_total_calls

    def test_failure_provenance_is_always_written_by_driver(self):
        from runtimes.pydantic_ai.driver import run_task
        from runtimes.pydantic_ai.interface import AgentInvocation
        ctx = self._ctx(provider_retries=0, max_total_calls=1)

        class FailingRuntime:
            def run(self, task, spec, context):
                from runtimes.pydantic_ai.interface import build_failure_record
                from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset
                rec = build_failure_record(
                    task=task, spec=spec, context=context, toolset=ReadOnlyToolset([]),
                    exc=TimeoutError("timed out"), usage_source="provider",
                    runtime_version="x", attempt_id="a1", parent_attempt_id=None,
                    started_at="s", finished_at="f", latency_s=1.0)
                return AgentInvocation(candidate={}, provenance=rec)

        res = run_task(FailingRuntime(), {"task_id": "t1", "agent": "judge", "inputs": []},
                       _spec(), ctx, shadow=True)
        self.assertFalse(res.accepted)
        self.assertTrue(res.provenance_path.exists())  # provenance persisted despite failure
        self.assertEqual(res.invocation.provenance.failure_category, "timeout")
        self.assertFalse(res.invocation.provenance.controller_mutated)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ProviderConfigurationTests(unittest.TestCase):
    def test_valid_and_extra_rejected(self):
        import pydantic
        from runtimes.pydantic_ai.models import ProviderConfiguration
        cfg = ProviderConfiguration(provider="anthropic", model_id="anthropic:claude-x")
        self.assertEqual(cfg.usage_source, "provider")
        self.assertGreaterEqual(cfg.provider_retries, 0)
        with self.assertRaises(pydantic.ValidationError):
            ProviderConfiguration(provider="anthropic", model_id="m", surprise=1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
