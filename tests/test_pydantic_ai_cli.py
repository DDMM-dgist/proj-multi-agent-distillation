"""Phase 2/D3+D6: provider credential preflight and the runtime CLI (modes + exit codes).

Network-free. The pydantic-ai CLI path never contacts a provider here (no credential ->
PROVIDER_UNAVAILABLE). The mock runtime path exercises the full validate/accept pipeline.
Skips when the optional ``pydantic`` extra is absent.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SPECS = str(ROOT / "agent_specs")

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _judge_task(task_id="jt1"):
    return {
        "schema_version": 1, "task_id": task_id, "agent": "judge",
        "created_at": "2026-08-07T00:00:00Z", "instruction": "review the evidence",
        "inputs": [], "criteria": ["c1"], "constraints": [],
        "context": {"review_lens": "evidence_provenance", "review_focus": "provenance"},
    }


def _judge_vote(lens="evidence_provenance", verdict="PASS"):
    return {"review_lens": lens, "verdict": verdict,
            "criteria_checked": [{"criterion": "c1", "value_read": "x", "ok": verdict == "PASS"}],
            "rationale": "checked", "required_fix": "" if verdict == "PASS" else "fix it"}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class PreflightTests(unittest.TestCase):
    def test_not_configured_without_model(self):
        from runtimes.pydantic_ai.provider import preflight_credentials
        r = preflight_credentials(env={})
        self.assertEqual(r.status, "NOT_CONFIGURED")

    def test_skipped_when_model_but_no_key(self):
        from runtimes.pydantic_ai.provider import preflight_credentials
        r = preflight_credentials(env={"PYDANTIC_AI_MODEL": "anthropic:claude-x"})
        self.assertEqual(r.status, "SKIPPED")
        self.assertFalse(r.key_present)
        self.assertEqual(r.provider, "anthropic")

    def test_blocked_on_unknown_provider(self):
        from runtimes.pydantic_ai.provider import preflight_credentials
        r = preflight_credentials(env={"PYDANTIC_AI_MODEL": "acme:model", "ACME_KEY": "x"})
        self.assertEqual(r.status, "BLOCKED")

    def test_ready_requires_key_and_sdk(self):
        from runtimes.pydantic_ai import provider
        env = {"PYDANTIC_AI_MODEL": "anthropic:claude-x", "ANTHROPIC_API_KEY": "sk-fake"}
        with mock.patch.object(provider, "_sdk_available", return_value=True):
            r = provider.preflight_credentials(env=env)
        self.assertEqual(r.status, "READY")
        self.assertTrue(r.key_present and r.sdk_present)

    def test_blocked_when_key_present_but_sdk_missing(self):
        from runtimes.pydantic_ai import provider
        env = {"PYDANTIC_AI_MODEL": "anthropic:claude-x", "ANTHROPIC_API_KEY": "sk-fake"}
        with mock.patch.object(provider, "_sdk_available", return_value=False):
            r = provider.preflight_credentials(env=env)
        self.assertEqual(r.status, "BLOCKED")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class CliTests(unittest.TestCase):
    def _write(self, d: Path, name: str, obj) -> Path:
        p = d / name
        p.write_text(json.dumps(obj))
        return p

    def test_mock_validate_only_success_no_acceptance(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            task = self._write(d, "task.json", _judge_task())
            resp = self._write(d, "resp.json", _judge_vote())
            ex = d / "exchange"
            code = cli.main(["run-task", "--runtime", "mock", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(task),
                             "--exchange-dir", str(ex), "--mock-response", str(resp),
                             "--mode", "validate-only"])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            # validate-only never accepts:
            self.assertFalse((ex / "results" / "jt1.json").exists())
            # provenance is always written:
            self.assertTrue(list((ex / "provenance").glob("jt1.*.json")))

    def test_mock_wrong_lens_rejected(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            task = self._write(d, "task.json", _judge_task())
            resp = self._write(d, "resp.json", _judge_vote(lens="scientific_validity"))
            code = cli.main(["run-task", "--runtime", "mock", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(task),
                             "--exchange-dir", str(d / "ex"), "--mock-response", str(resp),
                             "--mode", "validate-only"])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)

    def test_mock_primary_accepts_after_dispatch(self):
        from runtimes.pydantic_ai import cli
        from orchestration.exchange import FileExchangeRuntime
        from orchestration.specs import load_agent_specs
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ex = d / "exchange"
            spec = load_agent_specs(SPECS)["judge"]
            task = _judge_task()
            FileExchangeRuntime(str(ex)).dispatch(spec, task)  # primary accept needs a dispatched task
            tpath = self._write(d, "task.json", task)
            resp = self._write(d, "resp.json", _judge_vote())
            code = cli.main(["run-task", "--runtime", "mock", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(tpath),
                             "--exchange-dir", str(ex), "--mock-response", str(resp),
                             "--mode", "primary"])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            self.assertTrue((ex / "results" / "jt1.json").exists())

    def test_pydantic_ai_without_credentials_is_provider_unavailable(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {}, clear=True):
            d = Path(tmp)
            task = self._write(d, "task.json", _judge_task())
            code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(task),
                             "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            self.assertEqual(code, cli.EXIT_PROVIDER_UNAVAILABLE)

    def test_unknown_agent_blocked(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            task = self._write(d, "task.json", _judge_task())
            resp = self._write(d, "resp.json", _judge_vote())
            code = cli.main(["run-task", "--runtime", "mock", "--agent", "nope",
                             "--agent-specs-dir", SPECS, "--task", str(task),
                             "--exchange-dir", str(d / "ex"), "--mock-response", str(resp)])
            self.assertEqual(code, cli.EXIT_BLOCKED_POLICY)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
