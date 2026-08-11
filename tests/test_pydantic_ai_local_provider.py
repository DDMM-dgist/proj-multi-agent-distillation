"""Phase L1/L2: local (OpenAI-compatible / vLLM / Ollama) provider abstraction + fail-closed
local preflight. Network-free — NOTHING here launches a server or runs inference; the only socket
use is a bounded TCP connect to a port the test itself opens/closes. No Anthropic credential is
required or read on the local path.

Skips the constructibility/build cases when the ``openai`` SDK (the ``local-openai`` extra) is
absent; the pure-config preflight and selection cases run without it.
"""
from __future__ import annotations

import json
import socket
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

try:
    import openai  # noqa: F401
    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    _HAS_OPENAI = False


def _closed_port() -> int:
    """A port number with nothing listening (bind+close to reserve then release)."""
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class SelectProviderKindTests(unittest.TestCase):
    def test_explicit_provider_wins(self):
        from runtimes.pydantic_ai.provider import select_provider_kind
        self.assertEqual(select_provider_kind({"PYDANTIC_AI_PROVIDER": "local-openai"}), "local-openai")
        self.assertEqual(select_provider_kind({"PYDANTIC_AI_PROVIDER": "Ollama"}), "ollama")

    def test_anthropic_inferred_from_model_for_backcompat(self):
        from runtimes.pydantic_ai.provider import select_provider_kind
        self.assertEqual(select_provider_kind({"PYDANTIC_AI_MODEL": "anthropic:claude-x"}), "anthropic")

    def test_local_is_never_inferred_from_model_string(self):
        # An ollama-style id "qwen2.5:7b" must NOT be misread as a provider:model; local needs
        # explicit selection.
        from runtimes.pydantic_ai.provider import select_provider_kind
        self.assertEqual(select_provider_kind({"PYDANTIC_AI_MODEL": "qwen2.5:7b"}), "")

    def test_not_configured_is_empty(self):
        from runtimes.pydantic_ai.provider import select_provider_kind
        self.assertEqual(select_provider_kind({}), "")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class LocalPreflightConfigTests(unittest.TestCase):
    """Config-shape checks that need no openai SDK."""

    def test_not_selected_when_kind_is_not_local(self):
        from runtimes.pydantic_ai.provider import preflight_local, LOCAL_NOT_SELECTED
        r = preflight_local({"PYDANTIC_AI_MODEL": "anthropic:claude-x"})
        self.assertEqual(r.status, LOCAL_NOT_SELECTED)

    def test_model_not_configured(self):
        from runtimes.pydantic_ai.provider import preflight_local, LOCAL_MODEL_NOT_CONFIGURED
        r = preflight_local({"PYDANTIC_AI_PROVIDER": "local-openai"})
        self.assertEqual(r.status, LOCAL_MODEL_NOT_CONFIGURED)
        self.assertFalse(r.anthropic_key_required)

    def test_base_url_not_configured(self):
        from runtimes.pydantic_ai.provider import preflight_local, LOCAL_BASE_URL_NOT_CONFIGURED
        r = preflight_local({"PYDANTIC_AI_PROVIDER": "local-openai", "PYDANTIC_AI_MODEL": "m"})
        self.assertEqual(r.status, LOCAL_BASE_URL_NOT_CONFIGURED)

    def test_sdk_not_installed_reported(self):
        from runtimes.pydantic_ai import provider
        env = {"PYDANTIC_AI_PROVIDER": "local-openai", "PYDANTIC_AI_MODEL": "m",
               "PYDANTIC_AI_BASE_URL": "http://127.0.0.1:8000/v1"}
        with mock.patch.object(provider, "_openai_sdk_available", return_value=False):
            r = provider.preflight_local(env)
        self.assertEqual(r.status, provider.LOCAL_SDK_NOT_INSTALLED)
        self.assertFalse(r.sdk_present)

    def test_no_anthropic_credential_dependency(self):
        # The local preflight must never require/read an Anthropic key. Even with a bogus
        # ANTHROPIC_API_KEY absent, local config is judged on its own terms.
        from runtimes.pydantic_ai import provider
        env = {"PYDANTIC_AI_PROVIDER": "local-openai", "PYDANTIC_AI_MODEL": "m",
               "PYDANTIC_AI_BASE_URL": "http://127.0.0.1:8000/v1"}
        with mock.patch.object(provider, "_openai_sdk_available", return_value=False):
            r = provider.preflight_local(env)
        self.assertFalse(r.anthropic_key_required)
        self.assertEqual(r.max_total_calls, 1)  # bounded-call guard preserved

    def test_reachability_probe_is_pure_tcp(self):
        # _server_reachable is a plain TCP connect: False for a closed port, True for one we open.
        from runtimes.pydantic_ai.provider import _server_reachable
        self.assertFalse(_server_reachable(f"http://127.0.0.1:{_closed_port()}/v1", 0.3))
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
        try:
            port = srv.getsockname()[1]
            self.assertTrue(_server_reachable(f"http://127.0.0.1:{port}/v1", 0.5))
        finally:
            srv.close()


@unittest.skipUnless(_HAS_PYDANTIC and _HAS_OPENAI, "local-openai extra (openai SDK) not installed")
class LocalPreflightConstructTests(unittest.TestCase):
    """Constructibility + probe status — need the openai SDK, still no inference."""

    def _env(self, base_url, kind="local-openai"):
        return {"PYDANTIC_AI_PROVIDER": kind, "PYDANTIC_AI_MODEL": "smoke-model",
                "PYDANTIC_AI_BASE_URL": base_url}

    def test_ready_without_probe(self):
        from runtimes.pydantic_ai.provider import preflight_local, LOCAL_READY
        r = preflight_local(self._env("http://127.0.0.1:8000/v1"))
        self.assertEqual(r.status, LOCAL_READY)
        self.assertTrue(r.sdk_present and r.constructible)
        self.assertFalse(r.server_probed)
        self.assertFalse(r.anthropic_key_required)

    def test_not_running_when_probed_and_server_absent(self):
        from runtimes.pydantic_ai.provider import preflight_local, LOCAL_NOT_RUNNING
        r = preflight_local(self._env(f"http://127.0.0.1:{_closed_port()}/v1"),
                            probe=True, connect_timeout_s=0.3)
        self.assertEqual(r.status, LOCAL_NOT_RUNNING)   # operational, not a runtime failure
        self.assertTrue(r.server_probed)
        self.assertFalse(r.server_reachable)

    def test_ready_when_probed_and_server_reachable(self):
        from runtimes.pydantic_ai.provider import preflight_local, LOCAL_READY
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
        try:
            port = srv.getsockname()[1]
            r = preflight_local(self._env(f"http://127.0.0.1:{port}/v1"), probe=True)
            self.assertEqual(r.status, LOCAL_READY)
            self.assertTrue(r.server_reachable)
        finally:
            srv.close()

    def test_build_local_model_uses_placeholder_key_not_a_secret(self):
        from runtimes.pydantic_ai.provider import build_local_model
        m = build_local_model("local-openai", "smoke-model", "http://127.0.0.1:8000/v1")
        self.assertEqual(m.system, "openai")
        self.assertIn("127.0.0.1:8000", m.base_url)
        self.assertEqual(m.client.api_key, "api-key-not-set")  # non-secret local placeholder

    def test_build_local_model_ollama(self):
        from runtimes.pydantic_ai.provider import build_local_model
        m = build_local_model("ollama", "qwen2.5:7b", "http://127.0.0.1:11434/v1")
        self.assertEqual(m.system, "ollama")
        self.assertEqual(m.client.api_key, "api-key-not-set")

    def test_agent_builds_offline_with_local_model_tools_and_typed_output(self):
        # The exact runtime shape: an Agent with a typed output_type + a read-only tool, bound to
        # a local model, builds WITHOUT any network call.
        from pydantic_ai import Agent
        from runtimes.pydantic_ai.provider import build_local_model
        from runtimes.pydantic_ai.models import JudgeVoteModel
        agent = Agent(build_local_model("local-openai", "m", "http://127.0.0.1:8000/v1"),
                      output_type=JudgeVoteModel, system_prompt="judge")

        @agent.tool_plain
        def read_json(path: str) -> dict:  # noqa: ARG001
            return {}
        self.assertIsNotNone(agent)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class LocalRedactionTests(unittest.TestCase):
    def test_real_openai_key_redacted_placeholder_is_not(self):
        from runtimes.pydantic_ai.redaction import redact, secrets_from_env
        env = {"OPENAI_API_KEY": "sk-realsecretvalue1234567890abcd"}
        secrets = secrets_from_env(env)
        red = redact("connecting with sk-realsecretvalue1234567890abcd", secrets)
        self.assertNotIn("realsecretvalue", red)
        # the local placeholder is public and must NOT be flagged/masked as a secret
        self.assertNotIn("api-key-not-set", secrets_from_env({}))
        self.assertEqual(redact("api-key-not-set"), "api-key-not-set")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class LocalCliRoutingTests(unittest.TestCase):
    def _task(self, d: Path):
        t = {"schema_version": 1, "task_id": "lt1", "agent": "judge",
             "created_at": "2026-08-07T00:00:00Z", "instruction": "review",
             "inputs": [], "criteria": ["c1"], "constraints": [],
             "context": {"review_lens": "evidence_provenance", "review_focus": "f"}}
        p = d / "task.json"; p.write_text(json.dumps(t)); return p

    def test_local_without_base_url_is_provider_unavailable_no_anthropic_key(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"PYDANTIC_AI_PROVIDER": "local-openai",
                                               "PYDANTIC_AI_MODEL": "m"}, clear=True):
            d = Path(tmp)
            code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(self._task(d)),
                             "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            self.assertEqual(code, cli.EXIT_PROVIDER_UNAVAILABLE)  # missing base url, not a key

    @unittest.skipUnless(_HAS_OPENAI, "local-openai extra not installed")
    def test_local_ready_but_unconfirmed_requires_approval_and_builds_no_runtime(self):
        from runtimes.pydantic_ai import cli
        built = {"runtime": False}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ",
                                {"PYDANTIC_AI_PROVIDER": "local-openai", "PYDANTIC_AI_MODEL": "m",
                                 "PYDANTIC_AI_BASE_URL": "http://127.0.0.1:8000/v1"}, clear=True):
            d = Path(tmp)
            with mock.patch("runtimes.pydantic_ai.pydantic_ai_runtime.PydanticAIRuntime",
                            side_effect=lambda *a, **k: built.__setitem__("runtime", True)):
                code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                                 "--agent-specs-dir", SPECS, "--task", str(self._task(d)),
                                 "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            self.assertFalse(built["runtime"])  # no runtime/model constructed without confirmation

    @unittest.skipUnless(_HAS_OPENAI, "local-openai extra not installed")
    def test_local_not_running_is_operational_status(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ",
                                {"PYDANTIC_AI_PROVIDER": "local-openai", "PYDANTIC_AI_MODEL": "m",
                                 "PYDANTIC_AI_BASE_URL": f"http://127.0.0.1:{_closed_port()}/v1",
                                 "PYDANTIC_AI_SMOKE_CONFIRM": "yes"}, clear=True):
            d = Path(tmp)
            code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(self._task(d)),
                             "--exchange-dir", str(d / "ex"), "--mode", "shadow", "--probe-server"])
            # server absent -> operational PROVIDER_UNAVAILABLE, never a scientific/runtime failure
            self.assertEqual(code, cli.EXIT_PROVIDER_UNAVAILABLE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
