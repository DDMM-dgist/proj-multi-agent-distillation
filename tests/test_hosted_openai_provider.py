"""Hosted OpenAI as a first-class production provider (extension of Phase 2/D3's provider
layer). Confirms ``PYDANTIC_AI_PROVIDER=openai`` is a real, working choice through the actual
CLI/runtime dispatch -- not just an entry in the credential-env lookup table -- while
local-openai/ollama/anthropic keep their exact prior behavior. Network-free throughout: every
test either inspects preflight-only functions (no network) or replaces
``PydanticAIRuntime``/``build_provider_model`` with mocks, so no real HTTP/socket call is ever
made. A socket guard (``_no_sockets``) makes any accidental network attempt fail loudly.
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


def _judge_task(task_id="jt1"):
    return {
        "schema_version": 1, "task_id": task_id, "agent": "judge",
        "created_at": "2026-08-15T00:00:00Z", "instruction": "review the evidence",
        "inputs": [], "criteria": ["c1"], "constraints": [],
        "context": {"review_lens": "evidence_provenance", "review_focus": "provenance"},
    }


class _NoSockets:
    """Context manager that fails any test attempting a real TCP connection."""
    def __enter__(self):
        self._orig = socket.socket.connect

        def _blocked(*_a, **_k):
            raise AssertionError("no real network connection is permitted in this test")
        socket.socket.connect = _blocked
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._orig


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class HostedOpenAIProviderKindTests(unittest.TestCase):
    """A.1-A.7 audit turned into regression tests: openai is a real PROVIDER_KINDS/HOSTED_KINDS
    member, selected the same way anthropic is, with its own credential env var."""

    def test_openai_is_a_real_provider_kind(self):
        from runtimes.pydantic_ai import provider
        self.assertIn("openai", provider.PROVIDER_KINDS)
        self.assertIn("openai", provider.HOSTED_KINDS)
        self.assertIn("anthropic", provider.HOSTED_KINDS)
        # local kinds are a disjoint set, unaffected by adding a hosted kind
        self.assertEqual(provider.LOCAL_KINDS, ("local-openai", "ollama"))

    def test_select_provider_kind_honors_explicit_openai(self):
        from runtimes.pydantic_ai.provider import select_provider_kind
        self.assertEqual(
            select_provider_kind(env={"PYDANTIC_AI_PROVIDER": "openai"}), "openai")

    def test_missing_openai_api_key_is_not_ready(self):
        from runtimes.pydantic_ai.provider import preflight_credentials
        r = preflight_credentials(
            env={"PYDANTIC_AI_MODEL": "gpt-4o-mini"}, provider="openai")
        self.assertEqual(r.status, "SKIPPED")
        self.assertFalse(r.key_present)
        self.assertEqual(r.provider, "openai")

    def test_bare_model_id_is_qualified_with_the_explicit_provider_not_guessed(self):
        # A bare model id ("gpt-4o-mini", no "openai:" prefix) must be attributed to the
        # explicitly-selected provider, never silently defaulted to anthropic (the old
        # _provider_of() fallback for an unprefixed name).
        from runtimes.pydantic_ai.provider import preflight_credentials
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}, clear=True), \
             mock.patch("runtimes.pydantic_ai.provider._sdk_available", return_value=True):
            r = preflight_credentials(
                env={"PYDANTIC_AI_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "sk-fake"},
                provider="openai")
        self.assertEqual(r.status, "READY")
        self.assertEqual(r.provider, "openai")
        self.assertEqual(r.model_id, "openai:gpt-4o-mini")

    def test_openai_key_present_but_sdk_missing_is_blocked(self):
        from runtimes.pydantic_ai.provider import preflight_credentials
        with mock.patch("runtimes.pydantic_ai.provider._sdk_available", return_value=False):
            r = preflight_credentials(
                env={"PYDANTIC_AI_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "sk-fake"},
                provider="openai")
        self.assertEqual(r.status, "BLOCKED")

    def test_unknown_provider_still_fails_closed(self):
        from runtimes.pydantic_ai.provider import preflight_credentials, select_provider_kind
        self.assertEqual(
            select_provider_kind(env={"PYDANTIC_AI_PROVIDER": "totally-bogus"}), "totally-bogus")
        r = preflight_credentials(
            env={"PYDANTIC_AI_MODEL": "x"}, provider="totally-bogus")
        self.assertEqual(r.status, "BLOCKED")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class HostedOpenAICliDispatchTests(unittest.TestCase):
    """D.1-D.4: openai reaches the real production CLI dispatch (run-task), fails closed without
    a key, and reaches the ordinary model-construction call with no real network/API request."""

    def test_openai_without_api_key_is_provider_unavailable(self):
        from runtimes.pydantic_ai import cli
        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"PYDANTIC_AI_PROVIDER": "openai",
                                               "PYDANTIC_AI_MODEL": "gpt-4o-mini"}, clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(task_path),
                             "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            self.assertEqual(code, cli.EXIT_PROVIDER_UNAVAILABLE)

    def test_configured_openai_reaches_model_construction_with_no_real_call(self):
        from runtimes.pydantic_ai import cli
        built = {"model_id": None, "runtime": False}

        def _fake_build(model_id):
            built["model_id"] = model_id
            return model_id

        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ",
                                {"PYDANTIC_AI_PROVIDER": "openai", "PYDANTIC_AI_MODEL": "gpt-4o-mini",
                                 "OPENAI_API_KEY": "sk-fake", "PYDANTIC_AI_SMOKE_CONFIRM": "yes"},
                                clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            with mock.patch("runtimes.pydantic_ai.provider._sdk_available", return_value=True), \
                 mock.patch("runtimes.pydantic_ai.provider.build_provider_model",
                            side_effect=_fake_build), \
                 mock.patch("runtimes.pydantic_ai.pydantic_ai_runtime.PydanticAIRuntime",
                            side_effect=lambda *a, **k: built.__setitem__("runtime", True) or
                            mock.Mock()):
                cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                         "--agent-specs-dir", SPECS, "--task", str(task_path),
                         "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
        self.assertEqual(built["model_id"], "openai:gpt-4o-mini")
        self.assertTrue(built["runtime"])

    def test_openai_ready_but_unconfirmed_calls_no_provider(self):
        # Mirrors the existing anthropic confirmation-gate test: READY credentials are not
        # sufficient without PYDANTIC_AI_SMOKE_CONFIRM=yes.
        from runtimes.pydantic_ai import cli
        built = {"model": False, "runtime": False}
        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ",
                                {"PYDANTIC_AI_PROVIDER": "openai", "PYDANTIC_AI_MODEL": "gpt-4o-mini",
                                 "OPENAI_API_KEY": "sk-fake"}, clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            with mock.patch("runtimes.pydantic_ai.provider._sdk_available", return_value=True), \
                 mock.patch("runtimes.pydantic_ai.provider.build_provider_model",
                            side_effect=lambda m: built.__setitem__("model", True)), \
                 mock.patch("runtimes.pydantic_ai.pydantic_ai_runtime.PydanticAIRuntime",
                            side_effect=lambda *a, **k: built.__setitem__("runtime", True)):
                code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                                 "--agent-specs-dir", SPECS, "--task", str(task_path),
                                 "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            self.assertFalse(built["model"])
            self.assertFalse(built["runtime"])

    def test_api_key_never_appears_in_persisted_provenance(self):
        from runtimes.pydantic_ai import cli
        fake_key = "sk-should-never-be-persisted-anywhere"
        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ",
                                {"PYDANTIC_AI_PROVIDER": "openai", "PYDANTIC_AI_MODEL": "gpt-4o-mini",
                                 "OPENAI_API_KEY": fake_key, "PYDANTIC_AI_SMOKE_CONFIRM": "yes"},
                                clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            ex_dir = d / "ex"
            with mock.patch("runtimes.pydantic_ai.provider._sdk_available", return_value=True), \
                 mock.patch("runtimes.pydantic_ai.provider.build_provider_model",
                            side_effect=lambda m: m), \
                 mock.patch("runtimes.pydantic_ai.pydantic_ai_runtime.PydanticAIRuntime",
                            side_effect=lambda *a, **k: mock.Mock()):
                cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                         "--agent-specs-dir", SPECS, "--task", str(task_path),
                         "--exchange-dir", str(ex_dir), "--mode", "shadow"])
            for path in ex_dir.rglob("*"):
                if path.is_file():
                    self.assertNotIn(fake_key, path.read_text(errors="ignore"),
                                     f"leaked into {path}")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ExistingProviderKindsUnchangedTests(unittest.TestCase):
    """D.5-D.7: local-openai, ollama, and anthropic keep exactly their prior behavior."""

    def test_local_openai_still_routes_through_preflight_local_not_credentials(self):
        from runtimes.pydantic_ai import cli
        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"PYDANTIC_AI_PROVIDER": "local-openai",
                                               "PYDANTIC_AI_MODEL": "qwen2.5-7b-instruct"},
                                clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            with mock.patch("runtimes.pydantic_ai.provider.preflight_credentials") as pc:
                code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                                 "--agent-specs-dir", SPECS, "--task", str(task_path),
                                 "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            pc.assert_not_called()
            # no PYDANTIC_AI_BASE_URL -> the same LOCAL_BASE_URL_NOT_CONFIGURED pause as before
            self.assertEqual(code, cli.EXIT_PROVIDER_UNAVAILABLE)

    def test_ollama_still_routes_through_preflight_local_not_credentials(self):
        from runtimes.pydantic_ai import cli
        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"PYDANTIC_AI_PROVIDER": "ollama",
                                               "PYDANTIC_AI_MODEL": "qwen2.5:7b"}, clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            with mock.patch("runtimes.pydantic_ai.provider.preflight_credentials") as pc:
                code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                                 "--agent-specs-dir", SPECS, "--task", str(task_path),
                                 "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            pc.assert_not_called()
            self.assertEqual(code, cli.EXIT_PROVIDER_UNAVAILABLE)

    def test_anthropic_without_credentials_is_still_provider_unavailable(self):
        from runtimes.pydantic_ai import cli
        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"PYDANTIC_AI_MODEL": "anthropic:fake"}, clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                             "--agent-specs-dir", SPECS, "--task", str(task_path),
                             "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            self.assertEqual(code, cli.EXIT_PROVIDER_UNAVAILABLE)

    def test_anthropic_ready_but_unconfirmed_calls_no_provider_and_still_gets_its_own_kind(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.provider import PreflightResult
        built = {"model": False, "runtime": False}
        seen_kwargs = {}

        def _fake_preflight(env=None, *, provider=None):
            seen_kwargs["provider"] = provider
            return PreflightResult("READY", "ok", "anthropic", "anthropic:fake", True, True)

        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"PYDANTIC_AI_MODEL": "anthropic:fake"}, clear=True):
            d = Path(tmp)
            task_path = d / "task.json"
            task_path.write_text(json.dumps(_judge_task()))
            with mock.patch("runtimes.pydantic_ai.provider.preflight_credentials",
                            side_effect=_fake_preflight), \
                 mock.patch("runtimes.pydantic_ai.provider.build_provider_model",
                            side_effect=lambda m: built.__setitem__("model", True)), \
                 mock.patch("runtimes.pydantic_ai.pydantic_ai_runtime.PydanticAIRuntime",
                            side_effect=lambda *a, **k: built.__setitem__("runtime", True)):
                code = cli.main(["run-task", "--runtime", "pydantic-ai", "--agent", "judge",
                                 "--agent-specs-dir", SPECS, "--task", str(task_path),
                                 "--exchange-dir", str(d / "ex"), "--mode", "shadow"])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            self.assertFalse(built["model"])
            self.assertFalse(built["runtime"])
            # anthropic is still explicitly resolved via select_provider_kind, same as openai now
            self.assertEqual(seen_kwargs["provider"], "anthropic")

    def test_producer_context_policy_unchanged_for_mock_local_openai_ollama_anthropic(self):
        from runtimes.pydantic_ai.cli import producer_context_policy
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(producer_context_policy("mock", "mock")["source"], "model-default")
            self.assertEqual(
                producer_context_policy("local-openai", "qwen2.5-7b-instruct")["source"],
                "model-default")
            self.assertEqual(
                producer_context_policy("local-openai", "some-other-served-model")["source"],
                "model-default")
            self.assertEqual(
                producer_context_policy("ollama", "qwen2.5:7b")["source"], "model-default")
            self.assertEqual(
                producer_context_policy("anthropic", "anthropic:claude-x")["source"],
                "model-default")

    def test_producer_context_policy_requires_explicit_declaration_only_for_unknown_openai_model(self):
        from runtimes.pydantic_ai.cli import producer_context_policy
        with mock.patch.dict("os.environ", {}, clear=True):
            policy = producer_context_policy("openai", "openai:gpt-4o-mini")
            self.assertEqual(policy["source"], "undeclared")
        with mock.patch.dict("os.environ", {"PYDANTIC_AI_CONTEXT_WINDOW_TOKENS": "128000"},
                             clear=True):
            policy = producer_context_policy("openai", "openai:gpt-4o-mini")
            self.assertEqual(policy["source"], "env")
            self.assertEqual(policy["context_window_tokens"], 128000)

    def test_run_production_stage_fails_closed_before_dispatch_when_context_window_undeclared(self):
        from runtimes.pydantic_ai import cli
        with _NoSockets(), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ",
                                {"PYDANTIC_AI_PROVIDER": "openai", "PYDANTIC_AI_MODEL": "gpt-4o-mini",
                                 "OPENAI_API_KEY": "sk-fake", "PYDANTIC_AI_SMOKE_CONFIRM": "yes"},
                                clear=True):
            import yaml
            from workflow.controller import RunController
            root = Path(tmp)
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({
                "run_id": "openai-context-test", "inputs": [],
                "stages": [{
                    "name": "stage_a", "command": None, "outputs": ["artifacts/x.json"],
                    "pydantic_ai": {"role": "data-curator", "action": "build_dataset_manifest",
                                   "idempotency_key": "openai-context-test:stage_a:001",
                                   "parameters": {}},
                }],
            }))
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            controller = RunController(run_dir)
            with mock.patch("runtimes.pydantic_ai.provider._sdk_available", return_value=True):
                result = cli.run_production_stage(
                    controller, "stage_a", runtime="pydantic-ai", repo_root=str(ROOT))
            self.assertEqual(result.reason, cli.PRODUCER_CONTEXT_WINDOW_UNDECLARED)
            self.assertEqual(result.exit_code, cli.EXIT_VALIDATION_REJECTED)
            self.assertNotEqual(RunController(run_dir).stage("stage_a")["status"], "completed")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ProviderChoiceDoesNotAffectSemanticsTests(unittest.TestCase):
    """D.10: choosing a provider is purely a runtime-dispatch concern; it must not change the
    ActionProposal a stage produces, nor any Controller/Gate/recovery/run-campaign semantics."""

    def _workflow(self, root: Path) -> Path:
        import yaml
        workflow = root / "workflow.yaml"
        workflow.write_text(yaml.safe_dump({
            "run_id": "provider-invariance-test", "inputs": [],
            "stages": [{
                "name": "stage_a", "command": None, "outputs": ["artifacts/x.json"],
                "pydantic_ai": {"role": "data-curator", "action": "build_dataset_manifest",
                               "idempotency_key": "provider-invariance-test:stage_a:001",
                               "parameters": {}},
            }],
        }))
        return workflow

    def test_action_proposal_is_identical_regardless_of_provider_env(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.cli import _proposal_from_stage, _stage_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            RunController.initialize(self._workflow(root), run_dir)

            with mock.patch.dict("os.environ", {}, clear=True):
                c1 = RunController(run_dir)
                proposal_none, _ = _proposal_from_stage(c1, "stage_a", _stage_config(c1, "stage_a"))

            with mock.patch.dict("os.environ",
                                 {"PYDANTIC_AI_PROVIDER": "openai",
                                  "PYDANTIC_AI_MODEL": "gpt-4o-mini",
                                  "OPENAI_API_KEY": "sk-fake"}, clear=True):
                c2 = RunController(run_dir)
                proposal_openai, _ = _proposal_from_stage(c2, "stage_a", _stage_config(c2, "stage_a"))

            self.assertEqual(proposal_none, proposal_openai)


if __name__ == "__main__":
    unittest.main()
