"""Tests for the additive PydanticAI-compatible runtime PoC.

Two guard levels:
  * pydantic present  -> models, driver pipeline, and tool-security tests run (no network).
  * pydantic_ai present -> a REAL pydantic_ai.Agent is executed via TestModel (no network,
    no API key), exercising the same driver/validator/provenance path as the mock.

The whole module skips cleanly when the optional deps are absent (e.g. the core CI env),
so it never breaks the existing suite. The dedicated optional CI job installs
`.[pydantic-ai]` and asserts these do NOT skip.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import pydantic  # noqa: F401  (optional [pydantic-ai] extra)
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

try:
    import pydantic_ai  # noqa: F401
    _HAS_PYDANTIC_AI = True
except ImportError:
    _HAS_PYDANTIC_AI = False

from orchestration.exchange import FileExchangeRuntime, make_task
from orchestration.specs import load_agent_specs

ROOT = Path(__file__).resolve().parent.parent


def _valid_vote(lens):
    return json.dumps({"review_lens": lens, "verdict": "PASS",
                       "criteria_checked": [{"criterion": "artifact is complete",
                                             "value_read": "yes", "ok": True}],
                       "rationale": "ok", "required_fix": ""})


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic (optional [pydantic-ai] extra) not installed")
class RuntimePipelineTests(unittest.TestCase):
    def setUp(self):
        self.specs = load_agent_specs(str(ROOT / "agent_specs"))

    def _ctx(self, exchange_dir, allow=None):
        from runtimes.pydantic_ai.models import RuntimeContext
        return RuntimeContext(exchange_dir=str(exchange_dir), repo_root=str(ROOT),
                              read_allow_prefixes=allow or [str(exchange_dir)])

    def _judge_task(self, rt, spec, lens="evidence_provenance"):
        task = make_task("judge", "Review the gate.", criteria=["artifact is complete"],
                         context={"review_lens": lens, "review_focus": "Audit."})
        rt.dispatch(spec, task)
        return task

    # -- Pydantic models + schema drift ------------------------------------------

    def test_models_parse_valid_and_reject_extra_fields(self):
        from runtimes.pydantic_ai.models import JudgeVoteModel
        from pydantic import ValidationError
        m = JudgeVoteModel(review_lens="x", verdict="PASS",
                           criteria_checked=[{"criterion": "c", "ok": True}], rationale="ok")
        self.assertEqual(m.verdict, "PASS")
        with self.assertRaises(ValidationError):
            JudgeVoteModel(review_lens="x", verdict="PASS", criteria_checked=[],
                           rationale="ok", surprise=1)

    def test_schema_drift_detection_against_canonical_json_schema(self):
        from runtimes.pydantic_ai.models import JudgeVoteModel
        canonical = json.loads((ROOT / "orchestration/schema/judge_vote.schema.json").read_text())
        self.assertEqual(set(canonical["required"]), set(JudgeVoteModel.model_fields))

    # -- full pipeline via the mock runtime --------------------------------------

    def test_primary_mode_preserves_raw_and_records_result(self):
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = self._judge_task(rt, spec)
            res = run_task(MockAgentRuntime(lambda t, s, ts: (_valid_vote("evidence_provenance"), (100, 20))),
                           task, spec, self._ctx(exch))
            self.assertTrue(res.accepted)
            self.assertTrue((exch / "raw" / f"{task['task_id']}.json").is_file())
            self.assertTrue((exch / "results" / f"{task['task_id']}.json").is_file())
            self.assertEqual(res.invocation.provenance.prompt_tokens, 100)
            self.assertEqual(res.invocation.provenance.usage_source, "mock")

    def test_shadow_mode_validates_but_never_records(self):
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = self._judge_task(rt, spec, lens="scientific_validity")
            res = run_task(MockAgentRuntime(lambda t, s, ts: (_valid_vote("scientific_validity"), (10, 5))),
                           task, spec, self._ctx(exch), shadow=True)
            self.assertIsNotNone(res.validated)
            self.assertFalse(res.accepted)
            self.assertFalse((exch / "results" / f"{task['task_id']}.json").is_file())
            self.assertTrue(res.provenance_path.is_file())

    def test_invalid_output_is_rejected_by_existing_validation_not_pydantic(self):
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = self._judge_task(rt, spec, lens="evidence_provenance")
            res = run_task(MockAgentRuntime(lambda t, s, ts: (_valid_vote("WRONG_LENS"), (5, 5))),
                           task, spec, self._ctx(exch))
            self.assertFalse(res.accepted)
            self.assertIsNotNone(res.error)
            self.assertFalse((exch / "results" / f"{task['task_id']}.json").is_file())

    def test_non_json_output_is_captured_not_crashed(self):
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = self._judge_task(rt, spec)
            res = run_task(MockAgentRuntime(lambda t, s, ts: ("not json", (1, 1))),
                           task, spec, self._ctx(exch))
            self.assertFalse(res.accepted)
            self.assertIn("pydantic_parse",
                          [e.stage for e in res.invocation.provenance.validation_errors])

    def test_retry_attempts_keep_distinct_provenance_records(self):
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = self._judge_task(rt, spec)
            responder = lambda t, s, ts: (_valid_vote("evidence_provenance"), (1, 1))
            r1 = run_task(MockAgentRuntime(responder), task, spec, self._ctx(exch))
            r2 = run_task(MockAgentRuntime(responder), task, spec, self._ctx(exch))
            self.assertNotEqual(r1.provenance_path.name, r2.provenance_path.name)
            self.assertEqual(len(list((exch / "provenance").glob("*.json"))), 2)
            self.assertTrue((exch / "raw" / f"{task['task_id']}.json").is_file())
            self.assertTrue((exch / "raw" / f"{task['task_id']}.1.json").is_file())

    def test_mock_and_real_runtimes_share_the_build_invocation_path(self):
        # Guard against the mock taking a shortcut around the shared provenance path.
        import inspect
        from runtimes.pydantic_ai import mock_runtime, pydantic_ai_runtime
        self.assertIn("build_invocation", inspect.getsource(mock_runtime))
        self.assertIn("build_invocation", inspect.getsource(pydantic_ai_runtime))


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ToolSecurityTests(unittest.TestCase):
    def _toolset(self, allow, **kw):
        from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset
        return ReadOnlyToolset(allow, **kw)

    def test_allows_valid_text_inside_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("hi")
            self.assertEqual(self._toolset([tmp]).read_text(str(Path(tmp) / "a.txt")), "hi")

    def test_blocks_relative_and_absolute_paths_outside_prefix(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._toolset([str(Path(tmp) / "ok")])
            (Path(tmp) / "ok").mkdir()
            with self.assertRaises(ToolAccessError):
                ts.read_text("/etc/hostname")
            with self.assertRaises(ToolAccessError):
                ts.read_text(str(Path(tmp) / "ok" / ".." / "escape.txt"))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported on this OS")
    def test_symlink_escaping_the_prefix_is_blocked(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret_area = root / "outside"
            secret_area.mkdir()
            (secret_area / "loot.txt").write_text("secret")
            allow = root / "allow"
            allow.mkdir()
            link = allow / "link.txt"
            try:
                os.symlink(secret_area / "loot.txt", link)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation not permitted")
            with self.assertRaises(ToolAccessError):
                self._toolset([str(allow)]).read_text(str(link))  # resolves outside -> blocked

    def test_secret_path_components_are_refused(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in [".env", ".env.local"]:
                (root / rel).write_text("SECRET=1")
            (root / ".ssh").mkdir()
            (root / ".ssh" / "id_rsa").write_text("KEY")
            ts = self._toolset([str(root)])
            for rel in [".env", ".env.local", ".ssh/id_rsa"]:
                with self.assertRaises(ToolAccessError):
                    ts.read_text(str(root / rel))

    def test_binary_extension_is_refused(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "WAVECAR").write_bytes(b"\x00\x01\x02")
            with self.assertRaises(ToolAccessError):
                self._toolset([tmp]).read_text(str(Path(tmp) / "WAVECAR"))

    def test_per_file_size_cap_is_enforced(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.txt"
            big.write_text("x" * 2048)
            with self.assertRaises(ToolAccessError):
                self._toolset([tmp], max_file_bytes=1024).read_text(str(big))

    def test_invocation_byte_budget_is_enforced(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                (Path(tmp) / f"f{i}.txt").write_text("y" * 400)
            ts = self._toolset([tmp], invocation_byte_budget=1000)
            ts.read_text(str(Path(tmp) / "f0.txt"))
            ts.read_text(str(Path(tmp) / "f1.txt"))
            with self.assertRaises(ToolAccessError):
                ts.read_text(str(Path(tmp) / "f2.txt"))  # would exceed 1000-byte budget

    def test_refusals_are_recorded_for_audit(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._toolset([tmp])
            with self.assertRaises(ToolAccessError):
                ts.read_text("/etc/hostname")
            self.assertEqual(len(ts.invocations), 1)
            self.assertFalse(ts.invocations[0].ok)

    def test_read_json_returns_parsed_value_and_records_read_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "d.json").write_text('{"value": 3, "status": "ok"}')
            ts = self._toolset([tmp])
            self.assertEqual(ts.read_json(str(Path(tmp) / "d.json")),
                             {"value": 3, "status": "ok"})
            # recorded under 'read_json' exactly once (not double-logged as read_text)
            self.assertEqual([(i.tool, i.ok) for i in ts.invocations], [("read_json", True)])

    def test_read_json_invalid_json_records_a_single_failed_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bad.json").write_text("{not json")
            ts = self._toolset([tmp])
            with self.assertRaises(json.JSONDecodeError):
                ts.read_json(str(Path(tmp) / "bad.json"))
            # ok reflects the WHOLE operation: the read succeeded but the parse failed.
            self.assertEqual(len(ts.invocations), 1)
            self.assertEqual(ts.invocations[0].tool, "read_json")
            self.assertFalse(ts.invocations[0].ok)
            self.assertIn("JSON", ts.invocations[0].detail.upper())

    def test_read_text_invalid_utf8_records_failed_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.txt"
            bad.write_bytes(b"\xff\xfe\x00")
            ts = self._toolset([tmp])
            with self.assertRaises(UnicodeError):
                ts.read_text(str(bad))
            self.assertEqual(len(ts.invocations), 1)
            self.assertEqual(ts.invocations[0].tool, "read_text")
            self.assertFalse(ts.invocations[0].ok)

    def test_read_json_invalid_utf8_records_failed_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_bytes(b"\xff\xfe\x00")
            ts = self._toolset([tmp])
            with self.assertRaises(UnicodeError):   # decode fails before json.loads
                ts.read_json(str(bad))
            self.assertEqual(len(ts.invocations), 1)
            self.assertEqual(ts.invocations[0].tool, "read_json")
            self.assertFalse(ts.invocations[0].ok)

    def test_read_json_outside_and_secret_paths_are_blocked_and_recorded(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text('{"x":1}')
            allow = root / "ok"
            allow.mkdir()
            ts = self._toolset([str(allow)])
            with self.assertRaises(ToolAccessError):
                ts.read_json("/etc/hostname")           # outside allow-list
            with self.assertRaises(ToolAccessError):
                ts.read_json(str(root / ".env"))        # secret component
            self.assertEqual([i.tool for i in ts.invocations], ["read_json", "read_json"])
            self.assertTrue(all(not i.ok for i in ts.invocations))

    def test_manifest_lists_exactly_the_exposed_read_tools(self):
        from runtimes.pydantic_ai.tool_registry import EXPOSED_READ_TOOLS
        self.assertEqual(EXPOSED_READ_TOOLS,
                         ("read_text", "read_json", "read_csv_summary", "read_artifact_manifest"))


@unittest.skipUnless(_HAS_PYDANTIC_AI, "pydantic_ai not installed (optional [pydantic-ai] extra)")
class RealPydanticAiAgentTests(unittest.TestCase):
    """Executes a REAL pydantic_ai.Agent via TestModel — no network, no API key."""

    def setUp(self):
        self.specs = load_agent_specs(str(ROOT / "agent_specs"))

    def _ctx(self, exch):
        from runtimes.pydantic_ai.models import RuntimeContext
        return RuntimeContext(exchange_dir=str(exch), repo_root=str(ROOT),
                              provider="test", model_id="test-model",
                              read_allow_prefixes=[str(exch)])

    def test_real_agent_valid_output_flows_through_existing_validation_to_accept(self):
        from pydantic_ai.models.test import TestModel
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = make_task("judge", "Review.", criteria=["artifact is complete"],
                             context={"review_lens": "evidence_provenance", "review_focus": "x"})
            rt.dispatch(spec, task)
            tm = TestModel(call_tools=[], custom_output_args={
                "review_lens": "evidence_provenance", "verdict": "PASS",
                "criteria_checked": [{"criterion": "artifact is complete",
                                      "value_read": "yes", "ok": True}],
                "rationale": "ok", "required_fix": ""})
            res = run_task(PydanticAIRuntime(model=tm, usage_source="test-model"),
                           task, spec, self._ctx(exch))
            self.assertTrue(res.accepted)
            self.assertEqual(res.invocation.provenance.usage_source, "test-model")
            self.assertGreater(res.invocation.provenance.prompt_tokens, 0)
            self.assertTrue((exch / "results" / f"{task['task_id']}.json").is_file())

    def test_real_agent_wrong_lens_output_is_rejected(self):
        from pydantic_ai.models.test import TestModel
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = make_task("judge", "Review.", criteria=["artifact is complete"],
                             context={"review_lens": "evidence_provenance", "review_focus": "x"})
            rt.dispatch(spec, task)
            tm = TestModel(call_tools=[], custom_output_args={
                "review_lens": "WRONG", "verdict": "PASS",
                "criteria_checked": [{"criterion": "artifact is complete",
                                      "value_read": "yes", "ok": True}],
                "rationale": "ok", "required_fix": ""})
            res = run_task(PydanticAIRuntime(model=tm, usage_source="test-model"),
                           task, spec, self._ctx(exch))
            self.assertFalse(res.accepted)
            self.assertIsNotNone(res.error)

    def test_real_agent_tool_call_outside_allow_list_is_refused_not_crashed(self):
        # Default TestModel calls the registered read_text tool with a dummy arg; our
        # allow-list refuses it and the run still completes (refusal recorded).
        from pydantic_ai.models.test import TestModel
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = make_task("judge", "Review.", criteria=["artifact is complete"],
                             context={"review_lens": "evidence_provenance", "review_focus": "x"})
            rt.dispatch(spec, task)
            tm = TestModel(custom_output_args={  # call_tools defaults to calling tools
                "review_lens": "evidence_provenance", "verdict": "PASS",
                "criteria_checked": [{"criterion": "artifact is complete",
                                      "value_read": "yes", "ok": True}],
                "rationale": "ok", "required_fix": ""})
            res = run_task(PydanticAIRuntime(model=tm, usage_source="test-model"),
                           task, spec, self._ctx(exch))
            refusals = [i for i in res.invocation.provenance.tool_invocations if not i.ok]
            self.assertGreaterEqual(len(refusals), 1)

    def test_real_agent_registers_and_calls_both_read_tools(self):
        # TestModel(call_tools defaults to 'all') calls every registered tool, so both
        # read_text AND read_json appear in the provenance — proving read_json is wired
        # onto the real pydantic_ai.Agent, not just present in the toolset/manifest.
        from pydantic_ai.models.test import TestModel
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        from runtimes.pydantic_ai.driver import run_task
        from runtimes.pydantic_ai.tool_registry import EXPOSED_READ_TOOLS
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            spec = self.specs["judge"]
            task = make_task("judge", "Review.", criteria=["artifact is complete"],
                             context={"review_lens": "evidence_provenance", "review_focus": "x"})
            rt.dispatch(spec, task)
            tm = TestModel(custom_output_args={
                "review_lens": "evidence_provenance", "verdict": "PASS",
                "criteria_checked": [{"criterion": "artifact is complete",
                                      "value_read": "yes", "ok": True}],
                "rationale": "ok", "required_fix": ""})
            res = run_task(PydanticAIRuntime(model=tm, usage_source="test-model"),
                           task, spec, self._ctx(exch))
            called = {i.tool for i in res.invocation.provenance.tool_invocations}
            self.assertEqual(called, set(EXPOSED_READ_TOOLS))  # both tools registered + called

    def test_real_agent_reads_allowed_json_via_read_json_tool(self):
        # A model that calls read_json on an allowed JSON file: the tool runs, the parsed
        # value is returned to the model, and the invocation is logged as read_json (ok).
        from pydantic_ai.models.function import FunctionModel, AgentInfo
        from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, TextPart
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            exch.mkdir(parents=True, exist_ok=True)
            (exch / "evidence.json").write_text('{"k": 1}')
            spec = self.specs["judge"]
            task = make_task("judge", "Review.", criteria=["artifact is complete"],
                             context={"review_lens": "evidence_provenance", "review_focus": "x"})
            rt.dispatch(spec, task)
            target = str(exch / "evidence.json")

            calls = {"n": 0}

            def responder(messages, info: AgentInfo):
                # First turn: call read_json on the allowed file. Second turn: final output.
                if calls["n"] == 0:
                    calls["n"] += 1
                    return ModelResponse(parts=[ToolCallPart("read_json", {"path": target})])
                out = {"review_lens": "evidence_provenance", "verdict": "PASS",
                       "criteria_checked": [{"criterion": "artifact is complete",
                                             "value_read": "yes", "ok": True}],
                       "rationale": "ok", "required_fix": ""}
                # The final result is delivered via the output tool pydantic_ai injects.
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, out)])

            res = run_task(PydanticAIRuntime(model=FunctionModel(responder),
                                             usage_source="test-model"),
                           task, spec, self._ctx(exch))
            read_json_ok = [i for i in res.invocation.provenance.tool_invocations
                            if i.tool == "read_json" and i.ok]
            self.assertGreaterEqual(len(read_json_ok), 1)
            self.assertTrue(res.accepted)

    def _agent_calls_read_json_then_finishes(self, exch, target, spec):
        """A FunctionModel that calls read_json(target) then returns a valid vote."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        calls = {"n": 0}

        def responder(messages, info):
            if calls["n"] == 0:
                calls["n"] += 1
                return ModelResponse(parts=[ToolCallPart("read_json", {"path": target})])
            out = {"review_lens": "evidence_provenance", "verdict": "PASS",
                   "criteria_checked": [{"criterion": "artifact is complete",
                                         "value_read": "yes", "ok": True}],
                   "rationale": "ok", "required_fix": ""}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, out)])
        return responder

    def test_real_agent_invalid_json_is_refused_not_crashed(self):
        from pydantic_ai.models.function import FunctionModel
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            exch.mkdir(parents=True, exist_ok=True)
            (exch / "bad.json").write_text("{not json")
            spec = self.specs["judge"]
            task = make_task("judge", "Review.", criteria=["artifact is complete"],
                             context={"review_lens": "evidence_provenance", "review_focus": "x"})
            rt.dispatch(spec, task)
            responder = self._agent_calls_read_json_then_finishes(exch, str(exch / "bad.json"), spec)
            res = run_task(PydanticAIRuntime(model=FunctionModel(responder),
                                             usage_source="test-model"), task, spec, self._ctx(exch))
            failed = [i for i in res.invocation.provenance.tool_invocations
                      if i.tool == "read_json" and not i.ok]
            self.assertGreaterEqual(len(failed), 1)   # recorded ok=False
            self.assertTrue(res.accepted)             # run completed, no crash

    def test_real_agent_invalid_utf8_json_is_refused_not_crashed(self):
        from pydantic_ai.models.function import FunctionModel
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        from runtimes.pydantic_ai.driver import run_task
        with tempfile.TemporaryDirectory() as tmp:
            exch = Path(tmp) / "exchange"
            rt = FileExchangeRuntime(str(exch))
            exch.mkdir(parents=True, exist_ok=True)
            (exch / "enc.json").write_bytes(b"\xff\xfe\x00")
            spec = self.specs["judge"]
            task = make_task("judge", "Review.", criteria=["artifact is complete"],
                             context={"review_lens": "evidence_provenance", "review_focus": "x"})
            rt.dispatch(spec, task)
            responder = self._agent_calls_read_json_then_finishes(exch, str(exch / "enc.json"), spec)
            res = run_task(PydanticAIRuntime(model=FunctionModel(responder),
                                             usage_source="test-model"), task, spec, self._ctx(exch))
            failed = [i for i in res.invocation.provenance.tool_invocations
                      if i.tool == "read_json" and not i.ok]
            self.assertGreaterEqual(len(failed), 1)
            self.assertTrue(res.accepted)


if __name__ == "__main__":
    unittest.main()
