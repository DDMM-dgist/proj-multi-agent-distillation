"""R19 forensic-defect corrections: idempotent dispatch + partial-Judge-resume + atomic UTF-8
persistence (architecture-freeze revision v20; see tests/test_architecture_freeze.py).

Covers the exact production incident: an interrupted three-Judge gate on ``teacher_baseline``
where Judge 1's result was already accepted, Judge 2's raw-response write was cut short by a
UnicodeEncodeError (leaving a zero-byte raw file) under a non-UTF-8 default locale, and resuming
crashed again with a bare FileExistsError from FileExchangeRuntime.dispatch before even
re-invoking Judge 1. No network or real scientific compute is used anywhere in this file.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from orchestration.exchange import (FileExchangeRuntime, TaskPacketConflictError,
                                    atomic_write_text)
from orchestration.specs import load_agent_specs
from runtimes.pydantic_ai.cli import JudgeResumeConflictError, _judge_task, run_three_judge_gate
from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
from runtimes.pydantic_ai.models import RuntimeContext
from runtimes.pydantic_ai.production_router import run_role
from workflow.controller import RunController

ROOT = Path(__file__).resolve().parent.parent


def _task(task_id, agent, instruction="Do the thing.", **context):
    return {
        "schema_version": 1, "task_id": task_id, "agent": agent, "run_id": None,
        "created_at": "2026-01-01T00:00:00+00:00", "instruction": instruction,
        "inputs": [], "criteria": [], "constraints": [], "context": dict(context),
    }


class DispatchIdempotencyTests(unittest.TestCase):
    """Part A: FileExchangeRuntime.dispatch is idempotent/immutable, never a silent overwrite."""

    def setUp(self):
        self.specs = load_agent_specs(ROOT / "agent_specs", root=ROOT)

    def test_identical_redispatch_is_a_noop_reuse(self):
        spec = self.specs["data-curator"]
        task = _task("fixed-id-1", spec.name)
        with tempfile.TemporaryDirectory() as tmp:
            rt = FileExchangeRuntime(tmp)
            p1 = rt.dispatch(spec, task)
            before = p1.read_text(encoding="utf-8")
            p2 = rt.dispatch(spec, dict(task))  # a fresh dict, byte-identical content
            self.assertEqual(p1, p2)
            self.assertEqual(p2.read_text(encoding="utf-8"), before)  # never rewritten

    def test_conflicting_redispatch_fails_closed_and_never_overwrites(self):
        spec = self.specs["data-curator"]
        task = _task("fixed-id-2", spec.name, instruction="Original instruction.")
        with tempfile.TemporaryDirectory() as tmp:
            rt = FileExchangeRuntime(tmp)
            path = rt.dispatch(spec, task)
            original = path.read_text(encoding="utf-8")
            conflicting = _task("fixed-id-2", spec.name, instruction="DIFFERENT instruction.")
            with self.assertRaises(TaskPacketConflictError):
                rt.dispatch(spec, conflicting)
            self.assertEqual(path.read_text(encoding="utf-8"), original)  # untouched
            self.assertEqual(list(rt.outbox.glob("*.tmp")), [])  # no stray temp files

    def test_conflict_error_is_a_file_exists_error_subclass(self):
        # Existing bare `except FileExistsError:` call sites (e.g. runtimes/pydantic_ai/cli.py's
        # single-role-invoke command) must still catch a genuine identity conflict.
        self.assertTrue(issubclass(TaskPacketConflictError, FileExistsError))


class AtomicUtf8PersistenceTests(unittest.TestCase):
    """Part C: every exchange write is atomic + explicit UTF-8; a failed encode/write never
    truncates or corrupts an existing durable file."""

    def test_bare_write_text_is_the_r19_root_cause(self):
        # Reproduces the exact R19 mechanism: Path.write_text opens in 'w' mode (truncating to 0
        # bytes) BEFORE the TextIOWrapper attempts to encode; a non-ASCII character under a
        # non-UTF-8 encoding then raises mid-write, leaving a zero-byte file on disk -- exactly
        # matching the observed exchange/raw/teacher_baseline-judge-2.json.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            path.write_text("durable-original\n", encoding="utf-8")
            with self.assertRaises(UnicodeEncodeError):
                path.write_text("Judge rationale — an em dash.", encoding="ascii")
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_size, 0)  # the exact corrupted/truncated outcome

    def test_atomic_write_failure_leaves_existing_file_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            atomic_write_text(path, "durable-original\n")
            with self.assertRaises(UnicodeEncodeError):
                atomic_write_text(path, "Judge rationale — an em dash.", encoding="ascii")
            self.assertEqual(path.read_text(encoding="utf-8"), "durable-original\n")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])  # temp file cleaned up

    def test_atomic_write_succeeds_with_unicode_regardless_of_target_encoding_default(self):
        text = "Judge rationale — includes an em dash, and other text: café.\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            atomic_write_text(path, text)  # default encoding="utf-8"
            self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_exchange_accept_persists_unicode_raw_response(self):
        # End-to-end through the real production write path (accept -> raw + result), proving
        # the fix at the exact call site that failed in R19.
        specs = load_agent_specs(ROOT / "agent_specs", root=ROOT)
        spec = specs["data-curator"]
        task = _task("unicode-task-1", spec.name)
        with tempfile.TemporaryDirectory() as tmp:
            rt = FileExchangeRuntime(tmp)
            rt.dispatch(spec, task)
            raw = json.dumps({
                "schema_version": 1, "task_id": task["task_id"], "agent": spec.name,
                "status": "completed", "summary": "Lineage inspected — all clear.",
                "artifacts": [], "evidence": [], "requested_approval": None, "next_actions": [],
            })
            validated = rt.accept(spec, task["task_id"], raw)
            self.assertIn("—", validated["summary"])
            self.assertEqual((rt.raw / f"{task['task_id']}.json").read_text(encoding="utf-8"), raw)


def _agent_specs():
    return load_agent_specs(ROOT / "agent_specs", root=ROOT)


def _setup_gate_stage(root: Path):
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({"run_id": "r19-durability", "stages": [{
        "name": "s", "command": None, "outputs": ["artifacts/a.json"],
        "gate": {"criteria": ["c"]}}]}))
    run_dir = root / "run"
    c = RunController.initialize(workflow, run_dir)
    artifact = run_dir / "artifacts" / "a.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"ok": true}\n')
    c.complete_external_stage("s", [artifact])
    evidence = run_dir / "exchange" / "bounded_evidence" / "s.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"schema_version": 1}\n')
    return c, evidence


def _ctx_factory(run_dir):
    def factory(index):
        return RuntimeContext(exchange_dir=str(run_dir / "exchange"), repo_root=str(ROOT),
                              provider="mock", model_id="mock",
                              read_allow_prefixes=[], tools_enabled=False)
    return factory


def _counting_runtime_factory(call_log, verdict="PASS"):
    def factory(index):
        def responder(task, spec, toolset):
            call_log.append(index)
            ok = verdict == "PASS"
            return json.dumps({
                "review_lens": task["context"]["review_lens"], "verdict": verdict,
                "criteria_checked": [{"criterion": cr, "value_read": "ok", "ok": ok}
                                     for cr in task["criteria"]],
                "rationale": f"judge {index} checked frozen evidence", "required_fix": "" if ok else "fix it",
            }), (0, 0)
        return MockAgentRuntime(responder)
    return factory


def _pre_accept_judge(c, specs, evidence, run_dir, stage_name, index):
    """Faithfully simulate a PRIOR, already-completed production attempt for one Judge index,
    using the exact same _judge_task/dispatch/run_role(mode="primary") call chain the real
    run_three_judge_gate uses -- so the on-disk state left behind is indistinguishable from a
    real interrupted run."""
    gate_context = c.gate_context(stage_name)
    lens = gate_context["review_lenses"][index - 1]
    task = _judge_task(stage_name, index, lens, gate_context, evidence, c)
    exchange = FileExchangeRuntime(str(run_dir / "exchange"))
    exchange.dispatch(specs["judge"], task)
    ctx = _ctx_factory(run_dir)(index)

    def responder(task_, spec, toolset):
        return json.dumps({
            "review_lens": task_["context"]["review_lens"], "verdict": "PASS",
            "criteria_checked": [{"criterion": cr, "value_read": "ok", "ok": True}
                                 for cr in task_["criteria"]],
            "rationale": "pre-accepted in a prior (simulated) process", "required_fix": "",
        }), (0, 0)

    res = run_role(MockAgentRuntime(responder), task, specs["judge"], ctx, mode="primary")
    assert res.error is None and res.detail is not None, res.error
    return task


class JudgeGateResumeTests(unittest.TestCase):
    """Part B + Part D (Judge-gate lifecycle seams): partial-Judge resume never re-invokes an
    already-accepted Judge, never silently reuses/overwrites a conflicting binding, and always
    reaches the same terminal decision as an uninterrupted run."""

    def test_fresh_gate_invokes_all_three_judges_exactly_once(self):
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            call_log = []
            decision, path = run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(decision, "PASS")
            self.assertEqual(call_log, [1, 2, 3])
            self.assertEqual(c.stage("s")["gate"], "PASS")

    def test_resume_after_judge1_accepted_only_invokes_judge2_and_3(self):
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            _pre_accept_judge(c, specs, evidence, run_dir, "s", 1)  # simulate the crash point
            call_log = []
            decision, path = run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(decision, "PASS")
            self.assertEqual(call_log, [2, 3])  # Judge 1 reused, NOT re-invoked
            self.assertEqual(c.stage("s")["gate"], "PASS")
            bundle = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(bundle["votes"]), 3)

    def test_resume_after_judge1_and_judge2_accepted_only_invokes_judge3(self):
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            _pre_accept_judge(c, specs, evidence, run_dir, "s", 1)
            _pre_accept_judge(c, specs, evidence, run_dir, "s", 2)
            call_log = []
            decision, _ = run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(decision, "PASS")
            self.assertEqual(call_log, [3])

    def test_resume_after_all_three_accepted_before_gate_record_invokes_nobody(self):
        # Exactly "after Judge 3 accepted, before Gate record": the crash happens after every
        # Judge vote is durably accepted but before the votes bundle/record_gate ever ran.
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            for i in (1, 2, 3):
                _pre_accept_judge(c, specs, evidence, run_dir, "s", i)
            self.assertFalse((run_dir / "gates" / "s.production.votes.json").exists())
            call_log = []
            decision, path = run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(decision, "PASS")
            self.assertEqual(call_log, [])  # zero LLM calls: every vote was reused
            self.assertEqual(c.stage("s")["gate"], "PASS")  # record_gate ran exactly once
            self.assertTrue(path.exists())

    def test_zero_byte_raw_response_never_counts_as_accepted(self):
        # Exactly the R19 mechanism for Judge 2: a UnicodeEncodeError leaves a zero-byte raw
        # file with NO results/ entry -- resume must invoke that Judge, not skip it.
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            gate_context = c.gate_context("s")
            lens = gate_context["review_lenses"][0]
            task = _judge_task("s", 1, lens, gate_context, evidence, c)
            exchange = FileExchangeRuntime(str(run_dir / "exchange"))
            exchange.dispatch(specs["judge"], task)
            (exchange.raw / f"{task['task_id']}.json").write_text("")  # the zero-byte artifact
            self.assertFalse((exchange.inbox / f"{task['task_id']}.json").exists())
            call_log = []
            decision, _ = run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(decision, "PASS")
            self.assertEqual(call_log, [1, 2, 3])  # Judge 1 WAS invoked despite the raw litter

    def test_conflicting_task_packet_fails_closed(self):
        # Simulates a tampered/incompatible on-disk task packet under the deterministic task_id
        # (e.g. drifted gate criteria between an interrupted run and its resume).
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            gate_context = c.gate_context("s")
            lens = gate_context["review_lenses"][0]
            real_task = _judge_task("s", 1, lens, gate_context, evidence, c)
            exchange = FileExchangeRuntime(str(run_dir / "exchange"))
            tampered = dict(real_task)
            tampered["instruction"] = "TAMPERED instruction, not what the run actually derives."
            exchange.dispatch(specs["judge"], tampered)  # a foreign packet lands under this task_id
            call_log = []
            with self.assertRaises(TaskPacketConflictError):
                run_three_judge_gate(
                    c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(call_log, [])  # fails closed before ever invoking a Judge

    def test_accepted_result_with_mismatched_review_lens_fails_closed(self):
        # An accepted vote whose review_lens no longer matches the currently derived task's lens
        # must never be silently reused.
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            _pre_accept_judge(c, specs, evidence, run_dir, "s", 1)
            gate_context = c.gate_context("s")
            task = _judge_task("s", 1, gate_context["review_lenses"][0], gate_context, evidence, c)
            exchange = FileExchangeRuntime(str(run_dir / "exchange"))
            result_path = exchange.inbox / f"{task['task_id']}.json"
            stale = json.loads(result_path.read_text(encoding="utf-8"))
            stale["review_lens"] = "a-lens-that-does-not-match-the-current-task"
            result_path.write_text(json.dumps(stale))
            call_log = []
            with self.assertRaises(JudgeResumeConflictError):
                run_three_judge_gate(
                    c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(call_log, [])

    def test_accepted_result_without_accepted_provenance_fails_closed(self):
        # A results/ entry written OUTSIDE the real accept() pipeline (so no accepted=true
        # provenance record backs it) must never be trusted as a genuine accepted vote.
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            gate_context = c.gate_context("s")
            lens = gate_context["review_lenses"][0]
            task = _judge_task("s", 1, lens, gate_context, evidence, c)
            exchange = FileExchangeRuntime(str(run_dir / "exchange"))
            exchange.dispatch(specs["judge"], task)
            forged = {
                "review_lens": task["context"]["review_lens"], "verdict": "PASS",
                "criteria_checked": [{"criterion": cr, "value_read": "ok", "ok": True}
                                     for cr in task["criteria"]],
                "rationale": "forged, never actually accepted via the real pipeline",
                "required_fix": "",
            }
            (exchange.inbox / f"{task['task_id']}.json").write_text(json.dumps(forged))
            call_log = []
            with self.assertRaises(JudgeResumeConflictError):
                run_three_judge_gate(
                    c, "s", specs, _counting_runtime_factory(call_log), _ctx_factory(run_dir), evidence)
            self.assertEqual(call_log, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
