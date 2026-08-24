"""Regression: the immutable three-Judge task packets must be scoped to the recovery ITERATION so
a re-gate of the same stage with new evidence never collides with a prior iteration's packets.

Reproduces the C12F Stage-7 re-gate blocker generically (network-free, mock runtime only):

    * a stage is gated once (iteration 1) -> it leaves immutable ``{stage}-judge-{n}`` task/result/
      raw/provenance artifacts on disk;
    * a recovery opens a fresh iteration and the SAME stage is re-gated with CHANGED evidence.

Before the fix the Judge task_id was ``f"{stage}-judge-{index}"`` -- not iteration-scoped -- so the
second gate re-derived the identical task_id but with different packet content, which
``FileExchangeRuntime.dispatch`` correctly refused to overwrite, raising ``TaskPacketConflictError``
(the whole gate fail-closed). The fix keys the task_id on the canonical
``_current_iteration()["id"]`` (the same identity the Controller already uses to name saved vote
bundles), so iteration 1 keeps the historical unsuffixed identity and every later iteration gets a
DISTINCT immutable packet -- the prior iteration's audit trail is preserved byte-for-byte.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from orchestration.exchange import FileExchangeRuntime, TaskPacketConflictError
from orchestration.specs import load_agent_specs
from runtimes.pydantic_ai.cli import (_judge_task, _judge_task_id, run_three_judge_gate)
from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
from runtimes.pydantic_ai.models import RuntimeContext
from workflow.controller import RunController, now

ROOT = Path(__file__).resolve().parent.parent


def _agent_specs():
    return load_agent_specs(ROOT / "agent_specs", root=ROOT)


def _setup_gate_stage(root: Path):
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({"run_id": "judge-iter-scope", "stages": [{
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
    evidence.write_text('{"schema_version": 1, "iteration": 1}\n')
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
                "rationale": f"judge {index} checked frozen evidence",
                "required_fix": "" if ok else "fix it",
            }), (0, 0)
        return MockAgentRuntime(responder)
    return factory


def _malformed_then_valid_factory(call_log, *, bad_index, bad_attempts):
    """The ``bad_index`` lens emits a malformed REVISE (empty required_fix -> a hard Judge OUTPUT
    validation failure, i.e. INVALID_JUDGE_OUTPUT) for its first ``bad_attempts`` invocations, then a
    valid PASS; every other lens PASSes immediately."""
    seen = {}

    def factory(index):
        def responder(task, spec, toolset):
            call_log.append(index)
            n = seen.get(index, 0) + 1
            seen[index] = n
            if index == bad_index and n <= bad_attempts:
                return json.dumps({
                    "review_lens": task["context"]["review_lens"], "verdict": "REVISE",
                    "criteria_checked": [{"criterion": cr, "value_read": "read", "ok": False}
                                         for cr in task["criteria"]],
                    "rationale": f"judge {index} wants changes but omitted the concrete fix",
                    "required_fix": "",
                }), (0, 0)
            return json.dumps({
                "review_lens": task["context"]["review_lens"], "verdict": "PASS",
                "criteria_checked": [{"criterion": cr, "value_read": "ok", "ok": True}
                                     for cr in task["criteria"]],
                "rationale": f"judge {index} checked frozen evidence", "required_fix": "",
            }), (0, 0)
        return MockAgentRuntime(responder)
    return factory


def _open_new_iteration(c, new_id):
    """Simulate the Controller opening a fresh recovery iteration (as ``start_iteration`` does) so
    the defect under test -- the Judge task-packet identity, which keys on
    ``_current_iteration()['id']`` -- is exercised without dragging in the full recovery machinery
    (that circular-deadlock path is covered by tests/test_recovery_ordering_deadlock.py). The opened
    iteration carries no trigger, so record_gate's recovery-for-PASS guard does not apply."""
    c.state["iterations"].append({"id": new_id, "parent_iteration": new_id - 1,
                                  "status": "active", "started_at": now(), "trigger": None})
    c.save()


class JudgeTaskIterationScopeTests(unittest.TestCase):
    def test_task_id_is_unsuffixed_at_iteration_1_and_iteration_scoped_after(self):
        # Generic scoping (no hardcoded stage/run): iteration 1 keeps the historical identity so
        # existing runs' on-disk packets are preserved; any later iteration is distinctly scoped.
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = _setup_gate_stage(Path(tmp))
            self.assertEqual(_judge_task_id(c, "s", 1), "s-judge-1")
            self.assertEqual(_judge_task_id(c, "other_stage", 3), "other_stage-judge-3")
            _open_new_iteration(c, 2)
            self.assertEqual(_judge_task_id(c, "s", 1), "s-judge-1-iter002")
            self.assertEqual(_judge_task_id(c, "other_stage", 3), "other_stage-judge-3-iter002")
            _open_new_iteration(c, 3)
            self.assertEqual(_judge_task_id(c, "s", 2), "s-judge-2-iter003")

    def test_regate_in_new_iteration_creates_distinct_immutable_packets(self):
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            tasks_dir = run_dir / "exchange" / "tasks"
            results_dir = run_dir / "exchange" / "results"
            raw_dir = run_dir / "exchange" / "raw"

            # --- iteration 1 gate: the historical unsuffixed packets ---
            log1 = []
            decision1, _ = run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(log1), _ctx_factory(run_dir), evidence)
            self.assertEqual(decision1, "PASS")
            self.assertEqual(log1, [1, 2, 3])
            i1_task_ids = [f"s-judge-{n}" for n in (1, 2, 3)]
            for tid in i1_task_ids:
                self.assertTrue((tasks_dir / f"{tid}.json").exists(), tid)
            # Snapshot the iteration-1 artifacts to prove later they are never mutated.
            i1_snapshot = {}
            for d in (tasks_dir, results_dir, raw_dir):
                for tid in i1_task_ids:
                    p = d / f"{tid}.json"
                    i1_snapshot[str(p)] = p.read_bytes()

            # --- iteration 2 re-gate of the SAME stage with CHANGED evidence ---
            c = RunController(run_dir)
            _open_new_iteration(c, 2)
            evidence.write_text('{"schema_version": 1, "iteration": 2, "changed": true}\n')
            log2 = []
            decision2, path2 = run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(log2), _ctx_factory(run_dir), evidence)

            # (3) No TaskPacketConflictError: the re-gate completed end to end.
            self.assertEqual(decision2, "PASS")
            # (5) The prior iteration's accepted results did NOT satisfy the new gate -- all three
            #     lenses were genuinely invoked again this iteration.
            self.assertEqual(log2, [1, 2, 3])

            # (2) Distinct iteration-scoped immutable packets now exist.
            i2_task_ids = [f"s-judge-{n}-iter002" for n in (1, 2, 3)]
            for tid in i2_task_ids:
                self.assertTrue((tasks_dir / f"{tid}.json").exists(), tid)
                self.assertTrue((results_dir / f"{tid}.json").exists(), tid)

            # (1) The iteration-1 audit trail is preserved byte-for-byte (nothing archived/renamed).
            for path_str, original in i1_snapshot.items():
                self.assertEqual(Path(path_str).read_bytes(), original, path_str)

            # (4) The correct iteration's three reviews were collected, and both iterations' vote
            #     bundles persist under distinct, iteration-scoped Controller records.
            bundle2 = json.loads(path2.read_text(encoding="utf-8"))
            self.assertEqual(len(bundle2["votes"]), 3)
            gates_dir = run_dir / "gates"
            self.assertTrue((gates_dir / "s.iteration-001.votes.json").exists())
            self.assertTrue((gates_dir / "s.iteration-002.votes.json").exists())

    def test_unsuffixed_regate_would_conflict_proving_the_scope_is_load_bearing(self):
        # Control: WITHOUT iteration scoping, the iteration-2 re-gate re-derives "s-judge-1" with
        # different packet content and fails closed against iteration 1's immutable packet. This is
        # exactly the pre-fix C12F blocker; the scoping above is what avoids it.
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            log1 = []
            run_three_judge_gate(
                c, "s", specs, _counting_runtime_factory(log1), _ctx_factory(run_dir), evidence)

            c = RunController(run_dir)
            _open_new_iteration(c, 2)
            evidence.write_text('{"schema_version": 1, "iteration": 2, "changed": true}\n')
            gate_context = c.gate_context("s")
            lens = gate_context["review_lenses"][0]
            i2_task = _judge_task("s", 1, lens, gate_context, evidence, c)
            self.assertEqual(i2_task["task_id"], "s-judge-1-iter002")  # the fix in effect
            # Force the pre-fix (unsuffixed) identity for the changed-content packet and dispatch it
            # against the iteration-1 packet already on disk.
            unsuffixed = dict(i2_task, task_id="s-judge-1")
            exchange = FileExchangeRuntime(str(run_dir / "exchange"))
            with self.assertRaises(TaskPacketConflictError):
                exchange.dispatch(specs["judge"], unsuffixed)

    def test_invalid_output_retry_stays_within_the_iteration_scoped_task_id(self):
        # Requirement 6: the INVALID_JUDGE_OUTPUT bounded per-lens retry still works within a single
        # iteration/gate attempt -- the retry re-dispatches the SAME iteration-scoped task_id, so the
        # retry's raw output is preserved as a suffixed sibling, never a cross-iteration collision.
        specs = _agent_specs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, evidence = _setup_gate_stage(root)
            run_dir = root / "run"
            raw_dir = run_dir / "exchange" / "raw"

            c = RunController(run_dir)
            _open_new_iteration(c, 2)
            log = []
            decision, path = run_three_judge_gate(
                c, "s", specs,
                _malformed_then_valid_factory(log, bad_index=2, bad_attempts=1),
                _ctx_factory(run_dir), evidence)
            self.assertEqual(decision, "PASS")
            # lens 2 was retried once within the same gate attempt (initial malformed + valid retry).
            self.assertEqual(log, [1, 2, 2, 3])
            bundle = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([v["verdict"] for v in bundle["votes"]], ["PASS", "PASS", "PASS"])
            # The bounded retry re-dispatched the SAME iteration-scoped task_id (the accepted vote
            # lands under it) and never fell back to the unsuffixed pre-fix identity in any of the
            # exchange lanes -- so a re-gate's retry can never collide with a prior iteration.
            results_dir = run_dir / "exchange" / "results"
            tasks_dir = run_dir / "exchange" / "tasks"
            self.assertTrue((raw_dir / "s-judge-2-iter002.json").exists())
            self.assertTrue((results_dir / "s-judge-2-iter002.json").exists())
            self.assertTrue((tasks_dir / "s-judge-2-iter002.json").exists())
            for d in (raw_dir, results_dir, tasks_dir):
                self.assertFalse((d / "s-judge-2.json").exists(), d)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
