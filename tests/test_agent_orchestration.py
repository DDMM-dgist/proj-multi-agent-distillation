import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from orchestration.exchange import (FileExchangeRuntime, make_task,
                                    validate_agent_response, validate_judge_vote)
from orchestration.cli import main as orchestration_main
from orchestration.specs import load_agent_specs


ROOT = Path(__file__).resolve().parent.parent


class AgentOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.specs = load_agent_specs(ROOT / "agent_specs", root=ROOT)

    def test_registry_has_expected_runtime_neutral_roles(self):
        expected = {"orchestrator", "literature", "data-curator", "ml-trainer",
                    "simulation", "analyst", "judge"}
        self.assertEqual(set(self.specs), expected)
        self.assertEqual(self.specs["orchestrator"].role_type, "coordinator")
        self.assertEqual(self.specs["judge"].role_type, "reviewer")
        for spec in self.specs.values():
            self.assertTrue(spec.prompt)
            self.assertFalse(spec.prompt.lstrip().startswith("---"))
            self.assertNotIn("model: sonnet", spec.prompt)

    def test_json_schemas_are_valid_json_and_cover_exchange_types(self):
        names = {path.name for path in (ROOT / "orchestration/schema").glob("*.json")}
        self.assertEqual(names, {"agent_spec.schema.json", "agent_task.schema.json",
                                 "agent_result.schema.json", "judge_vote.schema.json"})
        for path in (ROOT / "orchestration/schema").glob("*.json"):
            self.assertIsInstance(json.loads(path.read_text()), dict)

    def test_file_exchange_round_trip_is_bound_to_task_and_agent(self):
        spec = self.specs["data-curator"]
        task = make_task(spec.name, "Inspect lineage.", run_id="mock-run")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            task_path = runtime.dispatch(spec, task)
            self.assertEqual(json.loads(task_path.read_text())["task_id"], task["task_id"])
            result = {
                "schema_version": 1,
                "task_id": task["task_id"],
                "agent": spec.name,
                "status": "completed",
                "summary": "Lineage inspected.",
                "artifacts": [],
                "evidence": [],
                "requested_approval": None,
                "next_actions": [],
            }
            result_path = runtime.inbox / f"{task['task_id']}.json"
            result_path.write_text(json.dumps(result))
            self.assertEqual(runtime.collect(spec, task["task_id"])["status"], "completed")

    def _completed_result(self, task_id, agent):
        return {"schema_version": 1, "task_id": task_id, "agent": agent,
                "status": "completed", "summary": "Lineage inspected.",
                "artifacts": [], "evidence": [], "requested_approval": None,
                "next_actions": []}

    def test_accept_result_preserves_raw_then_records_validated(self):
        spec = self.specs["data-curator"]
        task = make_task(spec.name, "Inspect lineage.", run_id="mock-run")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            runtime.dispatch(spec, task)
            raw = json.dumps(self._completed_result(task["task_id"], spec.name))
            validated = runtime.accept(spec, task["task_id"], raw)
            self.assertEqual(validated["status"], "completed")
            self.assertTrue((runtime.raw / f"{task['task_id']}.json").is_file())
            self.assertEqual((runtime.raw / f"{task['task_id']}.json").read_text(), raw)
            self.assertTrue((runtime.inbox / f"{task['task_id']}.json").is_file())

    def test_accept_result_preserves_raw_even_when_validation_fails(self):
        # The audit guarantee: a contract-violating response is still on disk.
        spec = self.specs["data-curator"]
        task = make_task(spec.name, "Inspect lineage.", run_id="mock-run")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            runtime.dispatch(spec, task)
            bad = json.dumps(self._completed_result("wrong-task-id", spec.name))
            with self.assertRaisesRegex(ValueError, "raw preserved at"):
                runtime.accept(spec, task["task_id"], bad)
            self.assertEqual((runtime.raw / f"{task['task_id']}.json").read_text(), bad)
            self.assertFalse((runtime.inbox / f"{task['task_id']}.json").is_file())

    def test_accept_result_preserves_raw_when_not_json(self):
        spec = self.specs["data-curator"]
        task = make_task(spec.name, "Inspect lineage.", run_id="mock-run")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            runtime.dispatch(spec, task)
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                runtime.accept(spec, task["task_id"], "this is not json")
            self.assertEqual((runtime.raw / f"{task['task_id']}.json").read_text(),
                             "this is not json")

    def test_accept_result_without_dispatched_task_writes_nothing(self):
        spec = self.specs["data-curator"]
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            raw = json.dumps(self._completed_result("orphan-task", spec.name))
            with self.assertRaises(FileNotFoundError):
                runtime.accept(spec, "orphan-task", raw)
            self.assertFalse((runtime.raw / "orphan-task.json").exists())

    def test_accept_result_resubmission_retains_prior_raw(self):
        spec = self.specs["data-curator"]
        task = make_task(spec.name, "Inspect lineage.", run_id="mock-run")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            runtime.dispatch(spec, task)
            first = json.dumps(self._completed_result(task["task_id"], spec.name))
            runtime.accept(spec, task["task_id"], first)
            second = json.dumps({**self._completed_result(task["task_id"], spec.name),
                                 "summary": "Re-run."})
            runtime.accept(spec, task["task_id"], second)
            self.assertEqual((runtime.raw / f"{task['task_id']}.json").read_text(), first)
            self.assertEqual((runtime.raw / f"{task['task_id']}.1.json").read_text(), second)

    def test_accept_result_rejects_judge_vote_with_wrong_lens(self):
        # The exchange accept path agrees with the controller's lens enforcement.
        spec = self.specs["judge"]
        task = make_task("judge", "Review the gate.", criteria=["artifact is complete"],
                         context={"review_lens": "scientific_validity",
                                  "review_focus": "Audit scientific validity."})
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            runtime.dispatch(spec, task)
            vote = json.dumps({"review_lens": "evidence_provenance", "verdict": "PASS",
                               "criteria_checked": [{"criterion": "artifact is complete",
                                                     "value_read": "yes", "ok": True}],
                               "rationale": "ok", "required_fix": ""})
            with self.assertRaisesRegex(ValueError, "raw preserved at"):
                runtime.accept(spec, task["task_id"], vote)
            self.assertTrue((runtime.raw / f"{task['task_id']}.json").is_file())

    def test_accept_result_cli_round_trip(self):
        spec = self.specs["data-curator"]
        task = make_task(spec.name, "Inspect lineage.", run_id="mock-run")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FileExchangeRuntime(tmp)
            task_path = runtime.dispatch(spec, task)
            response_path = Path(tmp) / "response.json"
            response_path.write_text(json.dumps(
                self._completed_result(task["task_id"], spec.name)))
            with redirect_stdout(io.StringIO()):
                orchestration_main(["accept-result", spec.name, str(task_path), tmp,
                                    "--response", str(response_path)])
            self.assertTrue((runtime.inbox / f"{task['task_id']}.json").is_file())
            self.assertTrue((runtime.raw / f"{task['task_id']}.json").is_file())

    def test_result_rejects_cross_agent_or_cross_task_response(self):
        result = {
            "schema_version": 1,
            "task_id": "wrong-task",
            "agent": "analyst",
            "status": "completed",
            "summary": "Done.",
            "artifacts": [],
            "evidence": [],
            "requested_approval": None,
            "next_actions": [],
        }
        with self.assertRaises(ValueError):
            task = make_task("data-curator", "Inspect lineage.")
            validate_agent_response(result, self.specs["data-curator"], task)

    def test_judge_uses_its_own_ordered_vote_contract(self):
        criteria = ["artifact hash matches", "threshold is met"]
        vote = {
            "review_lens": "scientific_validity",
            "verdict": "PASS",
            "criteria_checked": [
                {"criterion": criterion, "value_read": "verified", "ok": True}
                for criterion in criteria
            ],
            "rationale": "Both criteria were verified.",
            "required_fix": "",
        }
        self.assertEqual(validate_judge_vote(
            vote, criteria, "scientific_validity"
        )["verdict"], "PASS")
        vote["criteria_checked"].reverse()
        with self.assertRaises(ValueError):
            validate_judge_vote(vote, criteria, "scientific_validity")

    def test_judge_task_requires_lens_context_and_vote_must_echo_it(self):
        spec = self.specs["judge"]
        with self.assertRaisesRegex(ValueError, "requires review_lens"):
            validate_agent_response(
                {"review_lens": "scientific_validity", "verdict": "PASS",
                 "criteria_checked": [], "rationale": "ok", "required_fix": ""},
                spec, make_task("judge", "Review the gate."),
            )
        task = make_task(
            "judge", "Review the gate.", criteria=["artifact is complete"],
            context={"review_lens": "scientific_validity",
                     "review_focus": "Audit scientific validity."},
        )
        vote = {"review_lens": "evidence_provenance", "verdict": "PASS",
                "criteria_checked": [{"criterion": "artifact is complete",
                                      "value_read": "yes", "ok": True}],
                "rationale": "ok", "required_fix": ""}
        with self.assertRaisesRegex(ValueError, "does not match the dispatched task"):
            validate_agent_response(vote, spec, task)

    def test_make_task_cli_records_review_lens_context(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            orchestration_main([
                "make-task", "judge", "Review the gate.", tmp,
                "--criterion", "artifact is complete",
                "--context", "review_lens=evidence_provenance",
                "--context", "review_focus=Audit hashes and lineage.",
            ])
            tasks = list((Path(tmp) / "tasks").glob("*.json"))
            self.assertEqual(len(tasks), 1)
            context = json.loads(tasks[0].read_text())["context"]
            self.assertEqual(context["review_lens"], "evidence_provenance")
            self.assertEqual(context["review_focus"], "Audit hashes and lineage.")

    def test_runtime_entrypoints_are_thin_and_reference_canonical_roles(self):
        self.assertIn("agent_specs/orchestrator.yaml", (ROOT / "AGENTS.md").read_text())
        for name in self.specs:
            wrapper = ROOT / ".claude/agents" / f"{name}.md"
            self.assertTrue(wrapper.is_file(), wrapper)
            self.assertIn(f"agents/{name}.md", wrapper.read_text())
            self.assertIn(f"agent_specs/{name}.yaml", wrapper.read_text())


if __name__ == "__main__":
    unittest.main()
