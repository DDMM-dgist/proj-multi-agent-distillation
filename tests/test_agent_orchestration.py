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
