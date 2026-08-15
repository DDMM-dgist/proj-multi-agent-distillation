"""Deterministic integration fixture for the ``run-campaign`` outer production loop
(runtimes.pydantic_ai.cli.run_campaign): proves ONE campaign launch drives a real, generic stage
graph forward through multiple PASSing stages to terminal completion without anything manually
selecting each next stage, and proves the human-approval pause/resume boundary works across two
separate CLI process invocations (i.e. purely from durable Controller state, no in-memory state
carried between them).

Stage names here (``stage_a``/``stage_b``) are deliberately generic -- run_campaign/
_next_eligible_stage never reference a stage name, domain, or count; only each stage's own
``pydantic_ai.role``/``action`` (read from the workflow config, not any hardcoded route table)
determine what actually executes. The two actions used (``train_committee``, then
``evaluate_heldout_fidelity``, both role ``ml-trainer``) are real, already-registered executors so
the test exercises the genuine production dispatch path, not a stub.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


def _two_stage_workflow(root: Path) -> Path:
    dataset = root / "train.extxyz"
    frames = []
    for i in range(3):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"s{i}"
        a.info["parent_structure_id"] = f"p{i}"
        frames.append(a)
    write(str(dataset), frames)
    student_cfg = root / "student.yaml"
    student_cfg.write_text(
        "kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n"
        "predict:\n  factory: adapters.mock_model.MockCheckpointCalculator\n"
        "  checkpoint_arg: checkpoint\n  kwargs: {}\n")
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "synthetic-campaign",
        "inputs": [str(student_cfg), str(dataset)],
        "stages": [
            {
                "name": "stage_a",
                "command": None,
                "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
                "pydantic_ai": {
                    "role": "ml-trainer",
                    "action": "train_committee",
                    "approval_boundary": "costly_training",
                    "idempotency_key": "synthetic-campaign-stage-a-001",
                    "parameters": {
                        "student_config": str(student_cfg),
                        "dataset": str(dataset),
                        "output_dir": "{artifacts_dir}/committee",
                        "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
                    },
                },
                "gate": {"criteria": ["committee manifest is complete"]},
            },
            {
                "name": "stage_b",
                "command": None,
                "outputs": ["artifacts/heldout_labeled.extxyz", "artifacts/heldout_report.json"],
                "pydantic_ai": {
                    "role": "ml-trainer",
                    "action": "evaluate_heldout_fidelity",
                    "approval_boundary": "costly_training",
                    "idempotency_key": "synthetic-campaign-stage-b-001",
                    "parameters": {
                        "student_config": str(student_cfg),
                        "committee_manifest": "{artifacts_dir}/student_committee.manifest.json",
                        "frames_path": str(dataset),
                        "labeled_output": "{artifacts_dir}/heldout_labeled.extxyz",
                        "report_path": "{artifacts_dir}/heldout_report.json",
                    },
                },
                "gate": {"criteria": ["fidelity report is complete"]},
            },
        ],
    }))
    return workflow


class RunCampaignTests(unittest.TestCase):
    def test_pauses_for_approval_then_resumes_to_completion_across_two_stages(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _two_stage_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            # First launch: neither stage has been approved yet -- the campaign must stop at the
            # human-approval boundary rather than guess an approval, and it must not have touched
            # stage_b at all (no manual "next stage" selection happened).
            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            c = RunController(run_dir)
            self.assertEqual(c.stage("stage_a")["status"], "pending")
            self.assertEqual(c.stage("stage_b")["status"], "pending")

            # A human grants the approval out of band (never auto-approved by the campaign).
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training",
                                       "--note", "campaign test approval"]), cli.EXIT_SUCCESS)

            # Second, entirely separate CLI invocation: resumes purely from durable Controller
            # state and drives BOTH stages to PASS and the run to terminal completion in one call
            # -- nothing here re-supplies which stage is next.
            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            c = RunController(run_dir)
            self.assertEqual(c.stage("stage_a")["status"], "completed")
            self.assertEqual(c.stage("stage_a")["gate"], "PASS")
            self.assertEqual(c.stage("stage_b")["status"], "completed")
            self.assertEqual(c.stage("stage_b")["gate"], "PASS")
            self.assertTrue((run_dir / "artifacts" / "heldout_report.json").is_file())

            # stage_a's idempotency key must not have been replayed by the second invocation.
            training_events = [e for e in c.state["events"]
                               if e.get("type") == "external_stage_completed" and
                               e.get("stage") == "stage_a"]
            self.assertEqual(len(training_events), 1)

    def test_completed_campaign_is_idempotent_on_rerun(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _two_stage_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training", "--note", "t"]),
                             cli.EXIT_SUCCESS)
            self.assertEqual(cli.main(["run-campaign", "--runtime", "mock", "--run-dir",
                                       str(run_dir), "--auto-mock-judges"]), cli.EXIT_SUCCESS)
            # Re-running after terminal completion must not error, mutate state, or hang.
            before = RunController(run_dir).state
            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            after = RunController(run_dir).state
            self.assertEqual(before["stages"], after["stages"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
