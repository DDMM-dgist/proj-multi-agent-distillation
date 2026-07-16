import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController
from workflow.contracts import validate_md_manifest
from workflow.integrity import artifact_digest


ROOT = Path(__file__).resolve().parent.parent


class RunControllerTests(unittest.TestCase):
    def pass_gate(self, controller, stage):
        artifacts = {a["path"]: a["sha256"] for a in controller.stage_artifacts(stage)}
        vote_path = controller.run_dir / "gates" / f"{stage}.votes.json"
        criterion = "artifact is complete and internally consistent"
        def vote(judge_id):
            return {"judge_id": judge_id, "verdict": "PASS", "criteria_checked":
                    [{"criterion": criterion, "ok": True}], "rationale": "ok", "required_fix": ""}
        vote_path.write_text(json.dumps({"stage": stage, "criteria": [criterion],
                                        "artifact_sha256": artifacts, "decision": "PASS",
                                        "votes": [vote("judge-1"), vote("judge-2"), vote("judge-3")]}))
        controller.record_gate(stage, votes_path=vote_path)

    def test_gate_blocks_next_stage_and_state_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({
                "run_id": "test-run",
                "stages": [
                    {"name": "data", "command": [sys.executable, "-c",
                     "from pathlib import Path; Path('artifacts/data.txt').write_text('ok')"],
                     "outputs": ["artifacts/data.txt"]},
                    {"name": "train", "command": [sys.executable, "-c",
                     "from pathlib import Path; Path('artifacts/model.txt').write_text('ok')"],
                     "outputs": ["artifacts/model.txt"]},
                ],
            }))
            run_dir = root / "run"
            controller = RunController.initialize(cfg, run_dir)
            controller.run_stage("data")
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                controller.run_stage("train")
            self.pass_gate(controller, "data")
            RunController(run_dir).run_stage("train")
            resumed = RunController(run_dir)
            self.assertEqual(resumed.stage("train")["status"], "completed")
            self.assertEqual(len(resumed.state["artifacts"]), 2)

    def test_stage_can_import_project_packages_from_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{"name": "smoke",
                "command": ["{python}", "-c", "import adapters; from pathlib import Path; Path('artifacts/import-ok').write_text('ok')"],
                "outputs": ["artifacts/import-ok"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            controller.run_stage("smoke")
            self.assertEqual(controller.stage("smoke")["status"], "completed")

    def test_declared_inputs_are_snapshotted_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "student.yaml"
            source.write_text("kind: mock\n")
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "inputs": [str(source)],
                                           "stages": [{"name": "x", "command": None}]}))
            controller = RunController.initialize(cfg, root / "run")
            self.assertEqual(len(controller.state["inputs"]), 1)
            self.assertTrue(Path(controller.state["inputs"][0]["snapshot"]).is_file())

    def test_changed_source_input_blocks_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.yaml"
            source.write_text("original")
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "inputs": [str(source)],
                "stages": [{"name": "x", "command": [sys.executable, "-c", "pass"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            source.write_text("mutated")
            with self.assertRaisesRegex(RuntimeError, "changed after initialization"):
                controller.run_stage("x")
            controller.rebind_inputs()
            controller.run_stage("x")
            self.assertEqual(controller.stage("x")["status"], "completed")
            self.assertEqual(controller.state["events"][-1]["type"], "inputs_rebound")

    def test_large_directory_input_is_hash_bound_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "teacher-model"
            model.mkdir()
            (model / "weights.bin").write_text("v1")
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x",
                "inputs": [{"path": str(model), "copy": False}],
                "stages": [{"name": "x", "command": [sys.executable, "-c", "pass"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            self.assertIsNone(controller.state["inputs"][0]["snapshot"])
            (model / "weights.bin").write_text("v2")
            with self.assertRaisesRegex(RuntimeError, "changed after initialization"):
                controller.run_stage("x")

    def test_mock_workflow_runs_end_to_end_with_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = RunController.initialize(ROOT / "examples/mock/workflow.yaml",
                                                  Path(tmp) / "run")
            for stage in ("teacher_labeling", "dataset_split", "training", "evaluation"):
                controller.run_stage(stage)
                self.pass_gate(controller, stage)
            self.assertEqual(controller.stage("evaluation")["gate"], "PASS")
            self.assertTrue((controller.run_dir / "artifacts/accuracy_report.json").is_file())

    def test_modified_committee_checkpoint_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = RunController.initialize(ROOT / "examples/mock/workflow.yaml",
                                                  Path(tmp) / "run")
            for stage in ("teacher_labeling", "dataset_split", "training"):
                controller.run_stage(stage)
                self.pass_gate(controller, stage)
            checkpoint = controller.run_dir / "artifacts/committee/seed-1/mock-model.json"
            checkpoint.write_text('{"seed": 999}\n')
            with self.assertRaisesRegex(RuntimeError, "artifact integrity"):
                controller.run_stage("evaluation")

    def test_upstream_rerun_invalidates_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [
                {"name": "data", "command": [sys.executable, "-c",
                 "from pathlib import Path; Path('artifacts/data').write_text('v1')"],
                 "outputs": ["artifacts/data"]},
                {"name": "train", "command": [sys.executable, "-c",
                 "from pathlib import Path; Path('artifacts/model').write_text('m1')"],
                 "outputs": ["artifacts/model"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            controller.run_stage("data"); self.pass_gate(controller, "data")
            controller.run_stage("train"); self.pass_gate(controller, "train")
            controller.run_stage("data")
            self.assertEqual(controller.stage("train")["status"], "pending")
            self.assertEqual(controller.stage("train")["gate"], "pending")
            self.assertFalse(controller.stage_artifacts("train"))
            self.assertTrue(list((controller.run_dir / "stale").rglob("model")))

    def test_mutated_artifact_cannot_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{"name": "data",
                "command": [sys.executable, "-c", "from pathlib import Path; Path('artifacts/x').write_text('ok')"],
                "outputs": ["artifacts/x"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            controller.run_stage("data")
            artifact = Path(controller.stage_artifacts("data")[0]["path"])
            artifact.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                self.pass_gate(controller, "data")

    def test_pass_without_vote_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{"name": "data", "command": None}]}))
            controller = RunController.initialize(cfg, root / "run")
            artifact = root / "run" / "artifacts" / "x"
            artifact.write_text("x")
            controller.complete_external_stage("data", [artifact])
            with self.assertRaisesRegex(ValueError, "requires --votes"):
                controller.record_gate("data", "PASS")

    def test_gate_rejects_unfinished_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{"name": "data", "command": ["false"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            with self.assertRaisesRegex(RuntimeError, "completed"):
                controller.record_gate("data", "REVISE")

    def test_external_agent_artifact_can_complete_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{"name": "analysis", "command": None}]}))
            controller = RunController.initialize(cfg, root / "run")
            report = root / "run" / "artifacts" / "report.json"
            report.write_text("{}")
            controller.complete_external_stage("analysis", ["artifacts/report.json"])
            self.assertEqual(controller.stage("analysis")["status"], "completed")
            self.assertEqual(len(controller.state["artifacts"]), 1)

    def test_external_md_is_bound_to_approved_committee_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [
                {"name": "training", "command": None},
                {"name": "md", "command": None, "contract": {
                    "kind": "md_manifest", "manifest": "artifacts/md.manifest.json",
                    "committee_manifest": "artifacts/student_committee.manifest.json",
                    "required_evidence": ["input", "trajectory"]}},
            ]}))
            controller = RunController.initialize(cfg, root / "run")
            checkpoint = controller.run_dir / "artifacts" / "committee" / "seed-2" / "model.yaml"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("approved")
            committee = controller.run_dir / "artifacts" / "student_committee.manifest.json"
            committee.write_text(json.dumps({"models": [{"seed": 2, "path": str(checkpoint),
                                                         "integrity": artifact_digest(checkpoint)}]}))
            controller.complete_external_stage("training", [committee, checkpoint])
            self.pass_gate(controller, "training")

            md_manifest = controller.run_dir / "artifacts" / "md.manifest.json"
            md_input = controller.run_dir / "artifacts" / "prod_md.in"
            trajectory = controller.run_dir / "artifacts" / "trajectory.dump"
            md_input.write_text("input"); trajectory.write_text("trajectory")
            md_manifest.write_text(json.dumps({"selected_seed": 2, "checkpoint": str(checkpoint),
                                               "checkpoint_integrity": artifact_digest(checkpoint),
                                               "committee_manifest": str(committee),
                                               "evidence": [
                                                   {"role": "input", "path": str(md_input),
                                                    "integrity": artifact_digest(md_input)},
                                                   {"role": "trajectory", "path": str(trajectory),
                                                    "integrity": artifact_digest(trajectory)},
                                               ]}))
            controller.complete_external_stage("md", [md_manifest, md_input, trajectory])
            self.assertTrue(controller.state["events"][-1]["contract_validated"])

    def test_external_md_rejects_evidence_not_submitted_as_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [
                {"name": "training", "command": None},
                {"name": "md", "command": None, "contract": {
                    "kind": "md_manifest", "manifest": "artifacts/md.manifest.json",
                    "committee_manifest": "artifacts/student_committee.manifest.json",
                    "required_evidence": ["trajectory"]}},
            ]}))
            controller = RunController.initialize(cfg, root / "run")
            checkpoint = controller.run_dir / "artifacts" / "model.yaml"
            checkpoint.write_text("approved")
            committee = controller.run_dir / "artifacts" / "student_committee.manifest.json"
            committee.write_text(json.dumps({"models": [{"seed": 1, "path": str(checkpoint),
                                                         "integrity": artifact_digest(checkpoint)}]}))
            controller.complete_external_stage("training", [committee, checkpoint])
            self.pass_gate(controller, "training")
            trajectory = controller.run_dir / "artifacts" / "trajectory.dump"
            trajectory.write_text("trajectory")
            md_manifest = controller.run_dir / "artifacts" / "md.manifest.json"
            md_manifest.write_text(json.dumps({"selected_seed": 1, "checkpoint": str(checkpoint),
                                               "checkpoint_integrity": artifact_digest(checkpoint),
                                               "committee_manifest": str(committee),
                                               "evidence": [{"role": "trajectory",
                                                             "path": str(trajectory),
                                                             "integrity": artifact_digest(trajectory)}]}))
            with self.assertRaisesRegex(ValueError, "not submitted as a stage artifact"):
                controller.complete_external_stage("md", [md_manifest])
            self.assertEqual(controller.stage("md")["status"], "pending")

    def test_external_md_rejects_modified_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.yaml"
            checkpoint.write_text("approved")
            committee = root / "committee.json"
            committee.write_text(json.dumps({"models": [{"seed": 1, "path": str(checkpoint),
                                                         "integrity": artifact_digest(checkpoint)}]}))
            trajectory = root / "trajectory.dump"
            trajectory.write_text("original")
            manifest = root / "md.json"
            manifest.write_text(json.dumps({"selected_seed": 1, "checkpoint": str(checkpoint),
                                            "checkpoint_integrity": artifact_digest(checkpoint),
                                            "committee_manifest": str(committee),
                                            "evidence": [{"role": "trajectory",
                                                          "path": str(trajectory),
                                                          "integrity": artifact_digest(trajectory)}]}))
            trajectory.write_text("modified")
            with self.assertRaisesRegex(ValueError, "evidence integrity check failed"):
                validate_md_manifest(manifest, committee, [manifest, trajectory], ["trajectory"])

    def test_external_md_rejects_unapproved_checkpoint_without_mutating_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [
                {"name": "training", "command": None},
                {"name": "md", "command": None, "contract": {
                    "kind": "md_manifest", "manifest": "artifacts/md.manifest.json",
                    "committee_manifest": "artifacts/student_committee.manifest.json"}},
            ]}))
            controller = RunController.initialize(cfg, root / "run")
            approved = controller.run_dir / "artifacts" / "approved.yaml"
            unapproved = controller.run_dir / "artifacts" / "unapproved.yaml"
            approved.write_text("approved"); unapproved.write_text("other")
            committee = controller.run_dir / "artifacts" / "student_committee.manifest.json"
            committee.write_text(json.dumps({"models": [{"seed": 1, "path": str(approved),
                                                         "integrity": artifact_digest(approved)}]}))
            controller.complete_external_stage("training", [committee, approved])
            self.pass_gate(controller, "training")
            md_manifest = controller.run_dir / "artifacts" / "md.manifest.json"
            md_manifest.write_text(json.dumps({"selected_seed": 1, "checkpoint": str(unapproved),
                                               "checkpoint_integrity": artifact_digest(unapproved),
                                               "committee_manifest": str(committee)}))
            with self.assertRaisesRegex(ValueError, "not the selected committee checkpoint"):
                controller.complete_external_stage("md", [md_manifest])
            self.assertEqual(controller.stage("md")["status"], "pending")


if __name__ == "__main__":
    unittest.main()
