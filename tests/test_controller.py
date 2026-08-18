import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from workflow.controller import RunController, format_context
from workflow.contracts import validate_md_manifest
from workflow.integrity import artifact_digest


ROOT = Path(__file__).resolve().parent.parent


class RunControllerTests(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def test_contract_options_format_controller_context_recursively(self):
        value = {"profile": "{project_dir}/profile.yaml",
                 "nested": ["{run_dir}/x", 2]}
        self.assertEqual(format_context(value, {"project_dir": "/project",
                                                "run_dir": "/run"}),
                         {"profile": "/project/profile.yaml",
                          "nested": ["/run/x", 2]})

    def pass_gate(self, controller, stage):
        artifacts = {a["path"]: a["sha256"] for a in controller.stage_artifacts(stage)}
        vote_path = controller.run_dir / "gates" / f"{stage}.votes.json"
        criteria = controller.stage(stage).get("gate_criteria")
        lenses = controller.stage(stage).get("gate_review_lenses")
        self.assertTrue(criteria, "test workflow must bind gate criteria")
        self.assertEqual(len(lenses), 3, "test workflow must bind three review lenses")
        def vote(judge_id, lens):
            return {"judge_id": judge_id, "review_lens": lens["id"],
                    "verdict": "PASS", "criteria_checked":
                    [{"criterion": criterion, "value_read": "verified", "ok": True}
                     for criterion in criteria],
                    "rationale": "ok", "required_fix": ""}
        vote_path.write_text(json.dumps({"stage": stage, "criteria": criteria,
                                        "review_lenses": lenses,
                                        "artifact_sha256": artifacts, "decision": "PASS",
                                        "votes": [vote(f"judge-{index}", lens)
                                                  for index, lens in enumerate(lenses, 1)]}))
        controller.record_gate(stage, votes_path=vote_path)

    def write_vote_bundle(self, controller, stage, verdicts, *, runtime_failure=False):
        artifacts = {a["path"]: a["sha256"] for a in controller.stage_artifacts(stage)}
        vote_path = controller.run_dir / "gates" / f"{stage}.votes.json"
        criteria = controller.stage(stage).get("gate_criteria")
        lenses = controller.stage(stage).get("gate_review_lenses")
        votes = []
        for index, (verdict, lens) in enumerate(zip(verdicts, lenses), 1):
            ok = verdict == "PASS"
            rationale = "ok" if ok else "declared limitation needs human scope adjudication"
            required_fix = "" if ok else "human scope adjudication"
            if runtime_failure and index == len(verdicts):
                rationale = "Judge invocation failed"
                required_fix = "Re-run the judge"
            votes.append({
                "judge_id": f"judge-{index}", "review_lens": lens["id"],
                "verdict": verdict,
                "criteria_checked": [{"criterion": criterion, "value_read": "verified",
                                      "ok": ok} for criterion in criteria],
                "rationale": rationale, "required_fix": required_fix,
            })
        decision = "FAIL" if "FAIL" in verdicts else (
            "PASS" if verdicts == ["PASS"] * len(verdicts) else "REVISE")
        vote_path.write_text(json.dumps({
            "stage": stage, "criteria": criteria, "review_lenses": lenses,
            "artifact_sha256": artifacts, "decision": decision, "votes": votes,
        }))
        return vote_path

    def accept_limitation_proposal(self, path, stage="validation"):
        path.write_text(json.dumps({
            "schema_version": 1, "stage": stage,
            "proposed_decision": "ACCEPT_DECLARED_LIMITATION",
            "rationale": "The REVISE concerns an explicitly declared scope limitation.",
            "accepted_limitations": ["declared limitation remains NOT_ESTABLISHED"],
            "scope_effect": "restrict_scope_to_declared_limitations",
            "downstream_claim_restrictions": [
                "Do not claim applicability beyond the accepted declared limitation."
            ],
        }))
        return path

    def require_recovery_proposal(self, path, stage="validation"):
        path.write_text(json.dumps({
            "schema_version": 1, "stage": stage,
            "proposed_decision": "REQUIRE_SCIENTIFIC_RECOVERY",
            "rationale": "Human review requires additional scientific evidence.",
            "recovery_route": "use existing RecoveryPlan lifecycle",
        }))
        return path

    def completed_validation_controller(self, root, *, check_status="PASS"):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "adjudication", "stages": [
            {"name": "validation", "command": None, "outputs": ["artifacts/report.json"],
             "gate": {"criteria": [self.GATE_CRITERION]},
             "contract": {"kind": "validation_manifest", "manifest": "artifacts/report.json",
                          "validator": "validation.report.validate_validation_report",
                          "options": {"required_pass_observables": ["scope"]}}},
            {"name": "next", "command": [sys.executable, "-c",
             "from pathlib import Path; Path('artifacts/next.txt').write_text('ok')"],
             "outputs": ["artifacts/next.txt"]},
        ]}))
        controller = RunController.initialize(cfg, root / "run")
        report = controller.run_dir / "artifacts/report.json"
        evidence = controller.run_dir / "artifacts/evidence.txt"
        evidence.write_text("scope evidence")
        value = 1 if check_status == "PASS" else 0
        report.write_text(json.dumps({
            "schema_version": 1, "profile": "generic",
            "checks": [{"domain": "scope", "observable": "scope",
                        "status": check_status, "value": value, "unit": "none",
                        "criterion": {"operator": "equals", "target": 1}}],
            "evidence": [{"role": "scope_evidence", "path": str(evidence),
                          "integrity": artifact_digest(evidence)}],
        }))
        controller.complete_external_stage("validation", [report, evidence])
        return controller

    def test_gate_blocks_next_stage_and_state_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({
                "run_id": "test-run",
                "stages": [
                    {"name": "data", "command": [sys.executable, "-c",
                     "from pathlib import Path; Path('artifacts/data.txt').write_text('ok')"],
                     "outputs": ["artifacts/data.txt"],
                     "gate": {"criteria": [self.GATE_CRITERION]}},
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

    def test_bind_new_input_appends_hashed_snapshot_without_touching_existing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "student.yaml"
            source.write_text("kind: mock\n")
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "inputs": [str(source)],
                                           "stages": [{"name": "x", "command": None}]}))
            controller = RunController.initialize(cfg, root / "run")
            before_contract = controller.state["validation_contract"]
            before_stages = [dict(s) for s in controller.state["stages"]]

            new_source = root / "extra.json"
            new_source.write_text('{"a": 1}\n')
            record = controller.bind_new_input(new_source)

            self.assertEqual(len(controller.state["inputs"]), 2)
            self.assertEqual(controller.state["inputs"][0]["source"], str(source))
            self.assertEqual(record["source"], str(new_source.resolve()))
            self.assertTrue(Path(record["snapshot"]).is_file())
            self.assertEqual(Path(record["snapshot"]).name, "001-extra.json")
            self.assertEqual(record["sha256"], artifact_digest(new_source)["sha256"])
            self.assertEqual(controller.state["validation_contract"], before_contract)
            self.assertEqual([s["status"] for s in controller.state["stages"]],
                             [s["status"] for s in before_stages])
            self.assertEqual(controller.state["events"][-1]["type"], "input_bound")

    def test_bind_new_input_rejects_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x",
                                           "stages": [{"name": "x", "command": None}]}))
            controller = RunController.initialize(cfg, root / "run")
            with self.assertRaises(FileNotFoundError):
                controller.bind_new_input(root / "missing.json")
            self.assertEqual(len(controller.state["inputs"]), 0)

    def test_set_recovery_policy_accepts_mapping_and_none_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x",
                                           "stages": [{"name": "x", "command": None}]}))
            controller = RunController.initialize(cfg, root / "run")
            self.assertIsNone(controller.state.get("recovery_policy"))

            controller.set_recovery_policy({"max_recovery_attempts": 3})
            self.assertEqual(controller.state["recovery_policy"], {"max_recovery_attempts": 3})
            self.assertEqual(controller.state["events"][-1]["type"], "recovery_policy_updated")

            controller.set_recovery_policy(None)
            self.assertIsNone(controller.state["recovery_policy"])

            with self.assertRaisesRegex(ValueError, "mapping or None"):
                controller.set_recovery_policy(["not", "a", "mapping"])

    def test_failed_initialization_leaves_no_partial_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "inputs": ["missing.file"],
                                           "stages": [{"name": "x", "command": ["true"]}]}))
            run_dir = root / "run"
            with self.assertRaises(FileNotFoundError):
                RunController.initialize(cfg, run_dir)
            self.assertFalse(run_dir.exists())

    def test_initialization_rejects_outputs_outside_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "unsafe", "stages": [{
                "name": "work", "command": None, "outputs": ["../outside.txt"],
            }]}))
            with self.assertRaisesRegex(ValueError, "stay inside the run"):
                RunController.initialize(cfg, root / "run")
            self.assertFalse((root / "run").exists())

    def test_initialization_rejects_incomplete_external_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "bad-contract", "stages": [{
                "name": "validation", "command": None,
                "contract": {"kind": "validation_manifest",
                             "manifest": "artifacts/report.json"},
            }]}))
            with self.assertRaisesRegex(ValueError, "validator"):
                RunController.initialize(cfg, root / "run")
            self.assertFalse((root / "run").exists())

    def test_changed_project_code_blocks_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{
                "name": "x", "command": [sys.executable, "-c", "print(1)"]}]}))
            initial = {"available": True, "git_commit": "abc", "dirty": False,
                       "diff_sha256": None}
            changed = {"available": True, "git_commit": "def", "dirty": False,
                       "diff_sha256": None}
            with patch("workflow.controller.git_revision", return_value=initial):
                controller = RunController.initialize(cfg, root / "run")
            with patch("workflow.controller.git_revision", return_value=changed):
                with self.assertRaisesRegex(RuntimeError, "project code changed"):
                    controller.run_stage("x")

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
            self.assertIn("inputs_rebound", [e["type"] for e in controller.state["events"]])

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
            for stage in ("teacher_baseline", "data_coverage", "teacher_labeling",
                          "dataset_split", "training", "evaluation", "physical_validation"):
                controller.run_stage(stage)
                self.pass_gate(controller, stage)
            self.assertEqual(controller.stage("physical_validation")["gate"], "PASS")
            self.assertTrue((controller.run_dir / "artifacts/accuracy_report.json").is_file())
            self.assertTrue((controller.run_dir / "artifacts/validation_report.json").is_file())

    def test_modified_committee_checkpoint_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = RunController.initialize(ROOT / "examples/mock/workflow.yaml",
                                                  Path(tmp) / "run")
            for stage in ("teacher_baseline", "data_coverage", "teacher_labeling",
                          "dataset_split", "training"):
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
                 "outputs": ["artifacts/data"],
                 "gate": {"criteria": [self.GATE_CRITERION]}},
                {"name": "train", "command": [sys.executable, "-c",
                 "from pathlib import Path; Path('artifacts/model').write_text('m1')"],
                 "outputs": ["artifacts/model"],
                 "gate": {"criteria": [self.GATE_CRITERION]}}]}))
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
                "outputs": ["artifacts/x"],
                "gate": {"criteria": [self.GATE_CRITERION]}}]}))
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

    def test_gate_context_returns_verified_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "gate-context", "stages": [{
                "name": "data", "command": None,
                "outputs": ["artifacts/data.txt"],
                "gate": {"criteria": [self.GATE_CRITERION]},
            }]}))
            controller = RunController.initialize(cfg, root / "run")
            artifact = controller.run_dir / "artifacts/data.txt"
            artifact.write_text("data")
            controller.complete_external_stage("data", [artifact])
            context = controller.gate_context("data")
            self.assertEqual(context["stage"], "data")
            self.assertEqual(context["criteria"], [self.GATE_CRITERION])
            self.assertEqual(len(context["review_lenses"]), 3)
            self.assertEqual(context["artifact_sha256"],
                             {str(artifact): artifact_digest(artifact)["sha256"]})

            softened = "artifact exists"
            vote_path = controller.run_dir / "gates/softened.votes.json"
            votes = [{"judge_id": f"judge-{index}",
                      "review_lens": context["review_lenses"][index - 1]["id"],
                      "verdict": "PASS",
                      "criteria_checked": [{"criterion": softened,
                                            "value_read": "verified", "ok": True}],
                      "rationale": "ok", "required_fix": ""}
                     for index in range(1, 4)]
            vote_path.write_text(json.dumps({
                "stage": "data", "criteria": [softened],
                "review_lenses": context["review_lenses"],
                "artifact_sha256": context["artifact_sha256"],
                "decision": "PASS", "votes": votes,
            }))
            with self.assertRaisesRegex(ValueError, "run-bound gate criteria"):
                controller.record_gate("data", votes_path=vote_path)
            artifact.write_text("mutated")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                controller.gate_context("data")

    def test_synthetic_failed_judge_vote_records_revise_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "judge-failure", "stages": [{
                "name": "data", "command": None,
                "outputs": ["artifacts/data.txt"],
                "gate": {"criteria": ["artifact is complete"]},
            }]}))
            controller = RunController.initialize(cfg, root / "run")
            artifact = controller.run_dir / "artifacts/data.txt"
            artifact.write_text("data")
            controller.complete_external_stage("data", [artifact])
            criterion = "artifact is complete"
            lenses = controller.gate_context("data")["review_lenses"]
            passed = {"verdict": "PASS", "criteria_checked": [
                {"criterion": criterion, "value_read": "yes", "ok": True}],
                "rationale": "ok", "required_fix": ""}
            failed = {"verdict": "REVISE", "criteria_checked": [
                {"criterion": criterion, "value_read": "judge invocation failed",
                 "ok": False}], "rationale": "Judge invocation failed",
                "required_fix": "Re-run the judge"}
            bundle = controller.run_dir / "gates/data.votes.json"
            bundle.write_text(json.dumps({
                "gate": "data", "criteria": [criterion],
                "review_lenses": lenses,
                "artifact_sha256": controller.gate_context("data")["artifact_sha256"],
                "decision": "REVISE",
                "votes": [dict(passed, id=1, review_lens=lenses[0]["id"]),
                          dict(passed, id=2, review_lens=lenses[1]["id"]),
                          dict(failed, id=3, review_lens=lenses[2]["id"])],
            }))
            controller.record_gate("data", votes_path=bundle)
            self.assertEqual(controller.stage("data")["gate"], "REVISE")

    def test_revise_adjudication_accepts_declared_limitation_and_preserves_raw_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root)
            votes = self.write_vote_bundle(controller, "validation", ["PASS", "PASS", "REVISE"])
            controller.record_gate("validation", votes_path=votes)
            proposal = self.accept_limitation_proposal(root / "adjudication.json")
            adjudication = controller.propose_adjudication(proposal)
            self.assertEqual(adjudication["status"], "proposed")
            self.assertEqual(controller.stage("validation")["gate"], "REVISE")
            with self.assertRaisesRegex(RuntimeError, "recovery is pending"):
                controller.run_stage("next")
            controller.decide_adjudication("researcher", note="accept declared scope")
            self.assertIsNone(controller.state["pending_recovery"])
            self.assertEqual(controller.stage("validation")["gate"], "REVISE")
            self.assertEqual(controller.state["adjudications"][-1]["status"], "accepted")
            resolution = controller.stage("validation")["effective_resolution"]
            self.assertEqual(resolution["decision"], "ACCEPT_DECLARED_LIMITATION")
            self.assertEqual(resolution["raw_vote_verdicts"], ["PASS", "PASS", "REVISE"])
            self.assertIn("declared limitation", resolution["accepted_limitations"][0])
            controller.run_stage("next")
            self.assertEqual(controller.stage("next")["status"], "completed")

    def test_adjudication_requires_explicit_human_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root)
            votes = self.write_vote_bundle(controller, "validation", ["PASS", "PASS", "REVISE"])
            controller.record_gate("validation", votes_path=votes)
            controller.propose_adjudication(self.accept_limitation_proposal(root / "adj.json"))
            self.assertEqual(controller.state["pending_recovery"]["status"], "adjudication_proposed")
            with self.assertRaisesRegex(RuntimeError, "recovery is pending"):
                controller.run_stage("next")

    def test_adjudication_forbids_validator_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root, check_status="FAIL")
            votes = self.write_vote_bundle(controller, "validation", ["PASS", "PASS", "REVISE"])
            controller.record_gate("validation", votes_path=votes)
            with self.assertRaisesRegex(ValueError, "did not pass"):
                controller.propose_adjudication(self.accept_limitation_proposal(root / "adj.json"))

    def test_adjudication_forbids_fail_vote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root)
            votes = self.write_vote_bundle(controller, "validation", ["PASS", "REVISE", "FAIL"])
            controller.record_gate("validation", votes_path=votes)
            with self.assertRaisesRegex(RuntimeError, "only available for REVISE"):
                controller.propose_adjudication(self.accept_limitation_proposal(root / "adj.json"))

    def test_adjudication_forbids_missing_judge_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root)
            controller.record_gate("validation", "REVISE", evidence="manual revise")
            with self.assertRaisesRegex(RuntimeError, "vote record"):
                controller.propose_adjudication(self.accept_limitation_proposal(root / "adj.json"))

    def test_adjudication_forbids_runtime_failed_judge_vote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root)
            votes = self.write_vote_bundle(controller, "validation", ["PASS", "PASS", "REVISE"],
                                          runtime_failure=True)
            controller.record_gate("validation", votes_path=votes)
            with self.assertRaisesRegex(RuntimeError, "Judge runtime failure"):
                controller.propose_adjudication(self.accept_limitation_proposal(root / "adj.json"))

    def test_adjudication_requires_restrictive_scope_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root)
            votes = self.write_vote_bundle(controller, "validation", ["PASS", "PASS", "REVISE"])
            controller.record_gate("validation", votes_path=votes)
            proposal = root / "adj.json"
            proposal.write_text(json.dumps({
                "schema_version": 1, "stage": "validation",
                "proposed_decision": "ACCEPT_DECLARED_LIMITATION",
                "rationale": "expand",
                "accepted_limitations": ["limitation"],
                "scope_effect": "expand_applicability",
                "downstream_claim_restrictions": ["none"],
            }))
            with self.assertRaisesRegex(ValueError, "cannot expand"):
                controller.propose_adjudication(proposal)

    def test_adjudication_require_recovery_routes_to_existing_recovery_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self.completed_validation_controller(root)
            votes = self.write_vote_bundle(controller, "validation", ["PASS", "PASS", "REVISE"])
            controller.record_gate("validation", votes_path=votes)
            controller.propose_adjudication(self.require_recovery_proposal(root / "adj.json"))
            controller.decide_adjudication("researcher")
            self.assertEqual(controller.state["pending_recovery"]["status"], "required")
            plan = root / "recovery-plan.json"
            plan.write_text(json.dumps({
                "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
                "failure_category": "evidence_gap", "root_cause": "needs new evidence",
                "responsible_agent": "analyst", "return_stage": "validation",
                "proposed_changes": [{"type": "clarify_scope"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["validation"]},
                "estimated_cost": {},
            }))
            recovery = controller.propose_recovery(plan)
            self.assertEqual(recovery["status"], "proposed")

    def test_recovery_plan_requires_human_approval_and_starts_bound_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "recovery", "stages": [
                {"name": "data", "command": None, "outputs": ["artifacts/data.txt"],
                 "gate": {"criteria": [self.GATE_CRITERION]}},
                {"name": "teacher_labeling", "command": None,
                 "outputs": ["artifacts/labels.txt"],
                 "gate": {"criteria": [self.GATE_CRITERION]}},
                {"name": "training", "command": None,
                 "outputs": ["artifacts/model.txt"],
                 "gate": {"criteria": [self.GATE_CRITERION]}},
                {"name": "validation", "command": None,
                 "outputs": ["artifacts/validation.txt"],
                 "gate": {"criteria": [self.GATE_CRITERION]}},
            ]}))
            controller = RunController.initialize(cfg, root / "run")
            data = controller.run_dir / "artifacts/data.txt"
            data.write_text("data")
            controller.complete_external_stage("data", [data])
            self.pass_gate(controller, "data")
            labels = controller.run_dir / "artifacts/labels.txt"
            labels.write_text("labels")
            controller.complete_external_stage("teacher_labeling", [labels])
            self.pass_gate(controller, "teacher_labeling")
            model = controller.run_dir / "artifacts/model.txt"
            model.write_text("model")
            controller.complete_external_stage("training", [model])
            self.pass_gate(controller, "training")
            validation = controller.run_dir / "artifacts/validation.txt"
            validation.write_text("failed evidence")
            controller.complete_external_stage("validation", [validation])
            controller.record_gate("validation", "REVISE", evidence="coverage gap")
            with self.assertRaisesRegex(RuntimeError, "recovery is pending"):
                controller.complete_external_stage("validation", [validation])

            plan = root / "recovery-plan.json"
            plan.write_text(json.dumps({
                "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
                "failure_category": "dataset_coverage",
                "root_cause": "deployment liquid configurations are missing",
                "responsible_agent": "data-curator", "return_stage": "data",
                "proposed_changes": [{"type": "add_deployment_frames", "n_parents": 5}],
                "labeling": {"teacher_relabel": True, "new_dft": False},
                "student_training": {"retrain": True, "mode": "from_scratch"},
                "revalidation": {"reuse_profile": True, "targets": ["validation"]},
                "estimated_cost": {"gpu_hours": 1},
            }))
            recovery = controller.propose_recovery(plan)
            self.assertEqual(recovery["gate_binding"]["artifact_sha256"],
                             {str(validation): artifact_digest(validation)["sha256"]})
            with self.assertRaisesRegex(RuntimeError, "approved"):
                controller.start_iteration()
            controller.approve_recovery("researcher", "approved pilot")
            controller.start_iteration()
            self.assertIsNone(controller.state["pending_recovery"])
            self.assertEqual(controller.state["iterations"][-1]["id"], 2)
            self.assertEqual(controller.state["recoveries"][-1]["status"], "activated")
            self.assertEqual(controller.stage("data")["status"], "pending")
            self.assertEqual(controller.stage("validation")["status"], "pending")
            data.write_text("data iteration 2")
            controller.complete_external_stage("data", [data])
            self.pass_gate(controller, "data")
            labels.write_text("labels iteration 2")
            controller.complete_external_stage("teacher_labeling", [labels])
            self.pass_gate(controller, "teacher_labeling")
            model.write_text("model iteration 2")
            controller.complete_external_stage("training", [model])
            self.pass_gate(controller, "training")
            validation.write_text("revalidated evidence")
            controller.complete_external_stage("validation", [validation])
            with self.assertRaisesRegex(RuntimeError, "recovery execution"):
                self.pass_gate(controller, "validation")
            execution = root / "recovery-execution.json"
            execution.write_text(json.dumps({
                "schema_version": 1, "recovery_id": recovery["id"],
                "previous_iteration": 1, "current_iteration": 2,
                "changes": [{"type": "add_deployment_frames", "status": "APPLIED",
                             "evidence_artifacts": ["artifacts/data.txt"]}],
                "labeling": {"teacher_relabel": True,
                             "teacher_relabel_stage": "teacher_labeling",
                             "new_dft": False, "new_dft_stage": None},
                "student_training": {"retrain": True, "mode": "from_scratch",
                                     "stage": "training"},
                "revalidation": {"targets": ["validation"],
                                 "stages": ["validation"]},
            }))
            controller.verify_recovery_execution(execution)
            self.pass_gate(controller, "validation")
            self.assertEqual(controller.stage("validation")["gate"], "PASS")
            self.assertEqual(controller.state["recoveries"][-1]["status"], "resolved")

    def test_mutated_recovery_proposal_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "recovery-integrity", "stages": [{
                "name": "validation", "command": None,
                "outputs": ["artifacts/result.txt"],
            }]}))
            controller = RunController.initialize(cfg, root / "run")
            result = controller.run_dir / "artifacts/result.txt"
            result.write_text("result")
            controller.complete_external_stage("validation", [result])
            controller.record_gate("validation", "FAIL")
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
                "failure_category": "physical_validation", "root_cause": "unstable",
                "responsible_agent": "simulation", "return_stage": "validation",
                "proposed_changes": [{"type": "fix_protocol"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["stability"]},
                "estimated_cost": {},
            }))
            recovery = controller.propose_recovery(plan)
            Path(recovery["path"]).write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                controller.approve_recovery("researcher")

    def test_recovery_execution_rejects_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "unchanged-recovery", "stages": [{
                "name": "validation", "command": None,
                "outputs": ["artifacts/result.txt"],
                "gate": {"criteria": [self.GATE_CRITERION]},
            }]}))
            controller = RunController.initialize(cfg, root / "run")
            result = controller.run_dir / "artifacts/result.txt"
            result.write_text("same")
            controller.complete_external_stage("validation", [result])
            controller.record_gate("validation", "REVISE")
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
                "failure_category": "physical_validation", "root_cause": "revise protocol",
                "responsible_agent": "simulation", "return_stage": "validation",
                "proposed_changes": [{"type": "fix_protocol"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["stability"]},
                "estimated_cost": {},
            }))
            recovery = controller.propose_recovery(plan)
            controller.approve_recovery("researcher")
            controller.start_iteration()
            result.write_text("same")
            controller.complete_external_stage("validation", [result])
            execution = root / "execution.json"
            execution.write_text(json.dumps({
                "schema_version": 1, "recovery_id": recovery["id"],
                "previous_iteration": 1, "current_iteration": 2,
                "changes": [{"type": "fix_protocol", "status": "APPLIED",
                             "evidence_artifacts": ["artifacts/result.txt"]}],
                "labeling": {"teacher_relabel": False, "teacher_relabel_stage": None,
                             "new_dft": False, "new_dft_stage": None},
                "student_training": {"retrain": False, "mode": "none", "stage": None},
                "revalidation": {"targets": ["stability"], "stages": ["validation"]},
            }))
            with self.assertRaisesRegex(ValueError, "did not change artifacts"):
                controller.verify_recovery_execution(execution)

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

    def test_deterministic_stage_contract_fails_before_artifact_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            invalid_report = {"schema_version": 1, "profile": "generic",
                              "checks": [], "evidence": []}
            encoded = json.dumps(invalid_report).replace("{", "{{").replace("}", "}}")
            command = ("from pathlib import Path; "
                       f"Path('artifacts/report.json').write_text({encoded!r})")
            cfg.write_text(yaml.safe_dump({
                "run_id": "contract-failure",
                "stages": [{
                    "name": "validation", "command": [sys.executable, "-c", command],
                    "outputs": ["artifacts/report.json"],
                    "contract": {"kind": "validation_manifest",
                                 "manifest": "artifacts/report.json",
                                 "validator": "validation.report.validate_validation_report"},
                }],
            }))
            controller = RunController.initialize(cfg, root / "run")
            with self.assertRaisesRegex(ValueError, "non-empty checks"):
                controller.run_stage("validation")
            self.assertEqual(controller.stage("validation")["status"], "failed")
            self.assertEqual(controller.stage_artifacts("validation"), [])

    def test_failed_required_observable_is_recorded_but_cannot_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "trajectory.xyz"
            evidence.write_text("trajectory")
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({
                "run_id": "validation-pass-policy",
                "inputs": [str(evidence)],
                "stages": [{
                    "name": "validation", "command": None,
                    "outputs": ["artifacts/report.json"],
                    "gate": {"criteria": [self.GATE_CRITERION]},
                    "contract": {"kind": "validation_manifest",
                                 "manifest": "artifacts/report.json",
                                 "validator": "validation.report.validate_validation_report",
                                 "options": {
                                     "required_observables": ["density"],
                                     "required_pass_observables": ["density"],
                                 }},
                }],
            }))
            controller = RunController.initialize(cfg, root / "run")
            report = controller.run_dir / "artifacts" / "report.json"
            report.write_text(json.dumps({
                "schema_version": 1, "profile": "generic",
                "checks": [{"domain": "structure", "observable": "density",
                            "status": "FAIL", "value": 2.0, "unit": "g/cm3",
                            "criterion": {"operator": "max", "threshold": 1.0}}],
                "evidence": [{"role": "trajectory", "path": str(evidence),
                              "integrity": artifact_digest(evidence)}],
            }))
            controller.complete_external_stage("validation", [report])
            self.assertEqual(controller.stage("validation")["status"], "completed")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                self.pass_gate(controller, "validation")
            self.assertEqual(controller.stage("validation")["status"], "completed")
            self.assertEqual(controller.stage("validation")["gate"], "pending")

    def test_external_stage_requires_its_declared_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{
                "name": "analysis", "command": None,
                "outputs": ["artifacts/final_report.json"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            other = controller.run_dir / "artifacts" / "other.txt"
            other.write_text("not the declared report")
            with self.assertRaisesRegex(ValueError, "missing declared outputs"):
                controller.complete_external_stage("analysis", [other])
            self.assertEqual(controller.stage("analysis")["status"], "pending")

    def test_missing_executable_records_failed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{
                "name": "broken", "command": ["definitely-missing-binary-xyz"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            with self.assertRaisesRegex(RuntimeError, "could not be launched"):
                controller.run_stage("broken")
            resumed = RunController(controller.run_dir)
            self.assertEqual(resumed.stage("broken")["status"], "failed")
            self.assertEqual(resumed.state["events"][-1]["type"], "stage_failed")

    def test_external_md_is_bound_to_approved_committee_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [
                {"name": "training", "command": None,
                 "gate": {"criteria": [self.GATE_CRITERION]}},
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
                {"name": "training", "command": None,
                 "gate": {"criteria": [self.GATE_CRITERION]}},
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
                {"name": "training", "command": None,
                 "gate": {"criteria": [self.GATE_CRITERION]}},
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

    def test_failed_attempt_output_cannot_satisfy_a_successful_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            command = ("from pathlib import Path; import sys; "
                       "marker=Path('attempted'); output=Path('artifacts/result.txt'); "
                       "first=not marker.exists(); "
                       "output.write_text('partial') if first else None; "
                       "marker.write_text('yes'); sys.exit(1 if first else 0)")
            cfg.write_text(yaml.safe_dump({"run_id": "retry", "stages": [{
                "name": "work", "command": [sys.executable, "-c", command],
                "outputs": ["artifacts/result.txt"],
            }]}))
            controller = RunController.initialize(cfg, root / "run")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                controller.run_stage("work")
            with self.assertRaisesRegex(FileNotFoundError, "declared output missing"):
                controller.run_stage("work")
            self.assertEqual(controller.stage_artifacts("work"), [])
            self.assertFalse((controller.run_dir / "artifacts/result.txt").exists())
            self.assertTrue(any((controller.run_dir / "stale").rglob("result.txt")))

    def test_failed_rebind_does_not_poison_the_revision_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first.yaml", root / "second.yaml"
            first.write_text("value: 1\n"); second.write_text("value: 2\n")
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "rebind", "inputs": [
                {"path": str(first)}, {"path": str(second)}],
                "stages": [{"name": "work", "command": None}],
            }))
            controller = RunController.initialize(cfg, root / "run")
            second.unlink()
            with self.assertRaises(FileNotFoundError):
                controller.rebind_inputs()
            self.assertFalse((controller.run_dir / "inputs/revision-001").exists())
            second.write_text("value: 3\n")
            changes = controller.rebind_inputs()
            self.assertEqual(len(changes), 2)
            self.assertTrue((controller.run_dir / "inputs/revision-001").is_dir())

    def test_external_stage_can_replace_an_artifact_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "replace", "stages": [{
                "name": "external", "command": None,
                "outputs": ["artifacts/result.txt"],
            }]}))
            controller = RunController.initialize(cfg, root / "run")
            artifact = controller.run_dir / "artifacts/result.txt"
            artifact.write_text("first")
            controller.complete_external_stage("external", [artifact])
            artifact.write_text("second")
            controller.complete_external_stage("external", [artifact])
            self.assertEqual(artifact.read_text(), "second")
            self.assertEqual(len(controller.stage_artifacts("external")), 1)
            self.assertEqual(controller.stage("external")["attempts"], 2)

    def test_validation_retry_cannot_reuse_previous_same_stage_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "validation-retry", "stages": [{
                "name": "validation", "command": None,
                "outputs": ["artifacts/report.json"],
                "contract": {"kind": "validation_manifest",
                             "manifest": "artifacts/report.json",
                             "validator": "validation.report.validate_validation_report"},
            }]}))
            controller = RunController.initialize(cfg, root / "run")
            report = controller.run_dir / "artifacts/report.json"
            evidence = controller.run_dir / "artifacts/trajectory.dat"
            evidence.write_text("first attempt")

            def write_report():
                report.write_text(json.dumps({
                    "schema_version": 1, "profile": "generic",
                    "checks": [{"domain": "structure", "observable": "x",
                                "status": "RECORDED", "value": 1, "unit": "a.u.",
                                "criterion": None}],
                    "evidence": [{"role": "trajectory", "path": str(evidence),
                                  "integrity": artifact_digest(evidence)}],
                }))

            write_report()
            controller.complete_external_stage("validation", [report, evidence])
            write_report()
            with self.assertRaisesRegex(ValueError, "not bound to this run"):
                controller.complete_external_stage("validation", [report])
            self.assertEqual(controller.stage("validation")["status"], "completed")
            self.assertEqual(controller.stage("validation")["attempts"], 1)

    def test_validation_evidence_allowlist_excludes_downstream_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "evidence-order", "stages": [
                {"name": "upstream", "command": None},
                {"name": "validation", "command": None},
                {"name": "downstream", "command": None},
            ]}))
            controller = RunController.initialize(cfg, root / "run")
            upstream = root / "upstream.dat"
            downstream = root / "downstream.dat"
            current = root / "current.json"
            upstream.write_text("upstream")
            downstream.write_text("downstream")
            current.write_text("current")
            controller.register_artifact("upstream", upstream)
            controller.register_artifact("downstream", downstream)
            allowed = set(controller._validation_evidence_allowlist(
                [current], "validation"
            ))
            self.assertIn(upstream.resolve(), allowed)
            self.assertIn(current.resolve(), allowed)
            self.assertNotIn(downstream.resolve(), allowed)


if __name__ == "__main__":
    unittest.main()
