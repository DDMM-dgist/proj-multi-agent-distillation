"""Priority #3 requirement #2 (fail-closed provenance binding): propose_recovery hash-verifies an
attached diagnosis_binding (the Analyst -> RecoveryPlan bridge's provenance record, see
runtimes.pydantic_ai.recovery_bridge.DiagnosisBinding) against the actual artifact/evidence
content on disk -- a stale or missing diagnosis artifact must fail closed, not silently pass
through. diagnosis_binding itself stays OPTIONAL: a historical/manual plan that never attaches one
is unaffected.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController
from workflow.integrity import sha256_file


class DiagnosisBindingTests(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _controller(self, root):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "diagnosis-binding", "stages": [{
            "name": "validation", "command": None, "outputs": ["artifacts/result.txt"],
            "gate": {"criteria": [self.GATE_CRITERION]},
        }]}))
        controller = RunController.initialize(cfg, root / "run")
        result = controller.run_dir / "artifacts/result.txt"
        result.write_text("result")
        controller.complete_external_stage("validation", [result])
        controller.record_gate("validation", "REVISE")
        return controller

    def _base_plan(self, **overrides):
        plan = {
            "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
            "failure_category": "dataset_coverage", "root_cause": "low support in slice",
            "responsible_agent": "data-curator", "return_stage": "validation",
            "proposed_changes": [{"type": "add_deployment_frames"}],
            "labeling": {"teacher_relabel": False, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["validation"]},
            "estimated_cost": {},
        }
        plan.update(overrides)
        return plan

    def _propose(self, root, controller, plan):
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        return controller.propose_recovery(plan_path)

    def test_absent_diagnosis_binding_is_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            recovery = self._propose(root, controller, self._base_plan())
            self.assertEqual(recovery["status"], "proposed")

    def test_matching_diagnosis_binding_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            diagnosis = controller.run_dir / "diagnosis.json"
            diagnosis.write_text(json.dumps({"root_cause": "low support"}))
            evidence = controller.run_dir / "artifacts/result.txt"
            plan = self._base_plan(diagnosis_binding={
                "diagnosis_artifact_path": str(diagnosis),
                "diagnosis_artifact_sha256": sha256_file(diagnosis),
                "triggering_evidence": [
                    {"path": str(evidence), "sha256": sha256_file(evidence)},
                ],
            })
            recovery = self._propose(root, controller, plan)
            self.assertEqual(recovery["status"], "proposed")

    def test_diagnosis_artifact_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            diagnosis = controller.run_dir / "diagnosis.json"
            diagnosis.write_text(json.dumps({"root_cause": "low support"}))
            stale_sha256 = sha256_file(diagnosis)
            diagnosis.write_text(json.dumps({"root_cause": "edited after diagnosis"}))
            plan = self._base_plan(diagnosis_binding={
                "diagnosis_artifact_path": str(diagnosis),
                "diagnosis_artifact_sha256": stale_sha256,
                "triggering_evidence": [],
            })
            with self.assertRaisesRegex(ValueError, "missing or hash-mismatched"):
                self._propose(root, controller, plan)

    def test_diagnosis_artifact_path_missing_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan = self._base_plan(diagnosis_binding={
                "diagnosis_artifact_path": "nonexistent/diagnosis.json",
                "diagnosis_artifact_sha256": "a" * 64,
                "triggering_evidence": [],
            })
            with self.assertRaisesRegex(ValueError, "missing or hash-mismatched"):
                self._propose(root, controller, plan)

    def test_triggering_evidence_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            diagnosis = controller.run_dir / "diagnosis.json"
            diagnosis.write_text(json.dumps({"root_cause": "low support"}))
            evidence = controller.run_dir / "artifacts/result.txt"
            plan = self._base_plan(diagnosis_binding={
                "diagnosis_artifact_path": str(diagnosis),
                "diagnosis_artifact_sha256": sha256_file(diagnosis),
                "triggering_evidence": [{"path": str(evidence), "sha256": "b" * 64}],
            })
            with self.assertRaisesRegex(ValueError, "triggering_evidence"):
                self._propose(root, controller, plan)


if __name__ == "__main__":
    unittest.main()
