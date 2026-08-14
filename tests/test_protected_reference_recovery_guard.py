"""Priority #3 requirement #7: protected-reference isolation stays fail-closed through recovery.

Proves RunController._validate_protected_reference_roles (called from propose_recovery) rejects
a recovery plan that routes a run-declared protected-reference artifact role into a
training/acquisition input, output, OR a nested proposed_changes[*].artifact_roles entry (e.g. one
produced by runtimes.pydantic_ai.acquisition_targeting.AcquisitionTargetProposal/
DataRepairProposal) -- unless a separate, explicit protected_reference_reuse_authorization is
attached to that same plan -- and that an unrelated plan (no protected role touched anywhere)
proposes and approves exactly like before this guard existed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from runtimes.pydantic_ai.acquisition_targeting import AcquisitionTargetProposal, DataRepairProposal
from workflow.controller import RunController

_EVIDENCE = [{"path": "evidence/coverage.json", "sha256": "a" * 64}]


class ProtectedReferenceRecoveryGuardTests(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _controller(self, root, *, protected_reference_roles):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({
            "run_id": "protected-ref-guard",
            "protected_reference_roles": protected_reference_roles,
            "stages": [{
                "name": "validation", "command": None,
                "outputs": ["artifacts/result.txt"],
                "gate": {"criteria": [self.GATE_CRITERION]},
            }],
        }))
        controller = RunController.initialize(cfg, root / "run")
        result = controller.run_dir / "artifacts/result.txt"
        result.write_text("result")
        controller.complete_external_stage("validation", [result])
        controller.record_gate("validation", "REVISE")
        return controller

    def _base_plan(self, proposed_changes):
        return {
            "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
            "failure_category": "dataset_coverage", "root_cause": "low support in slice",
            "responsible_agent": "data-curator", "return_stage": "validation",
            "proposed_changes": proposed_changes,
            "labeling": {"teacher_relabel": False, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["validation"]},
            "estimated_cost": {},
        }

    def test_unrelated_plan_is_unaffected_by_the_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, protected_reference_roles=["teacher_train_partition"])
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(self._base_plan([{"type": "add_deployment_frames"}])))
            recovery = controller.propose_recovery(plan_path)
            self.assertEqual(recovery["status"], "proposed")

    def test_top_level_role_touching_protected_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, protected_reference_roles=["teacher_train_partition"])
            plan = self._base_plan([{"type": "add_deployment_frames"}])
            plan["required_input_artifact_roles"] = ["teacher_train_partition"]
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "protected-reference artifact role"):
                controller.propose_recovery(plan_path)

    def test_acquisition_target_proposal_artifact_role_alone_is_caught(self):
        # The protected role is declared ONLY inside proposed_changes[0].artifact_roles -- never
        # lifted into the plan's own top-level required_input_artifact_roles/
        # expected_output_artifact_roles -- which is exactly the gap this revision closes.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, protected_reference_roles=["teacher_train_partition"])
            target = AcquisitionTargetProposal(
                target_population="candidate_population", target_direction="teacher_support",
                rationale="low support fraction in slice", evidence_refs=_EVIDENCE,
                artifact_roles=["teacher_train_partition"],
            )
            plan = self._base_plan([target.to_proposed_change()])
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "protected-reference artifact role"):
                controller.propose_recovery(plan_path)

    def test_data_repair_proposal_artifact_role_alone_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, protected_reference_roles=["teacher_train_partition"])
            repair = DataRepairProposal(
                defect_description="mislabeled teacher forces", rationale="NaN forces detected",
                affected_artifact_refs=[{"path": "artifacts/labels.json", "sha256": "b" * 64}],
                evidence_refs=_EVIDENCE, artifact_roles=["teacher_train_partition"],
            )
            plan = self._base_plan([repair.to_proposed_change()])
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "protected-reference artifact role"):
                controller.propose_recovery(plan_path)

    def test_valid_override_lets_the_same_plan_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, protected_reference_roles=["teacher_train_partition"])
            target = AcquisitionTargetProposal(
                target_population="candidate_population", target_direction="teacher_support",
                rationale="low support fraction in slice", evidence_refs=_EVIDENCE,
                artifact_roles=["teacher_train_partition"],
            )
            plan = self._base_plan([target.to_proposed_change()])
            plan["protected_reference_reuse_authorization"] = {
                "authorized_by": "researcher", "rationale": "one-off approved reuse for audit",
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            recovery = controller.propose_recovery(plan_path)
            self.assertEqual(recovery["status"], "proposed")

    def test_no_protected_reference_roles_declared_means_no_guard_at_all(self):
        # A run that never declares protected_reference_roles behaves exactly as before v12 --
        # proposed_changes artifact_roles are inert.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, protected_reference_roles=[])
            target = AcquisitionTargetProposal(
                target_population="candidate_population", target_direction="teacher_support",
                rationale="low support fraction in slice", evidence_refs=_EVIDENCE,
                artifact_roles=["teacher_train_partition"],
            )
            plan = self._base_plan([target.to_proposed_change()])
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            recovery = controller.propose_recovery(plan_path)
            self.assertEqual(recovery["status"], "proposed")


if __name__ == "__main__":
    unittest.main()
