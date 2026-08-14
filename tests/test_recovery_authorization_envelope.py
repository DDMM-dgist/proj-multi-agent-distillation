"""Priority #3 requirement #3: RecoveryAuthorizationEnvelope enforcement, end to end.

Proves approving a RecoveryPlan (approve_recovery) never by itself authorizes a costly child
action: dispatch.authorize_and_execute (via controller_bridge.dispatch_via_controller) only
accepts a recovery-authorized action when a SEPARATE, explicit, hash-bound
RecoveryAuthorizationEnvelope (created by RunController.authorize_recovery_capabilities) actually
covers that exact action_type/resource-usage/artifact-role combination -- and that an envelope
scoped to something else, or missing entirely, never widens APPROVAL_GATED_ACTIONS; a normal
per-action approval keeps working unchanged regardless of any envelope.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
from workflow.controller import RunController


def _proposal(**overrides):
    proposal = {
        "requested_by_role": "data-curator", "action_type": "label_with_teacher",
        "idempotency_key": "k1", "run_id": "envelope-test", "stage": "s",
        "requested_at": "t", "rationale": "because",
    }
    proposal.update(overrides)
    return proposal


class RecoveryAuthorizationEnvelopeTests(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _activated_recovery_controller(self, root, *, protected_reference_roles=None):
        cfg_dict = {"run_id": "envelope-test", "stages": [{
            "name": "validation", "command": None, "outputs": ["artifacts/result.txt"],
            "gate": {"criteria": [self.GATE_CRITERION]},
        }]}
        if protected_reference_roles is not None:
            cfg_dict["protected_reference_roles"] = protected_reference_roles
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump(cfg_dict))
        controller = RunController.initialize(cfg, root / "run")
        result = controller.run_dir / "artifacts/result.txt"
        result.write_text("result")
        controller.complete_external_stage("validation", [result])
        controller.record_gate("validation", "REVISE")
        plan = root / "plan.json"
        plan.write_text(json.dumps({
            "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
            "failure_category": "physical_validation", "root_cause": "needs teacher relabel",
            "responsible_agent": "data-curator", "return_stage": "validation",
            "proposed_changes": [{"type": "relabel"}],
            "labeling": {"teacher_relabel": True, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["validation"]},
            "estimated_cost": {},
        }))
        controller.propose_recovery(plan)
        controller.approve_recovery("researcher", "approved pilot")
        controller.start_iteration()
        return controller

    def test_approving_the_recovery_never_authorizes_the_child_action_by_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._activated_recovery_controller(Path(tmp))
            outcome = dispatch_via_controller(_proposal(), controller=controller, mode="primary")
            self.assertEqual(outcome.status, "APPROVAL_REQUIRED")
            self.assertFalse(outcome.executed)

    def test_matching_envelope_authorizes_the_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._activated_recovery_controller(Path(tmp))
            envelope = controller.authorize_recovery_capabilities(
                "researcher", action_types=["label_with_teacher"])
            outcome = dispatch_via_controller(_proposal(), controller=controller, mode="primary")
            self.assertEqual(outcome.status, "DRY_RUN")  # no inline executor registered
            self.assertEqual(outcome.recovery_authorization_envelope_sha256,
                             envelope["envelope_sha256"])

    def test_envelope_scoped_to_a_different_action_type_never_widens(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._activated_recovery_controller(Path(tmp))
            controller.authorize_recovery_capabilities(
                "researcher", action_types=["some_other_action"])
            outcome = dispatch_via_controller(_proposal(), controller=controller, mode="primary")
            self.assertEqual(outcome.status, "APPROVAL_REQUIRED")

    def test_envelope_resource_limit_exceeded_falls_back_to_approval_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._activated_recovery_controller(Path(tmp))
            controller.authorize_recovery_capabilities(
                "researcher", action_types=["label_with_teacher"],
                resource_limits={"gpu_hours": 1})
            proposal = _proposal(parameters={"resource_usage": {"gpu_hours": 5}})
            outcome = dispatch_via_controller(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "APPROVAL_REQUIRED")

    def test_envelope_within_resource_limit_authorizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._activated_recovery_controller(Path(tmp))
            controller.authorize_recovery_capabilities(
                "researcher", action_types=["label_with_teacher"],
                resource_limits={"gpu_hours": 10})
            proposal = _proposal(parameters={"resource_usage": {"gpu_hours": 5}})
            outcome = dispatch_via_controller(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "DRY_RUN")

    def test_envelope_cannot_permit_a_protected_reference_artifact_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._activated_recovery_controller(
                Path(tmp), protected_reference_roles=["teacher_train_partition"])
            with self.assertRaisesRegex(ValueError, "protected-reference artifact"):
                controller.authorize_recovery_capabilities(
                    "researcher", action_types=["label_with_teacher"],
                    permitted_artifact_roles=["teacher_train_partition"])

    def test_normal_per_action_approval_still_works_without_any_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._activated_recovery_controller(Path(tmp))
            controller.grant_action_approval("costly_teacher_labeling")
            outcome = dispatch_via_controller(_proposal(), controller=controller, mode="primary")
            self.assertEqual(outcome.status, "DRY_RUN")
            self.assertIsNone(outcome.recovery_authorization_envelope_sha256)


if __name__ == "__main__":
    unittest.main()
