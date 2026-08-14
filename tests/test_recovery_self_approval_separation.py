"""Explicit separation of recovery proposal authority and approval authority.

The prior Priority #3 acceptance criteria required that a recovery proposal can never
self-approve; this was not implemented and is closed here. RunController.propose_recovery now
requires and records a provenance-bound proposer identity (workflow.actor_identity.ActorIdentity)
on every new proposal -- including a hand-authored manual plan.json, which is still a proposal
and is not exempt. approve_recovery and authorize_recovery_capabilities both then enforce, fail
closed:

  (1) the approving/authorizing actor must resolve to actor_kind == "human" -- an automated
      Agent/System actor can never satisfy either requirement regardless of what string or
      structured identity it supplies;
  (2) if the recovery has a recorded proposer identity, the approving/authorizing actor's
      canonical_id (whitespace/case-normalized, never a raw display-string `==`) must differ
      from it -- the same canonical actor cannot both propose and approve/authorize the same
      recovery.

No permissive same-human propose+approve mode exists anywhere in this framework, so none is
introduced here: every path in this file that has the same identity on both sides is rejected.
A historical recovery record that pre-dates this feature (no recorded proposer identity) skips
ONLY check (2); check (1) still applies unconditionally, so it never becomes silently
self-approved merely by lacking a proposer.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController


class _SelfApprovalFixture(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _controller(self, root):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "self-approval", "stages": [{
            "name": "validation", "command": None, "outputs": ["artifacts/result.txt"],
            "gate": {"criteria": [self.GATE_CRITERION]},
        }]}))
        controller = RunController.initialize(cfg, root / "run")
        result = controller.run_dir / "artifacts/result.txt"
        result.write_text("result")
        controller.complete_external_stage("validation", [result])
        controller.record_gate("validation", "REVISE")
        return controller

    def _base_plan(self, proposed_by, **overrides):
        plan = {
            "schema_version": 1, "proposed_by": proposed_by, "failed_stage": "validation",
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

    def _propose(self, root, controller, plan, name="plan.json"):
        plan_path = root / name
        plan_path.write_text(json.dumps(plan))
        return controller.propose_recovery(plan_path)


class ProposerIdentityIsRecordedTests(_SelfApprovalFixture):
    def test_propose_recovery_requires_proposed_by(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan = self._base_plan("researcher")
            del plan["proposed_by"]
            with self.assertRaisesRegex(ValueError, "proposed_by"):
                self._propose(root, controller, plan)

    def test_propose_recovery_records_the_resolved_proposer_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            recovery = self._propose(root, controller, self._base_plan("researcher"))
            self.assertEqual(recovery["proposed_by"],
                             {"actor_kind": "human", "canonical_id": "researcher",
                              "display_name": "researcher"})


class Scenario1AgentProposerTests(_SelfApprovalFixture):
    """1. agent proposes -> same agent cannot approve."""

    def test_same_agent_identity_cannot_approve_its_own_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan(
                {"actor_kind": "agent", "canonical_id": "data-curator"}))
            with self.assertRaisesRegex(ValueError, "cannot both propose and approve"):
                controller.approve_recovery("data-curator")


class Scenario2SystemProposerTests(_SelfApprovalFixture):
    """2. system/orchestrator proposes -> cannot self-approve."""

    def test_system_orchestrator_proposer_cannot_self_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan(
                {"actor_kind": "system", "canonical_id": "orchestrator"}))
            with self.assertRaisesRegex(ValueError, "cannot both propose and approve"):
                controller.approve_recovery("orchestrator")

    def test_an_agent_or_system_actor_can_never_itself_satisfy_human_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan("researcher"))
            with self.assertRaisesRegex(ValueError, "human approval actor"):
                controller.approve_recovery({"actor_kind": "system", "canonical_id": "lab-lead"})


class Scenario3And4HumanIdentityTests(_SelfApprovalFixture):
    """3. human proposer == human approver -> rejected. 4. different human approver -> accepted."""

    def test_same_human_identity_is_rejected_even_with_different_casing_or_whitespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan("researcher"))
            # Deliberately NOT an exact string match -- proves this is canonical-identity
            # equality (casefold + strip), not a fragile display-string comparison.
            with self.assertRaisesRegex(ValueError, "cannot both propose and approve"):
                controller.approve_recovery("  Researcher  ")

    def test_different_authorized_human_approver_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan("researcher"))
            recovery = controller.approve_recovery("lab-lead", "approved by a different reviewer")
            self.assertEqual(recovery["status"], "approved")
            self.assertEqual(recovery["human_approval"]["approved_by"],
                             {"actor_kind": "human", "canonical_id": "lab-lead",
                              "display_name": "lab-lead"})


class Scenario5And6EnvelopeAuthorizationTests(_SelfApprovalFixture):
    """5. no envelope before valid human approval. 6. invalid/self approval cannot authorize a
    child action."""

    def test_envelope_cannot_be_authorized_before_any_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan("researcher"))
            with self.assertRaisesRegex(RuntimeError, "no activated recovery"):
                controller.authorize_recovery_capabilities(
                    "lab-lead", action_types=["label_with_teacher"])

    def test_rejected_self_approval_never_advances_state_so_no_envelope_can_follow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            recovery = self._propose(root, controller, self._base_plan("researcher"))
            with self.assertRaises(ValueError):
                controller.approve_recovery("researcher")
            self.assertEqual(recovery["status"], "proposed")
            self.assertEqual(controller.state["pending_recovery"]["status"], "proposed")
            with self.assertRaisesRegex(RuntimeError, "no recovery is waiting in 'approved'"):
                controller.start_iteration()
            with self.assertRaisesRegex(RuntimeError, "no activated recovery"):
                controller.authorize_recovery_capabilities(
                    "lab-lead", action_types=["label_with_teacher"])

    def test_the_original_proposer_cannot_self_issue_the_envelope_even_after_a_valid_approval(self):
        # A different human validly approves the recovery (satisfying approve_recovery's own
        # check), but the ORIGINAL proposing actor then tries to issue the authorization
        # envelope itself -- this is a distinct, independent guard inside
        # authorize_recovery_capabilities, not merely a side effect of approve_recovery's check.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan("researcher"))
            controller.approve_recovery("lab-lead", "approved by a different reviewer")
            controller.start_iteration()
            with self.assertRaisesRegex(ValueError, "never be self-issued"):
                controller.authorize_recovery_capabilities(
                    "researcher", action_types=["label_with_teacher"])

    def test_a_valid_different_human_authorizer_can_issue_the_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self._propose(root, controller, self._base_plan("researcher"))
            controller.approve_recovery("lab-lead", "approved by a different reviewer")
            controller.start_iteration()
            envelope = controller.authorize_recovery_capabilities(
                "lab-lead", action_types=["label_with_teacher"])
            self.assertEqual(envelope["authorized_by"],
                             {"actor_kind": "human", "canonical_id": "lab-lead",
                              "display_name": "lab-lead"})


class Scenario7BackwardCompatibilityTests(_SelfApprovalFixture):
    """7. historical manifests without proposer-identity fields remain readable/backward
    compatible and do not silently become self-approved."""

    def test_a_recovery_record_missing_proposed_by_still_completes_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            recovery = self._propose(root, controller, self._base_plan("researcher"))
            # Simulate a recovery proposed under pre-feature code: the on-disk recovery
            # artifact and its recorded integrity hash are untouched, only the in-memory (and
            # on-disk manifest, via save()) record's proposed_by field is absent, exactly as a
            # historical manifest predating this feature would be.
            del recovery["proposed_by"]
            controller.save()
            reloaded = RunController(root / "run")
            self.assertNotIn("proposed_by", reloaded.state["recoveries"][-1])
            approved = reloaded.approve_recovery("researcher")
            self.assertEqual(approved["status"], "approved")

    def test_missing_proposed_by_does_not_silently_bypass_the_human_actor_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            recovery = self._propose(root, controller, self._base_plan("researcher"))
            del recovery["proposed_by"]
            controller.save()
            reloaded = RunController(root / "run")
            with self.assertRaisesRegex(ValueError, "human approval actor"):
                reloaded.approve_recovery({"actor_kind": "agent", "canonical_id": "researcher"})


class Scenario8NormalPerActionApprovalUnchangedTests(_SelfApprovalFixture):
    """8. the existing normal per-action approval mechanism remains unchanged."""

    def test_grant_and_has_action_approval_are_untouched_by_actor_identity_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            self.assertFalse(controller.has_action_approval("costly_teacher_labeling"))
            controller.grant_action_approval("costly_teacher_labeling")
            self.assertTrue(controller.has_action_approval("costly_teacher_labeling"))


if __name__ == "__main__":
    unittest.main()
