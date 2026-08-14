"""Trust boundary for the recorded recovery proposer identity: authority must come from the
TRUSTED caller/runtime context, never from an LLM-authored plan.json payload field.

``workflow.actor_identity.normalize_actor_identity`` intentionally treats a bare string as
``actor_kind="human"`` for manual usability -- acceptable only at a genuinely human-operated
entry point (the CLI). Before this revision, ``RunController.propose_recovery`` had exactly one
way to learn the proposer identity: ``plan["proposed_by"]``, read straight off disk. The one real
wired agent-facing path to it, ``runtimes.pydantic_ai.orchestrator_bridge._exec_propose_recovery``,
passed the plan_path through unchanged -- so an agent-authored plan.json could write
``proposed_by: "researcher"`` or ``{"actor_kind": "human", ...}`` and have it accepted as genuine
human provenance. This file proves that gap is closed: ``propose_recovery`` now accepts a
keyword-only, caller-supplied ``proposer`` that is authoritative over the payload, the
orchestrator bridge derives it from the Pydantic ``Literal["orchestrator"]``-typed
``requested_by_role`` (a field an LLM cannot forge to any other value), a conflicting payload
fails closed, the human-operated CLI/manual call shape (no ``proposer`` argument) is untouched,
and no agent-callable path can inject ``approved_by``/``authorized_by`` human-approval authority.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from runtimes.pydantic_ai import controller_bridge, dispatch
from runtimes.pydantic_ai.orchestrator_bridge import (
    OrchestratorActionProposal,
    dispatch_orchestrator_action,
)
from workflow.controller import RunController

GATE_CRITERION = "artifact is complete and internally consistent"


class _TrustBoundaryFixture(unittest.TestCase):
    def _controller(self, root):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "trust-boundary", "stages": [{
            "name": "validation", "command": None, "outputs": ["artifacts/result.txt"],
            "gate": {"criteria": [GATE_CRITERION]},
        }]}))
        controller = RunController.initialize(cfg, root / "run")
        result = controller.run_dir / "artifacts/result.txt"
        result.write_text("result")
        controller.complete_external_stage("validation", [result])
        controller.record_gate("validation", "REVISE")
        return controller

    def _plan(self, root, proposed_by=None, name="plan.json"):
        plan = {
            "schema_version": 1, "failed_stage": "validation",
            "failure_category": "dataset_coverage", "root_cause": "low support in slice",
            "responsible_agent": "data-curator", "return_stage": "validation",
            "proposed_changes": [{"type": "add_deployment_frames"}],
            "labeling": {"teacher_relabel": False, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["validation"]},
            "estimated_cost": {},
        }
        if proposed_by is not None:
            plan["proposed_by"] = proposed_by
        plan_path = root / name
        plan_path.write_text(json.dumps(plan))
        return plan_path

    def _orchestrator_proposal(self, plan_path, **overrides):
        payload = {
            "run_id": "trust-boundary", "stage": "validation", "requested_at": "t",
            "rationale": "recovering from a REVISE gate",
            "idempotency_key": "k1", "action_type": "propose_recovery",
            "parameters": {"plan_path": str(plan_path)},
        }
        payload.update(overrides)
        return OrchestratorActionProposal(**payload)


class Scenario1And2AgentCannotImpersonateHumanTests(_TrustBoundaryFixture):
    """1. Agent caller + proposed_by: "researcher" cannot impersonate a human.
    2. Agent caller + structured {actor_kind: "human"} cannot impersonate a human."""

    def test_agent_caller_with_bare_string_payload_proposer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(root, proposed_by="researcher")
            proposal = self._orchestrator_proposal(plan_path)
            outcome = dispatch_orchestrator_action(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "EXECUTOR_ERROR")
            self.assertIn("conflicts with the trusted", outcome.reason)
            self.assertEqual(controller.state.get("recoveries"), [])

    def test_agent_caller_with_structured_human_payload_proposer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(
                root, proposed_by={"actor_kind": "human", "canonical_id": "orchestrator"})
            proposal = self._orchestrator_proposal(plan_path)
            outcome = dispatch_orchestrator_action(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "EXECUTOR_ERROR")
            self.assertIn("conflicts with the trusted", outcome.reason)


class Scenario3CallerIdentityIsStampedTests(_TrustBoundaryFixture):
    """3. Agent caller identity is stamped/bound to the resulting recovery record."""

    def test_recovery_record_is_stamped_with_the_trusted_orchestrator_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(root, proposed_by=None)
            proposal = self._orchestrator_proposal(plan_path)
            outcome = dispatch_orchestrator_action(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "EXECUTED")
            recovery = controller.state["recoveries"][-1]
            self.assertEqual(recovery["proposed_by"],
                             {"actor_kind": "system", "canonical_id": "orchestrator",
                              "display_name": None})

    def test_a_non_conflicting_matching_payload_proposer_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(
                root, proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"})
            proposal = self._orchestrator_proposal(plan_path)
            outcome = dispatch_orchestrator_action(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "EXECUTED")


class Scenario4PayloadCallerMismatchFailsClosedTests(_TrustBoundaryFixture):
    """4. payload/caller identity mismatch fails closed (direct propose_recovery(proposer=...)
    call, independent of the orchestrator bridge's own EXECUTOR_ERROR wrapping above)."""

    def test_direct_propose_recovery_rejects_a_conflicting_payload_proposer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(
                root, proposed_by={"actor_kind": "agent", "canonical_id": "someone-else"})
            with self.assertRaisesRegex(ValueError, "conflicts with the trusted"):
                controller.propose_recovery(
                    plan_path, proposer={"actor_kind": "system", "canonical_id": "orchestrator"})
            self.assertEqual(controller.state.get("recoveries"), [])

    def test_direct_propose_recovery_accepts_no_payload_proposer_with_a_trusted_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(root, proposed_by=None)
            recovery = controller.propose_recovery(
                plan_path, proposer={"actor_kind": "system", "canonical_id": "orchestrator"})
            self.assertEqual(recovery["proposed_by"]["canonical_id"], "orchestrator")


class Scenario5CLIBackwardCompatibilityTests(_TrustBoundaryFixture):
    """5. trusted manual CLI bare-string human proposal remains backward compatible.

    The CLI's propose-recovery subcommand never passes a `proposer` kwarg (see
    workflow.controller.main), so this is exactly the pre-existing, unchanged call shape."""

    def test_omitting_proposer_still_trusts_a_bare_string_payload_as_human(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(root, proposed_by="researcher")
            recovery = controller.propose_recovery(plan_path)
            self.assertEqual(recovery["proposed_by"],
                             {"actor_kind": "human", "canonical_id": "researcher",
                              "display_name": "researcher"})

    def test_no_agent_driven_code_anywhere_shells_out_to_the_propose_recovery_cli(self):
        import subprocess
        root = Path(__file__).resolve().parent.parent
        hits = subprocess.run(
            ["grep", "-rl", "--include=*.py", "propose-recovery",
             str(root / "runtimes"), str(root / "orchestration")],
            capture_output=True, text=True,
        )
        self.assertEqual(hits.stdout.strip(), "")


class Scenario6NoAgentPathCanInjectApprovalIdentityTests(_TrustBoundaryFixture):
    """6. Agent cannot inject approved_by/authorized_by through any agent-callable path."""

    def test_dispatch_module_never_references_approve_or_authorize_recovery(self):
        import inspect
        source = inspect.getsource(dispatch)
        for needle in ("approved_by", "authorized_by", "approve_recovery",
                       "authorize_recovery_capabilities"):
            self.assertNotIn(needle, source)

    def test_controller_bridge_module_never_references_approve_or_authorize_recovery(self):
        import inspect
        source = inspect.getsource(controller_bridge)
        for needle in ("approved_by", "authorized_by", "approve_recovery",
                       "authorize_recovery_capabilities"):
            self.assertNotIn(needle, source)

    def test_orchestrator_bridge_human_approval_actions_are_unwired_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            for action_type in ("request_human_approval", "read_human_decision"):
                proposal = OrchestratorActionProposal(
                    run_id="trust-boundary", stage="validation", requested_at="t",
                    rationale="r", idempotency_key=f"k-{action_type}", action_type=action_type,
                    parameters={"approved_by": "attacker", "authorized_by": "attacker"},
                )
                outcome = dispatch_orchestrator_action(proposal, controller=controller,
                                                        mode="primary")
                self.assertEqual(outcome.status, "BLOCKED_CAPABILITY")


class Scenario7ValidHumanApprovalStillWorksTests(_TrustBoundaryFixture):
    """7. valid human approval still creates the authorization envelope normally."""

    def test_valid_human_approval_and_envelope_issuance_are_unaffected_by_the_trust_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(root, proposed_by=None)
            proposal = self._orchestrator_proposal(plan_path)
            outcome = dispatch_orchestrator_action(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "EXECUTED")
            approved = controller.approve_recovery("lab-lead", "approved by a human reviewer")
            self.assertEqual(approved["status"], "approved")
            controller.start_iteration()
            envelope = controller.authorize_recovery_capabilities(
                "lab-lead", action_types=["label_with_teacher"])
            self.assertEqual(envelope["authorized_by"],
                             {"actor_kind": "human", "canonical_id": "lab-lead",
                              "display_name": "lab-lead"})


class Scenario8ExistingSuitesStillPassTests(_TrustBoundaryFixture):
    """8. all existing self-approval and historical-manifest tests continue to pass (smoke check
    that the new `proposer` kwarg is additive-only; the full suites are the real proof and are
    run separately -- see tests/test_recovery_self_approval_separation.py)."""

    def test_omitted_proposer_self_approval_separation_is_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(root, proposed_by="researcher")
            controller.propose_recovery(plan_path)
            with self.assertRaisesRegex(ValueError, "cannot both propose and approve"):
                controller.approve_recovery("  Researcher  ")

    def test_historical_manifest_missing_proposed_by_is_still_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan_path = self._plan(root, proposed_by="researcher")
            recovery = controller.propose_recovery(plan_path)
            del recovery["proposed_by"]
            controller.save()
            reloaded = RunController(root / "run")
            self.assertNotIn("proposed_by", reloaded.state["recoveries"][-1])
            approved = reloaded.approve_recovery("researcher")
            self.assertEqual(approved["status"], "approved")


if __name__ == "__main__":
    unittest.main()
