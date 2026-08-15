"""Fail-closed negative paths for the bounded-autonomy recovery lifecycle (task #64).

Companion to test_full_lifecycle_integration.py's happy path. Each test below drives the SAME
real production entry points (workflow.controller.RunController,
runtimes.pydantic_ai.dispatch/controller_bridge/orchestrator_bridge/recovery_bridge/root_cause) up
to the exact point one fail-closed invariant is exercised, then asserts the specific failure. No
parallel/bypass path is constructed; no real scientific compute runs anywhere in this file.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtimes.pydantic_ai import bounded_evidence
from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
from runtimes.pydantic_ai.dispatch import ActionDescriptor, default_registry
from runtimes.pydantic_ai.orchestrator_bridge import (
    OrchestratorActionProposal, dispatch_orchestrator_action,
)
from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
from runtimes.pydantic_ai.models import EvidenceReference
from runtimes.pydantic_ai.root_cause import RootCauseClassification, validate_root_cause_classification
from workflow import contracts
from workflow.controller import RunController
from workflow.integrity import sha256_file

from test_full_lifecycle_integration import FixtureHelpers, DEPLOYMENT_DOMAIN, SPLIT_POLICY


class RecoveryNegativePathTest(FixtureHelpers):
    """Each test gets its own tmp run dir; helpers below drive shared setup so each test method
    focuses on exactly one fail-closed boundary."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    # --- shared: drive an initialized run up through "diagnosis + drafted (unproposed) plan" -
    def _drive_to_plan(self, controller, *, capability="data_repair", return_stage="prepare",
                       proposed_by=None, required_input_artifact_roles=None,
                       responsible_capability_override=None):
        baseline = controller.run_dir / "artifacts/baseline.json"
        self._write_json(baseline, {"role": "baseline", "revision": 1})
        controller.complete_external_stage("prepare", [baseline])
        self._gate(controller, "prepare", "PASS")

        evidence_path = controller.run_dir / "artifacts/evidence.json"
        self._write_json(evidence_path, self._coverage_evidence_payload())
        controller.complete_external_stage("produce_evidence", [evidence_path])
        self._gate(controller, "produce_evidence", "REVISE")

        classification = RootCauseClassification(
            run_id="fixture-run", stage="produce_evidence", failure_category="dataset_coverage",
            evidence_refs=[EvidenceReference(role="coverage_evidence", path=str(evidence_path),
                                             integrity={"sha256": sha256_file(evidence_path)})],
            evidence_summary="query environments are unsupported by the reference population",
            confidence=0.8, recommended_recovery_target=return_stage,
            recommended_next_action="augment the candidate pool",
        )
        validate_root_cause_classification(
            classification, available_artifacts=[str(evidence_path)],
            valid_recovery_targets=[s["name"] for s in controller.state["stages"]])
        diagnosis_path = self._write_json(controller.run_dir / "diagnosis.json",
                                          json.loads(classification.model_dump_json()))
        draft = build_recovery_plan_draft(
            classification,
            proposed_by=proposed_by or {"actor_kind": "system", "canonical_id": "orchestrator"},
            failed_stage="produce_evidence", capability=capability, return_stage=return_stage,
            proposed_changes=[{"type": "add_deployment_frames"}],
            labeling={"teacher_relabel": True, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
            diagnosis_artifact_path=str(diagnosis_path), diagnosis_artifact_sha256=sha256_file(diagnosis_path),
            required_input_artifact_roles=required_input_artifact_roles,
        )
        plan_json = draft.to_plan_json()
        if responsible_capability_override is not None:
            plan_json["responsible_capability"] = responsible_capability_override
        plan_path = self._write_json(controller.run_dir / "plan.json", plan_json)
        return evidence_path, diagnosis_path, plan_path

    # === 1. changed validation contract during recovery ====================================
    def test_changed_validation_contract_is_rejected(self):
        controller = self._init_controller(self.root)
        other_domain = {"slice_classes": ["group_z"], "parameter_range": [5, 5]}
        components = contracts.build_validation_contract_components(
            {"deployment_domain": other_domain},
            {"deployment_domain": other_domain, "shared_md_protocol": "protocol-v1",
             "checks": ["check_a", "check_b"]},
            {"split_policy": SPLIT_POLICY},
        )
        with self.assertRaisesRegex(ValueError, "already established.*different content"):
            controller.establish_validation_contract(components)

    # === 2. stale diagnosis artifact ========================================================
    def test_stale_diagnosis_artifact_is_rejected(self):
        controller = self._init_controller(self.root)
        _, diagnosis_path, plan_path = self._drive_to_plan(controller)
        diagnosis_path.write_text(diagnosis_path.read_text() + "\n")  # mutate after binding
        with self.assertRaisesRegex(ValueError, "hash-mismatched"):
            controller.propose_recovery(plan_path)

    # === 3. stale triggering evidence hash ==================================================
    def test_stale_triggering_evidence_hash_is_rejected(self):
        controller = self._init_controller(self.root)
        evidence_path, _, plan_path = self._drive_to_plan(controller)
        plan = json.loads(plan_path.read_text())
        self.assertTrue(plan["diagnosis_binding"]["triggering_evidence"])
        evidence_path.write_text(evidence_path.read_text() + " ")  # mutate cited evidence
        with self.assertRaisesRegex(ValueError, "hash-mismatched"):
            controller.propose_recovery(plan_path)

    # === 4. Agent attempting human identity spoofing ========================================
    def test_agent_cannot_spoof_human_identity_via_orchestrator_bridge(self):
        controller = self._init_controller(self.root)
        _, _, plan_path = self._drive_to_plan(
            controller, proposed_by={"actor_kind": "human", "canonical_id": "dr-lee"})
        proposal = OrchestratorActionProposal(
            run_id="fixture-run", stage="produce_evidence", requested_at="t",
            rationale="spoof attempt", idempotency_key="propose-spoof",
            action_type="propose_recovery",
            parameters={"run_dir": str(controller.run_dir), "plan_path": str(plan_path)})
        outcome = dispatch_orchestrator_action(proposal, controller=controller, mode="primary")
        self.assertEqual(outcome.status, "EXECUTOR_ERROR")
        self.assertIn("conflicts with the trusted", outcome.reason)
        # The failed proposal never bound a recovery: the gate is still waiting, untouched.
        self.assertEqual(controller.state["pending_recovery"]["status"], "required")
        self.assertEqual(controller.state.get("recoveries", []), [])

    # === 5. proposer attempting self-approval ===============================================
    def test_proposer_cannot_self_approve(self):
        controller = self._init_controller(self.root)
        _, _, plan_path = self._drive_to_plan(
            controller, proposed_by={"actor_kind": "human", "canonical_id": "dr-lee"})
        # Historical/manual call shape: no trusted `proposer` kwarg, payload is trusted outright.
        controller.propose_recovery(plan_path)
        with self.assertRaisesRegex(ValueError, "cannot both propose and approve"):
            controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})

    # === 6. RecoveryPlan approval without an authorization envelope for a delegated costly
    # action ==================================================================================
    def test_delegated_action_without_envelope_requires_approval(self):
        controller = self._init_controller(
            self.root, recovery_capability_roster={"data_repair": "data-curator"})
        _, _, plan_path = self._drive_to_plan(controller, capability="data_repair")
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        controller.start_iteration()
        # No authorize_recovery_capabilities call at all.
        registry = default_registry()
        registry["label_with_teacher"] = ActionDescriptor(
            action_type="label_with_teacher", role="data-curator",
            approval_boundary="costly_teacher_labeling", executor=lambda proposal: {"path": "x"})
        outcome = dispatch_via_controller(
            {"requested_by_role": "data-curator", "action_type": "label_with_teacher",
             "idempotency_key": "corrective-noenv", "run_id": "fixture-run", "stage": "prepare",
             "requested_at": "t", "rationale": "no envelope", "parameters": {}},
            controller=controller, registry=registry, mode="primary")
        self.assertEqual(outcome.status, "APPROVAL_REQUIRED")

    # === 7. child action outside the authorized action type =================================
    def test_child_action_outside_authorized_action_type_is_blocked(self):
        controller = self._init_controller(
            self.root, recovery_capability_roster={"data_repair": "data-curator"})
        _, _, plan_path = self._drive_to_plan(controller, capability="data_repair")
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        controller.start_iteration()
        controller.authorize_recovery_capabilities(
            {"actor_kind": "human", "canonical_id": "dr-lee"},
            action_types=["some_other_action_type"], resource_limits={"cpu_hours": 10})
        registry = default_registry()
        registry["label_with_teacher"] = ActionDescriptor(
            action_type="label_with_teacher", role="data-curator",
            approval_boundary="costly_teacher_labeling", executor=lambda proposal: {"path": "x"})
        outcome = dispatch_via_controller(
            {"requested_by_role": "data-curator", "action_type": "label_with_teacher",
             "idempotency_key": "corrective-wrongtype", "run_id": "fixture-run", "stage": "prepare",
             "requested_at": "t", "rationale": "wrong action type", "parameters": {}},
            controller=controller, registry=registry, mode="primary")
        self.assertEqual(outcome.status, "APPROVAL_REQUIRED")

    # === 8. child action exceeding its resource/budget envelope ==============================
    def test_child_action_exceeding_resource_envelope_is_blocked(self):
        controller = self._init_controller(
            self.root, recovery_capability_roster={"data_repair": "data-curator"})
        _, _, plan_path = self._drive_to_plan(controller, capability="data_repair")
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        controller.start_iteration()
        controller.authorize_recovery_capabilities(
            {"actor_kind": "human", "canonical_id": "dr-lee"},
            action_types=["label_with_teacher"], resource_limits={"cpu_hours": 1})
        registry = default_registry()
        registry["label_with_teacher"] = ActionDescriptor(
            action_type="label_with_teacher", role="data-curator",
            approval_boundary="costly_teacher_labeling", executor=lambda proposal: {"path": "x"})
        outcome = dispatch_via_controller(
            {"requested_by_role": "data-curator", "action_type": "label_with_teacher",
             "idempotency_key": "corrective-overbudget", "run_id": "fixture-run", "stage": "prepare",
             "requested_at": "t", "rationale": "over budget",
             "parameters": {"resource_usage": {"cpu_hours": 3}}},
            controller=controller, registry=registry, mode="primary")
        self.assertEqual(outcome.status, "APPROVAL_REQUIRED")

    # === 9. protected-reference artifact entering an acquisition/training role ==============
    def test_protected_reference_role_requires_explicit_override(self):
        controller = self._init_controller(
            self.root, protected_reference_roles=["protected_reference_set"])
        _, _, plan_path = self._drive_to_plan(
            controller, required_input_artifact_roles=["protected_reference_set"])
        with self.assertRaisesRegex(ValueError, "protected-reference artifact role"):
            controller.propose_recovery(plan_path)

    # === 10. unchanged/no-op recovery artifact ==============================================
    def test_unchanged_recovery_artifact_is_rejected(self):
        controller = self._init_controller(self.root)
        evidence_path, _, plan_path = self._drive_to_plan(
            controller, return_stage="produce_evidence")
        plan = json.loads(plan_path.read_text())
        plan["revalidation"]["targets"] = ["produce_evidence"]
        plan_path.write_text(json.dumps(plan))
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        recovery_id = controller.state["pending_recovery"]["recovery_id"]
        controller.start_iteration()
        # invalidate_from() quarantines (moves away) the prior artifact file; recreate it with
        # byte-IDENTICAL content to prove the controller detects "no genuine change".
        self._write_json(evidence_path, self._coverage_evidence_payload())
        controller.complete_external_stage("produce_evidence", [evidence_path])
        report = {
            "schema_version": 1, "recovery_id": recovery_id, "previous_iteration": 1,
            "current_iteration": 2,
            "changes": [{"type": "add_deployment_frames", "status": "APPLIED",
                        "evidence_artifacts": [str(evidence_path)]}],
            "labeling": {"teacher_relabel": True, "teacher_relabel_stage": "produce_evidence",
                        "new_dft": False, "new_dft_stage": None},
            "student_training": {"retrain": False, "mode": "none", "stage": None},
            "revalidation": {"targets": ["produce_evidence"], "stages": ["produce_evidence"]},
        }
        report_path = self._write_json(controller.run_dir / "execution_report.json", report)
        with self.assertRaisesRegex(ValueError, "did not change"):
            controller.verify_recovery_execution(report_path)

    # === 11. wrong re-entry stage (evidence cited from before the approved return stage) ====
    def test_execution_report_cannot_cite_a_stage_before_return_stage(self):
        controller = self._init_controller(self.root)
        evidence_path, _, plan_path = self._drive_to_plan(
            controller, return_stage="produce_evidence")
        plan = json.loads(plan_path.read_text())
        plan["labeling"] = {"teacher_relabel": False, "new_dft": False}
        plan["revalidation"] = {"reuse_profile": True, "targets": ["prepare", "produce_evidence"]}
        plan_path.write_text(json.dumps(plan))
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        recovery_id = controller.state["pending_recovery"]["recovery_id"]
        controller.start_iteration()  # only invalidates produce_evidence (return_stage)
        self._write_json(evidence_path,
                         self._coverage_evidence_payload(unsupported_count=1, seed="-changed"))
        controller.complete_external_stage("produce_evidence", [evidence_path])
        report = {
            "schema_version": 1, "recovery_id": recovery_id, "previous_iteration": 1,
            "current_iteration": 2,
            "changes": [{"type": "add_deployment_frames", "status": "APPLIED",
                        "evidence_artifacts": [str(evidence_path)]}],
            "labeling": {"teacher_relabel": False, "teacher_relabel_stage": None,
                        "new_dft": False, "new_dft_stage": None},
            "student_training": {"retrain": False, "mode": "none", "stage": None},
            # "prepare" precedes return_stage="produce_evidence": must fail closed.
            "revalidation": {"targets": ["prepare", "produce_evidence"],
                             "stages": ["prepare", "produce_evidence"]},
        }
        report_path = self._write_json(controller.run_dir / "execution_report.json", report)
        with self.assertRaisesRegex(ValueError, "precedes the approved return stage"):
            controller.verify_recovery_execution(report_path)

    # === 12. recovery PASS attempted before execution verification ==========================
    def test_pass_before_execution_verification_is_blocked(self):
        controller = self._init_controller(self.root)
        evidence_path, _, plan_path = self._drive_to_plan(
            controller, return_stage="produce_evidence")
        plan = json.loads(plan_path.read_text())
        plan["revalidation"]["targets"] = ["produce_evidence"]
        plan_path.write_text(json.dumps(plan))
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        controller.start_iteration()
        self._write_json(evidence_path,
                         self._coverage_evidence_payload(unsupported_count=1, seed="-changed"))
        controller.complete_external_stage("produce_evidence", [evidence_path])
        with self.assertRaisesRegex(RuntimeError, "cannot PASS until recovery execution"):
            self._gate(controller, "produce_evidence", "PASS")

    # === 13. repeated recovery signature beyond configured policy ===========================
    def test_repeated_recovery_signature_beyond_policy_is_rejected(self):
        controller = self._init_controller(
            self.root, recovery_policy={"max_repeated_signature": 1})
        evidence_path, _, plan_path = self._drive_to_plan(
            controller, return_stage="produce_evidence")
        plan = json.loads(plan_path.read_text())
        plan["revalidation"]["targets"] = ["produce_evidence"]
        plan_path.write_text(json.dumps(plan))
        controller.propose_recovery(plan_path)  # 1st proposal: prior_repeats=0, allowed
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        recovery_id = controller.state["pending_recovery"]["recovery_id"]
        controller.start_iteration()
        self._write_json(evidence_path,
                         self._coverage_evidence_payload(unsupported_count=1, seed="-changed"))
        controller.complete_external_stage("produce_evidence", [evidence_path])
        report = {
            "schema_version": 1, "recovery_id": recovery_id, "previous_iteration": 1,
            "current_iteration": 2,
            "changes": [{"type": "add_deployment_frames", "status": "APPLIED",
                        "evidence_artifacts": [str(evidence_path)]}],
            "labeling": {"teacher_relabel": True, "teacher_relabel_stage": "produce_evidence",
                        "new_dft": False, "new_dft_stage": None},
            "student_training": {"retrain": False, "mode": "none", "stage": None},
            "revalidation": {"targets": ["produce_evidence"], "stages": ["produce_evidence"]},
        }
        controller.verify_recovery_execution(
            self._write_json(controller.run_dir / "execution_report.json", report))
        self._gate(controller, "produce_evidence", "PASS")  # resolves recovery #1

        # Re-trigger with BYTE-IDENTICAL failing content -> identical recovery_signature.
        self._write_json(evidence_path, self._coverage_evidence_payload())
        controller.complete_external_stage("produce_evidence", [evidence_path])
        self._gate(controller, "produce_evidence", "REVISE")
        _, _, plan_path_2 = self._drive_second_plan_only(
            controller, evidence_path, return_stage="produce_evidence")
        with self.assertRaisesRegex(ValueError, "max_repeated_signature"):
            controller.propose_recovery(plan_path_2)

    def _drive_second_plan_only(self, controller, evidence_path, *, return_stage):
        classification = RootCauseClassification(
            run_id="fixture-run", stage="produce_evidence", failure_category="dataset_coverage",
            evidence_refs=[EvidenceReference(role="coverage_evidence", path=str(evidence_path),
                                             integrity={"sha256": sha256_file(evidence_path)})],
            evidence_summary="the same coverage gap recurs", confidence=0.8,
            recommended_recovery_target=return_stage,
            recommended_next_action="augment the candidate pool",
        )
        validate_root_cause_classification(
            classification, available_artifacts=[str(evidence_path)],
            valid_recovery_targets=[s["name"] for s in controller.state["stages"]])
        diagnosis_path = self._write_json(controller.run_dir / "diagnosis-2.json",
                                          json.loads(classification.model_dump_json()))
        draft = build_recovery_plan_draft(
            classification, proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"},
            failed_stage="produce_evidence", capability="data_repair", return_stage=return_stage,
            proposed_changes=[{"type": "add_deployment_frames"}],
            labeling={"teacher_relabel": True, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["produce_evidence"]},
            diagnosis_artifact_path=str(diagnosis_path), diagnosis_artifact_sha256=sha256_file(diagnosis_path),
        )
        plan_path = self._write_json(controller.run_dir / "plan-2.json", draft.to_plan_json())
        return None, diagnosis_path, plan_path

    # === 14. nonexistent/unregistered capability =============================================
    def test_unregistered_capability_is_rejected(self):
        controller = self._init_controller(
            self.root, recovery_capability_roster={"data_repair": "data-curator"})
        _, _, plan_path = self._drive_to_plan(
            controller, responsible_capability_override="no_such_capability")
        with self.assertRaisesRegex(ValueError, "not registered in this run's roster"):
            controller.propose_recovery(plan_path)

    # === 15. coverage representation/backend details never leak into the Analyst contract ===
    def test_coverage_representation_backend_details_never_leak(self):
        evidence_path = self.root / "evidence.json"
        self._write_json(evidence_path, self._coverage_evidence_payload())
        summary = bounded_evidence.summarize_artifact(evidence_path)
        coverage_summary = summary["structural_coverage_evidence"]
        self.assertEqual(set(coverage_summary), {
            "direction", "query_population", "reference_population", "n_query_environments",
            "n_query_structures", "reference_population_counts",
            "overall_distance_distribution", "query_slice_memberships",
            "query_slice_distance_distributions", "reference_slice_memberships",
            "reference_slice_distance_distributions", "provenance_hashes", "limitations",
        })
        leaked = json.dumps(coverage_summary).lower()
        for forbidden in ("soap", "cktree", "descriptor", "n_max", "l_max"):
            self.assertNotIn(forbidden, leaked)


if __name__ == "__main__":
    unittest.main()
