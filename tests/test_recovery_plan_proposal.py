"""Orchestrator RecoveryPlanProposal: the SECOND typed reasoning output (after
RootCauseClassification), closing the gap where build_recovery_plan_draft's scientific-choice
fields (capability/proposed_changes/labeling/student_training/revalidation/return_stage) cannot be
derived from a diagnosis alone. Network-free; skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import unittest

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

VALID_CAPABILITY_ROSTER = {"data_repair": "data-curator", "orchestration": "orchestrator"}
VALID_STAGES = {"prepare", "produce_evidence"}


def _classification(**over):
    from runtimes.pydantic_ai.root_cause import RootCauseClassification
    from runtimes.pydantic_ai.models import EvidenceReference
    base = dict(
        run_id="r", stage="produce_evidence", failure_category="dataset_coverage",
        evidence_refs=[EvidenceReference(role="coverage_evidence", path="runs/r/evidence.json",
                                        integrity={"sha256": "f" * 64})],
        evidence_summary="unsupported coverage in slice group_a",
        confidence=0.8, recommended_recovery_target="prepare",
        recommended_next_action="augment the candidate pool")
    base.update(over)
    return RootCauseClassification(**base)


def _proposal(**over):
    from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
    base = dict(
        run_id="r", failed_stage="produce_evidence", diagnosis_artifact_sha256="d" * 64,
        capability="data_repair", return_stage="prepare",
        proposed_changes=[{"type": "add_deployment_frames"}],
        labeling={"teacher_relabel": True, "new_dft": False},
        student_training={"retrain": False, "mode": "none"},
        revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
        rationale="coverage gap requires more deployment-representative frames")
    base.update(over)
    return RecoveryPlanProposal(**base)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class RecoveryPlanProposalTests(unittest.TestCase):
    def _validate(self, p, **over):
        from runtimes.pydantic_ai.recovery_bridge import validate_recovery_plan_proposal
        kwargs = dict(expected_failed_stage="produce_evidence",
                     expected_diagnosis_sha256="d" * 64,
                     capability_roster=VALID_CAPABILITY_ROSTER, valid_stage_names=VALID_STAGES)
        kwargs.update(over)
        return validate_recovery_plan_proposal(p, **kwargs)

    def test_valid_proposal_passes(self):
        p = self._validate(_proposal())
        self.assertEqual(p.capability, "data_repair")

    def test_wrong_failed_stage_rejected(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        with self.assertRaises(RecoveryPlanValidationError):
            self._validate(_proposal(failed_stage="prepare"))

    def test_stale_diagnosis_binding_rejected(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        with self.assertRaises(RecoveryPlanValidationError):
            self._validate(_proposal(diagnosis_artifact_sha256="e" * 64))

    def test_unregistered_capability_rejected(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        with self.assertRaises(RecoveryPlanValidationError):
            self._validate(_proposal(capability="not_a_capability"))

    def test_invalid_return_stage_rejected(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        with self.assertRaises(RecoveryPlanValidationError):
            self._validate(_proposal(return_stage="not_a_stage"))

    def test_empty_proposed_changes_rejected_by_shape(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError):
            _proposal(proposed_changes=[])

    def test_builds_draft_matching_direct_build_recovery_plan_draft(self):
        from runtimes.pydantic_ai.recovery_bridge import (
            build_recovery_plan_draft, build_recovery_plan_draft_from_proposal)
        classification = _classification()
        proposal = self._validate(_proposal())
        via_proposal = build_recovery_plan_draft_from_proposal(
            classification, proposal, proposed_by="dr-lee",
            diagnosis_artifact_path="runs/r/diagnosis.json", diagnosis_artifact_sha256="d" * 64)
        direct = build_recovery_plan_draft(
            classification, proposed_by="dr-lee", failed_stage="produce_evidence",
            capability="data_repair", return_stage="prepare",
            proposed_changes=[{"type": "add_deployment_frames"}],
            labeling={"teacher_relabel": True, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
            diagnosis_artifact_path="runs/r/diagnosis.json", diagnosis_artifact_sha256="d" * 64)
        self.assertEqual(via_proposal.to_plan_json(), direct.to_plan_json())

    def test_registered_as_reasoning_output_model(self):
        from runtimes.pydantic_ai.role_outputs import is_reasoning_output_model
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
        self.assertTrue(is_reasoning_output_model(RecoveryPlanProposal))


# --- Same R17 attempt-f89857df provenance file also contained proposed_changes/student_training/
# revalidation shapes that satisfied only the OLD bare dict[str, Any]/dict[str, bool] field types
# while silently violating propose_recovery's real structural requirements (workflow/controller.py
# _validate_protected_reference_roles and the inline proposed_changes/labeling/student_training/
# revalidation checks in propose_recovery itself). Each is the same hidden-constraint defect class
# as corrective_action, just on a different field; these regressions prove the now-typed
# ProposedChange/LabelingPlan/StudentTrainingPlan/RevalidationPlan submodels reject the exact
# real-world non-conforming shapes with a precise reason, while still accepting the legitimate
# descriptive extra keys (id/responsible_agent/action/acceptance_criteria/...) the R17 model
# produced alongside the required structural keys.
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ProposedChangesLabelingStudentTrainingRevalidationContractTests(unittest.TestCase):
    # The R17 candidate's proposed_changes items carried descriptive keys but never the
    # required "type" key propose_recovery actually enforces.
    R17_STYLE_PROPOSED_CHANGE_MISSING_TYPE = {
        "id": "pc-1", "responsible_agent": "data-curator",
        "action": "add deployment-representative frames to the candidate pool",
        "acceptance_criteria": ["coverage report shows group_a supported"],
    }

    def test_proposed_change_missing_type_rejected_by_schema(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError) as ctx:
            _proposal(proposed_changes=[self.R17_STYLE_PROPOSED_CHANGE_MISSING_TYPE])
        message = str(ctx.exception)
        self.assertIn("type", message)
        self.assertIn("proposed_changes", message)

    def test_proposed_change_keeps_descriptive_extra_keys(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
        p = RecoveryPlanProposal(**{
            **dict(run_id="r", failed_stage="produce_evidence",
                   diagnosis_artifact_sha256="d" * 64, capability="data_repair",
                   return_stage="prepare",
                   labeling={"teacher_relabel": True, "new_dft": False},
                   student_training={"retrain": False, "mode": "none"},
                   revalidation={"reuse_profile": True,
                                 "targets": ["prepare", "produce_evidence"]},
                   rationale="coverage gap requires more deployment-representative frames"),
            "proposed_changes": [{**self.R17_STYLE_PROPOSED_CHANGE_MISSING_TYPE,
                                   "type": "add_deployment_frames"}],
        })
        self.assertEqual(p.proposed_changes[0].id, "pc-1")
        self.assertEqual(p.proposed_changes[0].type, "add_deployment_frames")

    def test_labeling_missing_required_key_rejected_by_schema(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError) as ctx:
            _proposal(labeling={"teacher_relabel": True})
        message = str(ctx.exception)
        self.assertIn("new_dft", message)

    # The R17 candidate's student_training used retrain_required/strategy/artifacts_to_provide --
    # never the retrain/mode keys propose_recovery actually requires.
    def test_student_training_r17_style_keys_rejected_by_schema(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError) as ctx:
            _proposal(student_training={
                "retrain_required": False, "strategy": "none",
                "artifacts_to_provide": []})
        message = str(ctx.exception)
        self.assertIn("retrain", message)
        self.assertIn("mode", message)

    def test_student_training_retrain_mode_inconsistency_rejected(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError) as ctx:
            _proposal(student_training={"retrain": True, "mode": "none"})
        self.assertIn("inconsistent", str(ctx.exception))

    # The R17 candidate's revalidation used steps/gate_targets/gate_criteria_source -- never the
    # reuse_profile/targets keys propose_recovery actually requires.
    def test_revalidation_r17_style_keys_rejected_by_schema(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError) as ctx:
            _proposal(revalidation={
                "steps": ["re-run coverage report"], "gate_targets": ["produce_evidence"],
                "gate_criteria_source": "configs/runs/r/gates.yaml"})
        message = str(ctx.exception)
        self.assertIn("reuse_profile", message)
        self.assertIn("targets", message)

    def test_revalidation_empty_targets_rejected_by_schema(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError):
            _proposal(revalidation={"reuse_profile": True, "targets": []})


# --- Gap #5: build_recovery_plan_draft previously hash-verified `triggering_evidence` using
# whatever `sha256` the RootCauseClassification's EvidenceReference.integrity self-reported, never
# the controller's own already-registered artifact hash (RunController.state["artifacts"]) -- the
# same "model-claimed digest is prose audit only, never an authoritative integrity assertion"
# distinction the codebase already draws for actions.ActionProposalBase.
# advisory_claimed_config_hashes. artifact_sha256_lookup closes this: when supplied, it -- not the
# model-reported integrity -- is what DiagnosisBinding.triggering_evidence gets built from.
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class DiagnosisBindingHashTrustTests(unittest.TestCase):
    def test_artifact_sha256_lookup_overrides_model_reported_integrity(self):
        from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
        classification = _classification()
        wrong_hash = "f" * 64
        controller_true_hash = "1" * 64
        self.assertEqual(classification.evidence_refs[0].integrity["sha256"], wrong_hash)
        draft = build_recovery_plan_draft(
            classification, proposed_by="dr-lee", failed_stage="produce_evidence",
            capability="data_repair", return_stage="prepare",
            proposed_changes=[{"type": "add_deployment_frames"}],
            labeling={"teacher_relabel": True, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
            diagnosis_artifact_path="runs/r/diagnosis.json", diagnosis_artifact_sha256="d" * 64,
            artifact_sha256_lookup={"runs/r/evidence.json": controller_true_hash})
        bound = draft.diagnosis_binding.triggering_evidence[0]
        self.assertEqual(bound.path, "runs/r/evidence.json")
        self.assertEqual(bound.sha256, controller_true_hash)
        self.assertNotEqual(bound.sha256, wrong_hash)

    def test_missing_lookup_entry_fails_closed_rather_than_using_model_claimed_hash(self):
        """An artifact absent from the controller's own registered-artifact map must fail closed
        immediately (EvidenceHashRef requires a non-empty sha256) rather than silently falling
        back to a self-reported hash the controller never verified, and rather than silently
        building a binding with an empty/absent hash that would only be caught much later."""
        import pydantic as _pydantic
        from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
        classification = _classification()
        with self.assertRaises(_pydantic.ValidationError) as ctx:
            build_recovery_plan_draft(
                classification, proposed_by="dr-lee", failed_stage="produce_evidence",
                capability="data_repair", return_stage="prepare",
                proposed_changes=[{"type": "add_deployment_frames"}],
                labeling={"teacher_relabel": True, "new_dft": False},
                student_training={"retrain": False, "mode": "none"},
                revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
                diagnosis_artifact_path="runs/r/diagnosis.json", diagnosis_artifact_sha256="d" * 64,
                artifact_sha256_lookup={})
        self.assertIn("sha256", str(ctx.exception))

    def test_no_lookup_falls_back_to_model_reported_integrity(self):
        """Callers with no controller-registered artifacts to look up from (e.g. constructing a
        diagnosis by hand in a test) may omit artifact_sha256_lookup entirely and still get a
        binding -- from the model-reported integrity, which propose_recovery still fail-closed
        hash-verifies against the real file regardless of source."""
        from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
        classification = _classification()
        draft = build_recovery_plan_draft(
            classification, proposed_by="dr-lee", failed_stage="produce_evidence",
            capability="data_repair", return_stage="prepare",
            proposed_changes=[{"type": "add_deployment_frames"}],
            labeling={"teacher_relabel": True, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
            diagnosis_artifact_path="runs/r/diagnosis.json", diagnosis_artifact_sha256="d" * 64)
        bound = draft.diagnosis_binding.triggering_evidence[0]
        self.assertEqual(bound.sha256, classification.evidence_refs[0].integrity["sha256"])


# --- R17 forensic regression: corrective_action was exposed as an unconstrained
# Optional[dict[str, Any]] while validate_recovery_plan_proposal secretly required
# {"action_type": <non-empty str>, "parameters"?: dict}. The Orchestrator produced a
# well-formed-looking but structurally wrong candidate (prose/approval-caveat keys, no
# action_type) that passed pydantic-ai's own shape validation and was only rejected by the
# contextual validator, with the model never having been shown the required shape or the
# registered action vocabulary. See exchange/provenance/teacher_baseline-recovery-plan.
# attempt-f89857df-0d27-494d-a03f-d1b4fe04eb8d.json on runs/sio2-sox-allegro-simplenn-r17.
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class CorrectiveActionContractTests(unittest.TestCase):
    def _validate(self, p, **over):
        from runtimes.pydantic_ai.recovery_bridge import validate_recovery_plan_proposal
        kwargs = dict(expected_failed_stage="produce_evidence",
                     expected_diagnosis_sha256="d" * 64,
                     capability_roster=VALID_CAPABILITY_ROSTER, valid_stage_names=VALID_STAGES)
        kwargs.update(over)
        return validate_recovery_plan_proposal(p, **kwargs)

    # The exact rejected R17 candidate's corrective_action payload, verbatim.
    R17_REJECTED_CORRECTIVE_ACTION = {
        "human_approval_required": True,
        "estimated_dft_cost_note": (
            "Estimate depends on number and complexity of failing frames; compute precise "
            "cost from failing_frames.list.txt and configs before approval. Rough planning: "
            "O(1-100) targeted DFT single-point or small-relaxation jobs."),
        "escalation_contact": "researcher",
        "preconditions": [
            "obtain explicit human approval for DFT acquisitions and budget",
            "verify DFT method parameters align with run reference in configs; if missing, "
            "researcher to specify"],
    }

    def test_r17_rejected_candidate_shape_fails_schema_validation_with_precise_reason(self):
        import pydantic as _pydantic
        with self.assertRaises(_pydantic.ValidationError) as ctx:
            _proposal(corrective_action=self.R17_REJECTED_CORRECTIVE_ACTION)
        message = str(ctx.exception)
        # Precise, not generic: names the exact missing/forbidden fields, not just "invalid".
        self.assertIn("action_type", message)
        self.assertIn("corrective_action", message)

    def test_corrective_action_schema_exposes_action_type_and_parameters(self):
        """The structural requirement must be MACHINE-VISIBLE (in the generated JSON Schema),
        not only enforced after the fact by validate_recovery_plan_proposal."""
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
        schema = RecoveryPlanProposal.model_json_schema()
        defs = schema.get("$defs", schema.get("definitions", {}))
        corrective_schema = defs.get("CorrectiveAction")
        self.assertIsNotNone(corrective_schema, f"CorrectiveAction missing from $defs: {defs.keys()}")
        self.assertIn("action_type", corrective_schema["properties"])
        self.assertIn("parameters", corrective_schema["properties"])
        self.assertIn("action_type", corrective_schema.get("required", []))

    def test_schema_valid_registered_corrective_action_passes(self):
        p = self._validate(_proposal(corrective_action={
            "action_type": "build_dataset_manifest", "parameters": {"dataset": "d.extxyz"}}))
        self.assertEqual(p.corrective_action.action_type, "build_dataset_manifest")

    def test_corrective_action_none_remains_valid(self):
        p = self._validate(_proposal(corrective_action=None))
        self.assertIsNone(p.corrective_action)

    def test_corrective_action_wrong_role_action_type_rejected_by_contextual_validator(self):
        """action_type structurally valid (schema-wise) but not in the allowed set for the role
        responsible for the chosen capability -- e.g. an ml-trainer-only action proposed under
        capability="data_repair" (role data-curator) -- must still fail closed, contextually."""
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        with self.assertRaises(RecoveryPlanValidationError) as ctx:
            self._validate(_proposal(corrective_action={
                "action_type": "train_committee", "parameters": {}}))
        self.assertIn("train_committee", str(ctx.exception))

    def test_valid_actions_by_capability_derives_single_source_from_role_registry(self):
        from runtimes.pydantic_ai.recovery_bridge import valid_corrective_actions_by_capability
        from runtimes.pydantic_ai.actions import ROLE_ALLOWED_ACTIONS
        exposed = valid_corrective_actions_by_capability(VALID_CAPABILITY_ROSTER)
        self.assertEqual(set(exposed["data_repair"]), ROLE_ALLOWED_ACTIONS["data-curator"])
        # "orchestration" maps to role "orchestrator", which has no entry in
        # ROLE_ALLOWED_ACTIONS (it is not a producer role) -- correctly omitted, not KeyError'd.
        self.assertNotIn("orchestration", exposed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
