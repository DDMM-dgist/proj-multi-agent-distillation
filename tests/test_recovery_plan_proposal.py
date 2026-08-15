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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
