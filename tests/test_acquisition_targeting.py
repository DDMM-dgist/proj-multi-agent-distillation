"""Priority #3 requirement #6: typed, evidence-linked acquisition/repair targeting.

Proves AcquisitionTargetProposal/DataRepairProposal (a) are evidence-bound (empty evidence_refs
is rejected), (b) render into RecoveryPlanDraft.proposed_changes entries that pass
propose_recovery's own taxonomy validation, (c) never select any concrete AcquisitionPlan value
themselves, and (d) can only ADDITIVELY bind onto a real AcquisitionPlan -- an unbound plan and a
bound plan both pass runtimes.pydantic_ai.executors._validate_acquisition_plan unchanged.
"""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from runtimes.pydantic_ai.acquisition_targeting import (
    AcquisitionTargetProposal,
    DataRepairProposal,
    bind_acquisition_plan,
)
from runtimes.pydantic_ai.executors import _REQUIRED_ACQUISITION_PLAN_FIELDS, _validate_acquisition_plan
from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanDraft, RecoveryRouting

_EVIDENCE = [{"path": "evidence/coverage.json", "sha256": "a" * 64}]


def _plan() -> dict:
    return {
        "schema_version": 1, "eligible_source_categories": ["x"],
        "selected_parent_structure_ids": ["p1"], "selected_source_global_indices": [0],
        "n_parents": 1, "n_per_structure": 2, "T_K": 300, "beta": 0.1,
        "sigma_range_A": [0.01, 0.05], "cell_sigma": 0.01, "seed": 1,
        "expected_output_count": 2, "duplicate_handling": "skip",
        "protected_reference_exclusion_report": {
            "status": "PASS", "dft_labels_used_as_selection_scores": False,
        },
    }


class AcquisitionTargetProposalTests(unittest.TestCase):
    def test_empty_evidence_refs_is_rejected(self):
        with self.assertRaises(ValidationError):
            AcquisitionTargetProposal(
                target_population="p", target_direction="d", rationale="r", evidence_refs=[])

    def test_extra_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            AcquisitionTargetProposal(
                target_population="p", target_direction="d", rationale="r",
                evidence_refs=_EVIDENCE, unexpected="nope")

    def test_never_carries_concrete_acquisition_plan_fields(self):
        # `eligible_source_categories` is deliberately shared as an ADVISORY hint (see the
        # model docstring) -- the AcquisitionPlan's own value remains authoritative; every other
        # authoritative AcquisitionPlan field (parent/source selection, counts, seed, ...) must
        # never appear in a proposed_changes entry, since selecting those stays a separate,
        # campaign-specific step this typed model never performs.
        t = AcquisitionTargetProposal(
            target_population="candidate_population", target_direction="teacher_support",
            rationale="low support fraction in slice", evidence_refs=_EVIDENCE)
        change = t.to_proposed_change()
        authoritative_only = _REQUIRED_ACQUISITION_PLAN_FIELDS - {"eligible_source_categories"}
        self.assertFalse(authoritative_only & set(change))

    def test_binding_is_additive_only_and_plan_still_validates(self):
        t = AcquisitionTargetProposal(
            target_population="candidate_population", target_direction="teacher_support",
            rationale="low support fraction in slice", evidence_refs=_EVIDENCE)
        plan = _plan()
        bound = bind_acquisition_plan(plan, t)
        self.assertNotIn("target_binding", plan)  # original untouched
        self.assertEqual(bound["target_binding"]["target_proposal_sha256"], t.proposal_sha256())

        unbound_validated = _validate_acquisition_plan(_plan())
        bound_validated = _validate_acquisition_plan(bound)
        self.assertNotIn("target_binding", unbound_validated)
        self.assertIn("target_binding", bound_validated)

    def test_feeds_recovery_plan_draft_proposed_changes(self):
        t = AcquisitionTargetProposal(
            target_population="candidate_population", target_direction="teacher_support",
            rationale="low support fraction in slice", evidence_refs=_EVIDENCE)
        draft = RecoveryPlanDraft(
            proposed_by={"actor_kind": "agent", "canonical_id": "analyst"},
            failed_stage="teacher_baseline", failure_category="data_coverage", root_cause="x",
            routing=RecoveryRouting(capability="acquisition"), return_stage="teacher_baseline",
            proposed_changes=[t.to_proposed_change()], labeling={}, student_training={},
            revalidation={},
        )
        self.assertEqual(draft.to_plan_json()["proposed_changes"][0]["change_kind"],
                         "acquisition_target")


class DataRepairProposalTests(unittest.TestCase):
    def test_empty_affected_artifact_refs_is_rejected(self):
        with self.assertRaises(ValidationError):
            DataRepairProposal(
                defect_description="d", affected_artifact_refs=[], rationale="r",
                evidence_refs=_EVIDENCE)

    def test_feeds_recovery_plan_draft_proposed_changes(self):
        r = DataRepairProposal(
            defect_description="mislabeled teacher forces", rationale="NaN forces detected",
            affected_artifact_refs=[{"path": "artifacts/labels.json", "sha256": "b" * 64}],
            evidence_refs=_EVIDENCE,
        )
        draft = RecoveryPlanDraft(
            proposed_by={"actor_kind": "agent", "canonical_id": "analyst"},
            failed_stage="teacher_baseline", failure_category="data_quality", root_cause="x",
            routing=RecoveryRouting(capability="data_repair"), return_stage="teacher_baseline",
            proposed_changes=[r.to_proposed_change()], labeling={}, student_training={},
            revalidation={},
        )
        self.assertEqual(draft.to_plan_json()["proposed_changes"][0]["change_kind"], "data_repair")


if __name__ == "__main__":
    unittest.main()
