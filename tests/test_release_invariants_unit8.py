"""UNIT 8 tests: SET-LEVEL completeness invariants for the costly-approval and protected-data
gates (release-harness properties C and D).

Properties A (happy-path 1->12 traversal) and B (recovery routing for insufficient coverage /
insufficient dataset population / failed validation criterion, returning to the workflow) are
already exercised end to end by the existing integration suites -- ``test_synthetic_full_campaign_
lifecycle`` (all four automation paths across the twelve canonical stages),
``test_full_lifecycle_integration``, ``test_recovery_negative_paths`` and
``test_recovery_reacquisition_unit4`` (coverage-adequacy PARTIAL/COMPLETE, reacquisition
supersession, return-to-stage). Those prove the properties PER INSTANCE.

What no existing test asserts -- and what a release gate needs -- is that the two safety properties
hold over the WHOLE registered set, not just the specific actions/roles a given scenario happens to
touch:

  * (D) EVERY inherently-costly action is approval-gated and can NEVER be relaxed to no-approval by
    a self-asserted ``performs_teacher_inference=False`` flag; only the acquisition-family
    ``costly_teacher_labeling`` boundary is ever relaxable, and only for a NON-inherent action that
    affirmatively proves it runs no Teacher.
  * (C) The protected-data role restrictions are UNCONDITIONAL: ``derive_admissible_decision_space``
    surfaces the full restriction set for every evidence profile, including the fully-insufficient
    one -- protection never depends on which validation components happen to be admissible.
"""
from __future__ import annotations

import unittest

from runtimes.pydantic_ai import actions
from validation.teacher_evidence_profile import (
    PROTECTED_DATA_RESTRICTIONS,
    TeacherEvidenceProfile,
    derive_admissible_decision_space,
)


KNOWN_BOUNDARIES = {
    "costly_teacher_labeling", "costly_training", "production_md", "scheduler_submission",
}


def _boundary(action_type, parameters=None):
    default = actions.APPROVAL_GATED_ACTIONS.get(action_type)
    return actions.resolve_action_approval_boundary(action_type, default, parameters or {})


class CostlyApprovalCompletenessTests(unittest.TestCase):
    """(D) The approval gate is complete over the full costly-action set."""

    def test_every_inherently_costly_action_is_approval_gated(self):
        for action_type in actions._INHERENT_COSTLY_ACTIONS:
            self.assertIn(
                action_type, actions.APPROVAL_GATED_ACTIONS,
                f"inherently-costly action {action_type!r} has no approval boundary")
            self.assertIsNotNone(actions.APPROVAL_GATED_ACTIONS[action_type])

    def test_inherently_costly_action_never_relaxed_by_a_false_flag(self):
        # A proposal that self-asserts it performs no Teacher inference must NOT be able to shed the
        # approval boundary of an action whose costly effect is inherent (it always runs the Teacher
        # / trains / runs MD / submits a job, regardless of any declared flag).
        for action_type in actions._INHERENT_COSTLY_ACTIONS:
            resolved = _boundary(action_type, {"performs_teacher_inference": False})
            self.assertIsNotNone(
                resolved,
                f"{action_type!r} was relaxed to no-approval by a self-asserted false flag")
            self.assertEqual(resolved, actions.APPROVAL_GATED_ACTIONS[action_type])

    def test_all_gated_boundaries_are_known(self):
        for action_type, boundary in actions.APPROVAL_GATED_ACTIONS.items():
            self.assertIn(boundary, KNOWN_BOUNDARIES,
                          f"{action_type!r} maps to unknown boundary {boundary!r}")

    def test_only_costly_teacher_labeling_is_ever_relaxable(self):
        # For every gated action whose boundary is NOT costly_teacher_labeling, a false
        # performs_teacher_inference flag must leave the boundary untouched (never relaxed).
        for action_type, boundary in actions.APPROVAL_GATED_ACTIONS.items():
            if boundary == "costly_teacher_labeling":
                continue
            self.assertEqual(
                _boundary(action_type, {"performs_teacher_inference": False}), boundary,
                f"non-teacher-labeling boundary of {action_type!r} was unexpectedly relaxed")

    def test_teacher_labeling_relaxes_only_for_a_non_inherent_proven_no_teacher_action(self):
        # acquire_structures is the acquisition-family action that CAN relax when it affirmatively
        # proves it runs no Teacher; it must stay gated when the proof is absent or affirmatively
        # costly, and label_with_teacher (inherent) can never relax.
        self.assertIsNone(_boundary("acquire_structures", {"performs_teacher_inference": False}))
        self.assertEqual(
            _boundary("acquire_structures", {"performs_teacher_inference": True}),
            "costly_teacher_labeling")
        self.assertEqual(
            _boundary("acquire_structures", {}), "costly_teacher_labeling")  # fail-closed
        self.assertEqual(
            _boundary("label_with_teacher", {"performs_teacher_inference": False}),
            "costly_teacher_labeling")


class ProtectedDataRestrictionCompletenessTests(unittest.TestCase):
    """(C) Protected-data role restrictions are unconditional across every evidence profile."""

    def test_restriction_set_is_nonempty_and_covers_training_and_acquisition_roles(self):
        self.assertTrue(PROTECTED_DATA_RESTRICTIONS)
        # the roles that would consume protected DFT labels for model-shaping purposes must all be
        # prohibited from touching the protected reference population.
        for role in ("student_training", "student_validation_tuning", "acquisition_seed",
                     "augmentation_parent", "recovery_training"):
            self.assertIn(role, PROTECTED_DATA_RESTRICTIONS)

    def test_full_restriction_set_surfaced_for_a_rich_profile(self):
        profile = TeacherEvidenceProfile(
            teacher_model_available=True,
            operational_evaluation_population_available=True,
            original_training_db_available=True,
            original_labels_available=True,
            original_split_recovered=True,
            genuine_holdout_test_available=True,
            independent_external_reference_available=True,
            deployment_domain_population_available=True,
        )
        space = derive_admissible_decision_space(profile)
        self.assertEqual(list(space["protected_data_restrictions"]),
                         list(PROTECTED_DATA_RESTRICTIONS))

    def test_full_restriction_set_surfaced_even_for_an_insufficient_profile(self):
        # The fully-insufficient profile admits NOTHING, yet protection is independent of
        # admissibility: the complete restriction set must still be surfaced.
        profile = TeacherEvidenceProfile(teacher_model_available=False)
        space = derive_admissible_decision_space(profile)
        self.assertTrue(space["insufficient_evidence"])
        self.assertEqual(space["admissible_components"], [])
        self.assertEqual(list(space["protected_data_restrictions"]),
                         list(PROTECTED_DATA_RESTRICTIONS))


if __name__ == "__main__":
    unittest.main()
