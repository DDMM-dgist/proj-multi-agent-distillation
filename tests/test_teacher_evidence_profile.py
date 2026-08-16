"""Tests for the generic, evidence-driven Teacher validation ADMISSIBLE COMPONENT MODEL
(validation.teacher_evidence_profile). Every scenario here uses only boolean evidence facts --
never a material, dataset, or campaign name -- proving the mechanism is a reference
demonstration of a generic, evidence-driven decision, not a SiO2/Allegro-specific branch.

Components are additive (not a mutually-exclusive strategy enum): a profile's admissible set
may, and often does, contain more than one member.
"""
from __future__ import annotations

import unittest

from validation.teacher_evidence_profile import (
    CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING,
    DEPLOYMENT_APPLICABILITY,
    INDEPENDENT_REFERENCE_FIDELITY,
    OPERATIONAL_ROBUSTNESS,
    ORIGINAL_HELDOUT_FIDELITY,
    PROTECTED_DATA_RESTRICTIONS,
    TRAINING_CORPUS_CONSISTENCY,
    TeacherEvidenceProfile,
    derive_admissible_decision_space,
)


def _profile(**overrides):
    base = dict(
        teacher_model_available=True,
        operational_evaluation_population_available=False,
        original_training_db_available=False,
        original_labels_available=False,
        original_split_recovered=False,
        genuine_holdout_test_available=False,
        independent_external_reference_available=False,
        deployment_domain_population_available=False,
    )
    base.update(overrides)
    return TeacherEvidenceProfile(**base)


class AdmissibleDecisionSpaceTests(unittest.TestCase):
    def test_no_evidence_at_all_is_the_insufficient_evidence_floor(self):
        result = derive_admissible_decision_space(_profile())
        self.assertEqual(result["admissible_components"], [])
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["floor"], CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING)

    def test_custom_teacher_with_trustworthy_original_holdout(self):
        # custom Teacher + DB + labels + trustworthy split + genuine holdout + an operational
        # population -> {OPERATIONAL_ROBUSTNESS, ORIGINAL_HELDOUT_FIDELITY}. Note
        # TRAINING_CORPUS_CONSISTENCY's weaker requirement is trivially also satisfied here, so
        # it must be admissible too -- components are additive, never mutually exclusive.
        profile = _profile(
            operational_evaluation_population_available=True,
            original_training_db_available=True, original_labels_available=True,
            original_split_recovered=True, genuine_holdout_test_available=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY,
                          ORIGINAL_HELDOUT_FIDELITY})
        self.assertIn("held_out_fidelity",
                      result["components"][ORIGINAL_HELDOUT_FIDELITY]["allowed_claims"])

    def test_custom_teacher_with_db_but_no_trustworthy_split(self):
        # custom Teacher + DB but no trustworthy split -> {OPERATIONAL_ROBUSTNESS,
        # TRAINING_CORPUS_CONSISTENCY}; original held-out fidelity is not admissible.
        profile = _profile(
            operational_evaluation_population_available=True,
            original_training_db_available=True, original_labels_available=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY})
        self.assertNotIn(ORIGINAL_HELDOUT_FIDELITY, result["admissible_components"])
        self.assertNotIn("held_out_fidelity",
                         result["components"][TRAINING_CORPUS_CONSISTENCY]["allowed_claims"])
        self.assertIn("held_out_fidelity",
                      result["components"][TRAINING_CORPUS_CONSISTENCY]["prohibited_claims"])

    def test_umlip_with_genuine_independent_external_reference(self):
        # uMLIP (no DB) + genuine independent external reference + an operational population
        # -> {OPERATIONAL_ROBUSTNESS, INDEPENDENT_REFERENCE_FIDELITY}.
        profile = _profile(
            operational_evaluation_population_available=True,
            independent_external_reference_available=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, INDEPENDENT_REFERENCE_FIDELITY})

    def test_umlip_with_no_reference_but_an_operational_population_is_not_empty(self):
        # uMLIP + no independent reference, but the Teacher model plus a real operational
        # evaluation population exist -> {OPERATIONAL_ROBUSTNESS} only. NOT the insufficient-
        # evidence floor: an operationally-evaluable Teacher is always plannable.
        profile = _profile(operational_evaluation_population_available=True)
        result = derive_admissible_decision_space(profile)
        self.assertEqual(result["admissible_components"], [OPERATIONAL_ROBUSTNESS])
        self.assertFalse(result["insufficient_evidence"])

    def test_teacher_model_alone_with_no_population_is_insufficient(self):
        # teacher_model_available alone, with no operational population, does NOT satisfy
        # OPERATIONAL_ROBUSTNESS -- a Teacher with nothing to evaluate against has no
        # operational-robustness evidence merely because it is loadable.
        profile = _profile(teacher_model_available=True)
        result = derive_admissible_decision_space(profile)
        self.assertEqual(result["admissible_components"], [])
        self.assertTrue(result["insufficient_evidence"])

    def test_original_holdout_plus_deployment_domain_mismatch(self):
        # original holdout + a deployment-domain mismatch -> ORIGINAL_HELDOUT_FIDELITY and
        # DEPLOYMENT_APPLICABILITY are distinct and both admissible; held-out fidelity alone
        # does not satisfy the deployment-applicability requirement.
        profile = _profile(
            operational_evaluation_population_available=True,
            original_training_db_available=True, original_labels_available=True,
            original_split_recovered=True, genuine_holdout_test_available=True,
            deployment_domain_population_available=True,
            deployment_domain_matches_original_test_distribution=False,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY,
                          ORIGINAL_HELDOUT_FIDELITY, DEPLOYMENT_APPLICABILITY})

    def test_matching_deployment_domain_does_not_admit_deployment_applicability(self):
        profile = _profile(
            original_training_db_available=True, original_labels_available=True,
            original_split_recovered=True, genuine_holdout_test_available=True,
            deployment_domain_population_available=True,
            deployment_domain_matches_original_test_distribution=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertNotIn(DEPLOYMENT_APPLICABILITY, result["admissible_components"])

    def test_protected_data_restrictions_are_unconditional(self):
        # Present regardless of which components are admissible, including the empty case.
        for profile in (_profile(), _profile(independent_external_reference_available=True)):
            result = derive_admissible_decision_space(profile)
            self.assertEqual(set(result["protected_data_restrictions"]),
                             set(PROTECTED_DATA_RESTRICTIONS))

    def test_reference_sio2_allegro_campaign_evidence_resolves_as_expected(self):
        # This is the reference demonstration the user described: for THIS campaign, DB is
        # available, split is recovered and cross-version verified, and a genuine 1,142-frame
        # held-out test exists -- so the generic mechanism naturally admits
        # ORIGINAL_HELDOUT_FIDELITY (plus the weaker components its evidence subsumes). Nothing
        # here is a special-cased branch; it is the same decision logic exercised with this
        # campaign's actual evidence values.
        profile = TeacherEvidenceProfile(
            teacher_model_available=True,
            operational_evaluation_population_available=True,
            original_training_db_available=True,
            original_labels_available=True,
            original_split_recovered=True,
            original_split_confidence="VERIFIED_ZERO_DISCREPANCY_CROSS_VERSION",
            genuine_holdout_test_available=True,
            genuine_holdout_test_frame_count=1142,
            independent_external_reference_available=False,
            deployment_domain_population_available=True,
            deployment_domain_matches_original_test_distribution=None,
        )
        result = derive_admissible_decision_space(profile)
        self.assertIn(ORIGINAL_HELDOUT_FIDELITY, result["admissible_components"])
        self.assertNotIn(DEPLOYMENT_APPLICABILITY, result["admissible_components"])
        self.assertNotIn(INDEPENDENT_REFERENCE_FIDELITY, result["admissible_components"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
