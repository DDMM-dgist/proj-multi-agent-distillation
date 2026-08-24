"""Stage-8 (evaluation) / Stage-9 (uncertainty) separation.

Committee/per-seed disagreement and calibrated uncertainty are Stage-9 evidence. Stage 8 must be
able to gate on its own evaluation evidence WITHOUT requiring a downstream Stage-9 uncertainty
artifact as a prerequisite. This proves the framework's default evaluation StageReviewSpec never
demands `uncertainty_report` evidence, while the uncertainty stage's spec does -- so a corrected
run cannot re-introduce the recovery-004 defect (an evaluation gate criterion that required
committee force disagreement, i.e. a Stage-9 artifact, to be present in the Stage-8 accuracy
report).
"""
from __future__ import annotations

import unittest

from framework_v2.review_spec import default_stage_review_specs
from framework_v2.stages import CanonicalStage as S


def _required_evidence_classes(spec) -> set[str]:
    classes: set[str] = set()
    for criterion in spec.criteria:
        classes.update(criterion.required_evidence_classes)
    return classes


class Stage8DoesNotRequireStage9EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.specs = default_stage_review_specs()

    def test_evaluation_spec_does_not_require_uncertainty_report(self):
        evaluation = self.specs[S.EVALUATION.value]
        self.assertNotIn("uncertainty_report", _required_evidence_classes(evaluation))

    def test_uncertainty_spec_is_the_one_that_requires_uncertainty_report(self):
        uncertainty = self.specs[S.UNCERTAINTY.value]
        self.assertIn("uncertainty_report", _required_evidence_classes(uncertainty))

    def test_evaluation_spec_gates_on_its_own_evaluation_evidence(self):
        # Stage 8's required evidence is drawn only from its own evaluation artifacts, never a
        # Stage-9 uncertainty artifact.
        evaluation = self.specs[S.EVALUATION.value]
        required = _required_evidence_classes(evaluation)
        self.assertTrue(required)
        self.assertTrue(required <= {"evaluation_report", "scope_contract", "evaluation_policy"},
                        f"evaluation spec requires unexpected evidence classes: {required}")


if __name__ == "__main__":
    unittest.main()
