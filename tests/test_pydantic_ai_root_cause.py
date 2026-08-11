"""Phase 6/2: Analyst RootCauseClassification typed reasoning output + evidence validation.

Network-free; skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import unittest

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

AVAILABLE = {"runs/r/eval/errorc.json", "runs/r/committee/u.json"}
TARGETS = {"data_curation", "student_training", "teacher_baseline"}


def _classification(**over):
    from runtimes.pydantic_ai.root_cause import RootCauseClassification
    from runtimes.pydantic_ai.models import EvidenceReference
    base = dict(
        run_id="r", stage="validation", failure_category="student_fidelity",
        affected_channel="student_vs_teacher",
        affected_artifact_refs=[EvidenceReference(role="ml-trainer", path="runs/r/eval/errorc.json")],
        evidence_refs=[EvidenceReference(role="ml-trainer", path="runs/r/eval/errorc.json"),
                       EvidenceReference(role="ml-trainer", path="runs/r/committee/u.json")],
        evidence_summary="force MAE above threshold on held-out set",
        confidence=0.7, excluded_alternatives=["data_coverage"],
        recommended_recovery_target="student_training",
        recommended_next_action="retrain committee on augmented set")
    base.update(over)
    return RootCauseClassification(**base)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class RootCauseTests(unittest.TestCase):
    def _validate(self, c):
        from runtimes.pydantic_ai.root_cause import validate_root_cause_classification
        return validate_root_cause_classification(c, available_artifacts=AVAILABLE,
                                                  valid_recovery_targets=TARGETS)

    def test_valid_evidence_bound_classification(self):
        c = self._validate(_classification())
        self.assertEqual(c.failure_category, "student_fidelity")

    def test_nonexistent_artifact_reference_rejected(self):
        from runtimes.pydantic_ai.models import EvidenceReference
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        bad = _classification(evidence_refs=[EvidenceReference(role="x", path="runs/r/does_not_exist.json")])
        with self.assertRaises(RootCauseValidationError):
            self._validate(bad)

    def test_unsupported_category_rejected(self):
        import pydantic
        with self.assertRaises(pydantic.ValidationError):
            _classification(failure_category="totally_made_up")

    def test_missing_evidence_rejected(self):
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        with self.assertRaises(RootCauseValidationError):
            self._validate(_classification(evidence_refs=[]))

    def test_invalid_recovery_target_rejected(self):
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        with self.assertRaises(RootCauseValidationError):
            self._validate(_classification(recommended_recovery_target="not_a_stage"))

    def test_output_cannot_mutate_controller_or_verdict(self):
        # structural guarantee: the model has no field to change a gate verdict or controller state
        fields = set(_classification().model_fields)
        self.assertFalse({"verdict", "gate", "controller", "stage_status", "accepted"} & fields)

    def test_projects_to_typed_recovery_recommendation(self):
        from runtimes.pydantic_ai.root_cause import to_recovery_recommendation
        rec = to_recovery_recommendation(self._validate(_classification()))
        self.assertEqual(rec.target_stage, "student_training")
        self.assertTrue(rec.requires_human_approval)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
