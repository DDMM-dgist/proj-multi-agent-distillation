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


# --- R20 forensic-audit checklist item 6: teacher_baseline was misclassified as a
# reference_disagreement/teacher_vs_dft failure although it uses no DFT labels at all -- a
# classification asserting a Teacher-vs-DFT comparison must be rejected unless the failed stage's
# own evidence actually contains one (dft_comparison_evidence_present, computed deterministically
# by cli._stage_evidence_reveals_dft_comparison). See Scope F.
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class DftComparisonAssertionGatingTests(unittest.TestCase):
    def _validate(self, c, **over):
        from runtimes.pydantic_ai.root_cause import validate_root_cause_classification
        kwargs = dict(available_artifacts=AVAILABLE, valid_recovery_targets=TARGETS)
        kwargs.update(over)
        return validate_root_cause_classification(c, **kwargs)

    def test_reference_disagreement_category_rejected_without_dft_comparison_evidence(self):
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        c = _classification(failure_category="reference_disagreement",
                            affected_channel="teacher_support",
                            recommended_recovery_target="teacher_baseline")
        with self.assertRaises(RootCauseValidationError):
            self._validate(c, dft_comparison_evidence_present=False)

    def test_dft_named_channel_rejected_without_dft_comparison_evidence(self):
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        c = _classification(affected_channel="teacher_vs_dft",
                            recommended_recovery_target="teacher_baseline")
        with self.assertRaises(RootCauseValidationError):
            self._validate(c, dft_comparison_evidence_present=False)

    def test_non_dft_classification_unaffected_by_gate(self):
        # The default fixture (affected_channel="student_vs_teacher") makes no Teacher-vs-DFT
        # claim at all -- the gate must not reject it merely because evidence is absent.
        c = self._validate(_classification(), dft_comparison_evidence_present=False)
        self.assertEqual(c.affected_channel, "student_vs_teacher")

    def test_reference_disagreement_accepted_when_dft_comparison_evidence_present(self):
        # Same classification that was rejected above must be ACCEPTED once the failed stage's
        # evidence genuinely contains a Teacher-vs-DFT comparison (e.g. a real
        # reference_validation gate failure) -- the gate is evidence-bound, not a blanket ban.
        c = _classification(failure_category="reference_disagreement",
                            affected_channel="teacher_support",
                            recommended_recovery_target="teacher_baseline")
        validated = self._validate(c, dft_comparison_evidence_present=True)
        self.assertEqual(validated.failure_category, "reference_disagreement")


# --- R31 forensic finding: the recovery Analyst cited a plausible-but-wrong artifact path
# ('artifacts/train.extxyz') when the registered path is 'artifacts/dataset/train.extxyz'. The
# validator already fails closed on any unregistered path; the minimal grounding fix only makes the
# rejection ACTIONABLE by pointing a basename near-miss at the real registered path, so a retry
# cites the true path instead of re-inventing one. It must NOT widen what is accepted.
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class InventedArtifactPathGroundingTests(unittest.TestCase):
    def _validate(self, c, available):
        from runtimes.pydantic_ai.root_cause import validate_root_cause_classification
        return validate_root_cause_classification(
            c, available_artifacts=available, valid_recovery_targets=TARGETS)

    def test_basename_near_miss_surfaced_in_rejection(self):
        from runtimes.pydantic_ai.models import EvidenceReference
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        available = {"runs/r/artifacts/dataset/train.extxyz", "runs/r/committee/u.json"}
        bad = _classification(
            evidence_refs=[EvidenceReference(role="x", path="runs/r/artifacts/train.extxyz")],
            affected_artifact_refs=[EvidenceReference(role="x", path="runs/r/committee/u.json")])
        with self.assertRaises(RootCauseValidationError) as ctx:
            self._validate(bad, available)
        msg = str(ctx.exception)
        self.assertIn("did you mean", msg)
        self.assertIn("runs/r/artifacts/dataset/train.extxyz", msg)

    def test_no_hint_when_basename_has_no_registered_match(self):
        from runtimes.pydantic_ai.models import EvidenceReference
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        available = {"runs/r/committee/u.json"}
        bad = _classification(
            evidence_refs=[EvidenceReference(role="x", path="runs/r/nowhere/mystery.json")])
        with self.assertRaises(RootCauseValidationError) as ctx:
            self._validate(bad, available)
        self.assertNotIn("did you mean", str(ctx.exception))

    def test_hint_does_not_accept_invented_path(self):
        # The near-miss hint is diagnostic only: a classification citing the wrong path is still
        # rejected, never silently accepted or rewritten to the suggested path.
        from runtimes.pydantic_ai.models import EvidenceReference
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        available = {"runs/r/artifacts/dataset/train.extxyz"}
        bad = _classification(
            evidence_refs=[EvidenceReference(role="x", path="runs/r/artifacts/train.extxyz")])
        with self.assertRaises(RootCauseValidationError):
            self._validate(bad, available)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
