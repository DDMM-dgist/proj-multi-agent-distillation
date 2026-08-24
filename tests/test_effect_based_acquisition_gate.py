"""Regression tests for the effect-based human-approval boundary resolution.

The approval boundary an action requires is derived from the materially costly, non-trivially
reversible EFFECTS the action actually performs for a given proposal -- never from its action
name or the stage it occupies. The load-bearing defect this guards against: ``acquire_structures``
was gated behind ``costly_teacher_labeling`` purely because of its name/position, so a geometry-only
acquisition (cheap, reversible structure generation) was blocked one stage too early -- before the
reversible planning that DEFINES what will later be labeled.

These tests pin, at the unit level:

  * ``actions.resolve_action_approval_boundary`` relaxes ``costly_teacher_labeling`` to ``None`` ONLY
    for a proposal that affirmatively proves it performs no Teacher inference / creates no new labels;
  * an action whose Teacher inference is INHERENT (``label_with_teacher``) can NEVER be relaxed, even
    with a self-asserted ``performs_teacher_inference: False`` -- no existing expensive-compute safety
    rule is weakened;
  * boundaries other than ``costly_teacher_labeling`` (student training, production MD, scheduler
    submission) are never relaxed here;
  * ``cli._acquisition_incurs_teacher_inference`` classifies the ACTUAL bound recipe: both built-in
    recipes drive the REAL Teacher calculator during structure generation, so augment-atoms -> True
    (its executor unconditionally binds the Teacher for perturbation/relaxation) and teacher-md ->
    True; a configured ``adapter.acquire`` -> True; a missing / unreadable / non-dict / unknown-kind
    config fails closed -> True.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from runtimes.pydantic_ai import actions
from runtimes.pydantic_ai import cli


def _boundary(action_type, **parameters):
    default = actions.APPROVAL_GATED_ACTIONS.get(action_type)
    return actions.resolve_action_approval_boundary(action_type, default, parameters)


class ResolveApprovalBoundaryEffectBasedTests(unittest.TestCase):
    # --- acquisition family: relaxation is effect-based, driven by performs_teacher_inference -----
    def test_geometry_only_acquisition_relaxed_to_none(self):
        self.assertIsNone(_boundary("acquire_structures", performs_teacher_inference=False))

    def test_teacher_inference_acquisition_stays_gated(self):
        self.assertEqual(_boundary("acquire_structures", performs_teacher_inference=True),
                         "costly_teacher_labeling")

    def test_acquisition_without_effect_declaration_fails_closed(self):
        # No typed effect declaration -> the gate is kept (never mistaken for geometry-only).
        self.assertEqual(_boundary("acquire_structures"), "costly_teacher_labeling")

    def test_acquisition_declared_dft_labels_stays_gated(self):
        self.assertEqual(_boundary("acquire_structures", dft_labels_used=True),
                         "costly_teacher_labeling")

    # --- inherent-costly actions: NEVER relaxable, even with a self-asserted no-inference flag -----
    def test_label_with_teacher_never_relaxed_even_with_false_flag(self):
        self.assertEqual(_boundary("label_with_teacher", performs_teacher_inference=False),
                         "costly_teacher_labeling")
        self.assertEqual(
            _boundary("label_with_teacher",
                      dft_labels_used=False, protected_reference_labels_used=False),
            "costly_teacher_labeling")

    # --- other costly boundaries are distinct and never relaxed here ------------------------------
    def test_other_boundaries_untouched(self):
        self.assertEqual(_boundary("train_committee", performs_teacher_inference=False),
                         "costly_training")
        self.assertEqual(_boundary("evaluate_heldout_fidelity"), "costly_training")
        self.assertEqual(_boundary("run_student_md"), "production_md")
        self.assertEqual(_boundary("run_teacher_md"), "production_md")
        self.assertEqual(_boundary("submit_scheduler_job"), "scheduler_submission")

    def test_ungated_action_stays_none(self):
        self.assertIsNone(_boundary("some_cheap_action", performs_teacher_inference=False))

    # --- Teacher-evidence inference actions: costly COMPUTE gate is never relaxed by label provenance
    # Running the Teacher on GPU to build a baseline / reference comparison IS the guarded effect,
    # independently of whether the run also grows the training corpus. A "creates no new
    # DFT/protected-reference labels" declaration is about corpus growth, NOT compute, and so must
    # NOT relax this boundary (the defect that let a fresh 9,295-frame Teacher baseline dispatch on
    # GPU with action_approvals={}).
    def test_teacher_baseline_always_gated_even_with_no_new_labels(self):
        self.assertEqual(_boundary("build_teacher_baseline"), "costly_teacher_labeling")
        self.assertEqual(
            _boundary("build_teacher_baseline",
                      dft_labels_used=False, protected_reference_labels_used=False),
            "costly_teacher_labeling")

    def test_reference_validation_fresh_is_gated(self):
        # A fresh reference validation runs label_with_teacher over the reference population.
        self.assertEqual(_boundary("validate_teacher_reference"), "costly_teacher_labeling")
        self.assertEqual(
            _boundary("validate_teacher_reference",
                      dft_labels_used=False, protected_reference_labels_used=False),
            "costly_teacher_labeling")

    def test_reference_validation_verified_reuse_relaxed(self):
        # A bound prior verified historical_report drives the executor's verified-reuse path, which
        # recomputes metrics from already-materialized predictions and runs NO fresh Teacher.
        self.assertIsNone(
            _boundary("validate_teacher_reference",
                      historical_report={"reference_id": "prior", "metrics": {}}))


class AcquisitionRecipeClassifierTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _cfg(self, name, payload):
        p = self.tmp / name
        p.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return str(p)

    def test_augment_atoms_recipe_is_teacher_inference(self):
        # augment-atoms's executor unconditionally binds the Teacher calculator for the
        # perturbation/relaxation of every candidate structure, so its structure generation IS
        # materially costly Teacher inference and can never be relaxed by a self-asserted flag.
        path = self._cfg("augment.yaml", {"kind": "augment-atoms", "env": "augment"})
        self.assertTrue(cli._acquisition_incurs_teacher_inference(path))

    def test_augment_atoms_cannot_self_assert_no_teacher_inference(self):
        path = self._cfg("augment_liar.yaml", {
            "kind": "augment-atoms", "performs_teacher_inference": False})
        self.assertTrue(cli._acquisition_incurs_teacher_inference(path))

    def test_teacher_md_recipe_is_teacher_inference(self):
        path = self._cfg("teachermd.yaml", {"kind": "teacher-md", "T_K": 300.0})
        self.assertTrue(cli._acquisition_incurs_teacher_inference(path))

    def test_configured_adapter_callable_fails_closed(self):
        path = self._cfg("adapter.yaml", {
            "kind": "augment-atoms", "adapter": {"acquire": "pkg.mod:acquire_fn"}})
        self.assertTrue(cli._acquisition_incurs_teacher_inference(path))

    def test_missing_config_fails_closed(self):
        self.assertTrue(cli._acquisition_incurs_teacher_inference(
            str(self.tmp / "does-not-exist.yaml")))

    def test_none_path_fails_closed(self):
        self.assertTrue(cli._acquisition_incurs_teacher_inference(None))

    def test_non_dict_config_fails_closed(self):
        p = self.tmp / "scalar.yaml"
        p.write_text("just-a-string\n", encoding="utf-8")
        self.assertTrue(cli._acquisition_incurs_teacher_inference(str(p)))

    def test_unknown_kind_fails_closed(self):
        path = self._cfg("unknown.yaml", {"kind": "some-future-recipe"})
        self.assertTrue(cli._acquisition_incurs_teacher_inference(path))


if __name__ == "__main__":
    unittest.main()
