"""PC001 final teacher-validation gate — integrity tests (network-free; NO model/DFT/compute).

Verifies the authoritative teacher gate: no student metrics entered it; no threshold was invented;
outlier exclusion is domain-justified; target domain declared before performance; the preliminary run is
immutable and the final run is fresh; historical students are not set as the new-pipeline current student;
and downstream sequencing authorizes the distillation-dataset stage, NOT the student stage.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "runs" / "production_campaign_001" / "pc001-teacher-validation-final"
PRELIM = ROOT / "runs" / "production_campaign_001" / "pc001-teacher-validation"
EVAL_SRC = ROOT / "work" / "production_campaign_001_teacher_validation_final.py"


@unittest.skipUnless(FINAL.is_dir(), "PC001 final run not present")
class PC001TeacherValidationFinalTests(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads((FINAL / "teacher_validation_summary.json").read_text())
        self.crit = json.loads((FINAL / "criterion_results.json").read_text())
        self.rm = json.loads((FINAL / "run_manifest.json").read_text())

    def test_no_student_metrics_in_teacher_gate(self):
        # the evaluator must not READ any student error/prediction DATA file (naming historical students
        # as benchmark assets is allowed and required; ingesting their metrics is not).
        src = EVAL_SRC.read_text()
        # match the ACTUAL student data files/paths (never a bare 'error_b' token, which the docstring
        # uses only to state such data is NOT read)
        for banned in ("simplenn_vs", "error_b_clean", "error_c_simplenn", "error_b_simplenn",
                       "gpu_return_v5_committee/v5_committee_bundle", "SIMPLE_NN_DISTILLATION_CE",
                       "potential_saved_bestmodel"):
            self.assertNotIn(banned, src, f"student data/metric ingested in teacher gate: {banned}")
        self.assertIs(self.summary["student_metrics_used"], False)
        self.assertIs(self.crit["student_metrics_used"], False)

    def test_no_invented_threshold(self):
        self.assertIs(self.summary["invented_thresholds_used"], False)
        self.assertEqual(self.summary["threshold_provenance"]["source_grounded_teacher_vs_dft_threshold"], "NONE FOUND")
        # the deployed-student-comparison logic must be gone
        self.assertNotIn("below deployed student", EVAL_SRC.read_text().lower())

    def test_outlier_exclusion_is_domain_justified(self):
        out = json.loads((FINAL / "teacher_outlier_audit.json").read_text())["outliers"]
        big = [o for o in out if o["Fmae"] > 5]
        self.assertTrue(big, "expected the extreme outlier recorded")
        for o in big:
            self.assertFalse(o["in_scope"])                       # excluded by domain, not cherry-picking
            self.assertIn("out-of-scope", o["reason"])

    def test_target_domain_declared(self):
        self.assertIn("SiO2x_clustered_vacancy_voidsurface", self.summary["target_domain"])
        self.assertIn("amorphous_SiO2", self.summary["target_domain"])

    def test_seven_axes_present_no_average(self):
        axes = self.crit["axis_verdicts"]
        self.assertEqual(set(axes), {"A_DFT_REFERENCE_VALIDITY", "B_TEACHER_IDENTITY_AND_PROVENANCE",
                                     "C_TEACHER_FORCE_FIDELITY", "D_TEACHER_ENERGY_FIDELITY",
                                     "E_TARGET_DOMAIN_COVERAGE", "F_OUTLIER_AND_FAILURE_MODE",
                                     "G_TEACHER_PHYSICAL_CONSISTENCY"})
        for v in axes.values():
            self.assertIn(v, {"PASS", "REVISE", "FAIL", "UNRESOLVED"})

    def test_final_verdict_is_canonical_state(self):
        self.assertIn(self.summary["FINAL_TEACHER_VERDICT"],
                      {"TEACHER_ACCEPTED_FOR_DISTILLATION", "TEACHER_REVISE_BEFORE_DISTILLATION",
                       "TEACHER_REJECTED_FOR_TARGET_DOMAIN", "TEACHER_STATUS_UNRESOLVED"})
        self.assertNotEqual(self.summary["FINAL_TEACHER_VERDICT"], "ACCEPT_CONDITIONAL")

    def test_downstream_sequencing(self):
        ps = self.summary["pipeline_state"]
        if self.summary["FINAL_TEACHER_VERDICT"] == "TEACHER_ACCEPTED_FOR_DISTILLATION":
            self.assertIs(ps["DISTILLATION_DATASET_STAGE_AUTHORIZED"], True)
            self.assertIs(ps["STUDENT_STAGE_AUTHORIZED"], False)
            self.assertEqual(ps["next_campaign"], "PC002_DISTILLATION_DATASET_DESIGN")
        self.assertIs(ps["original_vs_v5_is_next_action"], False)

    def test_historical_students_not_current(self):
        ps = self.summary["pipeline_state"]
        self.assertEqual(ps["NEW_PIPELINE_CURRENT_STUDENT"], "NONE")
        self.assertIn("v5_committee", ps["EXISTING_HISTORICAL_STUDENT_ASSETS"])

    def test_no_model_calls_and_preliminary_immutable(self):
        prov = json.loads((FINAL / "provenance.json").read_text())
        for k in ("teacher_invoked", "no_dft", "no_md", "no_training", "no_network"):
            self.assertIn(k, prov)
        self.assertIs(prov["teacher_invoked"], False)
        self.assertIs(prov["student_data_read"], False)
        # preliminary run exists (immutable evidence) and is a distinct run
        if PRELIM.is_dir():
            self.assertTrue((PRELIM / "teacher_validation_verdict.json").exists())
            self.assertNotEqual(PRELIM.resolve(), FINAL.resolve())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
