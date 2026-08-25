"""FE-042 regression: the deterministic Stage-4 coverage-ADEQUACY gate control.

The three data_coverage Judges review report HONESTY (access mode, per-config_type counts, lineage,
protected-test exclusion) -- NOT configuration-space ADEQUACY. After FE-041 made the report honest, a
truthful ``COVERAGE_INSUFFICIENT`` report can therefore earn a unanimous 3/3 PASS on honesty while a
declared deployment structure class still has ZERO acquired representatives. That is the live ffv4r
defect: a PASS gate would advance to Stage 5 (teacher_labeling) over structurally unsupported regions.

FE-042 adds ONE deterministic control in ``RunController.record_gate``: a would-be PASS on a stage whose
registered coverage report carries ``coverage_assessment.assessment_status == "COVERAGE_INSUFFICIENT"``
is downgraded to a scientific REVISE that binds ``pending_recovery`` and recommends
``return_stage="acquisition"`` carrying the exact unsupported declared classes. It reads only the status
the FE-038/FE-039 gap gate already derived -- it invents no threshold, quota, size, or science, and is
inert for COVERAGE_SUFFICIENT / NOT_ASSESSABLE / any non-coverage stage.

The three required regressions:
  1. Judge PASS + COVERAGE_INSUFFICIENT -> Stage-4 REVISE -> acquisition recovery -> Stage 5 unreachable.
  2. Judge PASS + COVERAGE_SUFFICIENT  -> Stage-4 PASS   -> Stage 5 eligible.
  3. Replay the exact immutable ffv4r 7-gap Stage-4 fixture -> recovery receives the exact regions.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validation.coverage_assessment import aggregate_assessment_status
from validation.coverage_gap_assessment import build_structure_class_dimensions

from tests.test_full_lifecycle_integration import FixtureHelpers, GATE_CRITERION

_PROJECT = Path(__file__).resolve().parent.parent
_FFV4R_COVERAGE = (_PROJECT / "runs" / "sio2-sox-allegro-simplenn-ffv4r"
                   / "artifacts" / "data_coverage.json")

# Synthetic frozen label_map -- no material constant, no real class name (mirrors FE-039's fixture).
_SYN_LABEL_MAP = [
    {"raw_label": "ct_a1", "canonical_domain": "class_A", "claim_role": "primary_claim",
     "rationale": "syn"},
    {"raw_label": "ct_b1", "canonical_domain": "class_B", "claim_role": "primary_claim",
     "rationale": "syn"},
]
_SYN_DECLARED = ["class_A", "class_B"]


class Fe042CoverageAdequacyGateControlTests(FixtureHelpers):
    """record_gate must convert a 3/3-PASS honesty verdict into a REVISE iff coverage is INSUFFICIENT."""

    _STAGES = [
        {"name": "acquisition", "command": None, "outputs": ["artifacts/acquisition.json"],
         "gate": {"criteria": [GATE_CRITERION]}},
        {"name": "data_coverage", "command": None, "outputs": ["artifacts/data_coverage.json"],
         "gate": {"criteria": [GATE_CRITERION]}},
    ]

    def _controller_at_data_coverage(self, root, report_payload):
        """Init a 2-stage run, PASS acquisition, and complete data_coverage with the given report."""
        controller = self._init_controller(
            root, run_id="fe042-fixture", stages=self._STAGES,
            recovery_capability_roster={"data_repair": "data-curator",
                                        "orchestration": "orchestrator"})
        acq = controller.run_dir / "artifacts/acquisition.json"
        self._write_json(acq, {"role": "acquisition", "n_frames": 1})
        controller.complete_external_stage("acquisition", [acq])
        self._gate(controller, "acquisition", "PASS")

        report_path = controller.run_dir / "artifacts/data_coverage.json"
        self._write_json(report_path, report_payload)
        controller.complete_external_stage("data_coverage", [report_path])
        return controller, report_path

    @staticmethod
    def _coverage_report(declared, acquired_counts, label_map):
        dims = build_structure_class_dimensions(declared, acquired_counts, label_map)
        return {"coverage_assessment": {"assessment_status": aggregate_assessment_status(dims),
                                        "dimensions": dims}}

    # --- Regression 1 ---------------------------------------------------------------------------
    def test_judge_pass_but_insufficient_forces_revise_and_blocks_stage5(self):
        with tempfile.TemporaryDirectory() as tmp:
            # class_B has zero acquired representatives -> COVERAGE_INSUFFICIENT.
            report = self._coverage_report(_SYN_DECLARED, {"ct_a1": 4}, _SYN_LABEL_MAP)
            self.assertEqual(report["coverage_assessment"]["assessment_status"],
                             "COVERAGE_INSUFFICIENT")
            controller, _ = self._controller_at_data_coverage(Path(tmp), report)

            # The Judges unanimously PASS the HONESTY review...
            self._gate(controller, "data_coverage", "PASS")

            # ...but the deterministic adequacy control downgraded the recorded verdict to REVISE.
            self.assertEqual(controller.stage("data_coverage")["gate"], "REVISE")
            pending = controller.state["pending_recovery"]
            self.assertEqual(pending["status"], "required")
            self.assertEqual(pending["failed_stage"], "data_coverage")
            self.assertEqual(pending["verdict"], "REVISE")

            ca = pending["coverage_adequacy"]
            self.assertEqual(ca["control"], "fe042_coverage_adequacy")
            self.assertEqual(ca["assessment_status"], "COVERAGE_INSUFFICIENT")
            self.assertEqual(ca["recommended_return_stage"], "acquisition")
            self.assertEqual(ca["unsupported_structure_classes"], ["class_B"])

            # Stage 5 (the next stage) is unreachable: a pending recovery blocks all forward progress.
            with self.assertRaises(RuntimeError):
                controller._ensure_no_pending_recovery()

    # --- Regression 2 ---------------------------------------------------------------------------
    def test_judge_pass_and_sufficient_stays_pass_and_stage5_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            # every declared class has a representative -> COVERAGE_SUFFICIENT.
            report = self._coverage_report(_SYN_DECLARED, {"ct_a1": 1, "ct_b1": 1}, _SYN_LABEL_MAP)
            self.assertEqual(report["coverage_assessment"]["assessment_status"],
                             "COVERAGE_SUFFICIENT")
            controller, _ = self._controller_at_data_coverage(Path(tmp), report)

            self._gate(controller, "data_coverage", "PASS")

            # No downgrade: an adequate report PASSes and leaves Stage 5 eligible.
            self.assertEqual(controller.stage("data_coverage")["gate"], "PASS")
            self.assertIsNone(controller.state.get("pending_recovery"))
            controller._ensure_no_pending_recovery()  # does not raise -> forward progress allowed

    # --- Regression 3 ---------------------------------------------------------------------------
    def test_replay_immutable_ffv4r_fixture_routes_exact_unsupported_regions(self):
        if not _FFV4R_COVERAGE.is_file():
            self.skipTest("immutable ffv4r data_coverage fixture not present")
        fixture = json.loads(_FFV4R_COVERAGE.read_text())
        self.assertEqual(fixture["coverage_assessment"]["assessment_status"],
                         "COVERAGE_INSUFFICIENT")
        with tempfile.TemporaryDirectory() as tmp:
            controller, report_path = self._controller_at_data_coverage(Path(tmp), fixture)

            self._gate(controller, "data_coverage", "PASS")

            self.assertEqual(controller.stage("data_coverage")["gate"], "REVISE")
            ca = controller.state["pending_recovery"]["coverage_adequacy"]
            self.assertEqual(ca["recommended_return_stage"], "acquisition")
            # the recovery receives EXACTLY the seven zero-occupancy declared classes of the real run.
            self.assertEqual(sorted(ca["unsupported_structure_classes"]), [
                "amorphous_bulk_SiO2", "condensed_pure_Si_boundary", "crystalline_SiO2",
                "high_pressure_SiO2", "liquid_or_melt_SiO2", "oxygen_vacancy_SiO2", "surface_SiO2"])
            self.assertEqual(len(ca["unsupported_structure_classes"]), 7)
            # provenance points at the exact registered coverage artifact.
            registered = controller.stage_artifacts("data_coverage")[0]
            self.assertEqual(ca["report_sha256"], registered["sha256"])
            self.assertEqual(ca["report_path"], registered["path"])


if __name__ == "__main__":
    unittest.main()
