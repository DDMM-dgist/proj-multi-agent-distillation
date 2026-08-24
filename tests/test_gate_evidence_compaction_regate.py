"""Judge-facing gate-evidence compaction + gate-only correction re-gate (recovery-004 correction).

Two independent, generic mechanisms are proven here:

1. bounded_evidence: when a gate packet already carries a four-channel accuracy_report.json (which
   surfaces the exact evaluation population and every fidelity metric, aggregate + domain-resolved),
   a co-declared raw per-frame predictions .extxyz is redundant SCIENTIFIC content for the gate and
   is surfaced LINEAGE-ONLY -- its deterministic provenance (path/sha256/size/n_frames/n_atoms) is
   preserved while its bulky per-frame composition/category distribution is dropped, keeping the
   Judge prompt within the model context budget. Without a four-channel report present the extxyz
   keeps its full frame-level summary (no behaviour change). The four-channel summary itself is
   never touched by the compaction.

2. cli._is_gate_only_correction_regate: True only when the current iteration is an
   evidence_surfacing_correction re-gate of a stage already completed with all declared outputs on
   disk -- so a correction re-gate re-judges the same accepted artifacts without re-executing the
   (expensive, deterministic) production action.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from runtimes.pydantic_ai.bounded_evidence import (
    _FOUR_CHANNEL_DISPLAY_SIG_FIGS,
    _evaluation_population_block,
    _four_channel_accuracy_report_summary,
    _round_sig,
    build_bounded_evidence,
    summarize_artifact,
)
from runtimes.pydantic_ai.cli import (
    _gate_lineage_only_artifacts,
    _is_gate_only_correction_regate,
)
from workflow.integrity import sha256_file

_METRICS = {
    "e_raw_mae_meV": 1.0, "e_raw_rmse_meV": 2.0, "e_alignment_shift_meV": 0.5,
    "e_mae_meV": 0.9, "e_rmse_meV": 1.8, "f_mae": 0.01, "f_rmse": 0.02, "f_r2": 0.99,
}


def _group(n_frames, n_atoms):
    return {"n_frames": n_frames, "n_atoms": n_atoms, **_METRICS}


def _four_channel_report():
    channel = {"all": _group(10, 200),
               "domain_a": _group(6, 120), "domain_b": _group(4, 80)}
    return {"student_vs_teacher": channel, "student_vs_dft": channel,
            "teacher_vs_dft": channel}


def _write_extxyz(path, n_frames=3):
    frames = [Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    cell=[5, 5, 5], pbc=True) for _ in range(n_frames)]
    ase_write(str(path), frames, format="extxyz")


def _artifacts_from(payload):
    return {a["artifact_path"]: a for a in payload["artifacts"]}


class GateEvidenceCompactionTests(unittest.TestCase):
    def test_extxyz_lineage_only_when_four_channel_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy_report.json"
            report.write_text(json.dumps(_four_channel_report()))
            extxyz = root / "evaluated.extxyz"
            _write_extxyz(extxyz, n_frames=3)

            gate_artifacts = [str(extxyz), str(report)]
            lineage_only = _gate_lineage_only_artifacts(gate_artifacts)
            self.assertEqual([Path(p).name for p in lineage_only], ["evaluated.extxyz"])

            payload = build_bounded_evidence(
                gate_artifacts, root / "evidence.json", lineage_only=lineage_only)
            arts = _artifacts_from(payload)
            ext = arts[str(extxyz.resolve())]

            # lineage-only: deterministic provenance preserved, bulk dropped
            self.assertEqual(ext["summary_kind"], "lineage_reference")
            self.assertEqual(ext["n_frames"], 3)
            self.assertEqual(ext["n_atoms"], 9)
            self.assertIn("sha256", ext["integrity"])
            # no per-frame bulk keys leaked in
            self.assertNotIn("frames", ext)
            self.assertNotIn("composition", ext)
            self.assertNotIn("by_frame", ext)

            # the four-channel report summary is untouched by the compaction
            rep = arts[str(report.resolve())]
            self.assertNotEqual(rep.get("summary_kind"), "lineage_reference")
            blob = json.dumps(rep)
            self.assertIn("student_vs_teacher", blob)
            self.assertIn("domain_a", blob)
            self.assertIn("f_r2", blob)

    def test_extxyz_full_summary_when_no_four_channel_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # a JSON that is NOT a four-channel report
            plain = root / "notes.json"
            plain.write_text(json.dumps({"hello": "world"}))
            extxyz = root / "evaluated.extxyz"
            _write_extxyz(extxyz, n_frames=2)

            gate_artifacts = [str(extxyz), str(plain)]
            self.assertEqual(_gate_lineage_only_artifacts(gate_artifacts), [])

            payload = build_bounded_evidence(gate_artifacts, root / "evidence.json")
            ext = _artifacts_from(payload)[str(extxyz.resolve())]
            self.assertNotEqual(ext.get("summary_kind"), "lineage_reference")


class _StubStage:
    def __init__(self, status, outputs):
        self._d = {"status": status, "outputs": outputs}

    def get(self, k, default=None):
        return self._d.get(k, default)


class _StubController:
    def __init__(self, run_dir, iterations, stage):
        self.run_dir = run_dir
        self.state = {"iterations": iterations}
        self._stage = stage

    def stage(self, name):
        return self._stage


def _correction_iter(regate_stage="evaluation"):
    return {"id": 4, "trigger": {"kind": "evidence_surfacing_correction",
                                 "regate_stage": regate_stage, "failed_stage": None}}


class GateOnlyCorrectionRegateTests(unittest.TestCase):
    def _controller(self, root, iteration, status="completed", outputs=("out.json",),
                    write_outputs=True):
        for rel in outputs:
            if write_outputs:
                (root / rel).write_text("{}")
        return _StubController(root, [iteration], _StubStage(status, list(outputs)))

    def test_true_for_completed_correction_regate_with_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root, _correction_iter())
            self.assertTrue(_is_gate_only_correction_regate(c, "evaluation"))

    def test_false_when_not_a_correction_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recovery_iter = {"id": 2, "trigger": {"kind": "recovery",
                                                  "regate_stage": None, "failed_stage": "training"}}
            c = self._controller(root, recovery_iter)
            self.assertFalse(_is_gate_only_correction_regate(c, "evaluation"))

    def test_false_when_regate_stage_targets_a_different_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root, _correction_iter(regate_stage="training"))
            self.assertFalse(_is_gate_only_correction_regate(c, "evaluation"))

    def test_false_when_stage_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root, _correction_iter(), status="running")
            self.assertFalse(_is_gate_only_correction_regate(c, "evaluation"))

    def test_false_when_declared_output_missing_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root, _correction_iter(), write_outputs=False)
            self.assertFalse(_is_gate_only_correction_regate(c, "evaluation"))

    def test_false_when_stage_has_no_declared_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root, _correction_iter(), outputs=())
            self.assertFalse(_is_gate_only_correction_regate(c, "evaluation"))


def _hp(nf, na, base):
    """A group with deliberately high-precision (many-digit) float metrics, so 4-sig-fig display
    rounding is observable and n_frames/n_atoms exactness is testable."""
    return {
        "n_frames": nf, "n_atoms": na,
        "e_raw_mae_meV": base + 0.123456789, "e_raw_rmse_meV": base + 5.987654321,
        "e_alignment_shift_meV": -1.111111111, "e_mae_meV": base + 0.019283746,
        "e_rmse_meV": base + 4.55555555, "f_mae": 0.0452139876, "f_rmse": 0.1987654321,
        "f_r2": 0.9876543219,
    }


def _hp_report():
    ch = {"all": _hp(100, 5000, 61.0),
          "bulk_cryst": _hp(60, 3000, 22.0), "cluster": _hp(40, 2000, 503.0)}
    return {"student_vs_teacher": ch, "student_vs_dft": ch, "teacher_vs_dft": ch}


class FourChannelColumnarRoundingTests(unittest.TestCase):
    def test_round_sig_is_deterministic_and_preserves_ints_and_nonfinite(self):
        self.assertEqual(_round_sig(12.339999999), 12.34)
        self.assertEqual(_round_sig(0.0452139876), 0.04521)
        self.assertEqual(_round_sig(12.34), _round_sig(12.34))  # deterministic
        self.assertEqual(_round_sig(1000), 1000)                # int passthrough
        self.assertEqual(_round_sig(0), 0.0)
        self.assertNotEqual(_round_sig(float("nan")), _round_sig(float("nan")))  # nan!=nan
        self.assertEqual(_round_sig(float("inf")), float("inf"))

    def test_columnar_preserves_all_channels_groups_and_columns(self):
        summary = _four_channel_accuracy_report_summary(_hp_report())
        self.assertEqual(summary["channels_present"],
                         ["student_vs_dft", "student_vs_teacher", "teacher_vs_dft"])
        for ch in summary["channels"].values():
            self.assertEqual(ch["group_order"], ["bulk_cryst", "cluster"])
            self.assertEqual(ch["n_groups"], 2)
            cols = ch["by_group_columnar"]
            # all 8 metric columns + the 2 population columns retained, each array aligned to order
            for key in ("e_raw_mae_meV", "e_raw_rmse_meV", "e_alignment_shift_meV", "e_mae_meV",
                        "e_rmse_meV", "f_mae", "f_rmse", "f_r2", "n_frames", "n_atoms"):
                self.assertEqual(len(cols[key]), len(ch["group_order"]))

    def test_metrics_rounded_but_population_exact(self):
        summary = _four_channel_accuracy_report_summary(_hp_report())
        svt = summary["channels"]["student_vs_teacher"]
        # metric values rounded to 4 sig figs
        self.assertEqual(svt["aggregate"]["f_mae"], 0.04521)
        self.assertEqual(svt["by_group_columnar"]["f_mae"][0], 0.04521)
        self.assertEqual(svt["by_group_columnar"]["e_mae_meV"][0], 22.02)
        # population counts are EXACT (identity), never rounded
        self.assertEqual(svt["population"], {"n_frames": 100, "n_atoms": 5000})
        self.assertEqual(svt["by_group_columnar"]["n_frames"], [60, 40])
        self.assertEqual(svt["by_group_columnar"]["n_atoms"], [3000, 2000])
        # explicit disclosure that displayed values are rounded for review only
        self.assertIn(str(_FOUR_CHANNEL_DISPLAY_SIG_FIGS), summary["display_precision"])
        self.assertIn("full-precision", summary["display_precision"])

    def test_summary_is_derived_from_but_does_not_mutate_the_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "accuracy_report.json"
            report = _hp_report()
            path.write_text(json.dumps(report))
            before = sha256_file(path)

            summary = summarize_artifact(path)["four_channel_accuracy_report"]

            after = sha256_file(path)
            self.assertEqual(before, after)  # artifact byte-identical: not mutated
            # authoritative full-precision values still readable from the on-disk artifact
            on_disk = json.loads(path.read_text())
            self.assertEqual(on_disk["student_vs_teacher"]["bulk_cryst"]["f_mae"], 0.0452139876)
            # while the Judge-facing summary carries the rounded value derived from it
            svt = summary["channels"]["student_vs_teacher"]
            self.assertEqual(svt["by_group_columnar"]["f_mae"][0], 0.04521)


def _reference_validation_record(logical_frames=10):
    """A minimal but authoritative-shaped reference_validation.json record, mirroring the fields the
    reference_validation stage writes (see artifacts/reference_validation.json)."""
    return {
        "schema_version": 1,
        "stage": "reference_validation",
        "protected_reference_use": "teacher_vs_dft_reference_validation_only",
        "evidence_source": "VERIFIED_HISTORICAL_REUSE",
        "reference": {
            "reference_id": "recovered-original-heldout-test",
            "structures_path": "/data/recovered_original_holdout_test.xyz",
            "logical_frames": logical_frames,
            "protected_source_rows": logical_frames,
            "structures_integrity": {"kind": "file", "size": 123, "sha256": "aa" * 32},
        },
        "prediction_artifact": {
            "path": "/data/teacher_reference_predictions.extxyz",
            "integrity": {"kind": "file", "size": 456, "sha256": "bb" * 32},
            "n_frames": logical_frames,
            "labels": ["teacher_energy", "teacher_forces", "dft_energy", "dft_forces"],
        },
    }


class EvaluationPopulationBlockTests(unittest.TestCase):
    def test_block_is_derived_verbatim_with_lineage_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy_report.json"
            report.write_text(json.dumps(_four_channel_report()))  # all-group n_frames == 10
            (root / "reference_validation.json").write_text(
                json.dumps(_reference_validation_record(logical_frames=10)))

            block = _evaluation_population_block(json.loads(report.read_text()), report)
            self.assertEqual(block["population_id"], "recovered-original-heldout-test")
            self.assertEqual(block["protected_reference_use"],
                             "teacher_vs_dft_reference_validation_only")
            # provenance copied verbatim from the authoritative sibling record (never invented)
            self.assertEqual(block["source_structures"]["sha256"], "aa" * 32)
            self.assertEqual(block["source_structures"]["logical_frames"], 10)
            self.assertEqual(block["teacher_reference_predictions"]["sha256"], "bb" * 32)
            self.assertIn("dft_energy", block["teacher_reference_predictions"]["labels"])
            # lineage/hash binding present and holding
            lb = block["lineage_binding"]
            self.assertTrue(lb["frame_count_binds"])
            self.assertEqual(lb["reference_logical_frames"], 10)
            self.assertEqual(set(lb["accuracy_report_population_n_frames_by_channel"].values()), {10})

    def test_block_reaches_the_summary_that_feeds_the_judge_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy_report.json"
            report.write_text(json.dumps(_four_channel_report()))
            (root / "reference_validation.json").write_text(
                json.dumps(_reference_validation_record(logical_frames=10)))

            summary = summarize_artifact(report)["four_channel_accuracy_report"]
            self.assertIn("evaluation_population", summary)
            self.assertEqual(summary["evaluation_population"]["population_id"],
                             "recovered-original-heldout-test")

    def test_no_reference_record_surfaces_gap_not_a_fabricated_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy_report.json"
            report.write_text(json.dumps(_four_channel_report()))
            # deliberately NO sibling reference_validation.json
            block = _evaluation_population_block(json.loads(report.read_text()), report)
            self.assertIn("evidence_gap", block)
            self.assertNotIn("population_id", block)

    def test_frame_count_mismatch_is_flagged_not_asserted_as_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy_report.json"
            report.write_text(json.dumps(_four_channel_report()))  # all-group n_frames == 10
            (root / "reference_validation.json").write_text(
                json.dumps(_reference_validation_record(logical_frames=999)))  # deliberate mismatch
            block = _evaluation_population_block(json.loads(report.read_text()), report)
            self.assertFalse(block["lineage_binding"]["frame_count_binds"])
            self.assertIn("unverified", block["lineage_binding"]["statement"])

    def test_surfacing_does_not_mutate_either_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy_report.json"
            report.write_text(json.dumps(_four_channel_report()))
            ref = root / "reference_validation.json"
            ref.write_text(json.dumps(_reference_validation_record(logical_frames=10)))
            report_before, ref_before = sha256_file(report), sha256_file(ref)

            summarize_artifact(report)  # builds the Judge-facing summary + population block

            self.assertEqual(sha256_file(report), report_before)
            self.assertEqual(sha256_file(ref), ref_before)


if __name__ == "__main__":
    unittest.main()
