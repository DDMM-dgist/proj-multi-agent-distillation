"""UNIT 2 tests: Stage-8/9 consumption of the governed access-partition contract.

Covers the governed-isolation plumbing added on top of the UNIT 1 partition system:
  * validation.access_partition.materialize_partition_slice / enforce_and_materialize
  * workflow.steps.evaluate_committee report_fingerprints isolation
  * runtimes.pydantic_ai.executors Stage-8 / Stage-9 access-partition wiring
  * runtimes.pydantic_ai.bounded_evidence uncertainty governed_partition surfacing
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from validation.access_partition import (
    ROLE_CALIBRATION_EVAL,
    ROLE_CALIBRATION_FIT,
    ROLE_STUDENT_FINAL_EVALUATION,
    build_access_partition_contract,
    enforce_and_materialize,
    materialize_partition_slice,
    resolve_partition_fingerprints,
    write_access_partition_contract,
)


def _make_population(path, strata_counts, seeds=(1, 2), with_teacher=True, pos_shift=0.0):
    """Write a DFT-labeled eval population with per-seed committee forces embedded."""
    frames = []
    n = 0
    for stratum in sorted(strata_counts):
        for i in range(strata_counts[stratum]):
            atoms = Atoms(
                "H2O",
                positions=[[pos_shift, 0, 0], [0.9 + 0.001 * n, 0, 0], [0, 0.9, 0]],
                cell=[12, 12, 12], pbc=True,
            )
            atoms.info["config_type"] = stratum
            atoms.info["source_category"] = stratum
            atoms.info["source_local_index"] = i
            atoms.info["structure_id"] = f"{stratum}-{i}"
            atoms.info["dft_energy"] = -3.0 * (n + 1)
            atoms.arrays["dft_forces"] = np.full((3, 3), 0.01 * (n + 1))
            if with_teacher:
                atoms.info["teacher_energy"] = -3.0 * (n + 1) + 0.05
                atoms.arrays["teacher_forces"] = np.full((3, 3), 0.01 * (n + 1) + 0.002)
            for s in seeds:
                atoms.arrays[f"student_forces_seed{int(s):02d}"] = np.full((3, 3), 0.01 * (n + 1) + 0.001 * s)
            frames.append(atoms)
            n += 1
    write(str(path), frames, format="extxyz")
    return len(frames)


def _committee_manifest(path, seeds=(1, 2)):
    path.write_text(json.dumps({
        "schema_version": 1,
        "models": [{"seed": int(s), "path": f"/fake/seed-{s}.json"} for s in seeds],
    }, indent=2))
    return path


# ------------------------------------------------------------------ materialization + access


def test_materialize_preserves_arrays_and_count(tmp_path):
    xyz = tmp_path / "pop.extxyz"
    _make_population(xyz, {"a": 24, "b": 18, "c": 18})
    contract = build_access_partition_contract(xyz, "r", "recovered-original-holdout")
    out = tmp_path / "fit.extxyz"
    res = materialize_partition_slice(contract, ROLE_CALIBRATION_FIT, xyz, out,
                                      require_committee_seeds=[1, 2])
    from ase.io import read
    frames = read(str(out), index=":")
    assert len(frames) == res["n_frames"]
    assert res["n_frames"] == contract["partitions"][ROLE_CALIBRATION_FIT]["n_frames"]
    assert "student_forces_seed01" in frames[0].arrays
    assert "dft_forces" in frames[0].arrays


def test_materialize_fails_on_missing_fingerprints(tmp_path):
    xyz = tmp_path / "pop.extxyz"
    _make_population(xyz, {"a": 24, "b": 18, "c": 18})
    contract = build_access_partition_contract(xyz, "r", "recovered-original-holdout")
    other = tmp_path / "other.extxyz"
    _make_population(other, {"a": 24, "b": 18, "c": 18}, pos_shift=5.0)  # distinct geometries
    # 'other' shares stratum layout but geometries differ, so role fingerprints are absent.
    with pytest.raises(ValueError, match="were not found"):
        materialize_partition_slice(contract, ROLE_CALIBRATION_FIT, other, tmp_path / "x.extxyz")


def test_enforce_and_materialize_denies_unauthorized_role(tmp_path):
    xyz = tmp_path / "pop.extxyz"
    _make_population(xyz, {"a": 24, "b": 18, "c": 18})
    cpath = tmp_path / "ap.json"
    write_access_partition_contract(xyz, "r", "recovered-original-holdout", cpath)
    # evaluation stage may NOT materialize a calibration role.
    with pytest.raises(ValueError):
        enforce_and_materialize(cpath, "evaluation", ROLE_CALIBRATION_FIT, xyz, tmp_path / "x.extxyz")
    # uncertainty stage MAY materialize a calibration role.
    res = enforce_and_materialize(cpath, "uncertainty", ROLE_CALIBRATION_FIT, xyz,
                                  tmp_path / "fit.extxyz", require_committee_seeds=[1, 2])
    assert res["role"] == ROLE_CALIBRATION_FIT


# --------------------------------------------------------------------- Stage-8 isolation


def test_evaluate_committee_report_isolated_to_eval_slice(tmp_path):
    from workflow.steps import evaluate_committee, train_committee

    student_cfg = tmp_path / "student.yaml"
    student_cfg.write_text(
        "kind: mock\ncommittee:\n  seeds: [1, 2]\n"
        "predict:\n  factory: adapters.mock_model.MockCheckpointCalculator\n"
    )
    dataset = tmp_path / "train.extxyz"
    _make_population(dataset, {"a": 6, "b": 6})
    committee_manifest = tmp_path / "committee.json"
    train_committee(str(student_cfg), str(dataset), str(tmp_path / "committee"), str(committee_manifest))

    pop = tmp_path / "eval_pop.extxyz"
    n = _make_population(pop, {"a": 24, "b": 18, "c": 18})
    contract = build_access_partition_contract(pop, "r", "recovered-original-holdout")
    eval_fps = resolve_partition_fingerprints(contract, ROLE_STUDENT_FINAL_EVALUATION)

    labeled = tmp_path / "labeled.extxyz"
    report = tmp_path / "report.json"
    results = evaluate_committee(
        str(student_cfg), str(committee_manifest), str(pop), str(labeled), str(report),
        required_channels=["student_vs_dft", "teacher_vs_dft"],
        report_fingerprints=eval_fps)

    # Accuracy report restricted to the governed evaluation slice ...
    assert results["student_vs_dft"]["all"]["n_frames"] == len(eval_fps)
    assert results["student_vs_dft"]["all"]["n_frames"] < n
    # ... but committee predictions embedded on the FULL population (feeds Stage-9 slices).
    from ase.io import read
    assert len(read(str(labeled), index=":")) == n


# --------------------------------------------------------------------- Stage-9 isolation


def test_stage9_governed_fit_eval_isolation(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_uncertainty_report

    pop = tmp_path / "labeled_pop.extxyz"
    _make_population(pop, {"a": 24, "b": 18, "c": 18}, seeds=(1, 2))
    cpath = tmp_path / "ap.json"
    write_access_partition_contract(pop, "ref-x", "recovered-original-holdout", cpath)
    committee_manifest = _committee_manifest(tmp_path / "committee.json", seeds=(1, 2))
    report_path = tmp_path / "uncertainty_report.json"

    out = _exec_build_uncertainty_report({"parameters": {
        "committee_manifest": str(committee_manifest),
        "population_frames": str(pop),
        "report_path": str(report_path),
        "access_partition_path": str(cpath),
    }})
    report = json.loads(report_path.read_text())

    assert report["population"]["role"] == ROLE_CALIBRATION_FIT
    gp = report["governed_partition"]
    assert gp["fit_eval_disjoint"] is True
    assert gp["calibration_fit"]["role"] == ROLE_CALIBRATION_FIT
    assert gp["calibration_eval"]["role"] == ROLE_CALIBRATION_EVAL
    # fit and eval are disjoint governed slices
    assert (gp["calibration_fit"]["frame_fingerprints_sha256"]
            != gp["calibration_eval"]["frame_fingerprints_sha256"])
    assert "holdout_disagreement_summary" in gp["calibration_eval"]
    # primary report population size == fit slice size (isolated, not the whole pool)
    assert report["population"]["n_frames"] == gp["calibration_fit"]["n_frames"]
    assert out["report"]["governed_partition"]["fit_eval_disjoint"] is True


def test_stage9_ungoverned_path_unchanged(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_uncertainty_report

    pop = tmp_path / "labeled_pop.extxyz"
    _make_population(pop, {"a": 10, "b": 10}, seeds=(1, 2))
    committee_manifest = _committee_manifest(tmp_path / "committee.json", seeds=(1, 2))
    report_path = tmp_path / "uncertainty_report.json"
    _exec_build_uncertainty_report({"parameters": {
        "committee_manifest": str(committee_manifest),
        "population_frames": str(pop),
        "report_path": str(report_path),
    }})
    report = json.loads(report_path.read_text())
    assert "governed_partition" not in report
    assert report["population"]["role"] == "held_out_evaluation_population"


# --------------------------------------------------------------- Judge-facing surfacing


def test_uncertainty_adapter_surfaces_governed_partition(tmp_path):
    from runtimes.pydantic_ai.bounded_evidence import _uncertainty_report_summary

    pop = tmp_path / "labeled_pop.extxyz"
    _make_population(pop, {"a": 24, "b": 18, "c": 18}, seeds=(1, 2))
    cpath = tmp_path / "ap.json"
    write_access_partition_contract(pop, "ref-x", "recovered-original-holdout", cpath)
    from runtimes.pydantic_ai.executors import _exec_build_uncertainty_report
    committee_manifest = _committee_manifest(tmp_path / "committee.json", seeds=(1, 2))
    report_path = tmp_path / "uncertainty_report.json"
    _exec_build_uncertainty_report({"parameters": {
        "committee_manifest": str(committee_manifest),
        "population_frames": str(pop),
        "report_path": str(report_path),
        "access_partition_path": str(cpath),
    }})
    summary = _uncertainty_report_summary(json.loads(report_path.read_text()))
    gp = summary["governed_partition"]
    assert gp is not None
    assert gp["fit_eval_disjoint"] is True
    assert gp["calibration_fit"]["role"] == ROLE_CALIBRATION_FIT
    assert gp["calibration_eval"]["role"] == ROLE_CALIBRATION_EVAL


def test_uncertainty_adapter_governed_partition_absent_when_ungoverned():
    from runtimes.pydantic_ai.bounded_evidence import _uncertainty_report_summary
    summary = _uncertainty_report_summary({
        "schema_version": 1, "committee_manifest_sha256": "a" * 64, "seeds": [1, 2],
        "aggregate": "max", "u_frame_summary": {"mean": 0.1, "max": 0.2},
        "calibration": {"status": "uncalibrated", "caveat": "x"},
    })
    assert summary["governed_partition"] is None
