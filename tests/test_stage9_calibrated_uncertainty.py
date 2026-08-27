import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from validation.access_partition import write_access_partition_contract
from validation.uncertainty import validate_uncertainty_report


def _population(path, *, with_dft=True, seeds=(1, 2, 3)):
    frames = []
    n = 0
    for stratum in ("a", "b", "c"):
        for local in range(9):
            atoms = Atoms(
                "SiO2",
                positions=[[0, 0, 0], [1.6 + 0.01 * n, 0, 0], [0, 1.6, 0]],
                cell=[12, 12, 12],
                pbc=True,
            )
            atoms.info["source_category"] = stratum
            atoms.info["source_local_index"] = local
            atoms.info["structure_id"] = f"{stratum}-{local}"
            if with_dft:
                atoms.arrays["dft_forces"] = np.zeros((3, 3))
            for seed in seeds:
                atoms.arrays[f"student_forces_seed{seed:02d}"] = np.full((3, 3), 0.01 * seed + 0.001 * n)
            frames.append(atoms)
            n += 1
    write(str(path), frames, format="extxyz")


def _committee(path, seeds=(1, 2, 3)):
    path.write_text(json.dumps({"schema_version": 1, "models": [
        {"seed": int(seed), "path": f"/not/used/{seed}"} for seed in seeds
    ]}))
    return path


def _policy(path, target=90):
    path.write_text(json.dumps({
        "policy_id": "unc-test",
        "_scientific_semantics_note": {"nominal_coverage_target_percent": target},
    }))
    return path


def test_stage9_split_conformal_calibration_report(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_uncertainty_report

    pop = tmp_path / "population.extxyz"
    _population(pop)
    partition = tmp_path / "access_partition.json"
    write_access_partition_contract(pop, "ref", "recovered-original-holdout", partition)
    report = tmp_path / "uncertainty_report.json"

    _exec_build_uncertainty_report({"parameters": {
        "committee_manifest": str(_committee(tmp_path / "committee.json")),
        "population_frames": str(pop),
        "report_path": str(report),
        "access_partition_path": str(partition),
        "uncertainty_policy": str(_policy(tmp_path / "unc_policy.json")),
        "require_calibrated": True,
    }})
    payload = validate_uncertainty_report(report)
    calibration = payload["calibration"]
    assert calibration["status"] == "calibrated"
    assert calibration["nominal_coverage"] == 0.90
    assert calibration["quantity"] == "frame_max_abs_force_component_error_of_committee_mean_vs_dft"
    assert calibration["fit"]["frame_fingerprints_sha256"] != calibration["eval"]["frame_fingerprints_sha256"]
    assert calibration["coverage_eval"]["total_count"] == calibration["eval"]["n_frames"]
    assert calibration["decision"] == "HUMAN_SCIENTIFIC_INPUT_REQUIRED"
    assert "coverage_acceptance" not in calibration


def test_stage9_missing_bound_coverage_target_fails_closed(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_uncertainty_report

    pop = tmp_path / "population.extxyz"
    _population(pop)
    partition = tmp_path / "access_partition.json"
    write_access_partition_contract(pop, "ref", "recovered-original-holdout", partition)
    with pytest.raises(ValueError, match="requires access_partition_path and uncertainty_policy"):
        _exec_build_uncertainty_report({"parameters": {
            "committee_manifest": str(_committee(tmp_path / "committee.json")),
            "population_frames": str(pop),
            "report_path": str(tmp_path / "report.json"),
            "access_partition_path": str(partition),
            "require_calibrated": True,
        }})


def test_stage9_unlabeled_calibration_population_fails_closed(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_uncertainty_report

    pop = tmp_path / "population.extxyz"
    _population(pop, with_dft=False)
    partition = tmp_path / "access_partition.json"
    write_access_partition_contract(pop, "ref", "recovered-original-holdout", partition)
    with pytest.raises(ValueError, match="missing dft_forces"):
        _exec_build_uncertainty_report({"parameters": {
            "committee_manifest": str(_committee(tmp_path / "committee.json")),
            "population_frames": str(pop),
            "report_path": str(tmp_path / "report.json"),
            "access_partition_path": str(partition),
            "uncertainty_policy": str(_policy(tmp_path / "unc_policy.json")),
            "require_calibrated": True,
        }})


def test_calibrated_report_cannot_claim_pass_without_bound_acceptance(tmp_path):
    pop = tmp_path / "population.extxyz"
    _population(pop)
    partition = tmp_path / "access_partition.json"
    write_access_partition_contract(pop, "ref", "recovered-original-holdout", partition)
    from runtimes.pydantic_ai.executors import _exec_build_uncertainty_report
    report = tmp_path / "uncertainty_report.json"
    _exec_build_uncertainty_report({"parameters": {
        "committee_manifest": str(_committee(tmp_path / "committee.json")),
        "population_frames": str(pop),
        "report_path": str(report),
        "access_partition_path": str(partition),
        "uncertainty_policy": str(_policy(tmp_path / "unc_policy.json")),
        "require_calibrated": True,
    }})
    payload = json.loads(report.read_text())
    payload["calibration"]["decision"] = "PASS"
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cannot PASS without bound coverage_acceptance"):
        validate_uncertainty_report(report)
