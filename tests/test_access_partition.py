"""Tests for the governed protected-reference access-partition system."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from validation.access_partition import (
    CANONICAL_PARTITION_ROLES,
    ROLE_CALIBRATION_EVAL,
    ROLE_CALIBRATION_FIT,
    ROLE_STUDENT_FINAL_EVALUATION,
    FrameDescriptor,
    assert_stage_partition_access,
    build_access_partition_contract,
    derive_minimum_support_per_role,
    plan_stratified_partition,
    validate_access_partition_contract,
    write_access_partition_contract,
)


def _descriptors(strata_counts):
    """Build synthetic descriptors: {stratum: count} -> list[FrameDescriptor]."""
    out = []
    n = 0
    for stratum in sorted(strata_counts):
        for i in range(strata_counts[stratum]):
            fp = f"fp-{stratum}-{i:04d}-{n:06d}"
            out.append(FrameDescriptor(stratum, i, fp, [stratum, i]))
            n += 1
    return out


def _make_extxyz(path, strata_counts, with_labels=False):
    """Write a synthetic DFT-ish population; geometries unique per frame."""
    frames = []
    n = 0
    for stratum in sorted(strata_counts):
        for i in range(strata_counts[stratum]):
            atoms = Atoms(
                "H2",
                positions=[[0.0, 0.0, 0.0], [0.5 + 0.001 * n, 0.0, 0.0]],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
            atoms.info["config_type"] = stratum
            atoms.info["source_category"] = stratum
            atoms.info["source_local_index"] = i
            if with_labels:
                atoms.info["dft_energy"] = -1.234 * (n + 1)
                atoms.arrays["dft_forces"] = np.zeros((2, 3))
            frames.append(atoms)
            n += 1
    write(str(path), frames, format="extxyz")
    return len(frames)


# --------------------------------------------------------------------------- core algorithm


def test_plan_disjoint_and_covering():
    descriptors = _descriptors({"a": 40, "b": 25, "c": 7})
    assignment = plan_stratified_partition(descriptors, CANONICAL_PARTITION_ROLES)
    all_idx = [i for role in CANONICAL_PARTITION_ROLES for i in assignment[role]]
    assert len(all_idx) == len(descriptors)
    assert len(set(all_idx)) == len(descriptors)  # disjoint
    assert set(all_idx) == set(range(len(descriptors)))  # covering


def test_plan_is_deterministic():
    descriptors = _descriptors({"a": 33, "b": 33, "c": 34})
    a1 = plan_stratified_partition(descriptors, CANONICAL_PARTITION_ROLES)
    a2 = plan_stratified_partition(descriptors, CANONICAL_PARTITION_ROLES)
    assert a1 == a2


def test_plan_balances_role_sizes():
    descriptors = _descriptors({"a": 100, "b": 100, "c": 100})
    assignment = plan_stratified_partition(descriptors, CANONICAL_PARTITION_ROLES)
    sizes = sorted(len(assignment[r]) for r in CANONICAL_PARTITION_ROLES)
    assert sizes[-1] - sizes[0] <= 1  # near-equal thirds


def test_rare_categories_spread_across_roles():
    # Nine singleton categories; carried rotation must not dump all on one role.
    descriptors = _descriptors({f"cat{i}": 1 for i in range(9)})
    assignment = plan_stratified_partition(descriptors, CANONICAL_PARTITION_ROLES)
    sizes = sorted(len(assignment[r]) for r in CANONICAL_PARTITION_ROLES)
    assert sizes == [3, 3, 3]


def test_min_support_floor_tracks_strata():
    assert derive_minimum_support_per_role(24) == 24
    assert derive_minimum_support_per_role(0) == 1
    assert derive_minimum_support_per_role(1) == 1


def test_plan_requires_multiple_unique_roles():
    descriptors = _descriptors({"a": 10})
    with pytest.raises(ValueError):
        plan_stratified_partition(descriptors, ["only_one"])
    with pytest.raises(ValueError):
        plan_stratified_partition(descriptors, ["x", "x"])


# --------------------------------------------------------------------- contract build/replay


def test_build_and_validate_roundtrip(tmp_path):
    xyz = tmp_path / "pop.extxyz"
    n = _make_extxyz(xyz, {"a": 60, "b": 45, "c": 30}, with_labels=True)
    out = tmp_path / "access_partition.json"
    write_access_partition_contract(xyz, "ref-x", "recovered-original-holdout", out)
    contract = validate_access_partition_contract(out)
    assert contract["population"]["n_frames"] == n
    assert set(contract["partitions"]) == set(CANONICAL_PARTITION_ROLES)
    total = sum(contract["partitions"][r]["n_frames"] for r in CANONICAL_PARTITION_ROLES)
    assert total == n
    assert contract["disjointness"]["pairwise_disjoint"] is True


def test_assignment_ignores_dft_label_values(tmp_path):
    # Same geometry/category/index with vs without DFT labels -> identical assignment hash.
    xyz_labeled = tmp_path / "labeled.extxyz"
    xyz_bare = tmp_path / "bare.extxyz"
    _make_extxyz(xyz_labeled, {"a": 20, "b": 15, "c": 10}, with_labels=True)
    _make_extxyz(xyz_bare, {"a": 20, "b": 15, "c": 10}, with_labels=False)
    c_labeled = build_access_partition_contract(xyz_labeled, "r", "recovered-original-holdout")
    c_bare = build_access_partition_contract(xyz_bare, "r", "recovered-original-holdout")
    assert c_labeled["partition_assignment_sha256"] == c_bare["partition_assignment_sha256"]


def test_validate_detects_fingerprint_tampering(tmp_path):
    xyz = tmp_path / "pop.extxyz"
    _make_extxyz(xyz, {"a": 30, "b": 30, "c": 30})
    out = tmp_path / "ap.json"
    write_access_partition_contract(xyz, "r", "recovered-original-holdout", out)
    contract = json.loads(out.read_text())
    # Move a frame from one role to another (breaks replay + disjointness).
    victim = contract["partitions"][ROLE_CALIBRATION_FIT]["frame_fingerprints"].pop()
    contract["partitions"][ROLE_STUDENT_FINAL_EVALUATION]["frame_fingerprints"].append(victim)
    out.write_text(json.dumps(contract, indent=2))
    with pytest.raises(ValueError):
        validate_access_partition_contract(out)


def test_validate_detects_population_hash_drift(tmp_path):
    xyz = tmp_path / "pop.extxyz"
    _make_extxyz(xyz, {"a": 30, "b": 30, "c": 30})
    out = tmp_path / "ap.json"
    write_access_partition_contract(xyz, "r", "recovered-original-holdout", out)
    # Mutate the structures file after the contract was sealed.
    _make_extxyz(xyz, {"a": 31, "b": 30, "c": 30})
    with pytest.raises((RuntimeError, ValueError)):
        validate_access_partition_contract(out)


def test_validate_binds_expected_identity(tmp_path):
    xyz = tmp_path / "pop.extxyz"
    _make_extxyz(xyz, {"a": 30, "b": 30, "c": 30})
    out = tmp_path / "ap.json"
    write_access_partition_contract(xyz, "ref-good", "recovered-original-holdout", out)
    validate_access_partition_contract(out, expected_reference_id="ref-good")
    with pytest.raises(ValueError):
        validate_access_partition_contract(out, expected_reference_id="ref-wrong")


def test_population_too_small_fails_closed(tmp_path):
    # 12 strata -> floor 12 -> need 36 for 3 roles; only 12 frames present.
    xyz = tmp_path / "pop.extxyz"
    _make_extxyz(xyz, {f"cat{i}": 1 for i in range(12)})
    with pytest.raises(ValueError, match="POPULATION_TOO_SMALL_FOR_GOVERNED_PARTITION"):
        build_access_partition_contract(xyz, "r", "recovered-original-holdout")


def test_missing_stratum_fails_closed(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.5, 0, 0]], cell=[9, 9, 9], pbc=True)
    xyz = tmp_path / "pop.extxyz"
    write(str(xyz), [atoms], format="extxyz")
    with pytest.raises(ValueError, match="resolvable structural stratum"):
        build_access_partition_contract(xyz, "r", "recovered-original-holdout")


# ------------------------------------------------------------------------ access enforcement


def test_access_policy_allows_authorized_stage_role():
    contract = {"roles": list(CANONICAL_PARTITION_ROLES)}
    assert assert_stage_partition_access(contract, "evaluation", ROLE_STUDENT_FINAL_EVALUATION)
    assert assert_stage_partition_access(contract, "uncertainty", ROLE_CALIBRATION_FIT)
    assert assert_stage_partition_access(contract, "uncertainty", ROLE_CALIBRATION_EVAL)


def test_access_policy_denies_cross_role():
    contract = {"roles": list(CANONICAL_PARTITION_ROLES)}
    with pytest.raises(ValueError):
        assert_stage_partition_access(contract, "evaluation", ROLE_CALIBRATION_FIT)
    with pytest.raises(ValueError):
        assert_stage_partition_access(contract, "uncertainty", ROLE_STUDENT_FINAL_EVALUATION)


def test_access_policy_rejects_unknown_stage_and_role():
    contract = {"roles": list(CANONICAL_PARTITION_ROLES)}
    with pytest.raises(ValueError):
        assert_stage_partition_access(contract, "training", ROLE_STUDENT_FINAL_EVALUATION)
    with pytest.raises(ValueError):
        assert_stage_partition_access({"roles": []}, "evaluation", ROLE_STUDENT_FINAL_EVALUATION)
