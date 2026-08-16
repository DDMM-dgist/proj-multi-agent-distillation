"""Tests for the new `recovered-original-holdout` reference kind (Part C): validates an
algorithmically-reconstructed original Teacher train/validation/test split membership (e.g.
the genuine 1,142-frame test partition recovered for the SiO2/Allegro campaign), as opposed to
`protected-existing-dft`'s physically-recovered historical artifact -- generic over frame
count, split name, and campaign identity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from validation.protected_reference import (
    RECOVERED_HOLDOUT_REFERENCE_CLASS,
    validate_reference_config,
)
from workflow.integrity import sha256_file


def _split_manifest(path: Path, records) -> None:
    path.write_text(json.dumps({"records": records}))


def _frame(category, local_index, split_tag, x=0.0, energy=-1.0):
    a = Atoms("Cu", positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True)
    a.info["source_category"] = category
    a.info["source_local_index"] = local_index
    a.info["dft_energy"] = energy
    a.arrays["dft_forces"] = np.array([[0.0, 0.0, 0.0]])
    a.info["_split_tag_for_test"] = split_tag  # not read by the validator; documents intent
    return a


def _build(tmp_path, *, n_holdout=2, corrupt_split=False, ambiguous=False,
           missing_label=False, wrong_count=False, wrong_manifest_sha=False,
           duplicate_frame=False):
    records = [
        {"source_category": "bulk", "source_local_index": 0, "split": "train"},
        {"source_category": "bulk", "source_local_index": 1, "split": "test"},
        {"source_category": "bulk", "source_local_index": 2, "split": "test"},
    ]
    if ambiguous:
        records.append({"source_category": "bulk", "source_local_index": 1, "split": "train"})
    manifest_path = tmp_path / "split_manifest.json"
    _split_manifest(manifest_path, records)

    frames = [_frame("bulk", 1, "test", x=1.0), _frame("bulk", 2, "test", x=2.0)][:n_holdout]
    if corrupt_split:
        frames = [_frame("bulk", 0, "train")]  # train-partition frame, wrong target_split
    if missing_label:
        frames[0].info.pop("dft_energy")
    if duplicate_frame:
        frames.append(_frame("bulk", 1, "test", x=99.0))
    structures_path = tmp_path / "holdout.extxyz"
    write(str(structures_path), frames)

    reference = tmp_path / "recovered_reference.yaml"
    reference.write_text(yaml.safe_dump({
        "kind": "recovered-original-holdout",
        "reference_id": "test-recovered-holdout",
        "reference_class": RECOVERED_HOLDOUT_REFERENCE_CLASS,
        "status": "AVAILABLE_AND_VERIFIED",
        "target_split": "test",
        "split_source_manifest": str(manifest_path),
        "split_source_manifest_sha256":
            ("0" * 64) if wrong_manifest_sha else sha256_file(manifest_path),
        "frame_count": (len(frames) + 1) if wrong_count else len(frames),
        "structures": {
            "path": str(structures_path),
            "logical_frames": (len(frames) + 1) if wrong_count else len(frames),
            "sha256": sha256_file(structures_path),
        },
        "prohibited_uses": ["student_training", "student_validation_tuning", "acquisition_seed",
                            "augmentation_parent", "recovery_training"],
    }))
    return reference


def test_valid_recovered_holdout_reference_passes(tmp_path):
    reference = _build(tmp_path)
    result = validate_reference_config(reference)
    assert result["logical_frames"] == 2
    assert result["protected_source_rows"] == 2
    assert {"bulk", "bulk"} == {k[0] for k in result["protected_source_indices"]}


def test_train_partition_frame_is_rejected(tmp_path):
    reference = _build(tmp_path, corrupt_split=True)
    with pytest.raises(ValueError, match="target_split"):
        validate_reference_config(reference)


def test_ambiguous_split_key_is_rejected(tmp_path):
    reference = _build(tmp_path, ambiguous=True)
    with pytest.raises(ValueError, match="ambiguous"):
        validate_reference_config(reference)


def test_missing_dft_label_is_rejected(tmp_path):
    reference = _build(tmp_path, missing_label=True)
    with pytest.raises(ValueError, match="dft_energy"):
        validate_reference_config(reference)


def test_frame_count_mismatch_is_rejected(tmp_path):
    reference = _build(tmp_path, wrong_count=True)
    with pytest.raises(RuntimeError, match="frame-count mismatch"):
        validate_reference_config(reference)


def test_manifest_sha_mismatch_is_rejected(tmp_path):
    reference = _build(tmp_path, wrong_manifest_sha=True)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_reference_config(reference)


def test_duplicate_source_key_is_rejected(tmp_path):
    reference = _build(tmp_path, duplicate_frame=True)
    with pytest.raises(ValueError, match="duplicate source key"):
        validate_reference_config(reference)


def test_unknown_reference_kind_is_rejected(tmp_path):
    reference = tmp_path / "bad.yaml"
    reference.write_text(yaml.safe_dump({"kind": "not-a-real-kind"}))
    with pytest.raises(ValueError, match="not a recognized"):
        validate_reference_config(reference)
