"""Tests for the generic, material-agnostic ``protected-structure-identity`` reference kind:
a PROTECTION-ONLY reference that expresses a protected population by STRUCTURE IDENTITY
(source-pool indices + geometry fingerprints) and carries NO DFT/Teacher labels.

This kind exists so early-stage (Stages 1-7) disjointness checks -- acquisition,
teacher_labeling, dataset_split, training -- can verify a Student-side population does not
overlap the protected population using structure identity alone, without ever materializing or
reading protected DFT energy/force/stress truth (which is access-gated to Stages 8/9).

Test map (framework-fix spec, Option A, §4):
  A  geometry-only protection reference -> acquisition disjointness PASS
  B  geometry-only protection reference -> teacher_labeling disjointness PASS
  C  geometry-only protection reference -> dataset_split disjointness PASS
  D  protected frame present in candidate/train -> FAIL CLOSED (geometry + source-index + lineage)
  E  DFT/Teacher label access attempt via a protection reference at Stages 1-7 -> FAIL CLOSED
  F  existing DFT-labeled EVALUATION reference (Stage 8/9) still works
  G  existing supported reference kinds + capability classification regression PASS
  H  source-index / structures fingerprint mapping tampered -> FAIL CLOSED
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from validation.protected_reference import (
    PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS,
    RECOVERED_HOLDOUT_REFERENCE_CLASS,
    assert_reference_is_protection_capable,
    assert_reference_provides_dft_labels,
    reference_kind_of,
    reference_kind_provides_dft_labels,
    validate_protection_audit_report,
    validate_reference_config,
    write_protection_audit,
)
from workflow.integrity import sha256_file


# Prohibitions every held-out protection reference must declare (shared with recovered-holdout).
REQUIRED_PROHIBITIONS = [
    "student_training",
    "student_validation_tuning",
    "acquisition_seed",
    "augmentation_parent",
    "recovery_training",
]

# Source-pool rows that belong to the protected population, in this synthetic campaign.
PROTECTED_INDICES = [11, 12, 13]


def _identity_frame(x):
    """A geometry-only protected frame: species + positions + cell + pbc, NO labels."""
    return Atoms("Si", positions=[[x, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)


def _student_frame(x, source_index):
    """A Student-side frame carrying only the workflow lineage tag (no labels needed here)."""
    a = Atoms("Si", positions=[[x, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    a.info["parent_structure_id"] = f"seed-pool:{source_index}"
    return a


def _build_protection_reference(
    tmp_path,
    *,
    n_frames=3,
    add_label_frame=False,
    tamper_index_sha=False,
    tamper_structures_sha=False,
    wrong_index_count=False,
    duplicate_geometry=False,
):
    """Author a geometry-only ``protected-structure-identity`` reference.yaml and its inputs.

    Nothing about the material, frame count, or campaign is hardcoded in the framework: the
    validator reads whatever source-index file + geometry-only structures file the config names.
    """
    # (1) Protected source-pool indices file (newline-separated integers).
    indices = list(PROTECTED_INDICES)
    indices_path = tmp_path / "protected_source_indices.txt"
    indices_path.write_text("\n".join(str(i) for i in indices) + "\n")

    # (2) Geometry-only protected structures file (no DFT/Teacher labels).
    frames = [_identity_frame(float(i + 1)) for i in range(n_frames)]
    if duplicate_geometry:
        frames.append(_identity_frame(1.0))  # collides with frames[0]
    if add_label_frame:
        leaking = _identity_frame(99.0)
        leaking.info["dft_energy"] = -123.4  # forbidden label truth
        frames.append(leaking)
    structures_path = tmp_path / "protected_geometry.extxyz"
    write(str(structures_path), frames)

    index_sha = ("0" * 64) if tamper_index_sha else sha256_file(indices_path)
    struct_sha = ("0" * 64) if tamper_structures_sha else sha256_file(structures_path)

    reference = tmp_path / "protection_reference.yaml"
    reference.write_text(yaml.safe_dump({
        "kind": "protected-structure-identity",
        "reference_id": "test-protection-only",
        "reference_class": PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS,
        "status": "IDENTITY_AVAILABLE_AND_PROTECTED",
        "protected_source_rows": (len(indices) + 1) if wrong_index_count else len(indices),
        "protected_source_indices_file": str(indices_path),
        "protected_source_indices_sha256": index_sha,
        "structures": {
            "path": str(structures_path),
            "logical_frames": len(frames),
            "sha256": struct_sha,
        },
        "prohibited_uses": list(REQUIRED_PROHIBITIONS),
    }))
    return reference


def _write_audit_manifest(tmp_path, stage, dataset_path, selected_source_indices):
    manifest = tmp_path / f"{stage}_protection_audit_input.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "stage": stage,
        "selected_source_indices": list(selected_source_indices),
        "datasets": [{"role": "candidate", "path": str(dataset_path)}],
    }))
    return manifest


# --------------------------------------------------------------------------------------
# A / B / C -- geometry-only protection reference drives each early-stage disjointness PASS
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["acquisition", "teacher_labeling", "dataset_split"])
def test_geometry_only_reference_passes_early_stage_disjointness(tmp_path, stage):
    reference = _build_protection_reference(tmp_path)

    # A disjoint Student-side dataset: distinct geometries, non-protected source rows.
    student = tmp_path / "student_dataset.extxyz"
    write(str(student), [_student_frame(100.0, 1), _student_frame(200.0, 2)])

    manifest = _write_audit_manifest(tmp_path, stage, student, selected_source_indices=[1, 2])
    result = validate_protection_audit_report(manifest, reference_yaml=reference)

    assert result["status"] == "PASS"
    assert result["stage"] == stage
    assert result["reference_id"] == "test-protection-only"
    assert result["protected_source_rows"] == len(PROTECTED_INDICES)


def test_geometry_only_reference_validates_to_protection_dict(tmp_path):
    reference = _build_protection_reference(tmp_path)
    protection = validate_reference_config(reference)
    assert protection["protected_source_indices"] == set(PROTECTED_INDICES)
    assert protection["logical_frames"] == len(PROTECTED_INDICES)
    assert len(protection["reference_fingerprints"]) == len(PROTECTED_INDICES)
    # Early stages should obtain the same dict via the protection-capable accessor.
    assert assert_reference_is_protection_capable(reference) == protection


# --------------------------------------------------------------------------------------
# D -- a protected frame present in the candidate/train set FAILs CLOSED, three ways
# --------------------------------------------------------------------------------------
def test_protected_geometry_in_student_dataset_fails_closed(tmp_path):
    reference = _build_protection_reference(tmp_path)
    # frames[0] of the protection set is at x=1.0; reproduce that geometry Student-side.
    leaking = tmp_path / "leaking_geometry.extxyz"
    write(str(leaking), [_student_frame(1.0, 5)])
    manifest = _write_audit_manifest(tmp_path, "acquisition", leaking, selected_source_indices=[5])
    with pytest.raises(ValueError, match="protected reference geometry"):
        validate_protection_audit_report(manifest, reference_yaml=reference)


def test_protected_source_index_selected_fails_closed(tmp_path):
    reference = _build_protection_reference(tmp_path)
    student = tmp_path / "student_dataset.extxyz"
    write(str(student), [_student_frame(100.0, 11)])  # source row 11 is protected
    manifest = _write_audit_manifest(
        tmp_path, "acquisition", student, selected_source_indices=[11]
    )
    with pytest.raises(ValueError, match="leakage through source indices"):
        validate_protection_audit_report(manifest, reference_yaml=reference)


def test_protected_parent_lineage_fails_closed(tmp_path):
    reference = _build_protection_reference(tmp_path)
    # Distinct geometry, non-selected source index, but descended from a protected seed row.
    descendant = tmp_path / "descendant.extxyz"
    write(str(descendant), [_student_frame(300.0, 12)])  # parent seed-pool:12 is protected
    manifest = _write_audit_manifest(
        tmp_path, "acquisition", descendant, selected_source_indices=[2]
    )
    with pytest.raises(ValueError, match="protected reference descendant"):
        validate_protection_audit_report(manifest, reference_yaml=reference)


# --------------------------------------------------------------------------------------
# E -- a protection reference must never expose DFT/Teacher label truth to Stages 1-7
# --------------------------------------------------------------------------------------
def test_protection_reference_carrying_dft_label_is_rejected(tmp_path):
    reference = _build_protection_reference(tmp_path, add_label_frame=True)
    with pytest.raises(ValueError, match="forbidden label"):
        validate_reference_config(reference)


def test_protection_only_reference_cannot_be_used_as_evaluation_reference(tmp_path):
    reference = _build_protection_reference(tmp_path)
    # Early-stage protection access is allowed...
    assert assert_reference_is_protection_capable(reference)["reference_id"] == "test-protection-only"
    # ...but the Stage-8/9 evaluation-label path fails closed for a protection-only reference.
    with pytest.raises(ValueError, match="carries no DFT/Teacher labels"):
        assert_reference_provides_dft_labels(reference)


# --------------------------------------------------------------------------------------
# F -- the existing DFT-labeled EVALUATION reference path still works end to end
# --------------------------------------------------------------------------------------
def _build_recovered_holdout_reference(tmp_path):
    records = [
        {"source_category": "bulk", "source_local_index": 0, "split": "train"},
        {"source_category": "bulk", "source_local_index": 1, "split": "test"},
        {"source_category": "bulk", "source_local_index": 2, "split": "test"},
    ]
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"records": records}))

    def frame(local_index, x):
        a = Atoms("Cu", positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["source_category"] = "bulk"
        a.info["source_local_index"] = local_index
        a.info["dft_energy"] = -1.0
        a.arrays["dft_forces"] = np.array([[0.0, 0.0, 0.0]])
        return a

    frames = [frame(1, 1.0), frame(2, 2.0)]
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
        "split_source_manifest_sha256": sha256_file(manifest_path),
        "frame_count": len(frames),
        "structures": {
            "path": str(structures_path),
            "logical_frames": len(frames),
            "sha256": sha256_file(structures_path),
        },
        "prohibited_uses": list(REQUIRED_PROHIBITIONS),
    }))
    return reference


def test_existing_dft_labeled_evaluation_reference_still_works(tmp_path):
    reference = _build_recovered_holdout_reference(tmp_path)
    # Still validates as a protection-capable reference (Stages 1-7 disjointness)...
    protection = validate_reference_config(reference)
    assert protection["protected_source_rows"] == 2
    # ...AND is accepted on the Stage-8/9 DFT-label evaluation path.
    assert assert_reference_provides_dft_labels(reference)["reference_id"] == "test-recovered-holdout"


# --------------------------------------------------------------------------------------
# G -- capability classification + supported-kind regression
# --------------------------------------------------------------------------------------
def test_reference_kind_capability_classification(tmp_path):
    assert reference_kind_provides_dft_labels("protected-existing-dft") is True
    assert reference_kind_provides_dft_labels("recovered-original-holdout") is True
    assert reference_kind_provides_dft_labels("protected-structure-identity") is False
    with pytest.raises(ValueError, match="not a recognized"):
        reference_kind_provides_dft_labels("not-a-real-kind")

    protection_ref = _build_protection_reference(tmp_path)
    assert reference_kind_of(protection_ref) == "protected-structure-identity"


def test_unknown_reference_kind_still_rejected(tmp_path):
    reference = tmp_path / "bad.yaml"
    reference.write_text(yaml.safe_dump({"kind": "not-a-real-kind"}))
    with pytest.raises(ValueError, match="not a recognized"):
        validate_reference_config(reference)


# --------------------------------------------------------------------------------------
# H -- tampered source-index / structures fingerprint mapping FAILs CLOSED
# --------------------------------------------------------------------------------------
def test_tampered_source_index_sha_fails_closed(tmp_path):
    reference = _build_protection_reference(tmp_path, tamper_index_sha=True)
    with pytest.raises(RuntimeError, match="protected_source_indices_file SHA-256 mismatch"):
        validate_reference_config(reference)


def test_tampered_structures_sha_fails_closed(tmp_path):
    reference = _build_protection_reference(tmp_path, tamper_structures_sha=True)
    with pytest.raises(RuntimeError, match="structures SHA-256 mismatch"):
        validate_reference_config(reference)


def test_source_index_count_mismatch_fails_closed(tmp_path):
    reference = _build_protection_reference(tmp_path, wrong_index_count=True)
    with pytest.raises(ValueError, match="protected_source_rows"):
        validate_reference_config(reference)


def test_duplicate_geometry_in_reference_fails_closed(tmp_path):
    reference = _build_protection_reference(tmp_path, duplicate_geometry=True)
    with pytest.raises(ValueError, match="duplicate geometries"):
        validate_reference_config(reference)


# --------------------------------------------------------------------------------------
# write_protection_audit end-to-end (the artifact-producing contract path)
# --------------------------------------------------------------------------------------
def test_reference_validation_executor_rejects_protection_only_reference(tmp_path):
    """E (executor wiring): the DFT-label-reading reference_validation executor must fail
    closed if a protection-only structure-identity reference is misrouted to it, so protected
    labels are never read on behalf of an identity-only reference."""
    from runtimes.pydantic_ai import executors

    reference = _build_protection_reference(tmp_path)
    proposal = {"parameters": {
        "reference_yaml": str(reference),
        "teacher_config": str(tmp_path / "teacher.yaml"),
        "predictions_path": str(tmp_path / "preds.extxyz"),
        "report_path": str(tmp_path / "report.json"),
    }}
    with pytest.raises(ValueError, match="carries no DFT/Teacher labels"):
        executors._exec_validate_teacher_reference(proposal)


def test_write_protection_audit_produces_artifact_only_when_clean(tmp_path):
    reference = _build_protection_reference(tmp_path)
    student = tmp_path / "student_dataset.extxyz"
    write(str(student), [_student_frame(100.0, 1)])
    out = tmp_path / "acquisition_protection_audit.json"
    written = write_protection_audit(
        "acquisition",
        reference_yaml=reference,
        dataset_specs=[f"candidate={student}"],
        output=out,
        selected_source_indices=[1],
    )
    assert Path(written).is_file()
    payload = json.loads(Path(written).read_text())
    assert payload["stage"] == "acquisition"
