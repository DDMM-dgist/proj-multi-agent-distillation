"""Regression tests for Recovery id=5 (Stage-8 reference-population governance).

Covers the five governance failure/pass modes bound by the approved
RecoveryPlan #5 (canonical plan hash
``ea7cac2501179e2ef0d1d5b282aa44f20b54ede1ef32e1a4a946feec016f340a``):

  T1  a legacy ``protected_reference_use=..._only`` origin descriptor does
      NOT override / substitute for a separately pinned run-scoped
      ``allowed_uses`` list;
  T2  Student-vs-DFT PASSES governance when the exact use is present in the
      pinned pre-run authorization;
  T3  an actually-unauthorized Student-vs-DFT use FAILS governance closed;
  T4  population / source-hash / frame-count mismatch FAILS governance
      closed;
  T5  the Judge-facing evidence packet exposes BOTH semantics side-by-side
      without ambiguity (``historical_origin_descriptor`` disjoint from
      ``run_scoped_allowed_uses``).

Plus a scientific-artifact-untouched fixture: the recovery is governance-only
and must not modify any scientific artifact byte.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.reference_population_governance import (
    CHANNEL_TO_REQUIRED_ALLOWED_USE,
    resolve_run_scoped_allowed_uses,
    validate_stage8_reference_population_governance,
)


# ---------- Deterministic fixture builder --------------------------------------------------
# The fixtures materialise a minimal, self-contained run directory carrying the
# exact provenance shapes the framework produces at run init: a manifest.json
# with pinned inputs, one or more `inputs/*-reference.yaml` snapshots, a
# `protected_reference_manifest.json`, and an `artifacts/reference_validation.json`
# with the schema fields the validator reads.

_PROTECTED_MANIFEST_ALL_ALLOWED = {
    "schema_name": "ProtectedReferenceManifest",
    "schema_version": "preview-1",
    "reference_id": "test-protected-reference",
    "reference_class": "ORIGINAL_TEACHER_TEST",
    "student_policy": {
        "student_training": "PROHIBITED",
        "acquisition_parent": "PROHIBITED",
        "teacher_vs_dft_evaluation": "ALLOWED",
        "student_vs_dft_evaluation": "ALLOWED",
    },
}

_PROTECTED_MANIFEST_STUDENT_DFT_PROHIBITED = {
    **_PROTECTED_MANIFEST_ALL_ALLOWED,
    "student_policy": {
        "student_training": "PROHIBITED",
        "acquisition_parent": "PROHIBITED",
        "teacher_vs_dft_evaluation": "ALLOWED",
        "student_vs_dft_evaluation": "PROHIBITED",
    },
}


def _write_reference_yaml(path: Path, reference_id: str, protection_manifest: Path,
                          allowed_uses: list[str] | None = None) -> None:
    lines = [
        "kind: protected-existing-dft",
        f"reference_id: {reference_id}",
        "reference_class: ORIGINAL_TEACHER_TEST",
        f"protection_manifest: {protection_manifest}",
    ]
    if allowed_uses is not None:
        lines.append("allowed_uses:")
        for u in allowed_uses:
            lines.append(f"- {u}")
    path.write_text("\n".join(lines) + "\n")


def _build_run(tmp_path: Path, *,
               manifest_policy: dict,
               ref_snapshot_allowed_uses: list[str] | None,
               reference_id_in_ref_yaml: str = "test-protected-reference",
               reference_id_in_reference_validation: str = "test-recovered-subset",
               logical_frames: int = 100,
               structures_sha256: str = "0" * 64,
               protected_reference_use: str = "teacher_vs_dft_reference_validation_only",
               ) -> dict:
    run_dir = tmp_path / "runs" / "sio2-test"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "inputs").mkdir()

    # Protected reference manifest
    prm_path = tmp_path / "protected_reference_manifest.json"
    prm_path.write_text(json.dumps(manifest_policy) + "\n")

    # Pinned reference.yaml snapshot (may or may not declare allowed_uses)
    ref_snap_path = run_dir / "inputs" / "005-reference.yaml"
    _write_reference_yaml(
        ref_snap_path,
        reference_id=reference_id_in_ref_yaml,
        protection_manifest=prm_path,
        allowed_uses=ref_snapshot_allowed_uses,
    )

    # Manifest.json with pinned inputs (source == snapshot in-tests)
    manifest = {
        "schema_version": 10,
        "run_id": "sio2-test",
        "inputs": [
            {"source": str(ref_snap_path),
             "snapshot": str(ref_snap_path),
             "sha256": "deadbeef" * 8},
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")

    # reference_validation.json (Stage-2 output shape)
    rv_path = run_dir / "artifacts" / "reference_validation.json"
    rv_payload = {
        "schema_version": 1,
        "profile": "teacher_reference_validation",
        "stage": "reference_validation",
        "protected_reference_use": protected_reference_use,
        "evidence_source": "VERIFIED_HISTORICAL_REUSE",
        "reference": {
            "reference_id": reference_id_in_reference_validation,
            "reference_yaml": str(ref_snap_path),
            "structures_path": str(run_dir / "artifacts" / "structures.xyz"),
            "logical_frames": logical_frames,
            "structures_integrity": {"kind": "file", "size": 1, "sha256": structures_sha256},
            "protected_source_rows": logical_frames,
        },
        "prediction_artifact": {
            "path": str(run_dir / "artifacts" / "teacher_pred.extxyz"),
            "integrity": {"kind": "file", "size": 1, "sha256": "a" * 64},
            "n_frames": logical_frames,
            "labels": ["teacher_energy", "teacher_forces", "dft_energy", "dft_forces"],
        },
    }
    rv_path.write_text(json.dumps(rv_payload) + "\n")
    return {"run_dir": run_dir, "rv": rv_payload, "prm_path": prm_path}


# ---------- T1: legacy origin descriptor must not override allowed_uses ------------------

def test_t1_legacy_origin_descriptor_does_not_override_run_scoped_allowed_uses(tmp_path):
    ctx = _build_run(
        tmp_path,
        manifest_policy=_PROTECTED_MANIFEST_ALL_ALLOWED,
        ref_snapshot_allowed_uses=[
            "teacher_vs_dft_reference_validation",
            "student_vs_dft_reference_validation",
        ],
        # Legacy field intentionally narrower than the authoritative allowed_uses.
        protected_reference_use="teacher_vs_dft_reference_validation_only",
    )
    result = validate_stage8_reference_population_governance(
        run_dir=ctx["run_dir"],
        reference_validation_payload=ctx["rv"],
        accuracy_report_channels=["student_vs_dft", "teacher_vs_dft", "student_vs_teacher"],
        channel_frame_counts={"student_vs_dft": 100, "teacher_vs_dft": 100, "student_vs_teacher": 100},
        reference_yaml_source=str(ctx["run_dir"] / "inputs" / "005-reference.yaml"),
    )
    # The authoritative list is the run-scoped allowed_uses (both channels), which the
    # validator resolved from the pinned reference contract snapshot and NOT from the
    # narrower legacy origin descriptor.
    assert result["ok"] is True, result["failures"]
    assert result["historical_origin_descriptor"]["role"] == "historical_origin_descriptor"
    assert (result["historical_origin_descriptor"]["protected_reference_use"] ==
            "teacher_vs_dft_reference_validation_only")
    assert set(result["run_scoped_allowed_uses"]["values"]) == {
        "teacher_vs_dft_reference_validation",
        "student_vs_dft_reference_validation",
    }
    assert result["authorization_scope_authority"] == "pinned_run_scoped_allowed_uses_only"
    # Anti-substitution invariant: the check block explicitly labels the origin descriptor
    # as provenance-only.
    assert result["checks"]["historical_origin_descriptor_role"] == "provenance_only"


# ---------- T2: authorized Student-vs-DFT PASSES governance -------------------------------

def test_t2_authorized_student_vs_dft_passes_governance(tmp_path):
    ctx = _build_run(
        tmp_path,
        manifest_policy=_PROTECTED_MANIFEST_ALL_ALLOWED,
        ref_snapshot_allowed_uses=[
            "teacher_vs_dft_reference_validation",
            "student_vs_dft_reference_validation",
        ],
    )
    result = validate_stage8_reference_population_governance(
        run_dir=ctx["run_dir"],
        reference_validation_payload=ctx["rv"],
        accuracy_report_channels=["student_vs_dft"],
        channel_frame_counts={"student_vs_dft": 100},
        reference_yaml_source=str(ctx["run_dir"] / "inputs" / "005-reference.yaml"),
    )
    assert result["ok"] is True, result["failures"]
    assert (result["checks"]["required_allowed_uses_by_channel"] ==
            {"student_vs_dft": "student_vs_dft_reference_validation"})


# ---------- T3: genuinely unauthorized Student-vs-DFT FAILS governance closed -------------

def test_t3_unauthorized_student_vs_dft_fails_governance(tmp_path):
    # Neither the snapshot nor the manifest permits student_vs_dft.
    ctx = _build_run(
        tmp_path,
        manifest_policy=_PROTECTED_MANIFEST_STUDENT_DFT_PROHIBITED,
        ref_snapshot_allowed_uses=["teacher_vs_dft_reference_validation"],
    )
    result = validate_stage8_reference_population_governance(
        run_dir=ctx["run_dir"],
        reference_validation_payload=ctx["rv"],
        accuracy_report_channels=["student_vs_dft", "teacher_vs_dft"],
        channel_frame_counts={"student_vs_dft": 100, "teacher_vs_dft": 100},
        reference_yaml_source=str(ctx["run_dir"] / "inputs" / "005-reference.yaml"),
    )
    assert result["ok"] is False
    codes = {failure["code"] for failure in result["failures"]}
    # Either the direct-channel authorization check fires (channel_use_not_authorized) OR the
    # snapshot-vs-manifest reconciliation fires (authorization_source_inconsistent). Both are
    # fail-closed outcomes for the T3 scenario.
    assert codes & {"channel_use_not_authorized", "authorization_evidence_missing",
                    "authorization_source_inconsistent"}, result["failures"]


# ---------- T4: population / hash / frame-count mismatch FAILS closed ---------------------

def test_t4_frame_count_mismatch_fails_governance_closed(tmp_path):
    ctx = _build_run(
        tmp_path,
        manifest_policy=_PROTECTED_MANIFEST_ALL_ALLOWED,
        ref_snapshot_allowed_uses=[
            "teacher_vs_dft_reference_validation",
            "student_vs_dft_reference_validation",
        ],
    )
    # accuracy_report claims a different frame count than reference_validation.
    result = validate_stage8_reference_population_governance(
        run_dir=ctx["run_dir"],
        reference_validation_payload=ctx["rv"],
        accuracy_report_channels=["student_vs_dft", "teacher_vs_dft"],
        channel_frame_counts={"student_vs_dft": 99, "teacher_vs_dft": 100},
        reference_yaml_source=str(ctx["run_dir"] / "inputs" / "005-reference.yaml"),
    )
    assert result["ok"] is False
    codes = {failure["code"] for failure in result["failures"]}
    assert "frame_count_mismatch" in codes, result["failures"]


def test_t4_reference_identity_missing_fails_governance_closed(tmp_path):
    ctx = _build_run(
        tmp_path,
        manifest_policy=_PROTECTED_MANIFEST_ALL_ALLOWED,
        ref_snapshot_allowed_uses=[
            "teacher_vs_dft_reference_validation",
            "student_vs_dft_reference_validation",
        ],
    )
    ctx["rv"]["reference"]["reference_id"] = ""
    ctx["rv"]["reference"]["structures_integrity"] = {"kind": "file", "size": 1}
    result = validate_stage8_reference_population_governance(
        run_dir=ctx["run_dir"],
        reference_validation_payload=ctx["rv"],
        accuracy_report_channels=["student_vs_dft"],
        channel_frame_counts={"student_vs_dft": 100},
        reference_yaml_source=str(ctx["run_dir"] / "inputs" / "005-reference.yaml"),
    )
    assert result["ok"] is False
    codes = {failure["code"] for failure in result["failures"]}
    assert {"population_identity_missing", "source_hash_missing"} & codes, result["failures"]


# ---------- T5: Judge packet exposes both semantics side-by-side without ambiguity --------

def test_t5_evidence_packet_exposes_both_semantics_disjoint(tmp_path):
    """The bounded-evidence adapter must attach BOTH ``historical_origin_descriptor``
    and ``run_scoped_allowed_uses`` to the four_channel_accuracy_report summary,
    with the roles clearly labelled and disjoint from one another."""
    ctx = _build_run(
        tmp_path,
        manifest_policy=_PROTECTED_MANIFEST_ALL_ALLOWED,
        ref_snapshot_allowed_uses=[
            "teacher_vs_dft_reference_validation",
            "student_vs_dft_reference_validation",
        ],
    )
    from runtimes.pydantic_ai.bounded_evidence import _evaluation_population_block

    # Minimal four_channel_accuracy_report shape the block reads for frame counts.
    accuracy_report_payload = {
        "student_vs_dft": {"all": {"n_frames": 100, "n_atoms": 1000}},
        "student_vs_teacher": {"all": {"n_frames": 100, "n_atoms": 1000}},
        "teacher_vs_dft": {"all": {"n_frames": 100, "n_atoms": 1000}},
    }
    report_path = ctx["run_dir"] / "artifacts" / "accuracy_report.json"
    report_path.write_text(json.dumps(accuracy_report_payload) + "\n")

    block = _evaluation_population_block(accuracy_report_payload, report_path)
    # Both semantic keys exist.
    assert "historical_origin_descriptor" in block
    assert "run_scoped_allowed_uses" in block
    assert "governance_validation" in block
    # They are disjoint (the descriptor's role is provenance_only; it never appears as an
    # authoritative allowed_uses list).
    hod = block["historical_origin_descriptor"]
    rsu = block["run_scoped_allowed_uses"]
    assert hod is not None and hod["role"] == "historical_origin_descriptor"
    assert set(rsu["values"]) == {
        "teacher_vs_dft_reference_validation",
        "student_vs_dft_reference_validation",
    }
    # The explanatory note points to the run-scoped list, not the origin descriptor.
    assert "run_scoped_allowed_uses" in block["note"]
    assert "historical_origin_descriptor" in block["note"]
    # The validator ok is True on this in-scope fixture.
    assert block["governance_validation"]["ok"] is True


# ---------- Scientific-artifact-immutability guard ----------------------------------------
# The Recovery id=5 governance recovery must not modify ANY scientific artifact byte. This
# fixture asserts that the C12F Stage-8 artifacts remain at their pre-recovery sha256s at
# the moment this test runs, guarding the run's byte-level artifact invariant.

_CANONICAL_C12F_ARTIFACT_SHAS = {
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/evaluated.extxyz":
        "0d9c7ec387929739382d1879a5d4682764b35d62d16904c3a8695857865e9506",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/accuracy_report.json":
        "d39e12457426a271cc2d84a190d49bf2efbd22814ff4f2b7ed9d7f3fae18d892",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/reference_validation.json":
        "4a5e54eacf491da852f39a9bbdf11e84c5dded03d20c802e845a5a6a28cb325b",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relpath, expected_sha", sorted(_CANONICAL_C12F_ARTIFACT_SHAS.items()))
def test_c12f_scientific_artifacts_untouched_by_recovery_005(relpath, expected_sha):
    from workflow.integrity import sha256_file
    path = _project_root() / relpath
    if not path.is_file():
        pytest.skip(f"C12F artifact not present in this checkout: {relpath}")
    assert sha256_file(path) == expected_sha, (
        f"C12F scientific artifact {relpath!r} sha256 drifted from the pre-Recovery-005 "
        f"pin — Recovery id=5 must NOT modify any scientific artifact byte")
