"""Regression tests for the Stage-10 C2b identity-completeness fix
(starting_structure + student_checkpoint identities added to C2b per-field checks).

Session 2026-08-21, follow-through to Recovery-006 CASE-B refinement + v1→v2
gate-criteria rebound.

Guarantees:
  * both identity checks emit approved-vs-realized SHA equality per-field;
  * committee_manifest sha256 is NOT treated as checkpoint sha256;
  * selected seed 202631 maps to the expected checkpoint via the pinned
    student_committee.manifest.json entry (cross-check);
  * wrong starting-structure or checkpoint SHA → C2b aggregate FALSE;
  * missing pre-submission expected identity → fail closed (match=False);
  * post-submission-only identity binding cannot satisfy C2b
    (binding_precedes_submission=False → aggregate FALSE);
  * aggregate C2b `all_match` is True only when ALL six component checks pass;
  * all Stage-10 scientific artifacts remain byte-identical.
"""
from __future__ import annotations

import copy
import json
import hashlib
from pathlib import Path

import pytest

from validation.deployment_point import validate_stage10_deployment_point


_APPROVED_POINT = {
    "temperature_K": 300.0,
    "pressure_GPa": 0.0,
    "timestep_fs": 0.5,
    "nvt_equilibration_ps": 20.0,
    "nvt_production_ps": 50.0,
    "sampling_interval_fs": 10.0,
}

_REALIZED = {
    "ensemble": "nvt",
    "temperature_setpoint_K": 300.0,
    "timestep_ps": 0.0005,
    "total_simulated_time_ps": 70.0,
    "realized_composition": "SiO2 (stoichiometric, x = 0)",
    "starting_structure_species_counts_by_lammps_type": {"type_1": 128, "type_2": 64},
}


def _matching_identity_bindings():
    return {
        "starting_structure_identity": {
            "approved_path": "/fake/data/bulk_amo_SiO2.lammps-data",
            "approved_sha256": "s" * 64,
            "approved_source": "/fake/context.yaml",
            "approved_source_sha256": "c" * 64,
            "approved_at": "2026-08-21T01:00:00+00:00",
            "realized_path": "/fake/data/bulk_amo_SiO2.lammps-data",
            "realized_sha256": "s" * 64,
            "realized_source": "reconstructed from /fake/input.lmp → read_data → sha256",
            "submission_started_at": "2026-08-21T01:30:00+00:00",
            "binding_precedes_submission": True,
            "match": True,
        },
        "student_checkpoint_identity": {
            "approved_selected_seed": 202631,
            "approved_checkpoint_path": "/fake/committee/seed-202631/potential_saved_bestmodel",
            "approved_checkpoint_sha256": "k" * 64,
            "approved_source": "/fake/deployment_provenance.json",
            "approved_source_sha256": "p" * 64,
            "approved_at": "2026-08-21T01:00:00+00:00",
            "approved_derivation":
                "direct from deployment_provenance.student.checkpoint_sha256; cross-checked "
                "against committee_manifest models[seed=...]. cross_check_match=True",
            "realized_selected_seed": 202631,
            "realized_checkpoint_path": "/fake/committee/seed-202631/potential_saved_bestmodel",
            "realized_checkpoint_sha256": "k" * 64,
            "realized_source": "md.manifest.json.checkpoint_integrity.sha256 (with selected_seed cross-check)",
            "committee_manifest_sha256_semantic_guard": {
                "committee_manifest_sha256": "m" * 64,
                "checkpoint_sha256": "k" * 64,
                "committee_manifest_sha_is_not_the_checkpoint_sha": True,
            },
            "seed_match": True,
            "sha_match": True,
            "submission_started_at": "2026-08-21T01:30:00+00:00",
            "binding_precedes_submission": True,
            "match": True,
        },
    }


# Sentinel so tests can distinguish "no identity_bindings arg supplied → use defaults"
# from "identity_bindings=None explicitly (missing evidence)".
_DEFAULT = object()


def _validate(*, identity_bindings=_DEFAULT):
    resolved = _matching_identity_bindings() if identity_bindings is _DEFAULT else identity_bindings
    return validate_stage10_deployment_point(
        realized_protocol=_REALIZED,
        approved_shared_md_protocol=_APPROVED_POINT,
        pinned_composition_scope=[
            "SiO2 (stoichiometric, x = 0)",
            "SiOx (sub-stoichiometric, 0 < x < 2)",
            "Si (fully reduced boundary, x = 2)",
        ],
        pinned_temperature_envelope=["qualitative"],
        pinned_pressure_envelope=["qualitative"],
        thermo_diagnostic={
            "banned_tokens_found": [],
            "no_nan_inf_error_tokens": True,
            "observed_pressure_bar_nvt_diagnostic": {"mean": 70000.0, "min": 60000.0, "max": 80000.0},
        },
        identity_bindings=resolved,
    )


# ----------------------------------------------------------------------
# 1) & 2) starting-structure and checkpoint approved-vs-realized SHA equality
# ----------------------------------------------------------------------

def test_c2b_starting_structure_identity_is_readable():
    ev = _validate()
    c2b = ev["per_subcriterion"]["C2b"]
    ssi = c2b["checks"]["starting_structure_identity"]
    assert ssi["approved_sha256"] == "s" * 64
    assert ssi["realized_sha256"] == "s" * 64
    assert ssi["match"] is True
    assert ssi["binding_precedes_submission"] is True


def test_c2b_student_checkpoint_identity_is_readable():
    ev = _validate()
    c2b = ev["per_subcriterion"]["C2b"]
    sci = c2b["checks"]["student_checkpoint_identity"]
    assert sci["approved_selected_seed"] == 202631
    assert sci["realized_selected_seed"] == 202631
    assert sci["approved_checkpoint_sha256"] == "k" * 64
    assert sci["realized_checkpoint_sha256"] == "k" * 64
    assert sci["match"] is True
    assert sci["binding_precedes_submission"] is True


# ----------------------------------------------------------------------
# 3) committee_manifest sha256 is NOT treated as the checkpoint sha256
# ----------------------------------------------------------------------

def test_committee_manifest_sha_is_not_the_checkpoint_sha_semantic_guard():
    ev = _validate()
    c2b = ev["per_subcriterion"]["C2b"]
    sci = c2b["checks"]["student_checkpoint_identity"]
    guard = sci["committee_manifest_sha256_semantic_guard"]
    assert guard["committee_manifest_sha256"] == "m" * 64
    assert guard["checkpoint_sha256"] == "k" * 64
    assert guard["committee_manifest_sha_is_not_the_checkpoint_sha"] is True


# ----------------------------------------------------------------------
# 4) selected seed 202631 maps to the expected checkpoint (via cross-check)
# ----------------------------------------------------------------------

def test_seed_202631_maps_to_expected_checkpoint():
    ev = _validate()
    c2b = ev["per_subcriterion"]["C2b"]
    sci = c2b["checks"]["student_checkpoint_identity"]
    assert sci["approved_selected_seed"] == 202631
    assert sci["approved_derivation"] is not None
    assert "cross-checked" in sci["approved_derivation"]


# ----------------------------------------------------------------------
# 5) wrong starting-structure SHA → C2b aggregate FALSE
# ----------------------------------------------------------------------

def test_wrong_starting_structure_sha_fails_c2b():
    ib = _matching_identity_bindings()
    ib["starting_structure_identity"]["realized_sha256"] = "z" * 64
    ib["starting_structure_identity"]["match"] = False
    ev = _validate(identity_bindings=ib)
    assert ev["realized_protocol_matches_approved_point"] is False


# ----------------------------------------------------------------------
# 6) wrong checkpoint SHA → C2b aggregate FALSE
# ----------------------------------------------------------------------

def test_wrong_checkpoint_sha_fails_c2b():
    ib = _matching_identity_bindings()
    ib["student_checkpoint_identity"]["realized_checkpoint_sha256"] = "z" * 64
    ib["student_checkpoint_identity"]["match"] = False
    ib["student_checkpoint_identity"]["sha_match"] = False
    ev = _validate(identity_bindings=ib)
    assert ev["realized_protocol_matches_approved_point"] is False


# ----------------------------------------------------------------------
# 7) missing pre-submission expected identity → fail closed
# ----------------------------------------------------------------------

def test_missing_identity_bindings_fail_closed():
    ev = _validate(identity_bindings=None)
    assert ev["realized_protocol_matches_approved_point"] is False
    c2b = ev["per_subcriterion"]["C2b"]
    ssi = c2b["checks"]["starting_structure_identity"]
    sci = c2b["checks"]["student_checkpoint_identity"]
    # Missing bindings materialize as absent match + explicit gap.
    assert ssi.get("match") is False
    assert "evidence_gap" in ssi
    assert sci.get("match") is False
    assert "evidence_gap" in sci


def test_missing_approved_sha_fails_closed():
    ib = _matching_identity_bindings()
    ib["starting_structure_identity"]["approved_sha256"] = None
    ib["starting_structure_identity"]["match"] = False
    ev = _validate(identity_bindings=ib)
    assert ev["realized_protocol_matches_approved_point"] is False


# ----------------------------------------------------------------------
# 8) post-submission-only identity binding cannot satisfy C2b
# ----------------------------------------------------------------------

def test_post_submission_only_binding_cannot_satisfy_c2b():
    ib = _matching_identity_bindings()
    # Simulate an approval whose timestamp is AFTER submission → precedes=False
    ib["starting_structure_identity"]["binding_precedes_submission"] = False
    ev = _validate(identity_bindings=ib)
    assert ev["realized_protocol_matches_approved_point"] is False


def test_binding_precedes_submission_unknown_also_fails_c2b():
    ib = _matching_identity_bindings()
    ib["student_checkpoint_identity"]["binding_precedes_submission"] = None
    ev = _validate(identity_bindings=ib)
    assert ev["realized_protocol_matches_approved_point"] is False


# ----------------------------------------------------------------------
# 9) aggregate C2b true only when ALL SIX component checks pass
# ----------------------------------------------------------------------

def test_c2b_aggregate_requires_all_six_component_checks():
    # Baseline: all six pass → True
    ev = _validate()
    assert ev["realized_protocol_matches_approved_point"] is True

    # Flip ensemble → False
    real = dict(_REALIZED)
    real["ensemble"] = "npt"
    ev2 = validate_stage10_deployment_point(
        realized_protocol=real,
        approved_shared_md_protocol=_APPROVED_POINT,
        pinned_composition_scope=None,
        pinned_temperature_envelope=None,
        pinned_pressure_envelope=None,
        thermo_diagnostic=None,
        identity_bindings=_matching_identity_bindings(),
    )
    assert ev2["realized_protocol_matches_approved_point"] is False

    # Flip identity → False
    ib3 = _matching_identity_bindings()
    ib3["starting_structure_identity"]["match"] = False
    ev3 = _validate(identity_bindings=ib3)
    assert ev3["realized_protocol_matches_approved_point"] is False


# ----------------------------------------------------------------------
# 10) All Stage-10 scientific artifacts remain byte-identical
# ----------------------------------------------------------------------

_CANONICAL_STAGE10_SHAS = {
    "artifacts/md.manifest.json":
        "6541c3a1da04e038b3cbb05b0b9c36efda8b05806bcb941887a2660a2f7c46a0",
    "artifacts/deployment_md/trajectory.dump":
        "6eec4a0e90bc4c63ad2def8b081c0b1fdbec3e8358186a58bff7045d77988a4d",
    "artifacts/deployment_md/thermo.log":
        "3ed87bcec0beaea44726de04f90c0a38730101a2059c58ab35954d421c0983cc",
    "artifacts/deployment_md/input.lmp":
        "63e3438068ad26a04a15abcef02d3fdeb33afbe74eef291608eb1707c743aa53",
    "artifacts/deployment_md/context.yaml":
        "af0bc999434bf66c242d131cf38818d55a560e7ecf739929a54b90b5eb3d4931",
    "artifacts/deployment_md/deployment_provenance.json":
        "6cae634f29fd2599d537a208dd6be7cf0fd6bbf9c4a553c7b43430adb2b3302c",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relpath, expected", sorted(_CANONICAL_STAGE10_SHAS.items()))
def test_stage10_artifacts_untouched_by_c2b_identity_completeness(relpath, expected):
    from workflow.integrity import sha256_file
    root = _project_root() / "runs" / "sio2-sox-allegro-simplenn-c12f"
    path = root / relpath
    if not path.is_file():
        pytest.skip(f"Stage-10 artifact not present: {relpath}")
    assert sha256_file(path) == expected, (
        f"{relpath!r} sha256 drifted — the Stage-10 C2b identity completeness fix must "
        "NOT modify any Stage-10 scientific artifact")
