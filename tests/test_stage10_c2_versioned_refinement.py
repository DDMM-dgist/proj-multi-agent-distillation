"""Regression tests for the CASE-B versioned Stage-10 C2 refinement (Recovery id=6).

Bound by the authorized governance-only zero-compute recovery (session 2026-08-21).
Proves that:
  * the original C2 text is preserved BYTE-FOR-BYTE in the audit trail;
  * the refined C2a/C2b/C2c evaluate deterministically from pinned pre-init sources;
  * composition is derived from authoritative species counts in the LAMMPS data file;
  * the realized executable protocol is compared to the PRE-SUBMISSION pinned binding;
  * the 20/50 vs 10/60 analysis-window history is preserved as DISTINCT records and
    is not falsely claimed to match;
  * NVT has NO controlled pressure setpoint;
  * a qualitative pressure envelope yields
    ``quantitative_pressure_domain_evaluable=False``;
  * a numerical pressure envelope yields
    ``quantitative_pressure_domain_evaluable=True``;
  * observed pressure remains diagnostic-only;
  * no Stage-10 scientific artifact is modified;
  * a missing pre-submission authorization fails closed (per-field ``match=False``).
"""
from __future__ import annotations

import copy
import json
import hashlib
from pathlib import Path

import pytest

from validation.deployment_point import (
    ORIGINAL_C2_TEXT,
    STAGE10_C2_REFINEMENT_VERSION,
    validate_stage10_deployment_point,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_APPROVED_POINT = {
    "temperature_K": 300.0,
    "pressure_GPa": 0.0,
    "timestep_fs": 0.5,
    "nvt_equilibration_ps": 20.0,
    "nvt_production_ps": 50.0,
    "sampling_interval_fs": 10.0,
}


_REALIZED_MATCHING = {
    "ensemble": "nvt",
    "temperature_setpoint_K": 300.0,
    "timestep_ps": 0.0005,
    "total_simulated_time_ps": 70.0,
    "realized_composition": "SiO2 (stoichiometric, x = 0)",
    "starting_structure_species_counts_by_lammps_type": {"type_1": 128, "type_2": 64},
}


_QUALITATIVE_TP = {
    "temperature_K": ["full source-pool envelope (ambient through melt)"],
    "pressure_GPa": ["full source-pool envelope (ambient through high-pressure)"],
}


_NUMERIC_PRESSURE = {"min": 0.0, "max": 100.0}


_THERMO_FINITE = {
    "banned_tokens_found": [],
    "no_nan_inf_error_tokens": True,
    "observed_pressure_bar_nvt_diagnostic": {"mean": 70620.0, "min": 56000.0, "max": 83558.0},
}


_COMPOSITION_SCOPE = [
    "SiO2 (stoichiometric, x = 0)",
    "SiOx (sub-stoichiometric, 0 < x < 2)",
    "Si (fully reduced boundary, x = 2)",
]


_MATCHING_IDENTITY_BINDINGS = {
    "starting_structure_identity": {
        "approved_sha256": "s" * 64, "realized_sha256": "s" * 64,
        "match": True, "binding_precedes_submission": True,
    },
    "student_checkpoint_identity": {
        "approved_selected_seed": 202631, "realized_selected_seed": 202631,
        "approved_checkpoint_sha256": "k" * 64, "realized_checkpoint_sha256": "k" * 64,
        "seed_match": True, "sha_match": True,
        "match": True, "binding_precedes_submission": True,
    },
}


def _validate(**overrides):
    """Convenience wrapper — supply keyword overrides. Provides matching identity
    bindings by default so tests written before the C2b identity-completeness fix
    keep passing; individual tests can override ``identity_bindings`` when they need
    to model missing / mismatched identities."""
    kwargs = {
        "realized_protocol": _REALIZED_MATCHING,
        "approved_shared_md_protocol": _APPROVED_POINT,
        "pinned_composition_scope": _COMPOSITION_SCOPE,
        "pinned_temperature_envelope": _QUALITATIVE_TP["temperature_K"],
        "pinned_pressure_envelope": _QUALITATIVE_TP["pressure_GPa"],
        "thermo_diagnostic": _THERMO_FINITE,
        "analysis_window_history": [
            {"kind": "pre_init_validation_profile_pinned",
             "equilibration_ps": 20.0, "production_ps": 50.0},
            {"kind": "option_2_human_accepted_assessment_semantics",
             "equilibration_ps": 10.0, "assessment_ps": 60.0},
        ],
        "identity_bindings": dict(_MATCHING_IDENTITY_BINDINGS),
    }
    kwargs.update(overrides)
    return validate_stage10_deployment_point(**kwargs)


# ------------------------------------------------------------------
# Original C2 preserved verbatim in the audit trail
# ------------------------------------------------------------------

def test_original_c2_preserved_byte_for_byte_and_not_marked_pass():
    ev = _validate()
    assert (ev["original_c2_text_immutable_audit"] == ORIGINAL_C2_TEXT ==
            "MD run stayed inside the frozen deployment domain (composition, T, P)")
    # Original criterion cannot be marked directly evaluable against the qualitative contract:
    assert ev["original_c2_directly_evaluable_against_qualitative_contract"] is False
    # Refinement version identifier is present so downstream layers can distinguish binding.
    assert ev["refinement_version"] == STAGE10_C2_REFINEMENT_VERSION == "stage10_c2_case_b_v1"


# ------------------------------------------------------------------
# C2a — composition scope derived deterministically from authoritative sources
# ------------------------------------------------------------------

def test_c2a_composition_scope_valid_for_declared_composition():
    ev = _validate()
    c2a = ev["per_subcriterion"]["C2a"]
    assert c2a["realized_composition"] == "SiO2 (stoichiometric, x = 0)"
    assert c2a["pinned_composition_scope"] == _COMPOSITION_SCOPE
    assert c2a["composition_scope_valid"] is True
    assert ev["composition_scope_valid"] is True


def test_c2a_composition_scope_reports_none_when_realized_composition_unknown():
    ev = _validate(realized_protocol={**_REALIZED_MATCHING, "realized_composition": None})
    c2a = ev["per_subcriterion"]["C2a"]
    assert c2a["composition_scope_valid"] is None


def test_c2a_composition_scope_fails_when_not_in_scope():
    ev = _validate(realized_protocol={**_REALIZED_MATCHING,
                                       "realized_composition": "H2O bulk liquid"})
    c2a = ev["per_subcriterion"]["C2a"]
    assert c2a["composition_scope_valid"] is False


# ------------------------------------------------------------------
# C2b — realized protocol matched to PRE-SUBMISSION binding
# ------------------------------------------------------------------

def test_c2b_realized_protocol_matches_pre_submission_binding():
    ev = _validate()
    c2b = ev["per_subcriterion"]["C2b"]
    assert c2b["realized_protocol_matches_approved_point"] is True
    for f in ("temperature_setpoint_K", "timestep_ps", "total_simulated_time_ps", "ensemble"):
        assert c2b["checks"][f]["match"] is True, f"C2b failed on {f}: {c2b['checks'][f]}"


def test_c2b_fails_on_temperature_mismatch():
    ev = _validate(realized_protocol={**_REALIZED_MATCHING, "temperature_setpoint_K": 400.0})
    assert ev["realized_protocol_matches_approved_point"] is False
    assert ev["per_subcriterion"]["C2b"]["checks"]["temperature_setpoint_K"]["match"] is False


def test_c2b_fails_on_ensemble_mismatch():
    ev = _validate(realized_protocol={**_REALIZED_MATCHING, "ensemble": "npt"})
    assert ev["realized_protocol_matches_approved_point"] is False
    assert ev["per_subcriterion"]["C2b"]["checks"]["ensemble"]["match"] is False


def test_c2b_analysis_window_history_preserves_both_partitions_without_equating_them():
    ev = _validate()
    c2b = ev["per_subcriterion"]["C2b"]
    history = c2b["analysis_window_history"]
    assert len(history) == 2, history
    kinds = {h.get("kind") for h in history}
    assert "pre_init_validation_profile_pinned" in kinds
    assert "option_2_human_accepted_assessment_semantics" in kinds
    # The two partitions must not be equated:
    pre = next(h for h in history if h["kind"] == "pre_init_validation_profile_pinned")
    later = next(h for h in history if h["kind"] == "option_2_human_accepted_assessment_semantics")
    assert (pre["equilibration_ps"], pre["production_ps"]) != \
           (later["equilibration_ps"], later["assessment_ps"])
    # And the note must warn against claiming they coincide.
    note = c2b["analysis_window_history_note"]
    assert "NOT an exact match" in note or "do not claim they coincide" in note


def test_c2b_missing_pre_submission_authorization_fails_closed_per_field():
    # An empty approved binding AND empty identity bindings must fail closed on
    # every C2b component check (protocol equality + both identity checks).
    ev = _validate(approved_shared_md_protocol={}, identity_bindings={})
    c2b = ev["per_subcriterion"]["C2b"]
    assert c2b["realized_protocol_matches_approved_point"] is False
    assert all(v["match"] is False for v in c2b["checks"].values())


# ------------------------------------------------------------------
# C2c — ensemble-aware pressure semantics
# ------------------------------------------------------------------

def test_c2c_nvt_has_no_controlled_pressure_setpoint():
    ev = _validate()
    c2c = ev["per_subcriterion"]["C2c"]
    assert c2c["pressure_control_mode"] == "uncontrolled_observed_diagnostic_nvt"


def test_c2c_qualitative_pressure_envelope_returns_not_evaluable():
    ev = _validate()
    c2c = ev["per_subcriterion"]["C2c"]
    assert c2c["quantitative_pressure_domain_evaluable"] is False
    assert c2c["numerical_pressure_bounds_pinned"] is None


def test_c2c_numerical_pressure_envelope_flips_evaluable_true():
    ev = _validate(pinned_pressure_envelope=_NUMERIC_PRESSURE)
    c2c = ev["per_subcriterion"]["C2c"]
    assert c2c["quantitative_pressure_domain_evaluable"] is True
    assert c2c["numerical_pressure_bounds_pinned"] == {"min": 0.0, "max": 100.0}


def test_c2c_observed_pressure_diagnostic_is_labeled():
    ev = _validate()
    diag = ev["per_subcriterion"]["C2c"]["observed_pressure_diagnostic"]
    assert diag["observed_pressure_finite"] is True
    assert diag["observed_pressure_bar_mean"] == 70620.0
    assert "NOT a controlled setpoint" in diag["note"] or "diagnostic time-series" in diag["note"]


def test_c2c_npt_ensemble_reports_controlled_setpoint():
    ev = _validate(realized_protocol={**_REALIZED_MATCHING, "ensemble": "npt"})
    c2c = ev["per_subcriterion"]["C2c"]
    assert c2c["pressure_control_mode"] == "controlled_setpoint_npt"


# ------------------------------------------------------------------
# Global-domain claim must NOT collapse into a single PASS/FAIL
# ------------------------------------------------------------------

def test_global_domain_claim_supported_is_false_under_qualitative_contract():
    ev = _validate()
    # Even though composition PASSES and C2b PASSES, the global claim MUST be false
    # because T and P envelopes are qualitative.
    assert ev["global_domain_claim_supported"] is False
    # And the composition + protocol checks remain independently visible.
    assert ev["composition_scope_valid"] is True
    assert ev["realized_protocol_matches_approved_point"] is True


def test_global_domain_claim_becomes_evaluable_only_when_both_T_and_P_numerical():
    ev = _validate(
        pinned_temperature_envelope={"min": 100.0, "max": 3500.0},
        pinned_pressure_envelope={"min": 0.0, "max": 100.0},
    )
    # The validator now has numerical envelopes for BOTH T and P; global_domain_claim
    # remains False here because the validator does not itself perform T/P membership
    # against the observed diagnostics (that's a downstream policy). It correctly
    # reports the DIMENSIONS ARE NOW EVALUABLE without inventing bounds.
    assert ev["quantitative_pressure_domain_evaluable"] is True


# ------------------------------------------------------------------
# Not-claimed guarantees
# ------------------------------------------------------------------

def test_not_claimed_list_includes_the_expected_disclaimers():
    ev = _validate()
    expected_disclaimers = {
        "any numerical global temperature-domain PASS",
        "any numerical global pressure-domain PASS",
        "calibrated uncertainty",
        "physical realism of the resulting trajectory (see Stage 11)",
        "validation of the full SiO2-x deployment envelope",
    }
    assert expected_disclaimers.issubset(set(ev["not_claimed_by_this_evaluation"]))


# ------------------------------------------------------------------
# Live-artifact untouched fixture
# ------------------------------------------------------------------

_CANONICAL_STAGE10_SHAS = {
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/md.manifest.json":
        "6541c3a1da04e038b3cbb05b0b9c36efda8b05806bcb941887a2660a2f7c46a0",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/trajectory.dump":
        "6eec4a0e90bc4c63ad2def8b081c0b1fdbec3e8358186a58bff7045d77988a4d",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/thermo.log":
        "3ed87bcec0beaea44726de04f90c0a38730101a2059c58ab35954d421c0983cc",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/input.lmp":
        "63e3438068ad26a04a15abcef02d3fdeb33afbe74eef291608eb1707c743aa53",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/context.yaml":
        "af0bc999434bf66c242d131cf38818d55a560e7ecf739929a54b90b5eb3d4931",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/deployment_provenance.json":
        "6cae634f29fd2599d537a208dd6be7cf0fd6bbf9c4a553c7b43430adb2b3302c",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relpath, expected", sorted(_CANONICAL_STAGE10_SHAS.items()))
def test_stage10_artifacts_untouched_by_case_b_refinement(relpath, expected):
    from workflow.integrity import sha256_file
    path = _project_root() / relpath
    if not path.is_file():
        pytest.skip(f"Stage-10 artifact not present in this checkout: {relpath}")
    assert sha256_file(path) == expected, (
        f"{relpath!r} sha256 drifted — the CASE-B versioned Stage-10 C2 refinement "
        "must NOT modify any Stage-10 scientific artifact")
