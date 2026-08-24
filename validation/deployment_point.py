"""Deterministic deployment-point validator (Recovery id=6, CASE-B versioned Stage-10
C2 refinement).

Framework-generic — does NOT hardcode any material, campaign, temperature, or pressure
bound. Consumes:

  * the pinned pre-submission deployment-point authorization
    (``inputs/008-validation_profile.yaml::shared_md_protocol``);
  * the realized executable protocol
    (``artifacts/deployment_md/context.yaml`` + LAMMPS ``input.lmp`` ensemble detection);
  * the pinned pre-declared composition scope
    (``inputs/009-distillation_scope.yaml::deployment_domain.composition_scope``);
  * the pinned pre-declared global T/P envelope shape (qualitative vs numerical);
  * a bounded, deterministic pass over the already-written ``thermo.log`` (finite/
    non-finite check + observed pressure diagnostic).

Emits STRUCTURED FACTS separately for each Stage-10 C2 sub-criterion the CASE-B
refinement introduces (composition scope, realized-protocol match against the pre-
submission point, ensemble-aware pressure semantics). Never collapses them into a
single "global deployment domain PASS" boolean, and never invents numerical T/P
bounds beyond what a pinned pre-submission artifact already supplies.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


# The versioned CASE-B refinement is a fixed identifier attached to every result so
# a downstream Judge / Controller can distinguish this validator's output from any
# prior gate binding. Update if the sub-criterion semantics are ever revised.
STAGE10_C2_REFINEMENT_VERSION = "stage10_c2_case_b_v1"


# The ORIGINAL C2 text — preserved BYTE-FOR-BYTE for the immutable audit trail.
# This module MUST NOT overwrite/mutate the pinned run's `gate.criteria` list; the
# text lives here only as a comparison / audit reference so downstream layers can
# clearly show "here is what the original criterion said and why it isn't directly
# evaluable against a qualitative global T/P contract".
ORIGINAL_C2_TEXT = (
    "MD run stayed inside the frozen deployment domain (composition, T, P)"
)


def _get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def _is_qualitative_envelope(value: Any) -> bool:
    """A pinned envelope is 'qualitative' iff every entry is a free-text string with
    no numerical bound. A list of numbers or a {min, max} dict is 'numerical' and
    would make quantitative_pressure_domain_evaluable=True. Fail closed to qualitative
    if the envelope shape is unrecognised."""
    if value is None:
        return True
    if isinstance(value, list):
        return all(isinstance(v, str) for v in value)
    if isinstance(value, dict):
        return not any(isinstance(v, (int, float)) for v in value.values())
    return True


def _numeric_bounds_from_envelope(value: Any):
    """Extract ({min:.., max:..}) numerical bounds from a pinned envelope value.

    Only recognises explicit shapes: a dict with numeric ``min`` / ``max`` keys, or a
    list of exactly two numerics. Anything else (including a list of strings, a single
    number, or a mixed list) returns None. Never guesses from prose.
    """
    if isinstance(value, dict):
        lo, hi = value.get("min"), value.get("max")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return {"min": lo, "max": hi}
    if isinstance(value, list) and len(value) == 2 \
            and all(isinstance(v, (int, float)) for v in value):
        return {"min": min(value), "max": max(value)}
    return None


def _ensemble_from_input_lmp_text(text: str) -> Optional[str]:
    """Detect the active LAMMPS integrator (nvt/npt/nve) by parsing the last non-
    comment ``fix`` line whose style is a canonical ensemble token. Deterministic;
    never runs LAMMPS."""
    found: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 4 and tokens[0] == "fix":
            style = tokens[3].lower()
            if style in ("nvt", "npt", "nph", "nve", "nhc"):
                found.append(style)
    return found[-1] if found else None


def _protocol_matches(
    realized: dict,
    approved: dict,
    identity_bindings: Optional[dict] = None,
) -> dict:
    """Deterministic per-field match between a realized executable protocol and the
    pinned pre-submission binding. Six required per-field comparisons:
    ensemble, temperature_setpoint_K, timestep_ps, total_simulated_time_ps,
    starting_structure_identity, student_checkpoint_identity.

    ``identity_bindings`` (optional but required for C2b to be fully evaluable) is a
    dict of pre-computed approved-vs-realized identity comparisons produced by
    ``_build_identity_bindings`` in the calling site (bounded-evidence adapter),
    each with fields matching the operator's spec:

        starting_structure_identity: {approved_path, approved_sha256, approved_source,
            approved_source_sha256, approved_at, realized_path, realized_sha256,
            realized_source, match, binding_precedes_submission, ...}
        student_checkpoint_identity: {approved_selected_seed, approved_checkpoint_path,
            approved_checkpoint_sha256, approved_source, approved_source_sha256,
            approved_at, realized_selected_seed, realized_checkpoint_path,
            realized_checkpoint_sha256, realized_source, match,
            binding_precedes_submission, ...}

    If ``identity_bindings`` is absent OR either identity check is missing, the
    aggregate ``all_match`` MUST NOT be True (fail closed).
    """
    checks: dict = {}

    r_T = _get(realized, "temperature_setpoint_K")
    a_T = _get(approved, "temperature_K")
    checks["temperature_setpoint_K"] = {
        "realized": r_T, "approved": a_T,
        "match": (isinstance(r_T, (int, float)) and isinstance(a_T, (int, float))
                  and abs(float(r_T) - float(a_T)) < 1e-9),
    }

    r_ts_ps = _get(realized, "timestep_ps")
    a_ts_fs = _get(approved, "timestep_fs")
    a_ts_ps = float(a_ts_fs) * 1e-3 if isinstance(a_ts_fs, (int, float)) else _get(approved, "timestep_ps")
    checks["timestep_ps"] = {
        "realized_ps": r_ts_ps, "approved_ps": a_ts_ps,
        "match": (isinstance(r_ts_ps, (int, float)) and isinstance(a_ts_ps, (int, float))
                  and abs(float(r_ts_ps) - float(a_ts_ps)) < 1e-12),
    }

    r_total_ps = _get(realized, "total_simulated_time_ps")
    a_eq = _get(approved, "nvt_equilibration_ps")
    a_prod = _get(approved, "nvt_production_ps")
    a_total_ps = None
    if isinstance(a_eq, (int, float)) and isinstance(a_prod, (int, float)):
        a_total_ps = float(a_eq) + float(a_prod)
    elif isinstance(_get(approved, "simulated_time_ps"), (int, float)):
        a_total_ps = float(approved["simulated_time_ps"])
    checks["total_simulated_time_ps"] = {
        "realized_ps": r_total_ps, "approved_ps": a_total_ps,
        "match": (isinstance(r_total_ps, (int, float)) and isinstance(a_total_ps, (int, float))
                  and abs(float(r_total_ps) - float(a_total_ps)) < 1e-9),
    }

    r_ens = _get(realized, "ensemble")
    a_ens = _get(approved, "ensemble") or (
        # If the approved binding does not name an ensemble but supplies
        # nvt_equilibration_ps / nvt_production_ps, the pinned intent is NVT.
        "nvt" if (a_eq is not None or a_prod is not None) else None
    )
    checks["ensemble"] = {"realized": r_ens, "approved": a_ens,
                          "match": (r_ens is not None and a_ens is not None and r_ens == a_ens)}

    # Two REQUIRED identity checks (C2b text explicitly enumerates them). If the
    # caller did not supply them, aggregate cannot be True — record as absent.
    id_bindings = identity_bindings if isinstance(identity_bindings, dict) else {}
    ssi = id_bindings.get("starting_structure_identity") or {
        "match": False,
        "evidence_gap": "starting_structure_identity binding was not supplied to the C2b "
                        "validator",
    }
    sci = id_bindings.get("student_checkpoint_identity") or {
        "match": False,
        "evidence_gap": "student_checkpoint_identity binding was not supplied to the C2b "
                        "validator",
    }
    checks["starting_structure_identity"] = ssi
    checks["student_checkpoint_identity"] = sci

    # Aggregate: TRUE iff EVERY per-field check has match=True. Identity checks
    # additionally require binding_precedes_submission=True to prevent
    # post-submission-only bindings from satisfying C2b.
    def _field_ok(k: str, v: dict) -> bool:
        if not isinstance(v, dict) or v.get("match") is not True:
            return False
        if k in ("starting_structure_identity", "student_checkpoint_identity"):
            if v.get("binding_precedes_submission") is not True:
                return False
        return True

    all_match = all(_field_ok(k, v) for k, v in checks.items())
    return {"per_field": checks, "all_match": all_match}


def _observed_pressure_diagnostic(realized_thermo: Optional[dict]) -> dict:
    """Return the observed-pressure diagnostic from an already-computed thermo pass.

    ``realized_thermo`` is the ``thermo_diagnostic`` block already produced by the
    bounded-evidence adapter (single deterministic pass over ``thermo.log``); this
    function only extracts and labels it. Never re-reads the trajectory.
    """
    if not isinstance(realized_thermo, dict):
        return {"observed_pressure_bar": None, "observed_pressure_finite": None,
                "note": "thermo diagnostic not available"}
    banned = realized_thermo.get("banned_tokens_found") or []
    finite = realized_thermo.get("no_nan_inf_error_tokens", None) and not banned
    press = realized_thermo.get("observed_pressure_bar_nvt_diagnostic") or {}
    return {
        "observed_pressure_bar_mean": press.get("mean"),
        "observed_pressure_bar_min": press.get("min"),
        "observed_pressure_bar_max": press.get("max"),
        "observed_pressure_finite": bool(finite),
        "banned_tokens_found": list(banned),
        "note": (
            "observed pressure is an NVT (fixed-volume) diagnostic time-series only. "
            "It is NOT a controlled setpoint and NOT evidence of a numerical pressure-"
            "domain PASS/FAIL"
        ),
    }


def _species_counts_from_datafile(datafile_text: str) -> Optional[dict]:
    """Count atoms per LAMMPS type by parsing the ``Atoms`` section of a data file.
    Deterministic; never opens the trajectory; never uses chemistry-specific tokens.
    Returns None if the section is not present.
    """
    counts: dict = {}
    in_atoms = False
    for line in datafile_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Atoms"):
            in_atoms = True
            continue
        if in_atoms:
            if not stripped or stripped.startswith("#") or stripped.split()[0].isalpha():
                if counts:
                    break
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                key = f"type_{parts[1]}"
                counts[key] = counts.get(key, 0) + 1
    return counts or None


def validate_stage10_deployment_point(
    *,
    realized_protocol: dict,
    approved_shared_md_protocol: dict,
    pinned_composition_scope: Any,
    pinned_temperature_envelope: Any,
    pinned_pressure_envelope: Any,
    thermo_diagnostic: Optional[dict],
    analysis_window_history: Optional[list] = None,
    identity_bindings: Optional[dict] = None,
) -> dict:
    """Deterministic per-subcriterion evaluation of the CASE-B versioned Stage-10 C2.

    Returns a structured facts dict — never a single "global deployment domain PASS"
    boolean. Fields:

      composition_scope_valid
      realized_protocol_matches_approved_point
      pressure_control_mode
      quantitative_pressure_domain_evaluable
      observed_pressure_finite
      global_domain_claim_supported
      per_subcriterion  (nested detail for C2a, C2b, C2c)
    """
    # C2a — composition
    realized_composition = _get(realized_protocol, "realized_composition")
    scope_list = pinned_composition_scope if isinstance(pinned_composition_scope, list) else None
    composition_in_scope = None
    if realized_composition and scope_list:
        composition_in_scope = realized_composition in scope_list
    c2a = {
        "criterion":
            "The realized MD composition lies within the pre-declared composition scope.",
        "realized_composition": realized_composition,
        "pinned_composition_scope": scope_list,
        "composition_scope_valid": composition_in_scope,
        "provenance_source_pinned": "inputs/009-distillation_scope.yaml::deployment_domain.composition_scope",
    }

    # C2b — realized executable protocol matches pre-approved deployment point
    match = _protocol_matches(realized_protocol, approved_shared_md_protocol,
                              identity_bindings=identity_bindings)
    c2b = {
        "criterion":
            "The realized executable MD protocol matches the exact deployment point "
            "authorized before execution.",
        "approved_source_pinned":
            "inputs/008-validation_profile.yaml::shared_md_protocol",
        "checks": match["per_field"],
        "realized_protocol_matches_approved_point": match["all_match"],
        "analysis_window_history": analysis_window_history or [],
        "analysis_window_history_note": (
            "Analysis / assessment-window semantics are a separate governance record "
            "from the executable MD protocol. The executable protocol was NVT for the "
            "full 70 ps; the pre-init validation profile pinned a 20 ps equilibration "
            "/ 50 ps production analysis partition, while the later Option-2 human-"
            "approved Stage-10 assessment used a 10 ps equilibration / 60 ps "
            "assessment window for trajectory diagnostics. Both predate MD submission "
            "and both are preserved. They are DIFFERENT partitions of the same 70-ps "
            "trajectory, NOT an exact match; do not claim they coincide."
        ),
    }

    # C2c — ensemble-aware pressure semantics
    ensemble = _get(realized_protocol, "ensemble")
    if ensemble == "nvt":
        pressure_control_mode = "uncontrolled_observed_diagnostic_nvt"
    elif ensemble == "npt":
        pressure_control_mode = "controlled_setpoint_npt"
    elif ensemble == "nve":
        pressure_control_mode = "uncontrolled_observed_diagnostic_nve"
    elif ensemble is None:
        pressure_control_mode = "ensemble_undetermined"
    else:
        pressure_control_mode = f"other:{ensemble}"

    numerical_bounds = _numeric_bounds_from_envelope(pinned_pressure_envelope)
    quantitative_pressure_domain_evaluable = numerical_bounds is not None
    pressure_diag = _observed_pressure_diagnostic(thermo_diagnostic)
    c2c = {
        "criterion":
            "For fixed-volume NVT deployment, pressure is not a controlled setpoint. "
            "Observed virial pressure is reported as a diagnostic. Quantitative "
            "pressure-domain membership is claimed only if a pre-existing pinned "
            "numerical pressure-domain contract exists.",
        "pressure_control_mode": pressure_control_mode,
        "quantitative_pressure_domain_evaluable": bool(quantitative_pressure_domain_evaluable),
        "numerical_pressure_bounds_pinned": numerical_bounds,
        "pinned_pressure_envelope_verbatim": pinned_pressure_envelope,
        "observed_pressure_diagnostic": pressure_diag,
    }

    # Global-domain claim is supported only when EVERY dimension can be evaluated
    # numerically. In C12F: composition is enumerable (scope list), T is qualitative
    # (list of strings), P is qualitative — so global_domain_claim_supported=False.
    global_supported = (
        (c2a["composition_scope_valid"] is True)
        and (_numeric_bounds_from_envelope(pinned_temperature_envelope) is not None)
        and (quantitative_pressure_domain_evaluable is True)
    )

    return {
        "refinement_version": STAGE10_C2_REFINEMENT_VERSION,
        "original_c2_text_immutable_audit": ORIGINAL_C2_TEXT,
        "original_c2_directly_evaluable_against_qualitative_contract": False,
        "original_c2_directly_evaluable_reason": (
            "The pinned global T/P envelope is qualitative (free-text 'ambient "
            "through melt' / 'ambient through high-pressure'). No pre-existing "
            "numerical bounds exist, so the original C2 as written cannot be "
            "adjudicated as a numerical PASS or FAIL. This is a criterion-authoring "
            "defect (CASE B). The original text is preserved verbatim above; the "
            "versioned subcriteria below evaluate what the pinned contract can "
            "support without introducing new bounds (post-hoc criterion fitting)."
        ),
        "composition_scope_valid": c2a["composition_scope_valid"],
        "realized_protocol_matches_approved_point":
            c2b["realized_protocol_matches_approved_point"],
        "pressure_control_mode": pressure_control_mode,
        "quantitative_pressure_domain_evaluable":
            bool(quantitative_pressure_domain_evaluable),
        "observed_pressure_finite": pressure_diag.get("observed_pressure_finite"),
        "global_domain_claim_supported": bool(global_supported),
        "per_subcriterion": {"C2a": c2a, "C2b": c2b, "C2c": c2c},
        "not_claimed_by_this_evaluation": [
            "any numerical global temperature-domain PASS",
            "any numerical global pressure-domain PASS",
            "calibrated uncertainty",
            "physical realism of the resulting trajectory (see Stage 11)",
            "validation of the full SiO2-x deployment envelope",
        ],
    }


__all__ = [
    "STAGE10_C2_REFINEMENT_VERSION",
    "ORIGINAL_C2_TEXT",
    "validate_stage10_deployment_point",
]
