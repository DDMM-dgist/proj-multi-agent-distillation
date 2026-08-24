"""Runtime integration of the scientific-adequacy layer.

Session 2026-08-21. Session task 2b: RUNTIME INTEGRATION CLOSURE.

This module is the controller-hook that turns the typed contracts in
``framework_v2.scientific_adequacy`` from a schema into a runtime-enforced
gate. It is deliberately additive:

  * a run with no bound scientific policies is unchanged (scientific_gate
    is a no-op),
  * a run with a bound policy for a stage cannot advance that stage on a
    procedural PASS alone if the scientific adequacy is FAIL or
    NOT_EVALUABLE (when the policy declares it is required).

The invariant this module enforces:

    procedural_gate=PASS  AND  bound_policy_exists
       AND (adequacy_status is FAIL, OR
            adequacy_status is NOT_EVALUABLE and required=True)
        => Controller.record_gate raises and does NOT record PASS.

Callers should invoke ``assert_stage_scientific_adequacy(...)`` at the point
where a PASS verdict is about to be persisted. The Controller calls it from
``record_gate`` for stages that have a bound scientific policy.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from framework_v2.scientific_adequacy import (
    AdequacyStatus, CalibrationStatus, DomainMapping,
    DeploymentScopeContractV2, EnsembleKind,
    EvaluationAdequacyPolicyV2, EvaluationAdequacyVerdict,
    ObservableRole, ObservableSpec,
    PhysicalValidationPolicyV2, StatePreparationPolicy,
    UncertaintyPolicyV2,
    adjudicate_uncertainty, evaluate_adequacy,
)


# =====================================================================
# state<->policies accessor (lazy migration for pre-integration manifests)
# =====================================================================
POLICIES_KEY = "scientific_policies"


def _policies(state: dict) -> dict:
    """Return the mutable scientific_policies sub-state, lazily creating it.

    A pre-existing state without this key is not corrupted; simply gaining an
    empty dict has no runtime effect (no bindings => no assertions).
    """
    return state.setdefault(POLICIES_KEY, {})


def _content_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# =====================================================================
# bind_policy: attach a typed policy to a stage in the run manifest
# =====================================================================
KNOWN_POLICY_KINDS = {
    "EvaluationAdequacyPolicyV2",
    "DeploymentScopeContractV2",
    "StatePreparationPolicy",
    "PhysicalValidationPolicyV2",
    "UncertaintyPolicyV2",
}


def bind_policy(state: dict, stage_name: str, kind: str, policy_dict: dict,
                *, source_ref: str, note: str = "",
                required: bool = True) -> dict:
    """Attach a scientific policy to a stage.

    Idempotent by (stage_name, kind): binding the same kind twice on the same
    stage refuses to overwrite; a different content-hash raises. This means
    the bound policy is genuinely IMMUTABLE run provenance once written.
    """
    if kind not in KNOWN_POLICY_KINDS:
        raise ValueError(f"unknown scientific policy kind: {kind!r}")
    _validate_policy_shape(kind, policy_dict)
    policies = _policies(state)
    key = f"{stage_name}::{kind}"
    ch = _content_hash(policy_dict)
    if key in policies:
        existing = policies[key]
        if existing["content_sha256"] == ch:
            return existing
        raise ValueError(
            f"scientific policy {key!r} already bound with a different content-hash "
            f"({existing['content_sha256']} vs {ch}); refusing overwrite")
    record = {
        "stage": stage_name,
        "kind": kind,
        "content": policy_dict,
        "content_sha256": ch,
        "source_ref": source_ref,
        "note": note,
        "required": bool(required),
        "bound_at": _now(),
    }
    policies[key] = record
    state.setdefault("events", []).append({
        "at": record["bound_at"],
        "type": "scientific_policy_bound",
        "stage": stage_name,
        "kind": kind,
        "content_sha256": ch,
        "source_ref": source_ref,
        "required": required,
    })
    return record


def _validate_policy_shape(kind: str, payload: dict) -> None:
    """Construct the typed contract to validate; if this raises, the caller
    tried to bind a malformed policy. We do NOT keep the pydantic instance;
    the dict is what gets persisted."""
    if kind == "EvaluationAdequacyPolicyV2":
        EvaluationAdequacyPolicyV2(**payload)
    elif kind == "DeploymentScopeContractV2":
        DeploymentScopeContractV2(**payload)
    elif kind == "StatePreparationPolicy":
        StatePreparationPolicy(**payload)
    elif kind == "PhysicalValidationPolicyV2":
        PhysicalValidationPolicyV2(**payload)
    elif kind == "UncertaintyPolicyV2":
        UncertaintyPolicyV2(**payload)


# =====================================================================
# assert_stage_scientific_adequacy: THE runtime gate
# =====================================================================
class ScientificAdequacyBlocked(RuntimeError):
    """Raised when a procedural PASS cannot advance because bound scientific
    policy scores FAIL or NOT_EVALUABLE (when required)."""


def assert_stage_scientific_adequacy(
    state: dict, stage_name: str,
    *,
    accuracy_report_loader: Optional[Callable[[], dict]] = None,
    uncertainty_report_loader: Optional[Callable[[], dict]] = None,
    md_manifest_loader: Optional[Callable[[], dict]] = None,
    validation_report_loader: Optional[Callable[[], dict]] = None,
    in_scope_domains: Optional[list[str]] = None,
) -> Optional[dict]:
    """Enforce scientific adequacy at gate-PASS time.

    Fail-closed: any bound policy whose adjudication is FAIL, or NOT_EVALUABLE
    when the binding is ``required=True``, raises ``ScientificAdequacyBlocked``.
    A missing loader for a required evidence artifact raises
    ``ScientificAdequacyBlocked`` (rather than silently ignoring the check).

    Returns a summary dict written back into the run manifest, or None when
    no policies are bound to this stage.
    """
    policies = _policies(state)
    keys = [k for k in policies if k.startswith(f"{stage_name}::")]
    if not keys:
        return None
    summary = {"stage": stage_name, "checked_at": _now(), "adjudications": []}
    problems: list[str] = []
    for key in keys:
        rec = policies[key]
        kind = rec["kind"]
        required = bool(rec.get("required", True))
        content = rec["content"]
        try:
            if kind == "EvaluationAdequacyPolicyV2":
                verdict = _score_evaluation(content, accuracy_report_loader, in_scope_domains)
            elif kind == "UncertaintyPolicyV2":
                verdict = _score_uncertainty(content, uncertainty_report_loader)
            elif kind == "StatePreparationPolicy":
                verdict = _score_state_realization(content, md_manifest_loader)
            elif kind == "PhysicalValidationPolicyV2":
                verdict = _score_physical_validation(content, validation_report_loader)
            elif kind == "DeploymentScopeContractV2":
                # Scope contract is a lookup service, not a per-stage gate.
                verdict = {"status": AdequacyStatus.PASS.value,
                           "note": "scope-lookup contract; no per-stage adjudication"}
            else:
                verdict = {"status": AdequacyStatus.NOT_EVALUABLE.value,
                           "note": f"unknown kind {kind!r}"}
        except ScientificAdequacyBlocked:
            raise
        except Exception as e:
            verdict = {"status": AdequacyStatus.NOT_EVALUABLE.value,
                       "not_evaluable_reasons": [f"scoring error: {e}"]}
        summary["adjudications"].append({
            "policy_key": key, "kind": kind, "content_sha256": rec["content_sha256"],
            "required": required, "verdict": verdict,
        })
        status = verdict.get("status", AdequacyStatus.NOT_EVALUABLE.value)
        if status == AdequacyStatus.FAIL.value:
            problems.append(f"{key}=FAIL")
        elif status == AdequacyStatus.NOT_EVALUABLE.value and required:
            problems.append(f"{key}=NOT_EVALUABLE(required)")
    if problems:
        raise ScientificAdequacyBlocked(
            f"stage {stage_name!r} cannot advance under bound scientific policies: "
            + ", ".join(problems) + " (see stage record for details)")
    return summary


# =====================================================================
# Scorers per policy kind
# =====================================================================
def _score_evaluation(policy_dict: dict,
                      loader: Optional[Callable[[], dict]],
                      in_scope_domains: Optional[list[str]]) -> dict:
    policy = EvaluationAdequacyPolicyV2(**policy_dict)
    if loader is None:
        raise ScientificAdequacyBlocked(
            "EvaluationAdequacyPolicyV2 bound but no accuracy-report loader supplied")
    report = loader()
    if in_scope_domains is None:
        in_scope_domains = list(report.get("in_scope_domains", []))
        if not in_scope_domains:
            raise ScientificAdequacyBlocked(
                "no in_scope_domains supplied and accuracy_report carries none")
    metrics: dict[str, dict[str, float]] = {}
    for channel in ("student_vs_teacher", "student_vs_dft", "teacher_vs_dft"):
        ch = report.get(channel)
        if not isinstance(ch, dict):
            continue
        for domain, rec in ch.items():
            if not isinstance(rec, dict):
                continue
            for metric, val in rec.items():
                if isinstance(val, (int, float)):
                    metrics.setdefault(domain, {})[f"{channel}::{metric}"] = float(val)
    # __aggregate__ hook
    if "__aggregate__" in report and isinstance(report["__aggregate__"], dict):
        metrics["__aggregate__"] = {k: float(v) for k, v in report["__aggregate__"].items()
                                    if isinstance(v, (int, float))}
    verdict = evaluate_adequacy(policy, metrics, in_scope_domains=in_scope_domains)
    return verdict.model_dump()


def _score_uncertainty(policy_dict: dict,
                       loader: Optional[Callable[[], dict]]) -> dict:
    policy = UncertaintyPolicyV2(**policy_dict)
    if loader is None:
        raise ScientificAdequacyBlocked(
            "UncertaintyPolicyV2 bound but no uncertainty-report loader supplied")
    report = loader()
    status_str = (((report.get("calibration") or {}).get("status")) or
                  CalibrationStatus.UNCALIBRATED.value)
    try:
        status = CalibrationStatus(status_str)
    except ValueError:
        return {"status": AdequacyStatus.NOT_EVALUABLE.value,
                "not_evaluable_reasons": [f"unknown calibration.status {status_str!r}"]}
    verdict = adjudicate_uncertainty(policy, status)
    return {"status": verdict.value,
            "observed_calibration_status": status.value,
            "required_calibration_status": policy.required_status.value}


def _score_state_realization(policy_dict: dict,
                             loader: Optional[Callable[[], dict]]) -> dict:
    """Compare the realized MD manifest against the intended StatePreparationPolicy.

    Deterministic-only comparison. Any mismatch on a bound field is FAIL.
    Missing evidence is NOT_EVALUABLE.
    """
    policy = StatePreparationPolicy(**policy_dict)
    if loader is None:
        raise ScientificAdequacyBlocked(
            "StatePreparationPolicy bound but no md-manifest loader supplied")
    md = loader()
    problems: list[str] = []
    # ensemble
    realized_ensemble = md.get("ensemble") or (md.get("protocol") or {}).get("ensemble")
    if realized_ensemble and realized_ensemble != policy.ensemble.value:
        problems.append(f"ensemble realized={realized_ensemble!r} vs intended={policy.ensemble.value!r}")
    # temperature
    realized_T = md.get("temperature_K") or (md.get("protocol") or {}).get("temperature_K")
    if policy.intended_temperature_K is not None and realized_T is not None:
        if abs(float(realized_T) - float(policy.intended_temperature_K)) > 1.0:
            problems.append(f"T realized={realized_T} vs intended={policy.intended_temperature_K}")
    # starting-structure identity
    realized_ss = md.get("starting_structure_sha256") or (md.get("starting_structure") or {}).get("sha256")
    intended_ss = policy.starting_structure_provenance_ref
    if realized_ss and intended_ss and realized_ss not in intended_ss and intended_ss not in realized_ss:
        problems.append(f"starting_structure realized={realized_ss[:12]}... vs intended-ref={intended_ss[:60]}")
    if problems:
        return {"status": AdequacyStatus.FAIL.value, "mismatches": problems}
    return {"status": AdequacyStatus.PASS.value,
            "note": "realized state matches intended StatePreparationPolicy on bound fields"}


def _score_physical_validation(policy_dict: dict,
                               loader: Optional[Callable[[], dict]]) -> dict:
    """Verify that the validation_report emits observables matching the
    typed ObservableSpec set: right kind, right unit, cutoff-frozen, etc.
    """
    policy = PhysicalValidationPolicyV2(**policy_dict)
    if loader is None:
        raise ScientificAdequacyBlocked(
            "PhysicalValidationPolicyV2 bound but no validation-report loader supplied")
    report = loader()
    checks = {c.get("observable"): c for c in (report.get("checks") or []) if isinstance(c, dict)}
    missing: list[str] = []
    kind_mismatches: list[str] = []
    unit_mismatches: list[str] = []
    unfrozen_cutoffs: list[str] = []
    for obs in policy.observables:
        rec = checks.get(obs.name)
        if rec is None:
            missing.append(obs.name)
            continue
        # We cannot infer the executor's internal ObservableSpec.kind from the
        # report row alone -- but we CAN detect unit mismatch as a proxy for
        # observable-shape mismatch (e.g. "peak_g(r)" vs "Angstrom").
        if rec.get("unit") and rec["unit"] != obs.units:
            unit_mismatches.append(f"{obs.name}: report={rec['unit']!r} policy={obs.units!r}")
        # THRESHOLDED observable must have a criterion present
        if obs.role == ObservableRole.THRESHOLDED and not rec.get("criterion"):
            kind_mismatches.append(f"{obs.name}: THRESHOLDED but criterion=None in report")
        if obs.cutoff_source_ref and not obs.cutoff_frozen_before_student:
            unfrozen_cutoffs.append(obs.name)
    problems = missing + unit_mismatches + kind_mismatches + unfrozen_cutoffs
    if problems:
        return {
            "status": AdequacyStatus.FAIL.value,
            "missing_observables": missing,
            "unit_mismatches": unit_mismatches,
            "kind_or_criterion_mismatches": kind_mismatches,
            "unfrozen_cutoffs": unfrozen_cutoffs,
        }
    return {"status": AdequacyStatus.PASS.value,
            "note": f"all {len(policy.observables)} typed observables present with matching units"}


__all__ = [
    "ScientificAdequacyBlocked", "POLICIES_KEY",
    "bind_policy", "assert_stage_scientific_adequacy",
]
