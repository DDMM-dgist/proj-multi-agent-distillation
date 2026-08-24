"""Framework V2 -- deterministic validators for the late-lifecycle policy
contracts K/L/M (Sections 12).

The uncertainty, deployment-MD, and physical-validation stages each declare a
typed policy contract (``UncertaintyPolicy`` / ``DeploymentMDPolicy`` /
``PhysicalValidationPolicy``). Before R31-class silent gaps, these stages had
no framework-level deterministic check that the produced report actually
covered what the policy demanded -- a report could omit a required metric,
skip a stability check, or claim an observable within tolerance without the
tolerance ever being applied.

Each validator here is pure and deterministic: it consumes the policy plus the
stage's produced observations and emits ``DeterministicFact`` records plus a
single verdict in {``PASS``, ``REVISE``, ``FAIL``}. A missing-required-input
is ``REVISE`` (the stage can be re-run to produce it); an observable that ran
but breached tolerance is ``FAIL`` (a scientific result, not a gap). Facts are
authoritative -- an LLM Judge cannot negate them (Section 13).

No observable name, tolerance number, ensemble, or metric is hard-coded here;
they all come from the policy contract and the produced report.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from framework_v2.contracts import (
    DeploymentMDPolicy,
    PhysicalValidationPolicy,
    UncertaintyPolicy,
)
from framework_v2.facts import DeterministicFact, FactVerdict

PASS = "PASS"
REVISE = "REVISE"
FAIL = "FAIL"


@dataclasses.dataclass(frozen=True)
class PolicyValidationReport:
    verdict: str
    facts: tuple[DeterministicFact, ...]
    policy_id: str
    policy_sha256: str
    profile: str

    def gate_ok(self) -> bool:
        return self.verdict == PASS

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "profile": self.profile,
            "verdict": self.verdict,
            "gate_ok": self.gate_ok(),
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "facts": [f.model_dump(mode="json") for f in self.facts],
        }


def _verdict_from_facts(facts: Sequence[DeterministicFact]) -> str:
    """FAIL if any fact is a substantive FAIL; REVISE if any input is missing
    (UNCHECKED); else PASS. Missing inputs are distinguished from breaches by
    the ``UNCHECKED`` verdict a validator sets when it could not evaluate."""
    if any(f.verdict == FactVerdict.FAIL for f in facts):
        # A missing required input is recorded as UNCHECKED, not FAIL, so a
        # genuine FAIL means an evaluated breach -> not recoverable by re-run.
        return FAIL
    if any(f.verdict == FactVerdict.UNCHECKED for f in facts):
        return REVISE
    return PASS


def _num(value: Any):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


# =====================================================================
# K. UncertaintyPolicy
# =====================================================================
def validate_uncertainty(
    policy: UncertaintyPolicy,
    report: Mapping[str, Any],
) -> PolicyValidationReport:
    """Check that the uncertainty report covers every metric the policy
    requires and, if the policy references calibration evidence, that the
    report carries it.

    ``report`` is expected to expose ``metrics`` (a mapping metric-name ->
    value) and, optionally, ``calibration_evidence_ref``.
    """
    validator = "framework_v2.policy_validators.validate_uncertainty"
    produced = report.get("metrics") or {}
    facts: list[DeterministicFact] = []
    for metric in policy.metrics:
        present = metric in produced and produced[metric] is not None
        facts.append(DeterministicFact(
            fact_id=f"uncertainty-metric-{metric}",
            kind="uncertainty_metric_present",
            observed=produced.get(metric),
            expected=metric,
            verdict=FactVerdict.PASS if present else FactVerdict.UNCHECKED,
            validator=validator,
            rationale=f"uncertainty policy requires metric {metric!r}",
        ))
    if policy.calibration_evidence_ref:
        has_cal = bool(report.get("calibration_evidence_ref"))
        facts.append(DeterministicFact(
            fact_id="uncertainty-calibration-evidence",
            kind="uncertainty_calibration_evidence_present",
            observed=report.get("calibration_evidence_ref"),
            expected=policy.calibration_evidence_ref,
            verdict=FactVerdict.PASS if has_cal else FactVerdict.UNCHECKED,
            validator=validator,
            rationale="policy declares calibration evidence; report must carry it",
        ))
    return PolicyValidationReport(
        verdict=_verdict_from_facts(facts),
        facts=tuple(facts),
        policy_id=policy.policy_id,
        policy_sha256=policy.content_sha256(),
        profile="uncertainty_policy_validation_report",
    )


# =====================================================================
# L. DeploymentMDPolicy
# =====================================================================
def validate_deployment_md(
    policy: DeploymentMDPolicy,
    report: Mapping[str, Any],
) -> PolicyValidationReport:
    """Check that every declared ensemble ran, every stability check is
    present and passed, and wall time is within the policy budget.

    ``report`` is expected to expose ``ensembles_run`` (list of ensemble
    identifiers or dicts with an ``id``), ``stability_checks`` (mapping
    check-name -> "PASS"/"FAIL"/value), and optionally ``wall_time_s``.
    """
    validator = "framework_v2.policy_validators.validate_deployment_md"
    facts: list[DeterministicFact] = []

    def _ensemble_id(e):
        return e.get("id") if isinstance(e, Mapping) else e

    declared_ensembles = [_ensemble_id(e) for e in policy.ensembles]
    run_raw = report.get("ensembles_run") or []
    run_ids = {_ensemble_id(e) for e in run_raw}
    for ens in declared_ensembles:
        facts.append(DeterministicFact(
            fact_id=f"md-ensemble-{ens}",
            kind="deployment_md_ensemble_ran",
            observed=ens in run_ids,
            expected=True,
            verdict=FactVerdict.PASS if ens in run_ids else FactVerdict.UNCHECKED,
            validator=validator,
            rationale=f"policy declares ensemble {ens!r}",
        ))

    produced_checks = report.get("stability_checks") or {}
    for check in policy.stability_checks:
        if check not in produced_checks:
            verdict = FactVerdict.UNCHECKED
        else:
            result = produced_checks[check]
            passed = (result == "PASS") if isinstance(result, str) else bool(result)
            verdict = FactVerdict.PASS if passed else FactVerdict.FAIL
        facts.append(DeterministicFact(
            fact_id=f"md-stability-{check}",
            kind="deployment_md_stability_check",
            observed=produced_checks.get(check),
            expected="PASS",
            verdict=verdict,
            validator=validator,
            rationale=f"policy declares stability check {check!r}",
        ))

    if policy.max_wall_time_s is not None:
        wall = _num(report.get("wall_time_s"))
        if wall is None:
            verdict = FactVerdict.UNCHECKED
        else:
            verdict = (FactVerdict.PASS if wall <= policy.max_wall_time_s
                       else FactVerdict.FAIL)
        facts.append(DeterministicFact(
            fact_id="md-wall-time",
            kind="deployment_md_within_wall_time",
            observed=report.get("wall_time_s"),
            expected={"max_wall_time_s": policy.max_wall_time_s},
            verdict=verdict,
            validator=validator,
            rationale="deployment MD must finish within the policy wall-time budget",
        ))

    return PolicyValidationReport(
        verdict=_verdict_from_facts(facts),
        facts=tuple(facts),
        policy_id=policy.policy_id,
        policy_sha256=policy.content_sha256(),
        profile="deployment_md_policy_validation_report",
    )


# =====================================================================
# M. PhysicalValidationPolicy
# =====================================================================
def _within_tolerance(observed: float, reference: float, tol: Mapping[str, Any]) -> bool:
    abs_tol = _num(tol.get("abs_tol"))
    rel_tol = _num(tol.get("rel_tol"))
    diff = abs(observed - reference)
    ok = True
    checked = False
    if abs_tol is not None:
        checked = True
        ok = ok and diff <= abs_tol
    if rel_tol is not None:
        checked = True
        denom = abs(reference) if reference != 0 else None
        ok = ok and (denom is not None and diff / denom <= rel_tol)
    # If no tolerance was declared for this observable, require exact equality.
    if not checked:
        return observed == reference
    return ok


def validate_physical_validation(
    policy: PhysicalValidationPolicy,
    report: Mapping[str, Any],
) -> PolicyValidationReport:
    """Check that every declared observable was computed and, where a
    reference value + tolerance is declared, that the computed value is within
    tolerance.

    ``report`` is expected to expose ``observables`` (mapping observable-name
    -> computed numeric value). A declared observable absent from the report
    is ``UNCHECKED`` (re-runnable -> REVISE); a computed value breaching its
    declared tolerance is ``FAIL`` (a scientific result).
    """
    validator = "framework_v2.policy_validators.validate_physical_validation"
    produced = report.get("observables") or {}
    facts: list[DeterministicFact] = []
    for name in policy.observables:
        if name not in produced or produced[name] is None:
            facts.append(DeterministicFact(
                fact_id=f"pv-observable-{name}",
                kind="physical_observable_present",
                observed=None,
                expected=name,
                verdict=FactVerdict.UNCHECKED,
                validator=validator,
                rationale=f"policy declares observable {name!r}",
            ))
            continue
        observed = _num(produced[name])
        reference = _num(policy.reference_values.get(name))
        tol = policy.tolerance_config.get(name) or {}
        if observed is None:
            verdict = FactVerdict.UNCHECKED
            rationale = f"observable {name!r} is present but non-numeric"
        elif reference is None:
            # No reference to compare against -> presence-only PASS.
            verdict = FactVerdict.PASS
            rationale = f"observable {name!r} computed; no reference value declared"
        else:
            ok = _within_tolerance(observed, reference, tol)
            verdict = FactVerdict.PASS if ok else FactVerdict.FAIL
            rationale = (f"observable {name!r} compared to reference "
                         f"{reference} with tolerance {dict(tol)}")
        facts.append(DeterministicFact(
            fact_id=f"pv-observable-{name}",
            kind="physical_observable_within_tolerance",
            observed=produced[name],
            expected={"reference": policy.reference_values.get(name),
                      "tolerance": dict(tol)},
            verdict=verdict,
            validator=validator,
            rationale=rationale,
        ))
    return PolicyValidationReport(
        verdict=_verdict_from_facts(facts),
        facts=tuple(facts),
        policy_id=policy.policy_id,
        policy_sha256=policy.content_sha256(),
        profile="physical_validation_policy_validation_report",
    )


__all__ = [
    "PASS", "REVISE", "FAIL",
    "PolicyValidationReport",
    "validate_uncertainty",
    "validate_deployment_md",
    "validate_physical_validation",
]
