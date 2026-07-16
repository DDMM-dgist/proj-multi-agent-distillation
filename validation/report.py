"""Common envelope and deterministic checks for physical-validation reports."""
import json
import math
from pathlib import Path

from workflow.integrity import artifact_digest, verify_artifact


REPORT_STATUSES = {"PASS", "FAIL", "RECORDED", "NOT_EVALUATED"}


def evidence_record(role, path):
    path = Path(path).resolve()
    return {"role": str(role), "path": str(path), "integrity": artifact_digest(path)}


def criterion_passes(value, criterion):
    operator = criterion.get("operator")
    if operator == "max_abs":
        return abs(float(value)) <= float(criterion["threshold"])
    if operator == "max":
        return float(value) <= float(criterion["threshold"])
    if operator == "min":
        return float(value) >= float(criterion["threshold"])
    if operator == "target_tolerance":
        return abs(float(value) - float(criterion["target"])) <= float(criterion["tolerance"])
    if operator == "equals":
        return value == criterion["target"]
    raise ValueError(f"unknown validation criterion operator: {operator!r}")


def _validate_criterion(criterion, observable):
    if not isinstance(criterion, dict):
        raise ValueError(f"validation criterion must be an object for {observable}")
    operator = criterion.get("operator")
    required = {
        "max_abs": ("threshold",),
        "max": ("threshold",),
        "min": ("threshold",),
        "target_tolerance": ("target", "tolerance"),
        "equals": ("target",),
    }
    if operator not in required:
        raise ValueError(f"unknown validation criterion operator: {operator!r}")
    if operator == "equals":
        if "target" not in criterion:
            raise ValueError(f"validation criterion is missing target for {observable}")
        return
    for field in required[operator]:
        if not _finite_scalar(criterion.get(field)):
            raise ValueError(f"validation criterion {field} must be finite for {observable}")
    if operator == "target_tolerance" and criterion["tolerance"] < 0:
        raise ValueError(f"validation criterion tolerance must be non-negative for {observable}")


def make_check(domain, observable, value=None, unit=None, criterion=None, details=None,
               reason=None):
    if criterion is None:
        status = "NOT_EVALUATED" if reason else "RECORDED"
    else:
        status = "PASS" if criterion_passes(value, criterion) else "FAIL"
    result = {"domain": domain, "observable": observable, "status": status,
              "value": value, "unit": unit, "criterion": criterion}
    if details is not None:
        result["details"] = details
    if reason:
        result["reason"] = reason
    return result


def _finite_scalar(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_validation_report(manifest_path, required_observables=None,
                               submitted_artifacts=None, require_submitted_evidence=False,
                               required_pass_observables=None,
                               allowed_evidence=None, enforce_required_pass=False):
    """Validate the common report envelope without knowing observable-specific internals."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1 or not payload.get("profile"):
        raise ValueError("validation report requires schema_version=1 and profile")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("validation report requires non-empty checks")
    observables = set()
    for check in checks:
        observable = check.get("observable")
        status = check.get("status")
        if not check.get("domain") or not observable or status not in REPORT_STATUSES:
            raise ValueError("validation checks require domain, observable, and a valid status")
        if observable in observables:
            raise ValueError(f"validation observable is duplicated: {observable}")
        observables.add(observable)
        criterion = check.get("criterion")
        if criterion is not None:
            _validate_criterion(criterion, observable)
            value = check.get("value")
            if criterion.get("operator") != "equals" and not _finite_scalar(value):
                raise ValueError(f"criterion value must be finite for {observable}")
            expected = "PASS" if criterion_passes(value, criterion) else "FAIL"
            if status != expected:
                raise ValueError(f"validation status is inconsistent for {observable}")
        elif status in {"PASS", "FAIL"}:
            raise ValueError(f"validation check without a criterion cannot be {status}: {observable}")
    missing = set(required_observables or []) - observables
    if missing:
        raise ValueError("validation report is missing observables: " + ", ".join(sorted(missing)))
    if enforce_required_pass:
        status_by_observable = {check["observable"]: check["status"] for check in checks}
        failed = {observable for observable in (required_pass_observables or [])
                  if status_by_observable.get(observable) != "PASS"}
        if failed:
            raise ValueError("required validation observables did not pass: " +
                             ", ".join(sorted(failed)))

    submitted = {Path(path).resolve() for path in (submitted_artifacts or [])}
    allowed = [Path(path).resolve() for path in allowed_evidence] \
        if allowed_evidence is not None else None
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("validation report requires non-empty evidence")
    roles = set()
    for item in evidence:
        role, raw_path, integrity = item.get("role"), item.get("path"), item.get("integrity")
        if not role or not raw_path or not isinstance(integrity, dict):
            raise ValueError("validation evidence requires role, path, and integrity")
        if role in roles:
            raise ValueError(f"validation evidence role is duplicated: {role}")
        roles.add(role)
        path = Path(raw_path).expanduser()
        path = path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
        verify_artifact(path, integrity)
        if allowed is not None and not any(path == root or path.is_relative_to(root)
                                           for root in allowed):
            raise ValueError(f"validation evidence is not bound to this run: {path}")
        if require_submitted_evidence and path not in submitted:
            raise ValueError(f"validation evidence was not submitted as a stage artifact: {path}")
    return payload
