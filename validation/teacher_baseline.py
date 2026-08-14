"""Deterministic contract for a teacher-first validation baseline."""
import hashlib
import json
from pathlib import Path

from validation.report import validate_validation_report


PURPOSES = {"student_teacher_fidelity", "physical_accuracy", "deployment_stability"}
REFERENCE_SOURCES = {"teacher", "dft", "experiment", "other"}
APPLICABILITY_STATUSES = {"SUPPORTED", "CONDITIONAL", "NOT_ESTABLISHED"}


def validate_teacher_baseline_report(manifest_path, required_observables=None,
                                     required_pass_observables=None,
                                     accepted_applicability=None,
                                     submitted_artifacts=None, allowed_evidence=None,
                                     enforce_required_pass=False,
                                     validation_contract_path=None):
    """Validate scope, provenance, and frozen targets before student training.

    If validation_contract_path is given, the report's deployment_domain must hash-match the
    run's locked validation contract's teacher_applicability_domain component exactly — a
    re-executed teacher_baseline (e.g. via recovery) may only pass under the SAME frozen Teacher
    applicability domain, never a redefined one.
    """
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    validate_validation_report(
        manifest_path,
        required_observables=required_observables,
        required_pass_observables=required_pass_observables,
        submitted_artifacts=submitted_artifacts,
        allowed_evidence=allowed_evidence,
        enforce_required_pass=enforce_required_pass,
    )
    teacher = payload.get("teacher")
    if not isinstance(teacher, dict) or not teacher.get("config"):
        raise ValueError("teacher baseline requires teacher.config")
    config_path = Path(teacher["config"]).expanduser()
    config_path = (config_path.resolve() if config_path.is_absolute() else
                   (manifest_path.parent / config_path).resolve())
    config_evidence = [item for item in payload["evidence"]
                       if item.get("role") == "teacher_config"]
    if len(config_evidence) != 1:
        raise ValueError("teacher baseline requires exactly one teacher_config evidence role")
    evidence_path = Path(config_evidence[0]["path"]).expanduser()
    evidence_path = (evidence_path.resolve() if evidence_path.is_absolute() else
                     (manifest_path.parent / evidence_path).resolve())
    if evidence_path != config_path:
        raise ValueError("teacher.config does not match teacher_config evidence")
    scope_value = payload.get("distillation_scope")
    if not isinstance(scope_value, str) or not scope_value.strip():
        raise ValueError("teacher baseline requires distillation_scope")
    scope_path = Path(scope_value).expanduser()
    scope_path = (scope_path.resolve() if scope_path.is_absolute() else
                  (manifest_path.parent / scope_path).resolve())
    scope_evidence = [item for item in payload["evidence"]
                      if item.get("role") == "distillation_scope"]
    if len(scope_evidence) != 1:
        raise ValueError("teacher baseline requires exactly one distillation_scope evidence role")
    evidence_path = Path(scope_evidence[0]["path"]).expanduser()
    evidence_path = (evidence_path.resolve() if evidence_path.is_absolute() else
                     (manifest_path.parent / evidence_path).resolve())
    if evidence_path != scope_path:
        raise ValueError("distillation_scope does not match distillation_scope evidence")
    profile_value = payload.get("validation_profile")
    if not isinstance(profile_value, str) or not profile_value.strip():
        raise ValueError("teacher baseline requires validation_profile")
    profile_path = Path(profile_value).expanduser()
    profile_path = (profile_path.resolve() if profile_path.is_absolute() else
                    (manifest_path.parent / profile_path).resolve())
    profile_evidence = [item for item in payload["evidence"]
                        if item.get("role") == "validation_profile"]
    if len(profile_evidence) != 1:
        raise ValueError("teacher baseline requires exactly one validation_profile evidence role")
    evidence_path = Path(profile_evidence[0]["path"]).expanduser()
    evidence_path = (evidence_path.resolve() if evidence_path.is_absolute() else
                     (manifest_path.parent / evidence_path).resolve())
    if evidence_path != profile_path:
        raise ValueError("validation_profile does not match validation_profile evidence")
    domain = payload.get("deployment_domain")
    if not isinstance(domain, dict) or not domain:
        raise ValueError("teacher baseline requires a non-empty deployment_domain")
    if validation_contract_path is not None:
        contract = json.loads(Path(validation_contract_path).read_text())
        locked = contract["components"]["teacher_applicability_domain"]
        domain_hash = hashlib.sha256(
            json.dumps(domain, indent=2, sort_keys=True).encode()
        ).hexdigest()
        if domain_hash != locked["sha256"]:
            raise ValueError(
                "teacher baseline deployment_domain does not match the run's locked "
                "validation contract; a genuine change to the Teacher applicability domain "
                "requires a new run, not a re-executed teacher_baseline stage"
            )
    applicability = payload.get("applicability")
    if not isinstance(applicability, dict) or applicability.get("status") not in APPLICABILITY_STATUSES:
        raise ValueError("teacher baseline requires a valid applicability.status")
    limitations = applicability.get("limitations", [])
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise ValueError("teacher applicability limitations must be a list of strings")
    accepted = set(accepted_applicability or {"SUPPORTED", "CONDITIONAL"})
    if enforce_required_pass and applicability["status"] not in accepted:
        raise ValueError("teacher applicability is outside the accepted statuses")
    for check in payload["checks"]:
        if check.get("purpose") not in PURPOSES:
            raise ValueError(f"teacher baseline check has invalid purpose: {check.get('observable')}")
        if check.get("reference_source") not in REFERENCE_SOURCES:
            raise ValueError(
                f"teacher baseline check has invalid reference_source: {check.get('observable')}"
            )
        if not isinstance(check.get("protocol"), str) or not check["protocol"].strip():
            raise ValueError(f"teacher baseline check requires protocol: {check.get('observable')}")
    return payload
