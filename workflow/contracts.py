"""Deterministic contracts for scheduler/agent-produced external stages."""
import importlib
import inspect
import json
from pathlib import Path

from workflow.integrity import artifact_digest, verify_artifact

# Optional top-level `teacher_validation_objectives: list[str]` key a validation_profile.yaml MAY
# declare (additive; a profile that omits it gets identical behavior to before this vocabulary
# existed -- see runtimes.pydantic_ai.teacher_validation_plan, the consumer that reads it as
# context for an autonomous Teacher-validation-planning proposal). Every value must be drawn from
# this fixed, generic vocabulary: never a material/dataset/campaign-specific string. New
# objectives are added here, never invented ad hoc by a config or a proposal.
TEACHER_VALIDATION_OBJECTIVES = (
    "require_predictive_fidelity_when_evidence_supports_it",
    "assess_deployment_applicability_when_domain_evidence_exists",
    "prohibit_unsupported_generalization_claims",
)


def parse_teacher_validation_objectives(profile_cfg):
    """Parse and validate the OPTIONAL `teacher_validation_objectives` key from a parsed
    validation_profile.yaml mapping. Returns ``[]`` if the key is absent (fully additive/backward
    compatible); raises ValueError if present but not a list of values drawn from
    ``TEACHER_VALIDATION_OBJECTIVES``."""
    if not isinstance(profile_cfg, dict):
        raise ValueError("validation_profile must be a mapping")
    objectives = profile_cfg.get("teacher_validation_objectives")
    if objectives is None:
        return []
    if (not isinstance(objectives, list) or
            any(not isinstance(item, str) or item not in TEACHER_VALIDATION_OBJECTIVES
                for item in objectives)):
        raise ValueError(
            "validation_profile.teacher_validation_objectives must be a list drawn from: "
            + ", ".join(TEACHER_VALIDATION_OBJECTIVES)
        )
    return list(objectives)


def _resolve(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _same_integrity(left, right):
    return all(left.get(key) == right.get(key) for key in ("kind", "size", "sha256"))


def validate_md_manifest(manifest_path, expected_committee_manifest, submitted_artifacts,
                         required_evidence=None):
    """Bind an external MD result to a checkpoint and its submitted evidence files."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    required = {"selected_seed", "checkpoint", "checkpoint_integrity", "committee_manifest"}
    missing = required - set(payload)
    if missing:
        raise ValueError("MD manifest is missing: " + ", ".join(sorted(missing)))

    committee_path = _resolve(payload["committee_manifest"], base)
    expected_committee_manifest = Path(expected_committee_manifest).resolve()
    if committee_path != expected_committee_manifest:
        raise ValueError("MD manifest does not reference the approved committee manifest")
    committee = json.loads(committee_path.read_text())
    seed = int(payload["selected_seed"])
    candidates = [model for model in committee.get("models", []) if int(model.get("seed", -1)) == seed]
    if len(candidates) != 1:
        raise ValueError(f"selected seed {seed} is not unique in the committee manifest")

    checkpoint = _resolve(payload["checkpoint"], base)
    approved = candidates[0]
    if checkpoint != Path(approved["path"]).resolve():
        raise ValueError("MD checkpoint is not the selected committee checkpoint")
    verify_artifact(checkpoint, approved["integrity"])
    current = artifact_digest(checkpoint)
    if not _same_integrity(current, payload["checkpoint_integrity"]):
        raise ValueError("MD checkpoint integrity does not match the manifest")

    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("MD manifest requires a non-empty evidence list")
    submitted = {Path(path).resolve() for path in submitted_artifacts}
    roles = set()
    for item in evidence:
        if not isinstance(item, dict) or not item.get("role") or not item.get("path"):
            raise ValueError("each MD evidence item requires role, path, and integrity")
        role = str(item["role"])
        if role in roles:
            raise ValueError(f"MD evidence role is duplicated: {role}")
        roles.add(role)
        path = _resolve(item["path"], base)
        if path not in submitted:
            raise ValueError(f"MD evidence was not submitted as a stage artifact: {path}")
        try:
            verify_artifact(path, item.get("integrity", {}))
        except (FileNotFoundError, RuntimeError) as exc:
            raise ValueError(f"MD evidence integrity check failed for role {role}: {path}") from exc
        current = artifact_digest(path)
        if not _same_integrity(current, item.get("integrity", {})):
            raise ValueError(f"MD evidence integrity does not match for role {role}: {path}")

    missing_roles = set(required_evidence or []) - roles
    if missing_roles:
        raise ValueError("MD manifest is missing required evidence roles: " +
                         ", ".join(sorted(missing_roles)))
    return payload


def build_validation_contract_components(scope_cfg, profile_cfg, policy_cfg):
    """Build the write-once validation-contract components from parsed config mappings.

    ``scope_cfg``'s deployment_domain is the sole authoritative source; if ``profile_cfg``'s
    own copy differs, this hard-fails rather than silently picking one (the two declarations
    must be reconciled in the source configs first). This is the single shared construction
    path used both by the manual ``establish-validation-contract`` CLI helper
    (workflow.steps.establish_validation_contract_from_configs) and by
    ``RunController.initialize()`` when a workflow declares ``validation_contract_sources`` —
    callers differ only in where the three configs come from (arbitrary paths vs. a run's own
    snapshotted source copies), never in how the components are derived from them.
    """
    authoritative_domain = scope_cfg.get("deployment_domain") if isinstance(scope_cfg, dict) else None
    if not isinstance(authoritative_domain, dict) or not authoritative_domain:
        raise ValueError("distillation_scope requires a non-empty deployment_domain")
    duplicate_domain = profile_cfg.get("deployment_domain") if isinstance(profile_cfg, dict) else None
    if duplicate_domain != authoritative_domain:
        raise ValueError(
            "validation_profile's deployment_domain does not match the authoritative "
            "deployment_domain declared in distillation_scope; reconcile the source "
            "configs before establishing the validation contract"
        )
    checks = profile_cfg.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("validation_profile requires a non-empty checks list")
    validation_scope = {"shared_md_protocol": profile_cfg.get("shared_md_protocol"),
                        "checks": checks}
    split_policy = policy_cfg.get("split_policy") if isinstance(policy_cfg, dict) else None
    if not isinstance(split_policy, dict) or not split_policy:
        raise ValueError("dataset_policy requires a non-empty split_policy")
    return {"teacher_applicability_domain": authoritative_domain,
           "validation_scope": validation_scope, "dataset_split_policy": split_policy}


def validate_validation_manifest(manifest_path, validator, options=None, submitted_artifacts=None,
                                 allowed_evidence=None, enforce_required_pass=False):
    """Dispatch an external validation artifact to a config-selected validator.

    The core knows only the validation contract. Observable-specific logic is
    selected by a hash-bound dotted callable path; built-ins live under
    ``validation/`` and external adapters may live in their own package.
    """
    if not isinstance(validator, str) or "." not in validator:
        raise ValueError("validation contract requires a dotted validator callable")
    module_name, callable_name = validator.rsplit(".", 1)
    function = getattr(importlib.import_module(module_name), callable_name, None)
    if not callable(function):
        raise ValueError(f"validation contract callable is invalid: {validator}")
    kwargs = dict(options or {})
    if "submitted_artifacts" in inspect.signature(function).parameters:
        kwargs["submitted_artifacts"] = submitted_artifacts
    if "allowed_evidence" in inspect.signature(function).parameters:
        kwargs["allowed_evidence"] = allowed_evidence
    if "enforce_required_pass" in inspect.signature(function).parameters:
        kwargs["enforce_required_pass"] = enforce_required_pass
    return function(manifest_path, **kwargs)
