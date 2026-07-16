"""Deterministic contracts for scheduler/agent-produced external stages."""
import importlib
import json
from pathlib import Path

from workflow.integrity import artifact_digest, verify_artifact


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


def validate_validation_manifest(manifest_path, validator, options=None):
    """Dispatch an external validation artifact to a config-selected validator.

    The core knows only the validation contract. Observable-specific logic lives
    under ``validation/`` and is selected by dotted callable path.
    """
    if not isinstance(validator, str) or not validator.startswith("validation."):
        raise ValueError("validation contract requires a validator under validation.*")
    module_name, callable_name = validator.rsplit(".", 1)
    function = getattr(importlib.import_module(module_name), callable_name, None)
    if not callable(function):
        raise ValueError(f"validation contract callable is invalid: {validator}")
    return function(manifest_path, **dict(options or {}))
