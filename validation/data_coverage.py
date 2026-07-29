"""Deterministic contract for dataset-source and deployment-coverage reports."""
import json
import math
from pathlib import Path

from validation.report import validate_evidence


ACCESS_MODES = {"full", "representative", "unavailable"}
COVERAGE_STATUSES = {"COMPLETE", "PARTIAL", "NOT_ASSESSABLE"}


def _nonnegative_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_data_coverage_report(manifest_path, required_source_categories=None,
                                  accepted_statuses=None, submitted_artifacts=None,
                                  allowed_evidence=None, enforce_required_pass=False):
    """Validate an auditable coverage assessment without imposing one descriptor."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("data coverage report requires schema_version=1")
    access = payload.get("teacher_training_data_access")
    if access not in ACCESS_MODES:
        raise ValueError("data coverage report has invalid teacher_training_data_access")
    status = payload.get("coverage_status")
    if status not in COVERAGE_STATUSES:
        raise ValueError("data coverage report has invalid coverage_status")
    if access == "unavailable" and status != "NOT_ASSESSABLE":
        raise ValueError("unavailable teacher data must be reported as NOT_ASSESSABLE")
    if not isinstance(payload.get("deployment_domain"), dict) or not payload["deployment_domain"]:
        raise ValueError("data coverage report requires a non-empty deployment_domain")
    sources = payload.get("dataset_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data coverage report requires non-empty dataset_sources")
    categories, fraction_total, source_evidence_roles = set(), 0.0, set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("category"), str):
            raise ValueError("each dataset source requires a category")
        category = source["category"].strip()
        if not category or category in categories:
            raise ValueError(f"dataset source category is empty or duplicated: {category!r}")
        categories.add(category)
        evidence_role = source.get("evidence_role")
        if not isinstance(evidence_role, str) or not evidence_role.strip():
            raise ValueError(f"dataset source requires evidence_role: {category}")
        source_evidence_roles.add(evidence_role)
        if not _nonnegative_integer(source.get("n_parents")):
            raise ValueError(f"dataset source n_parents must be a non-negative integer: {category}")
        if not _nonnegative_integer(source.get("n_frames")):
            raise ValueError(f"dataset source n_frames must be a non-negative integer: {category}")
        label_sources = source.get("label_sources")
        if (not isinstance(label_sources, list) or not label_sources or
                any(not isinstance(item, str) or not item.strip() for item in label_sources)):
            raise ValueError(f"dataset source requires non-empty label_sources: {category}")
        if "fraction" in source:
            fraction = source["fraction"]
            if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
                    not math.isfinite(fraction) or not 0 <= fraction <= 1):
                raise ValueError(f"dataset source fraction must be in [0, 1]: {category}")
            fraction_total += float(fraction)
    fractions_present = ["fraction" in source for source in sources]
    if any(fractions_present):
        if not all(fractions_present):
            raise ValueError("dataset source fractions must be provided for every source or none")
        if not math.isclose(fraction_total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("dataset source fractions must sum to 1")
    missing = set(required_source_categories or []) - categories
    if missing:
        raise ValueError("data coverage report is missing source categories: " +
                         ", ".join(sorted(missing)))
    dimensions = payload.get("coverage_dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("data coverage report requires coverage_dimensions")
    replay = payload.get("replay_policy")
    if not isinstance(replay, dict) or not isinstance(replay.get("enabled"), bool):
        raise ValueError("data coverage report requires replay_policy.enabled")
    if replay["enabled"]:
        if "teacher_training_replay" not in categories:
            raise ValueError("enabled replay policy requires a teacher_training_replay source")
        if (not isinstance(replay.get("selection_method"), str) or
                not replay["selection_method"].strip() or
                not _nonnegative_integer(replay.get("target_count"))):
            raise ValueError("enabled replay policy requires selection_method and target_count")
    for field in ("identified_gaps", "limitations"):
        value = payload.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"data coverage report {field} must be a list of strings")
    evidence_roles = validate_evidence(manifest_path, payload.get("evidence"),
                                       submitted_artifacts, False, allowed_evidence,
                                       label="data coverage")
    missing_evidence = source_evidence_roles - evidence_roles
    if missing_evidence:
        raise ValueError("dataset source evidence roles are missing: " +
                         ", ".join(sorted(missing_evidence)))
    policy_value = payload.get("dataset_policy")
    if not isinstance(policy_value, str) or not policy_value.strip():
        raise ValueError("data coverage report requires dataset_policy")
    policy_path = Path(policy_value).expanduser()
    policy_path = (policy_path.resolve() if policy_path.is_absolute() else
                   (manifest_path.parent / policy_path).resolve())
    policy_evidence = [item for item in payload["evidence"]
                       if item.get("role") == "dataset_policy"]
    if len(policy_evidence) != 1:
        raise ValueError("data coverage report requires exactly one dataset_policy evidence role")
    evidence_path = Path(policy_evidence[0]["path"]).expanduser()
    evidence_path = (evidence_path.resolve() if evidence_path.is_absolute() else
                     (manifest_path.parent / evidence_path).resolve())
    if evidence_path != policy_path:
        raise ValueError("dataset_policy does not match dataset_policy evidence")
    accepted = set(accepted_statuses or COVERAGE_STATUSES)
    if enforce_required_pass and status not in accepted:
        raise ValueError("data coverage status is outside the accepted statuses")
    return payload
