"""Deterministic contract for dataset-source and deployment-coverage reports."""
import hashlib
import json
import math
from pathlib import Path

import yaml
from ase.io import read

from validation.report import validate_evidence


ACCESS_MODES = {"full", "representative", "unavailable"}
COVERAGE_STATUSES = {"COMPLETE", "PARTIAL", "NOT_ASSESSABLE"}


def _nonnegative_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nested(payload, dotted):
    value = payload
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"coverage statistics field is missing: {dotted}")
        value = value[key]
    return value


def _label_sources(frames, field):
    labels = set()
    for atoms in frames:
        value = atoms.info.get(field, "unlabeled")
        values = value if isinstance(value, (list, tuple)) else [value]
        labels.update(str(item) for item in values)
    return sorted(labels)


def _source_statistics(path, config):
    kind = config.get("kind")
    if kind == "ase":
        frames = read(path, index=":")
        grouping_key = config.get("grouping_key", "parent_structure_id")
        parents = []
        for index, atoms in enumerate(frames):
            if grouping_key not in atoms.info:
                raise ValueError(
                    f"coverage source frame {index} is missing grouping key {grouping_key!r}"
                )
            parents.append(str(atoms.info[grouping_key]))
        return {"n_frames": len(frames), "n_parents": len(set(parents)),
                "label_sources": _label_sources(frames, config.get("label_source_field",
                                                                    "label_source"))}
    if kind == "json":
        payload = json.loads(Path(path).read_text())
        return {
            "n_frames": _nested(payload, config.get("n_frames_field", "n_frames")),
            "n_parents": _nested(payload, config.get("n_parents_field", "n_parents")),
            "label_sources": _nested(payload,
                                      config.get("label_sources_field", "label_sources")),
        }
    raise ValueError("coverage source statistics.kind must be ase or json")


def validate_data_coverage_report(manifest_path, required_source_categories=None,
                                  accepted_statuses=None, submitted_artifacts=None,
                                  allowed_evidence=None, enforce_required_pass=False,
                                  validation_contract_path=None):
    """Validate an auditable coverage assessment without imposing one descriptor.

    If validation_contract_path is given, this report's deployment_domain and its
    dataset_policy file's split_policy block must hash-match the run's locked validation
    contract exactly — a re-executed data_coverage stage may only pass under the SAME frozen
    Teacher applicability domain and dataset split policy, never redefined ones.
    """
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    contract = (json.loads(Path(validation_contract_path).read_text())
               if validation_contract_path is not None else None)
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
    if contract is not None:
        locked_domain = contract["components"]["teacher_applicability_domain"]
        domain_hash = hashlib.sha256(
            json.dumps(payload["deployment_domain"], indent=2, sort_keys=True).encode()
        ).hexdigest()
        if domain_hash != locked_domain["sha256"]:
            raise ValueError(
                "data coverage deployment_domain does not match the run's locked validation "
                "contract; a genuine change to the Teacher applicability domain requires a new "
                "run, not a re-executed data_coverage stage"
            )
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
        if evidence_role in source_evidence_roles:
            raise ValueError(f"dataset source evidence_role is duplicated: {evidence_role}")
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
    if status != "NOT_ASSESSABLE" and not dimensions:
        raise ValueError("assessable coverage requires non-empty coverage_dimensions")
    for name, dimension in dimensions.items():
        if (not isinstance(name, str) or not name.strip() or not isinstance(dimension, dict) or
                not isinstance(dimension.get("method"), str) or not dimension["method"].strip()):
            raise ValueError("every coverage dimension requires a non-empty method")
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
    evidence_by_role = {item["role"]: item for item in payload["evidence"]}
    for source in sources:
        statistics = source.get("statistics")
        if not isinstance(statistics, dict):
            raise ValueError(
                f"dataset source requires deterministic statistics: {source['category']}"
            )
        evidence = evidence_by_role[source["evidence_role"]]
        evidence_path = Path(evidence["path"]).expanduser()
        evidence_path = (evidence_path.resolve() if evidence_path.is_absolute() else
                         (manifest_path.parent / evidence_path).resolve())
        actual = _source_statistics(evidence_path, statistics)
        if (not _nonnegative_integer(actual.get("n_parents")) or
                not _nonnegative_integer(actual.get("n_frames")) or
                not isinstance(actual.get("label_sources"), list) or
                any(not isinstance(item, str) or not item.strip()
                    for item in actual["label_sources"])):
            raise ValueError(
                f"dataset source statistics are invalid for {source['category']}"
            )
        for field in ("n_parents", "n_frames"):
            if actual[field] != source[field]:
                raise ValueError(
                    f"dataset source {field} does not match evidence for {source['category']}"
                )
        if sorted(actual["label_sources"]) != sorted(source["label_sources"]):
            raise ValueError(
                f"dataset source label_sources do not match evidence for {source['category']}"
            )
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
    if contract is not None:
        locked_split_policy = contract["components"]["dataset_split_policy"]
        policy_cfg = yaml.safe_load(policy_path.read_text())
        split_policy = policy_cfg.get("split_policy") if isinstance(policy_cfg, dict) else None
        if not isinstance(split_policy, dict) or not split_policy:
            raise ValueError("data coverage dataset_policy file requires a non-empty split_policy")
        split_policy_hash = hashlib.sha256(
            json.dumps(split_policy, indent=2, sort_keys=True).encode()
        ).hexdigest()
        if split_policy_hash != locked_split_policy["sha256"]:
            raise ValueError(
                "data coverage dataset_policy split_policy does not match the run's locked "
                "validation contract; a genuine change to the dataset split policy requires a "
                "new run, not a re-executed data_coverage stage"
            )
    accepted = set(accepted_statuses or COVERAGE_STATUSES)
    if enforce_required_pass and status not in accepted:
        raise ValueError("data coverage status is outside the accepted statuses")
    return payload
