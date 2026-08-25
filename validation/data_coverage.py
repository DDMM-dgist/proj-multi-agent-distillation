"""Deterministic contract for dataset-source and deployment-coverage reports.

schema_version=1 is the original single teacher_training_data_access/coverage_dimensions
shape and is preserved byte-for-byte below (_validate_v1) -- no behavior change.

schema_version=2 (_validate_v2) is the Priority #2 generic directed-coverage evidence
shape: it requires a non-empty `directed_coverage` list, each entry an explicit,
self-describing directed coverage question -- `direction`, `query_population`,
`reference_population`, and `reference_role` (a free-form campaign-supplied label,
e.g. "teacher_train_partition", "deployment_target_population",
"student_training_dataset", "candidate_population", "other") -- rather than a fixed
set of hardcoded block names. This validator does not hardcode which directions a
campaign must report; an optional `required_directions` parameter (mirroring
schema_version=1's `required_source_categories`) lets a specific campaign require
particular direction names via configuration. Whenever an entry's `reference_role` is
"teacher_train_partition", it must record validation/test as explicitly excluded and
carry a split_membership_verification block that is not allowed to claim cryptographic
verification while a reconstruction caveat is open (see configs/provenance/PROVENANCE.md)
-- this is a conditional rule keyed on `reference_role`, not on a specific block name.
It also requires protected_reference_status (a pointer only, whose required-frame-count
fields are cross-checked against the hash-verified reference_validation report file it
points to -- never a hardcoded literal count in this module), and an OPTIONAL
reference_population_partition_overlap (an arbitrary, campaign-defined partition
breakdown of some reference population, checked only for internal count consistency --
this validator does not hardcode any specific partition names, population, or size,
since those are per-dataset facts, not architectural invariants). It does not choose a
pass/fail threshold, an acquisition count, or a parent-selection policy; those remain
later, separately human-approved decisions (see coverage/__init__.py).
"""
import hashlib
import json
import math
from pathlib import Path

import yaml
from ase.io import read

from validation.report import validate_evidence
from workflow.integrity import sha256_file


ACCESS_MODES = {"full", "representative", "unavailable"}
COVERAGE_STATUSES = {"COMPLETE", "PARTIAL", "NOT_ASSESSABLE"}
TEACHER_TRAIN_PARTITION_ROLE = "teacher_train_partition"
REQUIRED_TEACHER_TRAINING_EXCLUSIONS = {"validation", "test"}
SPLIT_MEMBERSHIP_VERIFICATION_STATUSES = {
    "reconstructed_unverified_cross_version_rng", "cryptographically_verified",
}
PROTECTED_REFERENCE_POINTER_ROLE = "protected_reference_pointer"


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
                                  validation_contract_path=None, required_directions=None):
    """Validate an auditable coverage assessment without imposing one descriptor.

    Dispatches on the manifest's schema_version: 1 is the original single-descriptor
    shape (_validate_v1, unchanged); 2 is the Priority #2 generic directed-coverage
    evidence shape (_validate_v2). See this module's docstring for the
    schema_version=2 contract. `required_directions`, if given, only applies to
    schema_version=2 reports (mirrors schema_version=1's `required_source_categories`).
    """
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    schema_version = payload.get("schema_version")
    if schema_version == 2:
        return _validate_v2(manifest_path, payload, submitted_artifacts, allowed_evidence,
                            required_directions)
    if schema_version != 1:
        raise ValueError("data coverage report requires schema_version=1 or schema_version=2")
    return _validate_v1(manifest_path, payload, required_source_categories, accepted_statuses,
                        submitted_artifacts, allowed_evidence, enforce_required_pass,
                        validation_contract_path)


def _validate_v1(manifest_path, payload, required_source_categories, accepted_statuses,
                 submitted_artifacts, allowed_evidence, enforce_required_pass,
                 validation_contract_path):
    """Validate an auditable coverage assessment without imposing one descriptor.

    If validation_contract_path is given, this report's deployment_domain and its
    dataset_policy file's split_policy block must hash-match the run's locked validation
    contract exactly — a re-executed data_coverage stage may only pass under the SAME frozen
    Teacher applicability domain and dataset split policy, never redefined ones.
    """
    contract = (json.loads(Path(validation_contract_path).read_text())
               if validation_contract_path is not None else None)
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
    if "coverage_assessment" in payload:
        from validation.coverage_assessment import validate_coverage_assessment
        validate_coverage_assessment(payload["coverage_assessment"])
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


def _resolve_relative(manifest_path, raw):
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _require_evidence_role(payload, evidence_by_role, role_field_name, block, label):
    evidence_role = block.get(role_field_name)
    if evidence_role is None:
        return None
    if not isinstance(evidence_role, str) or not evidence_role.strip():
        raise ValueError(f"{label} {role_field_name} must be a non-empty string if given")
    if evidence_role not in evidence_by_role:
        raise ValueError(f"{label} {role_field_name} does not resolve to any submitted evidence")
    return evidence_by_role[evidence_role]


def _validate_directed_coverage_entry(manifest_path, payload, evidence_by_role, entry, index):
    label = f"directed_coverage[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{label} must be an object")
    for field in ("direction", "query_population", "reference_population", "reference_role"):
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}.{field} must be a non-empty string")
    for field in ("n_reference_frames", "n_reference_atoms"):
        if not _nonnegative_integer(entry.get(field)):
            raise ValueError(f"{label}.{field} must be a non-negative integer")
    structural_method = entry.get("structural_method")
    if structural_method is not None and (
        not isinstance(structural_method, str) or not structural_method.strip()
    ):
        raise ValueError(f"{label}.structural_method must be a non-empty string if given")

    if entry["reference_role"] == TEACHER_TRAIN_PARTITION_ROLE:
        excluded = entry.get("excluded_partitions")
        if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
            raise ValueError(f"{label}.excluded_partitions must be a list of strings")
        missing_exclusions = REQUIRED_TEACHER_TRAINING_EXCLUSIONS - set(excluded)
        if missing_exclusions:
            raise ValueError(
                f"{label}.excluded_partitions must include "
                f"{sorted(REQUIRED_TEACHER_TRAINING_EXCLUSIONS)}; missing "
                f"{sorted(missing_exclusions)} -- reference_role={TEACHER_TRAIN_PARTITION_ROLE!r} "
                "evidence must never be computed against validation or test frames"
            )
        manifest_value = entry.get("reference_manifest_path")
        if not isinstance(manifest_value, str) or not manifest_value.strip():
            raise ValueError(
                f"{label} with reference_role={TEACHER_TRAIN_PARTITION_ROLE!r} requires "
                "reference_manifest_path"
            )
        reference_manifest_path = _resolve_relative(manifest_path, manifest_value)
        if not reference_manifest_path.is_file():
            raise ValueError(f"{label}.reference_manifest_path does not exist: {reference_manifest_path}")
        recorded_hash = entry.get("reference_manifest_sha256")
        if not isinstance(recorded_hash, str) or not recorded_hash.strip():
            raise ValueError(
                f"{label} with reference_role={TEACHER_TRAIN_PARTITION_ROLE!r} requires "
                "reference_manifest_sha256"
            )
        actual_hash = sha256_file(reference_manifest_path)
        if actual_hash != recorded_hash:
            raise ValueError(
                f"{label}.reference_manifest_sha256 does not match the actual reference_manifest_path "
                f"file ({reference_manifest_path}); a genuinely reconstructed split membership must be "
                "evidenced by the real file, never asserted"
            )
        verification = entry.get("split_membership_verification")
        if not isinstance(verification, dict):
            raise ValueError(
                f"{label} with reference_role={TEACHER_TRAIN_PARTITION_ROLE!r} requires "
                "split_membership_verification"
            )
        status = verification.get("status")
        if status not in SPLIT_MEMBERSHIP_VERIFICATION_STATUSES:
            raise ValueError(
                f"{label}.split_membership_verification.status must be one of "
                f"{sorted(SPLIT_MEMBERSHIP_VERIFICATION_STATUSES)}"
            )
        if status != "cryptographically_verified":
            caveat = verification.get("caveat")
            if not isinstance(caveat, str) or not caveat.strip():
                raise ValueError(
                    f"{label}.split_membership_verification requires a non-empty caveat unless "
                    "status is 'cryptographically_verified' -- a reconstruction caveat must stay "
                    "disclosed, not silently dropped"
                )

    _require_evidence_role(payload, evidence_by_role, "evidence_role", entry, label)


def _validate_directed_coverage(manifest_path, payload, evidence_by_role, required_directions=None):
    entries = payload.get("directed_coverage")
    if not isinstance(entries, list) or not entries:
        raise ValueError("schema_version=2 report requires a non-empty directed_coverage list")
    seen_directions = set()
    for index, entry in enumerate(entries):
        _validate_directed_coverage_entry(manifest_path, payload, evidence_by_role, entry, index)
        seen_directions.add(entry["direction"])
    missing_directions = set(required_directions or []) - seen_directions
    if missing_directions:
        raise ValueError(
            "schema_version=2 report is missing required directed_coverage directions: " +
            ", ".join(sorted(missing_directions))
        )


def _validate_protected_reference_status(manifest_path, payload):
    block = payload.get("protected_reference_status")
    if not isinstance(block, dict):
        raise ValueError("schema_version=2 report requires protected_reference_status")
    if block.get("role") != PROTECTED_REFERENCE_POINTER_ROLE:
        raise ValueError(
            f"protected_reference_status.role must be {PROTECTED_REFERENCE_POINTER_ROLE!r} -- "
            "this block is a pointer to the independent DFT validation channel only, and must "
            "never inline or recompute that channel's own metrics"
        )
    report_path_value = block.get("report_path")
    if not isinstance(report_path_value, str) or not report_path_value.strip():
        raise ValueError("protected_reference_status requires report_path")
    report_path = _resolve_relative(manifest_path, report_path_value)
    if not report_path.is_file():
        raise ValueError(f"protected_reference_status.report_path does not exist: {report_path}")
    recorded_hash = block.get("report_sha256")
    if not isinstance(recorded_hash, str) or not recorded_hash.strip():
        raise ValueError("protected_reference_status requires report_sha256")
    actual_hash = sha256_file(report_path)
    if actual_hash != recorded_hash:
        raise ValueError(
            "protected_reference_status.report_sha256 does not match the actual report_path file "
            f"({report_path}); this pointer must be bound to real, unmodified evidence"
        )
    report_payload = json.loads(report_path.read_text())
    reference_block = report_payload.get("reference")
    if not isinstance(reference_block, dict):
        raise ValueError(
            "protected_reference_status.report_path must contain a `reference` block with "
            "logical_frames/protected_source_rows (see validation.reference_validation)"
        )
    expected_logical = reference_block.get("logical_frames")
    expected_rows = reference_block.get("protected_source_rows")
    if not _nonnegative_integer(expected_logical) or not _nonnegative_integer(expected_rows):
        raise ValueError(
            "protected_reference_status.report_path reference block must declare integer "
            "logical_frames/protected_source_rows"
        )
    # The expected counts come from the hash-verified report file itself -- this campaign's
    # actual frame counts, whatever they are -- never a hardcoded literal in this module.
    if block.get("required_logical_frames") != expected_logical:
        raise ValueError(
            "protected_reference_status.required_logical_frames must match the hash-verified "
            "report_path's reference.logical_frames exactly, not an independently supplied value"
        )
    if block.get("required_protected_source_rows") != expected_rows:
        raise ValueError(
            "protected_reference_status.required_protected_source_rows must match the hash-verified "
            "report_path's reference.protected_source_rows exactly, not an independently supplied "
            "value"
        )


def _validate_reference_population_partition_overlap(payload):
    """Optional, generic replacement for the old fixed-shape
    operational_baseline_membership block: an arbitrary, campaign-defined
    train/validation/test-like partition breakdown of some reference population,
    checked only for internal count consistency. Does not hardcode any specific
    partition names, population, or size -- those are per-dataset facts, not
    architectural invariants -- and is entirely optional, since not every
    campaign has an overlapping-partition question to report.
    """
    block = payload.get("reference_population_partition_overlap")
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError("reference_population_partition_overlap must be an object if given")
    total = block.get("total")
    if not _nonnegative_integer(total):
        raise ValueError("reference_population_partition_overlap.total must be a non-negative integer")
    partitions = block.get("partitions")
    if not isinstance(partitions, dict) or not partitions:
        raise ValueError("reference_population_partition_overlap.partitions must be a non-empty object")
    computed_total = 0
    for name, count in partitions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("reference_population_partition_overlap.partitions keys must be non-empty strings")
        if not _nonnegative_integer(count):
            raise ValueError(
                f"reference_population_partition_overlap.partitions[{name!r}] must be a "
                "non-negative integer"
            )
        computed_total += count
    if computed_total != total:
        raise ValueError(
            "reference_population_partition_overlap partitions must sum to total "
            f"({computed_total} != {total})"
        )


def _validate_v2_dataset_policy(manifest_path, payload, evidence_by_role):
    policy_value = payload.get("dataset_policy")
    if not isinstance(policy_value, str) or not policy_value.strip():
        raise ValueError("data coverage report requires dataset_policy")
    policy_path = _resolve_relative(manifest_path, policy_value)
    policy_evidence = evidence_by_role.get("dataset_policy")
    if policy_evidence is None:
        raise ValueError("data coverage report requires exactly one dataset_policy evidence role")
    evidence_path = _resolve_relative(manifest_path, policy_evidence["path"])
    if evidence_path != policy_path:
        raise ValueError("dataset_policy does not match dataset_policy evidence")
    policy_cfg = yaml.safe_load(policy_path.read_text())
    provenance = policy_cfg.get("provenance") if isinstance(policy_cfg, dict) else None
    if not isinstance(provenance, dict):
        raise ValueError(
            "schema_version=2 dataset_policy requires a non-empty provenance block with "
            "source_dataset_access / split_membership_status / deployed_checkpoint_linkage_status "
            "-- the legacy single teacher_training_data_access field does not separate these"
        )
    for field in ("source_dataset_access", "split_membership_status",
                  "deployed_checkpoint_linkage_status"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"schema_version=2 dataset_policy.provenance requires {field}")


def _validate_v2(manifest_path, payload, submitted_artifacts, allowed_evidence, required_directions=None):
    """Validate the Priority #2 generic directed-coverage evidence report.

    This never selects a pass/fail threshold, acquisition count, or parent-selection
    policy -- it validates that the required evidence, exclusions, and provenance are
    present and internally consistent (see this module's docstring).
    """
    validate_evidence(manifest_path, payload.get("evidence"), submitted_artifacts, False,
                      allowed_evidence, label="data coverage")
    evidence_by_role = {item["role"]: item for item in payload.get("evidence") or []}
    _validate_directed_coverage(manifest_path, payload, evidence_by_role, required_directions)
    _validate_protected_reference_status(manifest_path, payload)
    _validate_reference_population_partition_overlap(payload)
    _validate_v2_dataset_policy(manifest_path, payload, evidence_by_role)
    for field in ("identified_gaps", "limitations"):
        value = payload.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"data coverage report {field} must be a list of strings")
    return payload
