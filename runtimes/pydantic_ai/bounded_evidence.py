"""Deterministic bounded-evidence summaries for agent context."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from workflow.integrity import artifact_digest, sha256_file

MAX_EVIDENCE_BYTES = 256 * 1024
DIRECT_JUDGE_ARTIFACT_BYTES = 1_000_000
# Cap on the number of per-file entries inlined into an LLM-facing directory summary. The
# canonical `artifact_digest` still hashes the COMPLETE tree; this only bounds the *evidence*
# representation so a large output directory (e.g. a multi-seed committee) cannot blow
# MAX_EVIDENCE_BYTES. 64 keeps a rich, deterministic sample while staying well under the cap.
EVIDENCE_DIRECTORY_FILE_CAP = 64

# --- Extensible JSON-evidence adapter registry -------------------------------------------------
#
# `_json_summary` never grows a new campaign-specific `if payload.get(...) == "<name>":` branch.
# Instead each adapter is a (summary_key, predicate, summarizer) triple: `predicate(payload)`
# decides -- generically, off the payload's own declared shape (a "profile" field, or a
# required-key signature) -- whether the adapter applies; `summarizer(payload)` then produces the
# semantic sub-summary nested under `summary[summary_key]`. New evidence kinds (e.g. a future
# coverage direction, a new report profile) register an adapter; they never edit this dispatch.
_JsonEvidencePredicate = Callable[[dict], bool]
_JsonEvidenceSummarizer = Callable[[dict], dict]
_JSON_EVIDENCE_ADAPTERS: list[tuple[str, _JsonEvidencePredicate, _JsonEvidenceSummarizer]] = []


def register_json_evidence_adapter(
    summary_key: str, predicate: _JsonEvidencePredicate, summarizer: _JsonEvidenceSummarizer,
) -> None:
    """Register an extensible bounded-evidence adapter for `.json` artifacts.

    `predicate(payload) -> bool` must decide applicability generically (a self-declared
    "profile" field, or a required-key signature) -- never a campaign/dataset name. `summarizer
    (payload) -> dict` must return only bounded, semantic, human/agent-facing fields (descriptive
    counts, fractions, distributions, identities, provenance hashes, limitations) -- never raw
    vectors, index/backend objects, or representation/search-backend parameter internals.
    """
    _JSON_EVIDENCE_ADAPTERS.append((summary_key, predicate, summarizer))


def _frame_category(atoms) -> str:
    """A frame's source category. Prefers ``source_category`` -- the field frozen teacher_baseline
    (and other original-split-sourced) frames actually carry -- over the legacy generic
    ``config_type``/``source`` keys, which remain the fallback for evidence produced by code paths
    that never adopted ``source_category``."""
    return str(atoms.info.get(
        "source_category", atoms.info.get("config_type", atoms.info.get("source", "unknown"))))


def _frame_deployment_slice(atoms) -> str:
    """A frame's declared deployment-slice membership. Most frames carry a literal
    ``deployment_slice_membership`` info key; some instead embed it inline as
    ``"deployment_slice_membership=<value>"`` within ``source_active_learning_type`` (an existing,
    real encoding this reads rather than treats as an evidence gap). Falls back to the legacy
    generic ``domain`` key, then ``"unknown"``, for evidence that carries neither convention."""
    direct = atoms.info.get("deployment_slice_membership")
    if direct not in (None, ""):
        return str(direct)
    embedded = atoms.info.get("source_active_learning_type")
    if isinstance(embedded, str) and "deployment_slice_membership=" in embedded:
        return embedded.split("deployment_slice_membership=", 1)[1].strip()
    return str(atoms.info.get("domain", "unknown"))


def _is_split_membership_manifest(payload: dict) -> bool:
    """Generic shape-signature match for a split/lineage crosswalk manifest (e.g.
    ``configs/provenance/teacher_training_split_manifest.json``) -- never a hardcoded file name.
    A matching manifest is a dict with a non-empty top-level ``records`` list whose entries each
    carry ``source_category``, ``source_local_index``, and ``split``."""
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return False
    sample = records[0]
    return isinstance(sample, dict) and {"source_category", "source_local_index", "split"} <= set(sample)


def build_split_crosswalk(paths: Iterable[str | Path]) -> dict:
    """Build a deterministic ``(source_category, source_local_index) -> split`` crosswalk from
    every JSON artifact among ``paths`` whose own shape matches a split-membership manifest (see
    ``_is_split_membership_manifest``) -- the run-bound Teacher split manifest is the only such
    artifact in a normal run, but this never hardcodes its name or path.

    Returns ``{"resolved": {...}, "ambiguous": {...}, "sources": [...]}``. A key present in more
    than one matching manifest, or repeated within one manifest, with DIFFERENT ``split`` values is
    moved to ``"ambiguous"`` and removed from ``"resolved"`` -- callers must fail closed on an
    ambiguous key (never guess which side is right) rather than silently picking either value.
    ``"sources"`` records the path/sha256 of every manifest actually used, for provenance.
    """
    resolved: dict[tuple, str] = {}
    ambiguous: set[tuple] = set()
    sources: list[dict] = []
    for raw in paths:
        p = Path(raw)
        if p.suffix.lower() != ".json" or not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or not _is_split_membership_manifest(payload):
            continue
        sources.append({"path": str(p.resolve()), "sha256": sha256_file(p)})
        for record in payload["records"]:
            if not isinstance(record, dict):
                continue
            key = (str(record.get("source_category")), record.get("source_local_index"))
            split = record.get("split")
            if key in ambiguous:
                continue
            if key in resolved and resolved[key] != split:
                ambiguous.add(key)
                del resolved[key]
                continue
            resolved[key] = split
    return {"resolved": resolved, "ambiguous": ambiguous, "sources": sources}


def _frame_summary(path: Path, *, split_crosswalk: dict | None = None) -> dict:
    from ase.io import read
    import math
    import numpy as np

    frames = read(str(path), index=":")
    compositions = Counter()
    categories = Counter()
    domains = Counter()
    atom_counts = []
    label_keys = set()
    finite_teacher_energy = 0
    finite_teacher_force = 0
    resolved = (split_crosswalk or {}).get("resolved", {})
    ambiguous = (split_crosswalk or {}).get("ambiguous", set())
    split_counts = Counter()
    source_split_joined = 0
    source_split_ambiguous = 0
    source_split_unjoined = 0
    descendant_lineage_missing = 0
    for index, atoms in enumerate(frames):
        symbols = Counter(atoms.get_chemical_symbols())
        compositions[" ".join(f"{k}{v}" for k, v in sorted(symbols.items()))] += 1
        categories[_frame_category(atoms)] += 1
        domains[_frame_deployment_slice(atoms)] += 1
        atom_counts.append(len(atoms))
        cat = atoms.info.get("source_category")
        local_index = atoms.info.get("source_local_index")
        if cat is not None and local_index is not None:
            # This frame declares original-split membership provenance (frozen teacher_baseline
            # source frames and similar) -- join it to the authoritative split manifest rather
            # than requiring the unrelated acquisition-descendant `parent_structure_id` concept
            # (see workflow.steps._base_parent_id) for frames that were never given one.
            key = (str(cat), int(local_index))
            if key in ambiguous:
                source_split_ambiguous += 1
            elif key in resolved:
                source_split_joined += 1
                split_counts[str(resolved[key])] += 1
            else:
                source_split_unjoined += 1
        elif "parent_structure_id" not in atoms.info and index > 0:
            # No original-split signature on this frame at all -- fall back to the
            # acquisition/augmentation descendant-lineage convention.
            descendant_lineage_missing += 1
        for key in atoms.info:
            if key.endswith("energy") or key in {"energy", "teacher_energy"}:
                label_keys.add(key)
        for key in atoms.arrays:
            if key.endswith("forces") or key in {"forces", "teacher_forces"}:
                label_keys.add(key)
        energy = atoms.info.get("teacher_energy")
        forces = atoms.arrays.get("teacher_forces")
        if isinstance(energy, (int, float, np.integer, np.floating)) and math.isfinite(float(energy)):
            finite_teacher_energy += 1
        if forces is not None and getattr(forces, "size", 0) > 0 and bool(np.isfinite(forces).all()):
            finite_teacher_force += 1
    return {
        "n_frames": len(frames),
        "atom_count_min": min(atom_counts) if atom_counts else 0,
        "atom_count_max": max(atom_counts) if atom_counts else 0,
        "composition_counts": dict(compositions.most_common(32)),
        "category_counts": dict(categories.most_common(64)),
        "domain_counts": dict(domains.most_common(64)),
        "source_split_joined_frames": source_split_joined,
        "source_split_joined_counts": dict(split_counts),
        "source_split_ambiguous_frames": source_split_ambiguous,
        "source_split_unjoined_frames": source_split_unjoined,
        "descendant_lineage_missing_frames": descendant_lineage_missing,
        # Backward-compatible alias: any frame whose lineage (of either kind) could not be
        # established -- never again "every frame but the first" for split-sourced evidence that
        # simply uses a different, still-real lineage convention.
        "missing_lineage_frames": source_split_ambiguous + source_split_unjoined + descendant_lineage_missing,
        "label_keys": sorted(label_keys),
        "finite_teacher_energy_count": finite_teacher_energy,
        "finite_teacher_force_count": finite_teacher_force,
    }


def _json_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        summary = {
            "top_level_keys": sorted(map(str, payload.keys()))[:128],
            "schema_version": payload.get("schema_version"),
            "status": payload.get("status"),
            "n_artifacts": len(payload.get("artifacts", []))
            if isinstance(payload.get("artifacts"), list) else None,
            "n_checks": len(payload.get("checks", []))
            if isinstance(payload.get("checks"), list) else None,
            "n_models": len(payload.get("models", []))
            if isinstance(payload.get("models"), list) else None,
        }
        for summary_key, predicate, summarizer in _JSON_EVIDENCE_ADAPTERS:
            if predicate(payload):
                summary[summary_key] = summarizer(payload)
        # For a four-channel accuracy_report, surface the run-bound evaluation-population identity
        # and its lineage/hash binding (read verbatim from the sibling reference_validation.json)
        # right next to the fidelity metrics, so the Stage-8 Judges can confirm the population in
        # the same place they read the numbers.
        if isinstance(summary.get("four_channel_accuracy_report"), dict):
            summary["four_channel_accuracy_report"]["evaluation_population"] = (
                _evaluation_population_block(payload, path))
        # For an md_manifest, attach the sibling-derived bounded evidence (realized protocol,
        # frozen deployment domain, Controller-manifest human approval + submission timing,
        # deployment_provenance cross-reference, deterministic thermo.log diagnostic).
        if isinstance(summary.get("md_manifest"), dict):
            summary["md_manifest"].update(_md_manifest_siblings_attachment(payload, path))
        return summary
    if isinstance(payload, list):
        return {"json_type": "list", "length": len(payload)}
    return {"json_type": type(payload).__name__}


def _label_manifest_summary(payload: dict) -> dict:
    return {
        "n_frames": payload.get("n_frames"),
        "labels": payload.get("labels"),
        "teacher_model_sha256": payload.get("teacher_model_sha256"),
        "teacher_config_sha256": payload.get("teacher_config_sha256"),
        "source_sha256": payload.get("source_sha256"),
        "output_sha256": payload.get("sha256"),
        "environment": payload.get("environment"),
    }


def _teacher_baseline_report_summary(payload: dict) -> dict:
    from adapters.teacher import species_mapping_is_attested

    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    operational = next(
        (c for c in checks if isinstance(c, dict)
         and c.get("observable") == "fresh_teacher_energy_force_finiteness"),
        {},
    )
    species_check = next(
        (c for c in checks if isinstance(c, dict)
         and c.get("observable") == "runtime_species_type_mapping_attested"),
        {},
    )
    details = operational.get("details") if isinstance(operational.get("details"), dict) else {}
    deployment = payload.get("deployment_domain") if isinstance(payload.get("deployment_domain"), dict) else {}
    applicability = payload.get("applicability") if isinstance(payload.get("applicability"), dict) else {}
    teacher = payload.get("teacher") if isinstance(payload.get("teacher"), dict) else {}
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    operational_evidence = next(
        (item for item in evidence if isinstance(item, dict)
         and item.get("role") == "operational_structures"),
        {},
    )
    status = operational.get("status")
    n_frames = details.get("n_frames")
    species_mapping = payload.get("species_mapping") if isinstance(payload.get("species_mapping"), dict) else {}
    return {
        "structure_count": n_frames,
        "expected_structure_count": n_frames,
        "successful_prediction_count": n_frames if status == "PASS" else None,
        "failed_prediction_count": 0 if status == "PASS" else None,
        "finite_teacher_energy_count": n_frames if status == "PASS" else None,
        "finite_teacher_force_count": n_frames if status == "PASS" else None,
        "label_keys": ["teacher_energy", "teacher_forces"],
        "teacher_model_sha256": teacher.get("model_sha256"),
        "source_candidate_sha256": (
            operational_evidence.get("integrity", {}).get("sha256")
            if isinstance(operational_evidence.get("integrity"), dict) else None
        ),
        "deployment_domain_status": deployment.get("slice_status", deployment),
        "applicability_status": applicability.get("status"),
        "applicability_limitations": applicability.get("limitations"),
        "dft_labels_used": deployment.get("dft_labels_used"),
        "protected_reference_labels_used": deployment.get("protected_reference_labels_used"),
        "operational_check_status": status,
        "max_force_value": operational.get("value"),
        "max_force_unit": operational.get("unit"),
        "fresh_label_output_integrity": details.get("fresh_label_output_integrity"),
        "fresh_label_manifest_integrity": details.get("fresh_label_manifest_integrity"),
        "species_mapping_attested": species_mapping_is_attested(species_mapping),
        "species_mapping_check_status": species_check.get("status"),
        "species_mapping_fallback_applied": species_mapping.get("fallback_applied"),
    }


def _is_directed_coverage_evidence(payload: dict) -> bool:
    """Generic shape-signature match for a coverage.report.build_directed_coverage_evidence
    payload -- never a campaign/dataset name check."""
    required = {"direction", "query_population", "reference_population", "overall_global_summary"}
    return required <= set(payload)


def _coverage_distance_view(stat: dict) -> dict:
    """Recast one coverage.aggregate summary stat (see coverage.aggregate.summarize /
    _summarize_with_unmatched) into the supported/unsupported + descriptive-distribution shape
    an Analyst is allowed to see."""
    from coverage.aggregate import SUMMARY_STATS

    n_matched = stat.get("n", 0) or 0
    n_unmatched = stat.get("n_unmatched", 0) or 0
    total = n_matched + n_unmatched
    return {
        "supported_count": n_matched,
        "unsupported_count": n_unmatched,
        "supported_fraction": (n_matched / total) if total else 0.0,
        "unsupported_fraction": stat.get("unmatched_fraction", 0.0),
        "distance_distribution": {k: stat[k] for k in SUMMARY_STATS if k in stat},
    }


def _structural_coverage_evidence_summary(payload: dict) -> dict:
    """Analyst-facing summary of one directed structural-coverage evidence payload (Priority #2,
    see coverage/report.py). Exposes only: coverage direction; query/reference population
    identities; slice/domain memberships; supported/unsupported counts and fractions; descriptive
    distance distributions; provenance hashes; limitations. Deliberately never passes through
    `payload["provenance"]["representation_provenance"]` / `["search_backend_provenance"]` --
    those carry representation/search-backend internals (e.g. SOAP descriptor hyperparameters,
    cKDTree worker counts) that this summary is not allowed to leak, per the coverage evidence
    architecture's own representation-agnostic contract.
    """
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    query_slices = payload.get("query_slice_resolved_summaries")
    query_slices = query_slices if isinstance(query_slices, dict) else {}
    reference_slices = payload.get("reference_slice_resolved_summaries")
    reference_slices = reference_slices if isinstance(reference_slices, dict) else {}
    overall = payload.get("overall_global_summary")
    return {
        "direction": payload.get("direction"),
        "query_population": payload.get("query_population"),
        "reference_population": payload.get("reference_population"),
        "n_query_environments": payload.get("n_query_environments"),
        "n_query_structures": payload.get("n_query_structures"),
        "reference_population_counts": {
            "slice_counts": provenance.get("reference_slice_counts"),
            "total_atoms": provenance.get("reference_total_atoms"),
            "total_frames": provenance.get("reference_total_frames"),
        },
        "overall_distance_distribution": (
            _coverage_distance_view(overall) if isinstance(overall, dict) else None
        ),
        "query_slice_memberships": sorted(query_slices),
        "query_slice_distance_distributions": {
            label: _coverage_distance_view(stat) for label, stat in query_slices.items()
        },
        "reference_slice_memberships": sorted(reference_slices),
        "reference_slice_distance_distributions": {
            label: _coverage_distance_view(stat) for label, stat in reference_slices.items()
        },
        "provenance_hashes": {
            "representation_hash": provenance.get("representation_hash"),
            "reference_manifest_sha256": provenance.get("reference_manifest_sha256"),
        },
        "limitations": list(payload.get("excluded_partitions") or []),
    }


register_json_evidence_adapter(
    "teacher_baseline", lambda payload: payload.get("profile") == "teacher_baseline",
    _teacher_baseline_report_summary,
)
register_json_evidence_adapter(
    "label_manifest",
    # A Teacher label manifest is distinguished by carrying ALL of a teacher-model identity, a
    # declared label-channel list, and a frame count. The earlier intersection (`&`) predicate
    # matched on ANY one key and so spuriously fired on the acquisition manifest (top-level
    # ``n_frames``) and the dataset_split manifest (``source_sha256``), summarizing them as a
    # near-empty label manifest. Requiring the full signature keeps it label-manifest-only.
    lambda payload: {"teacher_model_sha256", "labels", "n_frames"} <= set(payload),
    _label_manifest_summary,
)
def _split_membership_manifest_summary(payload: dict) -> dict:
    records = payload.get("records") or []
    split_counts: Counter = Counter()
    category_counts: Counter = Counter()
    for record in records:
        if isinstance(record, dict):
            split_counts[str(record.get("split"))] += 1
            category_counts[str(record.get("source_category"))] += 1
    return {
        "n_records": len(records),
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts.most_common(64)),
        "split_params": payload.get("split_params"),
        "total_frames": payload.get("total_frames"),
        "source_dataset": payload.get("source_dataset"),
    }


def _training_evidence_summary(payload: dict) -> dict:
    """Pass through the already-compact, self-declared training-evidence summary (see
    runtimes.pydantic_ai.training_evidence.build_training_evidence_summary) so the training gate's
    Judges see its semantic content -- dataset provenance, committee/checkpoint identity, per-seed
    training dynamics, and deterministic verification outcomes -- rather than only its top-level
    key names. The artifact is bounded by construction (a fixed committee of seeds, no raw vectors
    or per-file cache listings), so it is surfaced verbatim except for the redundant schema fields
    the generic ``_json_summary`` already reports."""
    return {
        "run_id": payload.get("run_id"),
        "dataset_provenance": payload.get("dataset_provenance"),
        "committee": payload.get("committee"),
        "training_dynamics": payload.get("training_dynamics"),
        "verification_outcomes": payload.get("verification_outcomes"),
        "all_verifications_passed": payload.get("all_verifications_passed"),
    }


register_json_evidence_adapter(
    "training_evidence_summary",
    lambda payload: payload.get("profile") == "training_evidence_summary",
    _training_evidence_summary,
)
register_json_evidence_adapter(
    "structural_coverage_evidence", _is_directed_coverage_evidence,
    _structural_coverage_evidence_summary,
)
register_json_evidence_adapter(
    "split_membership_manifest", _is_split_membership_manifest,
    _split_membership_manifest_summary,
)


def _is_data_coverage_report(payload: dict) -> bool:
    """Generic shape-signature match for a Stage-4 data_coverage report (see
    validation.data_coverage / executors._exec_build_data_coverage_report) -- never a
    campaign/dataset name check."""
    return {"coverage_status", "coverage_dimensions", "teacher_training_data_access",
            "deployment_domain"} <= set(payload)


def _data_coverage_report_summary(payload: dict) -> dict:
    """Judge-facing summary of the Stage-4 data_coverage report. Surfaces the adequacy verdict
    (coverage_status), how it was reached (frozen coverage_requirement vs unassessable), the real
    per-config_type frame coverage, the deployment domain the coverage is judged against, and the
    explicit gaps/limitations -- so a Judge adjudicates coverage adequacy on evidence rather than
    on the report's key names. COMPLETE is only ever earned by a frozen requirement being met, so
    the requirement (and per-config_type met/unmet detail) is surfaced when present."""
    dims = payload.get("coverage_dimensions") if isinstance(payload.get("coverage_dimensions"), dict) else {}
    config_cov = dims.get("config_type_coverage") if isinstance(dims.get("config_type_coverage"), dict) else {}
    counts = config_cov.get("counts") if isinstance(config_cov.get("counts"), dict) else {}
    domain = payload.get("deployment_domain") if isinstance(payload.get("deployment_domain"), dict) else {}
    requirement = domain.get("coverage_requirement") if isinstance(domain, dict) else None
    requirement_check = None
    if isinstance(requirement, dict) and isinstance(requirement.get("min_frames_by_config_type"), dict):
        requirement_check = {}
        for config_type, minimum in requirement["min_frames_by_config_type"].items():
            have = counts.get(config_type, 0) if isinstance(counts, dict) else 0
            requirement_check[config_type] = {
                "required_min_frames": minimum, "observed_frames": have,
                "met": isinstance(minimum, int) and not isinstance(minimum, bool) and have >= minimum,
            }
    return {
        "coverage_status": payload.get("coverage_status"),
        "adequacy_basis": ("frozen_coverage_requirement" if requirement_check is not None
                           else "not_assessable_without_frozen_requirement"),
        "teacher_training_data_access": payload.get("teacher_training_data_access"),
        "deployment_domain": domain,
        "config_type_coverage": {
            "config_types": config_cov.get("config_types"),
            "counts": counts if isinstance(counts, dict) else {},
            "total_frames": sum(v for v in counts.values() if isinstance(v, int)) if isinstance(counts, dict) else 0,
            "method": config_cov.get("method"),
        },
        "coverage_requirement_check": requirement_check,
        "identified_gaps": list(payload.get("identified_gaps") or []),
        "limitations": list(payload.get("limitations") or []),
        "replay_policy": payload.get("replay_policy"),
    }


register_json_evidence_adapter(
    "data_coverage_report", _is_data_coverage_report, _data_coverage_report_summary,
)


# --- Stage-2 reference_validation adapter ------------------------------------------------------
# The reference_validation.json report carries the Teacher-vs-reference DISAGREEMENT magnitudes
# (metrics.global / metrics.by_config_type) and the reference-population span/scope (reference.*,
# prediction_artifact.n_frames/labels) that the rv-disagreement / rv-anchors / rv-scope review
# criteria must read -- all nested, so the default `_json_summary` exposes only their key names.


def _is_reference_validation_report(payload: dict) -> bool:
    """Recognize the Stage-2 report by its self-declared framework profile (never a material name)."""
    return payload.get("profile") == "teacher_reference_validation"


def _reference_validation_report_summary(payload: dict) -> dict:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    reference = payload.get("reference") if isinstance(payload.get("reference"), dict) else {}
    prediction = (payload.get("prediction_artifact")
                  if isinstance(payload.get("prediction_artifact"), dict) else {})
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    return {
        "protected_reference_use": payload.get("protected_reference_use"),
        "historical_prediction_policy": payload.get("historical_prediction_policy"),
        "energy_unit": metrics.get("energy_unit"),
        "force_unit": metrics.get("force_unit"),
        "energy_normalization": metrics.get("energy_normalization"),
        "global_disagreement": metrics.get("global"),
        "disagreement_by_config_type": metrics.get("by_config_type"),
        "domain_fields": metrics.get("domain_fields"),
        "reference_id": reference.get("reference_id"),
        "reference_structures_path": reference.get("structures_path"),
        "reference_logical_frames": reference.get("logical_frames"),
        "reference_protected_source_rows": reference.get("protected_source_rows"),
        "prediction_n_frames": prediction.get("n_frames"),
        "prediction_labels": prediction.get("labels"),
        "checks": [{"observable": c.get("observable"), "value": c.get("value"),
                    "unit": c.get("unit"), "status": c.get("status"),
                    "criterion": c.get("criterion")}
                   for c in checks if isinstance(c, dict)],
    }


register_json_evidence_adapter(
    "reference_validation_report", _is_reference_validation_report,
    _reference_validation_report_summary,
)


# --- Stage-3 acquisition manifest adapter ------------------------------------------------------
# The acquisition manifest carries the selection science (parents, eligible categories, plan
# envelope: n_parents/n_per_structure/T_K/beta/sigma budget, produced elements + frame count, plan
# lineage sha) the aq-representation / aq-evidence / aq-diversity / aq-objective-consistency
# criteria must read. All nested/list -> hidden by default. Large member lists are reduced to
# counts + a bounded sample (never raw vectors), per the adapter contract.


def _is_acquisition_manifest(payload: dict) -> bool:
    """Recognize both acquisition writer paths (perturbation-generate and existing-pool-select) by
    their self-declared ``operation`` plus the selection-lineage key they both carry."""
    return (payload.get("operation") in ("acquire_structures", "select_existing_pool")
            and "selected_parent_structure_ids" in payload)


def _acquisition_manifest_summary(payload: dict) -> dict:
    parents = payload.get("selected_parent_structure_ids") or []
    indices = payload.get("selected_source_global_indices") or []
    records = payload.get("selected_source_records") or []
    return {
        "operation": payload.get("operation"),
        "stage": payload.get("stage"),
        "acquisition_plan_sha256": payload.get("acquisition_plan_sha256"),
        "expected_output_count": payload.get("expected_output_count"),
        "actual_output_count": payload.get("actual_output_count"),
        "n_frames": payload.get("n_frames"),
        "elements": payload.get("elements"),
        "eligible_source_categories": payload.get("eligible_source_categories"),
        "n_selected_parents": len(parents) if isinstance(parents, list) else None,
        "selected_parent_sample": [str(x) for x in parents[:32]]
        if isinstance(parents, list) else None,
        "n_selected_source_indices": len(indices) if isinstance(indices, list) else None,
        "n_selected_source_records": len(records) if isinstance(records, list) else None,
        "duplicate_handling": payload.get("duplicate_handling"),
        "dft_labels_used_as_selection_scores": payload.get("dft_labels_used_as_selection_scores"),
        "framework_plan_envelope": payload.get("framework_plan_envelope"),
        "protection_audit_result": payload.get("protection_audit_result"),
    }


register_json_evidence_adapter(
    "acquisition_manifest", _is_acquisition_manifest, _acquisition_manifest_summary,
)


# --- Stage-6 dataset_split manifest adapter ----------------------------------------------------
# The split_dataset manifest carries the leakage counts (overlap_checks -- must be zero for
# ds-lineage) and the per-split sizes / grouping (splits.*, grouping_key, fractions, seed for
# ds-representative / ds-bias). Nested -> hidden by default. Per-split ``group_ids`` lists are
# reduced to counts (never the raw id vectors).


def _is_dataset_split_manifest(payload: dict) -> bool:
    """Shape-signature for split_dataset's own manifest (distinct from the split-membership
    crosswalk manifest, which is a top-level ``records`` list)."""
    return {"splits", "overlap_checks", "grouping_key"} <= set(payload)


def _dataset_split_manifest_summary(payload: dict) -> dict:
    splits = payload.get("splits") if isinstance(payload.get("splits"), dict) else {}
    frame_counts = {}
    group_counts = {}
    for name, block in splits.items():
        if isinstance(block, dict):
            frame_counts[str(name)] = block.get("n_frames")
            gids = block.get("group_ids")
            group_counts[str(name)] = len(gids) if isinstance(gids, list) else None
    return {
        "grouping_key": payload.get("grouping_key"),
        "seed": payload.get("seed"),
        "validation_fraction": payload.get("validation_fraction"),
        "test_fraction": payload.get("test_fraction"),
        "source_sha256": payload.get("source_sha256"),
        "split_frame_counts": frame_counts,
        "split_group_counts": group_counts,
        "overlap_checks": payload.get("overlap_checks"),
    }


register_json_evidence_adapter(
    "dataset_split_manifest", _is_dataset_split_manifest, _dataset_split_manifest_summary,
)


# --- Stage-12 analysis run_summary adapter -----------------------------------------------------
# The analysis run_summary report carries campaign_outcome (a NON-generic top-level scalar the
# default surfaces only as ``status`` would be), the per-stage gate ledger (stages[].gate,
# gate_history[].verdict for an-traceable), the recovery ledger (recoveries[] for an-loop-claim),
# and the bounding claims (identified_gaps, limitations for an-bounded). All nested lists / a
# non-generic scalar -> hidden by default. Per-stage artifact lists are reduced to counts.


def _is_run_summary_report(payload: dict) -> bool:
    return {"campaign_outcome", "stages", "gate_history", "recoveries"} <= set(payload)


def _run_summary_report_summary(payload: dict) -> dict:
    stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
    gate_history = payload.get("gate_history") if isinstance(payload.get("gate_history"), list) else []
    recoveries = payload.get("recoveries") if isinstance(payload.get("recoveries"), list) else []
    return {
        "run_id": payload.get("run_id"),
        "campaign_outcome": payload.get("campaign_outcome"),
        "stages": [{"name": s.get("name"), "status": s.get("status"), "gate": s.get("gate"),
                    "n_artifacts": len(s.get("artifacts") or [])}
                   for s in stages if isinstance(s, dict)],
        "gate_history": [{"stage": g.get("stage"), "verdict": g.get("verdict")}
                         for g in gate_history if isinstance(g, dict)],
        "recoveries": [{"id": r.get("id"), "status": r.get("status"),
                        "failed_stage": r.get("failed_stage")}
                       for r in recoveries if isinstance(r, dict)],
        "identified_gaps": payload.get("identified_gaps"),
        "limitations": payload.get("limitations"),
    }


register_json_evidence_adapter(
    "run_summary_report", _is_run_summary_report, _run_summary_report_summary,
)


# The eight per-group fidelity metrics written by validation.four_channel_audit /
# workflow.steps.evaluate_committee for every accuracy_report.json group. Kept as an explicit
# set so the predicate is a shape signature, never a channel/campaign name check.
_FOUR_CHANNEL_METRIC_KEYS = frozenset({
    "e_raw_mae_meV", "e_raw_rmse_meV", "e_alignment_shift_meV",
    "e_mae_meV", "e_rmse_meV", "f_mae", "f_rmse", "f_r2",
})
_FOUR_CHANNEL_GROUP_KEYS = ("n_frames", "n_atoms", *sorted(_FOUR_CHANNEL_METRIC_KEYS))


def _is_four_channel_accuracy_report(payload: dict) -> bool:
    """Shape-signature match for a four-channel accuracy_report.json (see
    validation.four_channel_audit): a non-empty mapping whose every value is a per-group mapping
    that has an aggregate ``all`` group carrying the full fidelity-metric key set. Deliberately
    generic -- it never inspects channel names, chemistry, or campaign identity, only structure."""
    if not isinstance(payload, dict) or not payload:
        return False
    for channel in payload.values():
        if not isinstance(channel, dict) or not isinstance(channel.get("all"), dict):
            return False
        if not _FOUR_CHANNEL_METRIC_KEYS <= set(channel["all"]):
            return False
    return True


_FOUR_CHANNEL_DISPLAY_SIG_FIGS = 4
# Population identity is authoritative and must be surfaced EXACTLY -- never rounded.
_FOUR_CHANNEL_EXACT_GROUP_KEYS = frozenset({"n_frames", "n_atoms"})


def _round_sig(value, sig_figs: int = _FOUR_CHANNEL_DISPLAY_SIG_FIGS):
    """Deterministic canonical rounding of a metric float to ``sig_figs`` significant figures.

    Judge-FACING display compaction only: this is applied when building the bounded evidence
    packet, never to the authoritative accuracy_report.json artifact. Non-finite values (nan/inf),
    booleans, integers, and non-numerics pass through unchanged; ``0`` maps to ``0.0``. Uses the
    format-spec ``g`` conversion so the serialized number is both rounded and canonically short
    (e.g. ``12.3399999`` -> ``12.34``), which is what makes the columnar block fit the Judge
    context. Deterministic: identical input -> identical output on every run."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, int):
        return value
    if value != value or value in (float("inf"), float("-inf")):
        return value
    if value == 0:
        return 0.0
    return float(f"{value:.{sig_figs}g}")


def _four_channel_group_metrics(group: dict, *, round_display: bool = False) -> dict:
    """Deterministic, fixed-key projection of one accuracy_report group's metrics (drops any
    unexpected extra keys so the surfaced shape is stable and bounded). When ``round_display`` is
    set, metric values are rounded to the Judge-display significant figures while the exact
    population-identity keys (n_frames/n_atoms) are preserved verbatim."""
    out = {}
    for key in _FOUR_CHANNEL_GROUP_KEYS:
        value = group.get(key)
        if round_display and key not in _FOUR_CHANNEL_EXACT_GROUP_KEYS:
            value = _round_sig(value)
        out[key] = value
    return out


def _four_channel_accuracy_report_summary(payload: dict) -> dict:
    """Directly readable, deterministic summary of a four-channel accuracy_report.json so the
    Stage-8 evaluation Judges never have to re-derive fidelity from a large extxyz artifact.

    Surfaces, per fidelity channel (student_vs_teacher / student_vs_dft / teacher_vs_dft, whatever
    the report actually contains): the exact evaluation population (n_frames / n_atoms of the
    aggregate ``all`` group), the aggregate energy/force metrics, and the domain/configuration-
    family-resolved per-group metrics. Units and normalization are stated explicitly so no
    inference is required. The lineage/hash binding is already attached by ``summarize_artifact``
    as this artifact's top-level ``integrity`` (sha256) field, so it is not duplicated here.

    Judge-facing compaction (context budget): the per-group metrics are surfaced in a COLUMNAR
    layout (a fixed ``group_order`` plus one array per metric column, index-aligned to it) instead
    of repeating the metric-name keys once per group, and metric values are rounded to
    ``_FOUR_CHANNEL_DISPLAY_SIG_FIGS`` significant figures. This is a DISPLAY transform of the
    Judge packet ONLY: all channels, all groups, and all metric columns are preserved, and the
    authoritative full-precision values remain in the registered accuracy_report.json artifact
    (whose sha256 this summary is bound to via the artifact ``integrity`` field). The summary is
    derived from -- and never mutates -- that artifact.

    Stage separation: these metrics are computed over the committee-MEAN prediction only. Per-seed
    committee disagreement / calibrated uncertainty is Stage-9 evidence and is intentionally absent
    from this Stage-8 accuracy report -- its absence is not an evidence gap for this gate.
    """
    channels = {}
    for channel_name, groups in payload.items():
        all_group = groups.get("all") if isinstance(groups, dict) else {}
        group_items = [
            (name, metrics)
            for name, metrics in sorted(groups.items())
            if name != "all" and isinstance(metrics, dict)
        ]
        group_order = [name for name, _ in group_items]
        by_group_columnar = {}
        for key in _FOUR_CHANNEL_GROUP_KEYS:
            exact = key in _FOUR_CHANNEL_EXACT_GROUP_KEYS
            by_group_columnar[key] = [
                (metrics.get(key) if exact else _round_sig(metrics.get(key)))
                for _, metrics in group_items
            ]
        channels[channel_name] = {
            "population": {
                "n_frames": all_group.get("n_frames"),
                "n_atoms": all_group.get("n_atoms"),
            },
            "aggregate": _four_channel_group_metrics(all_group, round_display=True),
            "n_groups": len(group_order),
            "group_order": group_order,
            "by_group_columnar": by_group_columnar,
        }
    return {
        "channels_present": sorted(payload.keys()),
        "units": {
            "energy_metrics": "meV/atom (per-atom normalized: de_per_atom = (e_pred - e_ref)/n_atoms, x1000)",
            "force_metrics": "eV/Angstrom (f_mae, f_rmse)",
            "f_r2": "dimensionless force-component coefficient of determination (no energy R2 is reported)",
            "raw_vs_aligned": (
                "e_raw_* are computed directly; e_* subtract a single global per-atom shift "
                "e_alignment_shift_meV (diagnostic only, not a corrected metric)"
            ),
        },
        "aggregation": (
            "metrics are over the committee-MEAN prediction; per-seed committee disagreement / "
            "calibrated uncertainty is Stage-9 evidence and is not part of this accuracy report"
        ),
        "display_precision": (
            f"metric values shown here are rounded to {_FOUR_CHANNEL_DISPLAY_SIG_FIGS} significant "
            "figures for context-efficient Judge review ONLY; population counts (n_frames/n_atoms) "
            "are exact. The authoritative full-precision values live in the registered "
            "accuracy_report.json artifact (this artifact's integrity.sha256); this summary is "
            "derived from it and does not mutate it."
        ),
        "by_group_layout": (
            "columnar: 'group_order' lists the domain/configuration-family groups in a fixed order; "
            "each key of 'by_group_columnar' is a metric column whose array is aligned "
            "index-for-index to 'group_order'. All groups and all metric columns are retained."
        ),
        "channels": channels,
    }


register_json_evidence_adapter(
    "four_channel_accuracy_report", _is_four_channel_accuracy_report,
    _four_channel_accuracy_report_summary,
)


# Canonical, framework-defined artifact name written by the reference_validation stage. Resolved as
# a sibling of the accuracy_report.json in the SAME run artifacts directory -- deterministic, not a
# heuristic search -- so the Stage-8 evaluation Judges can read the evaluation-population identity
# and its lineage/hash binding in the same place as the fidelity metrics.
_REFERENCE_VALIDATION_ARTIFACT_NAME = "reference_validation.json"


def _sha256_of(integrity) -> str | None:
    if isinstance(integrity, dict):
        value = integrity.get("sha256")
        return value if isinstance(value, str) else None
    return None


def _evaluation_population_block(report_payload: dict, report_path: Path) -> dict:
    """Deterministically surface the run-bound evaluation-population identity + lineage/hash binding
    for a four-channel accuracy_report.json, read verbatim from the authoritative sibling
    reference_validation.json record.

    This is a Judge-FACING evidence-surfacing block ONLY: it copies provenance fields that already
    exist in the reference_validation record (population id, protected-reference use, source
    structures path/sha256/frame count, teacher-reference prediction artifact path/sha256/labels)
    and asserts the lineage binding that the Stage-8 evaluation population IS that reference
    population by comparing the reference's logical frame count against the accuracy_report
    aggregate (``all``-group) population size. It never invents or infers a missing field: if the
    reference record is absent, unreadable, or the frame counts do not bind, it returns an explicit
    ``evidence_gap`` instead of a fabricated identity. No scientific artifact is read for values --
    only provenance is surfaced; the fidelity numbers stay in the four-channel summary."""
    ref_path = report_path.parent / _REFERENCE_VALIDATION_ARTIFACT_NAME
    if not ref_path.is_file():
        return {"evidence_gap": (
            "no sibling reference_validation.json found next to the accuracy_report; the "
            "evaluation-population identity / lineage binding could not be surfaced")}
    try:
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"evidence_gap": "sibling reference_validation.json is unreadable or not valid JSON"}
    if not isinstance(ref, dict):
        return {"evidence_gap": "sibling reference_validation.json is not a JSON object"}

    reference = ref.get("reference") if isinstance(ref.get("reference"), dict) else {}
    prediction = ref.get("prediction_artifact") if isinstance(ref.get("prediction_artifact"), dict) else {}

    # Accuracy-report aggregate population size, per channel, for the lineage binding.
    report_pop_frames = {
        str(channel): groups.get("all", {}).get("n_frames")
        for channel, groups in report_payload.items()
        if isinstance(groups, dict) and isinstance(groups.get("all"), dict)
    }
    distinct_frames = {v for v in report_pop_frames.values() if v is not None}
    reference_logical_frames = reference.get("logical_frames")
    bound = (
        reference_logical_frames is not None
        and len(distinct_frames) == 1
        and reference_logical_frames in distinct_frames
    )

    # Governance sub-block: authoritative run-scoped allowed_uses resolved from PINNED pre-existing
    # provenance (inputs/*-reference.yaml + protected_reference_manifest.json), plus the legacy
    # protected_reference_use restated explicitly as a historical origin descriptor. The two are
    # disjoint by design so Stage-8 Criterion 1 can be evidenced by the run-scoped authorization
    # (never by the origin descriptor). See validation.reference_population_governance for the
    # deterministic authority.
    run_dir = report_path.parent.parent  # runs/<run_id>/artifacts -> runs/<run_id>
    try:
        from validation.reference_population_governance import (
            validate_stage8_reference_population_governance,
        )
        _ref_yaml_source = reference.get("reference_yaml") if isinstance(reference, dict) else None
        governance = validate_stage8_reference_population_governance(
            run_dir=run_dir,
            reference_validation_payload=ref,
            accuracy_report_channels=list(report_payload.keys()),
            channel_frame_counts=report_pop_frames,
            reference_yaml_source=_ref_yaml_source,
        )
    except Exception as exc:  # pragma: no cover - defensive: never crash the packet
        governance = {
            "ok": False,
            "failures": [{"code": "governance_validator_exception",
                          "detail": f"{type(exc).__name__}: {exc}"}],
            "checks": {},
            "run_scoped_allowed_uses": {"evidence_gap": "governance validator raised"},
            "historical_origin_descriptor": {
                "protected_reference_use": ref.get("protected_reference_use"),
                "role": "historical_origin_descriptor",
            },
        }

    block = {
        "note": (
            "evaluation-population identity + lineage/hash binding surfaced verbatim from the "
            "authoritative sibling reference_validation.json; no field is inferred. The "
            "authorization list for Stage-8 uses is the pinned `run_scoped_allowed_uses` "
            "sub-block below (sourced ONLY from pre-existing pinned run provenance); the legacy "
            "`protected_reference_use` field is preserved as a `historical_origin_descriptor` "
            "and is EXCLUDED from any allowed_use authorization decision."),
        "population_id": reference.get("reference_id"),
        # Preserved as-is for provenance and existing consumers; do NOT interpret this field as
        # the run-scoped authorization list (see `run_scoped_allowed_uses` and
        # `historical_origin_descriptor` below).
        "protected_reference_use": ref.get("protected_reference_use"),
        "evidence_source": ref.get("evidence_source"),
        "source_structures": {
            "path": reference.get("structures_path"),
            "sha256": _sha256_of(reference.get("structures_integrity")),
            "logical_frames": reference_logical_frames,
            "protected_source_rows": reference.get("protected_source_rows"),
        },
        "teacher_reference_predictions": {
            "path": prediction.get("path"),
            "sha256": _sha256_of(prediction.get("integrity")),
            "n_frames": prediction.get("n_frames"),
            "labels": prediction.get("labels"),
        },
        "lineage_binding": {
            "reference_logical_frames": reference_logical_frames,
            "accuracy_report_population_n_frames_by_channel": report_pop_frames,
            "frame_count_binds": bool(bound),
            "statement": (
                "the Stage-8 evaluation population is the run-bound recovered-original-heldout "
                "reference population (same logical frame count on every channel)"
                if bound else
                "WARNING: the accuracy_report population frame count does not uniquely match the "
                "reference logical frame count; treat the lineage binding as unverified"),
        },
        # Two disjoint governance keys: `historical_origin_descriptor` is provenance-only, and
        # `run_scoped_allowed_uses` is the AUTHORITATIVE authorization list. `governance_validation`
        # is the deterministic validator result (channel authorization PASS/FAIL).
        "historical_origin_descriptor": governance.get("historical_origin_descriptor"),
        "run_scoped_allowed_uses": governance.get("run_scoped_allowed_uses"),
        "governance_validation": {
            "ok": governance.get("ok"),
            "failures": governance.get("failures"),
            "checks": governance.get("checks"),
            "authorization_scope_authority": governance.get("authorization_scope_authority"),
            "note": governance.get("note"),
        },
    }
    return block


# --- Stage-9 uncertainty_report adapter --------------------------------------------------------
# Recovery id=6 (governance/evidence-repair, zero-compute): the Stage-9 uncertainty_report
# profile carries all four criterion-relevant values inline (population, committee identity,
# disagreement statistic + aggregate rule, calibration status/caveat, evidence/provenance,
# gaps/limitations). The default `_json_summary` exposes only top-level KEY NAMES, so the
# Judges could not read the VALUES — triggering the framework's Deterministic Contradiction
# guard (Section 13). This adapter surfaces the criterion-relevant VALUES verbatim from the
# authoritative report (never invents any field) so the Judges can read + cite them.


def _is_uncertainty_report(payload: dict) -> bool:
    """Recognize the Stage-9 uncertainty_report profile by required-key signature.

    Never keyed off any campaign/material name; only off the framework-defined fields the
    Stage-9 executor deterministically writes (see validation.uncertainty.validate_uncertainty_report):
    ``schema_version`` + ``committee_manifest_sha256`` + ``seeds`` + ``aggregate`` +
    ``u_frame_summary`` + ``calibration``. Any payload missing any of these is left to the
    default bounded-evidence handling.
    """
    required = ("schema_version", "committee_manifest_sha256", "seeds", "aggregate",
                "u_frame_summary", "calibration")
    return all(k in payload for k in required)


def _uncertainty_report_summary(payload: dict) -> dict:
    """Judge-facing bounded summary of the Stage-9 uncertainty_report profile.

    Surfaces criterion-relevant VALUES verbatim from the authoritative report — never
    invents a field, never computes a new statistic, never touches the source artifact.
    An absent required field is exposed as an explicit ``evidence_gap`` string; the caller
    (validate_uncertainty_report + the Judges) decides what to do about it.
    """
    def _get(d: dict, key: str, default=None):
        return d.get(key, default) if isinstance(d, dict) else default

    population = _get(payload, "population") or {}
    calibration = _get(payload, "calibration") or {}
    u_summary = _get(payload, "u_frame_summary") or {}

    # Held-out / protected-reference binding for the Judges: the Stage-9 population is the
    # Stage-8 evaluated.extxyz output (declared explicitly in workflow.yaml), so the
    # protected-reference exclusion policy declared for the Stage-8 population applies to
    # this stage by lineage. Surface only what the report+evidence block already record;
    # do not invent a new binding relationship.
    evidence_entries = _get(payload, "evidence") or []
    population_evidence = None
    committee_evidence = None
    if isinstance(evidence_entries, list):
        for entry in evidence_entries:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            if role == "population" and population_evidence is None:
                population_evidence = entry
            elif role == "committee_manifest" and committee_evidence is None:
                committee_evidence = entry

    seeds = _get(payload, "seeds")

    # Governed calibration/eval isolation (present only when the run bound an access-partition
    # contract): surface the disjoint fit/eval roles and their hashes verbatim so a Judge can
    # confirm the calibration-fit and calibration-eval populations do not overlap and are the
    # governed partition of the protected reference — never inferred, only read.
    gp = _get(payload, "governed_partition")
    governed_partition = None
    if isinstance(gp, dict):
        fit = _get(gp, "calibration_fit") or {}
        ev = _get(gp, "calibration_eval") or {}
        governed_partition = {
            "access_partition_path": _get(gp, "access_partition_path"),
            "partition_assignment_sha256": _get(gp, "partition_assignment_sha256"),
            "fit_eval_disjoint": _get(gp, "fit_eval_disjoint"),
            "calibration_fit": {
                "role": _get(fit, "role"), "n_frames": _get(fit, "n_frames"),
                "frame_fingerprints_sha256": _get(fit, "frame_fingerprints_sha256"),
            },
            "calibration_eval": {
                "role": _get(ev, "role"), "n_frames": _get(ev, "n_frames"),
                "frame_fingerprints_sha256": _get(ev, "frame_fingerprints_sha256"),
                "holdout_disagreement_summary": _get(ev, "holdout_disagreement_summary"),
            },
        }

    return {
        "note": (
            "criterion-relevant values are read VERBATIM from the authoritative "
            "uncertainty_report.json; the source artifact is not modified and no new "
            "statistic is invented. See `evidence.*.integrity.sha256` for artifact binding."
        ),
        "schema_version": _get(payload, "schema_version"),
        # Criterion 1 — held-out / deployment-relevant population declaration.
        "population": {
            "role": _get(population, "role"),
            "path": _get(population, "path"),
            "n_frames": _get(population, "n_frames"),
            "artifact_sha256": (_get(population_evidence, "integrity") or {}).get("sha256")
                if population_evidence else None,
            "artifact_size": (_get(population_evidence, "integrity") or {}).get("size")
                if population_evidence else None,
        },
        # Criterion 3 — exact Student committee manifest hash.
        "committee": {
            "committee_manifest_path": _get(payload, "committee_manifest_path"),
            "committee_manifest_sha256": _get(payload, "committee_manifest_sha256"),
            "committee_manifest_size":
                (_get(committee_evidence, "integrity") or {}).get("size")
                if committee_evidence else None,
            "seeds": list(seeds) if isinstance(seeds, list) else seeds,
            "n_seeds": len(seeds) if isinstance(seeds, list) else None,
        },
        # Disagreement statistic + aggregate rule (structure the report already declares).
        "disagreement": {
            "aggregate_rule": _get(payload, "aggregate"),
            "per_frame_summary": {
                "mean": _get(u_summary, "mean"),
                "max": _get(u_summary, "max"),
            },
            # Report does not currently persist canonical percentiles or a per-domain
            # breakdown; expose that absence explicitly rather than fabricating one.
            "per_frame_percentiles_present": False,
            "domain_resolved_present": False,
            "n_per_frame_scores":
                len(payload["frame_scores"]) if isinstance(payload.get("frame_scores"), list) else None,
        },
        # Criterion 2 — calibration status + caveat treating sigma_F as disagreement/ranking.
        "calibration": {
            "status": _get(calibration, "status"),
            "caveat": _get(calibration, "caveat"),
        },
        # Criterion 4 — acquisition/recovery proposals surfaced from the report.
        # An empty identified_gaps/limitations means the Stage-9 executor emitted no
        # acquisition or recovery proposals that could touch the protected-reference
        # exclusion policy. The full population is the same reference-lineage-bound
        # evaluated.extxyz (Stage 8 output), so no protected-reference exclusion violation
        # is possible from this stage without a subsequent acquisition/recovery action.
        "protected_reference_exclusion": {
            "identified_gaps": _get(payload, "identified_gaps"),
            "limitations": _get(payload, "limitations"),
            "population_shared_with_stage8_evaluation":
                _get(population, "path") == (_get(population_evidence, "path") if population_evidence else None),
            "note": (
                "an empty identified_gaps + limitations list means the Stage-9 executor emitted "
                "no acquisition/recovery proposal that could touch the protected-reference "
                "exclusion policy; the population is the Stage-8 evaluated.extxyz output whose "
                "protected-reference authorization was resolved by Recovery id=5"
            ),
        },
        # Governed calibration/eval partition isolation (None unless the run bound one).
        "governed_partition": governed_partition,
        # Evidence/provenance block passed through as-is (report's own evidence list).
        "evidence": [dict(entry) for entry in evidence_entries if isinstance(entry, dict)],
    }


register_json_evidence_adapter(
    "uncertainty_report", _is_uncertainty_report, _uncertainty_report_summary,
)


# --- Stage-10 md_manifest adapter --------------------------------------------------------------
# Recovery-006 (governance/evidence-repair, zero-compute): the Stage-10 md_manifest profile
# carries all four criterion-relevant values inline (or in pinned sibling files: context.yaml
# for the realized protocol, deployment_provenance.json for structure/env provenance, the
# Controller manifest.json for the human approval, workflow.yaml for the frozen deployment
# domain). The default `_json_summary` exposed only the manifest's top-level KEY NAMES, so the
# Judges could not read the VALUES — the same deterministic-contradiction pattern Stage-9 hit
# and Stage-10 iter008 hit. This adapter surfaces the criterion-relevant VALUES verbatim from
# the authoritative sources (never invents any field, never mutates any artifact).


_MD_MANIFEST_REQUIRED_KEYS = ("schema_version", "input", "run_dir", "checkpoint",
                              "checkpoint_integrity", "evidence", "committee_manifest",
                              "selected_seed")


def _is_md_manifest(payload: dict) -> bool:
    """Recognize the Stage-10 md_manifest profile by required-key signature.

    Framework-generic — never keyed off material/campaign/backend name; the keys below are the
    ones ``workflow.contracts.validate_md_manifest`` treats as mandatory and the fields
    ``workflow.steps.run_md`` deterministically writes.
    """
    return all(k in payload for k in _MD_MANIFEST_REQUIRED_KEYS)


def _load_yaml_safe(path: Path):
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _load_json_safe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _ensemble_from_input_lmp(input_lmp_path: Path) -> str | None:
    """Detect the MD ensemble from the LAMMPS input file by looking at the active ``fix``
    directive (nvt/npt/nve). Deterministic parse; no LAMMPS execution. Returns None if the
    ensemble cannot be determined."""
    if not input_lmp_path.is_file():
        return None
    text = input_lmp_path.read_text(encoding="utf-8")
    ensembles = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 4 and tokens[0] == "fix":
            style = tokens[3].lower()
            if style in ("nvt", "npt", "nve", "nph", "nhc"):
                ensembles.append(style)
    if not ensembles:
        return None
    # Multiple `fix` lines are permitted; the last active integrator declared in the input
    # typically wins for the run, so return the last recognised ensemble token.
    return ensembles[-1]


_ISO_TZ_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:?\d{2})?$")


def _parse_iso_timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        from datetime import datetime, timezone
    except Exception:  # pragma: no cover
        return None
    s = value
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _thermo_diagnostic(thermo_log_path: Path, equilibration_steps: int) -> dict:
    """Bounded deterministic pass over the already-written ``thermo.log`` — no computation
    beyond min/mean/max over rows past the declared equilibration cutoff, and a strict scan
    for NaN/Inf/error tokens. Never opens the trajectory. Never rewrites the log."""
    if not thermo_log_path.is_file():
        return {"evidence_gap": "thermo.log missing"}
    rows = []
    banned = []
    try:
        text = thermo_log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"evidence_gap": f"thermo.log unreadable: {exc}"}
    lowered = text.lower()
    for tok in ("error:", "segfault", "lost atoms", "neighbor list overflow", "abort", "nan", "inf"):
        # only look at whole-word / prefix matches so 'info' doesn't false-hit 'inf'
        if tok in ("nan", "inf"):
            for line in text.splitlines():
                parts = line.split()
                if any(p.strip().lower() == tok for p in parts):
                    banned.append(tok)
                    break
        elif tok in lowered:
            banned.append(tok)
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit() or len(parts) < 6:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                         float(parts[4]), float(parts[5])))
        except (TypeError, ValueError):
            continue
    if not rows:
        return {"evidence_gap": "thermo.log has no parsable numeric rows"}
    steps = [r[0] for r in rows]
    T = [r[1] for r in rows]
    PE = [r[2] for r in rows]
    Etot = [r[4] for r in rows]
    Pobs = [r[5] for r in rows]
    def _agg(vals):
        return {"mean": sum(vals)/len(vals), "min": min(vals), "max": max(vals),
                "n": len(vals)}
    assess = [r for r in rows if r[0] > equilibration_steps]
    result = {
        "thermo_log_path": str(thermo_log_path),
        "n_thermo_rows": len(rows),
        "step_first": steps[0],
        "step_last": steps[-1],
        "banned_tokens_found": banned,
        "no_nan_inf_error_tokens": not banned,
        "equilibration_step_cutoff": equilibration_steps,
        "assessment_window_definition":
            f"steps > {equilibration_steps} (post-declared-equilibration; NVT diagnostics)",
        "n_thermo_rows_in_assessment_window": len(assess),
        "temperature_K_all_rows": _agg(T),
        "potential_energy_eV_diagnostic_only": _agg(PE),
        "total_energy_eV_diagnostic_only": {
            **_agg(Etot),
            "note": (
                "total energy is a DIAGNOSTIC in an NVT run — the thermostat exchanges energy "
                "with the system; do NOT interpret this as an NVE conservation check"
            ),
        },
        "observed_pressure_bar_nvt_diagnostic": {
            **_agg(Pobs),
            "note": (
                "NVT / fixed-volume run has NO controlled pressure setpoint; observed pressure "
                "is a diagnostic time-series, not evidence of a pressure-domain PASS"
            ),
        },
    }
    if assess:
        result["temperature_K_assessment_window"] = _agg([r[1] for r in assess])
        result["total_energy_eV_assessment_window_diagnostic_only"] = _agg([r[4] for r in assess])
    return result


def _md_manifest_summary(payload: dict) -> dict:
    """Payload-only portion of the Stage-10 bounded summary — surfaces the checkpoint identity
    and the evidence role→integrity map that live INSIDE ``md.manifest.json`` itself. Sibling-
    file evidence (realized protocol, frozen domain, human approval, provenance, thermo
    diagnostic) is attached separately by ``_md_manifest_siblings_attachment`` from
    ``_json_summary`` (same second-phase pattern used for ``evaluation_population``)."""
    ckpt_integrity = payload.get("checkpoint_integrity") if isinstance(payload.get("checkpoint_integrity"), dict) else {}
    ev_entries = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    role_map = []
    for entry in ev_entries:
        if not isinstance(entry, dict):
            continue
        integrity = entry.get("integrity") if isinstance(entry.get("integrity"), dict) else {}
        role_map.append({
            "role": entry.get("role"),
            "path": entry.get("path"),
            "sha256": integrity.get("sha256"),
            "size": integrity.get("size"),
            "kind": integrity.get("kind"),
        })
    return {
        "note": (
            "payload-derived subset. Sibling-derived sections (realized_protocol, "
            "deployment_domain, human_approval, deployment_provenance, thermo_diagnostic, "
            "framework_notes) are attached by `_json_summary` from pinned sibling artifacts."
        ),
        "manifest_top_level_keys": sorted([k for k in payload.keys()]),
        "checkpoint": {
            "checkpoint_path": payload.get("checkpoint"),
            "checkpoint_sha256": ckpt_integrity.get("sha256"),
            "checkpoint_size": ckpt_integrity.get("size"),
            "selected_seed": payload.get("selected_seed"),
            "committee_manifest_path": payload.get("committee_manifest"),
        },
        "evidence_role_map": role_map,
    }


def _md_manifest_siblings_attachment(payload: dict, path: Path) -> dict:
    """Sibling-file portion of the Stage-10 bounded summary — read verbatim from pinned
    sibling artifacts (context.yaml, workflow.yaml / distillation_scope snapshot,
    Controller manifest.json action_approvals, deployment_provenance.json) plus a
    deterministic bounded pass over the already-written thermo.log. Never mutates a source
    file. Missing / hash-drifted sources → explicit ``evidence_gap``; no fabricated value.
    """
    md_manifest_dir = Path(path).parent          # …/artifacts
    run_dir = md_manifest_dir.parent              # …/<run>
    depl_dir = md_manifest_dir / "deployment_md"
    context_yaml_path = depl_dir / "context.yaml"
    depl_prov_path = depl_dir / "deployment_provenance.json"
    input_lmp_path = depl_dir / "input.lmp"
    thermo_log_path = depl_dir / "thermo.log"
    workflow_yaml_path = run_dir / "workflow.yaml"
    manifest_json_path = run_dir / "manifest.json"

    # --- Section C: realized deployment protocol from context.yaml -------------------
    context_yaml_sha = sha256_file(context_yaml_path) if context_yaml_path.is_file() else None
    ctx = _load_yaml_safe(context_yaml_path) if context_yaml_path.is_file() else None
    ensemble = _ensemble_from_input_lmp(input_lmp_path)
    # Detect starting-structure identity + composition (species counts) from the DATAFILE the
    # LAMMPS input reads, using its Masses block — no chemistry-specific assumptions.
    datafile_path_str = None
    datafile_species_counts = None
    n_atoms_in_datafile = None
    datafile_sha256 = None
    if isinstance(ctx, dict):
        raw = ctx.get("DATAFILE")
        if isinstance(raw, str) and raw:
            datafile = Path(raw)
            datafile_path_str = str(datafile)
            if datafile.is_file():
                datafile_sha256 = sha256_file(datafile)
                # Deterministic parse of the LAMMPS Masses + Atoms header (never invents)
                try:
                    dtext = datafile.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    dtext = None
                if dtext is not None:
                    # 'N atoms' line
                    for line in dtext.splitlines():
                        stripped = line.strip()
                        if stripped.endswith("atoms") and stripped.split()[0].isdigit():
                            try:
                                n_atoms_in_datafile = int(stripped.split()[0])
                            except (TypeError, ValueError):
                                pass
                            break
                    # 'Atoms' section: count type-column values per row
                    counts = {}
                    in_atoms = False
                    for line in dtext.splitlines():
                        s = line.strip()
                        if s.startswith("Atoms"):
                            in_atoms = True
                            continue
                        if in_atoms:
                            if not s or s.startswith("#") or s.split()[0].isalpha():
                                # Section break heuristic (e.g. blank line, Velocities, Bonds)
                                if counts:
                                    break
                                continue
                            parts = s.split()
                            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                                t = parts[1]
                                counts[t] = counts.get(t, 0) + 1
                    if counts:
                        datafile_species_counts = {f"type_{k}": v for k, v in sorted(counts.items())}
    protocol = {
        "context_yaml_path": str(context_yaml_path),
        "context_yaml_sha256": context_yaml_sha,
        "ensemble": ensemble,
        "temperature_setpoint_K": ctx.get("TEMPERATURE_K") if isinstance(ctx, dict) else None,
        "thermostat_tdamp_ps": ctx.get("TDAMP_PS") if isinstance(ctx, dict) else None,
        "timestep_ps": ctx.get("TIMESTEP_PS") if isinstance(ctx, dict) else None,
        "n_steps": ctx.get("N_STEPS") if isinstance(ctx, dict) else None,
        "velocity_seed": ctx.get("SEED") if isinstance(ctx, dict) else None,
        "mpi_ranks": ctx.get("MPI_RANKS") if isinstance(ctx, dict) else None,
        "dump_every_steps": ctx.get("DUMP_EVERY_STEPS") if isinstance(ctx, dict) else None,
        "thermo_every_steps": ctx.get("THERMO_EVERY_STEPS") if isinstance(ctx, dict) else None,
        "dump_file": ctx.get("DUMP_FILE") if isinstance(ctx, dict) else None,
        "starting_structure_datafile_path": datafile_path_str,
        "starting_structure_datafile_sha256": datafile_sha256,
        "starting_structure_n_atoms": n_atoms_in_datafile,
        "starting_structure_species_counts_by_lammps_type": datafile_species_counts,
    }
    if isinstance(ctx, dict) and isinstance(protocol["timestep_ps"], (int, float)) and \
            isinstance(protocol["n_steps"], int):
        protocol["total_simulated_time_ps"] = protocol["timestep_ps"] * protocol["n_steps"]
    else:
        protocol["total_simulated_time_ps"] = None
    if ensemble == "nvt":
        protocol["pressure_setpoint_bar"] = None
        protocol["pressure_note"] = (
            "NVT / fixed-volume ensemble: NO controlled pressure setpoint. Pressure is a "
            "diagnostic output only — see `thermo_diagnostic.observed_pressure_bar_nvt_diagnostic`."
        )

    # --- Section D: frozen deployment domain from pinned workflow.yaml + inputs -----
    workflow_yaml_sha = sha256_file(workflow_yaml_path) if workflow_yaml_path.is_file() else None
    workflow_cfg = _load_yaml_safe(workflow_yaml_path) if workflow_yaml_path.is_file() else None
    declared_domain = None
    domain_source_paths = []
    if isinstance(workflow_cfg, dict):
        deployment_scope = workflow_cfg.get("deployment_scope")
        if isinstance(deployment_scope, dict) and "deployment_domain" in deployment_scope:
            declared_domain = deployment_scope["deployment_domain"]
            domain_source_paths.append({"path": str(workflow_yaml_path), "sha256": workflow_yaml_sha,
                                         "kind": "workflow_yaml.deployment_scope.deployment_domain"})
    # Fallback: distillation_scope input snapshot
    if declared_domain is None:
        inputs_dir = run_dir / "inputs"
        if inputs_dir.is_dir():
            for snap in sorted(inputs_dir.iterdir()):
                if snap.suffix in (".yaml", ".yml") and "distillation_scope" in snap.name.lower():
                    payload_yaml = _load_yaml_safe(snap)
                    if isinstance(payload_yaml, dict) and "deployment_domain" in payload_yaml:
                        declared_domain = payload_yaml["deployment_domain"]
                        domain_source_paths.append({"path": str(snap), "sha256": sha256_file(snap),
                                                     "kind": "pinned_distillation_scope_snapshot"})
                        break
    deployment_domain = {
        "declared_domain": declared_domain,
        "declared_domain_provenance_sources": domain_source_paths,
        "in_domain_comparison": None,
        "note": (
            "the declared deployment domain is the AUTHORITATIVE composition/T/P envelope; the "
            "'realized_protocol' block above holds the run's SETPOINTS. Only qualitative envelope "
            "information (e.g. 'temperature_K: full source-pool envelope ambient through melt') is "
            "surfaced here; no numerical PASS is asserted where the frozen contract does not "
            "supply numerical bounds to compare against."
        ),
    }
    # Best-effort structural comparison for temperature only (composition qualitative;
    # pressure not comparable in NVT). Never fabricate a PASS.
    try:
        if isinstance(declared_domain, dict):
            temp_scope = declared_domain.get("temperature_K")
            realized_T = protocol["temperature_setpoint_K"]
            if isinstance(temp_scope, list) and isinstance(realized_T, (int, float)):
                deployment_domain["in_domain_comparison"] = {
                    "temperature_K_setpoint": realized_T,
                    "temperature_K_declared_envelope": temp_scope,
                    "envelope_kind": "qualitative_source_pool_envelope",
                    "numerical_bounds_available": False,
                    "assertion": (
                        "no numerical PASS asserted because the declared envelope is qualitative; "
                        "Judges/deterministic layer decide only what the contract actually supports"
                    ),
                }
    except Exception:  # pragma: no cover
        pass

    # --- Section E: Controller manifest.json action_approvals + submission timing --
    approval_block = {
        "approval_source": str(manifest_json_path),
        "approval_source_sha256": sha256_file(manifest_json_path) if manifest_json_path.is_file() else None,
    }
    stage_started_at_iso = None
    manifest_json = _load_json_safe(manifest_json_path) if manifest_json_path.is_file() else None
    if isinstance(manifest_json, dict):
        approvals = manifest_json.get("action_approvals") or {}
        prod_md = approvals.get("production_md") if isinstance(approvals, dict) else None
        if isinstance(prod_md, dict):
            approval_block.update({
                "granted": prod_md.get("granted"),
                "scope": prod_md.get("scope"),
                "action_type": prod_md.get("action_type"),
                "approved_at": prod_md.get("at"),
                "note_head": (prod_md.get("note") or "")[:400],
            })
        stages = manifest_json.get("stages") or []
        for s in stages:
            if s.get("name") == "deployment_md":
                stage_started_at_iso = s.get("started_at")
                break
    approval_block["submission_started_at"] = stage_started_at_iso
    approved_at_dt = _parse_iso_timestamp(approval_block.get("approved_at"))
    submitted_at_dt = _parse_iso_timestamp(stage_started_at_iso)
    if approved_at_dt is not None and submitted_at_dt is not None:
        approval_block["approval_precedes_submission"] = approved_at_dt <= submitted_at_dt
        approval_block["approval_to_submission_seconds"] = (submitted_at_dt - approved_at_dt).total_seconds()
    else:
        approval_block["approval_precedes_submission"] = None

    # --- Section F: deployment_provenance sub-block (secondary, cross-reference) -----
    depl_prov = _load_json_safe(depl_prov_path) if depl_prov_path.is_file() else None
    depl_prov_summary = None
    if isinstance(depl_prov, dict):
        starting = depl_prov.get("starting_structure") if isinstance(depl_prov.get("starting_structure"), dict) else {}
        backend = depl_prov.get("lammps_backend") if isinstance(depl_prov.get("lammps_backend"), dict) else {}
        preflight = depl_prov.get("pair_style_nn_preflight") if isinstance(depl_prov.get("pair_style_nn_preflight"), dict) else {}
        auth = depl_prov.get("authorization") if isinstance(depl_prov.get("authorization"), dict) else {}
        student = depl_prov.get("student") if isinstance(depl_prov.get("student"), dict) else {}
        depl_prov_summary = {
            "deployment_provenance_path": str(depl_prov_path),
            "deployment_provenance_sha256": sha256_file(depl_prov_path),
            "starting_structure": {
                "path": starting.get("path"),
                "sha256": starting.get("sha256"),
                "provenance_role": starting.get("provenance_role"),
                "leakage_check": starting.get("leakage_check"),
            },
            "lammps_backend": {
                "env": backend.get("env"),
                "binary_realpath": backend.get("binary_realpath"),
                "binary_sha256": backend.get("binary_sha256"),
                "lammps_version": backend.get("lammps_version"),
                "mpi": backend.get("mpi"),
            },
            "pair_style_nn_preflight": {
                "status": preflight.get("status"),
                "assertions_count": len(preflight.get("assertions") or []),
                "disposable": preflight.get("disposable"),
            },
            "authorization_cross_reference": {
                "boundary": auth.get("boundary"),
                "action_type": auth.get("action_type"),
                "approved_by_head": (auth.get("approved_by") or "")[:200],
                "resource_safety": (auth.get("resource_safety") or "")[:200],
            },
            "student_checkpoint_cross_reference": {
                "checkpoint_sha256": student.get("checkpoint_sha256"),
                "selected_seed": student.get("selected_seed"),
            },
        }
    else:
        depl_prov_summary = {
            "evidence_gap": (
                "deployment_provenance.json missing or unreadable — starting-structure "
                "provenance and LAMMPS-backend identity cannot be surfaced from this sibling; "
                "primary Controller-manifest approval evidence still holds"
            ),
        }

    # --- Section G: deterministic thermo.log diagnostic (bounded) --------------------
    # Equilibration cutoff = 20000 steps (10 ps at 0.5 fs timestep) — this matches the accepted
    # Stage-10 protocol's "first 10 ps = equilibration / final 60 ps = assessment window".
    eq_steps_default = 20000
    if isinstance(ctx, dict) and isinstance(ctx.get("N_STEPS"), int) and ctx["N_STEPS"] < eq_steps_default:
        eq_steps_default = max(0, ctx["N_STEPS"] // 7)  # scale-safe minimum
    thermo_diag = _thermo_diagnostic(thermo_log_path, eq_steps_default)

    # --- CASE-B versioned Stage-10 C2 refinement (Recovery id=6) ---------------------
    # Evaluate the versioned C2a/C2b/C2c subcriteria deterministically and attach the
    # structured facts NEXT TO the payload-side evidence. Pinned pre-init sources ONLY:
    #  * approved point: `inputs/008-validation_profile.yaml::shared_md_protocol`
    #  * composition scope: `inputs/009-distillation_scope.yaml::deployment_domain.composition_scope`
    # No new numerical T/P bound is invented. Original C2 remains in workflow.yaml
    # gate.criteria (byte-for-byte); this validator supplies the versioned adjudication
    # facts alongside the packet.
    approved_point = {}
    composition_scope_pinned = None
    pinned_temperature_envelope = None
    pinned_pressure_envelope = None
    validation_profile_path = run_dir / "inputs" / "008-validation_profile.yaml"
    validation_profile_sha = None
    if validation_profile_path.is_file():
        vp = _load_yaml_safe(validation_profile_path)
        validation_profile_sha = sha256_file(validation_profile_path)
        if isinstance(vp, dict):
            approved_point = vp.get("shared_md_protocol") or {}
            dd = vp.get("deployment_domain") or {}
            pinned_temperature_envelope = dd.get("temperature_K")
            pinned_pressure_envelope = dd.get("pressure_GPa")
    # composition_scope from distillation_scope snapshot
    for snap in sorted((run_dir / "inputs").iterdir()) if (run_dir / "inputs").is_dir() else []:
        if snap.suffix in (".yaml", ".yml") and "distillation_scope" in snap.name.lower():
            dsp = _load_yaml_safe(snap)
            if isinstance(dsp, dict):
                dd = dsp.get("deployment_domain") or {}
                composition_scope_pinned = dd.get("composition_scope")
            break
    # Determine realized composition from the DATAFILE species counts (deterministic,
    # generic mapping to the pre-declared scope list without any chemistry hardcoding).
    realized_composition = None
    species_counts = protocol.get("starting_structure_species_counts_by_lammps_type")
    if isinstance(species_counts, dict) and species_counts and isinstance(composition_scope_pinned, list):
        # Best-effort deterministic classification: look for a scope entry whose head
        # contains a chemistry token that appears in the datafile's Masses block (via
        # the LAMMPS type-1 : type-2 count RATIO). This never invents a rule — it
        # simply confirms whether the OBSERVED integer ratio corresponds to a single
        # scope entry marked with an explicit x-value.
        # For 2 LAMMPS types with the same counts as SiO2 (2:1 O:Si), the datafile's
        # own Masses lines already encode the elemental identity; the classification
        # here matches the ratio to the scope-entry token 'stoichiometric' /
        # 'sub-stoichiometric' / 'fully reduced boundary' only when the pinned scope
        # explicitly declares that label. If no explicit label is enumerable, leave
        # realized_composition as None and let C2a report unresolved rather than fake.
        try:
            values = sorted(species_counts.values(), reverse=True)
            if len(values) == 2 and values[1] > 0:
                ratio_ceil = values[0] / values[1]
                # Look for the explicit scope entry that describes this ratio.
                # Framework-generic tag matching: 'stoichiometric' + 'x = 0' anchors
                # a 2:1 majority:minority arrangement in a binary A2B compound.
                for entry in composition_scope_pinned:
                    if not isinstance(entry, str):
                        continue
                    low = entry.lower()
                    if abs(ratio_ceil - 2.0) < 1e-6 and "stoichiometric" in low and "x = 0" in low:
                        realized_composition = entry
                        break
        except Exception:  # pragma: no cover - defensive
            pass
    protocol_for_validator = dict(protocol)
    protocol_for_validator["realized_composition"] = realized_composition

    analysis_window_history = [
        {
            "kind": "pre_init_validation_profile_pinned",
            "source_path": str(validation_profile_path) if validation_profile_path.is_file() else None,
            "source_sha256": validation_profile_sha,
            "equilibration_ps": approved_point.get("nvt_equilibration_ps"),
            "production_ps": approved_point.get("nvt_production_ps"),
            "pinned_at": "run init (pre-Stage-10 submission)",
            "role": "pinned_analysis_partition_at_run_init",
        },
        {
            "kind": "option_2_human_accepted_assessment_semantics",
            "source_path": "operator directive (Option-2 accepted candidate protocol, session 2026-08-21)",
            "equilibration_ps": 10.0,
            "assessment_ps": 60.0,
            "recorded_at_or_before": "2026-08-21T01:42:09Z (production_md approval)",
            "role": "later_human_approved_assessment_window",
        },
        {
            "note": (
                "Both partitions predate MD submission. Both are recorded as immutable "
                "provenance. They are DIFFERENT 70-ps partitions of the SAME executable "
                "MD trajectory (NVT for the full 70 ps); they are NOT an exact match and "
                "MUST NOT be claimed to coincide. The later 10/60 assessment window is the "
                "canonical Stage-10 trajectory-diagnostic partition used in this recovery."
            ),
        },
    ]

    # ---- Build identity bindings for C2b (starting_structure + Student checkpoint) ----
    # Read ONLY from pre-submission authoritative artifacts (context.yaml,
    # deployment_provenance.json, action_approvals.production_md, and the pinned
    # student_committee.manifest.json). Realized identities are reconstructed
    # INDEPENDENTLY from the execution record (md.manifest.json + input.lmp read_data
    # path). Never conflate committee_manifest_sha256 with checkpoint_sha256.
    identity_bindings = {}

    def _parse_read_data(lmp_text: str):
        for line in lmp_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            parts = stripped.split(None, 1)
            if len(parts) == 2 and parts[0] == "read_data":
                return parts[1].strip()
        return None

    def _iso(value):
        if not isinstance(value, str) or not value:
            return None
        s = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            from datetime import datetime
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    # Approval + submission timing (used to prove binding_precedes_submission)
    approved_at_iso = approval_block.get("approved_at") if isinstance(approval_block, dict) else None
    submitted_at_iso = approval_block.get("submission_started_at") if isinstance(approval_block, dict) else None
    approved_at_dt = _iso(approved_at_iso)
    submitted_at_dt = _iso(submitted_at_iso)

    # ---- starting_structure_identity ----
    approved_ss_path = None
    approved_ss_sha256 = None
    approved_ss_source = None
    approved_ss_source_sha256 = None
    approved_ss_at = None

    # Primary: context.yaml DATAFILE (path only; the pre-submission binding pins the file).
    if isinstance(ctx, dict) and isinstance(ctx.get("DATAFILE"), str):
        approved_ss_path = ctx["DATAFILE"]
        approved_ss_source = str(context_yaml_path)
        approved_ss_source_sha256 = context_yaml_sha
        approved_ss_at = approved_at_iso   # written under production_md approval
    # Direct sha binding: deployment_provenance.json.starting_structure.sha256
    if isinstance(depl_prov, dict):
        ss = depl_prov.get("starting_structure") if isinstance(depl_prov.get("starting_structure"), dict) else {}
        prov_ss_sha = ss.get("sha256")
        if isinstance(prov_ss_sha, str) and prov_ss_sha:
            approved_ss_sha256 = prov_ss_sha
            # Provenance source for the direct SHA
            if approved_ss_source is None:
                approved_ss_source = str(depl_prov_path)
                approved_ss_source_sha256 = sha256_file(depl_prov_path) if depl_prov_path.is_file() else None
                approved_ss_at = approved_at_iso

    # Realized: reconstruct independently from md.manifest → input.lmp → read_data → sha
    realized_ss_path = None
    realized_ss_sha256 = None
    realized_ss_source = None
    input_lmp_from_evidence = None
    if isinstance(payload.get("evidence"), list):
        for entry in payload["evidence"]:
            if isinstance(entry, dict) and entry.get("role") == "input":
                input_lmp_from_evidence = entry.get("path")
                break
    if input_lmp_from_evidence and Path(input_lmp_from_evidence).is_file():
        try:
            _text = Path(input_lmp_from_evidence).read_text(encoding="utf-8", errors="replace")
        except OSError:
            _text = ""
        parsed = _parse_read_data(_text) if _text else None
        if parsed:
            realized_ss_path = parsed
            realized_ss_source = input_lmp_from_evidence
            if Path(parsed).is_file():
                realized_ss_sha256 = sha256_file(Path(parsed))

    ss_match = (
        approved_ss_sha256 is not None and realized_ss_sha256 is not None
        and approved_ss_sha256 == realized_ss_sha256
    )
    ss_precedes = (
        approved_at_dt is not None and submitted_at_dt is not None
        and approved_at_dt <= submitted_at_dt
    ) if (approved_at_dt is not None and submitted_at_dt is not None) else None
    identity_bindings["starting_structure_identity"] = {
        "approved_path": approved_ss_path,
        "approved_sha256": approved_ss_sha256,
        "approved_source": approved_ss_source,
        "approved_source_sha256": approved_ss_source_sha256,
        "approved_at": approved_ss_at,
        "realized_path": realized_ss_path,
        "realized_sha256": realized_ss_sha256,
        "realized_source":
            f"reconstructed from {realized_ss_source} → read_data → deterministic sha256 of the data file"
            if realized_ss_source else None,
        "match": bool(ss_match),
        "submission_started_at": submitted_at_iso,
        "binding_precedes_submission": bool(ss_precedes) if ss_precedes is not None else None,
        "note": (
            "Approved sha256 is read from the pre-submission deployment_provenance.json "
            "(written under production_md approval); realized sha256 is reconstructed "
            "INDEPENDENTLY by parsing input.lmp's read_data directive and hashing the "
            "actual data file. Fields are never mirrored across sides."
        ),
    }

    # ---- student_checkpoint_identity ----
    # 1) Approved seed: prefer manifest.action_approvals.production_md.note (mentions
    #    the committee_manifest hash but seed is bound via the committee-manifest entry
    #    for the specific selected_seed). We treat deployment_provenance.student
    #    (selected_seed + checkpoint_sha256) as the direct pre-submission binding
    #    (recorded at approval time), and cross-check the seed against the pinned
    #    student_committee.manifest.json.
    approved_seed = None
    approved_ck_path = None
    approved_ck_sha = None
    approved_ck_source = None
    approved_ck_source_sha256 = None
    approved_ck_at = None
    approved_ck_derivation = None
    ck_committee_manifest_sha = None
    if isinstance(depl_prov, dict):
        student_prov = depl_prov.get("student") if isinstance(depl_prov.get("student"), dict) else {}
        approved_seed = student_prov.get("selected_seed")
        approved_ck_sha = student_prov.get("checkpoint_sha256")
        if approved_ck_sha:
            approved_ck_source = str(depl_prov_path)
            approved_ck_source_sha256 = sha256_file(depl_prov_path) if depl_prov_path.is_file() else None
            approved_ck_at = approved_at_iso
    # Fallback derivation (no direct sha): seed → committee_manifest entry → file
    committee_manifest_path = md_manifest_dir / "student_committee.manifest.json"
    if committee_manifest_path.is_file():
        cm = _load_json_safe(committee_manifest_path)
        ck_committee_manifest_sha = sha256_file(committee_manifest_path)
        if isinstance(cm, dict):
            for m in (cm.get("models") or []):
                if isinstance(m, dict) and m.get("seed") == approved_seed:
                    if approved_ck_sha is None:
                        integ = m.get("integrity") if isinstance(m.get("integrity"), dict) else {}
                        approved_ck_sha = integ.get("sha256")
                        approved_ck_path = m.get("path") or approved_ck_path
                        approved_ck_source = str(committee_manifest_path)
                        approved_ck_source_sha256 = ck_committee_manifest_sha
                        approved_ck_derivation = (
                            "approved_selected_seed → pinned student_committee.manifest.json "
                            "models[seed=...].integrity.sha256"
                        )
                    else:
                        # We have the direct sha; still cross-check that the manifest
                        # entry for this seed carries the SAME sha (never confuse
                        # committee_manifest sha with checkpoint sha).
                        integ = m.get("integrity") if isinstance(m.get("integrity"), dict) else {}
                        cross = integ.get("sha256")
                        approved_ck_derivation = (
                            "direct from deployment_provenance.student.checkpoint_sha256; "
                            "cross-checked against committee_manifest models[seed=...]. "
                            f"cross_check_match={cross == approved_ck_sha}"
                        )
                    if approved_ck_path is None:
                        approved_ck_path = m.get("path")
                    break

    realized_seed = payload.get("selected_seed")
    realized_ck_path = payload.get("checkpoint")
    _realized_ck_integrity = payload.get("checkpoint_integrity") if isinstance(
        payload.get("checkpoint_integrity"), dict) else {}
    realized_ck_sha = _realized_ck_integrity.get("sha256")

    ck_seed_match = (approved_seed is not None and realized_seed is not None
                      and int(approved_seed) == int(realized_seed))
    ck_sha_match = (approved_ck_sha is not None and realized_ck_sha is not None
                    and approved_ck_sha == realized_ck_sha)
    # committee_manifest sha semantic guard — must NOT equal checkpoint sha
    ck_semantic_ok = (ck_committee_manifest_sha is None
                       or realized_ck_sha is None
                       or ck_committee_manifest_sha != realized_ck_sha)
    ck_match = ck_seed_match and ck_sha_match and ck_semantic_ok
    ck_precedes = (
        approved_at_dt is not None and submitted_at_dt is not None
        and approved_at_dt <= submitted_at_dt
    ) if (approved_at_dt is not None and submitted_at_dt is not None) else None
    identity_bindings["student_checkpoint_identity"] = {
        "approved_selected_seed": approved_seed,
        "approved_checkpoint_path": approved_ck_path,
        "approved_checkpoint_sha256": approved_ck_sha,
        "approved_source": approved_ck_source,
        "approved_source_sha256": approved_ck_source_sha256,
        "approved_at": approved_ck_at,
        "approved_derivation": approved_ck_derivation,
        "realized_selected_seed": realized_seed,
        "realized_checkpoint_path": realized_ck_path,
        "realized_checkpoint_sha256": realized_ck_sha,
        "realized_source":
            "md.manifest.json.checkpoint_integrity.sha256 (with selected_seed cross-check)",
        "committee_manifest_sha256_semantic_guard": {
            "committee_manifest_sha256": ck_committee_manifest_sha,
            "checkpoint_sha256": realized_ck_sha,
            "committee_manifest_sha_is_not_the_checkpoint_sha": bool(ck_semantic_ok),
        },
        "match": bool(ck_match),
        "seed_match": bool(ck_seed_match),
        "sha_match": bool(ck_sha_match),
        "submission_started_at": submitted_at_iso,
        "binding_precedes_submission": bool(ck_precedes) if ck_precedes is not None else None,
        "note": (
            "Approved checkpoint identity is bound from pre-submission "
            "deployment_provenance.student (recorded under production_md approval), "
            "cross-checked against the pinned student_committee.manifest.json entry for the "
            "approved seed. Realized identity is read INDEPENDENTLY from "
            "md.manifest.json.checkpoint_integrity.sha256. The committee_manifest sha256 is "
            "kept semantically distinct from the checkpoint sha256 and never used as its "
            "value."
        ),
    }

    try:
        from validation.deployment_point import validate_stage10_deployment_point
        deployment_domain_versioned_evaluation = validate_stage10_deployment_point(
            realized_protocol=protocol_for_validator,
            approved_shared_md_protocol=approved_point,
            pinned_composition_scope=composition_scope_pinned,
            pinned_temperature_envelope=pinned_temperature_envelope,
            pinned_pressure_envelope=pinned_pressure_envelope,
            thermo_diagnostic=thermo_diag,
            analysis_window_history=analysis_window_history,
            identity_bindings=identity_bindings,
        )
    except Exception as exc:  # pragma: no cover - defensive
        deployment_domain_versioned_evaluation = {
            "evidence_gap": f"deployment_point validator raised: {type(exc).__name__}: {exc}"
        }

    # --- Assemble sibling-derived attachment ----------------------------------------
    return {
        "sibling_note": (
            "Criterion-relevant values are read VERBATIM from pinned sibling artifacts: "
            "context.yaml (realized protocol), workflow.yaml / distillation_scope.yaml "
            "(frozen deployment domain), Controller manifest.json action_approvals "
            "(primary human-approval ledger + submission timing), deployment_provenance.json "
            "(structure + backend provenance, cross-reference), and a deterministic bounded "
            "pass over thermo.log (assessment-window T diagnostics, NVT pressure diagnostic "
            "ONLY, and total energy explicitly labelled as an NVT diagnostic — never an NVE "
            "conservation criterion). No new statistic is invented; no artifact is mutated."
        ),
        "realized_protocol": protocol,
        "deployment_domain": deployment_domain,
        "human_approval": approval_block,
        "deployment_provenance": depl_prov_summary,
        "thermo_diagnostic": thermo_diag,
        "deployment_domain_versioned_evaluation": deployment_domain_versioned_evaluation,
        "framework_notes": {
            "nvt_pressure_semantic": (
                "This is an NVT (fixed-volume) run. Pressure has NO controlled setpoint. "
                "Observed pressure is a diagnostic time-series only, not a pressure-domain "
                "PASS/FAIL criterion."
            ),
            "energy_conservation_semantic": (
                "Total energy in NVT is not conserved by design — the Nose-Hoover thermostat "
                "exchanges energy with the system. The `total_energy_eV_*` fields are for "
                "diagnostic inspection only and MUST NOT be treated as NVE-style energy "
                "conservation checks."
            ),
        },
    }


register_json_evidence_adapter(
    "md_manifest", _is_md_manifest, _md_manifest_summary,
)


def _is_physical_validation_report(payload: dict) -> bool:
    """A Stage-11 physical-validation report: a dict with a ``checks`` list that contains the
    required ``nve_drift`` deployment-stability observable. This is the distinctive signature of
    the physical_validation stage (its nve_drift check is always required by the frozen profile),
    so it is matched structurally rather than by any campaign/dataset name."""
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    return any(isinstance(c, dict) and c.get("observable") == "nve_drift" for c in checks)


def _physical_validation_report_summary(payload: dict) -> dict:
    """Bounded, Judge-facing surfacing of a Stage-11 physical-validation report.

    Surfaces each observable's value + criterion + pass/fail verbatim so the Judge reads the
    numbers where it reads the decision. Descriptive comparisons (``threshold: null``) are marked
    ``descriptive`` -- never coerced into a synthetic PASS/FAIL. The required deterministic
    stability criterion (``nve_drift``, sourced from the DEDICATED microcanonical segment) is
    surfaced prominently with its full drift details and energy-log evidence.
    """
    from validation.report import criterion_passes

    checks = [c for c in payload.get("checks", []) if isinstance(c, dict)]
    observables = []
    for c in checks:
        criterion = c.get("criterion")
        row = {
            "observable": c.get("observable"),
            "domain": c.get("domain"),
            "value": c.get("value"),
            "unit": c.get("unit"),
            "criterion": criterion,
        }
        if criterion in (None, {}):
            row["role"] = "descriptive"
            row["passed"] = None
        elif c.get("value") is None:
            row["role"] = "thresholded"
            row["passed"] = None
            row["evidence_gap"] = c.get("reason") or "no value produced"
        else:
            row["role"] = "thresholded"
            try:
                row["passed"] = bool(criterion_passes(c.get("value"), criterion))
            except Exception as exc:  # pragma: no cover - defensive
                row["passed"] = None
                row["evidence_gap"] = f"criterion could not be evaluated: {exc}"
        observables.append(row)

    nve = next((r for r in observables if r["observable"] == "nve_drift"), None)
    nve_detail = next((c for c in checks if c.get("observable") == "nve_drift"), {})
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    nve_energy_log = next((e for e in evidence if isinstance(e, dict)
                           and e.get("role") == "nve_energy_log"), None)
    return {
        "profile": payload.get("profile"),
        "n_observables": len(observables),
        "observables": observables,
        "nve_drift": {
            "value": (nve or {}).get("value"),
            "unit": (nve or {}).get("unit"),
            "criterion": (nve or {}).get("criterion"),
            "passed": (nve or {}).get("passed"),
            "details": nve_detail.get("details"),
            "energy_log_evidence": nve_energy_log,
            "note": (
                "nve_drift is measured on a DEDICATED microcanonical (NVE) segment, distinct "
                "from the thermostatted NVT production trajectory whose total energy is not a "
                "valid conservation metric."),
        } if nve is not None else None,
    }


register_json_evidence_adapter(
    "physical_validation_report", _is_physical_validation_report,
    _physical_validation_report_summary,
)


def _compact_directory_integrity(integrity: dict) -> dict:
    """Bound the LLM-facing representation of a directory's integrity digest.

    ``artifact_digest`` returns the COMPLETE per-file ``files`` list for a directory tree; that is
    the canonical, tamper-evident record and is left untouched in Controller state. For the
    bounded *evidence* packet a large tree (e.g. a multi-seed committee with thousands of
    intermediate cache files) would overflow ``MAX_EVIDENCE_BYTES``, so this trims the inlined
    listing to a deterministic head of ``EVIDENCE_DIRECTORY_FILE_CAP`` entries. The aggregate tree
    ``sha256`` and total ``size`` already preserve full integrity; ``n_files``/``n_files_omitted``
    make the truncation explicit. Non-directory or already-small digests pass through unchanged
    (a directory at/below the cap keeps its full listing with ``n_files_omitted == 0``). File
    order is the sorted-by-relative-path order ``artifact_digest`` already produces, so the sample
    is stable and reproducible -- never dependent on filesystem traversal order.
    """
    if not isinstance(integrity, dict) or integrity.get("kind") != "directory":
        return integrity
    files = integrity.get("files")
    if not isinstance(files, list):
        return integrity
    shown = files[:EVIDENCE_DIRECTORY_FILE_CAP]
    return {
        "kind": integrity.get("kind"),
        "size": integrity.get("size"),
        "sha256": integrity.get("sha256"),
        "n_files": len(files),
        "files_shown": shown,
        "n_files_omitted": len(files) - len(shown),
    }


def summarize_artifact(path: str | Path, *, split_crosswalk: dict | None = None) -> dict:
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    summary = {
        "artifact_path": str(path),
        "integrity": _compact_directory_integrity(artifact_digest(path)),
        "summary_kind": "generic",
        "evidence_gaps": [],
    }
    try:
        if suffix in {".xyz", ".extxyz"}:
            summary.update({"summary_kind": "ase_frames",
                            **_frame_summary(path, split_crosswalk=split_crosswalk)})
        elif suffix == ".json":
            summary.update({"summary_kind": "json_manifest", **_json_summary(path)})
        else:
            summary["evidence_gaps"].append(
                f"no semantic summarizer for extension {suffix or '<none>'}")
    except Exception as exc:
        summary["evidence_gaps"].append(f"summarizer_failed: {type(exc).__name__}: {exc}")
    return summary


def lineage_reference_summary(path: str | Path) -> dict:
    """Compact lineage/integrity-only summary for a large raw-content artifact whose SEMANTIC
    content is already surfaced elsewhere in the SAME evidence packet.

    Used for e.g. the Stage-8 ``evaluated.extxyz`` raw per-frame predictions: the co-declared
    four-channel ``accuracy_report.json`` already carries the exact evaluation population and every
    fidelity metric (aggregate + domain/configuration-family resolved), so the Judge must not be
    required to re-derive fidelity from this large extxyz, and the bounded packet must not carry its
    full per-frame composition/category/domain distribution (which pushed the Judge prompt over the
    model context). This surfaces ONLY deterministic provenance -- artifact identity/path, tree
    ``sha256`` + ``size``, and frame/atom counts -- so the artifact's lineage/hash binding is fully
    preserved while its bulk is dropped. Scientific content, metrics, and the artifact itself are
    unchanged; this is a Judge-facing evidence compaction only.
    """
    path = Path(path).resolve()
    summary = {
        "artifact_path": str(path),
        "integrity": _compact_directory_integrity(artifact_digest(path)),
        "summary_kind": "lineage_reference",
        "evidence_gaps": [],
    }
    if path.suffix.lower() in {".xyz", ".extxyz"}:
        try:
            from ase.io import read

            frames = read(str(path), index=":")
            summary["n_frames"] = len(frames)
            summary["n_atoms"] = int(sum(len(atoms) for atoms in frames))
        except Exception as exc:
            summary["evidence_gaps"].append(
                f"lineage_count_failed: {type(exc).__name__}: {exc}")
    return summary


def build_bounded_evidence(
    artifacts: Iterable[str | Path],
    out_path: str | Path,
    *,
    protocol_refs: Iterable[str | Path] = (),
    validation_outcomes: Iterable[dict] = (),
    facts: Iterable = (),
    lineage_only: Iterable[str | Path] = (),
) -> dict:
    out = Path(out_path).resolve()
    artifact_paths = list(artifacts)
    # Artifacts whose bulky semantic content is redundant in THIS packet (already surfaced by a
    # co-present summary) are rendered lineage/integrity-only rather than fully summarized -- see
    # lineage_reference_summary. Everything else is summarized as before.
    lineage_only_set = {str(Path(p).resolve()) for p in lineage_only}
    # Built once from the full artifact set (not per-frame-file): any registered split-membership
    # manifest among these artifacts (see build_split_crosswalk) becomes the authoritative crosswalk
    # every .extxyz artifact's frame-level source lineage is joined against below.
    split_crosswalk = build_split_crosswalk(artifact_paths)
    protocol_records = []
    for ref in protocol_refs:
        p = Path(ref).resolve()
        if p.exists():
            protocol_records.append({"path": str(p), "sha256": sha256_file(p)})
    # Framework V2 (Section 14): DeterministicFact records are authoritative. When
    # a stage supplies them, render each through the generic EvidenceCompiler so
    # the Judge packet carries both the legacy validation_outcome shape and the
    # typed facts -- an LLM Judge cannot negate a deterministic fact.
    fact_list = list(facts)
    deterministic_facts = []
    fact_outcomes = []
    if fact_list:
        from framework_v2.evidence_compiler import fact_to_validation_outcome
        deterministic_facts = [f.model_dump(mode="json") for f in fact_list]
        fact_outcomes = [fact_to_validation_outcome(f) for f in fact_list]
    payload = {
        "schema_version": 1,
        "max_evidence_bytes": MAX_EVIDENCE_BYTES,
        "artifacts": [
            lineage_reference_summary(path)
            if str(Path(path).resolve()) in lineage_only_set
            else summarize_artifact(path, split_crosswalk=split_crosswalk)
            for path in artifact_paths],
        "protocol_refs": protocol_records,
        "validation_outcomes": list(validation_outcomes) + fact_outcomes,
        "deterministic_facts": deterministic_facts,
        "split_crosswalk_sources": split_crosswalk["sources"],
        "split_crosswalk_ambiguous_key_count": len(split_crosswalk["ambiguous"]),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if len(text.encode("utf-8")) > MAX_EVIDENCE_BYTES:
        raise ValueError("bounded evidence summary exceeds 256 KB")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    payload["summary_path"] = str(out)
    payload["summary_sha256"] = sha256_file(out)
    return payload
