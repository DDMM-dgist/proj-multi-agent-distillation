"""Deterministic bounded-evidence summaries for agent context."""
from __future__ import annotations

import json
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
    lambda payload: bool({"teacher_model_sha256", "source_sha256", "n_frames"} & set(payload)),
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


def build_bounded_evidence(
    artifacts: Iterable[str | Path],
    out_path: str | Path,
    *,
    protocol_refs: Iterable[str | Path] = (),
    validation_outcomes: Iterable[dict] = (),
) -> dict:
    out = Path(out_path).resolve()
    artifact_paths = list(artifacts)
    # Built once from the full artifact set (not per-frame-file): any registered split-membership
    # manifest among these artifacts (see build_split_crosswalk) becomes the authoritative crosswalk
    # every .extxyz artifact's frame-level source lineage is joined against below.
    split_crosswalk = build_split_crosswalk(artifact_paths)
    protocol_records = []
    for ref in protocol_refs:
        p = Path(ref).resolve()
        if p.exists():
            protocol_records.append({"path": str(p), "sha256": sha256_file(p)})
    payload = {
        "schema_version": 1,
        "max_evidence_bytes": MAX_EVIDENCE_BYTES,
        "artifacts": [summarize_artifact(path, split_crosswalk=split_crosswalk)
                      for path in artifact_paths],
        "protocol_refs": protocol_records,
        "validation_outcomes": list(validation_outcomes),
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
