"""Deterministic bounded-evidence summaries for agent context."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from workflow.integrity import artifact_digest, sha256_file

MAX_EVIDENCE_BYTES = 256 * 1024
DIRECT_JUDGE_ARTIFACT_BYTES = 1_000_000

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


def _frame_summary(path: Path) -> dict:
    from ase.io import read
    import math
    import numpy as np

    frames = read(str(path), index=":")
    compositions = Counter()
    categories = Counter()
    domains = Counter()
    atom_counts = []
    missing_lineage = 0
    label_keys = set()
    finite_teacher_energy = 0
    finite_teacher_force = 0
    for index, atoms in enumerate(frames):
        symbols = Counter(atoms.get_chemical_symbols())
        compositions[" ".join(f"{k}{v}" for k, v in sorted(symbols.items()))] += 1
        categories[str(atoms.info.get("config_type", atoms.info.get("source", "unknown")))] += 1
        domains[str(atoms.info.get("domain", "unknown"))] += 1
        atom_counts.append(len(atoms))
        if "parent_structure_id" not in atoms.info and index > 0:
            missing_lineage += 1
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
        "missing_lineage_frames": missing_lineage,
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
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    operational = next(
        (c for c in checks if isinstance(c, dict)
         and c.get("observable") == "fresh_teacher_energy_force_finiteness"),
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
register_json_evidence_adapter(
    "structural_coverage_evidence", _is_directed_coverage_evidence,
    _structural_coverage_evidence_summary,
)


def summarize_artifact(path: str | Path) -> dict:
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    summary = {
        "artifact_path": str(path),
        "integrity": artifact_digest(path),
        "summary_kind": "generic",
        "evidence_gaps": [],
    }
    try:
        if suffix in {".xyz", ".extxyz"}:
            summary.update({"summary_kind": "ase_frames", **_frame_summary(path)})
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
    protocol_records = []
    for ref in protocol_refs:
        p = Path(ref).resolve()
        if p.exists():
            protocol_records.append({"path": str(p), "sha256": sha256_file(p)})
    payload = {
        "schema_version": 1,
        "max_evidence_bytes": MAX_EVIDENCE_BYTES,
        "artifacts": [summarize_artifact(path) for path in artifacts],
        "protocol_refs": protocol_records,
        "validation_outcomes": list(validation_outcomes),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if len(text.encode("utf-8")) > MAX_EVIDENCE_BYTES:
        raise ValueError("bounded evidence summary exceeds 256 KB")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    payload["summary_path"] = str(out)
    payload["summary_sha256"] = sha256_file(out)
    return payload
