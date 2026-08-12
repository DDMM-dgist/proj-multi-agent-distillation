"""Deterministic bounded-evidence summaries for agent context."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from workflow.integrity import artifact_digest, sha256_file

MAX_EVIDENCE_BYTES = 256 * 1024


def _frame_summary(path: Path) -> dict:
    from ase.io import read

    frames = read(str(path), index=":")
    compositions = Counter()
    categories = Counter()
    domains = Counter()
    atom_counts = []
    missing_lineage = 0
    for index, atoms in enumerate(frames):
        symbols = Counter(atoms.get_chemical_symbols())
        compositions[" ".join(f"{k}{v}" for k, v in sorted(symbols.items()))] += 1
        categories[str(atoms.info.get("config_type", atoms.info.get("source", "unknown")))] += 1
        domains[str(atoms.info.get("domain", "unknown"))] += 1
        atom_counts.append(len(atoms))
        if "parent_structure_id" not in atoms.info and index > 0:
            missing_lineage += 1
    return {
        "n_frames": len(frames),
        "atom_count_min": min(atom_counts) if atom_counts else 0,
        "atom_count_max": max(atom_counts) if atom_counts else 0,
        "composition_counts": dict(compositions.most_common(32)),
        "category_counts": dict(categories.most_common(64)),
        "domain_counts": dict(domains.most_common(64)),
        "missing_lineage_frames": missing_lineage,
    }


def _json_summary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return {
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
    if isinstance(payload, list):
        return {"json_type": "list", "length": len(payload)}
    return {"json_type": type(payload).__name__}


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
    out.write_text(text + "\n")
    payload["summary_path"] = str(out)
    payload["summary_sha256"] = sha256_file(out)
    return payload
