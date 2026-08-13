"""Deterministic bounded-evidence summaries for agent context."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from workflow.integrity import artifact_digest, sha256_file

MAX_EVIDENCE_BYTES = 256 * 1024
DIRECT_JUDGE_ARTIFACT_BYTES = 1_000_000


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
        if payload.get("profile") == "teacher_baseline":
            summary["teacher_baseline"] = _teacher_baseline_report_summary(payload)
        if {"teacher_model_sha256", "source_sha256", "n_frames"} & set(payload):
            summary["label_manifest"] = {
                "n_frames": payload.get("n_frames"),
                "labels": payload.get("labels"),
                "teacher_model_sha256": payload.get("teacher_model_sha256"),
                "teacher_config_sha256": payload.get("teacher_config_sha256"),
                "source_sha256": payload.get("source_sha256"),
                "output_sha256": payload.get("sha256"),
                "environment": payload.get("environment"),
            }
        return summary
    if isinstance(payload, list):
        return {"json_type": "list", "length": len(payload)}
    return {"json_type": type(payload).__name__}


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
