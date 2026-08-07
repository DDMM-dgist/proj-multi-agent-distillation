"""v6 -> v7 run-manifest migration, performed ONLY on a copy (never in place).

The v7 bump is purely additive operational metadata (runtime attempt references, action
idempotency, stale-running runner metadata). This module upgrades a COPY of a run directory so
the original — including a frozen baseline — is never modified. On any failure the destination
copy is removed and the source is left untouched. See SCHEMA_MIGRATION.md for rollback.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

SUPPORTED_TARGET = 7


def apply_v7_fields(state: dict) -> dict:
    """Add missing v7 additive fields with safe empty defaults. Never changes an existing field."""
    state.setdefault("runtime_attempts", [])
    state.setdefault("idempotency", {})
    return state


def migrate_run_manifest(src_run_dir, dst_run_dir) -> Path:
    """Copy ``src_run_dir`` to ``dst_run_dir`` and upgrade the COPY's manifest to schema_version 7.

    The source is never modified. If the destination already exists, or the source manifest is
    missing/invalid/newer-than-target, the destination copy is removed and the error is raised —
    the original is always preserved. Returns the destination manifest path.
    """
    src = Path(src_run_dir).resolve()
    dst = Path(dst_run_dir).resolve()
    if not (src / "manifest.json").exists():
        raise FileNotFoundError(f"source run has no manifest.json: {src}")
    if dst.exists():
        raise FileExistsError(f"destination already exists (refusing to overwrite): {dst}")
    shutil.copytree(src, dst)
    try:
        manifest = dst / "manifest.json"
        state = json.loads(manifest.read_text())
        original_version = state.get("schema_version")
        if original_version is None:
            raise ValueError("source manifest has no schema_version")
        if original_version > SUPPORTED_TARGET:
            raise ValueError(f"cannot downgrade schema_version {original_version} -> {SUPPORTED_TARGET}")
        apply_v7_fields(state)
        state["schema_version"] = SUPPORTED_TARGET
        state.setdefault("events", []).append({
            "type": "schema_migrated", "from": original_version, "to": SUPPORTED_TARGET,
            "at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
        tmp = manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(manifest)
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)  # preserve original; drop the partial copy
        raise
    return manifest
