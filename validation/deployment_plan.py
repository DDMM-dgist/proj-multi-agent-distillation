"""Reusable Stage-10 deployment-MD readiness contract.

The contract separates scientific binding (which deployment point/protocol/checkpoint) from
operational provisioning (whether the requested backend is actually runnable).  It never chooses a
starting structure, seed, protocol, or backend default on behalf of a campaign.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from validation.deployment_resolution import resolve_selected_checkpoint, load_shared_md_protocol
from workflow.integrity import sha256_file

READY = "READY"
HUMAN_SCIENTIFIC_INPUT_REQUIRED = "HUMAN_SCIENTIFIC_INPUT_REQUIRED"
HUMAN_OPERATIONAL_INPUT_REQUIRED = "HUMAN_OPERATIONAL_INPUT_REQUIRED"


def _load_mapping(value, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        raise ValueError(f"{label} is required")
    path = Path(value)
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _bind_starting_structure(contract) -> dict[str, Any]:
    spec = _load_mapping(contract, "starting_structure_contract")
    path_value = spec.get("path") or spec.get("structures_path")
    if not path_value:
        raise ValueError(
            "HUMAN_SCIENTIFIC_INPUT_REQUIRED: Stage-10 starting structure must be bound by an "
            "explicit deployment-point selection contract")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"deployment starting structure does not exist: {path}")
    observed = sha256_file(path)
    expected = spec.get("sha256") or spec.get("structures_sha256")
    if expected and observed != expected:
        raise ValueError("deployment starting structure sha256 mismatch")
    if not spec.get("deployment_point_id"):
        raise ValueError("starting_structure_contract.deployment_point_id is required")
    if not spec.get("selection_rule"):
        raise ValueError("starting_structure_contract.selection_rule is required")
    return {
        "deployment_point_id": spec["deployment_point_id"],
        "selection_rule": spec["selection_rule"],
        "path": str(path),
        "sha256": observed,
        "role": spec.get("role", "deployment_starting_structure"),
        "source_population_sha256": spec.get("source_population_sha256"),
    }


def _backend_status(md_config: dict[str, Any]) -> dict[str, Any]:
    status = md_config.get("provisioning_status")
    if status != "PROVISIONED":
        return {
            "status": HUMAN_OPERATIONAL_INPUT_REQUIRED,
            "provisioning_status": status or "UNDECLARED",
            "reason": "MD backend is not canonically provisioned; do not run production MD",
            "kind": md_config.get("kind"),
            "env": md_config.get("env"),
            "binary": md_config.get("binary"),
            "preflight": md_config.get("preflight"),
        }
    return {
        "status": READY,
        "provisioning_status": status,
        "kind": md_config.get("kind"),
        "env": md_config.get("env"),
        "binary": md_config.get("binary"),
        "preflight": md_config.get("preflight"),
    }


def build_stage10_deployment_plan(*, md_config, validation_profile, committee_manifest,
                                  starting_structure_contract, selected_seed=None,
                                  select_by=None, velocity_seed=None) -> dict[str, Any]:
    """Build a pre-execution deployment-MD plan with fail-closed readiness semantics."""
    md_cfg = _load_mapping(md_config, "md_config")
    protocol = load_shared_md_protocol(validation_profile)
    start = _bind_starting_structure(starting_structure_contract)
    if velocity_seed is None:
        raise ValueError(
            "HUMAN_SCIENTIFIC_INPUT_REQUIRED: Stage-10 velocity_seed/seed policy is not bound")
    student = resolve_selected_checkpoint(
        committee_manifest, selected_seed=selected_seed, select_by=select_by)
    backend = _backend_status(md_cfg)
    scientific_status = READY
    executable = backend["status"] == READY and scientific_status == READY
    return {
        "schema_version": 1,
        "contract_kind": "stage10_deployment_md_plan",
        "scientific_status": scientific_status,
        "operational_status": backend["status"],
        "executable": executable,
        "starting_structure": start,
        "shared_md_protocol": protocol,
        "velocity_seed": int(velocity_seed),
        "student": student,
        "backend": backend,
        "human_input_required": ([] if executable else [backend["status"]]),
    }


def validate_stage10_deployment_plan(plan) -> dict[str, Any]:
    payload = _load_mapping(plan, "stage10 deployment plan") if not isinstance(plan, dict) else plan
    if payload.get("schema_version") != 1:
        raise ValueError("Stage-10 deployment plan requires schema_version=1")
    if payload.get("contract_kind") != "stage10_deployment_md_plan":
        raise ValueError("Stage-10 deployment plan has wrong contract_kind")
    for field in ("scientific_status", "operational_status", "executable", "starting_structure", "shared_md_protocol", "student", "backend"):
        if field not in payload:
            raise ValueError(f"Stage-10 deployment plan missing {field}")
    if payload["scientific_status"] != READY:
        raise ValueError("Stage-10 scientific deployment contract is not complete")
    if payload["executable"] and payload["operational_status"] != READY:
        raise ValueError("Stage-10 plan cannot be executable while backend is not READY")
    start = payload["starting_structure"]
    path = Path(start.get("path", ""))
    if not path.is_file() or sha256_file(path) != start.get("sha256"):
        raise ValueError("Stage-10 starting structure provenance is not hash-valid")
    student = payload["student"]
    if not student.get("checkpoint_sha256") or not student.get("checkpoint_path"):
        raise ValueError("Stage-10 plan must bind a concrete Student checkpoint")
    return payload


__all__ = [
    "READY",
    "HUMAN_SCIENTIFIC_INPUT_REQUIRED",
    "HUMAN_OPERATIONAL_INPUT_REQUIRED",
    "build_stage10_deployment_plan",
    "validate_stage10_deployment_plan",
]
