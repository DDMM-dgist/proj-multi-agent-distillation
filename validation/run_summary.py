"""Deterministic contract for end-of-campaign run-summary reports.

Mirrors the structural-contract style of ``validation.uncertainty`` / ``validation.data_coverage``:
validates shape and provenance only. Every stage/gate/artifact fact in the report must be a
byte-exact mirror of the CURRENT ``RunController`` state (assembled mechanically by
``runtimes.pydantic_ai.cli._assemble_run_summary_state``), never a free-form Analyst narrative
substituting for it.
"""
import json
from pathlib import Path

from validation.report import validate_evidence

STAGE_STATUSES = {"pending", "running", "completed", "failed", "interrupted", "not_applicable"}
GATE_STATUSES = {"pending", "PASS", "REVISE", "FAIL", "NOT_APPLICABLE"}


def validate_run_summary_report(manifest_path, submitted_artifacts=None, allowed_evidence=None,
                                enforce_required_pass=False):
    """Validate a run-summary report's shape and provenance.

    Requires: a run_id, a non-empty stage list with valid status/gate and hash-bound artifact
    references, a gate_history list of real recorded verdicts, a recoveries list, and an explicit
    campaign_outcome string. Never itself re-derives these facts from the run_dir -- that is the
    assembler's job; this only checks the shape the assembler is contracted to produce.
    """
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("run summary requires schema_version=1")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise ValueError("run summary requires a non-empty run_id")

    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("run summary requires a non-empty stages list")
    seen_stages = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"stages[{index}] must be an object")
        name = stage.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"stages[{index}].name must be a non-empty string")
        if name in seen_stages:
            raise ValueError(f"run summary stage is duplicated: {name}")
        seen_stages.add(name)
        if stage.get("status") not in STAGE_STATUSES:
            raise ValueError(f"stages[{index}].status must be one of {sorted(STAGE_STATUSES)}")
        if stage.get("gate") not in GATE_STATUSES:
            raise ValueError(f"stages[{index}].gate must be one of {sorted(GATE_STATUSES)}")
        artifacts = stage.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(f"stages[{index}].artifacts must be a list")
        for a_index, artifact in enumerate(artifacts):
            if (not isinstance(artifact, dict) or
                    not isinstance(artifact.get("path"), str) or not artifact["path"].strip() or
                    not isinstance(artifact.get("sha256"), str) or not artifact["sha256"].strip()):
                raise ValueError(f"stages[{index}].artifacts[{a_index}] requires path and sha256")

    gate_history = payload.get("gate_history")
    if not isinstance(gate_history, list):
        raise ValueError("run summary requires a gate_history list")
    for index, event in enumerate(gate_history):
        if (not isinstance(event, dict) or not isinstance(event.get("stage"), str) or
                not event["stage"].strip() or event.get("verdict") not in GATE_STATUSES):
            raise ValueError(f"gate_history[{index}] requires stage and a valid verdict")

    recoveries = payload.get("recoveries")
    if not isinstance(recoveries, list):
        raise ValueError("run summary requires a recoveries list")
    for index, recovery in enumerate(recoveries):
        if not isinstance(recovery, dict) or "id" not in recovery or "status" not in recovery:
            raise ValueError(f"recoveries[{index}] requires id and status")

    outcome = payload.get("campaign_outcome")
    if not isinstance(outcome, str) or not outcome.strip():
        raise ValueError("run summary requires a non-empty campaign_outcome")
    unresolved = payload.get("unresolved_human_inputs", [])
    if not isinstance(unresolved, list):
        raise ValueError("run summary unresolved_human_inputs must be a list")
    if outcome == "ALL_STAGES_PASSED":
        bad_stages = [stage["name"] for stage in stages
                      if stage["status"] not in {"completed", "not_applicable"}
                      or stage["gate"] not in {"PASS", "NOT_APPLICABLE"}]
        if bad_stages:
            raise ValueError("ALL_STAGES_PASSED requires every required stage completed and gated")
        missing_artifacts = [stage["name"] for stage in stages
                             if stage["status"] == "completed" and not stage.get("artifacts")]
        if missing_artifacts:
            raise ValueError("ALL_STAGES_PASSED requires registered artifact hashes for completed stages")
        active_recoveries = [r for r in recoveries
                             if r.get("status") not in {"superseded", "resolved", "completed", "cancelled", "withdrawn"}]
        if active_recoveries:
            raise ValueError("ALL_STAGES_PASSED cannot be reported with active/pending recoveries")
        if unresolved:
            raise ValueError("ALL_STAGES_PASSED cannot be reported with unresolved human input")

    validate_evidence(manifest_path, payload.get("evidence"), submitted_artifacts, False,
                      allowed_evidence, label="run_summary")
    for field in ("identified_gaps", "limitations"):
        value = payload.get(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"run summary {field} must be a list of strings")
    return payload
