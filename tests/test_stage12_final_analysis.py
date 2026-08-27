import json
from pathlib import Path

import pytest

from validation.run_summary import validate_run_summary_report
from workflow.integrity import artifact_digest


def _artifact(tmp_path, name="artifact.txt"):
    p = tmp_path / name
    p.write_text("x\n")
    return {"path": str(p), "sha256": artifact_digest(p)["sha256"]}


def _report(tmp_path, *, outcome="ALL_STAGES_PASSED", stage_status="completed", gate="PASS",
            artifacts=True, recoveries=None, unresolved=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    evidence_file = tmp_path / "snapshot.json"
    evidence_file.write_text("{}\n")
    payload = {
        "schema_version": 1,
        "run_id": "r",
        "stages": [{"name": "analysis", "status": stage_status, "gate": gate,
                     "artifacts": [_artifact(tmp_path)] if artifacts else []}],
        "gate_history": [{"stage": "analysis", "verdict": gate}],
        "recoveries": recoveries if recoveries is not None else [],
        "campaign_outcome": outcome,
        "unresolved_human_inputs": unresolved if unresolved is not None else [],
        "identified_gaps": [],
        "limitations": [],
        "evidence": [{"role": "run_state_snapshot", "path": str(evidence_file),
                      "integrity": artifact_digest(evidence_file)}],
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(payload))
    return path


def test_run_summary_all_pass_requires_completed_gated_artifacts(tmp_path):
    validate_run_summary_report(_report(tmp_path))
    with pytest.raises(ValueError, match="completed and gated"):
        validate_run_summary_report(_report(tmp_path / "a", stage_status="pending"))
    with pytest.raises(ValueError, match="registered artifact"):
        validate_run_summary_report(_report(tmp_path / "b", artifacts=False))


def test_run_summary_all_pass_blocks_active_recovery_and_unresolved_input(tmp_path):
    with pytest.raises(ValueError, match="active/pending recoveries"):
        validate_run_summary_report(_report(
            tmp_path / "r", recoveries=[{"id": 1, "status": "proposed", "failed_stage": "x"}]))
    with pytest.raises(ValueError, match="unresolved human input"):
        validate_run_summary_report(_report(tmp_path / "h", unresolved=["Stage-9 coverage test"]))


def test_run_summary_deterministic_failure_cannot_be_overridden_by_judges(tmp_path):
    with pytest.raises(ValueError, match="completed and gated"):
        validate_run_summary_report(_report(tmp_path, gate="FAIL"))


def test_executor_generates_markdown_from_same_structured_summary(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_generate_run_summary

    snapshot = tmp_path / "run_state.snapshot.json"
    state = {
        "run_id": "r",
        "stages": [{"name": "analysis", "status": "completed", "gate": "PASS",
                    "artifacts": [_artifact(tmp_path, "analysis-artifact.txt")]}],
        "gate_history": [{"stage": "analysis", "verdict": "PASS", "at": "now"}],
        "recoveries": [],
    }
    snapshot.write_text(json.dumps(state))
    report_path = tmp_path / "analysis.json"
    md_path = tmp_path / "analysis.md"
    res = _exec_generate_run_summary({"parameters": {
        "run_state_path": str(snapshot),
        "report_path": str(report_path),
        "markdown_path": str(md_path),
    }})
    payload = validate_run_summary_report(report_path)
    md = md_path.read_text()
    assert payload["campaign_outcome"] == "ALL_STAGES_PASSED"
    assert "Campaign outcome: `ALL_STAGES_PASSED`" in md
    assert "| analysis | completed | PASS | 1 |" in md
    assert res["markdown_integrity"]["sha256"] == artifact_digest(md_path)["sha256"]


def test_executor_keeps_in_progress_when_required_state_missing(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_generate_run_summary

    snapshot = tmp_path / "run_state.snapshot.json"
    snapshot.write_text(json.dumps({
        "run_id": "r",
        "stages": [{"name": "evaluation", "status": "pending", "gate": "pending", "artifacts": []}],
        "gate_history": [],
        "recoveries": [],
    }))
    report_path = tmp_path / "analysis.json"
    res = _exec_generate_run_summary({"parameters": {
        "run_state_path": str(snapshot),
        "report_path": str(report_path),
    }})
    assert res["report"]["campaign_outcome"] == "IN_PROGRESS"
    assert Path(res["markdown_path"]).is_file()
