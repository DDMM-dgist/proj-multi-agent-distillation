"""Stage D-2 C1 advisory-Judge APPEND-ONLY records + semantic-transition mapping.

deterministic_authoritative=false: the advisory semantic verdict is GENUINE and is NOT rebound to the
Axis-A PASS. This module builds the append-only attempt artifacts from a Judge provenance record and
maps the advisory verdict to the workflow transition. Pure; NO network/model/GPU. Used by the Judge
runner (not invoked here) and by tests.

Preservation contract (enforced by ``assert_appendonly``): the historical DEFERRED
``judge_interpretation.json`` and every Axis-A scientific artifact are NEVER overwritten; the first
real Judge attempt writes FRESH names only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# advisory Judge verdict -> STAGE_D2_C1_TRANSITION (genuine; not forced toward PASS)
SEMANTIC_TRANSITION = {"PASS": "ADVANCE", "REVISE": "REVISE", "FAIL": "FAIL_STOP"}

PRESERVE_BYTE_IDENTICAL = ("msd.csv", "msd_summary.json", "criterion_results.json", "approval.json",
                           "judge_interpretation.json", "execution_wrapper_snapshot.py")
ATTEMPT1 = ("judge_interpretation_attempt1.json", "judge_provenance_attempt1.json",
            "semantic_transition_attempt1.json")


def semantic_transition(advisory_verdict: str) -> str:
    """Map the advisory verdict to the transition. Unknown/None -> conservative FAIL_STOP."""
    return SEMANTIC_TRANSITION.get(advisory_verdict, "FAIL_STOP")


def assert_appendonly(run_dir) -> None:
    """Refuse to proceed unless the append-only preconditions hold: the deferred interpretation exists
    (to be preserved) and NONE of the attempt1 filenames exist yet (fresh only)."""
    rd = Path(run_dir)
    if not (rd / "judge_interpretation.json").is_file():
        raise FileNotFoundError("historical deferred judge_interpretation.json missing — refuse")
    existing = [f for f in ATTEMPT1 if (rd / f).exists()]
    if existing:
        raise FileExistsError(f"append-only violation: attempt artifacts already exist: {existing}")


def snapshot_hashes(run_dir) -> dict:
    rd = Path(run_dir)
    return {f: hashlib.sha256((rd / f).read_bytes()).hexdigest()
            for f in PRESERVE_BYTE_IDENTICAL if (rd / f).is_file()}


def assert_preserved(run_dir, before: dict) -> None:
    """Verify every preserved artifact is byte-identical to ``before`` (post-Judge integrity check)."""
    after = snapshot_hashes(run_dir)
    drift = [f for f in before if after.get(f) != before[f]]
    if drift:
        raise AssertionError(f"preserved artifacts changed (history rewritten): {drift}")


def build_attempt_records(judge_prov: dict, *, axis_a_verdict: str = "PASS") -> tuple:
    """From a Judge attempt provenance dict, build (interpretation, provenance, semantic) attempt-1
    records. The advisory verdict is the LLM's GENUINE verdict — not rebound to Axis-A."""
    parsed = judge_prov.get("parsed_result") or {}
    verdict = parsed.get("verdict")
    transition = semantic_transition(verdict)
    canonical_ok = (judge_prov.get("validation_errors") in ([], None))
    interpretation = {
        "attempt": 1, "deterministic_authoritative": False,
        "advisory_verdict": verdict, "criteria_checked": parsed.get("criteria_checked"),
        "rationale": parsed.get("rationale"), "required_fix": parsed.get("required_fix"),
        "criterion_contradictions": len(judge_prov.get("criterion_contradictions") or []),
        "axis_a_authoritative_verdict": f"{axis_a_verdict} (preserved; advisory verdict NOT rebound to it)",
        "note": "advisory semantic verdict is genuine (deterministic_authoritative=false).",
    }
    provenance = {
        "attempt": 1, "provider": judge_prov.get("provider"), "model_id": judge_prov.get("model_id"),
        "usage_source": judge_prov.get("usage_source"), "prompt_sha256": judge_prov.get("prompt_sha256"),
        "tool_invocations": judge_prov.get("tool_invocations"),
        "validation_errors": judge_prov.get("validation_errors"), "canonical_validation_ok": canonical_ok,
        "prompt_tokens": judge_prov.get("prompt_tokens"), "completion_tokens": judge_prov.get("completion_tokens"),
        "latency_s": judge_prov.get("latency_s"), "attempt_id": judge_prov.get("attempt_id"),
        "retry_category": judge_prov.get("retry_category"), "parent_attempt_id": judge_prov.get("parent_attempt_id"),
        "accepted_verdict": judge_prov.get("accepted_verdict"),
        "verdict_overridden": bool(judge_prov.get("verdict_overridden")),
    }
    semantic = {
        "stage": "stage_d2_c1_semantic", "attempt": 1,
        "STAGE_D2_C1_AXIS_A": axis_a_verdict,
        "advisory_judge_verdict": verdict, "STAGE_D2_C1_TRANSITION": transition,
        "references": {"original_run_provenance": "provenance.json",
                       "axis_a_criterion_results": "criterion_results.json",
                       "judge_attempt_provenance": "judge_provenance_attempt1.json",
                       "advisory_interpretation": "judge_interpretation_attempt1.json"},
        "note": ("advisory (deterministic_authoritative=false); PASS->ADVANCE, REVISE->REVISE, "
                 "FAIL->FAIL_STOP; not forced toward PASS. pbc_hard_guarantee=false must be honored."),
    }
    return interpretation, provenance, semantic


def write_attempt_records(run_dir, judge_prov: dict, *, axis_a_verdict: str = "PASS") -> dict:
    """Append-only write of the three attempt-1 records + run_manifest.after_judge.json. Refuses if any
    attempt file exists; verifies preserved artifacts are byte-identical afterward. Never touches the
    deferred judge_interpretation.json or any Axis-A scientific artifact."""
    rd = Path(run_dir)
    assert_appendonly(rd)
    before = snapshot_hashes(rd)
    interp, prov, semantic = build_attempt_records(judge_prov, axis_a_verdict=axis_a_verdict)
    (rd / "judge_interpretation_attempt1.json").write_text(json.dumps(interp, indent=2) + "\n")
    (rd / "judge_provenance_attempt1.json").write_text(json.dumps(prov, indent=2) + "\n")
    (rd / "semantic_transition_attempt1.json").write_text(json.dumps(semantic, indent=2) + "\n")
    consolidated = {"_note": "NEW consolidated manifest; does NOT overwrite the original run_manifest.json",
                    "STAGE_D2_C1_AXIS_A": axis_a_verdict,
                    "STAGE_D2_C1_SEMANTIC_JUDGE": semantic["advisory_judge_verdict"],
                    "STAGE_D2_C1_TRANSITION": semantic["STAGE_D2_C1_TRANSITION"],
                    "attempt_artifacts": list(ATTEMPT1)}
    (rd / "run_manifest.after_judge.json").write_text(json.dumps(consolidated, indent=2) + "\n")
    assert_preserved(rd, before)
    return semantic
