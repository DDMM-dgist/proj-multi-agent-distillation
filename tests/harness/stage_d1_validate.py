#!/usr/bin/env python3
"""Network-free validation of the Stage D-1 auditable-decision replay fixtures. NO model/GPU.

Confirms each checkpoint is well-formed and portable and that the evidence is METRICS-ONLY (the
historical verdict must NOT leak into what the agent reads): validate_task PASS; agent==judge;
router strategy judge_gate; no machine-specific absolute path; evidence file resolves within the
allow-list and reads as a JSON object; the evidence carries NO verdict/decision key; task criteria
== the frozen ordered_criteria; review_lens+focus present; expectation schema complete with a
valid historical_verdict and non-empty acceptable_verdicts. Used by the runner + regression tests.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = "tests/fixtures/stage_d1_replay"
VERDICTS = {"PASS", "REVISE", "FAIL"}
_LEAK_KEYS = {"verdict", "judge_decision", "decision", "historical_verdict", "gate_decision", "pass"}


def validate_all(repo_root):
    root = Path(repo_root).resolve()
    gold = json.loads((root / BASE / "golden_decisions.json").read_text())
    from orchestration.specs import load_agent_specs
    from orchestration.exchange import validate_task
    from runtimes.pydantic_ai.production_router import acceptance_strategy
    from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset, ToolAccessError
    specs = load_agent_specs(str(root / "agent_specs")); spec = specs["judge"]
    ev_dir = root / BASE / "evidence"
    msgs, ok = [], True

    def check(cond, m):
        nonlocal ok
        msgs.append(("OK   " if cond else "FAIL ") + m); ok = ok and bool(cond)

    check(8 <= len(gold) or len(gold) >= 7, f"checkpoint count is compact ({len(gold)})")
    for cid, exp in sorted(gold.items()):
        tp = root / BASE / "tasks" / f"{cid}.json"
        check(tp.exists(), f"{cid}: task exists")
        if not tp.exists():
            continue
        task = json.loads(tp.read_text())
        try:
            validate_task(task, spec); check(True, f"{cid}: validate_task PASS")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{cid}: validate_task -> {type(exc).__name__}: {exc}")
        check(task.get("agent") == "judge", f"{cid}: agent == judge")
        check(acceptance_strategy(spec) == "judge_gate", f"{cid}: strategy judge_gate")
        check("/tmp/" not in json.dumps(task) and "/home/" not in json.dumps(task),
              f"{cid}: portable (no absolute path)")
        ctx = task.get("context", {})
        check(bool(ctx.get("review_lens")) and bool(ctx.get("review_focus")),
              f"{cid}: review_lens + review_focus present")
        check(task.get("criteria") == exp.get("ordered_criteria"),
              f"{cid}: task.criteria == frozen ordered_criteria")
        # evidence: resolves + reads + is metrics-only (no leaked verdict)
        rel = exp["evidence_file"]
        abspath = root / rel
        check(abspath.exists(), f"{cid}: evidence file exists ({rel})")
        if abspath.exists():
            ev = json.loads(abspath.read_text())
            check(isinstance(ev, dict), f"{cid}: evidence is a JSON object")
            leaked = _LEAK_KEYS & {str(k).lower() for k in (ev or {})}
            check(not leaked, f"{cid}: evidence is METRICS-ONLY (no leaked verdict key {sorted(leaked)})")
            ts = ReadOnlyToolset([str(ev_dir)]); cwd = os.getcwd()
            try:
                os.chdir(root); ts.read_json(rel)
                check(True, f"{cid}: evidence reads within the allow-list")
            except ToolAccessError as exc:
                check(False, f"{cid}: evidence refused: {exc}")
            finally:
                os.chdir(cwd)
        # expectation schema
        check(exp.get("historical_verdict") in VERDICTS, f"{cid}: historical_verdict valid")
        av = set(exp.get("acceptable_verdicts", []))
        check(bool(av) and av <= VERDICTS, f"{cid}: acceptable_verdicts subset of {VERDICTS}")
        check(exp.get("historical_verdict") in av, f"{cid}: historical_verdict in acceptable_verdicts")
        check(exp.get("expected_controller_mutation") is False, f"{cid}: expected_controller_mutation false")
        check(exp.get("expected_paid_api_calls") == 0, f"{cid}: expected_paid_api_calls 0")
        check(bool(exp.get("source")), f"{cid}: source provenance recorded")
        # a must_not_pass checkpoint must not list PASS as acceptable
        if exp.get("must_not_pass"):
            check("PASS" not in av, f"{cid}: must_not_pass => PASS not acceptable")
    return ok, msgs


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msgs = validate_all(root)
    print("\n".join(msgs))
    print("STAGE_D1_FIXTURES:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
