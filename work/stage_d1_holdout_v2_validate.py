#!/usr/bin/env python3
"""Network-free validation of the STAGE D-1 HOLDOUT V2 package (architecture v2). NO model/GPU.

Checks: the built cases == the frozen SELECTION_MANIFEST; tasks well-formed + portable; evidence
metrics-only with no verdict leak; criteria == frozen ordered_criteria; specs use ONLY frozen
operators + generic keys; authoritative block attaches and equals a fresh evaluation; recorded
deterministic predictions match; expected model qwen2.5-7b-instruct. Plus a v2 axis-A check: an
ADVERSARIAL LLM vote, passed through the real acceptance path (validate_agent_response), is BOUND to
the deterministic severity — proving the accepted verdict is policy-owned before any live run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = "examples/stage_d1_holdout_v2"
VERDICTS = {"PASS", "REVISE", "FAIL"}
_LEAK = {"verdict", "judge_decision", "decision", "historical_verdict", "gate_decision", "pass"}
_SPEC_KEYS = {"criterion", "operator", "lhs", "rhs", "invalidating", "all", "any"}
EXPECTED_MODEL = "qwen2.5-7b-instruct"
_WRONG = {"PASS": "FAIL", "REVISE": "FAIL", "FAIL": "PASS"}


def _ops(item, acc):
    for comb in ("all", "any"):
        for sub in item.get(comb, []):
            _ops(sub, acc)
    if "operator" in item:
        acc.add(item["operator"])


def validate_all(repo_root):
    root = Path(repo_root).resolve()
    gold = json.loads((root / BASE / "golden_decisions.json").read_text())
    manifest = json.loads((root / BASE / "SELECTION_MANIFEST.json").read_text())
    preds = json.loads((root / BASE / "DETERMINISTIC_PREDICTIONS.json").read_text())
    from orchestration.specs import load_agent_specs
    from orchestration.exchange import validate_task, validate_agent_response
    from runtimes.pydantic_ai.production_router import acceptance_strategy
    from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset, ToolAccessError
    from runtimes.pydantic_ai.criterion_eval import _OPERATORS, evaluate_criteria, derive_severity
    specs = load_agent_specs(str(root / "agent_specs")); spec = specs["judge"]
    ev_dir = root / BASE / "evidence"
    msgs, ok = [], True

    def check(cond, m):
        nonlocal ok
        msgs.append(("OK   " if cond else "FAIL ") + m); ok = ok and bool(cond)

    # selection integrity: built golden cases correspond 1:1 to the frozen manifest
    man_targets = {m["selection_hash"] for m in manifest}
    gold_hashes = {exp.get("selection_hash") for exp in gold.values()}
    check(len(manifest) == len(gold) == 7, f"7 cases == manifest ({len(manifest)}/{len(gold)})")
    check(man_targets == gold_hashes, "golden cases match the frozen selection hashes")

    for cid, exp in sorted(gold.items()):
        tp = root / BASE / "tasks" / f"{cid}.json"; cp = root / BASE / "criteria" / f"{cid}.json"
        check(tp.exists() and cp.exists(), f"{cid}: task + criteria exist")
        if not (tp.exists() and cp.exists()):
            continue
        task = json.loads(tp.read_text()); cspec = json.loads(cp.read_text())
        try:
            validate_task(task, spec); check(True, f"{cid}: validate_task PASS")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{cid}: validate_task -> {type(exc).__name__}: {exc}")
        check(task.get("agent") == "judge", f"{cid}: agent == judge")
        check(acceptance_strategy(spec) == "judge_gate", f"{cid}: strategy judge_gate")
        check("/tmp/" not in json.dumps(task) and "/home/" not in json.dumps(task), f"{cid}: portable")
        check(task.get("criteria") == exp.get("ordered_criteria"), f"{cid}: criteria == ordered_criteria")
        ops = set(); keys_ok = True
        for item in cspec:
            _ops(item, ops); keys_ok = keys_ok and (set(item) - _SPEC_KEYS == set())
        check(not (ops - _OPERATORS), f"{cid}: frozen operators only (new: {sorted(ops - _OPERATORS)})")
        check(keys_ok, f"{cid}: generic spec keys only")
        rel = exp["evidence_file"]; abspath = root / rel
        check(abspath.exists(), f"{cid}: evidence exists")
        if abspath.exists():
            ev = json.loads(abspath.read_text())
            check(not (_LEAK & {str(k).lower() for k in ev}), f"{cid}: metrics-only (no verdict leak)")
            ts = ReadOnlyToolset([str(ev_dir)]); cwd = os.getcwd()
            try:
                os.chdir(root); ts.read_json(rel); check(True, f"{cid}: evidence reads within allow-list")
            except ToolAccessError as exc:
                check(False, f"{cid}: evidence refused: {exc}")
            finally:
                os.chdir(cwd)
            results = evaluate_criteria(ev, cspec)
            ctx = task["context"]; block = ctx.get("deterministic_criterion_results")
            check(block is not None and [b["result"] for b in block] == [r.result for r in results],
                  f"{cid}: authoritative block == fresh evaluation")
            check(ctx.get("deterministic_authoritative") is True, f"{cid}: authoritative mode set")
            sev = derive_severity(results)
            check(ctx.get("deterministic_suggested_severity") == sev, f"{cid}: block severity == fresh")
            check(preds.get(cid, {}).get("deterministic_suggested_severity") == sev,
                  f"{cid}: recorded prediction == fresh ({sev})")
            # v2 AXIS-A: an adversarial vote is bound to the deterministic verdict by the real path
            det_ok = [b["result"] for b in block]
            adv = {"review_lens": ctx["review_lens"], "verdict": _WRONG[sev],
                   "criteria_checked": [{"criterion": c, "value_read": "adv", "ok": (not det_ok[i])}
                                        for i, c in enumerate(task["criteria"])],
                   "rationale": "adversarial.", "required_fix": "adv."}
            bound = validate_agent_response(adv, spec, task)
            check(bound["verdict"] == sev, f"{cid}: acceptance BINDS adversarial vote -> {sev}")
        check(exp.get("historical_verdict") in VERDICTS, f"{cid}: historical_verdict valid")
        av = set(exp.get("acceptable_verdicts", []))
        check(bool(av) and av <= VERDICTS and exp.get("historical_verdict") in av,
              f"{cid}: acceptable_verdicts valid")
        if exp.get("must_not_pass"):
            check("PASS" not in av, f"{cid}: must_not_pass => PASS not acceptable")
        check(bool(exp.get("source")), f"{cid}: source recorded")
    return ok, msgs


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msgs = validate_all(root)
    print("\n".join(msgs))
    print("STAGE_D1_HOLDOUT_V2_FIXTURES:", "PASS" if ok else "FAIL")
    print("EXPECTED_MODEL:", EXPECTED_MODEL)
    sys.exit(0 if ok else 1)
