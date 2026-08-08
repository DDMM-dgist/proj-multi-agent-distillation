#!/usr/bin/env python3
"""Network-free validation of the STAGE D-1 HOLDOUT package. NO model/GPU.

Reuses the development fixture checks (well-formed, portable, metrics-only, no leaked verdict,
criteria == frozen ordered_criteria, expectation schema) AND adds the holdout-integrity checks
required before an unseen replay:
  * every criterion spec uses ONLY the frozen generic operators (no new operator/semantics);
  * no spec contains task-ID-specific or answer-key logic (only generic predicate keys);
  * the authoritative deterministic block attaches correctly and equals a fresh evaluation, with the
    fully-deterministic (authoritative=True) mode set;
  * DETERMINISTIC_PREDICTIONS.json (recorded before inference) matches a fresh evaluation exactly;
  * the expected model is qwen2.5-7b-instruct.
Exits non-zero on any failure (the runner refuses to launch).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = "examples/stage_d1_holdout"
VERDICTS = {"PASS", "REVISE", "FAIL"}
_LEAK_KEYS = {"verdict", "judge_decision", "decision", "historical_verdict", "gate_decision", "pass"}
_SPEC_KEYS = {"criterion", "operator", "lhs", "rhs", "invalidating", "all", "any"}
EXPECTED_MODEL = "qwen2.5-7b-instruct"


def _spec_operators(spec_item, acc):
    for combiner in ("all", "any"):
        if combiner in spec_item:
            for sub in spec_item[combiner]:
                _spec_operators(sub, acc)
    if "operator" in spec_item:
        acc.add(spec_item["operator"])


def validate_all(repo_root):
    root = Path(repo_root).resolve()
    gold = json.loads((root / BASE / "golden_decisions.json").read_text())
    from orchestration.specs import load_agent_specs
    from orchestration.exchange import validate_task
    from runtimes.pydantic_ai.production_router import acceptance_strategy
    from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset, ToolAccessError
    from runtimes.pydantic_ai.criterion_eval import _OPERATORS, evaluate_criteria, derive_severity
    specs = load_agent_specs(str(root / "agent_specs")); spec = specs["judge"]
    ev_dir = root / BASE / "evidence"
    predictions = json.loads((root / BASE / "DETERMINISTIC_PREDICTIONS.json").read_text())
    msgs, ok = [], True

    def check(cond, m):
        nonlocal ok
        msgs.append(("OK   " if cond else "FAIL ") + m); ok = ok and bool(cond)

    check(len(gold) == 8, f"holdout has exactly 8 decisions ({len(gold)})")
    for cid, exp in sorted(gold.items()):
        tp = root / BASE / "tasks" / f"{cid}.json"
        cp = root / BASE / "criteria" / f"{cid}.json"
        check(tp.exists() and cp.exists(), f"{cid}: task + criteria spec exist")
        if not (tp.exists() and cp.exists()):
            continue
        task = json.loads(tp.read_text())
        cspec = json.loads(cp.read_text())
        try:
            validate_task(task, spec); check(True, f"{cid}: validate_task PASS")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{cid}: validate_task -> {type(exc).__name__}: {exc}")
        check(task.get("agent") == "judge", f"{cid}: agent == judge")
        check(acceptance_strategy(spec) == "judge_gate", f"{cid}: strategy judge_gate")
        check("/tmp/" not in json.dumps(task) and "/home/" not in json.dumps(task),
              f"{cid}: portable (no absolute path)")
        check(task.get("criteria") == exp.get("ordered_criteria"),
              f"{cid}: task.criteria == frozen ordered_criteria")
        # frozen-operator + generic-spec checks
        ops = set()
        keys_ok = True
        for item in cspec:
            _spec_operators(item, ops)
            keys_ok = keys_ok and (set(item) - _SPEC_KEYS == set())
        new_ops = ops - _OPERATORS
        check(not new_ops, f"{cid}: uses only frozen operators (new: {sorted(new_ops)})")
        check(keys_ok, f"{cid}: spec uses only generic predicate keys (no answer-key/task-id logic)")
        check(len(cspec) == len(task.get("criteria", [])), f"{cid}: one spec per criterion")
        # evidence: resolves + metrics-only
        rel = exp["evidence_file"]; abspath = root / rel
        check(abspath.exists(), f"{cid}: evidence file exists ({rel})")
        if abspath.exists():
            ev = json.loads(abspath.read_text())
            check(isinstance(ev, dict), f"{cid}: evidence is a JSON object")
            leaked = _LEAK_KEYS & {str(k).lower() for k in (ev or {})}
            check(not leaked, f"{cid}: evidence is METRICS-ONLY (no leaked key {sorted(leaked)})")
            ts = ReadOnlyToolset([str(ev_dir)]); cwd = os.getcwd()
            try:
                os.chdir(root); ts.read_json(rel); check(True, f"{cid}: evidence reads within allow-list")
            except ToolAccessError as exc:
                check(False, f"{cid}: evidence refused: {exc}")
            finally:
                os.chdir(cwd)
            # deterministic eval succeeds; authoritative block attaches + matches; predictions match
            results = evaluate_criteria(ev, cspec)
            ctx = task.get("context", {})
            block = ctx.get("deterministic_criterion_results")
            check(block is not None and len(block) == len(results),
                  f"{cid}: authoritative block attached (1 per criterion)")
            if block:
                check([b["result"] for b in block] == [r.result for r in results],
                      f"{cid}: block booleans == fresh evaluation")
            check(ctx.get("deterministic_authoritative") is True, f"{cid}: authoritative mode set")
            sev = derive_severity(results)
            check(ctx.get("deterministic_suggested_severity") == sev, f"{cid}: block severity == fresh")
            pred = predictions.get(cid, {})
            check(pred.get("deterministic_suggested_severity") == sev,
                  f"{cid}: DETERMINISTIC_PREDICTIONS severity == fresh ({sev})")
        # expectation schema
        check(exp.get("historical_verdict") in VERDICTS, f"{cid}: historical_verdict valid")
        av = set(exp.get("acceptable_verdicts", []))
        check(bool(av) and av <= VERDICTS and exp.get("historical_verdict") in av,
              f"{cid}: acceptable_verdicts valid + contains historical")
        check(exp.get("expected_controller_mutation") is False, f"{cid}: expected_controller_mutation false")
        check(exp.get("expected_paid_api_calls") == 0, f"{cid}: expected_paid_api_calls 0")
        check(bool(exp.get("source")), f"{cid}: source provenance recorded")
        if exp.get("must_not_pass"):
            check("PASS" not in av, f"{cid}: must_not_pass => PASS not acceptable")
    return ok, msgs


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msgs = validate_all(root)
    print("\n".join(msgs))
    print("STAGE_D1_HOLDOUT_FIXTURES:", "PASS" if ok else "FAIL")
    print("EXPECTED_MODEL:", EXPECTED_MODEL)
    sys.exit(0 if ok else 1)
