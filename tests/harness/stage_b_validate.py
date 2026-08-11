#!/usr/bin/env python3
"""Network-free validation of the Stage B frozen fixtures. NO provider, NO model, NO GPU.

Exits 0 iff all seven role tasks are well-formed and portable:
  - validate_task PASS (canonical task schema; judge also needs review_lens + review_focus)
  - agent field == role, and the router acceptance strategy is the expected one
  - judge: the repo-relative evidence path resolves inside the read allow-list and reads back
  - producers: the intended action is allowed for the role, NOT approval-gated, and not an
    out-of-scope/unavailable capability (so a dry-run dispatch yields DRY_RUN with no side effect)

Used by tests/harness/stage_b_local_smoke.sh (pre-launch gate) and tests/test_pydantic_ai_stage_b_fixtures.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIX_DIR = "tests/fixtures/stage_b_smoke"

# role -> (fixture filename, expected acceptance strategy, expected producer action or None)
STAGE_B_TASKS = {
    "orchestrator": ("orchestrator.json", "typed_result", None),
    "literature":   ("literature.json",   "typed_result", None),
    "data-curator": ("data-curator.json", "producer_dispatch", "inspect_dataset"),
    "ml-trainer":   ("ml-trainer.json",   "producer_dispatch", "compute_committee_disagreement"),
    "simulation":   ("simulation.json",   "producer_dispatch", "compute_nve_drift"),
    "analyst":      ("analyst.json",      "producer_dispatch", "compare_force_errors"),
    "judge":        ("judge.json",        "judge_gate", None),
}


def validate_all(repo_root):
    root = Path(repo_root).resolve()
    from orchestration.specs import load_agent_specs
    from orchestration.exchange import validate_task
    from runtimes.pydantic_ai.production_router import acceptance_strategy
    from runtimes.pydantic_ai.actions import (
        ROLE_ALLOWED_ACTIONS, APPROVAL_GATED_ACTIONS, CAPABILITY_REGISTRY)
    from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset, ToolAccessError

    specs = load_agent_specs(str(root / "agent_specs"))
    msgs, ok = [], True

    def check(cond, m):
        nonlocal ok
        msgs.append(("OK   " if cond else "FAIL ") + m)
        ok = ok and bool(cond)

    for role, (fname, strat, action) in STAGE_B_TASKS.items():
        p = root / FIX_DIR / fname
        check(p.exists(), f"{role}: fixture {fname} exists")
        if not p.exists():
            continue
        task = json.loads(p.read_text())
        spec = specs[role]
        try:
            validate_task(task, spec)
            check(True, f"{role}: validate_task PASS")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{role}: validate_task -> {type(exc).__name__}: {exc}")
        check(task.get("agent") == role, f"{role}: agent field == '{role}'")
        check(acceptance_strategy(spec) == strat, f"{role}: acceptance strategy == '{strat}'")

        if role == "judge":
            ctx = task.get("context", {})
            check(bool(ctx.get("review_lens")) and bool(ctx.get("review_focus")),
                  "judge: context has non-empty review_lens + review_focus")
            art_dir = root / FIX_DIR / "artifacts"
            ts = ReadOnlyToolset([str(art_dir)])
            rel = f"{FIX_DIR}/artifacts/evidence.json"
            cwd = os.getcwd()
            try:
                os.chdir(root)  # mirror runtime: model calls the repo-relative path, cwd==repo
                val = ts.read_json(rel)
                check(val.get("structure_count") == 12,
                      "judge: evidence.json reads structure_count==12 within the allow-list")
            except ToolAccessError as exc:
                check(False, f"judge: evidence read refused: {exc}")
            finally:
                os.chdir(cwd)

        if action is not None:
            check(action in ROLE_ALLOWED_ACTIONS.get(role, set()),
                  f"{role}: action '{action}' is allowed for the role")
            check(action not in APPROVAL_GATED_ACTIONS,
                  f"{role}: action '{action}' is NOT approval-gated (safe dry-run)")
            check(action not in CAPABILITY_REGISTRY,
                  f"{role}: action '{action}' is not an out-of-scope/unavailable capability")
    return ok, msgs


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msgs = validate_all(root)
    print("\n".join(msgs))
    print("STAGE_B_FIXTURES:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
