#!/usr/bin/env python3
"""Network-free validation of the Stage C golden fixtures + frozen expectations. NO model/GPU.

Confirms, before any live inference, that every golden task is well-formed, portable, and matches
its frozen expectation: validate_task PASS; agent == expected_role; router strategy ==
expected_route_strategy; NO machine-specific absolute paths; artifact reads resolve within the
allow-list with the expected ok flag (the missing-artifact negative case must be genuinely absent);
producer actions are authorized-as-expected (allowed+dry-run-safe, or intentionally approval-gated);
judge ordered_criteria == task criteria. Used by tests/harness/stage_c_golden_shadow.sh + the regression test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = "tests/fixtures/stage_c_golden"


def _load(repo_root):
    root = Path(repo_root).resolve()
    gold = json.loads((root / BASE / "golden_expectations.json").read_text())
    return root, gold


def validate_all(repo_root):
    root, gold = _load(repo_root)
    from orchestration.specs import load_agent_specs
    from orchestration.exchange import validate_task
    from runtimes.pydantic_ai.production_router import acceptance_strategy
    from runtimes.pydantic_ai.actions import (
        ROLE_ALLOWED_ACTIONS, APPROVAL_GATED_ACTIONS, CAPABILITY_REGISTRY)
    from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset, ToolAccessError

    specs = load_agent_specs(str(root / "agent_specs"))
    art_dir = root / BASE / "artifacts"
    msgs, ok = [], True

    def check(cond, m):
        nonlocal ok
        msgs.append(("OK   " if cond else "FAIL ") + m); ok = ok and bool(cond)

    for tid, exp in sorted(gold.items()):
        tp = root / BASE / "tasks" / f"{tid}.json"
        check(tp.exists(), f"{tid}: task fixture exists")
        if not tp.exists():
            continue
        task = json.loads(tp.read_text())
        role = exp["expected_role"]
        spec = specs.get(role)
        check(spec is not None, f"{tid}: role '{role}' is a known spec")
        if spec is None:
            continue
        try:
            validate_task(task, spec); check(True, f"{tid}: validate_task PASS")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{tid}: validate_task -> {type(exc).__name__}: {exc}")
        check(task.get("agent") == role, f"{tid}: agent == '{role}'")
        check(acceptance_strategy(spec) == exp["expected_route_strategy"],
              f"{tid}: strategy == {exp['expected_route_strategy']}")
        # portability: no machine-specific absolute paths anywhere in the task
        blob = json.dumps(task)
        check("/tmp/" not in blob and "/home/" not in blob,
              f"{tid}: no absolute /tmp or /home path (portable)")

        # artifact reads resolve (or are intentionally absent) within the allow-list
        for aref in exp.get("expected_artifact_reads", []):
            rel, expect_ok = aref["path"], aref["ok"]
            abspath = root / rel
            if expect_ok:
                check(abspath.exists(), f"{tid}: expected-readable artifact exists: {rel}")
                ts = ReadOnlyToolset([str(art_dir)])
                cwd = os.getcwd()
                try:
                    os.chdir(root)
                    val = ts.read_json(rel)
                    check(isinstance(val, dict), f"{tid}: {rel} reads as JSON within allow-list")
                except ToolAccessError as exc:
                    check(False, f"{tid}: {rel} refused: {exc}")
                finally:
                    os.chdir(cwd)
            else:
                check(not abspath.exists(),
                      f"{tid}: negative-case artifact is genuinely ABSENT: {rel}")

        # producer action authorization matches the expectation
        action = exp.get("expected_action_type")
        if action is not None and exp["expected_route_strategy"] == "producer_dispatch":
            check(action in ROLE_ALLOWED_ACTIONS.get(role, set()),
                  f"{tid}: action '{action}' is in the role's proposable set")
            if exp.get("expected_outcome") == "DRY_RUN":
                check(action not in APPROVAL_GATED_ACTIONS and action not in CAPABILITY_REGISTRY,
                      f"{tid}: allowed action '{action}' is dry-run-safe (not gated/out-of-scope)")
            elif exp.get("expected_outcome") == "APPROVAL_REQUIRED":
                gated = action in APPROVAL_GATED_ACTIONS or (
                    action in CAPABILITY_REGISTRY and
                    getattr(CAPABILITY_REGISTRY[action], "status", "") == "APPROVAL_REQUIRED")
                check(gated, f"{tid}: negative action '{action}' is genuinely approval-gated")

        # judge: ordered criteria must equal the task's criteria; lens present
        if role == "judge":
            check(exp.get("ordered_criteria") == task.get("criteria"),
                  f"{tid}: golden ordered_criteria == task.criteria")
            ctx = task.get("context", {})
            check(bool(ctx.get("review_lens")) and bool(ctx.get("review_focus")),
                  f"{tid}: judge context has review_lens + review_focus")

        # expectation schema completeness
        for field in ("expected_role", "expected_route_strategy", "expected_controller_mutation",
                      "expected_paid_api_calls", "expected_fabricated_sources", "forbidden_tools"):
            check(field in exp, f"{tid}: expectation has '{field}'")
        check(exp.get("expected_controller_mutation") is False, f"{tid}: expected_controller_mutation == false")
        check(exp.get("expected_paid_api_calls") == 0, f"{tid}: expected_paid_api_calls == 0")
        check(exp.get("expected_fabricated_sources") == 0, f"{tid}: expected_fabricated_sources == 0")

    check(len(gold) >= 8, f"golden set has >= 8 tasks (have {len(gold)})")
    return ok, msgs


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msgs = validate_all(root)
    print("\n".join(msgs))
    print("STAGE_C_FIXTURES:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
