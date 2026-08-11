#!/usr/bin/env python3
"""Attach the DETERMINISTIC criterion results to each Stage D-1 Judge task (integration step).

For every checkpoint: load its frozen evidence + generic criterion spec, run the deterministic
evaluator (runtimes/pydantic_ai/criterion_eval.py), and inject the typed results into the task's
``context`` as authoritative facts via criterion_eval.attach_to_task. This places the deterministic
layer UPSTREAM of the LLM Judge — the numeric comparisons are settled before the model reasons, so
the Stage D-1 `0.339 > 0.376` class of error cannot recur. Idempotent (re-running overwrites the
same context keys). Run AFTER stage_d1_gen_fixtures.py and stage_d1_gen_criteria.py. Deterministic;
NO network/model/GPU. This does NOT run inference and does NOT change historical verdicts/expectations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "examples" / "stage_d1_replay"
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai.criterion_eval import attach_to_task, evaluate_criteria  # noqa: E402


def main():
    tasks = sorted((BASE / "tasks").glob("*.json"))
    n = 0
    for tf in tasks:
        cid = tf.stem
        spec_f = BASE / "criteria" / f"{cid}.json"
        if not spec_f.exists():
            print(f"skip {cid}: no criterion spec"); continue
        task = json.loads(tf.read_text())
        evidence = json.loads((BASE / "evidence" / f"{cid}.json").read_text())
        specs = json.loads(spec_f.read_text())
        results = evaluate_criteria(evidence, specs)
        assert len(results) == len(task["criteria"]), f"{cid}: results != criteria count"
        tf.write_text(json.dumps(attach_to_task(task, results), indent=2) + "\n")
        n += 1
    print(f"attached deterministic criterion results to {n} Stage D-1 tasks")


if __name__ == "__main__":
    main()
