#!/usr/bin/env python3
"""Stage D-2 C1 Attempt-2 PROVENANCE CLOSURE — deterministic, NO LLM, NO inference.

Generates the four missing append-only Attempt-2 records from the PRESERVED Attempt-2 exchange
provenance ONLY (the runner's writer step did not emit them). Copies the JudgeVote verdict /
criteria_checked / rationale / required_fix and the provider/model/usage/tool/provenance fields
EXACTLY from the recorded attempt; nothing is reconstructed from memory or rewritten. Append-only:
refuses if the attempt-2 outputs already exist; verifies the historical + scientific artifacts remain
byte-identical. The advisory verdict (REVISE) is genuine and is NOT rebound to the Axis-A PASS.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "harness"))
from stage_d2_judge_map import (assert_appendonly, assert_preserved, attempt_names,  # noqa: E402
                                build_attempt_records, snapshot_hashes)

RUN_DIR = ROOT / "tests" / "fixtures" / "stage_d2" / "d2c1-posthoc-msd-random_x006"
PDIR = RUN_DIR / "judge_exchange" / "exchange" / "provenance"
ATTEMPT = 2
ATTEMPT2_EXCHANGE_TOKEN = "02d531f6"   # the preserved Attempt-2 exchange provenance
ATTEMPT1_EXCHANGE_TOKEN = "2fe2fc26"   # the preserved Attempt-1 (failed) exchange provenance


def _find(token):
    hits = [p for p in glob.glob(str(PDIR / "*.json")) if token in p]
    if not hits:
        raise FileNotFoundError(f"exchange provenance for {token} not found under {PDIR}")
    return hits[0]


def main():
    a2_path = _find(ATTEMPT2_EXCHANGE_TOKEN)
    a1_path = _find(ATTEMPT1_EXCHANGE_TOKEN)
    jp = json.loads(Path(a2_path).read_text())

    # source-of-truth guards (no rewriting): must be a valid typed + canonical Attempt-2 result
    parsed = jp.get("parsed_result") or {}
    assert parsed.get("verdict") == "REVISE", "attempt-2 verdict is not REVISE — refuse to close"
    assert jp.get("validation_errors") in ([], None), "attempt-2 canonical validation not clean"
    assert not jp.get("failure_category"), "attempt-2 provenance carries a failure_category"

    assert_appendonly(RUN_DIR, ATTEMPT)
    before = snapshot_hashes(RUN_DIR)
    names = attempt_names(ATTEMPT)   # (interp, prov, semantic, run_manifest.after_judge)

    interp, prov, semantic = build_attempt_records(jp, attempt=ATTEMPT, axis_a_verdict="PASS")
    # exact copies from the recorded attempt (assert build_attempt_records didn't alter them)
    assert interp["advisory_verdict"] == parsed["verdict"] == "REVISE"
    assert interp["criteria_checked"] == parsed.get("criteria_checked")
    assert interp["rationale"] == parsed.get("rationale")
    assert interp["required_fix"] == parsed.get("required_fix")

    # augment the semantic record with the mandated explicit references + closure provenance
    rel = lambda p: str(Path(p).relative_to(RUN_DIR))  # noqa: E731
    semantic["axis_a_verdict"] = "PASS"
    semantic["semantic_judge_verdict"] = "REVISE"
    semantic["final_transition"] = "REVISE"
    semantic["closure"] = {"method": "deterministic (no LLM); copied from preserved exchange provenance",
                           "attempt1_exchange_provenance": rel(a1_path),
                           "attempt2_exchange_provenance": rel(a2_path),
                           "criterion_results": "criterion_results.json",
                           "original_run_provenance": "provenance.json"}

    (RUN_DIR / names[0]).write_text(json.dumps(interp, indent=2) + "\n")
    (RUN_DIR / names[1]).write_text(json.dumps(prov, indent=2) + "\n")
    (RUN_DIR / names[2]).write_text(json.dumps(semantic, indent=2) + "\n")
    consolidated = {"_note": "consolidated attempt-2 manifest; additive; does NOT overwrite the "
                    "original run_manifest.json, the deferred file, or an earlier attempt",
                    "attempt": ATTEMPT, "STAGE_D2_C1_AXIS_A": "PASS",
                    "STAGE_D2_C1_SEMANTIC_JUDGE": "REVISE", "STAGE_D2_C1_TRANSITION": "REVISE",
                    "STAGE_D2_C1": "AXIS_A_PASS__SEMANTIC_REVISE",
                    "attempt_artifacts": list(names)}
    (RUN_DIR / names[3]).write_text(json.dumps(consolidated, indent=2) + "\n")

    assert_preserved(RUN_DIR, before)   # historical + scientific artifacts byte-identical
    import hashlib
    print(json.dumps({"closed": True, "verdict": "REVISE", "final_transition": "REVISE",
                      "files": {n: hashlib.sha256((RUN_DIR / n).read_bytes()).hexdigest() for n in names}},
                     indent=2))


if __name__ == "__main__":
    main()
