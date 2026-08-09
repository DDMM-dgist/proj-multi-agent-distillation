#!/usr/bin/env python3
"""Stage D-2 C1 PROVENANCE CLOSURE (no execution; no MSD rerun; scientific artifacts untouched).

Two methodological corrections, applied only to metadata files:
  (1) Workflow transition is NOT ADVANCE yet: Axis-A PASS, semantic Judge PENDING ->
      STAGE_D2_C1_TRANSITION = READY_FOR_ADVISORY_JUDGE.
  (2) Execution-wrapper provenance caveat: work/stage_d2_execute.py was created AFTER the approved
      preparation HEAD b5762a1d and is not contained in it. Snapshot the EXACT wrapper source into the
      run dir, record its sha256, and mark wrapper_committed_at_execution=false.

Only provenance.json + run_manifest.json are updated; msd.csv + msd_summary.json + criterion_results.json
+ approval.json are verified byte-identical and NEVER modified. Deterministic; NO network/model/GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "stage_d2" / "d2c1-posthoc-msd-random_x006"
WRAPPER = ROOT / "work" / "stage_d2_execute.py"
PACKAGE_HEAD = "b5762a1d57fad9dd16fe557702bb311117c38786"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    prov = json.loads((RUN_DIR / "provenance.json").read_text())
    # (0) prove the scientific results are UNALTERED vs what the run recorded
    for name in ("msd.csv", "msd_summary.json", "criterion_results.json", "approval.json"):
        assert _sha(RUN_DIR / name) == prov["artifacts_sha256"][name], f"{name} changed since execution!"

    # (2) snapshot the exact execution wrapper as a provenance-only artifact
    wrapper_sha = _sha(WRAPPER)
    (RUN_DIR / "execution_wrapper_snapshot.py").write_text(WRAPPER.read_text())
    assert _sha(RUN_DIR / "execution_wrapper_snapshot.py") == wrapper_sha, "snapshot mismatch"

    execution_wrapper = {
        "snapshot_file": "execution_wrapper_snapshot.py", "sha256": wrapper_sha,
        "orchestrated_by": "work/stage_d2_execute.py",
        "created_after_approved_head": PACKAGE_HEAD,
        "wrapper_committed_at_execution": False,
        "note": ("the orchestration wrapper was NOT contained in the approved frozen HEAD "
                 f"{PACKAGE_HEAD}; the prepared trusted executor (stage_d2_executor.py) itself was "
                 "unchanged, source sha before/after matched, Stage D-1 unchanged, output isolated, "
                 "Axis-A deterministic criteria passed."),
    }
    # (1) transition correction (metadata only) + caveat records
    prov["execution_wrapper"] = execution_wrapper
    prov["STAGE_D2_C1_AXIS_A"] = "PASS"
    prov["STAGE_D2_C1_SEMANTIC_JUDGE"] = "PENDING"
    prov["STAGE_D2_C1_TRANSITION"] = "READY_FOR_ADVISORY_JUDGE"
    prov["STAGE_D2_C1_EXECUTION_ATTEMPT_1"] = "AXIS_A_PASS_WITH_EXECUTION_WRAPPER_PROVENANCE_CAVEAT"
    prov["transition"] = "READY_FOR_ADVISORY_JUDGE"   # supersede the earlier ADVANCE
    prov["transition_note"] = ("ADVANCE/REVISE is finalized only after the separately-approved advisory "
                               "semantic Judge. The human 'looks bounded/solid-like' observation is "
                               "descriptive only and does not substitute for the formal Judge result.")
    prov["advisory_judge"] = "PENDING"
    (RUN_DIR / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    manifest = json.loads((RUN_DIR / "run_manifest.json").read_text())
    manifest.update({"transition": "READY_FOR_ADVISORY_JUDGE", "STAGE_D2_C1_AXIS_A": "PASS",
                     "STAGE_D2_C1_SEMANTIC_JUDGE": "PENDING", "advisory_judge": "PENDING",
                     "accepted": False, "accepted_note": "Axis-A artifact validity PASS; final "
                     "ADVANCE pending advisory Judge", "execution_wrapper": execution_wrapper})
    (RUN_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({"wrapper_sha256": wrapper_sha, "wrapper_committed_at_execution": False,
                      "transition": "READY_FOR_ADVISORY_JUDGE",
                      "scientific_artifacts_unaltered": True,
                      "snapshot": "execution_wrapper_snapshot.py"}, indent=2))


if __name__ == "__main__":
    main()
