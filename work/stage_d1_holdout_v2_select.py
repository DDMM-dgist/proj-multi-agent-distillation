#!/usr/bin/env python3
"""STAGE D-1 HOLDOUT V2 — deterministic selection (per work/stage_d1_holdout_v2_plan.md).

Selection depends ONLY on the predeclared key sha256("gate|target") plus the declared gate-family
stratification and per-family quotas. It does NOT read historical verdicts, difficulty, deterministic
severity, or model-pass likelihood — those are not inputs here. Emits SELECTION_MANIFEST.json with only
{target, gate_family, selection_hash, selection_rank, source}. Freeze this BEFORE building fixtures.
Deterministic; NO network/model/GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "examples" / "stage_d1_holdout_v2"

# The frozen candidate pool (gate, target, family, source) = remaining real auditable decisions NOT in
# the 7 development or 8 consumed-holdout-v1 cases. Family + source only; NO verdict is recorded here.
POOL = [
    ("dft-label-judge-gate", "cell_009", "dft-physical", "scan_labeled_structures/manifest.csv"),
    ("judge-gate-clustered", "clustered_cell_000", "dft-physical", "scan_labeled_structures/manifest.csv"),
    ("judge-gate-clustered", "clustered_cell_003", "dft-physical", "scan_labeled_structures/manifest.csv"),
    ("judge-gate-clustered", "clustered_cell_006", "dft-physical", "scan_labeled_structures/manifest.csv"),
    ("judge-gate-clustered", "clustered_cell_007", "dft-physical", "scan_labeled_structures/manifest.csv"),
    ("judge-gate-clustered", "clustered_cell_008", "dft-physical", "scan_labeled_structures/manifest.csv"),
    ("production-sizing-gate", "production-cell-12288-cubic", "production-protocol", "coordination_log.csv"),
    ("cristobalite-build-gate", "cristobalite-12288-seed", "production-protocol", "coordination_log.csv"),
    ("er-finetune-gate", "teacher-ER-finetune-AB", "committee-model-selection", "coordination_log.csv"),
    ("ph-pipeline-gate", "persistent-homology-pipeline", "science-analysis", "coordination_log.csv"),
    ("production-science-gate", "paper2-production-findings", "science-analysis", "coordination_log.csv"),
    ("meltquench-protocol-gate", "production_12288-meltquench-protocol", "production-protocol", "coordination_log.csv"),
]
# Declared quotas per family (balanced coverage; the pool has no FAIL case in any family — recorded as
# HOLDOUT_V2_UNSEEN_FAIL_CASE_COVERAGE=0, not manufactured). Chosen before hashing; independent of verdict.
QUOTA = {"dft-physical": 2, "production-protocol": 2, "committee-model-selection": 1, "science-analysis": 2}


def _key(gate, target):
    return hashlib.sha256(f"{gate}|{target}".encode()).hexdigest()


def select():
    from collections import defaultdict
    byfam = defaultdict(list)
    for gate, target, fam, src in POOL:
        byfam[fam].append((_key(gate, target), gate, target, src))
    manifest = []
    for fam in sorted(byfam):
        for rank, (h, gate, target, src) in enumerate(sorted(byfam[fam])):   # ascending hash
            if rank < QUOTA.get(fam, 0):
                manifest.append({"target": target, "gate": gate, "gate_family": fam,
                                 "selection_hash": h, "selection_rank": rank, "source": src})
    return sorted(manifest, key=lambda m: (m["gate_family"], m["selection_rank"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = select()
    (OUT / "SELECTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"HOLDOUT_V2_SELECTION_MANIFEST ({len(manifest)} cases; selection = sha256 rank only):")
    for m in manifest:
        print(f"  {m['gate_family']:26s} rank={m['selection_rank']} {m['target']:38s} "
              f"gate={m['gate']:26s} sha={m['selection_hash'][:16]}")


if __name__ == "__main__":
    main()
