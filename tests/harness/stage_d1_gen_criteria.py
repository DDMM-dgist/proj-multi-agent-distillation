#!/usr/bin/env python3
"""Generate GENERIC deterministic criterion specs for the Stage D-1 checkpoints (predicates, NOT
answers). Each spec is a list of field-referencing predicates aligned 1:1 (same order) with the
checkpoint's frozen ordered_criteria; the deterministic evaluator (runtimes/pydantic_ai/
criterion_eval.py) computes the booleans + severity. NO per-task answer is encoded (no `if task==...`);
only generic predicates over evidence fields + an ``invalidating`` flag on physical-validity criteria.
Emits tests/fixtures/stage_d1_replay/criteria/<cid>.json. Deterministic; NO network/model/GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests" / "fixtures" / "stage_d1_replay"
OUT = BASE / "criteria"

# Predicate lists per checkpoint, in the SAME ORDER as ordered_criteria. Field refs only.
DFT = [
    {"operator": "in_range", "lhs": {"field": "E_per_atom_eV"}, "rhs": {"low": -11, "high": -8}, "invalidating": True},
    {"operator": "le", "lhs": {"field": "max_force_eV_A"}, "rhs": {"const": 50}, "invalidating": True},
]
COMMITTEE = [
    {"operator": "le", "lhs": {"field": "u_max_mean_deploy"}, "rhs": {"field": "original_u_max_mean_deploy"}},
    {"operator": "le", "lhs": {"field": "error_c_eV_A"}, "rhs": {"const": 0.368}},
    {"operator": "le", "lhs": {"field": "F_RMSE_eV_A"}, "rhs": {"field": "original_F_RMSE_eV_A"}},
]
PREDICATES = {
    "d1-dft-cell_001": DFT, "d1-dft-clustered_cell_002": DFT, "d1-dft-cc001": DFT,
    "d1-committee-v3": COMMITTEE, "d1-committee-v5": COMMITTEE,
    "d1-data-provenance": [
        {"operator": "eq", "lhs": {"field": "split_manifest_committed"}, "rhs": {"const": True}},
        {"operator": "eq", "lhs": {"field": "leakage_resolved"}, "rhs": {"const": True}},
    ],
    "d1-physical-validation": [
        {"operator": "in_range", "lhs": {"field": "density_g_cm3"}, "rhs": {"low": 2.20, "high": 2.23}},
        {"all": [
            {"operator": "approx", "lhs": {"field": "si_o_peak_A"}, "rhs": {"value": 1.61, "tol": 0.05}},
            {"operator": "approx", "lhs": {"field": "cn_si"}, "rhs": {"value": 4, "tol": 0.1}},
            {"operator": "approx", "lhs": {"field": "cn_o"}, "rhs": {"value": 2, "tol": 0.1}}]},
        {"all": [
            {"operator": "lt", "lhs": {"field": "nve_drift_meV_atom_ns"}, "rhs": {"const": 0.05}},
            {"operator": "eq", "lhs": {"field": "msd_plateau"}, "rhs": {"const": True}}]},
    ],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold = json.loads((BASE / "golden_decisions.json").read_text())
    for cid, preds in PREDICATES.items():
        ordered = gold[cid]["ordered_criteria"]
        assert len(preds) == len(ordered), f"{cid}: {len(preds)} preds != {len(ordered)} criteria"
        spec = [dict(p, criterion=ordered[i]) for i, p in enumerate(preds)]
        (OUT / f"{cid}.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {len(PREDICATES)} generic criterion specs to {OUT}")


if __name__ == "__main__":
    main()
