#!/usr/bin/env python3
"""TRACK A / PC-A1 — GPU-side completion check + quick fidelity sanity.
Runs on the GPU server AFTER compile_model.sh. This is NOT the authoritative
acceptance test (that is the CPU-side 373-frame target-domain held-out screen at
the join point) — it only confirms the fine-tuned teacher loads and produces a
sane force MAE on the shipped internal validation set.

Usage:  python scripts/verify_and_eval.py <compiled.nequip.pth>
"""
import sys, json
import numpy as np
from ase.io import read
from nequip.ase import NequIPCalculator

def main(model):
    calc = NequIPCalculator.from_compiled_model(model, device="cpu",
            chemical_species_to_atom_type_map={"O": "O", "Si": "Si"})
    val = read("dataset/fine_tune_val.xyz", index=":")
    # domain tag from config_type via the same central mapping
    DOMAIN = {"amorphous_SiO2": {"bulk_amo","quench","quench_int_AL","liquid"},
              "SiO2x_dilute_vacancy": {"vacancy_int_AL","vacancy","SiOx_int_AL"},
              "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL","quench_max_AL","surfaces_max_AL"}}
    def dfam(c):
        for d,s in DOMAIN.items():
            if c in s: return d
        return "replay_broad"
    acc = {}
    for a in val:
        Fd = a.calc.results.get("forces") if (a.calc and a.calc.results) else None
        if Fd is None: continue
        Fd = np.asarray(Fd)
        m = a.copy(); m.calc = calc
        m.get_potential_energy(); Ft = m.get_forces()
        mae = float(np.abs(Ft - Fd).mean())
        d = dfam(a.info.get("config_type","?"))
        acc.setdefault(d, []).append((mae, len(a)))
        acc.setdefault("ALL", []).append((mae, len(a)))
    out = {}
    for d, rows in acc.items():
        w = sum(n for _, n in rows)
        out[d] = {"N": len(rows), "force_component_MAE_eV_A": round(sum(m*n for m,n in rows)/w, 4)}
    print(json.dumps({"model": model, "internal_val_force_MAE_by_domain": out,
                      "note": "sanity only; authoritative acceptance = CPU-side 373-frame held-out screen"}, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1])
