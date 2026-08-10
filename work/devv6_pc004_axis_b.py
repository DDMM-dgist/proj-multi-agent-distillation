#!/usr/bin/env python3
"""PC004 Axis B (§12): Teacher v6 and Student committee vs the protected 11 SCAN DFT cells.
Channels (kept distinct from Axis A student_vs_teacher): teacher_vs_dft_11, student_vs_dft_11.
The 11 SCAN cells are READ_ONLY / VALIDATION_ONLY / NEVER_TRAINING. No new DFT.
Usage: devv6_pc004_axis_b.py <teacher.v6.yaml> <student_committee.manifest.json|NONE> <out.json>
Runs teacher channel via NequIPCalculator (this env=allegro); student channel via each committee
potential's predict adapter (spawns its own env). Deterministic; force component MAE (eV/A)."""
import sys, json, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase.io import read
from adapters import load_config
from adapters.teacher import load_teacher

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
SCAN = f"{RES}/scan_labeled_structures/sio2x_AL_labels_11cells.xyz"

def dft_forces(a):
    for k in ("dft_forces", "forces"):
        if a.calc and a.calc.results.get("forces") is not None and k == "forces":
            return np.asarray(a.calc.results["forces"])
        if k in a.arrays:
            return np.asarray(a.arrays[k])
    return None

def comp_mae(F, Fd):
    return float(np.abs(np.asarray(F) - np.asarray(Fd)).mean())

def main(teacher_cfg_path, committee_manifest, out):
    cells = read(SCAN, index=":")
    dfts = [dft_forces(a) for a in cells]
    res = {"n_dft_cells": len(cells), "cells_read_only_validation_only": True, "channels": {}}
    # teacher_vs_dft_11
    try:
        calc = load_teacher(load_config(teacher_cfg_path))
        errs = []
        for a, Fd in zip(cells, dfts):
            if Fd is None: continue
            m = a.copy(); m.calc = calc
            m.get_potential_energy(); errs.append(comp_mae(m.get_forces(), Fd))
        res["channels"]["teacher_vs_dft_11"] = {"status": "EXECUTED", "n": len(errs),
            "force_comp_MAE_eV_A": round(float(np.mean(errs)), 4) if errs else None,
            "per_cell": [round(e, 4) for e in errs]}
    except Exception as e:
        res["channels"]["teacher_vs_dft_11"] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    # student_vs_dft_11 (committee)
    if committee_manifest and committee_manifest != "NONE" and Path(committee_manifest).exists():
        cm = json.loads(Path(committee_manifest).read_text())
        res["channels"]["student_vs_dft_11"] = {"status": "PENDING_WIRE",
            "note": "committee present; per-member LAMMPS predict vs DFT-11 to be run via predict adapter",
            "n_members": len(cm.get("models", []))}
    else:
        res["channels"]["student_vs_dft_11"] = {"status": "WAIT_STUDENT",
            "note": "no committee manifest yet; runs after training completes"}
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "NONE", sys.argv[3] if len(sys.argv) > 3 else "/dev/stdout")
