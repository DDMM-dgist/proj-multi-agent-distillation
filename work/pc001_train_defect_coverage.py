#!/usr/bin/env python3
"""Root-cause diagnostic: how the defect structures entered the EXACT seed-123 TRAIN split of dataset.xyz.
Read-only; DFT forces from the data file; NO model inference / DFT / MD / training. Determines whether the
teacher's defect-domain degradation is coverage-, diversity-, model/training-, or DFT-label-limited."""
import json
import numpy as np
from collections import defaultdict
import torch
from ase.io import read

DATASET = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/03_allegro_train/dataset.xyz"
OUT = "/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/pc001_source_artifacts/train_defect_coverage.json"
N = 11424
DOMAIN = {
    "amorphous_SiO2": {"bulk_amo", "quench", "quench_int_AL", "liquid"},
    "SiO2x_dilute_vacancy": {"vacancy_int_AL", "vacancy", "SiOx_int_AL"},
    "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL", "quench_max_AL", "surfaces_max_AL"},
}
gen = torch.Generator().manual_seed(123)
sp = torch.utils.data.random_split(list(range(N)), [0.8, 0.1, 0.1], generator=gen)
train_idx = set(int(i) for i in sp[0].indices); test_idx = set(int(i) for i in sp[2].indices)

allA = read(DATASET, index=":")
frames = []
for i, a in enumerate(allA):
    syms = a.get_chemical_symbols(); nSi = syms.count("Si"); nO = syms.count("O")
    Fd = a.calc.results.get("forces") if (a.calc is not None and a.calc.results) else None
    if Fd is None: continue
    Fd = np.asarray(Fd)
    lx = (1.0 - (nO / nSi) / 2.0) if nSi > 0 else None   # O-deficiency vs SiO2 (0 = stoich)
    frames.append({"i": i, "cfg": a.info.get("config_type", "?"), "nat": len(a), "nSi": nSi, "nO": nO,
                   "local_x": lx, "dft_comp_RMS": float(np.sqrt((Fd.reshape(-1)**2).mean())),
                   "split": "train" if i in train_idx else ("test" if i in test_idx else "val")})

def dfam(c):
    for d, s in DOMAIN.items():
        if c in s: return d
    return None

def summ(rows):
    if not rows: return None
    lx = [r["local_x"] for r in rows if r["local_x"] is not None]
    rms = [r["dft_comp_RMS"] for r in rows]; nat = [r["nat"] for r in rows]
    return {"n": len(rows),
            "local_x": {"min": round(min(lx),3), "max": round(max(lx),3), "mean": round(float(np.mean(lx)),3), "std": round(float(np.std(lx)),3)} if lx else None,
            "dft_comp_RMS": {"min": round(min(rms),3), "max": round(max(rms),3), "mean": round(float(np.mean(rms)),3), "std": round(float(np.std(rms)),3)},
            "natoms": {"min": min(nat), "max": max(nat)},
            "distinct_natoms": len(set(nat)), "distinct_lx_rounded": len(set(round(x,2) for x in lx)) if lx else 0}

# per target domain: TRAIN coverage + diversity, and per raw config family
train = [f for f in frames if f["split"] == "train"]
test = [f for f in frames if f["split"] == "test"]
report = {"split_counts": {"train": len(train), "val": sum(1 for f in frames if f['split']=='val'), "test": len(test), "total_with_forces": len(frames)}}
report["domains"] = {}
for d in DOMAIN:
    tr = [f for f in train if dfam(f["cfg"]) == d]; te = [f for f in test if dfam(f["cfg"]) == d]
    # interpolation: fraction of TEST frames whose dft_comp_RMS within TRAIN [min,max] of same domain
    if tr and te:
        lo, hi = min(f["dft_comp_RMS"] for f in tr), max(f["dft_comp_RMS"] for f in tr)
        interp = sum(1 for f in te if lo <= f["dft_comp_RMS"] <= hi) / len(te)
    else:
        interp = None
    report["domains"][d] = {"TRAIN": summ(tr), "TEST": summ(te),
                            "train_fraction_of_all_train": round(len(tr)/len(train), 4),
                            "test_force_scale_within_train_range_frac": round(interp, 3) if interp is not None else None,
                            "per_config_family_TRAIN": {cfg: summ([f for f in tr if f["cfg"] == cfg]) for cfg in DOMAIN[d]}}
# global generalization gap (from committed exact-heldout: train 0.149 vs test 0.156 force)
report["global_generalization_gap"] = {"train_force_MAE": 0.149, "test_force_MAE": 0.156, "gap": 0.007,
    "note": "near-zero => the model fits train and held-out EQUALLY; defect error is NOT an overfitting/under-coverage gap"}

# root-cause determination (deterministic evidence)
dil = report["domains"]["SiO2x_dilute_vacancy"]; clu = report["domains"]["SiO2x_clustered_vacancy_voidsurface"]
coverage_counts_ok = dil["TRAIN"]["n"] >= 200 and clu["TRAIN"]["n"] >= 200
interp_ok = (dil["test_force_scale_within_train_range_frac"] or 0) >= 0.9 and (clu["test_force_scale_within_train_range_frac"] or 0) >= 0.9
report["root_cause"] = {
    "coverage_counts_sufficient": coverage_counts_ok,
    "test_defects_are_interpolation_not_extrapolation": interp_ok,
    "near_zero_generalization_gap": True,
    "determination": None}  # filled below in prose
print(json.dumps({"split_counts": report["split_counts"],
                  "dilute_TRAIN": {k: dil["TRAIN"][k] for k in ("n","local_x","dft_comp_RMS","distinct_natoms","distinct_lx_rounded")},
                  "clustered_TRAIN": {k: clu["TRAIN"][k] for k in ("n","local_x","dft_comp_RMS","distinct_natoms","distinct_lx_rounded")},
                  "amorphous_TRAIN_n": report["domains"]["amorphous_SiO2"]["TRAIN"]["n"],
                  "interp_dilute": dil["test_force_scale_within_train_range_frac"],
                  "interp_clustered": clu["test_force_scale_within_train_range_frac"],
                  "coverage_counts_ok": coverage_counts_ok, "interp_ok": interp_ok}, indent=2))
json.dump(report, open(OUT, "w"), indent=2)
print("wrote", OUT)
