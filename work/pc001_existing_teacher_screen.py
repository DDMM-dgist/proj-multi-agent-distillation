#!/usr/bin/env python3
"""Phase 1 — cheap existing-Teacher candidate screen on the 373-frame target-domain held-out subset.
Base b56e20ff metrics REUSED from the committed exact-heldout per-frame CSV (no rerun). Candidate
teacher_finetuned_v2 (b3be4d2a) evaluated fresh on the SAME 373 held-out frames via NequIPCalculator
(valid path). DFT scale re-read from dataset.xyz (DATA). NO Student/DFT/MD/training/Judge. VASP untouched."""
import sys, time, json, csv
import numpy as np
from pathlib import Path
import torch

import os
BASE_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
# candidate configurable via env; default = finetuned_v2
FT = os.environ.get("CAND_PATH", "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/gpu_return_v2/return_to_cpu/teacher/teacher_finetuned_v2.nequip.pth")
CAND_LABEL = os.environ.get("CAND_LABEL", "finetuned_v2")
CAND_SHA = os.environ.get("CAND_SHA", "b3be4d2aa33ec5cd")
RUN_NAME = os.environ.get("RUN_NAME", "pc001-existing-teacher-screen")
DATASET = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/03_allegro_train/dataset.xyz"
ROOT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime")
RUN = ROOT / "runs" / "production_campaign_001" / RUN_NAME
TEST_CSV = ROOT / "runs/production_campaign_001/pc001-teacher-exact-heldout-validation/fresh_test_per_frame_metrics.csv"
N = 11424
DOMAIN = {"amorphous_SiO2": {"bulk_amo","quench","quench_int_AL","liquid"},
          "SiO2x_dilute_vacancy": {"vacancy_int_AL","vacancy","SiOx_int_AL"},
          "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL","quench_max_AL","surfaces_max_AL"}}
CENTRAL = {c for s in DOMAIN.values() for c in s}
def dfam(c):
    for d,s in DOMAIN.items():
        if c in s: return d
    return None
def agg(rows, key_err="err_comp_MAE"):
    if not rows: return None
    at3=sum(r["nat"]*3 for r in rows)
    return {"N":len(rows),
            "DFT_comp_RMS":round(float(np.sqrt(sum(r["dft_rms"]**2*r["nat"]*3 for r in rows)/at3)),4),
            "err_comp_MAE":round(float(sum(r["err_mae"]*r["nat"]*3 for r in rows)/at3),4),
            "err_comp_RMSE":round(float(np.sqrt(sum(r["err_rmse"]**2*r["nat"]*3 for r in rows)/at3)),4),
            "normalized_RMSE":round(float(np.sqrt(sum(r["err_rmse"]**2*r["nat"]*3 for r in rows)/at3)/np.sqrt(sum(r["dft_rms"]**2*r["nat"]*3 for r in rows)/at3)),4),
            "energy_MAE_meV":round(float(np.mean([abs(r["dEpa"]) for r in rows if r.get("dEpa") is not None])),2) if any(r.get("dEpa") is not None for r in rows) else None}

def main():
    RUN.mkdir(parents=True, exist_ok=False)
    from ase.io import read
    from nequip.ase import NequIPCalculator
    # target-domain held-out frames + base metrics from committed CSV
    base = {}
    for r in csv.DictReader(open(TEST_CSV)):
        if r["config_type"] in CENTRAL:
            base[int(r["dataset_index"])] = {"cfg": r["config_type"], "nat": int(r["natoms"]),
                "err_mae": float(r["force_component_MAE_eV_A"]), "err_rmse": float(r["force_component_RMSE_eV_A"]),
                "dEpa": float(r["dE_meV_atom"])}
    idxs = sorted(base)
    # load candidate
    try:
        calc = NequIPCalculator.from_compiled_model(FT, device="cpu", chemical_species_to_atom_type_map={"O":"O","Si":"Si"})
        load_ok = True; load_err = None
    except Exception as e:
        load_ok = False; load_err = f"{type(e).__name__}: {str(e)[:200]}"
    allA = read(DATASET, index=":")
    base_rows, cand_rows = [], []
    t0=time.time()
    for i in idxs:
        a = allA[i]; nat = len(a)
        Fd = a.calc.results.get("forces") if (a.calc is not None and a.calc.results) else None
        Ed = a.calc.results.get("energy") if (a.calc is not None and a.calc.results) else None
        if Fd is None: continue
        Fd = np.asarray(Fd); dft_rms = float(np.sqrt((Fd.reshape(-1)**2).mean()))
        b = base[i]
        base_rows.append({"cfg": b["cfg"], "nat": nat, "dft_rms": dft_rms, "err_mae": b["err_mae"], "err_rmse": b["err_rmse"], "dEpa": b["dEpa"]})
        if load_ok:
            m = a.copy(); m.calc = calc
            Et = float(m.get_potential_energy()); Ft = m.get_forces()
            comp = np.abs(Ft - Fd)
            cand_rows.append({"cfg": b["cfg"], "nat": nat, "dft_rms": dft_rms,
                              "err_mae": float(comp.mean()), "err_rmse": float(np.sqrt((comp**2).mean())),
                              "dEpa": (Et - Ed)/nat*1000.0 if Ed is not None else None})
    elapsed = time.time()-t0

    result = {"screen_set_size": len(base_rows), "candidate_label": CAND_LABEL, "candidate_path": FT,
              "candidates": {"base_b56e20ff": BASE_SHA, CAND_LABEL: CAND_SHA},
              "candidate_load_ok": load_ok, "candidate_load_error": load_err,
              "walltime_min": round(elapsed/60,1)}
    domains = {}
    for d in DOMAIN:
        b = agg([r for r in base_rows if dfam(r["cfg"])==d])
        c = agg([r for r in cand_rows if dfam(r["cfg"])==d]) if load_ok else None
        rec = {"N": b["N"], "base": b, "candidate": c}
        if c:
            rec["delta_err_MAE"] = round(c["err_comp_MAE"]-b["err_comp_MAE"],4)
            rec["delta_normalized_RMSE"] = round(c["normalized_RMSE"]-b["normalized_RMSE"],4)
            rec["improved"] = c["err_comp_MAE"] < b["err_comp_MAE"]
        domains[d] = rec
    result["domains"] = domains
    result["caveat"] = ("If this candidate IMPROVES oxygen-deficient domains, verify NONE of the 373 held-out "
                        "frames overlap the candidate's fine-tune training set before advancing (train-on-test). "
                        "A null/worse result is unaffected by potential leakage (leakage would only help).")
    # advancement (no invented threshold): clear improvement in dilute+clustered WITHOUT amorphous degradation
    if load_ok:
        dil = domains["SiO2x_dilute_vacancy"]; clu = domains["SiO2x_clustered_vacancy_voidsurface"]; amo = domains["amorphous_SiO2"]
        dil_impr = dil["delta_err_MAE"] < -0.005   # meaningfully lower dilute error
        clu_impr = clu["delta_err_MAE"] < -0.005
        amo_ok = amo["delta_err_MAE"] <= 0.01      # no material amorphous degradation
        advance = (dil_impr or clu_impr) and amo_ok and (dil["delta_err_MAE"]<=0.005 and clu["delta_err_MAE"]<=0.005)
        result["advancement"] = {
            "amorphous_delta_MAE": amo["delta_err_MAE"], "dilute_delta_MAE": dil["delta_err_MAE"], "clustered_delta_MAE": clu["delta_err_MAE"],
            "clear_improvement_in_oxygen_deficient": bool(dil_impr or clu_impr),
            "no_amorphous_degradation": bool(amo_ok),
            "ADVANCE_to_full_1142_validation": bool(advance),
            "decision": "ADVANCE" if advance else "DO_NOT_ADVANCE (changes tiny/mixed/regressive — not a clear oxygen-deficient improvement without amorphous degradation)"}
    else:
        result["advancement"] = {"decision": "CANDIDATE_UNSCREENABLE", "reason": load_err}
    (RUN/"existing_teacher_candidate_screen.json").write_text(json.dumps(result, indent=2, default=float)+"\n")
    print(json.dumps({"candidate": CAND_LABEL}
                     | {k:result[k] for k in ("screen_set_size","candidate_load_ok","walltime_min")}
                     | {"domains":{d:{"base_MAE":domains[d]["base"]["err_comp_MAE"],
                                      "cand_MAE":(domains[d]["candidate"]["err_comp_MAE"] if domains[d]["candidate"] else None),
                                      "delta":domains[d].get("delta_err_MAE")} for d in DOMAIN},
                        "decision":result["advancement"]["decision"]}, indent=2, default=float))

if __name__ == "__main__":
    sys.exit(main())
