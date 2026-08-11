#!/usr/bin/env python3
"""PC001 force-scale-normalized held-out fidelity — from ALREADY-GENERATED artifacts only.
Per-frame Teacher error metrics come from the committed exact-heldout per_frame_metrics.csv; DFT force
SCALE is re-read from dataset.xyz (existing DFT reference DATA — NOT a model forward). NO Allegro/DFT/
Student inference. Distinguishes A (absolute-scale rise) vs B (relative-fidelity degradation) vs C."""
import csv, json
import numpy as np
from collections import defaultdict
import torch
from ase.io import read

DATASET = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/03_allegro_train/dataset.xyz"
R = "/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime/runs/production_campaign_001/pc001-teacher-exact-heldout-validation"
N = 11424
DOMAIN = {
    "amorphous_SiO2": {"bulk_amo", "quench", "quench_int_AL", "liquid"},
    "SiO2x_dilute_vacancy": {"vacancy_int_AL", "vacancy", "SiOx_int_AL"},
    "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL", "quench_max_AL", "surfaces_max_AL"},
    "surfaces": {"surfaces", "surfaces_int_AL"},
    "ambient_crystalline_SiO2": {"bulk_cryst", "SiOx_crystal_amorphous_interfaces"},
}

# exact test indices (deterministic; no inference)
gen = torch.Generator().manual_seed(123)
test_idx = sorted(int(i) for i in torch.utils.data.random_split(list(range(N)), [0.8, 0.1, 0.1], generator=gen)[2].indices)

# per-frame Teacher error metrics from the committed CSV
err = {}
for r in csv.DictReader(open(f"{R}/fresh_test_per_frame_metrics.csv")):
    err[int(r["dataset_index"])] = {"cfg": r["config_type"], "nat": int(r["natoms"]),
        "err_comp_MAE": float(r["force_component_MAE_eV_A"]), "err_comp_RMSE": float(r["force_component_RMSE_eV_A"]),
        "vec_mean": float(r["force_vector_error_mean_eV_A"]), "vec_max": float(r["force_error_max_eV_A"])}

# DFT force scale re-read from dataset.xyz (DATA, not inference)
allA = read(DATASET, index=":")
per = []
for idx in test_idx:
    a = allA[idx]
    Fd = a.calc.results.get("forces") if (a.calc is not None and a.calc.results) else None
    if Fd is None or idx not in err:
        continue
    Fd = np.asarray(Fd)
    comp = Fd.reshape(-1)
    mag = np.linalg.norm(Fd, axis=1)
    e = err[idx]
    per.append({"idx": idx, "cfg": e["cfg"], "nat": e["nat"],
                "dft_comp_RMS": float(np.sqrt((comp**2).mean())), "dft_comp_meanabs": float(np.abs(comp).mean()),
                "dft_mag_mean": float(mag.mean()), "dft_mag_median": float(np.median(mag)), "dft_mag_p95": float(np.quantile(mag, .95)),
                "err_comp_MAE": e["err_comp_MAE"], "err_comp_RMSE": e["err_comp_RMSE"], "vec_mean": e["vec_mean"]})

def dfam(c):
    for d, s in DOMAIN.items():
        if c in s: return d
    return None

def agg(rows):
    if not rows: return None
    A = lambda k: float(np.mean([r[k] for r in rows]))
    # atom-weighted DFT scale (representative of the domain's force magnitudes)
    tot_at = sum(r["nat"] for r in rows)
    dft_rms_aw = float(np.sqrt(sum(r["dft_comp_RMS"]**2 * r["nat"] * 3 for r in rows) / (tot_at * 3)))
    dft_meanabs_aw = float(sum(r["dft_comp_meanabs"] * r["nat"] * 3 for r in rows) / (tot_at * 3))
    err_rmse_aw = float(np.sqrt(sum(r["err_comp_RMSE"]**2 * r["nat"] * 3 for r in rows) / (tot_at * 3)))
    err_mae_aw = float(sum(r["err_comp_MAE"] * r["nat"] * 3 for r in rows) / (tot_at * 3))
    return {"n": len(rows),
            "DFT_comp_RMS": round(dft_rms_aw, 4), "DFT_comp_meanabs": round(dft_meanabs_aw, 4),
            "DFT_permag_mean": round(A("dft_mag_mean"), 4), "DFT_permag_median": round(A("dft_mag_median"), 4), "DFT_permag_p95": round(A("dft_mag_p95"), 4),
            "err_comp_MAE": round(err_mae_aw, 4), "err_comp_RMSE": round(err_rmse_aw, 4),
            "normalized_comp_RMSE": round(err_rmse_aw / dft_rms_aw, 4),
            "normalized_comp_MAE": round(err_mae_aw / dft_meanabs_aw, 4),
            "normalized_vector_error": round(A("vec_mean") / A("dft_mag_mean"), 4)}

domain = {d: agg([r for r in per if dfam(r["cfg"]) == d]) for d in DOMAIN}
domain = {d: v for d, v in domain.items() if v}
# per-config within amorphous/dilute/clustered
per_config = {}
for d in ("amorphous_SiO2", "SiO2x_dilute_vacancy", "SiO2x_clustered_vacancy_voidsurface"):
    for cfg in DOMAIN[d]:
        sub = [r for r in per if r["cfg"] == cfg]
        if sub: per_config[cfg] = agg(sub)

amo, dil, clu = domain["amorphous_SiO2"], domain["SiO2x_dilute_vacancy"], domain["SiO2x_clustered_vacancy_voidsurface"]
# A/B/C: compare absolute vs normalized rise amorphous->clustered
abs_rise = clu["err_comp_RMSE"] / amo["err_comp_RMSE"]
dft_rise = clu["DFT_comp_RMS"] / amo["DFT_comp_RMS"]
norm_rise = clu["normalized_comp_RMSE"] / amo["normalized_comp_RMSE"]
classification = ("A_ABSOLUTE_ERROR_RISE_ONLY" if norm_rise < 1.15
                  else "B_TRUE_RELATIVE_FIDELITY_DEGRADATION" if norm_rise >= 1.4
                  else "C_MIXED")
out = {
    "note": "DFT force scale re-read from dataset.xyz (DATA); Teacher error metrics from committed exact-heldout per_frame_metrics.csv; NO model forward.",
    "cosine_similarity_and_component_R2": "NOT_PERSISTED — teacher per-atom force arrays were not saved in the exact-heldout run; computing them would require a re-forward (forbidden). Normalized error + DFT scale below fully address the A/B/C question.",
    "domain": domain, "per_config_family": per_config,
    "amorphous_dilute_clustered": {
        "err_comp_RMSE": [amo["err_comp_RMSE"], dil["err_comp_RMSE"], clu["err_comp_RMSE"]],
        "DFT_comp_RMS": [amo["DFT_comp_RMS"], dil["DFT_comp_RMS"], clu["DFT_comp_RMS"]],
        "normalized_comp_RMSE": [amo["normalized_comp_RMSE"], dil["normalized_comp_RMSE"], clu["normalized_comp_RMSE"]],
        "normalized_comp_MAE": [amo["normalized_comp_MAE"], dil["normalized_comp_MAE"], clu["normalized_comp_MAE"]],
        "normalized_vector_error": [amo["normalized_vector_error"], dil["normalized_vector_error"], clu["normalized_vector_error"]],
        "abs_err_rise_clu_over_amo": round(abs_rise, 3), "DFT_scale_rise_clu_over_amo": round(dft_rise, 3),
        "normalized_err_rise_clu_over_amo": round(norm_rise, 3)},
    "classification": classification,
}
json.dump(out, open(f"{R}/force_scale_normalized_domain_metrics.json", "w"), indent=2)
print(json.dumps({"domain_norm_RMSE": {d: v["normalized_comp_RMSE"] for d, v in domain.items()},
                  "domain_DFT_RMS": {d: v["DFT_comp_RMS"] for d, v in domain.items()},
                  "domain_abs_err_RMSE": {d: v["err_comp_RMSE"] for d, v in domain.items()},
                  "amo_dil_clu_normRMSE": out["amorphous_dilute_clustered"]["normalized_comp_RMSE"],
                  "abs_rise": abs_rise, "dft_scale_rise": dft_rise, "norm_rise": norm_rise,
                  "classification": classification}, indent=2))
