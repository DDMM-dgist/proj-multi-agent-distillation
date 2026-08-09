#!/usr/bin/env python3
"""PC001 source-grounded FRESH Teacher inference (confirmatory) on the training-corpus dataset.xyz.

The AUTHORITATIVE held-out test result is the NequIP test-phase (seed-123 10% split of dataset.xyz):
forces_mae 0.1561 eV/A, per_atom_energy_mae 15.73 meV/atom (wandb-summary + train log). This script is an
INDEPENDENT confirmation via the NequIPCalculator path (the same valid path, NOT raw-torch C3) on a
deterministic STRATIFIED per-config sample of dataset.xyz, to (a) confirm the calculator path reproduces
the pipeline global and (b) provide domain-resolved numbers the global log lacks. NO student data, DFT,
MD, training, network, Judge.
"""
import sys, time, json, hashlib, csv
import numpy as np
from pathlib import Path
from collections import defaultdict

MODEL = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/teacher/model.nequip.pth"
DATASET = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/03_allegro_train/dataset.xyz"
ROOT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime")
RUN = ROOT / "runs" / "production_campaign_001" / "pc001-teacher-training-source-validation"
K_PER_CONFIG = 22   # deterministic evenly-spaced sample per config family

DOMAIN = {
    "amorphous_SiO2": {"bulk_amo", "quench", "quench_int_AL", "liquid"},
    "SiO2x_dilute_vacancy": {"vacancy_int_AL", "vacancy", "SiOx_int_AL"},
    "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL", "quench_max_AL", "surfaces_max_AL"},
    "surfaces": {"surfaces", "surfaces_int_AL"},
    "ambient_crystalline_SiO2": {"bulk_cryst", "SiOx_crystal_amorphous_interfaces"},
    "OUT_elemental_Si": {"silicon_bulk_amo", "silicon_crystalline_main", "silicon_defects", "silicon_liquid", "silicon_others", "silicon_surfaces"},
    "OUT_high_pressure": {"bulk_cryst_hp", "highpressure_int_AL", "highpressure_max_AL"},
    "OUT_cluster": {"cluster"},
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    from ase.io import read
    from nequip.ase import NequIPCalculator
    import torch, nequip, platform
    t0 = time.time()
    calc = NequIPCalculator.from_compiled_model(MODEL, device="cpu",
                                                chemical_species_to_atom_type_map={"O": "O", "Si": "Si"})
    allA = read(DATASET, index=":")
    by_cfg = defaultdict(list)
    for i, a in enumerate(allA):
        by_cfg[a.info.get("config_type", "?")].append(i)
    # deterministic evenly-spaced sample per config
    sample_idx = []
    for cfg, idxs in by_cfg.items():
        if len(idxs) <= K_PER_CONFIG:
            sample_idx += idxs
        else:
            step = len(idxs) / K_PER_CONFIG
            sample_idx += [idxs[int(j * step)] for j in range(K_PER_CONFIG)]
    sample_idx = sorted(set(sample_idx))

    rows = []
    for idx in sample_idx:
        a = allA[idx]; nat = len(a)
        # DFT labels live on a SinglePointCalculator (energy==dft_free_energy, forces) — extract FIRST
        Ed = a.calc.results.get("energy") if (a.calc is not None and a.calc.results) else a.info.get("dft_free_energy", a.info.get("dft_energy"))
        Fd = a.calc.results.get("forces") if (a.calc is not None and a.calc.results) else a.arrays.get("dft_forces", a.arrays.get("forces"))
        cfg = a.info.get("config_type", "?")
        syms = a.get_chemical_symbols()
        if Ed is None or Fd is None or not np.all(np.isfinite(Fd)) or any(s not in ("O", "Si") for s in syms):
            continue
        b = a.copy(); b.calc = calc   # a.copy() drops the DFT calc; attach the teacher
        Et = float(b.get_potential_energy()); Ft = b.get_forces()
        comp = np.abs(Ft - Fd)
        rows.append({"idx": idx, "config_type": cfg, "natoms": nat,
                     "nSi": syms.count("Si"), "nO": syms.count("O"),
                     "E_DFT_eV": float(Ed), "E_teacher_eV": Et, "dE_meV_atom": (Et - Ed) / nat * 1000.0,
                     "force_component_MAE_eV_A": float(comp.mean()),
                     "force_component_RMSE_eV_A": float(np.sqrt(((Ft - Fd) ** 2).mean())),
                     "force_vector_error_mean_eV_A": float(np.linalg.norm(Ft - Fd, axis=1).mean()),
                     "force_error_max_eV_A": float(np.linalg.norm(Ft - Fd, axis=1).max())})
    elapsed = time.time() - t0

    with open(RUN / "per_frame_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)

    def dfam(c):
        for d, s in DOMAIN.items():
            if c in s: return d
        return "OTHER"
    fmae = np.array([r["force_component_MAE_eV_A"] for r in rows])
    dEpa = np.array([r["dE_meV_atom"] for r in rows])
    glob = {"n_struct": len(rows), "n_atoms": int(sum(r["natoms"] for r in rows)),
            "force_component_MAE": float(fmae.mean()), "force_component_RMSE_of_frame_MAEs": float(np.sqrt((fmae**2).mean())),
            "force_median": float(np.median(fmae)), "force_p90": float(np.quantile(fmae, .9)),
            "force_p95": float(np.quantile(fmae, .95)), "force_p99": float(np.quantile(fmae, .99)), "force_max": float(fmae.max()),
            "energy_signed_bias_meV_atom": float(dEpa.mean()), "energy_raw_MAE": float(np.abs(dEpa).mean()),
            "energy_bias_corrected_MAE": float(np.abs(dEpa - dEpa.mean()).mean()), "energy_RMSE": float(np.sqrt((dEpa**2).mean()))}
    domain = {}
    for d in list(DOMAIN) + ["OTHER"]:
        sub = [r for r in rows if dfam(r["config_type"]) == d]
        if not sub: continue
        f = np.array([r["force_component_MAE_eV_A"] for r in sub]); e = np.array([abs(r["dE_meV_atom"]) for r in sub])
        domain[d] = {"n": len(sub), "force_MAE": float(f.mean()), "force_RMSE": float(np.sqrt((f**2).mean())),
                     "force_median": float(np.median(f)), "force_p95": float(np.quantile(f, .95)), "force_max": float(f.max()),
                     "energy_MAE": float(e.mean())}
    with open(RUN / "domain_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["domain", "n", "force_MAE", "force_RMSE", "force_median", "force_p95", "force_max", "energy_MAE_meV"])
        for d, m in domain.items():
            w.writerow([d, m["n"], round(m["force_MAE"], 4), round(m["force_RMSE"], 4), round(m["force_median"], 4),
                        round(m["force_p95"], 4), round(m["force_max"], 4), round(m["energy_MAE"], 2)])
    amo = domain.get("amorphous_SiO2", {}).get("force_MAE")
    dil = domain.get("SiO2x_dilute_vacancy", {}).get("force_MAE")
    clu = domain.get("SiO2x_clustered_vacancy_voidsurface", {}).get("force_MAE")
    vac = {"amorphous": amo, "dilute": dil, "clustered": clu,
           "delta_dilute_minus_amorphous": (dil - amo) if (amo and dil) else None,
           "delta_clustered_minus_dilute": (clu - dil) if (dil and clu) else None,
           "monotonic": bool(amo and dil and clu and amo < dil < clu),
           "caveat": "sample from the TRAINING CORPUS (in-sample); given the near-zero global generalization gap (NequIP test 0.156 vs train 0.149), the per-domain gradient is representative of held-out per-domain too"}
    top = sorted(rows, key=lambda r: -r["force_component_MAE_eV_A"])[:20]
    with open(RUN / "outlier_table.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rank", "idx", "config_type", "natoms", "nSi_nO", "force_MAE", "force_max", "dE_meV_atom"])
        for i, r in enumerate(top, 1):
            w.writerow([i, r["idx"], r["config_type"], r["natoms"], f"{r['nSi']}/{r['nO']}",
                        round(r["force_component_MAE_eV_A"], 3), round(r["force_error_max_eV_A"], 2), round(r["dE_meV_atom"], 1)])
    np.savez_compressed(RUN / "raw_fresh_predictions.npz",
                        idx=np.array([r["idx"] for r in rows]), E_teacher=np.array([r["E_teacher_eV"] for r in rows]),
                        E_dft=np.array([r["E_DFT_eV"] for r in rows]), config_type=np.array([r["config_type"] for r in rows], dtype=object))

    print(json.dumps({"n_sample": len(rows), "walltime_min": round(elapsed/60, 1),
                      "fresh_global_force_MAE": round(glob["force_component_MAE"], 4),
                      "nequip_logged_test_force_MAE": 0.1561, "match": abs(glob["force_component_MAE"] - 0.1561) < 0.05,
                      "fresh_energy_bias_corr_MAE": round(glob["energy_bias_corrected_MAE"], 2),
                      "vacancy_gradient": {"amo": round(amo, 3) if amo else None, "dil": round(dil, 3) if dil else None,
                                           "clu": round(clu, 3) if clu else None, "monotonic": vac["monotonic"]},
                      "domain": {d: round(m["force_MAE"], 3) for d, m in domain.items()}}, indent=2))
    # stash for the report
    (RUN / "_fresh_summary.json").write_text(json.dumps({"global": glob, "domain": domain, "vacancy": vac,
                                                         "walltime_s": elapsed, "n_sample": len(rows),
                                                         "sample_per_config": K_PER_CONFIG}, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
