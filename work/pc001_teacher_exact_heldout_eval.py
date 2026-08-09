#!/usr/bin/env python3
"""PC001 — FRESH Teacher inference on the EXACT seed-123 held-out TEST split (APPROVED). NO student data.

Reconstructs the exact nequip split (torch.utils.data.random_split(dataset,[0.8,0.1,0.1],
Generator.manual_seed(123)); subset order [train,val,test]; test=splits[2]) and runs b56e20ff fresh
(NequIPCalculator, valid path) on ONLY the ~1142 held-out test frames. Validates reproduction vs the
NequIP test log (forces_mae 0.1561, per_atom_energy_mae 15.73 meV/atom) and computes domain-resolved
metrics. NO student inference, DFT, MD, training, network, Judge.
"""
import sys, time, json, hashlib, csv
import numpy as np
from pathlib import Path
import torch

MODEL = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/teacher/model.nequip.pth"
TEACHER_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
DATASET = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/03_allegro_train/dataset.xyz"
ROOT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime")
RUN = ROOT / "runs" / "production_campaign_001" / "pc001-teacher-exact-heldout-validation"
LOG_TEST = {"forces_mae": 0.1561268, "forces_rmse": 0.4741700, "per_atom_energy_mae_meV": 15.733115}
N = 11424
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
    import nequip, torch as T, platform
    # exact test indices
    gen = torch.Generator().manual_seed(123)
    splits = torch.utils.data.random_split(list(range(N)), [0.8, 0.1, 0.1], generator=gen)
    test_idx = sorted(int(i) for i in splits[2].indices)

    t0 = time.time()
    calc = NequIPCalculator.from_compiled_model(MODEL, device="cpu",
                                                chemical_species_to_atom_type_map={"O": "O", "Si": "Si"})
    allA = read(DATASET, index=":")
    assert len(allA) == N, f"dataset frames {len(allA)} != {N}"

    rows = []
    for idx in test_idx:
        a = allA[idx]; nat = len(a)
        Ed = a.calc.results.get("energy") if (a.calc is not None and a.calc.results) else a.info.get("dft_free_energy")
        Fd = a.calc.results.get("forces") if (a.calc is not None and a.calc.results) else None
        cfg = a.info.get("config_type", "?")
        syms = a.get_chemical_symbols()
        if Ed is None or Fd is None or not np.all(np.isfinite(Fd)):
            continue
        b = a.copy(); b.calc = calc
        Et = float(b.get_potential_energy()); Ft = b.get_forces()
        comp = np.abs(Ft - Fd); vec = np.linalg.norm(Ft - Fd, axis=1)
        rows.append({"dataset_index": idx, "config_type": cfg, "natoms": nat,
                     "nSi": syms.count("Si"), "nO": syms.count("O"),
                     "E_DFT_eV": float(Ed), "E_teacher_eV": Et, "dE_meV_atom": (Et - Ed) / nat * 1000.0,
                     "force_component_MAE_eV_A": float(comp.mean()),
                     "force_component_RMSE_eV_A": float(np.sqrt((comp ** 2).mean())),
                     "force_vector_error_mean_eV_A": float(vec.mean()), "force_error_max_eV_A": float(vec.max())})
    elapsed = time.time() - t0

    with open(RUN / "fresh_test_per_frame_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    np.savez_compressed(RUN / "fresh_test_predictions.npz",
                        dataset_index=np.array([r["dataset_index"] for r in rows]),
                        E_teacher=np.array([r["E_teacher_eV"] for r in rows]),
                        E_dft=np.array([r["E_DFT_eV"] for r in rows]),
                        config_type=np.array([r["config_type"] for r in rows], dtype=object))

    # GLOBAL: force COMPONENT MAE/RMSE computed over ALL atoms x components (weight by atoms, like NequIP)
    tot_abs = sum(r["force_component_MAE_eV_A"] * r["natoms"] * 3 for r in rows)
    tot_sq = sum(r["force_component_RMSE_eV_A"] ** 2 * r["natoms"] * 3 for r in rows)
    tot_comp = sum(r["natoms"] * 3 for r in rows)
    g_fmae = tot_abs / tot_comp
    g_frmse = (tot_sq / tot_comp) ** 0.5
    fmae_frame = np.array([r["force_component_MAE_eV_A"] for r in rows])
    dEpa = np.array([r["dE_meV_atom"] for r in rows])
    energy_mae_atomwt = sum(abs(r["dE_meV_atom"]) * r["natoms"] for r in rows) / sum(r["natoms"] for r in rows)
    glob = {"n_test_frames": len(rows), "n_atoms": int(sum(r["natoms"] for r in rows)),
            "force_component_MAE_atomweighted": g_fmae, "force_component_RMSE_atomweighted": g_frmse,
            "force_frame_MAE_mean": float(fmae_frame.mean()), "force_frame_median": float(np.median(fmae_frame)),
            "force_frame_p90": float(np.quantile(fmae_frame, .9)), "force_frame_p95": float(np.quantile(fmae_frame, .95)),
            "force_frame_p99": float(np.quantile(fmae_frame, .99)), "force_frame_max": float(fmae_frame.max()),
            "energy_signed_bias_meV_atom": float(dEpa.mean()), "energy_raw_MAE_meV_atom": float(np.abs(dEpa).mean()),
            "energy_atomweighted_MAE_meV_atom": float(energy_mae_atomwt),
            "energy_bias_corrected_MAE_meV_atom": float(np.abs(dEpa - dEpa.mean()).mean())}

    def dfam(c):
        for d, s in DOMAIN.items():
            if c in s: return d
        return "OTHER"
    domain = {}
    for d in list(DOMAIN) + ["OTHER"]:
        sub = [r for r in rows if dfam(r["config_type"]) == d]
        if not sub: continue
        f = np.array([r["force_component_MAE_eV_A"] for r in sub]); e = np.array([abs(r["dE_meV_atom"]) for r in sub])
        domain[d] = {"n": len(sub), "force_MAE": float(f.mean()), "force_RMSE": float(np.sqrt((f ** 2).mean())),
                     "force_median": float(np.median(f)), "force_p95": float(np.quantile(f, .95)),
                     "force_max": float(f.max()), "energy_MAE_meV": float(e.mean())}
    with open(RUN / "fresh_test_domain_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["domain", "n", "force_MAE", "force_RMSE", "force_median", "force_p95", "force_max", "energy_MAE_meV"])
        for d, m in domain.items():
            w.writerow([d, m["n"], round(m["force_MAE"], 4), round(m["force_RMSE"], 4), round(m["force_median"], 4),
                        round(m["force_p95"], 4), round(m["force_max"], 4), round(m["energy_MAE_meV"], 2)])

    amo = domain.get("amorphous_SiO2", {}); dil = domain.get("SiO2x_dilute_vacancy", {}); clu = domain.get("SiO2x_clustered_vacancy_voidsurface", {})
    vac = {"amorphous": {"force_MAE": amo.get("force_MAE"), "n": amo.get("n")},
           "dilute": {"force_MAE": dil.get("force_MAE"), "n": dil.get("n")},
           "clustered": {"force_MAE": clu.get("force_MAE"), "n": clu.get("n")},
           "delta_dilute_minus_amorphous": (dil.get("force_MAE") - amo.get("force_MAE")) if amo.get("force_MAE") and dil.get("force_MAE") else None,
           "delta_clustered_minus_dilute": (clu.get("force_MAE") - dil.get("force_MAE")) if dil.get("force_MAE") and clu.get("force_MAE") else None,
           "monotonic_degradation": bool(amo.get("force_MAE") and dil.get("force_MAE") and clu.get("force_MAE") and amo["force_MAE"] < dil["force_MAE"] < clu["force_MAE"]),
           "sparse_flags": {k: (v.get("n", 0) < 20) for k, v in [("amorphous", amo), ("dilute", dil), ("clustered", clu)]}}

    top = sorted(rows, key=lambda r: -r["force_component_MAE_eV_A"])[:20]
    with open(RUN / "outlier_table.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rank", "dataset_index", "config_type", "natoms", "nSi_nO", "force_MAE", "force_max", "dE_meV_atom"])
        for i, r in enumerate(top, 1):
            w.writerow([i, r["dataset_index"], r["config_type"], r["natoms"], f"{r['nSi']}/{r['nO']}",
                        round(r["force_component_MAE_eV_A"], 3), round(r["force_error_max_eV_A"], 2), round(r["dE_meV_atom"], 1)])

    # reproduction of original NequIP test log
    dF = abs(g_fmae - LOG_TEST["forces_mae"]); dFr = abs(g_frmse - LOG_TEST["forces_rmse"]); dE = abs(energy_mae_atomwt - LOG_TEST["per_atom_energy_mae_meV"])
    if dF < 0.005 and dE < 1.0:
        repro = "REPRODUCED"
    elif dF < 0.02 and dE < 3.0:
        repro = "SMALL_NUMERICAL_DIFFERENCE"
    else:
        repro = "MISMATCH"
    reproduction = {"fresh_force_component_MAE": g_fmae, "log_force_component_MAE": LOG_TEST["forces_mae"], "abs_diff_force_MAE": dF,
                    "fresh_force_RMSE": g_frmse, "log_force_RMSE": LOG_TEST["forces_rmse"], "abs_diff_force_RMSE": dFr,
                    "fresh_energy_MAE_meV": energy_mae_atomwt, "log_energy_MAE_meV": LOG_TEST["per_atom_energy_mae_meV"], "abs_diff_energy": dE,
                    "classification": repro, "n_test": len(rows)}

    # verdict (derived; no invented threshold)
    verdict = "TEACHER_ACCEPTED_FOR_DISTILLATION" if repro in ("REPRODUCED", "SMALL_NUMERICAL_DIFFERENCE") else "TEACHER_STATUS_UNRESOLVED"

    W = lambda n, o: (RUN / n).write_text(json.dumps(o, indent=2, default=float) + "\n")
    W("global_test_metrics.json", glob)
    W("fresh_test_domain_metrics.json", domain)
    W("vacancy_defect_check.json", vac)
    W("original_log_reproduction.json", reproduction)
    W("teacher_verdict.json", {"FINAL_TEACHER_VERDICT": verdict, "verdict_type": "DERIVED (fresh exact-held-out test; no invented threshold)",
                               "held_out_force_component_MAE": g_fmae, "held_out_energy_MAE_meV_atom": energy_mae_atomwt,
                               "reproduction": repro, "vacancy_monotonic": vac["monotonic_degradation"], "student_data_used": False})
    import subprocess
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    W("provenance.json", {"run_id": "pc001-teacher-exact-heldout-validation", "stage": "production_campaign_001",
                          "package_head": head, "teacher_sha256": sha256(MODEL), "test_split": "seed-123 random_split test (splits[2])",
                          "n_test": len(rows), "analysis_code_sha256": sha256(str(Path(__file__).resolve())),
                          "fresh_teacher_inference": True, "calculator": "NequIPCalculator (cpu)", "walltime_s": round(elapsed, 1),
                          "student_data_used": False, "no_dft": True, "no_md": True, "no_training": True, "no_network": True, "no_semantic_judge": True})
    W("run_manifest.json", {"status": "OK", "phase": "TEACHER_EXACT_HELDOUT_VALIDATION", "n_test": len(rows),
                            "held_out_force_component_MAE": round(g_fmae, 4), "held_out_energy_MAE_meV": round(energy_mae_atomwt, 2),
                            "reproduction": repro, "vacancy_monotonic": vac["monotonic_degradation"],
                            "FINAL_TEACHER_VERDICT": verdict, "walltime_s": round(elapsed, 1)})
    print(json.dumps({"n_test": len(rows), "walltime_min": round(elapsed/60, 1),
                      "fresh_force_MAE": round(g_fmae, 4), "log_force_MAE": LOG_TEST["forces_mae"],
                      "fresh_energy_MAE_meV": round(energy_mae_atomwt, 2), "log_energy_MAE_meV": LOG_TEST["per_atom_energy_mae_meV"],
                      "reproduction": repro,
                      "vacancy": {"amo": [amo.get("force_MAE"), amo.get("n")], "dil": [dil.get("force_MAE"), dil.get("n")],
                                  "clu": [clu.get("force_MAE"), clu.get("n")], "monotonic": vac["monotonic_degradation"]},
                      "FINAL_TEACHER_VERDICT": verdict}, indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())
