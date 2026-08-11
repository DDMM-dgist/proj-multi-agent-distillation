#!/usr/bin/env python3
"""PC001 root-cause discriminator: Teacher b56e20ff error on the SAME central domains in TRAIN vs held-out
TEST. Fresh Allegro inference (APPROVED) on the central-domain TRAIN frames; TEST reused from the committed
exact-heldout per-frame metrics. IDENTICAL NequIPCalculator path + metric definitions as the test run.
NO Student, DFT, MD, training, network, Judge. VASP untouched."""
import sys, time, json, csv
import numpy as np
from pathlib import Path
import torch

MODEL = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/teacher/model.nequip.pth"
DATASET = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/03_allegro_train/dataset.xyz"
ROOT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime")
RUN = ROOT / "runs" / "production_campaign_001" / "pc001-teacher-exact-heldout-validation"   # append into the authoritative run
TEST_CSV = RUN / "fresh_test_per_frame_metrics.csv"
N = 11424
DOMAIN = {
    "amorphous_SiO2": {"bulk_amo", "quench", "quench_int_AL", "liquid"},
    "SiO2x_dilute_vacancy": {"vacancy_int_AL", "vacancy", "SiOx_int_AL"},
    "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL", "quench_max_AL", "surfaces_max_AL"},
}
CENTRAL_CFGS = {c for s in DOMAIN.values() for c in s}


def dfam(c):
    for d, s in DOMAIN.items():
        if c in s: return d
    return None


def per_frame_record(cfg, nat, Fd, Ft):
    comp = np.abs(Ft - Fd)
    return {"config_type": cfg, "natoms": nat,
            "dft_comp_RMS": float(np.sqrt((Fd.reshape(-1) ** 2).mean())),
            "dft_comp_meanabs": float(np.abs(Fd.reshape(-1)).mean()),
            "err_comp_MAE": float(comp.mean()),
            "err_comp_RMSE": float(np.sqrt((comp ** 2).mean())),
            "vec_mean": float(np.linalg.norm(Ft - Fd, axis=1).mean())}


def agg(rows):
    if not rows: return None
    at3 = sum(r["natoms"] * 3 for r in rows)
    dft_rms = float(np.sqrt(sum(r["dft_comp_RMS"] ** 2 * r["natoms"] * 3 for r in rows) / at3))
    dft_ma = float(sum(r["dft_comp_meanabs"] * r["natoms"] * 3 for r in rows) / at3)
    err_mae = float(sum(r["err_comp_MAE"] * r["natoms"] * 3 for r in rows) / at3)
    err_rmse = float(np.sqrt(sum(r["err_comp_RMSE"] ** 2 * r["natoms"] * 3 for r in rows) / at3))
    fm = np.array([r["err_comp_MAE"] for r in rows])
    return {"N": len(rows), "DFT_comp_RMS": round(dft_rms, 4), "err_comp_MAE": round(err_mae, 4),
            "err_comp_RMSE": round(err_rmse, 4), "normalized_MAE": round(err_mae / dft_ma, 4),
            "normalized_RMSE": round(err_rmse / dft_rms, 4), "frame_median": round(float(np.median(fm)), 4),
            "frame_p90": round(float(np.quantile(fm, .9)), 4), "frame_p95": round(float(np.quantile(fm, .95)), 4),
            "frame_max": round(float(fm.max()), 4)}


def main():
    from ase.io import read
    from nequip.ase import NequIPCalculator
    gen = torch.Generator().manual_seed(123)
    sp = torch.utils.data.random_split(list(range(N)), [0.8, 0.1, 0.1], generator=gen)
    train_idx = [i for i in (int(x) for x in sp[0].indices)]

    calc = NequIPCalculator.from_compiled_model(MODEL, device="cpu",
                                                chemical_species_to_atom_type_map={"O": "O", "Si": "Si"})
    allA = read(DATASET, index=":")
    # TRAIN central-domain frames
    train_rows = []
    t0 = time.time()
    train_central = [i for i in train_idx if allA[i].info.get("config_type", "?") in CENTRAL_CFGS]
    counts = {}
    for i in train_central:
        counts[allA[i].info.get("config_type")] = counts.get(allA[i].info.get("config_type"), 0) + 1
    for i in train_central:
        a = allA[i]; cfg = a.info.get("config_type", "?"); nat = len(a)
        Fd = a.calc.results.get("forces") if (a.calc is not None and a.calc.results) else None
        if Fd is None or not np.all(np.isfinite(Fd)):
            continue
        Fd = np.asarray(Fd)
        b = a.copy(); b.calc = calc
        b.get_potential_energy(); Ft = b.get_forces()
        r = per_frame_record(cfg, nat, Fd, Ft); r["idx"] = i; train_rows.append(r)
    elapsed = time.time() - t0

    # TEST central-domain frames: err from committed CSV, DFT scale re-read (identical definitions; no inference)
    test_err = {int(r["dataset_index"]): r for r in csv.DictReader(open(TEST_CSV))}
    test_idx = [int(x) for x in sp[2].indices]
    test_rows = []
    for i in test_idx:
        a = allA[i]; cfg = a.info.get("config_type", "?")
        if cfg not in CENTRAL_CFGS or i not in test_err:
            continue
        Fd = a.calc.results.get("forces") if (a.calc is not None and a.calc.results) else None
        if Fd is None: continue
        Fd = np.asarray(Fd); e = test_err[i]
        test_rows.append({"config_type": cfg, "natoms": int(e["natoms"]),
                          "dft_comp_RMS": float(np.sqrt((Fd.reshape(-1) ** 2).mean())),
                          "dft_comp_meanabs": float(np.abs(Fd.reshape(-1)).mean()),
                          "err_comp_MAE": float(e["force_component_MAE_eV_A"]),
                          "err_comp_RMSE": float(e["force_component_RMSE_eV_A"]),
                          "vec_mean": float(e["force_vector_error_mean_eV_A"])})

    # aggregate per domain + per config
    domains = {}
    for d in DOMAIN:
        tr = agg([r for r in train_rows if dfam(r["config_type"]) == d])
        te = agg([r for r in test_rows if dfam(r["config_type"]) == d])
        ratio_mae = round(te["err_comp_MAE"] / tr["err_comp_MAE"], 3) if (tr and te and tr["err_comp_MAE"]) else None
        ratio_norm = round(te["normalized_RMSE"] / tr["normalized_RMSE"], 3) if (tr and te and tr["normalized_RMSE"]) else None
        # classification (transparent rule): compare TRAIN vs TEST domain error
        cls = "UNRESOLVED"
        if tr and te:
            if ratio_mae <= 1.25:
                cls = "A_MODEL_TRAINING_UNDERFIT"   # train already ~ as bad as test
            elif ratio_mae >= 1.5:
                cls = "B_GENERALIZATION_COVERAGE_LIMITATION"  # train much better than test
            else:
                cls = "C_MIXED"
        per_cfg = {}
        for c in DOMAIN[d]:
            per_cfg[c] = {"TRAIN": agg([r for r in train_rows if r["config_type"] == c]),
                          "TEST": agg([r for r in test_rows if r["config_type"] == c])}
        domains[d] = {"TRAIN": tr, "TEST": te, "test_over_train_err_MAE_ratio": ratio_mae,
                      "test_over_train_normalized_RMSE_ratio": ratio_norm, "classification": cls,
                      "per_config": per_cfg}

    # csv
    with open(RUN / "train_vs_test_domain_fidelity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "split", "N", "DFT_comp_RMS", "err_comp_MAE", "err_comp_RMSE",
                    "normalized_MAE", "normalized_RMSE", "frame_median", "frame_p90", "frame_p95", "frame_max"])
        for d, v in domains.items():
            for sname in ("TRAIN", "TEST"):
                m = v[sname]
                if m: w.writerow([d, sname, m["N"], m["DFT_comp_RMS"], m["err_comp_MAE"], m["err_comp_RMSE"],
                                  m["normalized_MAE"], m["normalized_RMSE"], m["frame_median"], m["frame_p90"], m["frame_p95"], m["frame_max"]])

    convergence = {"final_epoch": 158, "lr_final": 9.77e-6, "min_lr": 1e-6,
                   "train_forces_mae": 0.14942, "val_forces_mae": 0.16266, "test_forces_mae": 0.15613,
                   "train_forces_rmse": 0.24781, "test_forces_rmse": 0.47417,
                   "early_stopping": "patience 20 on val0_epoch/weighted_sum; LR decayed ReduceLROnPlateau to ~min_lr => plateaued",
                   "force_energy_loss_weights": {"total_energy": 1.0, "forces": 1.0},
                   "train_val_gap_force_mae": 0.0132,
                   "interpretation": "training converged/plateaued (LR ~1e-5, early-stopping monitored, train~val close) => NOT gross optimization undertraining"}

    # overall root cause synthesis
    dil = domains["SiO2x_dilute_vacancy"]; clu = domains["SiO2x_clustered_vacancy_voidsurface"]; amo = domains["amorphous_SiO2"]
    out = {
        "note": "Teacher error on the SAME central domains, TRAIN vs held-out TEST (identical NequIPCalculator path + metric defs). TRAIN fresh inference; TEST reused from committed exact-heldout per-frame metrics.",
        "train_central_counts": counts, "n_train_evaluated": len(train_rows), "walltime_min": round(elapsed / 60, 1),
        "domains": domains, "training_convergence": convergence,
        "per_domain_classification": {d: v["classification"] for d, v in domains.items()},
        "downgrade_note": "supersedes the earlier TEACHER_MODEL_TRAINING_LIMITATION claim (which was ROOT_CAUSE_PENDING); count/composition/force-scale coverage remain established, local-environment/diversity sufficiency were NOT claimed."
    }
    (RUN / "train_vs_test_root_cause.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps({"train_counts": counts, "n_train_eval": len(train_rows), "walltime_min": round(elapsed/60, 1),
                      "TRAIN_err_MAE": {d: domains[d]["TRAIN"]["err_comp_MAE"] for d in DOMAIN},
                      "TEST_err_MAE": {d: domains[d]["TEST"]["err_comp_MAE"] for d in DOMAIN},
                      "TRAIN_norm_RMSE": {d: domains[d]["TRAIN"]["normalized_RMSE"] for d in DOMAIN},
                      "TEST_norm_RMSE": {d: domains[d]["TEST"]["normalized_RMSE"] for d in DOMAIN},
                      "test_over_train_ratio": {d: domains[d]["test_over_train_err_MAE_ratio"] for d in DOMAIN},
                      "classification": {d: domains[d]["classification"] for d in DOMAIN}}, indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())
