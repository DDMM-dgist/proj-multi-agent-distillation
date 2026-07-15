#!/usr/bin/env python3
"""Four-channel error audit: teacher-vs-DFT, student-vs-teacher, student-vs-DFT,
and (optionally) student-MD-trajectory-vs-DFT. This is the core diagnostic
methodology (see agents/analyst.md) and is fully material/architecture-
independent — it only reads labels off structures, never model internals.

Expected input: one extxyz file where each frame carries whichever of the
following labels are available (missing ones simply skip that channel):
  atoms.info["dft_energy"],        atoms.arrays["dft_forces"]
  atoms.info["teacher_energy"],    atoms.arrays["teacher_forces"]
  atoms.info["student_energy"],    atoms.arrays["student_forces"]
      (or, for a committee: atoms.info["student_energy_seed01"], ... "_seedNN",
       atoms.arrays["student_forces_seed01"], ... — averaged automatically)
  atoms.info["config_type"]  (optional — enables a per-config_type breakdown)

Usage:
    python validation/four_channel_audit.py labeled_frames.extxyz [--per-config-type]
"""
import argparse
import re
from collections import defaultdict

import numpy as np
from ase.io import read


def _seed_keys(info_or_arrays, prefix):
    return sorted(k for k in info_or_arrays if re.fullmatch(rf"{prefix}_seed\d+", k))


def committee_mean_energy(atoms, prefix):
    if f"{prefix}_energy" in atoms.info:
        return atoms.info[f"{prefix}_energy"]
    keys = _seed_keys(atoms.info, f"{prefix}_energy")
    if not keys:
        return None
    return float(np.mean([atoms.info[k] for k in keys]))


def committee_mean_forces(atoms, prefix):
    if f"{prefix}_forces" in atoms.arrays:
        return atoms.arrays[f"{prefix}_forces"]
    keys = _seed_keys(atoms.arrays, f"{prefix}_forces")
    if not keys:
        return None
    return np.mean([atoms.arrays[k] for k in keys], axis=0)


def r2(ref, pred):
    ref, pred = np.asarray(ref).ravel(), np.asarray(pred).ravel()
    ss_res = ((pred - ref) ** 2).sum()
    ss_tot = ((ref - ref.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def channel(frames, ref_prefix, pred_prefix, per_config_type=False):
    """Per-atom, shift-corrected energy MAE/RMSE/R2 + atom-weighted force MAE/RMSE/R2
    for one (ref, pred) pair, e.g. ref_prefix='dft', pred_prefix='teacher'."""
    rows = []
    for a in frames:
        e_ref = committee_mean_energy(a, ref_prefix)
        e_pred = committee_mean_energy(a, pred_prefix)
        f_ref = committee_mean_forces(a, ref_prefix)
        f_pred = committee_mean_forces(a, pred_prefix)
        if e_ref is None or e_pred is None or f_ref is None or f_pred is None:
            continue
        rows.append(dict(
            n=len(a), e_ref=e_ref, e_pred=e_pred, f_ref=f_ref, f_pred=f_pred,
            config_type=a.info.get("config_type", "unlabeled"),
        ))
    if not rows:
        return None

    def summarize(subset):
        n_atoms_total = sum(r["n"] for r in subset)
        de_per_atom = np.array([(r["e_pred"] - r["e_ref"]) / r["n"] for r in subset])
        shift = de_per_atom.mean()
        e_mae = np.abs(de_per_atom - shift).mean()
        e_rmse = np.sqrt(((de_per_atom - shift) ** 2).mean())
        f_ref_all = np.concatenate([r["f_ref"] for r in subset])
        f_pred_all = np.concatenate([r["f_pred"] for r in subset])
        f_mae = np.abs(f_pred_all - f_ref_all).mean()
        f_rmse = np.sqrt(((f_pred_all - f_ref_all) ** 2).mean())
        f_r2 = r2(f_ref_all, f_pred_all)
        return dict(n_frames=len(subset), n_atoms=n_atoms_total,
                    e_mae_meV=e_mae * 1000, e_rmse_meV=e_rmse * 1000,
                    f_mae=f_mae, f_rmse=f_rmse, f_r2=f_r2)

    out = {"all": summarize(rows)}
    if per_config_type:
        by_ct = defaultdict(list)
        for r in rows:
            by_ct[r["config_type"]].append(r)
        for ct, subset in by_ct.items():
            out[ct] = summarize(subset)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", help="extxyz file with labeled frames")
    ap.add_argument("--per-config-type", action="store_true",
                     help="also break down each channel by atoms.info['config_type']")
    args = ap.parse_args()

    frames = read(args.frames, index=":")
    print(f"loaded {len(frames)} frames from {args.frames}")

    channels = [
        ("(a) teacher vs DFT", "dft", "teacher"),
        ("(b) student vs teacher", "teacher", "student"),
        ("(c) student vs DFT", "dft", "student"),
    ]
    for label, ref, pred in channels:
        result = channel(frames, ref, pred, per_config_type=args.per_config_type)
        if result is None:
            print(f"{label}: SKIPPED (missing labels for '{ref}' and/or '{pred}')")
            continue
        a = result["all"]
        print(f"{label}: n={a['n_frames']} frames / {a['n_atoms']} atoms | "
              f"E_MAE={a['e_mae_meV']:.2f} meV/atom E_RMSE={a['e_rmse_meV']:.2f} meV/atom | "
              f"F_MAE={a['f_mae']:.4f} F_RMSE={a['f_rmse']:.4f} F_R2={a['f_r2']:.3f}")
        if args.per_config_type:
            for ct, s in result.items():
                if ct == "all":
                    continue
                print(f"    {ct:32s} n={s['n_frames']:4d}  "
                      f"E_MAE={s['e_mae_meV']:6.2f}  F_MAE={s['f_mae']:.4f}  F_R2={s['f_r2']:.3f}")

    print(
        "\nDiagnostic reminder: if (b) ~ (a), the residual is teacher-limited, not "
        "distillation-limited (see agents/analyst.md). Channel (d) — student-MD "
        "trajectory vs DFT single-points on carved snapshots — is computed "
        "separately once production trajectories exist; see agents/simulation.md's "
        "small-cell recipe."
    )


if __name__ == "__main__":
    main()
