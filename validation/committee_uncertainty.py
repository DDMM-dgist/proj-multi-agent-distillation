#!/usr/bin/env python3
"""Committee-uncertainty ranking: does per-frame sigma_F (student-committee
force disagreement) predict the true student-vs-teacher force error?

Expected input: an extxyz file where each frame has per-seed student force
arrays (atoms.arrays["student_forces_seed01"], "_seed02", ...) AND a teacher
reference (atoms.arrays["teacher_forces"]) to compute the true error against.

Usage:
    python validation/committee_uncertainty.py labeled_frames.extxyz
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read

# Work whether or not `pip install -e .` was run (see pyproject.toml).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapters.uncertainty import committee_force_std, pearson, spearman, top_decile_enrichment


def _seed_force_stack(atoms, prefix):
    keys = sorted(k for k in atoms.arrays if re.fullmatch(rf"{prefix}_seed\d+", k))
    if not keys:
        return None
    return np.stack([atoms.arrays[k] for k in keys])  # (n_seeds, n_atoms, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames")
    ap.add_argument("--aggregate", choices=["max", "mean"], default="max")
    ap.add_argument("--top-fraction", type=float, default=0.10)
    args = ap.parse_args()

    frames = read(args.frames, index=":")
    sigma_scores, true_errors = [], []
    skipped = 0
    for a in frames:
        seed_forces = _seed_force_stack(a, "student_forces")
        teacher_forces = a.arrays.get("teacher_forces")
        if seed_forces is None or teacher_forces is None:
            skipped += 1
            continue
        _, frame_sigma = committee_force_std(seed_forces, aggregate=args.aggregate)
        student_mean = seed_forces.mean(axis=0)
        true_frame_error = np.abs(student_mean - teacher_forces).mean()
        sigma_scores.append(frame_sigma)
        true_errors.append(true_frame_error)

    if skipped:
        print(f"skipped {skipped}/{len(frames)} frames (missing per-seed forces or teacher reference)")
    if len(sigma_scores) < 10:
        raise SystemExit(f"only {len(sigma_scores)} usable frames — need more for a meaningful ranking")

    sigma_scores, true_errors = np.array(sigma_scores), np.array(true_errors)
    r = pearson(sigma_scores, true_errors)
    rho = spearman(sigma_scores, true_errors)
    recall, enrichment, k = top_decile_enrichment(sigma_scores, true_errors, args.top_fraction)

    print(f"n_frames={len(sigma_scores)}")
    print(f"sigma_F <-> student-teacher force error:  Pearson r={r:.3f}  Spearman rho={rho:.3f}")
    print(f"top-{args.top_fraction:.0%} sigma_F recall of top-{args.top_fraction:.0%} true error: "
          f"{recall:.1%}  ({enrichment:.1f}x over random)  k={k}")
    print(
        "\nFraming reminder: this ranks student-teacher FIDELITY. It is only "
        "rank-correlated (not necessarily magnitude-correlated) with the "
        "teacher's own DFT error — do not report sigma_F as calibrated "
        "uncertainty or a teacher-accuracy estimator (see configs/uncertainty.yaml)."
    )


if __name__ == "__main__":
    main()
