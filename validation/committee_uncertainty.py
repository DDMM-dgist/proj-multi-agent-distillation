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
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from ase.io import read

# Work whether or not `pip install -e .` was run (see pyproject.toml).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapters.uncertainty import committee_force_std, pearson, spearman, top_decile_enrichment
from validation.report import evidence_record


def _seed_force_stack(atoms, prefix):
    keys = sorted(k for k in atoms.arrays if re.fullmatch(rf"{prefix}_seed\d+", k))
    if not keys:
        return None
    return np.stack([atoms.arrays[k] for k in keys])  # (n_seeds, n_atoms, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames")
    ap.add_argument("--aggregate", choices=["max", "mean"], default=None,
                    help="atom-to-frame reduction; overrides the config")
    ap.add_argument("--config", help="optional configs/uncertainty.yaml")
    ap.add_argument("--top-fraction", type=float, default=None,
                    help="top-ranked fraction; overrides the config")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail if any frame lacks teacher or committee forces")
    ap.add_argument("--output", help="optional JSON report path")
    args = ap.parse_args()
    aggregate, top_fraction, require_complete = args.aggregate, args.top_fraction, args.require_complete
    if args.config:
        with open(args.config) as handle:
            config = yaml.safe_load(handle)
        aggregate = aggregate or config.get("aggregate")
        top_fraction = top_fraction if top_fraction is not None else config.get("top_fraction")
        require_complete = require_complete or bool(config.get("require_complete", False))
    aggregate = aggregate or "mean"
    top_fraction = 0.10 if top_fraction is None else float(top_fraction)
    if not 0 < top_fraction <= 1:
        raise SystemExit("top_fraction must be in (0, 1]")

    frames = read(args.frames, index=":")
    sigma_scores, true_errors, seed_counts = [], [], []
    skipped = 0
    for a in frames:
        seed_forces = _seed_force_stack(a, "student_forces")
        teacher_forces = a.arrays.get("teacher_forces")
        if seed_forces is None or teacher_forces is None:
            skipped += 1
            continue
        _, frame_sigma = committee_force_std(seed_forces, aggregate=aggregate)
        seed_counts.append(seed_forces.shape[0])
        student_mean = seed_forces.mean(axis=0)
        true_frame_error = np.abs(student_mean - teacher_forces).mean()
        sigma_scores.append(frame_sigma)
        true_errors.append(true_frame_error)

    if skipped:
        print(f"skipped {skipped}/{len(frames)} frames (missing per-seed forces or teacher reference)")
    if require_complete and skipped:
        raise SystemExit(f"required uncertainty coverage is incomplete: {len(frames)-skipped}/{len(frames)} frames")
    if len(set(seed_counts)) > 1:
        raise SystemExit("committee seed count is inconsistent across frames")
    if len(sigma_scores) < 10:
        raise SystemExit(f"only {len(sigma_scores)} usable frames — need more for a meaningful ranking")

    sigma_scores, true_errors = np.array(sigma_scores), np.array(true_errors)
    r = pearson(sigma_scores, true_errors)
    rho = spearman(sigma_scores, true_errors)
    if not np.isfinite([r, rho]).all():
        raise SystemExit("uncertainty ranking correlation is undefined for constant or non-finite data")
    recall, enrichment, k = top_decile_enrichment(sigma_scores, true_errors, top_fraction)

    result = {"schema_version": 1, "n_frames": len(sigma_scores),
              "n_total_frames": len(frames), "n_skipped_frames": skipped,
              "n_committee_seeds": seed_counts[0],
              "aggregate": aggregate, "pearson": r, "spearman": rho,
              "top_fraction": top_fraction, "top_k": k,
              "top_error_recall": recall, "enrichment_over_random": enrichment,
              "evidence": [evidence_record("evaluated_frames", args.frames)]}
    if args.config:
        result["evidence"].append(evidence_record("uncertainty_config", args.config))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(f"n_frames={len(sigma_scores)}")
    print(f"aggregate={aggregate}")
    print(f"sigma_F <-> student-teacher force error:  Pearson r={r:.3f}  Spearman rho={rho:.3f}")
    print(f"top-{top_fraction:.0%} sigma_F recall of top-{top_fraction:.0%} true error: "
          f"{recall:.1%}  ({enrichment:.1f}x over random)  k={k}")
    print(
        "\nFraming reminder: this tests student-committee disagreement against "
        "student-teacher error. It does not by itself establish calibration "
        "against the teacher's own reference error; report the tested "
        "relationship and coverage."
    )


if __name__ == "__main__":
    main()
