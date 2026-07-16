"""Uncertainty adapter: committee-based fidelity ranking.

`kind: committee-force-std` (sigma_F) is architecture-agnostic by construction
— it only needs N force predictions from N same-`kind` student seeds on the
same structures, never touching model internals. This is why it needs no
per-`kind` dispatch branch: any student adapter that returns per-atom forces
can be committee-ensembled this way.

The committee score measures student disagreement. Its relationship to
student-teacher error is evaluated separately, and it is not by itself a
calibrated estimate of teacher-reference error.
"""
import numpy as np
from scipy.stats import spearmanr


def committee_force_std(forces_per_seed, aggregate="max"):
    """forces_per_seed: array-like, shape (n_seeds, n_atoms, 3).
    Returns per-atom sigma_F (n_atoms,) and one aggregated per-frame score.

    sigma_F is the RMS standard deviation over the three Cartesian force
    components, matching the manuscript definition
    sqrt(sum_{m,alpha}(F_malpha-Fbar_alpha)^2 / (3 M)).
    """
    F = np.asarray(forces_per_seed)
    if F.ndim != 3 or F.shape[0] < 2 or F.shape[2] != 3:
        raise ValueError("forces_per_seed must have shape (n_seeds>=2, n_atoms, 3)")
    per_atom_std = F.std(axis=0)                      # (n_atoms, 3)
    per_atom_sigma = np.sqrt(np.mean(per_atom_std ** 2, axis=-1))
    if aggregate == "max":
        frame_score = per_atom_sigma.max()
    elif aggregate == "mean":
        frame_score = per_atom_sigma.mean()
    else:
        raise ValueError(f"unknown aggregate={aggregate!r}")
    return per_atom_sigma, frame_score


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """Tie-correct Spearman rank correlation."""
    return float(spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).statistic)


def top_decile_enrichment(sigma_scores, true_error_scores, fraction=0.10):
    """What fraction of the top-error frames are recovered by ranking on
    sigma_F alone (the "find low-fidelity regions without DFT" claim).
    Returns (recall, enrichment_over_random, k).
    """
    n = len(sigma_scores)
    k = max(1, round(fraction * n))
    top_sigma = set(np.argsort(sigma_scores)[::-1][:k])
    top_error = set(np.argsort(true_error_scores)[::-1][:k])
    recall = len(top_sigma & top_error) / k
    enrichment = recall / fraction
    return recall, enrichment, k
