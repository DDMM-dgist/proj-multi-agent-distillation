"""Framework V2 -- CandidateSelector (diversity + fail-closed disjointness).

Selects a diverse subset of generated candidates for canonical labeling. In the
INITIAL acquisition phase the only admissible selection signal is
descriptor-space diversity/novelty (no model uncertainty / no expected
information gain). This module implements deterministic farthest-point sampling
(FPS) over candidate descriptors, then applies the MANDATORY protected-reference
disjointness check: if any selected candidate overlaps a protected reference the
selection fails closed (never silently drops the overlap), and DFT labels are
never used as selection scores.

Both the descriptor provider and the disjointness checker are injected so the
selector stays independent of any particular descriptor implementation or the
protected-reference machinery, and is fully testable with fakes.
"""
from __future__ import annotations

from typing import Callable, Sequence

from framework_v2.acquisition.contracts import (
    CandidateGenerationResult,
    CandidateSelectionResult,
    ProtectedDisjointnessReport,
)


def farthest_point_selection(
    vectors: Sequence[Sequence[float]], k: int, *, seed_index: int = 0
) -> list[int]:
    """Deterministic FPS: greedily pick the point farthest (max-min Euclidean
    distance) from those already chosen, starting from ``seed_index``.

    Returns the selected indices in selection order. Deterministic given the
    input ordering and seed_index -- no RNG."""
    import numpy as np

    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(vectors, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    if not (0 <= seed_index < n):
        raise ValueError("seed_index out of range")

    selected = [seed_index]
    # min distance from every point to the selected set
    min_dist = np.linalg.norm(arr - arr[seed_index], axis=1)
    while len(selected) < k:
        # farthest point; ties broken by lowest index (argmax is stable)
        nxt = int(np.argmax(min_dist))
        if nxt in selected:
            # all remaining points are duplicates of selected; pick next unused
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            nxt = remaining[0]
        selected.append(nxt)
        d = np.linalg.norm(arr - arr[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return selected


def select_candidates(
    *,
    selection_id: str,
    generation_result: CandidateGenerationResult,
    descriptors: Sequence[Sequence[float]],
    k: int,
    disjointness_checker: Callable[[list[str]], ProtectedDisjointnessReport],
    selector: str = "farthest_point_sampling",
    seed_index: int = 0,
) -> CandidateSelectionResult:
    """Diversity-select ``k`` candidates then verify protected disjointness.

    ``descriptors`` is aligned to ``generation_result.candidate_ids``.
    Fails closed (via CandidateSelectionResult's validator) if the
    disjointness report is not PASS."""
    ids = list(generation_result.candidate_ids)
    if len(descriptors) != len(ids):
        raise ValueError(
            "descriptors must align 1:1 with generation_result.candidate_ids"
        )
    if not ids:
        raise ValueError("no candidates to select from")

    chosen_idx = farthest_point_selection(descriptors, k, seed_index=seed_index)
    selected_ids = [ids[i] for i in chosen_idx]

    report = disjointness_checker(selected_ids)

    diversity_evidence = {
        "selector": selector,
        "n_candidates": len(ids),
        "n_selected": len(selected_ids),
        "seed_index": seed_index,
        "selection_order_indices": chosen_idx,
    }

    return CandidateSelectionResult(
        selection_id=selection_id,
        generation_result_sha256=generation_result.content_sha256(),
        selector=selector,
        selector_config={"k": k, "seed_index": seed_index},
        selected_candidate_ids=selected_ids,
        diversity_evidence=diversity_evidence,
        disjointness_report=report,
    )
