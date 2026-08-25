"""Performance-hotfix equivalence proof for ``_marginal_novelty_curve`` (FE-027 P3 sizing).

The shipped ``framework_v2.acquisition.generic_coverage._marginal_novelty_curve`` was changed from
an O(n^3) naive farthest-point traversal (Python-list membership scans) to an O(n^2 * d) traversal
using an O(1) boolean chosen-mask and an incremental running min-distance. FE-027 sizing SEMANTICS
must be UNCHANGED: same seed (index 0), same argmax-with-lowest-index tie-breaking, same per-axis
missing-axis-aware distance, same emitted marginal-novelty curve, and therefore the same knee /
saturation.

``_LEGACY_marginal_novelty_curve`` below is a verbatim copy of the pre-hotfix implementation, kept
here as the executable oracle. It reuses the module's real ``_distance`` (identical arithmetic and
summation order), so the optimized path must match the oracle BIT-FOR-BIT, not merely within a
tolerance. Both an instrumented (order-exposing) legacy and optimized traversal are compared so the
SELECTED SEQUENCE and tie-breaking -- not only the curve -- are proven identical.
"""
from __future__ import annotations

import random
import unittest

try:
    import ase  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from framework_v2.acquisition.generic_coverage import (
    FrameworkSizingParams,
    _axis_scales,
    _distance,
    _marginal_novelty_curve,
    compute_saturation,
)


# --------------------------------------------------------------------------------------------
# Verbatim pre-hotfix oracle (O(n^3), Python-list membership). DO NOT "optimize" this copy.
# --------------------------------------------------------------------------------------------
def _LEGACY_marginal_novelty_curve(vectors, axes, scales):
    n = len(vectors)
    if n < 2:
        return []
    chosen = [0]
    min_dist = [_distance(vectors[i], vectors[0], axes, scales) for i in range(n)]
    curve = []
    for _ in range(n - 1):
        best_i, best_d = -1, -1.0
        for i in range(n):
            if i in chosen:
                continue
            if min_dist[i] > best_d:
                best_d, best_i = min_dist[i], i
        if best_i < 0:
            break
        curve.append(best_d)
        chosen.append(best_i)
        for i in range(n):
            if i not in chosen:
                d = _distance(vectors[i], vectors[best_i], axes, scales)
                if d < min_dist[i]:
                    min_dist[i] = d
    return curve


def _legacy_order_and_curve(vectors, axes, scales):
    """Instrumented legacy traversal exposing the chosen index order AND the curve."""
    n = len(vectors)
    if n < 2:
        return ([0] if n == 1 else []), []
    chosen = [0]
    min_dist = [_distance(vectors[i], vectors[0], axes, scales) for i in range(n)]
    curve = []
    for _ in range(n - 1):
        best_i, best_d = -1, -1.0
        for i in range(n):
            if i in chosen:
                continue
            if min_dist[i] > best_d:
                best_d, best_i = min_dist[i], i
        if best_i < 0:
            break
        curve.append(best_d)
        chosen.append(best_i)
        for i in range(n):
            if i not in chosen:
                d = _distance(vectors[i], vectors[best_i], axes, scales)
                if d < min_dist[i]:
                    min_dist[i] = d
    return chosen, curve


def _optimized_order_and_curve(vectors, axes, scales):
    """Instrumented optimized traversal (mask + incremental min-dist) exposing order AND curve.
    Mirrors the shipped ``_marginal_novelty_curve`` logic exactly, additionally recording order."""
    n = len(vectors)
    if n < 2:
        return ([0] if n == 1 else []), []
    chosen_mask = [False] * n
    chosen_mask[0] = True
    order = [0]
    v0 = vectors[0]
    min_dist = [_distance(vectors[i], v0, axes, scales) for i in range(n)]
    curve = []
    for _ in range(n - 1):
        best_i, best_d = -1, -1.0
        for i in range(n):
            if chosen_mask[i]:
                continue
            di = min_dist[i]
            if di > best_d:
                best_d, best_i = di, i
        if best_i < 0:
            break
        curve.append(best_d)
        chosen_mask[best_i] = True
        order.append(best_i)
        vb = vectors[best_i]
        for i in range(n):
            if not chosen_mask[i]:
                d = _distance(vectors[i], vb, axes, scales)
                if d < min_dist[i]:
                    min_dist[i] = d
    return order, curve


@unittest.skipUnless(_HAS_DEPS, "pydantic/ase not installed")
class MarginalNoveltyCurveEquivalence(unittest.TestCase):

    def _assert_full_equivalence(self, vectors, axes, scales):
        """Selected order, curve values, and the shipped curve must match the oracle exactly."""
        legacy_order, legacy_curve = _legacy_order_and_curve(vectors, axes, scales)
        opt_order, opt_curve = _optimized_order_and_curve(vectors, axes, scales)
        shipped_curve = _marginal_novelty_curve(vectors, axes, scales)
        oracle_curve = _LEGACY_marginal_novelty_curve(vectors, axes, scales)

        # Selected SEQUENCE + tie-breaking identical.
        self.assertEqual(opt_order, legacy_order)
        # Curve length == n - 1 (every non-seed point is visited exactly once), when n >= 2.
        if len(vectors) >= 2:
            self.assertEqual(len(shipped_curve), len(vectors) - 1)
        # Marginal-novelty values bit-identical across all four computations.
        self.assertEqual(opt_curve, legacy_curve)
        self.assertEqual(shipped_curve, oracle_curve)
        self.assertEqual(shipped_curve, legacy_curve)

    def _assert_saturation_equivalence(self, vectors, axes, scales):
        for params in (
            FrameworkSizingParams(),
            FrameworkSizingParams(knee_novelty_fraction=0.25, min_frames_for_curve=2),
            FrameworkSizingParams(knee_novelty_fraction=0.5, min_frames_for_curve=5),
        ):
            shipped = compute_saturation(vectors, axes, scales, params)
            # Recompute knee/saturation from the oracle curve directly.
            if len(vectors) < max(2, params.min_frames_for_curve):
                oracle = (0.0, 1.0)
            else:
                curve = _LEGACY_marginal_novelty_curve(vectors, axes, scales)
                if not curve or curve[0] <= 0.0:
                    oracle = (1.0, 0.0)
                else:
                    knee = params.knee_novelty_fraction * curve[0]
                    plateaued = sum(1 for d in curve if d <= knee)
                    sat = plateaued / len(curve)
                    oracle = (sat, max(0.0, 1.0 - sat))
            self.assertEqual(shipped, oracle)

    # -- 1. small hand-checkable -----------------------------------------------------------
    def test_small_hand_checkable_1d(self):
        axes = ["x"]
        scales = {"x": 1.0}
        vectors = [{"x": 0.0}, {"x": 10.0}, {"x": 1.0}, {"x": 9.0}]
        order, curve = _optimized_order_and_curve(vectors, axes, scales)
        # Seed=idx0 (x=0). Farthest is idx1 (x=10, dist 10). Then min-dist to {0,10}:
        #   idx2 x=1 -> 1 ; idx3 x=9 -> 1  => tie -> lowest index -> idx2 (dist 1).
        #   remaining idx3 -> min(|9-0|,|9-10|,|9-1|)=1.
        self.assertEqual(order, [0, 1, 2, 3])
        self.assertEqual(curve, [10.0, 1.0, 1.0])
        self._assert_full_equivalence(vectors, axes, scales)

    def test_small_hand_checkable_scaled(self):
        axes = ["x"]
        scales = {"x": 2.0}  # non-unit scale must divide identically in both paths
        vectors = [{"x": 0.0}, {"x": 4.0}, {"x": 2.0}]
        order, curve = _optimized_order_and_curve(vectors, axes, scales)
        self.assertEqual(order, [0, 1, 2])
        self.assertEqual(curve, [2.0, 1.0])  # (4-0)/2=2 ; then min(|2-0|,|2-4|)/2 = 1
        self._assert_full_equivalence(vectors, axes, scales)

    # -- 2. random seeded, multiple dims ---------------------------------------------------
    def test_random_seeded_multiple_dims(self):
        for seed in range(12):
            rng = random.Random(1000 + seed)
            dim = 1 + (seed % 4)
            n = 5 + (seed * 7) % 60
            axes = [f"a{k}" for k in range(dim)]
            vectors = [{ax: rng.uniform(-5, 5) for ax in axes} for _ in range(n)]
            scales = _axis_scales(vectors, axes)
            self._assert_full_equivalence(vectors, axes, scales)
            self._assert_saturation_equivalence(vectors, axes, scales)

    # -- 3. duplicates ---------------------------------------------------------------------
    def test_duplicates_and_all_identical(self):
        axes = ["x", "y"]
        scales = {"x": 1.0, "y": 1.0}
        identical = [{"x": 0.3, "y": -0.2} for _ in range(8)]
        self._assert_full_equivalence(identical, axes, scales)
        self._assert_saturation_equivalence(identical, axes, scales)
        # Mixed with duplicate clusters.
        mixed = [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 0.0},
                 {"x": 5.0, "y": 5.0}, {"x": 5.0, "y": 5.0},
                 {"x": 5.0, "y": 5.0}, {"x": 1.0, "y": 1.0}]
        self._assert_full_equivalence(mixed, axes, scales)
        self._assert_saturation_equivalence(mixed, axes, scales)

    # -- 4. ties (symmetric configuration -> lowest index wins) ----------------------------
    def test_symmetric_ties_lowest_index(self):
        axes = ["x"]
        scales = {"x": 1.0}
        # Symmetric around seed: idx1=+3, idx2=-3 equidistant; lowest index (1) must win first.
        vectors = [{"x": 0.0}, {"x": 3.0}, {"x": -3.0}, {"x": 3.0}, {"x": -3.0}]
        order, _ = _optimized_order_and_curve(vectors, axes, scales)
        self.assertEqual(order[1], 1)  # tie broken to lowest index
        self._assert_full_equivalence(vectors, axes, scales)
        self._assert_saturation_equivalence(vectors, axes, scales)

    # -- 5. missing-axis (axis absent in some vectors) -------------------------------------
    def test_missing_axis(self):
        axes = ["x", "y"]
        scales = {"x": 1.0, "y": 1.0}
        # Some frames lack 'y' entirely (mirrors single-atom frames lacking neighbor distance).
        vectors = [
            {"x": 0.0, "y": 0.0},
            {"x": 2.0},            # missing y
            {"x": 0.0, "y": 3.0},
            {"x": 4.0},            # missing y
            {"y": 1.0},            # missing x
            {"x": 1.0, "y": 1.0},
        ]
        self._assert_full_equivalence(vectors, axes, scales)
        self._assert_saturation_equivalence(vectors, axes, scales)

    def test_missing_axis_random(self):
        for seed in range(6):
            rng = random.Random(7000 + seed)
            axes = ["p", "q", "r"]
            n = 10 + seed * 5
            vectors = []
            for _ in range(n):
                v = {}
                for ax in axes:
                    if rng.random() > 0.3:  # ~30% of axes missing per frame
                        v[ax] = rng.uniform(0, 10)
                vectors.append(v)
            scales = _axis_scales(vectors, axes)
            self._assert_full_equivalence(vectors, axes, scales)
            self._assert_saturation_equivalence(vectors, axes, scales)

    # -- 6/7. single- and two-point degenerate cases --------------------------------------
    def test_single_point(self):
        axes = ["x"]
        scales = {"x": 1.0}
        self.assertEqual(_marginal_novelty_curve([{"x": 1.0}], axes, scales), [])
        self.assertEqual(_marginal_novelty_curve([], axes, scales), [])

    def test_two_points(self):
        axes = ["x", "y"]
        scales = {"x": 1.0, "y": 1.0}
        vectors = [{"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 4.0}]
        curve = _marginal_novelty_curve(vectors, axes, scales)
        self.assertEqual(curve, [5.0])  # 3-4-5 triangle
        self._assert_full_equivalence(vectors, axes, scales)

    # -- 8. multiple dims, larger n (still exact) ------------------------------------------
    def test_multiple_dims_larger(self):
        rng = random.Random(424242)
        axes = ["a", "b", "c", "d"]
        vectors = [{ax: rng.gauss(0, 1) for ax in axes} for _ in range(150)]
        scales = _axis_scales(vectors, axes)
        self._assert_full_equivalence(vectors, axes, scales)
        self._assert_saturation_equivalence(vectors, axes, scales)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
