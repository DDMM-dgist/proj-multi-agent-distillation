"""Framework V2 -- generic two-axis coverage + evidence-driven sizing (FE-027 P3).

Two coverage questions must be answered from evidence, generically, with no per-material rule and
no human-supplied ``target_count``:

  * §5/§8 TARGET_REGIME_COVERAGE -- for each discovered+role-assigned target regime, how saturated
    is it by the EXISTING pool, and (as an OUTPUT, never a human input) how much more acquisition
    is warranted? Saturation here is a real, reproducible curve computed from the generic
    descriptor vectors P1 already produced: a deterministic farthest-point traversal yields the
    marginal-novelty curve (each step's minimum distance to the already-chosen set), and the
    fraction of that curve that has flattened below a VERSIONED knee tolerance is the saturation.
    The only tunables are the fields of :class:`FrameworkSizingParams` -- explicit, versioned
    framework knobs, not per-material magic numbers.

  * §6 TEACHER_DISTRIBUTION_COVERAGE -- a SEPARATE, first-class axis: is the frozen Teacher's own
    training distribution KNOWN, PARTIALLY_KNOWN, or UNKNOWN relative to the target regimes? This
    is NOT the same question as target-regime coverage and must not be conflated with it. When the
    campaign carries no admissible evidence of the Teacher's training distribution the honest
    answer is UNKNOWN -- which is recorded, never fabricated, and is NOT an automatic failure.

Sizing (§8) turns the coverage evidence into per-regime acquisition counts: a saturated regime
warrants zero new acquisition; an unsaturated regime warrants a bounded batch (``batch_growth_factor``
x its current count), and the whole recommendation is clamped under an optional ``ComputeCeiling``.
``target_count = current_count + recommended_new`` is therefore an OUTPUT of this deterministic
analysis, exactly as the objective-to-evidence autonomy requirement demands.
"""
from __future__ import annotations

import dataclasses
import enum
import math
from typing import Optional

from framework_v2.acquisition.contracts import (
    ComputeCeiling,
    CoverageGapAnalysis,
    RelevanceRole,
)
from framework_v2.acquisition.coverage_gap import RegimeCoverageInput
from framework_v2.acquisition.generic_regions import build_frame_regime_classifier
from framework_v2.contracts import ContractBase, DomainRepresentation


@dataclasses.dataclass(frozen=True)
class FrameworkSizingParams:
    """Versioned framework knobs for the generic coverage + sizing analysis.

    These are portability-stable framework parameters (documented, versioned), NOT per-material
    magic numbers: the SAME values apply to every material. ``knee_novelty_fraction`` is the
    fraction of the initial marginal novelty below which a farthest-point step counts as
    "plateaued"; ``saturation_threshold`` is the plateaued-fraction at/above which a regime is
    saturated; ``min_frames_for_curve`` is the minimum members below which saturation is treated as
    0 (insufficient data is honestly non-saturating, never imputed); ``batch_growth_factor`` bounds
    how large an unsaturated regime's next acquisition batch may be relative to its current count.
    """
    knee_novelty_fraction: float = 0.1
    saturation_threshold: float = 0.8
    min_frames_for_curve: int = 3
    batch_growth_factor: float = 1.0
    version: str = "generic_sizing_v1"


# --------------------------------------------------------------------------------------------
# §6 -- Teacher training-distribution coverage (first-class, separate axis)
# --------------------------------------------------------------------------------------------
class TeacherCoverageStatus(str, __import__("enum").Enum):
    KNOWN = "KNOWN"
    PARTIALLY_KNOWN = "PARTIALLY_KNOWN"
    UNKNOWN = "UNKNOWN"


class TeacherDistributionCoverage(ContractBase):
    """First-class evidence of how well the frozen Teacher's TRAINING distribution is known
    relative to the target regimes -- deliberately distinct from TARGET_REGIME_COVERAGE.

    ``per_regime_overlap`` is populated only when concrete overlap evidence exists; for an UNKNOWN
    status it stays empty (the framework never fabricates an overlap number). An UNKNOWN status is
    a recorded finding, not an automatic failure -- downstream review decides what it means for the
    claim."""
    coverage_id: str
    status: TeacherCoverageStatus
    target_regime_model_sha256: str
    per_regime_overlap: dict[str, float] = {}
    rationale: str = ""
    evidence_refs: list[str] = []


def assess_teacher_distribution_coverage(
    *,
    coverage_id: str,
    target_regime_model_sha256: str,
    teacher_distribution_evidence: Optional[dict] = None,
) -> TeacherDistributionCoverage:
    """Deterministically classify Teacher-distribution coverage from whatever evidence exists.

    The generic metadata-free path has no evidence of the Teacher's training distribution, so the
    honest classification is UNKNOWN -- recorded, not fabricated, and not an auto-fail. When a
    campaign DOES supply per-regime overlap evidence, a complete map yields KNOWN and a partial map
    yields PARTIALLY_KNOWN."""
    if not teacher_distribution_evidence:
        return TeacherDistributionCoverage(
            coverage_id=coverage_id,
            status=TeacherCoverageStatus.UNKNOWN,
            target_regime_model_sha256=target_regime_model_sha256,
            rationale=("no admissible evidence of the frozen Teacher's training distribution is "
                       "available for this campaign; coverage is UNKNOWN (recorded, not fabricated)"))
    overlap = dict(teacher_distribution_evidence.get("per_regime_overlap", {}))
    complete = bool(teacher_distribution_evidence.get("complete", False))
    status = TeacherCoverageStatus.KNOWN if complete else TeacherCoverageStatus.PARTIALLY_KNOWN
    return TeacherDistributionCoverage(
        coverage_id=coverage_id,
        status=status,
        target_regime_model_sha256=target_regime_model_sha256,
        per_regime_overlap=overlap,
        rationale=teacher_distribution_evidence.get("rationale", ""),
        evidence_refs=list(teacher_distribution_evidence.get("evidence_refs", [])))


# --------------------------------------------------------------------------------------------
# §5/§8 -- target-regime coverage from the real farthest-point marginal-novelty curve
# --------------------------------------------------------------------------------------------
def _axis_scales(vectors: list[dict], axes: list[str]) -> dict[str, float]:
    """Per-axis pool-wide range, so axes with different units are comparable. Range 0 -> scale 1
    (the axis contributes no distance), never a divide-by-zero or an invented value."""
    scales: dict[str, float] = {}
    for ax in axes:
        vals = [v[ax] for v in vectors if ax in v]
        if not vals:
            scales[ax] = 1.0
            continue
        span = max(vals) - min(vals)
        scales[ax] = span if span > 0 else 1.0
    return scales


def _distance(a: dict, b: dict, axes: list[str], scales: dict[str, float]) -> float:
    total = 0.0
    for ax in axes:
        if ax in a and ax in b:
            d = (a[ax] - b[ax]) / scales[ax]
            total += d * d
    return math.sqrt(total)


def _marginal_novelty_curve(vectors: list[dict], axes: list[str], scales: dict[str, float]):
    """Deterministic farthest-point traversal. Start at index 0 (frames arrive in stable pool
    order); each subsequent step picks the point maximizing its min-distance to the chosen set,
    ties broken by lowest index. Returns the list of those chosen min-distances (marginal novelty),
    excluding the seed.

    Complexity is O(n^2 * d): an O(1) boolean chosen-mask replaces the earlier O(len(chosen)) list
    membership scans, and the min-distance-to-chosen-set is carried incrementally. Distances use the
    identical per-axis pure-Python ``_distance`` (same missing-axis handling, same float arithmetic
    and summation order), so the emitted sequence -- seed at index 0, argmax with lowest-index
    tie-breaking, curve values -- is bit-identical to the naive traversal it replaces."""
    n = len(vectors)
    if n < 2:
        return []
    chosen_mask = [False] * n
    chosen_mask[0] = True
    v0 = vectors[0]
    min_dist = [_distance(vectors[i], v0, axes, scales) for i in range(n)]
    curve: list[float] = []
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
        vb = vectors[best_i]
        for i in range(n):
            if not chosen_mask[i]:
                d = _distance(vectors[i], vb, axes, scales)
                if d < min_dist[i]:
                    min_dist[i] = d
    return curve


def compute_saturation(member_vectors: list[dict], axes: list[str],
                       scales: dict[str, float], params: FrameworkSizingParams):
    """Return (saturation, novelty_headroom) in [0, 1] from the marginal-novelty plateau fraction.

    Fewer than ``min_frames_for_curve`` members -> saturation 0 (insufficient data is honestly
    non-saturating). Otherwise saturation is the fraction of farthest-point steps whose marginal
    novelty has fallen to/below ``knee_novelty_fraction`` of the first (largest) step."""
    if len(member_vectors) < max(2, params.min_frames_for_curve):
        return 0.0, 1.0
    curve = _marginal_novelty_curve(member_vectors, axes, scales)
    if not curve or curve[0] <= 0.0:
        # A regime whose members are descriptor-space-identical is fully saturated (no novelty).
        return 1.0, 0.0
    knee = params.knee_novelty_fraction * curve[0]
    plateaued = sum(1 for d in curve if d <= knee)
    saturation = plateaued / len(curve)
    return saturation, max(0.0, 1.0 - saturation)


def compute_target_regime_coverage_inputs(
    pool, representation: DomainRepresentation, target_regime_model, *,
    params: FrameworkSizingParams, axes: Optional[list[str]] = None,
) -> list[RegimeCoverageInput]:
    """Build one ``RegimeCoverageInput`` per target regime from the real pool distribution.

    ``current_count`` is the number of pooled frames the executable P2 classifier assigns to the
    regime; ``saturation``/``novelty_headroom`` come from that regime's members' generic descriptor
    vectors via the farthest-point plateau curve. Nothing here supplies a target_count (sizing is a
    separate, later step)."""
    from framework_v2.acquisition.generic_representation import PRIMARY_CONTINUOUS_VARIABLES

    axes = list(axes if axes is not None else PRIMARY_CONTINUOUS_VARIABLES)
    classify = build_frame_regime_classifier(representation)
    features_by_id = {f.item_id: f.features for f in pool.frames}

    by_regime: dict[str, list[dict]] = {}
    for frame_id, feats in features_by_id.items():
        rid = classify(frame_id)
        if rid is not None:
            by_regime.setdefault(rid, []).append(feats)

    role_of = {r.regime_id: r.relevance_role for r in target_regime_model.regimes}
    all_vectors = list(features_by_id.values())
    scales = _axis_scales(all_vectors, axes)

    inputs: list[RegimeCoverageInput] = []
    for regime in target_regime_model.regimes:
        members = by_regime.get(regime.regime_id, [])
        saturation, headroom = compute_saturation(members, axes, scales, params)
        inputs.append(RegimeCoverageInput(
            regime_id=regime.regime_id,
            relevance_role=role_of[regime.regime_id],
            current_count=len(members),
            saturation=saturation,
            novelty_headroom=headroom,
            target_count=None))
    return inputs


# --------------------------------------------------------------------------------------------
# §8 -- evidence-driven acquisition sizing (target_count as OUTPUT, clamped by ceiling)
# --------------------------------------------------------------------------------------------
class AcquisitionSizing(ContractBase):
    """The deterministic sizing OUTPUT: recommended NEW-frame counts per regime + derived targets.

    ``recommended_new`` is the count the deterministic analysis warrants for each regime;
    ``target_count`` = current_count + recommended_new is the OUTPUT target (never a human input).
    ``ceiling_clamped`` records whether an optional compute ceiling reduced the recommendation."""
    sizing_id: str
    coverage_gap_sha256: str
    params_version: str
    recommended_new: dict[str, int]
    target_count: dict[str, int]
    ceiling_clamped: bool = False
    rationale: str = ""


def recommend_acquisition_sizing(
    coverage: CoverageGapAnalysis, *, params: FrameworkSizingParams,
    sizing_id: str, compute_ceiling: Optional[ComputeCeiling] = None,
) -> AcquisitionSizing:
    """Turn per-regime coverage evidence into per-regime acquisition counts.

    A saturated regime -> 0 new frames. An unsaturated CORE_TARGET / ADJACENT_PHYSICS /
    GENERATION_PATHWAY / BOUNDARY_GUARDRAIL regime -> a bounded batch of
    ``round(batch_growth_factor * current_count)`` (a re-evaluated exploration step, not a fixed
    target), floored at 1 when a gap exists but the regime is currently empty. OUT_OF_TARGET
    regimes -> 0. The total is clamped under ``compute_ceiling.max_candidates_generated`` if set,
    scaling all regimes down proportionally and deterministically."""
    recommended: dict[str, int] = {}
    for c in coverage.per_regime:
        if c.saturated or c.gap_score <= 0.0 or c.relevance_role == RelevanceRole.OUT_OF_TARGET_ACQUISITION:
            recommended[c.regime_id] = 0
            continue
        n = int(round(params.batch_growth_factor * c.current_count))
        recommended[c.regime_id] = max(1, n)

    ceiling_clamped = False
    if compute_ceiling is not None and compute_ceiling.max_candidates_generated is not None:
        cap = int(compute_ceiling.max_candidates_generated)
        total = sum(recommended.values())
        if total > cap and total > 0:
            ceiling_clamped = True
            scale = cap / total
            scaled = {k: int(math.floor(v * scale)) for k, v in recommended.items()}
            # Distribute any remaining budget deterministically (largest original first).
            remaining = cap - sum(scaled.values())
            for k in sorted(recommended, key=lambda k: (-recommended[k], k)):
                if remaining <= 0:
                    break
                if recommended[k] > 0:
                    scaled[k] += 1
                    remaining -= 1
            recommended = scaled

    current_of = {c.regime_id: c.current_count for c in coverage.per_regime}
    target_count = {k: current_of.get(k, 0) + v for k, v in recommended.items()}
    return AcquisitionSizing(
        sizing_id=sizing_id,
        coverage_gap_sha256=coverage.content_sha256(),
        params_version=params.version,
        recommended_new=recommended,
        target_count=target_count,
        ceiling_clamped=ceiling_clamped,
        rationale=("acquisition size derived from per-regime saturation/novelty (farthest-point "
                   "plateau) with a bounded growth step, clamped under any compute ceiling"))


# --------------------------------------------------------------------------------------------
# FE-028 -- existing-pool LABELING-population sizing (population size as OUTPUT, not a human input)
# --------------------------------------------------------------------------------------------
class LabelingPopulationSizingEvidence(ContractBase):
    """Deterministic sizing of the EXISTING-pool labeling population (FE-028).

    When a target regime is already covered by an eligible existing pool (no new-configuration
    gap), Stage-3 does not GENERATE frames -- it SELECTS a representative existing subset for
    canonical Teacher labeling. The size of that subset is an OUTPUT of this analysis, never a human
    input: a deterministic farthest-point traversal over the eligible descriptor vectors (the SAME
    raw-Euclidean FPS ``selection.farthest_point_selection`` uses, so sizing and selection agree)
    yields the marginal-novelty curve, and the knee of that curve -- the point of diminishing
    representation benefit -- sets the recommended population. When no defensible smaller subset
    exists the conservative fallback is the FULL eligible population. An optional
    ``target_labeled_population`` acts ONLY as an upper bound and an optional compute ceiling clamps
    the count; neither is required and the framework never asks a human 'how many frames?'.

    ``selected_positions`` are indices into the eligible descriptor array (in FPS order); the first
    ``recommended_population_size`` of them are the labeling population. The LLM never sets any of
    these numbers."""
    sizing_id: str
    coverage_gap_sha256: str
    params_version: str
    eligible_population_size: int
    protected_excluded_count: int
    recommended_population_size: int
    selected_positions: list[int]
    knee_index: Optional[int] = None
    first_marginal_novelty: float = 0.0
    knee_marginal_novelty: float = 0.0
    fallback_full_population: bool = False
    target_constraint_applied: bool = False
    ceiling_clamped: bool = False
    budget_insufficient: bool = False
    rationale: str = ""


def _fps_order_and_curve(vectors):
    """Numpy farthest-point order + marginal-novelty curve, matching
    ``selection.farthest_point_selection`` exactly (raw Euclidean, seed_index=0, ties -> lowest
    index). Returns (order, curve): ``order`` has length n (FPS visitation order); ``curve`` has
    length n-1, the min-distance-to-chosen-set (marginal novelty) at each step after the seed."""
    import numpy as np

    arr = np.asarray(vectors, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return [], []
    order = [0]
    min_dist = np.linalg.norm(arr - arr[0], axis=1)
    curve: list[float] = []
    while len(order) < n:
        nxt = int(np.argmax(min_dist))
        if nxt in order:
            remaining = [i for i in range(n) if i not in order]
            if not remaining:
                break
            nxt = remaining[0]
        curve.append(float(min_dist[nxt]))
        order.append(nxt)
        d = np.linalg.norm(arr - arr[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return order, curve


def recommend_labeling_population_sizing(
    eligible_vectors: list[list[float]], *, params: FrameworkSizingParams,
    sizing_id: str, coverage_gap_sha256: str, protected_excluded_count: int = 0,
    target_labeled_population: Optional[int] = None,
    max_teacher_label_calls: Optional[int] = None,
) -> LabelingPopulationSizingEvidence:
    """Deterministically size the existing-pool labeling population from descriptor evidence alone.

    ``eligible_vectors`` are the descriptor vectors of the protected-reference-EXCLUDED eligible
    pool, in stable pool order. The knee of the farthest-point marginal-novelty curve sets the
    justified subset size; absent a knee the conservative fallback is the full eligible population.
    ``target_labeled_population`` (optional) is only an upper bound; ``max_teacher_label_calls``
    (optional) clamps the count. The result is a typed evidence object -- never a human-supplied N,
    never an LLM-invented N."""
    n = len(eligible_vectors)
    if n <= 0:
        raise ValueError("recommend_labeling_population_sizing requires a non-empty eligible pool")

    order, curve = _fps_order_and_curve(eligible_vectors)
    knee_index: Optional[int] = None
    first_novelty = float(curve[0]) if curve else 0.0
    knee_novelty = 0.0
    fallback_full = False

    if n < max(2, params.min_frames_for_curve) or not curve:
        # Too few members to justify a smaller subset -> conservative full population.
        k = n
        fallback_full = True
    elif first_novelty <= 0.0:
        # Descriptor-space-identical eligible pool: one frame represents it fully.
        k = 1
        knee_index = 0
        knee_novelty = 0.0
    else:
        knee = params.knee_novelty_fraction * first_novelty
        j = next((idx for idx, dist in enumerate(curve) if dist <= knee), None)
        if j is None:
            # Novelty never plateaus -> no defensible smaller subset -> full population.
            k = n
            fallback_full = True
        else:
            knee_index = j
            knee_novelty = float(curve[j])
            k = j + 1  # seed + j high-novelty picks; order[j+1] onward is plateau

    target_applied = False
    if target_labeled_population is not None and k > int(target_labeled_population) > 0:
        k = int(target_labeled_population)
        target_applied = True

    ceiling_clamped = False
    budget_insufficient = False
    if max_teacher_label_calls is not None:
        cap = int(max_teacher_label_calls)
        if cap < 1:
            budget_insufficient = True
        elif k > cap:
            k = cap
            ceiling_clamped = True

    k = max(1, min(k, n))
    return LabelingPopulationSizingEvidence(
        sizing_id=sizing_id,
        coverage_gap_sha256=coverage_gap_sha256,
        params_version=params.version,
        eligible_population_size=n,
        protected_excluded_count=int(protected_excluded_count),
        recommended_population_size=k,
        selected_positions=[int(i) for i in order[:k]],
        knee_index=knee_index,
        first_marginal_novelty=first_novelty,
        knee_marginal_novelty=knee_novelty,
        fallback_full_population=fallback_full,
        target_constraint_applied=target_applied,
        ceiling_clamped=ceiling_clamped,
        budget_insufficient=budget_insufficient,
        rationale=("existing-pool labeling population sized from the farthest-point marginal-novelty "
                   "knee over the protected-excluded eligible descriptors; conservative fallback is "
                   "the full eligible population; optional target/ceiling applied only as bounds"))


__all__ = [
    "FrameworkSizingParams",
    "TeacherCoverageStatus",
    "TeacherDistributionCoverage",
    "assess_teacher_distribution_coverage",
    "compute_saturation",
    "compute_target_regime_coverage_inputs",
    "AcquisitionSizing",
    "recommend_acquisition_sizing",
    "LabelingPopulationSizingEvidence",
    "recommend_labeling_population_sizing",
]
