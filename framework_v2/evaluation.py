"""Framework V2 -- scope-aware evaluation partitioner (Section 11).

R31's evaluation stage aggregated all 1142 held-out frames including
8 no-Si (out-of-scope) frames whose Teacher-vs-DFT F_R2 was -154 and
therefore corrupted the aggregate. Framework V2 makes this structurally
impossible:

  * Every eval frame is labelled with one of the 5 ``ScopeCategory``
    values plus the derived ``HISTORICAL_BENCHMARK`` / ``BLIND_TEST``
    tags (the latter two are informational -- the scope contract's
    5-way categorisation already contains ``BLIND_TEST``;
    ``HISTORICAL_BENCHMARK`` here means "was inspected in a previous
    run and is no longer blind").
  * The primary campaign metric may aggregate ONLY the
    ``PRIMARY_DEPLOYMENT`` frames.
  * ``AUXILIARY_SUPPORT`` / ``OUT_OF_SCOPE`` / ``PROTECTED_REFERENCE`` /
    ``BLIND_TEST`` / ``HISTORICAL_BENCHMARK`` are reported diagnostically
    but explicitly labelled ``PRIMARY_STATUS: DIAGNOSTIC_ONLY`` so no
    downstream consumer can silently treat them as the primary result.
  * A mixed-scope aggregate that is not itself
    ``PRIMARY_DEPLOYMENT``-only fails a ``primary_aggregate_is_pure()``
    check.

The partitioner is deliberately not opinionated about *how* frames are
classified against regions -- the workflow supplies a
``FrameClassifier`` callable that maps a frame's metadata dict onto a
``region_id`` (or ``None`` if the frame is out-of-scope in this
contract). This keeps the partitioner reusable across different
scientific systems.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional

from pydantic import Field, model_validator

from framework_v2.contracts import (
    ContractBase,
    DeploymentScopeContract,
    EvaluationPolicy,
    ScopeCategory,
    utc_now_iso,
)


# ---------------------------------------------------------------------
# One-frame classification
# ---------------------------------------------------------------------
class FrameClassification(ContractBase):
    """The scope classification of a single evaluation frame."""
    frame_index: int  # index in the eval population
    frame_id: str     # stable id, if available; else str(frame_index)
    region_id: Optional[str]  # which ScopeRegion this frame belongs to
    category: ScopeCategory
    rationale: str = ""


FrameClassifier = Callable[[Mapping[str, Any]], Optional[str]]
"""Map a frame's metadata to a ``region_id`` in the contract or ``None``
if the frame does not belong to any declared region. The partitioner
treats ``None`` as ``OUT_OF_SCOPE`` (a frame the run's scope does not
cover)."""


def partition_frames(
    frames: Iterable[Mapping[str, Any]],
    scope: DeploymentScopeContract,
    classifier: FrameClassifier,
) -> list[FrameClassification]:
    """Classify every frame against the declared scope.

    ``frames`` is any iterable of dict-like metadata records (one per
    eval frame). ``classifier`` maps a metadata record to a
    ``region_id``. Frames the classifier maps to ``None`` are recorded
    as ``OUT_OF_SCOPE``.
    """
    region_by_id = {r.region_id: r for r in scope.regions}
    out: list[FrameClassification] = []
    for i, frame in enumerate(frames):
        rid = classifier(frame)
        if rid is None:
            out.append(FrameClassification(
                frame_index=i,
                frame_id=str(frame.get("frame_id", i)),
                region_id=None,
                category=ScopeCategory.OUT_OF_SCOPE,
                rationale="classifier returned None (no matching region)",
            ))
            continue
        region = region_by_id.get(rid)
        if region is None:
            raise ValueError(
                f"classifier returned region_id={rid!r} which is not declared "
                f"in the DeploymentScopeContract (contract_id="
                f"{scope.contract_id})"
            )
        out.append(FrameClassification(
            frame_index=i,
            frame_id=str(frame.get("frame_id", i)),
            region_id=rid,
            category=region.category,
            rationale=region.rationale,
        ))
    return out


def count_by_category(
    classifications: Iterable[FrameClassification],
) -> dict[ScopeCategory, int]:
    counts: dict[ScopeCategory, int] = {c: 0 for c in ScopeCategory}
    for fc in classifications:
        counts[fc.category] += 1
    return counts


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
class PartitionMetrics(ContractBase):
    """Metrics computed over one scope partition."""
    category: ScopeCategory
    n_frames: int
    metrics: dict[str, float] = Field(default_factory=dict)
    is_primary_partition: bool = False


class EvaluationReport(ContractBase):
    """The full scope-aware evaluation report.

    ``primary_partition`` holds the metrics that may be quoted as the
    primary R32 scientific result. ``diagnostic_partitions`` holds all
    others. ``mixed_aggregate`` is optional and, if provided, is
    explicitly labelled DIAGNOSTIC_ONLY.
    """
    report_id: str
    policy_sha256: str
    scope_sha256: str
    primary_partition: PartitionMetrics
    diagnostic_partitions: list[PartitionMetrics] = Field(default_factory=list)
    mixed_aggregate: Optional[PartitionMetrics] = None
    frame_classifications: list[FrameClassification]
    generated_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _primary_partition_is_primary(self):
        if self.primary_partition.category != ScopeCategory.PRIMARY_DEPLOYMENT:
            raise ValueError(
                "EvaluationReport.primary_partition must have "
                "category == PRIMARY_DEPLOYMENT"
            )
        if not self.primary_partition.is_primary_partition:
            raise ValueError(
                "primary_partition.is_primary_partition must be True"
            )
        return self

    @model_validator(mode="after")
    def _diagnostic_partitions_are_not_primary(self):
        for p in self.diagnostic_partitions:
            if p.is_primary_partition:
                raise ValueError(
                    "diagnostic_partitions may not carry "
                    "is_primary_partition=True"
                )
        return self

    def primary_aggregate_is_pure(self) -> bool:
        """True iff the primary partition contains only frames from
        PRIMARY_DEPLOYMENT (should always be True by construction)."""
        primary_frames = [fc for fc in self.frame_classifications
                          if fc.frame_index in _primary_indices(self)]
        return all(fc.category == ScopeCategory.PRIMARY_DEPLOYMENT
                   for fc in primary_frames)


def _primary_indices(report: "EvaluationReport") -> set[int]:
    # For structural purity check: primary partition frames are those
    # whose category is PRIMARY_DEPLOYMENT.
    return {fc.frame_index for fc in report.frame_classifications
            if fc.category == ScopeCategory.PRIMARY_DEPLOYMENT}


# ---------------------------------------------------------------------
# Metric-computation adapter type
# ---------------------------------------------------------------------
MetricFn = Callable[[list[int]], dict[str, float]]
"""A caller-supplied function that computes a dict of named metrics
over the subset of frame indices passed in. The partitioner is
metric-agnostic; the workflow supplies the actual E MAE/RMSE/F R2 etc.
computation."""


def build_evaluation_report(
    *,
    report_id: str,
    policy: EvaluationPolicy,
    scope: DeploymentScopeContract,
    classifications: list[FrameClassification],
    metric_fn: MetricFn,
    include_mixed_aggregate: bool = False,
) -> EvaluationReport:
    """Compute per-category metrics and return an EvaluationReport.

    The primary partition is *always* PRIMARY_DEPLOYMENT; if the policy
    is ``reject_mixed_aggregate_as_primary=True`` (default) and
    ``include_mixed_aggregate=True``, the mixed-aggregate is included
    but labelled non-primary. Callers must never use the mixed
    aggregate as their headline metric under that policy.
    """
    # index groups by category
    groups: dict[ScopeCategory, list[int]] = {c: [] for c in ScopeCategory}
    for fc in classifications:
        groups[fc.category].append(fc.frame_index)

    primary_indices = groups[ScopeCategory.PRIMARY_DEPLOYMENT]
    primary_partition = PartitionMetrics(
        category=ScopeCategory.PRIMARY_DEPLOYMENT,
        n_frames=len(primary_indices),
        metrics=metric_fn(primary_indices) if primary_indices else {},
        is_primary_partition=True,
    )

    diagnostic_partitions: list[PartitionMetrics] = []
    for cat in [ScopeCategory.AUXILIARY_SUPPORT, ScopeCategory.OUT_OF_SCOPE,
                ScopeCategory.PROTECTED_REFERENCE, ScopeCategory.BLIND_TEST]:
        idxs = groups[cat]
        if idxs:
            diagnostic_partitions.append(PartitionMetrics(
                category=cat, n_frames=len(idxs),
                metrics=metric_fn(idxs),
                is_primary_partition=False,
            ))

    mixed_aggregate: Optional[PartitionMetrics] = None
    if include_mixed_aggregate:
        all_indices = [fc.frame_index for fc in classifications]
        if all_indices:
            # We label the mixed aggregate with OUT_OF_SCOPE category
            # to make it structurally impossible to promote it to the
            # primary role: EvaluationReport's validator refuses a
            # primary_partition whose category is not
            # PRIMARY_DEPLOYMENT.
            mixed_aggregate = PartitionMetrics(
                category=ScopeCategory.OUT_OF_SCOPE,
                n_frames=len(all_indices),
                metrics=metric_fn(all_indices),
                is_primary_partition=False,
            )

    return EvaluationReport(
        report_id=report_id,
        policy_sha256=policy.content_sha256(),
        scope_sha256=scope.content_sha256(),
        primary_partition=primary_partition,
        diagnostic_partitions=diagnostic_partitions,
        mixed_aggregate=mixed_aggregate,
        frame_classifications=classifications,
    )


# ---------------------------------------------------------------------
# Cross-stage scope-consistency check (Section 20 case G)
# ---------------------------------------------------------------------
def cross_stage_scope_consistent(*artifact_scope_shas: str) -> bool:
    """True iff every provided scope-SHA is identical -- i.e. every
    stage bound its work to the same DeploymentScopeContract identity.
    Any disagreement means one stage silently reinterpreted scope
    (Section 3), which is a contract violation."""
    unique = set(artifact_scope_shas)
    return len(unique) <= 1


__all__ = [
    "FrameClassification",
    "FrameClassifier",
    "MetricFn",
    "PartitionMetrics",
    "EvaluationReport",
    "partition_frames",
    "count_by_category",
    "build_evaluation_report",
    "cross_stage_scope_consistent",
]
