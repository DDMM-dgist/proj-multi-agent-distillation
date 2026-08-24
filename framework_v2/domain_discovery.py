"""Framework V2 -- generic domain-regime discovery (Section 5).

R31's regime structure (the SiO2-x "CORE / ANCHOR / BOUNDARY" partition and
its fixed counts) was hand-authored in prose and YAML. There was no generic,
evidence-driven capability that could look at an actual source universe and
*discover* what regimes exist, so a wrong or stale regime map could not be
caught by the framework. Section 5 requires a generic discoverer that:

  * consumes a source universe expressed as items with named descriptor
    variables (continuous and/or categorical),
  * discovers regimes deterministically from the data distribution -- NOT
    from any element-, composition-, or campaign-specific rule,
  * emits a typed ``DomainRepresentation`` contract bound by content-SHA to
    the ``DeploymentScopeContract`` it was discovered against,
  * classifies each discovered regime's overlap with the declared scope
    regions via a caller-supplied deterministic membership evaluator (the
    caller owns its domain semantics; this module owns only the generic
    discovery mechanics).

Nothing here hard-codes a regime count, an element, a density, a coordination
number, or the labels CORE/ANCHOR/BOUNDARY. Every tunable (how finely a
continuous axis is split, how many items make a regime "real") is supplied by
a ``DiscoveryConfig`` the caller sources from evidence/policy; the module
invents no numbers.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Optional, Sequence

from framework_v2.contracts import (
    DeploymentScopeContract,
    DomainRegime,
    DomainRepresentation,
    ScopeCategory,
)


@dataclasses.dataclass(frozen=True)
class SourceItem:
    """One member of the source universe.

    ``features`` maps a descriptor-variable name to its value. Continuous
    variables carry a real number; categorical variables carry any hashable
    label (typically a string). A variable absent from a given item is
    treated as missing and the item is excluded from that variable's
    partitioning (recorded as an evidence gap), never imputed.
    """
    item_id: str
    features: dict


@dataclasses.dataclass(frozen=True)
class DiscoveryConfig:
    """All discovery tunables, supplied by the caller (never defaulted to a
    magic number inside the module).

    ``continuous_variables`` / ``categorical_variables`` name the descriptor
    axes to partition on. ``max_intervals_per_continuous_axis`` caps how many
    contiguous intervals a single continuous axis may be split into.
    ``gap_significance`` is the minimum normalized gap (relative to the axis
    range) at which a continuous axis is split. ``min_items_per_regime`` is
    the count below which an occupied cell is reported as a *sparse* regime
    (still emitted -- absence is evidence -- but flagged so downstream code
    can refuse to treat it as representative). ``hierarchy`` optionally
    declares an ordered nesting of variables; when set, the representation's
    ``kind`` is ``hierarchical``.
    """
    continuous_variables: Sequence[str] = ()
    categorical_variables: Sequence[str] = ()
    max_intervals_per_continuous_axis: int = 4
    gap_significance: float = 0.15
    min_items_per_regime: int = 1
    hierarchy: Sequence[str] = ()

    def __post_init__(self):
        if not self.continuous_variables and not self.categorical_variables:
            raise ValueError(
                "DiscoveryConfig requires at least one continuous or "
                "categorical variable to partition on"
            )
        if self.max_intervals_per_continuous_axis < 1:
            raise ValueError("max_intervals_per_continuous_axis must be >= 1")
        if not 0.0 < self.gap_significance <= 1.0:
            raise ValueError("gap_significance must be in (0, 1]")
        if self.min_items_per_regime < 1:
            raise ValueError("min_items_per_regime must be >= 1")


@dataclasses.dataclass(frozen=True)
class _Interval:
    """A half-open interval [lo, hi) (the last interval of an axis is closed)."""
    lo: float
    hi: float
    closed_hi: bool

    def contains(self, value: float) -> bool:
        if value < self.lo:
            return False
        if self.closed_hi:
            return value <= self.hi
        return value < self.hi

    def label(self, var: str) -> str:
        rb = "]" if self.closed_hi else ")"
        return f"{var} in [{_fmt(self.lo)}, {_fmt(self.hi)}{rb}"


def _fmt(x: float) -> str:
    # Compact, deterministic float rendering for membership-rule strings.
    return repr(round(float(x), 6))


def _discover_intervals(values: Sequence[float], config: DiscoveryConfig) -> list[_Interval]:
    """Deterministic 1-D gap splitting.

    Sort the unique values; walk the sorted gaps largest-first, splitting at
    a gap iff it is at least ``gap_significance`` of the axis range, until at
    most ``max_intervals_per_continuous_axis`` intervals exist. Ties break by
    the lower boundary so the result is fully deterministic.
    """
    uniq = sorted(set(values))
    if not uniq:
        return []
    lo, hi = uniq[0], uniq[-1]
    span = hi - lo
    if span <= 0 or len(uniq) == 1:
        return [_Interval(lo, hi, True)]
    gaps = []
    for i in range(len(uniq) - 1):
        gaps.append((uniq[i + 1] - uniq[i], uniq[i], uniq[i + 1]))
    # Candidate split points: gaps that clear the significance threshold,
    # ordered by descending gap size then ascending left boundary.
    significant = [g for g in gaps if g[0] / span >= config.gap_significance]
    significant.sort(key=lambda g: (-g[0], g[1]))
    n_splits = min(len(significant), config.max_intervals_per_continuous_axis - 1)
    split_after = sorted(significant[i][1] for i in range(n_splits))
    intervals: list[_Interval] = []
    cursor = lo
    for boundary_left in split_after:
        # boundary is the value just below the gap; interval is [cursor, next_lo)
        next_lo = next(g[2] for g in gaps if g[1] == boundary_left)
        intervals.append(_Interval(cursor, next_lo, False))
        cursor = next_lo
    intervals.append(_Interval(cursor, hi, True))
    return intervals


def _infer_kind(config: DiscoveryConfig) -> str:
    if config.hierarchy:
        return "hierarchical"
    has_cont = bool(config.continuous_variables)
    has_cat = bool(config.categorical_variables)
    if has_cont and has_cat:
        return "hybrid"
    if has_cat:
        return "categorical"
    return "continuous"


def discover_domain(
    *,
    representation_id: str,
    descriptor: str,
    items: Sequence[SourceItem],
    config: DiscoveryConfig,
    scope_contract: DeploymentScopeContract,
    region_classifier: Optional[Callable[[SourceItem], Iterable[str]]] = None,
    evidence_ref: str = "",
) -> DomainRepresentation:
    """Discover regimes from ``items`` and emit a ``DomainRepresentation``.

    ``region_classifier`` (optional) maps an item to the set of
    ``ScopeRegion.region_id`` values it satisfies -- this is the caller's
    deterministic domain-membership evaluator. A discovered regime's
    ``within_scope_categories`` is the union of the scope categories of the
    regions its member items fall into; a regime whose items match no
    declared region gets an empty list, which is itself an auditable finding
    (the source universe contains data outside every declared scope region).

    The returned representation is bound by ``linked_scope_contract_sha256``
    to ``scope_contract.content_sha256()`` so downstream contracts cannot
    silently pair it with a different scope.
    """
    if not items:
        raise ValueError("discover_domain requires a non-empty source universe")

    # --- partition each continuous axis --------------------------------------
    axis_intervals: dict[str, list[_Interval]] = {}
    evidence_gaps: dict[str, int] = {}
    for var in config.continuous_variables:
        vals = []
        missing = 0
        for it in items:
            v = it.features.get(var)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
            else:
                missing += 1
        if missing:
            evidence_gaps[var] = missing
        axis_intervals[var] = _discover_intervals(vals, config)

    # --- assign every item to a cell (a tuple of per-variable labels) --------
    def cell_of(it: SourceItem) -> Optional[tuple]:
        key: list = []
        for var in config.continuous_variables:
            v = it.features.get(var)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return None
            iv = next((iv for iv in axis_intervals[var] if iv.contains(float(v))), None)
            if iv is None:
                return None
            key.append((var, "continuous", iv))
        for var in config.categorical_variables:
            v = it.features.get(var)
            if v is None:
                return None
            key.append((var, "categorical", v))
        return tuple(key)

    cells: dict[tuple, list[SourceItem]] = {}
    unresolved = 0
    for it in items:
        c = cell_of(it)
        if c is None:
            unresolved += 1
            continue
        cells.setdefault(c, []).append(it)

    # --- build a DomainRegime per occupied cell ------------------------------
    regimes: list[DomainRegime] = []
    sparse_regime_ids: list[str] = []
    for idx, (cell, members) in enumerate(sorted(cells.items(), key=_cell_sort_key)):
        rule_parts: list[str] = []
        label_parts: list[str] = []
        for (var, vtype, payload) in cell:
            if vtype == "continuous":
                rule_parts.append(payload.label(var))
                label_parts.append(f"{var}:{_fmt(payload.lo)}-{_fmt(payload.hi)}")
            else:
                rule_parts.append(f"{var} == {payload!r}")
                label_parts.append(f"{var}={payload}")
        # scope-category overlap from member items
        categories: set[ScopeCategory] = set()
        if region_classifier is not None:
            for it in members:
                for region_id in region_classifier(it):
                    region = scope_contract.region(region_id)
                    if region is not None:
                        categories.add(region.category)
        regime_id = f"{representation_id}-r{idx:03d}"
        if len(members) < config.min_items_per_regime:
            sparse_regime_ids.append(regime_id)
        member_refs = [it.item_id for it in members]
        regimes.append(DomainRegime(
            regime_id=regime_id,
            label="; ".join(label_parts) if label_parts else regime_id,
            membership_rule="; ".join(rule_parts),
            membership_evidence_refs=(
                ([evidence_ref] if evidence_ref else []) + member_refs
            ),
            within_scope_categories=sorted(categories, key=lambda c: c.value),
        ))

    sensitivity_report = {
        "n_items": len(items),
        "n_regimes": len(regimes),
        "n_unresolved_items": unresolved,
        "sparse_regime_ids": sparse_regime_ids,
        "min_items_per_regime": config.min_items_per_regime,
        "continuous_axis_intervals": {
            var: [iv.label(var) for iv in ivs] for var, ivs in axis_intervals.items()
        },
        "per_variable_missing_counts": evidence_gaps,
        "config": {
            "continuous_variables": list(config.continuous_variables),
            "categorical_variables": list(config.categorical_variables),
            "max_intervals_per_continuous_axis": config.max_intervals_per_continuous_axis,
            "gap_significance": config.gap_significance,
            "hierarchy": list(config.hierarchy),
        },
    }

    return DomainRepresentation(
        representation_id=representation_id,
        kind=_infer_kind(config),
        descriptor=descriptor,
        regimes=regimes,
        sensitivity_report=sensitivity_report,
        linked_scope_contract_sha256=scope_contract.content_sha256(),
    )


def _cell_sort_key(item):
    cell = item[0]
    key = []
    for (var, vtype, payload) in cell:
        if vtype == "continuous":
            key.append((var, payload.lo, payload.hi))
        else:
            key.append((var, str(payload)))
    return key


def primary_regimes_present(
    representation: DomainRepresentation,
) -> list[DomainRegime]:
    """Regimes overlapping PRIMARY_DEPLOYMENT scope. Convenience used by the
    partition-representativeness validator (Section 8)."""
    return [
        r for r in representation.regimes
        if ScopeCategory.PRIMARY_DEPLOYMENT in r.within_scope_categories
    ]


__all__ = [
    "SourceItem",
    "DiscoveryConfig",
    "discover_domain",
    "primary_regimes_present",
]
