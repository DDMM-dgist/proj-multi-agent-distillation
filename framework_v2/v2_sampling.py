"""V2 sampler and stopping-policy interfaces.

Samplers answer WHICH structures to add.  Stopping policies answer WHETHER a
region still needs data.  Keeping these independent lets later experiments
compare Random/FPS/DIRECT-like selectors under the same closure criterion.

DIRECT-like here is a *structural-stratified diversity* selector, not a
reproduction of published DIRECT.  It guarantees per-region coverage when the
budget permits and fills residual budget by global structural farthest-point
diversity (relative to the already-selected set) rather than by any region-ID
ordering.  Under-budget requests return a typed unresolved status instead of
silently favouring whichever region sorts first.
"""
from __future__ import annotations

import random
from enum import Enum
from typing import Any, Mapping

import numpy as np
from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase
from framework_v2.structural_representation import StructuralRepresentation


class SamplerKind(str, Enum):
    RANDOM = "RANDOM"
    FPS = "FPS"
    DIRECT_LIKE = "DIRECT_LIKE"
    UNCERTAINTY = "UNCERTAINTY"
    UNCERTAINTY_DIVERSITY = "UNCERTAINTY_DIVERSITY"


class SelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    SELECTION_BUDGET_INSUFFICIENT = "SELECTION_BUDGET_INSUFFICIENT"


class RegionClosureState(str, Enum):
    OPEN = "OPEN"
    LEARNING = "LEARNING"
    RECOVER = "RECOVER"
    CLOSED = "CLOSED"
    EVIDENCE_NOT_EVALUATED = "EVIDENCE_NOT_EVALUATED"
    HUMAN_SCIENTIFIC_INPUT_REQUIRED = "HUMAN_SCIENTIFIC_INPUT_REQUIRED"


class CriterionComparator(str, Enum):
    LE = "LE"
    LT = "LT"
    GE = "GE"
    GT = "GT"
    EQ = "EQ"


class CriterionBindingStatus(str, Enum):
    BOUND = "BOUND"
    UNBOUND = "UNBOUND"


class CriterionRole(str, Enum):
    """WHY an observable/criterion is evaluated (H12 observable-role axis).

    ``SCIENTIFIC_REQUIRED``   -- a scientific success criterion (the target property).
    ``OPERATIONAL_REQUIRED``  -- an operational model-fidelity criterion (e.g.
                                 Student-vs-Teacher energy/force RMSE).
    ``NUMERICAL_GUARD``       -- a numerical/physical-stability guard (e.g. NVE
                                 energy drift); a gate, but never a scientific target.
    ``EVIDENCE_ONLY``         -- a non-gating diagnostic observable.

    ``NUMERICAL_GUARD`` is treated as a closure gate exactly like the other
    non-evidence roles (it can block closure); it is distinguished only so that a
    stability guard is never mistaken for a scientific target property.
    """

    SCIENTIFIC_REQUIRED = "SCIENTIFIC_REQUIRED"
    OPERATIONAL_REQUIRED = "OPERATIONAL_REQUIRED"
    NUMERICAL_GUARD = "NUMERICAL_GUARD"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"


class EvidenceStatus(str, Enum):
    EVALUATED = "EVALUATED"
    NOT_EVALUATED = "NOT_EVALUATED"


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------
class SamplerRequest(ContractBase):
    sampler: SamplerKind
    candidate_ids: list[str]
    n_select: int
    region_by_candidate: dict[str, str] = Field(default_factory=dict)
    deficient_region_ids: list[str] = Field(default_factory=list)
    protected_candidate_ids: list[str] = Field(default_factory=list)
    seed: int = 0
    uncertainty_by_candidate: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid(self):
        if self.n_select <= 0:
            raise ValueError("SamplerRequest n_select must be positive")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        protected = set(self.protected_candidate_ids)
        if protected & set(self.candidate_ids) and self.parameters.get("allow_protected"):
            raise ValueError("protected candidate override is not allowed in V2")
        return self


class SamplerResult(ContractBase):
    sampler: SamplerKind
    selected_ids: list[str]
    request_sha256: str
    rationale: str
    status: SelectionStatus = SelectionStatus.SELECTED
    unresolved_reason: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)


def _eligible(request: SamplerRequest) -> list[str]:
    ids = [cid for cid in request.candidate_ids if cid not in set(request.protected_candidate_ids)]
    if request.deficient_region_ids:
        allowed = set(request.deficient_region_ids)
        ids = [cid for cid in ids if request.region_by_candidate.get(cid) in allowed]
    return ids


def _group_by_region(ids: list[str], region_by_candidate: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for cid in ids:
        region = region_by_candidate.get(cid)
        if region is None:
            raise ValueError(f"candidate {cid!r} lacks region assignment")
        grouped.setdefault(region, []).append(cid)
    return grouped


def _fps(ids: list[str], representation: StructuralRepresentation) -> list[str]:
    matrix_by_id = {
        sid: np.asarray(row, dtype=float)
        for sid, row in zip(representation.structure_ids, representation.matrix)
    }
    missing = [cid for cid in ids if cid not in matrix_by_id]
    if missing:
        raise ValueError("representation is missing candidate ids: " + ", ".join(missing))
    chosen: list[str] = []
    remaining = list(ids)
    if not remaining:
        return []
    chosen.append(remaining.pop(0))
    while remaining:
        def min_dist(cid: str) -> float:
            return min(float(np.linalg.norm(matrix_by_id[cid] - matrix_by_id[c])) for c in chosen)
        pick = max(remaining, key=lambda cid: (min_dist(cid), -ids.index(cid)))
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def _fps_fill(
    selected: list[str],
    pool: list[str],
    representation: StructuralRepresentation,
    k: int,
) -> list[str]:
    """Greedily add k pool ids maximising min distance to the selected set."""
    matrix_by_id = {
        sid: np.asarray(row, dtype=float)
        for sid, row in zip(representation.structure_ids, representation.matrix)
    }
    missing = [cid for cid in (selected + pool) if cid not in matrix_by_id]
    if missing:
        raise ValueError("representation is missing candidate ids: " + ", ".join(missing))
    chosen: list[str] = []
    anchor = list(selected)
    remaining = list(pool)
    while remaining and len(chosen) < k:
        def min_dist(cid: str) -> float:
            ref = anchor + chosen
            if not ref:
                return 0.0
            return min(float(np.linalg.norm(matrix_by_id[cid] - matrix_by_id[c])) for c in ref)
        pick = max(remaining, key=lambda cid: (min_dist(cid), -pool.index(cid)))
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def _direct_like_select(
    eligible: list[str],
    region_by_candidate: Mapping[str, str],
    n_select: int,
    representation: StructuralRepresentation | None,
) -> tuple[list[str] | None, dict[str, Any]]:
    """Return (selected_or_None, provenance).

    ``None`` means SELECTION_BUDGET_INSUFFICIENT (budget below the number of
    eligible regions).  Coverage is guaranteed first (one per region); residual
    budget is filled by global structural FPS when a representation is present,
    otherwise by stable round-robin over regions.
    """
    grouped = _group_by_region(eligible, region_by_candidate)
    region_order = list(grouped.keys())  # first-appearance order, not sorted IDs
    if n_select < len(region_order):
        return None, {"n_regions": len(region_order), "n_select": n_select}

    # step 1: one representative per region (stable-first within region)
    selected = [grouped[r][0] for r in region_order]
    remaining = n_select - len(selected)
    provenance: dict[str, Any] = {
        "region_order": region_order,
        "per_region_representative": dict(zip(region_order, selected)),
        "residual_budget": remaining,
    }

    if remaining > 0:
        pool = [c for c in eligible if c not in set(selected)]
        if representation is not None:
            fill = _fps_fill(selected, pool, representation, remaining)
            provenance["residual_fill"] = "global_fps_diversity"
        else:
            fill = _round_robin_fill(grouped, region_order, set(selected), remaining)
            provenance["residual_fill"] = "stable_round_robin_no_diversity_claim"
        selected.extend(fill)

    # deterministic output order = original eligible order
    ordered = sorted(set(selected), key=lambda c: eligible.index(c))
    return ordered, provenance


def _round_robin_fill(
    grouped: Mapping[str, list[str]],
    region_order: list[str],
    already: set[str],
    k: int,
) -> list[str]:
    fill: list[str] = []
    taken = set(already)
    while len(fill) < k:
        progressed = False
        for region in region_order:
            for cid in grouped[region]:
                if cid not in taken:
                    fill.append(cid)
                    taken.add(cid)
                    progressed = True
                    break
            if len(fill) >= k:
                break
        if not progressed:
            break
    return fill


def sample_candidates(
    request: SamplerRequest,
    *,
    representation: StructuralRepresentation | None = None,
) -> SamplerResult:
    ids = _eligible(request)
    protected_excluded = len(request.candidate_ids) - len(
        [c for c in request.candidate_ids if c not in set(request.protected_candidate_ids)]
    )

    if request.sampler == SamplerKind.DIRECT_LIKE and request.region_by_candidate:
        if request.n_select > len(ids):
            raise ValueError("not enough eligible candidates for DIRECT_LIKE budget")
        selected, prov = _direct_like_select(
            ids, request.region_by_candidate, request.n_select, representation
        )
        prov = {**prov, "protected_excluded": protected_excluded,
                "representation_sha256": representation.content_sha256() if representation else None}
        if selected is None:
            return SamplerResult(
                sampler=request.sampler, selected_ids=[],
                request_sha256=request.content_sha256(),
                rationale="DIRECT-like structural-stratified selection under-budget",
                status=SelectionStatus.SELECTION_BUDGET_INSUFFICIENT,
                unresolved_reason=(
                    f"n_select={request.n_select} < number of eligible regions="
                    f"{prov['n_regions']}; cannot guarantee region coverage"
                ),
                provenance=prov,
            )
        return SamplerResult(
            sampler=request.sampler, selected_ids=selected[: request.n_select],
            request_sha256=request.content_sha256(),
            rationale="DIRECT-like: per-region coverage then structural-diversity residual",
            provenance=prov,
        )

    if len(ids) < request.n_select:
        raise ValueError("not enough eligible candidates after region/protected filtering")

    if request.sampler == SamplerKind.RANDOM:
        rng = random.Random(request.seed)
        ordered = list(ids)
        rng.shuffle(ordered)
    elif request.sampler == SamplerKind.FPS:
        if representation is None:
            raise ValueError("FPS sampler requires structural representation")
        ordered = _fps(ids, representation)
    elif request.sampler == SamplerKind.DIRECT_LIKE:
        # no region metadata -> fall back to pure structural diversity
        if representation is None:
            raise ValueError("DIRECT_LIKE sampler requires regions or representation")
        ordered = _fps(ids, representation)
    elif request.sampler == SamplerKind.UNCERTAINTY:
        ordered = sorted(
            ids,
            key=lambda cid: (-float(request.uncertainty_by_candidate.get(cid, 0.0)), ids.index(cid)),
        )
    elif request.sampler == SamplerKind.UNCERTAINTY_DIVERSITY:
        if representation is None:
            raise ValueError("UNCERTAINTY_DIVERSITY requires structural representation")
        top = sorted(
            ids,
            key=lambda cid: (-float(request.uncertainty_by_candidate.get(cid, 0.0)), ids.index(cid)),
        )[: max(request.n_select * 3, request.n_select)]
        ordered = _fps(top, representation)
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(f"unsupported sampler {request.sampler}")

    return SamplerResult(
        sampler=request.sampler,
        selected_ids=ordered[: request.n_select],
        request_sha256=request.content_sha256(),
        rationale="V2 sampler selected eligible training-side candidates only",
        provenance={"protected_excluded": protected_excluded},
    )


# --------------------------------------------------------------------------
# stopping / closure
# --------------------------------------------------------------------------
def criterion_passes(measured, value, comparator: CriterionComparator) -> bool:
    if comparator in (CriterionComparator.EQ,):
        return measured == value
    m = float(measured)
    v = float(value)
    if comparator == CriterionComparator.LE:
        return m <= v
    if comparator == CriterionComparator.LT:
        return m < v
    if comparator == CriterionComparator.GE:
        return m >= v
    if comparator == CriterionComparator.GT:
        return m > v
    raise ValueError(f"unsupported comparator {comparator}")  # pragma: no cover


class SignalCriterion(ContractBase):
    signal: str
    role: CriterionRole
    binding_status: CriterionBindingStatus
    comparator: CriterionComparator | None = None
    value: float | int | str | None = None
    units: str = ""
    provenance: list[str] = Field(default_factory=list)
    unbound_reason: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.role == CriterionRole.EVIDENCE_ONLY:
            return self
        if self.binding_status == CriterionBindingStatus.BOUND:
            if self.comparator is None or self.value is None or not self.provenance:
                raise ValueError("BOUND required criterion needs comparator, value, provenance")
        if self.binding_status == CriterionBindingStatus.UNBOUND:
            if self.comparator is not None or self.value is not None:
                raise ValueError("UNBOUND criterion cannot carry comparator/value")
        return self


class SignalCriterionEvaluation(ContractBase):
    signal: str
    criterion_binding_status: CriterionBindingStatus
    evidence_status: EvidenceStatus
    role: CriterionRole
    measured_value: float | int | str | None = None
    passed: bool | None = None
    reason: str


class RegionStoppingPolicy(ContractBase):
    policy_id: str
    criteria: list[SignalCriterion]

    def evaluate_signals(
        self, signals: Mapping[str, float | int | str | None]
    ) -> list[SignalCriterionEvaluation]:
        out: list[SignalCriterionEvaluation] = []
        for c in self.criteria:
            if c.role == CriterionRole.EVIDENCE_ONLY:
                out.append(SignalCriterionEvaluation(
                    signal=c.signal,
                    criterion_binding_status=c.binding_status,
                    evidence_status=(EvidenceStatus.EVALUATED if c.signal in signals
                                     else EvidenceStatus.NOT_EVALUATED),
                    role=c.role,
                    measured_value=signals.get(c.signal),
                    passed=None,
                    reason="evidence-only signal does not affect closure",
                ))
                continue
            if c.binding_status == CriterionBindingStatus.UNBOUND:
                out.append(SignalCriterionEvaluation(
                    signal=c.signal,
                    criterion_binding_status=c.binding_status,
                    evidence_status=EvidenceStatus.NOT_EVALUATED,
                    role=c.role,
                    passed=None,
                    reason="required criterion is unbound",
                ))
                continue
            if c.signal not in signals or signals[c.signal] is None:
                out.append(SignalCriterionEvaluation(
                    signal=c.signal,
                    criterion_binding_status=c.binding_status,
                    evidence_status=EvidenceStatus.NOT_EVALUATED,
                    role=c.role,
                    passed=None,
                    reason="required evidence not evaluated",
                ))
                continue
            passed = criterion_passes(signals[c.signal], c.value, c.comparator)
            out.append(SignalCriterionEvaluation(
                signal=c.signal,
                criterion_binding_status=c.binding_status,
                evidence_status=EvidenceStatus.EVALUATED,
                role=c.role,
                measured_value=signals[c.signal],
                passed=passed,
                reason="passed" if passed else "failed required criterion",
            ))
        return out

    def state_for(
        self, signals: Mapping[str, float | int | str | None]
    ) -> RegionClosureState:
        evaluations = self.evaluate_signals(signals)
        required = [e for e in evaluations if e.role != CriterionRole.EVIDENCE_ONLY]
        if any(e.criterion_binding_status == CriterionBindingStatus.UNBOUND for e in required):
            return RegionClosureState.HUMAN_SCIENTIFIC_INPUT_REQUIRED
        if any(e.evidence_status == EvidenceStatus.NOT_EVALUATED for e in required):
            return RegionClosureState.EVIDENCE_NOT_EVALUATED
        if any(e.passed is False for e in required):
            return RegionClosureState.RECOVER
        return RegionClosureState.CLOSED


__all__ = [
    "CriterionBindingStatus",
    "CriterionComparator",
    "CriterionRole",
    "EvidenceStatus",
    "RegionClosureState",
    "RegionStoppingPolicy",
    "SamplerKind",
    "SamplerRequest",
    "SamplerResult",
    "SelectionStatus",
    "SignalCriterion",
    "SignalCriterionEvaluation",
    "criterion_passes",
    "sample_candidates",
]
