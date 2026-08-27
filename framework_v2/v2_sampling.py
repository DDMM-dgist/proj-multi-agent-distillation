"""V2 sampler and stopping-policy interfaces.

Samplers answer WHICH structures to add.  Stopping policies answer WHETHER a
region still needs data.  Keeping these independent lets later experiments
compare Random/FPS/DIRECT-like selectors under the same closure criterion.
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


class RegionClosureState(str, Enum):
    OPEN = "OPEN"
    LEARNING = "LEARNING"
    RECOVER = "RECOVER"
    CLOSED = "CLOSED"
    HUMAN_SCIENTIFIC_INPUT_REQUIRED = "HUMAN_SCIENTIFIC_INPUT_REQUIRED"


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


def _eligible(request: SamplerRequest) -> list[str]:
    ids = [cid for cid in request.candidate_ids if cid not in set(request.protected_candidate_ids)]
    if request.deficient_region_ids:
        allowed = set(request.deficient_region_ids)
        ids = [cid for cid in ids if request.region_by_candidate.get(cid) in allowed]
    return ids


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


def sample_candidates(
    request: SamplerRequest,
    *,
    representation: StructuralRepresentation | None = None,
) -> SamplerResult:
    ids = _eligible(request)
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
        if request.region_by_candidate:
            counts: dict[str, int] = {}
            ordered = sorted(
                ids,
                key=lambda cid: (
                    counts.setdefault(request.region_by_candidate.get(cid, ""), 0),
                    request.region_by_candidate.get(cid, ""),
                    ids.index(cid),
                ),
            )
        else:
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
    )


class RegionStoppingPolicy(ContractBase):
    policy_id: str
    required_signals: list[str]
    thresholds: dict[str, float | int | str] = Field(default_factory=dict)
    threshold_provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _thresholds_bound(self):
        missing = [s for s in self.required_signals if s not in self.thresholds]
        if missing:
            raise ValueError(
                "RegionStoppingPolicy has unbound required signal thresholds: "
                + ", ".join(missing)
            )
        missing_provenance = [s for s in self.required_signals if s not in self.threshold_provenance]
        if missing_provenance:
            raise ValueError(
                "RegionStoppingPolicy thresholds lack provenance: "
                + ", ".join(missing_provenance)
            )
        return self

    def state_for(self, signals: Mapping[str, float | int | str]) -> RegionClosureState:
        for signal in self.required_signals:
            value = signals.get(signal)
            threshold = self.thresholds[signal]
            if value is None:
                return RegionClosureState.HUMAN_SCIENTIFIC_INPUT_REQUIRED
            if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
                if float(value) > float(threshold):
                    return RegionClosureState.RECOVER
            elif value != threshold:
                return RegionClosureState.RECOVER
        return RegionClosureState.CLOSED


__all__ = [
    "RegionClosureState",
    "RegionStoppingPolicy",
    "SamplerKind",
    "SamplerRequest",
    "SamplerResult",
    "sample_candidates",
]
