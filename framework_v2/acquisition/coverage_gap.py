"""Framework V2 -- CoverageGapAnalysis (INITIAL-phase evidence, binding #3).

Per-regime coverage gaps are the evidence the strategy planner reasons over.
In the INITIAL acquisition phase (no trained Student, no calibrated
uncertainty) this analysis may use ONLY coverage/novelty/diversity/redundancy/
saturation/source-coverage/generator-yield evidence -- never model uncertainty
or expected-information-gain. That restriction is enforced by
``framework_v2.acquisition.validators`` against the produced contract; this
module simply assembles the evidence transparently.

``gap_score`` is a transparent, deterministic function of the *novelty
headroom* and *saturation* of a regime:

    saturated  = saturation >= saturation_threshold        (threshold from policy)
    gap_score  = 0.0                        if saturated
               = max(0, novelty_headroom) * (1 - saturation)   otherwise

The saturation threshold is never invented by the core -- the caller supplies
it from policy/evidence. All inputs (counts, saturation, novelty headroom) are
computed upstream from the descriptor-space coverage machinery
(``coverage/`` NN-distance + aggregate); this module does not compute
descriptors itself.
"""
from __future__ import annotations

import dataclasses

from framework_v2.acquisition.contracts import (
    AcquisitionPhase,
    CoverageGapAnalysis,
    RegimeCoverage,
    RelevanceRole,
)


@dataclasses.dataclass(frozen=True)
class RegimeCoverageInput:
    """Upstream-computed coverage evidence for one target regime.

    ``saturation`` in [0, 1] is the fraction of the regime's descriptor space
    already covered (diminishing-return signal). ``novelty_headroom`` in
    [0, 1] is how much genuinely new descriptor space remains reachable. Both
    come from the descriptor-space coverage machinery, never from a
    material-specific rule here."""
    regime_id: str
    relevance_role: RelevanceRole
    current_count: int
    saturation: float
    novelty_headroom: float
    target_count: int | None = None


def compute_regime_coverage(
    inp: RegimeCoverageInput, *, saturation_threshold: float
) -> RegimeCoverage:
    """Transparent, deterministic per-regime gap computation."""
    if not (0.0 <= saturation_threshold <= 1.0):
        raise ValueError("saturation_threshold must be in [0, 1]")
    sat = inp.saturation
    saturated = sat >= saturation_threshold
    if saturated:
        gap_score = 0.0
    else:
        gap_score = max(0.0, inp.novelty_headroom) * (1.0 - sat)
    return RegimeCoverage(
        regime_id=inp.regime_id,
        relevance_role=inp.relevance_role,
        current_count=inp.current_count,
        target_count=inp.target_count,
        saturation=sat,
        novelty_headroom=inp.novelty_headroom,
        gap_score=gap_score,
        saturated=saturated,
    )


def build_coverage_gap_analysis(
    *,
    analysis_id: str,
    phase: AcquisitionPhase,
    target_regime_model_sha256: str,
    region_resolution_sha256: str,
    regime_inputs: list[RegimeCoverageInput],
    saturation_threshold: float,
    available_source_coverage: dict[str, int] | None = None,
) -> CoverageGapAnalysis:
    """Assemble the per-regime coverage-gap analysis contract.

    Deterministic. The INITIAL/MODEL_INFORMED phase is recorded on the
    contract; the deterministic validator enforces that INITIAL analyses carry
    no uncertainty/EIG evidence."""
    per_regime = [
        compute_regime_coverage(i, saturation_threshold=saturation_threshold)
        for i in regime_inputs
    ]
    return CoverageGapAnalysis(
        analysis_id=analysis_id,
        phase=phase,
        target_regime_model_sha256=target_regime_model_sha256,
        region_resolution_sha256=region_resolution_sha256,
        per_regime=per_regime,
        available_source_coverage=dict(available_source_coverage or {}),
    )
