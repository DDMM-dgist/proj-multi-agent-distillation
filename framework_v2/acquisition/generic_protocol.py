"""Framework V2 -- generic ProtocolEnvelopeBuilder for both backends (FE-027 P4, §10/§11).

A candidate-generation backend needs an ADMISSIBLE decision space (which knobs, within which
bounds) for the Agent to propose a concrete recipe inside. FE-026's specialized plugin supplied
those bounds as material constants (SiO2 sigma/T/beta ranges). The generic path must instead derive
them from DATA + PHYSICS, so a brand-new material gets a usable envelope with no hand-authored
numbers.

The envelope has two parts, both material-agnostic:

  * INPUT-knob bounds -- for LOCAL_PERTURBATION, the displacement magnitude is bounded by a fraction
    of the pool's OWN nearest-neighbor spacing (you must not displace atoms by an appreciable
    fraction of the interatomic distance), and the cell-strain magnitude by a versioned fraction.
    These are derived from ``mean_min_neighbor_distance_A`` -- a generic feature P1 already computed
    -- never from an element pair or phase.

  * OUTPUT-admissibility bounds -- applied to whatever a backend produces, regardless of how:
    a generated structure is admissible only if its minimum interatomic distance stays above a
    fraction of the pool's nearest-neighbor scale (no atomic overlap) and its cell volume changes
    by no more than a versioned fraction. This is what makes TEACHER_DRIVEN_MD safe generically:
    rather than fabricate an absolute temperature ceiling (which raw structure does not determine),
    the envelope constrains the PHYSICAL admissibility of the produced structures and bounds the
    number of MD steps by the compute ceiling. Absolute-T is presence-checked only, and the fact
    that raw structure alone does not bound it is recorded, not papered over.

Everything tunable lives in the versioned :class:`EnvelopeParams`; the same values apply to every
material.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from framework_v2.acquisition.contracts import (
    AcquisitionStrategyKind,
    ComputeCeiling,
)
from framework_v2.contracts import ContractBase
from pydantic import Field


@dataclasses.dataclass(frozen=True)
class EnvelopeParams:
    """Versioned, material-agnostic framework knobs for the protocol envelope.

    ``max_displacement_frac_of_nn`` bounds a perturbation displacement as a fraction of the pool's
    nearest-neighbor spacing; ``max_cell_strain_frac`` bounds cell strain; ``min_interatomic_frac_of_nn``
    is the output-admissibility floor on the minimum interatomic distance (as a fraction of the same
    scale); ``max_volume_change_frac`` bounds admissible cell-volume change."""
    max_displacement_frac_of_nn: float = 0.3
    max_cell_strain_frac: float = 0.1
    min_interatomic_frac_of_nn: float = 0.6
    max_volume_change_frac: float = 0.3
    version: str = "generic_protocol_v1"


class ProtocolEnvelope(ContractBase):
    """The admissible generation-recipe decision space for one backend.

    ``param_bounds`` are numeric (lo, hi) bounds the Agent's proposal must satisfy;
    ``presence_required_keys`` are knobs the proposal must supply but which raw structure does not
    numerically bound (e.g. absolute MD temperature) -- their physical admissibility is enforced by
    ``output_admissibility`` instead. ``nn_scale_A`` records the pool nearest-neighbor scale the
    bounds were derived from, so the derivation is auditable."""
    envelope_id: str
    strategy_kind: AcquisitionStrategyKind
    params_version: str
    nn_scale_A: float
    param_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    presence_required_keys: list[str] = Field(default_factory=list)
    output_admissibility: dict[str, float] = Field(default_factory=dict)
    unbounded_from_raw_structure: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""

    @property
    def required_param_keys(self) -> tuple[str, ...]:
        return tuple(self.param_bounds.keys()) + tuple(self.presence_required_keys)


def _pool_nn_scale(pool) -> float:
    """The conservative pool-wide nearest-neighbor spacing: the MINIMUM per-frame mean nearest
    neighbor distance (angstrom). Fails closed if no frame carries the geometry axis."""
    vals = [
        f.features["mean_min_neighbor_distance_A"]
        for f in pool.frames
        if "mean_min_neighbor_distance_A" in f.features
    ]
    if not vals:
        from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
        raise AcquisitionCapabilityGap(
            "no pooled frame carries a computable nearest-neighbor distance; the generic protocol "
            "envelope cannot derive physical generation bounds from this pool",
            gap_kind="PROTOCOL_ENVELOPE_UNGROUNDED")
    return float(min(vals))


def _output_admissibility(nn_scale: float, params: EnvelopeParams) -> dict[str, float]:
    return {
        "min_interatomic_distance_A": params.min_interatomic_frac_of_nn * nn_scale,
        "max_volume_change_frac": params.max_volume_change_frac,
    }


def build_perturbation_envelope(
    pool, *, params: EnvelopeParams, envelope_id: str, evidence_ref: str = "",
) -> ProtocolEnvelope:
    """Derive the LOCAL_PERTURBATION envelope from the pool's own nearest-neighbor scale."""
    nn = _pool_nn_scale(pool)
    max_disp = params.max_displacement_frac_of_nn * nn
    return ProtocolEnvelope(
        envelope_id=envelope_id,
        strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
        params_version=params.version,
        nn_scale_A=nn,
        param_bounds={
            "displacement_sigma_A": (0.0, max_disp),
            "cell_strain_frac": (0.0, params.max_cell_strain_frac),
        },
        presence_required_keys=["seed"],
        output_admissibility=_output_admissibility(nn, params),
        evidence_refs=[evidence_ref] if evidence_ref else [],
        rationale=("displacement bounded by a fraction of the pool nearest-neighbor spacing; "
                   "generated structures are admissible only within the output bounds"))


def build_md_envelope(
    pool, *, params: EnvelopeParams, envelope_id: str,
    compute_ceiling: Optional[ComputeCeiling] = None, evidence_ref: str = "",
) -> ProtocolEnvelope:
    """Derive the TEACHER_DRIVEN_MD envelope. MD step count is bounded by the compute ceiling; the
    absolute temperature is presence-checked only (raw structure does not bound it), with physical
    admissibility enforced by the output bounds instead."""
    nn = _pool_nn_scale(pool)
    param_bounds: dict[str, tuple[float, float]] = {}
    if compute_ceiling is not None and compute_ceiling.max_md_steps_total is not None:
        param_bounds["n_md_steps"] = (1.0, float(int(compute_ceiling.max_md_steps_total)))
    return ProtocolEnvelope(
        envelope_id=envelope_id,
        strategy_kind=AcquisitionStrategyKind.TEACHER_DRIVEN_MD,
        params_version=params.version,
        nn_scale_A=nn,
        param_bounds=param_bounds,
        presence_required_keys=["temperature_K", "seed"],
        output_admissibility=_output_admissibility(nn, params),
        unbounded_from_raw_structure=["temperature_K"],
        evidence_refs=[evidence_ref] if evidence_ref else [],
        rationale=("MD step count bounded by the compute ceiling; absolute temperature is not "
                   "derivable from raw structure and is presence-checked, with physical "
                   "admissibility enforced by the output bounds on generated structures"))


def check_output_admissible(
    envelope: ProtocolEnvelope, *, min_interatomic_distance_A: float, volume_change_frac: float,
) -> list[str]:
    """Deterministically check a generated structure against the envelope's output bounds. Returns
    issues; empty iff admissible."""
    issues: list[str] = []
    floor = envelope.output_admissibility.get("min_interatomic_distance_A")
    if floor is not None and min_interatomic_distance_A < floor:
        issues.append(
            f"generated structure minimum interatomic distance {min_interatomic_distance_A:.4f} A "
            f"is below the admissible floor {floor:.4f} A (atomic overlap)")
    cap = envelope.output_admissibility.get("max_volume_change_frac")
    if cap is not None and abs(volume_change_frac) > cap:
        issues.append(
            f"generated structure cell-volume change {volume_change_frac:.4f} exceeds the "
            f"admissible bound {cap:.4f}")
    return issues


__all__ = [
    "EnvelopeParams",
    "ProtocolEnvelope",
    "build_perturbation_envelope",
    "build_md_envelope",
    "check_output_admissible",
]
