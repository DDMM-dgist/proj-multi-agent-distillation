"""Framework V2 — ValidationProfile (Section D): define the claim before testing it.

The human provides the high-level scientific/application objective. A PydanticAI
scientific Producer proposes the detailed, versioned ValidationProfile
appropriate to that objective; the human approves the high-level claim envelope
ONCE before expensive execution, after which the profile is frozen.

The profile is the contract that every downstream validation policy
(EvaluationPolicy, UncertaintyPolicy, DeploymentMDPolicy,
PhysicalValidationPolicy) derives from. It is deliberately material-agnostic:
channel ``observable`` names, ``kind`` groupings, tolerances, and the list of
unsupported properties are campaign OUTPUTS filled by the Producer from
evidence, never hard-coded here. The core only enforces the *shape* of a
well-formed claim envelope.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, model_validator

from .contracts import ContractBase, utc_now_iso


class ChannelKind(str, Enum):
    """Generic grouping of a validation channel. No material observable is
    implied by any member; a campaign maps its observables onto these."""
    STRUCTURAL = "structural"
    THERMODYNAMIC = "thermodynamic"
    DYNAMICAL = "dynamical"
    MECHANICAL = "mechanical"
    FIDELITY = "fidelity"          # Student<->Teacher / Teacher<->reference agreement
    OTHER = "other"


class ValidationChannel(ContractBase):
    """One validation channel: a named observable the claim will be tested on.

    ``observable`` is a free campaign-chosen string (e.g. an RDF peak, a
    diffusion coefficient, an elastic constant, an energy MAE) — the core never
    enumerates material observables. ``common`` distinguishes channels that
    apply to essentially any MLIP claim (e.g. energy/force fidelity, MD
    stability) from application-specific channels.
    """
    channel_id: str
    observable: str
    kind: ChannelKind
    common: bool = False
    required: bool = True
    reference: Optional[str] = None          # what it is compared against (Teacher/DFT/expt)
    tolerance: Optional[dict[str, Any]] = None
    rationale: str = ""


class ValidationProfile(ContractBase):
    """The versioned, frozen claim envelope for a campaign (Section D)."""
    profile_id: str
    version: int = 1
    objective: str                            # human-provided high-level objective
    intended_deployment_claim: str            # what the potential is claimed to do
    claim_boundaries: list[str] = Field(default_factory=list)   # explicit limits
    channels: list[ValidationChannel] = Field(default_factory=list)
    unsupported_properties: list[str] = Field(default_factory=list)
    applicable_references: list[str] = Field(default_factory=list)
    linked_scope_contract_sha256: Optional[str] = None
    # human approval of the high-level envelope (Section AD): recorded here, but
    # the authoritative approval event is a Controller ledger record.
    human_approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    established_at: str = Field(default_factory=utc_now_iso)

    def channel(self, channel_id: str) -> Optional[ValidationChannel]:
        for c in self.channels:
            if c.channel_id == channel_id:
                return c
        return None

    def required_channels(self) -> list[ValidationChannel]:
        return [c for c in self.channels if c.required]

    def common_channels(self) -> list[ValidationChannel]:
        return [c for c in self.channels if c.common]

    def application_channels(self) -> list[ValidationChannel]:
        return [c for c in self.channels if not c.common]

    @model_validator(mode="after")
    def _unique_channel_ids(self):
        ids = [c.channel_id for c in self.channels]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate ValidationChannel.channel_id: {dupes}")
        return self

    @model_validator(mode="after")
    def _claim_is_declared(self):
        if not self.intended_deployment_claim.strip():
            raise ValueError("ValidationProfile.intended_deployment_claim must be non-empty")
        return self

    @model_validator(mode="after")
    def _unsupported_not_also_channel(self):
        """A property cannot be both an accepted validation channel observable
        and declared explicitly unsupported (Section D: unsupported properties
        must not be used as acceptance criteria)."""
        obs = {c.observable for c in self.channels}
        clash = sorted(obs.intersection(self.unsupported_properties))
        if clash:
            raise ValueError(
                f"observable(s) {clash} are both a validation channel and listed "
                f"as unsupported_properties — an unsupported property cannot be an "
                f"acceptance criterion"
            )
        return self
