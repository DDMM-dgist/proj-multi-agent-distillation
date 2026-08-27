"""Region-directed recovery contracts for V2."""
from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso
from framework_v2.error_tracking import ErrorLedger
from framework_v2.v2_sampling import SamplerKind, SamplerRequest


class RecoveryAction(str, Enum):
    ADD_TRAINING_SIDE_CANDIDATES = "ADD_TRAINING_SIDE_CANDIDATES"
    REDISTILL_STUDENT = "REDISTILL_STUDENT"


class RegionRecoveryPlan(ContractBase):
    plan_id: str
    campaign_id: str
    iteration: int
    deficient_region_ids: list[str]
    eligible_training_candidate_ids: list[str]
    protected_candidate_ids: list[str] = Field(default_factory=list)
    sampler: SamplerKind
    n_select: int
    action: RecoveryAction = RecoveryAction.ADD_TRAINING_SIDE_CANDIDATES
    teacher_retraining_allowed: bool = False
    new_dft_allowed: bool = False
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    established_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _safety(self):
        if not self.deficient_region_ids:
            raise ValueError("RegionRecoveryPlan requires deficient regions")
        if self.teacher_retraining_allowed:
            raise ValueError("V2 recovery must keep the Teacher frozen")
        if self.new_dft_allowed:
            raise ValueError("V2 recovery must not request new DFT")
        if set(self.eligible_training_candidate_ids) & set(self.protected_candidate_ids):
            raise ValueError("protected structures cannot be eligible recovery candidates")
        return self

    def sampler_request(self, region_by_candidate: dict[str, str]) -> SamplerRequest:
        return SamplerRequest(
            sampler=self.sampler,
            candidate_ids=list(self.eligible_training_candidate_ids),
            n_select=self.n_select,
            region_by_candidate=region_by_candidate,
            deficient_region_ids=list(self.deficient_region_ids),
            protected_candidate_ids=list(self.protected_candidate_ids),
        )


def plan_region_recovery(
    ledger: ErrorLedger,
    *,
    iteration: int,
    eligible_training_candidate_ids: list[str],
    protected_candidate_ids: list[str],
    sampler: SamplerKind,
    n_select: int,
    rationale: str,
) -> RegionRecoveryPlan:
    deficient = ledger.deficient_regions(iteration)
    if not deficient:
        raise ValueError("no deficient regions require recovery")
    return RegionRecoveryPlan(
        plan_id=f"{ledger.campaign_id}_iter{iteration}_region_recovery",
        campaign_id=ledger.campaign_id,
        iteration=iteration,
        deficient_region_ids=deficient,
        eligible_training_candidate_ids=eligible_training_candidate_ids,
        protected_candidate_ids=protected_candidate_ids,
        sampler=sampler,
        n_select=n_select,
        rationale=rationale,
        evidence_refs=[ledger.content_sha256()],
    )


__all__ = [
    "RecoveryAction",
    "RegionRecoveryPlan",
    "plan_region_recovery",
]
