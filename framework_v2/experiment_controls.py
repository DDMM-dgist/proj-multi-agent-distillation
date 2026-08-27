"""Replay and supercell experiment-control contracts for V2."""
from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase
from framework_v2.protected_eligibility import EligibilityCheckResult


class ReplayEligibilityRole(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    PROTECTED = "PROTECTED"


class DFTReplayPolicy(ContractBase):
    """DFT replay policy.

    V2 does not trust an inline ``frame_roles`` map as its own authority for
    what may be replayed.  Instead the selected frames must equal the frames a
    validated :class:`EligibilityCheckResult` already cleared as TRAIN-eligible
    (see :func:`build_replay_training_plan`).
    """

    policy_id: str
    enabled: bool = False
    ratio: float = 0.0
    selected_frame_ids: list[str] = Field(default_factory=list)
    eligibility_manifest_sha256: str | None = None
    provenance_refs: list[str] = Field(default_factory=list)
    energy_alignment: str = "explicit_if_teacher_and_dft_coexist"
    force_shift_allowed: bool = False

    @model_validator(mode="after")
    def _valid(self):
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError("replay ratio must be in [0, 1]")
        if self.enabled and self.ratio <= 0.0:
            raise ValueError("enabled replay requires a positive explicit ratio")
        if self.ratio > 0.0 and not self.enabled:
            raise ValueError("nonzero replay ratio requires enabled=True")
        if self.force_shift_allowed:
            raise ValueError("forces must not receive arbitrary shifts")
        if self.enabled and not self.provenance_refs:
            raise ValueError("enabled replay requires provenance")
        return self


class ReplayTrainingPlan(ContractBase):
    plan_id: str
    policy_sha256: str
    selected_frame_ids: list[str]
    ratio: float
    eligibility_result_sha256: str
    provenance_refs: list[str] = Field(default_factory=list)


def build_replay_training_plan(
    policy: DFTReplayPolicy,
    eligibility: EligibilityCheckResult,
    *,
    plan_id: str | None = None,
) -> ReplayTrainingPlan:
    if not policy.enabled:
        raise ValueError("replay disabled")
    if set(policy.selected_frame_ids) != set(eligibility.eligible_candidate_ids):
        raise ValueError("replay selected IDs do not match validated eligibility")
    return ReplayTrainingPlan(
        plan_id=plan_id or f"{policy.policy_id}_replay_plan",
        policy_sha256=policy.content_sha256(),
        selected_frame_ids=list(policy.selected_frame_ids),
        ratio=policy.ratio,
        eligibility_result_sha256=eligibility.content_sha256(),
        provenance_refs=list(policy.provenance_refs),
    )


class SupercellUse(str, Enum):
    TRAINING_STRATEGY = "TRAINING_STRATEGY"
    TRANSFERABILITY_VALIDATION = "TRANSFERABILITY_VALIDATION"


class SupercellStrategy(ContractBase):
    """A supercell replication strategy.

    A TRAINING_STRATEGY strategy is a *plan*: it may be authored before any
    Teacher labels exist because the generated candidates are labeled afterwards
    (see :class:`SupercellExecutionRecord`) and only then admitted to training.
    """

    strategy_id: str
    use: SupercellUse
    parent_ids: list[str]
    replication_matrix: list[list[int]]
    perturbation_policy_sha256: str | None = None
    teacher_labeling_provenance: list[str] = Field(default_factory=list)
    lineage_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self):
        if len(self.replication_matrix) != 3 or any(len(row) != 3 for row in self.replication_matrix):
            raise ValueError("replication_matrix must be 3x3")
        if not self.parent_ids:
            raise ValueError("SupercellStrategy requires parent_ids")
        return self


class SupercellExecutionRecord(ContractBase):
    strategy_sha256: str
    generated_candidate_ids: list[str]
    parent_ids: list[str]
    replication_matrix: list[list[int]]
    teacher_labeling_provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self):
        if not self.generated_candidate_ids:
            raise ValueError("supercell execution requires generated candidates")
        return self

    def admits_to_training(self) -> bool:
        """Generated candidates may enter the training set only once labeled."""
        return bool(self.teacher_labeling_provenance)


__all__ = [
    "DFTReplayPolicy",
    "ReplayEligibilityRole",
    "ReplayTrainingPlan",
    "SupercellExecutionRecord",
    "SupercellStrategy",
    "SupercellUse",
    "build_replay_training_plan",
]
