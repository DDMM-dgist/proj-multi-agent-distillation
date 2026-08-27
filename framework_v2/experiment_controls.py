"""Replay and supercell experiment-control contracts for V2."""
from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase


class ReplayEligibilityRole(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    PROTECTED = "PROTECTED"


class DFTReplayPolicy(ContractBase):
    policy_id: str
    enabled: bool = False
    ratio: float = 0.0
    eligible_frame_ids: list[str] = Field(default_factory=list)
    frame_roles: dict[str, ReplayEligibilityRole] = Field(default_factory=dict)
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
        for frame_id in self.eligible_frame_ids:
            if self.frame_roles.get(frame_id) != ReplayEligibilityRole.TRAIN:
                raise ValueError("DFT replay may use only TRAIN-role frames")
        if self.force_shift_allowed:
            raise ValueError("forces must not receive arbitrary shifts")
        if self.enabled and not self.provenance_refs:
            raise ValueError("enabled replay requires provenance")
        return self


class SupercellUse(str, Enum):
    TRAINING_STRATEGY = "TRAINING_STRATEGY"
    TRANSFERABILITY_VALIDATION = "TRANSFERABILITY_VALIDATION"


class SupercellStrategy(ContractBase):
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
        if self.use == SupercellUse.TRAINING_STRATEGY and not self.teacher_labeling_provenance:
            raise ValueError("training supercell strategy requires Teacher labeling provenance")
        return self


__all__ = [
    "DFTReplayPolicy",
    "ReplayEligibilityRole",
    "SupercellStrategy",
    "SupercellUse",
]
