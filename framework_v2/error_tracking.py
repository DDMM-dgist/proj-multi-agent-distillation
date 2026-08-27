"""Region-resolved V2 error and efficiency ledgers."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso
from framework_v2.v2_sampling import RegionClosureState


class ReferenceChannel(str, Enum):
    STUDENT_VS_TEACHER = "student_vs_teacher"
    STUDENT_VS_DFT = "student_vs_dft"
    TEACHER_VS_DFT = "teacher_vs_dft"
    TARGET_PROPERTY = "target_property"


class RawEfficiencyRecord(ContractBase):
    selected_structures: int = 0
    cumulative_training_structures: int = 0
    added_structures: int = 0
    teacher_evaluations: int = 0
    replay_structures: int = 0
    epochs: int = 0
    continuation_rounds: int = 0
    gpu_time_seconds: float | None = None
    wall_time_seconds: float | None = None
    recovery_iterations: int = 0
    llm_calls: int = 0
    judge_calls: int = 0
    token_count: int | None = None
    orchestration_latency_seconds: float | None = None
    energy_error: float | None = None
    force_error: float | None = None
    target_property_metrics: dict[str, float] = Field(default_factory=dict)
    region_coverage: dict[str, float] = Field(default_factory=dict)
    unresolved_region_count: int = 0
    outcome: str = ""
    stopping_reason: str = ""

    @model_validator(mode="after")
    def _nonnegative_counts(self):
        for field in (
            "selected_structures",
            "cumulative_training_structures",
            "added_structures",
            "teacher_evaluations",
            "replay_structures",
            "epochs",
            "continuation_rounds",
            "recovery_iterations",
            "llm_calls",
            "judge_calls",
            "unresolved_region_count",
        ):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")
        return self


class RegionErrorRecord(ContractBase):
    campaign_id: str
    iteration: int
    region_id: str
    region_membership_sha256: str
    state: RegionClosureState
    region_candidate_count: int = 0
    selected_structures: list[str] = Field(default_factory=list)
    newly_added_structures: list[str] = Field(default_factory=list)
    cumulative_training_population_sha256: str = ""
    sampling_strategy: str = ""
    candidate_source: str = ""
    replay_exposure: dict[str, Any] = Field(default_factory=dict)
    supercell_lineage: list[str] = Field(default_factory=list)
    energy_error: float | None = None
    force_error: float | None = None
    reference_channel: ReferenceChannel = ReferenceChannel.STUDENT_VS_TEACHER
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    target_property_metrics: dict[str, Any] = Field(default_factory=dict)
    target_property_criterion: dict[str, Any] | None = None
    target_property_criterion_provenance: list[str] = Field(default_factory=list)
    target_property_requirement: str = "evidence_only"
    coverage: dict[str, Any] = Field(default_factory=dict)
    efficiency: RawEfficiencyRecord = Field(default_factory=RawEfficiencyRecord)
    failure_reason: str = ""
    intervention: str = ""
    before_metric: dict[str, float] = Field(default_factory=dict)
    after_metric: dict[str, float] = Field(default_factory=dict)
    delta: dict[str, float] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    recorded_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _valid(self):
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.state == RegionClosureState.RECOVER and not self.failure_reason:
            raise ValueError("RECOVER records require failure_reason")
        if self.target_property_criterion and not self.target_property_criterion_provenance:
            raise ValueError("target-property criteria require provenance")
        return self


class ErrorLedger(ContractBase):
    ledger_id: str
    campaign_id: str
    records: list[RegionErrorRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_iteration_region(self):
        keys = [(r.iteration, r.region_id) for r in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("ErrorLedger contains duplicate iteration/region records")
        return self

    def records_for_iteration(self, iteration: int) -> list[RegionErrorRecord]:
        return [r for r in self.records if r.iteration == iteration]

    def deficient_regions(self, iteration: int) -> list[str]:
        return [
            r.region_id
            for r in self.records_for_iteration(iteration)
            if r.state == RegionClosureState.RECOVER
        ]

    def append(self, record: RegionErrorRecord) -> "ErrorLedger":
        if record.campaign_id != self.campaign_id:
            raise ValueError("record campaign_id does not match ledger")
        return ErrorLedger(
            ledger_id=self.ledger_id,
            campaign_id=self.campaign_id,
            records=[*self.records, record],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)
        )

    @classmethod
    def load(cls, path: str | Path) -> "ErrorLedger":
        return cls.model_validate(json.loads(Path(path).read_text()))


__all__ = [
    "ErrorLedger",
    "RawEfficiencyRecord",
    "ReferenceChannel",
    "RegionErrorRecord",
]
