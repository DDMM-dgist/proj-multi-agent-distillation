"""Region-resolved V2 error and efficiency ledgers."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso
from framework_v2.region_evaluation import (
    EvaluationPopulationRegionBinding,
    RegionEvaluationRecord,
)
from framework_v2.v2_sampling import (
    RegionClosureState,
    RegionStoppingPolicy,
    SignalCriterionEvaluation,
)


class ReferenceChannel(str, Enum):
    STUDENT_VS_TEACHER = "student_vs_teacher"
    STUDENT_VS_DFT = "student_vs_dft"
    TEACHER_VS_DFT = "teacher_vs_dft"
    TARGET_PROPERTY = "target_property"


# Numeric efficiency fields: "unknown" is not "zero"; a measured value must
# carry measurement provenance so cost evidence is never fabricated.
NUMERIC_EFFICIENCY_FIELDS = (
    "selected_structures",
    "cumulative_training_structures",
    "added_structures",
    "teacher_evaluations",
    "replay_structures",
    "epochs",
    "continuation_rounds",
    "gpu_time_seconds",
    "wall_time_seconds",
    "recovery_iterations",
    "llm_calls",
    "judge_calls",
    "token_count",
    "orchestration_latency_seconds",
    "energy_error",
    "force_error",
    "unresolved_region_count",
)

_NONNEGATIVE_EFFICIENCY_FIELDS = (
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
    "token_count",
    "gpu_time_seconds",
    "wall_time_seconds",
    "orchestration_latency_seconds",
    "unresolved_region_count",
)


class RawEfficiencyRecord(ContractBase):
    selected_structures: int | None = None
    cumulative_training_structures: int | None = None
    added_structures: int | None = None
    teacher_evaluations: int | None = None
    replay_structures: int | None = None
    epochs: int | None = None
    continuation_rounds: int | None = None
    gpu_time_seconds: float | None = None
    wall_time_seconds: float | None = None
    recovery_iterations: int | None = None
    llm_calls: int | None = None
    judge_calls: int | None = None
    token_count: int | None = None
    orchestration_latency_seconds: float | None = None
    energy_error: float | None = None
    force_error: float | None = None
    target_property_metrics: dict[str, float] = Field(default_factory=dict)
    region_coverage: dict[str, float] = Field(default_factory=dict)
    unresolved_region_count: int | None = None
    outcome: str = ""
    stopping_reason: str = ""
    measurement_provenance: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _measured_values_have_provenance(self):
        for field in _NONNEGATIVE_EFFICIENCY_FIELDS:
            value = getattr(self, field)
            if value is not None and value < 0:
                raise ValueError(f"{field} must be non-negative")
        for field in NUMERIC_EFFICIENCY_FIELDS:
            if getattr(self, field) is not None and field not in self.measurement_provenance:
                raise ValueError(f"{field} is measured but lacks provenance")
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
    criterion_evaluations: list[SignalCriterionEvaluation] = Field(default_factory=list)
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


def _failure_reason(
    state: RegionClosureState, evals: list[SignalCriterionEvaluation]
) -> str:
    if state == RegionClosureState.RECOVER:
        failed = [e.signal for e in evals if e.passed is False]
        if failed:
            return "failed required criteria: " + ", ".join(sorted(failed))
        return "region requires recovery"
    if state == RegionClosureState.HUMAN_SCIENTIFIC_INPUT_REQUIRED:
        return "unbound required criterion needs human scientific input"
    if state == RegionClosureState.EVIDENCE_NOT_EVALUATED:
        return "required closure evidence not evaluated"
    return ""


def build_error_ledger_iteration(
    *,
    ledger: ErrorLedger,
    iteration: int,
    evaluation_binding: EvaluationPopulationRegionBinding,
    region_evaluations: list[RegionEvaluationRecord],
    closure_policy: RegionStoppingPolicy,
    target_validation_sha256: str | None,
    training_population_sha256: str,
    efficiency: RawEfficiencyRecord,
    uncertainty_evidence_sha256: str | None = None,
    coverage_evidence_sha256: str | None = None,
) -> ErrorLedger:
    out = ledger
    for rev in sorted(region_evaluations, key=lambda r: r.region_id):
        evals = closure_policy.evaluate_signals(rev.namespaced_signals)
        state = closure_policy.state_for(rev.namespaced_signals)
        record = RegionErrorRecord(
            campaign_id=ledger.campaign_id,
            iteration=iteration,
            region_id=rev.region_id,
            region_membership_sha256=rev.region_membership_sha256,
            state=state,
            cumulative_training_population_sha256=training_population_sha256,
            energy_error=rev.energy_rmse_meV_per_atom,
            force_error=rev.force_component_rmse_eV_per_angstrom,
            uncertainty={
                k: v for k, v in rev.namespaced_signals.items() if k.startswith("uncertainty.")
            },
            target_property_metrics={
                k: v for k, v in rev.namespaced_signals.items() if k.startswith("target.")
            },
            coverage={
                k: v for k, v in rev.namespaced_signals.items() if k.startswith("coverage.")
            },
            efficiency=efficiency,
            failure_reason=_failure_reason(state, evals),
            evidence_refs=[
                evaluation_binding.content_sha256(),
                rev.content_sha256(),
                *filter(
                    None,
                    [
                        target_validation_sha256,
                        uncertainty_evidence_sha256,
                        coverage_evidence_sha256,
                    ],
                ),
            ],
            criterion_evaluations=evals,
        )
        out = out.append(record)
    return out


__all__ = [
    "ErrorLedger",
    "NUMERIC_EFFICIENCY_FIELDS",
    "RawEfficiencyRecord",
    "ReferenceChannel",
    "RegionErrorRecord",
    "build_error_ledger_iteration",
]
