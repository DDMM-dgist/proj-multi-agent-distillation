"""V2-H10: thin, non-executing workflow-integration and evidence adapters.

This module is glue only.  It plans the paper-facing seven-step workflow
(SPECIFY -> DISCOVER -> CURATE -> DISTILL -> TRACK -> RECOVER -> VALIDATE),
surfaces coverage / convergence / efficiency evidence in provenance-carrying
records, maps V2 request contracts to the *existing* executors, and emits an
adapter-ready final-evidence record.  It never calls HPC, Teacher inference,
Student training, MD, DFT, replay, or supercell jobs; it only produces
hash-pinned plans and requests that a human authorizes downstream.

Source-confirmed reuse (see V2ExecutorAdapterMap): FE-067 target validation via
``validation.teacher_physical_validation.evaluate_observable`` and FE-068 final
summary via ``validation.run_summary.validate_run_summary_report``.  Convergence
evidence is adapted from the existing ``framework_v2.convergence`` report
(``build_convergence_report``) rather than introducing a second authority.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase
from framework_v2.error_tracking import ErrorLedger, RawEfficiencyRecord
from framework_v2.v2_sampling import RegionClosureState


class V2WorkflowStep(str, Enum):
    SPECIFY = "SPECIFY"
    DISCOVER = "DISCOVER"
    CURATE = "CURATE"
    DISTILL = "DISTILL"
    TRACK = "TRACK"
    RECOVER = "RECOVER"
    VALIDATE = "VALIDATE"


class V2WorkflowStatus(str, Enum):
    READY_TO_PLAN = "READY_TO_PLAN"
    READY_TO_EXECUTE_EXTERNAL_ACTION = "READY_TO_EXECUTE_EXTERNAL_ACTION"
    WAITING_FOR_ARTIFACT = "WAITING_FOR_ARTIFACT"
    SCIENTIFIC_INPUT_REQUIRED = "SCIENTIFIC_INPUT_REQUIRED"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    VALIDATION_READY = "VALIDATION_READY"
    COMPLETE = "COMPLETE"


class CoverageEvidenceRecord(ContractBase):
    record_id: str
    campaign_id: str
    iteration: int
    structural_region_manifest_sha256: str
    region_id: str
    metric_name: str
    measured_value: float | int | str | None
    definition: str
    aggregation: str
    provenance: list[str]
    population_sha256: str | None = None
    criterion_sha256: str | None = None

    @model_validator(mode="after")
    def _valid(self):
        if not self.definition.strip():
            raise ValueError("coverage evidence requires explicit definition")
        if not self.aggregation.strip():
            raise ValueError("coverage evidence requires aggregation definition")
        if not self.provenance:
            raise ValueError("coverage evidence requires provenance")
        return self


class ConvergenceKind(str, Enum):
    TRAINING = "TRAINING"
    RECOVERY_ITERATION = "RECOVERY_ITERATION"
    CAMPAIGN_CLOSURE = "CAMPAIGN_CLOSURE"


class ConvergenceEvidenceRecord(ContractBase):
    record_id: str
    campaign_id: str
    iteration: int
    kind: ConvergenceKind
    artifact_sha256: str
    epochs: int | None = None
    continuation_rounds: int | None = None
    stopping_criterion_sha256: str | None = None
    stopping_reason: str = ""
    converged: bool | None = None
    unresolved_reason: str = ""
    provenance: list[str]

    @model_validator(mode="after")
    def _valid(self):
        if not self.provenance:
            raise ValueError("convergence evidence requires provenance")
        if self.converged is None and not self.unresolved_reason:
            raise ValueError("unresolved convergence requires unresolved_reason")
        return self


class ParetoRecord(ContractBase):
    campaign_id: str
    iteration: int
    region_id: str | None = None
    cumulative_training_structures: int | None = None
    added_structures: int | None = None
    teacher_evaluations: int | None = None
    replay_structures: int | None = None
    gpu_time_seconds: float | None = None
    wall_time_seconds: float | None = None
    recovery_iterations: int | None = None
    epochs: int | None = None
    continuation_rounds: int | None = None
    llm_calls: int | None = None
    judge_calls: int | None = None
    energy_error: float | None = None
    force_error: float | None = None
    target_property_metrics: dict[str, Any] = Field(default_factory=dict)
    region_coverage: dict[str, Any] = Field(default_factory=dict)
    unresolved_region_count: int | None = None
    provenance: list[str]


class EfficiencyEvidenceBundle(ContractBase):
    bundle_id: str
    campaign_id: str
    error_ledger_sha256: str
    convergence_evidence_sha256s: list[str] = Field(default_factory=list)
    pareto_records: list[ParetoRecord]
    final_record: ParetoRecord | None = None
    provenance: list[str]


class V2WorkflowPlan(ContractBase):
    plan_id: str
    campaign_id: str
    current_step: V2WorkflowStep
    status: V2WorkflowStatus
    execution_allowed: bool = False

    human_target_sha256: str | None = None
    target_operationalization_sha256: str | None = None
    target_validation_contract_sha256: str | None = None
    structural_representation_sha256: str | None = None
    structural_region_manifest_sha256: str | None = None
    sampler_result_sha256: str | None = None
    teacher_labeling_request_sha256: str | None = None
    student_training_request_sha256: str | None = None
    evaluation_binding_sha256: str | None = None
    error_ledger_sha256: str | None = None
    recovery_bundle_sha256: str | None = None
    final_validation_request_sha256: str | None = None
    final_evidence_sha256: str | None = None

    unresolved_reason: str = ""
    provenance: list[str] = Field(default_factory=list)


class ExecutorEndpointStatus(str, Enum):
    CONFIRMED_REUSABLE = "CONFIRMED_REUSABLE"
    NEEDS_SOURCE_CONFIRMATION = "NEEDS_SOURCE_CONFIRMATION"
    NEW_ADAPTER_REQUIRED = "NEW_ADAPTER_REQUIRED"


class V2ExecutorEndpoint(ContractBase):
    v2_request_type: str
    expected_existing_executor_capability: str
    source_file: str
    known_existing_symbol: str | None = None
    status: ExecutorEndpointStatus
    notes: str = ""


class V2ExecutorAdapterMap(ContractBase):
    map_id: str
    endpoints: list[V2ExecutorEndpoint]

    @model_validator(mode="after")
    def _unique_request_types(self):
        request_types = [e.v2_request_type for e in self.endpoints]
        if len(set(request_types)) != len(request_types):
            raise ValueError("duplicate V2 request type in executor adapter map")
        return self


class FinalTargetValidationRequest(ContractBase):
    request_id: str
    campaign_id: str
    error_ledger_sha256: str
    target_validation_contract_sha256: str
    final_student_committee_sha256: str
    protected_evaluation_population_sha256: str
    structural_region_manifest_sha256: str
    evaluation_binding_sha256: str
    physical_validation_policy_sha256: str | None = None
    required_region_ids: list[str]
    provenance: list[str]


class V2FinalEvidenceRecord(ContractBase):
    record_id: str
    campaign_id: str
    human_target_sha256: str
    target_operationalization_sha256: str
    target_validation_request_sha256: str
    error_ledger_sha256: str
    efficiency_bundle_sha256: str
    convergence_evidence_sha256s: list[str]
    recovery_history_sha256s: list[str] = Field(default_factory=list)
    teacher_frozen: bool
    new_dft_performed: bool
    protected_population_sha256s: list[str]
    fe067_bridge_ref: str | None = None
    fe068_bridge_ref: str | None = None
    provenance: list[str]

    @model_validator(mode="after")
    def _invariants(self):
        if not self.teacher_frozen:
            raise ValueError("V2 final evidence requires frozen Teacher invariant")
        if self.new_dft_performed:
            raise ValueError("V2 final evidence forbids new DFT")
        if not self.protected_population_sha256s:
            raise ValueError("V2 final evidence requires protected population identity")
        return self


def build_v2_workflow_plan(
    *,
    plan_id: str,
    campaign_id: str,
    human_target_sha256: str,
) -> V2WorkflowPlan:
    return V2WorkflowPlan(
        plan_id=plan_id,
        campaign_id=campaign_id,
        current_step=V2WorkflowStep.SPECIFY,
        status=V2WorkflowStatus.READY_TO_PLAN,
        human_target_sha256=human_target_sha256,
        provenance=[human_target_sha256],
    )


def advance_v2_workflow_plan(
    plan: V2WorkflowPlan,
    *,
    produced_artifact_sha256: str | None = None,
    produced_artifact_type: str,
    unresolved_status: V2WorkflowStatus | None = None,
    unresolved_reason: str = "",
) -> V2WorkflowPlan:
    if unresolved_status is not None:
        return plan.model_copy(
            update={
                "status": unresolved_status,
                "unresolved_reason": unresolved_reason,
            }
        )

    update: dict[str, Any] = {"provenance": [*plan.provenance, produced_artifact_sha256]}

    if plan.current_step == V2WorkflowStep.SPECIFY:
        if produced_artifact_type == "TargetOperationalizationResult":
            update["target_operationalization_sha256"] = produced_artifact_sha256
            update["current_step"] = V2WorkflowStep.DISCOVER
            update["status"] = V2WorkflowStatus.READY_TO_PLAN
        elif produced_artifact_type == "TargetOperationalizationPending":
            update["status"] = V2WorkflowStatus.SCIENTIFIC_INPUT_REQUIRED
        else:
            raise ValueError("SPECIFY expects target operationalization artifact")

    elif plan.current_step == V2WorkflowStep.DISCOVER:
        if produced_artifact_type != "StructuralRegionManifest":
            raise ValueError("DISCOVER expects StructuralRegionManifest")
        update["structural_region_manifest_sha256"] = produced_artifact_sha256
        update["current_step"] = V2WorkflowStep.CURATE
        update["status"] = V2WorkflowStatus.READY_TO_PLAN

    elif plan.current_step == V2WorkflowStep.CURATE:
        if produced_artifact_type == "SamplerResult":
            update["sampler_result_sha256"] = produced_artifact_sha256
            update["current_step"] = V2WorkflowStep.DISTILL
            update["status"] = V2WorkflowStatus.READY_TO_EXECUTE_EXTERNAL_ACTION
        elif produced_artifact_type == "SelectionBudgetInsufficient":
            update["status"] = V2WorkflowStatus.EVIDENCE_INCOMPLETE
            update["unresolved_reason"] = "selection budget insufficient under bound policy"
        else:
            raise ValueError("CURATE expects sampler result or unresolved selection")

    elif plan.current_step == V2WorkflowStep.DISTILL:
        if produced_artifact_type not in {"TeacherLabelingRequest", "RedistillationRequest"}:
            raise ValueError("DISTILL emits external execution request")
        update["teacher_labeling_request_sha256"] = produced_artifact_sha256
        update["current_step"] = V2WorkflowStep.TRACK
        update["status"] = V2WorkflowStatus.WAITING_FOR_ARTIFACT

    elif plan.current_step == V2WorkflowStep.TRACK:
        if produced_artifact_type != "ErrorLedger":
            raise ValueError("TRACK expects ErrorLedger")
        update["error_ledger_sha256"] = produced_artifact_sha256
        update["status"] = V2WorkflowStatus.RECOVERY_REQUIRED

    elif plan.current_step == V2WorkflowStep.RECOVER:
        if produced_artifact_type != "RecoveryExecutionBundle":
            raise ValueError("RECOVER expects RecoveryExecutionBundle")
        update["recovery_bundle_sha256"] = produced_artifact_sha256
        update["current_step"] = V2WorkflowStep.DISTILL
        update["status"] = V2WorkflowStatus.READY_TO_EXECUTE_EXTERNAL_ACTION

    elif plan.current_step == V2WorkflowStep.VALIDATE:
        if produced_artifact_type == "FinalTargetValidationRequest":
            update["final_validation_request_sha256"] = produced_artifact_sha256
            update["status"] = V2WorkflowStatus.WAITING_FOR_ARTIFACT
        elif produced_artifact_type == "V2FinalEvidenceRecord":
            update["final_evidence_sha256"] = produced_artifact_sha256
            update["status"] = V2WorkflowStatus.COMPLETE
        else:
            raise ValueError("VALIDATE expects final validation request/evidence")

    return plan.model_copy(update=update)


def latest_region_states(
    ledger: ErrorLedger,
    required_region_ids: list[str],
) -> dict[str, RegionClosureState]:
    latest: dict[str, RegionClosureState] = {}
    for rid in required_region_ids:
        rows = [r for r in ledger.records if r.region_id == rid]
        if not rows:
            latest[rid] = RegionClosureState.EVIDENCE_NOT_EVALUATED
            continue
        rows = sorted(rows, key=lambda r: r.iteration)
        latest[rid] = rows[-1].state
    return latest


def route_after_tracking(
    plan: V2WorkflowPlan,
    *,
    ledger: ErrorLedger,
    required_region_ids: list[str],
) -> V2WorkflowPlan:
    latest = latest_region_states(ledger, required_region_ids)
    if all(state == RegionClosureState.CLOSED for state in latest.values()):
        return plan.model_copy(
            update={
                "current_step": V2WorkflowStep.VALIDATE,
                "status": V2WorkflowStatus.VALIDATION_READY,
                "error_ledger_sha256": ledger.content_sha256(),
            }
        )
    if any(state == RegionClosureState.RECOVER for state in latest.values()):
        return plan.model_copy(
            update={
                "current_step": V2WorkflowStep.RECOVER,
                "status": V2WorkflowStatus.RECOVERY_REQUIRED,
                "error_ledger_sha256": ledger.content_sha256(),
            }
        )
    return plan.model_copy(
        update={
            "status": V2WorkflowStatus.EVIDENCE_INCOMPLETE,
            "error_ledger_sha256": ledger.content_sha256(),
            "unresolved_reason": (
                "latest required regions are not all CLOSED and no evaluated "
                "RECOVER state is available"
            ),
        }
    )


def coverage_signals_from_records(
    records: list[CoverageEvidenceRecord],
    *,
    region_id: str,
) -> dict[str, float | int | str | None]:
    signals: dict[str, float | int | str | None] = {}
    for record in records:
        if record.region_id != region_id:
            continue
        key = f"coverage.{record.metric_name}"
        if record.measured_value is None:
            signals[key] = None
        else:
            signals[key] = record.measured_value
    return signals


def convergence_from_existing_artifact(
    *,
    record_id: str,
    campaign_id: str,
    iteration: int,
    kind: ConvergenceKind,
    artifact_sha256: str,
    epochs: int | None,
    continuation_rounds: int | None,
    stopping_criterion_sha256: str | None,
    stopping_reason: str,
    converged: bool | None,
    provenance: list[str],
) -> ConvergenceEvidenceRecord:
    return ConvergenceEvidenceRecord(
        record_id=record_id,
        campaign_id=campaign_id,
        iteration=iteration,
        kind=kind,
        artifact_sha256=artifact_sha256,
        epochs=epochs,
        continuation_rounds=continuation_rounds,
        stopping_criterion_sha256=stopping_criterion_sha256,
        stopping_reason=stopping_reason,
        converged=converged,
        unresolved_reason=(
            "" if converged is not None else "convergence artifact did not provide a resolved state"
        ),
        provenance=provenance,
    )


def pareto_records_from_ledger(ledger: ErrorLedger) -> list[ParetoRecord]:
    records: list[ParetoRecord] = []
    for row in sorted(ledger.records, key=lambda r: (r.iteration, r.region_id)):
        eff = row.efficiency
        records.append(
            ParetoRecord(
                campaign_id=ledger.campaign_id,
                iteration=row.iteration,
                region_id=row.region_id,
                cumulative_training_structures=eff.cumulative_training_structures,
                added_structures=eff.added_structures,
                teacher_evaluations=eff.teacher_evaluations,
                replay_structures=eff.replay_structures,
                gpu_time_seconds=eff.gpu_time_seconds,
                wall_time_seconds=eff.wall_time_seconds,
                recovery_iterations=eff.recovery_iterations,
                epochs=eff.epochs,
                continuation_rounds=eff.continuation_rounds,
                llm_calls=eff.llm_calls,
                judge_calls=eff.judge_calls,
                energy_error=row.energy_error,
                force_error=row.force_error,
                target_property_metrics=row.target_property_metrics,
                region_coverage=row.coverage,
                unresolved_region_count=eff.unresolved_region_count,
                provenance=[row.content_sha256(), eff.content_sha256()],
            )
        )
    return records


def build_efficiency_evidence_bundle(
    *,
    bundle_id: str,
    ledger: ErrorLedger,
    convergence_records: list[ConvergenceEvidenceRecord] | None = None,
) -> EfficiencyEvidenceBundle:
    pareto = pareto_records_from_ledger(ledger)
    final = pareto[-1] if pareto else None
    convergence_records = convergence_records or []
    return EfficiencyEvidenceBundle(
        bundle_id=bundle_id,
        campaign_id=ledger.campaign_id,
        error_ledger_sha256=ledger.content_sha256(),
        convergence_evidence_sha256s=[r.content_sha256() for r in convergence_records],
        pareto_records=pareto,
        final_record=final,
        provenance=[ledger.content_sha256(), *[r.content_sha256() for r in convergence_records]],
    )


def default_v2_executor_adapter_map() -> V2ExecutorAdapterMap:
    return V2ExecutorAdapterMap(
        map_id="default_v2_executor_adapter_map",
        endpoints=[
            V2ExecutorEndpoint(
                v2_request_type="TeacherLabelingRequest",
                expected_existing_executor_capability="canonical Teacher labeling / label selected candidate structures",
                source_file="runtimes/pydantic_ai/executors.py",
                known_existing_symbol=None,
                status=ExecutorEndpointStatus.NEEDS_SOURCE_CONFIRMATION,
                notes="inspect existing teacher-label execution action before binding",
            ),
            V2ExecutorEndpoint(
                v2_request_type="TrainingDatasetUpdateRequest",
                expected_existing_executor_capability="build/update Student training dataset from prior train population + new labels",
                source_file="runtimes/pydantic_ai/executors.py",
                known_existing_symbol=None,
                status=ExecutorEndpointStatus.NEEDS_SOURCE_CONFIRMATION,
            ),
            V2ExecutorEndpoint(
                v2_request_type="RedistillationRequest",
                expected_existing_executor_capability="Student committee training/redistillation",
                source_file="runtimes/pydantic_ai/executors.py",
                known_existing_symbol=None,
                status=ExecutorEndpointStatus.NEEDS_SOURCE_CONFIRMATION,
            ),
            V2ExecutorEndpoint(
                v2_request_type="NextEvaluationRequest",
                expected_existing_executor_capability="protected evaluation / Stage-8 style E-F comparison",
                source_file="runtimes/pydantic_ai/executors.py",
                known_existing_symbol=None,
                status=ExecutorEndpointStatus.NEEDS_SOURCE_CONFIRMATION,
            ),
            V2ExecutorEndpoint(
                v2_request_type="FinalTargetValidationRequest",
                expected_existing_executor_capability="FE-067 physical/target validation",
                source_file="validation/teacher_physical_validation.py",
                known_existing_symbol="evaluate_observable",
                status=ExecutorEndpointStatus.CONFIRMED_REUSABLE,
            ),
            V2ExecutorEndpoint(
                v2_request_type="V2FinalEvidenceRecord",
                expected_existing_executor_capability="FE-068 deterministic final analysis producer",
                source_file="validation/run_summary.py",
                known_existing_symbol="validate_run_summary_report",
                status=ExecutorEndpointStatus.CONFIRMED_REUSABLE,
                notes="exact executor-side final summary builder needs source confirmation",
            ),
        ],
    )


def build_final_target_validation_request(
    *,
    request_id: str,
    campaign_id: str,
    ledger: ErrorLedger,
    required_region_ids: list[str],
    target_validation_contract_sha256: str,
    final_student_committee_sha256: str,
    protected_evaluation_population_sha256: str,
    structural_region_manifest_sha256: str,
    evaluation_binding_sha256: str,
    physical_validation_policy_sha256: str | None = None,
) -> FinalTargetValidationRequest:
    latest = latest_region_states(ledger, required_region_ids)
    not_closed = {
        rid: state for rid, state in latest.items() if state != RegionClosureState.CLOSED
    }
    if not_closed:
        raise ValueError(
            "final target validation requires all latest required region states CLOSED; "
            + ", ".join(f"{rid}={state.value}" for rid, state in sorted(not_closed.items()))
        )
    return FinalTargetValidationRequest(
        request_id=request_id,
        campaign_id=campaign_id,
        error_ledger_sha256=ledger.content_sha256(),
        target_validation_contract_sha256=target_validation_contract_sha256,
        final_student_committee_sha256=final_student_committee_sha256,
        protected_evaluation_population_sha256=protected_evaluation_population_sha256,
        structural_region_manifest_sha256=structural_region_manifest_sha256,
        evaluation_binding_sha256=evaluation_binding_sha256,
        physical_validation_policy_sha256=physical_validation_policy_sha256,
        required_region_ids=required_region_ids,
        provenance=[ledger.content_sha256(), target_validation_contract_sha256],
    )


def build_v2_final_evidence_record(
    *,
    record_id: str,
    campaign_id: str,
    human_target_sha256: str,
    target_operationalization_sha256: str,
    final_validation_request: FinalTargetValidationRequest,
    ledger: ErrorLedger,
    efficiency_bundle: EfficiencyEvidenceBundle,
    convergence_records: list[ConvergenceEvidenceRecord],
    recovery_history_sha256s: list[str],
    protected_population_sha256s: list[str],
    fe067_bridge_ref: str | None = None,
    fe068_bridge_ref: str | None = None,
) -> V2FinalEvidenceRecord:
    return V2FinalEvidenceRecord(
        record_id=record_id,
        campaign_id=campaign_id,
        human_target_sha256=human_target_sha256,
        target_operationalization_sha256=target_operationalization_sha256,
        target_validation_request_sha256=final_validation_request.content_sha256(),
        error_ledger_sha256=ledger.content_sha256(),
        efficiency_bundle_sha256=efficiency_bundle.content_sha256(),
        convergence_evidence_sha256s=[r.content_sha256() for r in convergence_records],
        recovery_history_sha256s=recovery_history_sha256s,
        teacher_frozen=True,
        new_dft_performed=False,
        protected_population_sha256s=protected_population_sha256s,
        fe067_bridge_ref=fe067_bridge_ref,
        fe068_bridge_ref=fe068_bridge_ref,
        provenance=[
            final_validation_request.content_sha256(),
            ledger.content_sha256(),
            efficiency_bundle.content_sha256(),
            *[r.content_sha256() for r in convergence_records],
        ],
    )


__all__ = [
    "ConvergenceEvidenceRecord",
    "ConvergenceKind",
    "CoverageEvidenceRecord",
    "EfficiencyEvidenceBundle",
    "ExecutorEndpointStatus",
    "FinalTargetValidationRequest",
    "ParetoRecord",
    "V2ExecutorAdapterMap",
    "V2ExecutorEndpoint",
    "V2FinalEvidenceRecord",
    "V2WorkflowPlan",
    "V2WorkflowStatus",
    "V2WorkflowStep",
    "advance_v2_workflow_plan",
    "build_efficiency_evidence_bundle",
    "build_final_target_validation_request",
    "build_v2_final_evidence_record",
    "build_v2_workflow_plan",
    "convergence_from_existing_artifact",
    "coverage_signals_from_records",
    "default_v2_executor_adapter_map",
    "latest_region_states",
    "pareto_records_from_ledger",
    "route_after_tracking",
]
