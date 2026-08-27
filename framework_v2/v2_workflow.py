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


class FinalValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"


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
    teacher_label_artifact_sha256: str | None = None
    training_dataset_artifact_sha256: str | None = None
    student_committee_sha256: str | None = None
    student_training_request_sha256: str | None = None
    evaluation_binding_sha256: str | None = None
    error_ledger_sha256: str | None = None
    recovery_bundle_sha256: str | None = None
    final_validation_request_sha256: str | None = None
    final_validation_result_sha256: str | None = None
    final_validation_status: FinalValidationStatus | None = None
    final_analysis_evidence_sha256: str | None = None
    final_evidence_sha256: str | None = None

    unresolved_reason: str = ""
    provenance: list[str] = Field(default_factory=list)


class ExecutorEndpointStatus(str, Enum):
    CONFIRMED_REUSABLE = "CONFIRMED_REUSABLE"
    REUSABLE_VIA_THIN_ADAPTER = "REUSABLE_VIA_THIN_ADAPTER"
    NEEDS_SOURCE_CONFIRMATION = "NEEDS_SOURCE_CONFIRMATION"
    NEW_ADAPTER_REQUIRED = "NEW_ADAPTER_REQUIRED"


class V2ExecutorEndpoint(ContractBase):
    v2_request_type: str
    expected_existing_executor_capability: str
    source_file: str
    known_existing_symbol: str | None = None
    registered_action_type: str | None = None
    requires_hpc_approval: bool = False
    approval_boundary: str | None = None
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


class FinalTargetValidationResult(ContractBase):
    """Deterministic outcome of FE-067 final target validation.

    A request alone can never COMPLETE the workflow; this typed result is the
    RESULT identity the final evidence record must bind.  It carries the request
    SHA it answers, the FE-067 evidence SHA, the PASS/FAIL/UNRESOLVED verdict,
    validation provenance, and the final Student and protected-evaluation
    identities so the verdict cannot be silently re-attributed.
    """

    result_id: str
    campaign_id: str
    request_sha256: str
    status: FinalValidationStatus
    fe067_evidence_sha256: str
    final_student_committee_sha256: str
    protected_evaluation_population_sha256: str
    validation_provenance: list[str]
    per_region_status: dict[str, str] = Field(default_factory=dict)
    unresolved_reason: str = ""

    @model_validator(mode="after")
    def _valid(self):
        if not self.validation_provenance:
            raise ValueError("final validation result requires validation provenance")
        if self.status == FinalValidationStatus.UNRESOLVED and not self.unresolved_reason:
            raise ValueError("unresolved final validation requires unresolved_reason")
        return self


class V2FinalEvidenceRecord(ContractBase):
    record_id: str
    campaign_id: str
    human_target_sha256: str
    target_operationalization_sha256: str
    target_validation_request_sha256: str
    final_validation_result_sha256: str
    final_validation_status: FinalValidationStatus
    final_analysis_evidence_sha256: str
    final_student_committee_sha256: str
    protected_evaluation_population_sha256: str
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
        if self.final_validation_status != FinalValidationStatus.PASS:
            raise ValueError("V2 final evidence requires a PASS final validation result")
        if not self.final_validation_result_sha256.strip():
            raise ValueError("V2 final evidence must bind the final validation RESULT identity")
        if not self.final_analysis_evidence_sha256.strip():
            raise ValueError("V2 final evidence requires deterministic FE-068 analysis evidence")
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
        # DISTILL never advances on a *request*.  It emits an external
        # execution request and then waits until the real artifact chain
        # (Teacher labels -> updated dataset -> Student committee) actually
        # completes.  Only a Student-ready committee artifact permits TRACK.
        if produced_artifact_type in {"TeacherLabelingRequest", "RedistillationRequest"}:
            update["teacher_labeling_request_sha256"] = produced_artifact_sha256
            update["status"] = V2WorkflowStatus.WAITING_FOR_ARTIFACT
        elif produced_artifact_type == "TeacherLabelArtifact":
            if plan.teacher_labeling_request_sha256 is None:
                raise ValueError(
                    "TeacherLabelArtifact requires an emitted labeling request first"
                )
            update["teacher_label_artifact_sha256"] = produced_artifact_sha256
            update["status"] = V2WorkflowStatus.WAITING_FOR_ARTIFACT
        elif produced_artifact_type == "TrainingDatasetArtifact":
            if plan.teacher_label_artifact_sha256 is None:
                raise ValueError(
                    "TrainingDatasetArtifact requires completed Teacher labels first"
                )
            update["training_dataset_artifact_sha256"] = produced_artifact_sha256
            update["status"] = V2WorkflowStatus.WAITING_FOR_ARTIFACT
        elif produced_artifact_type == "StudentCommitteeArtifact":
            if plan.training_dataset_artifact_sha256 is None:
                raise ValueError(
                    "StudentCommitteeArtifact requires an updated training dataset first"
                )
            update["student_committee_sha256"] = produced_artifact_sha256
            update["current_step"] = V2WorkflowStep.TRACK
            update["status"] = V2WorkflowStatus.READY_TO_EXECUTE_EXTERNAL_ACTION
        else:
            raise ValueError(
                "DISTILL emits an external execution request and only advances on the "
                "Student-ready committee artifact"
            )

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
        # Re-entering DISTILL must re-gate through the staged artifact chain;
        # do not carry a prior iteration's intermediate artifacts forward.
        update["teacher_labeling_request_sha256"] = None
        update["teacher_label_artifact_sha256"] = None
        update["training_dataset_artifact_sha256"] = None
        update["student_committee_sha256"] = None

    elif plan.current_step == V2WorkflowStep.VALIDATE:
        # A *request* is never sufficient to COMPLETE.  The chain is:
        # request -> FinalTargetValidationResult(PASS) -> deterministic final
        # analysis evidence (FE-068) -> V2FinalEvidenceRecord -> COMPLETE.
        if produced_artifact_type == "FinalTargetValidationRequest":
            update["final_validation_request_sha256"] = produced_artifact_sha256
            update["status"] = V2WorkflowStatus.WAITING_FOR_ARTIFACT
        elif produced_artifact_type == "V2FinalEvidenceRecord":
            if (
                plan.final_validation_result_sha256 is None
                or plan.final_validation_status != FinalValidationStatus.PASS
            ):
                raise ValueError(
                    "COMPLETE requires a PASS FinalTargetValidationResult before the "
                    "final evidence record"
                )
            if plan.final_analysis_evidence_sha256 is None:
                raise ValueError(
                    "COMPLETE requires deterministic final analysis evidence (FE-068)"
                )
            update["final_evidence_sha256"] = produced_artifact_sha256
            update["status"] = V2WorkflowStatus.COMPLETE
        else:
            raise ValueError("VALIDATE expects final validation request/evidence")

    return plan.model_copy(update=update)


def record_final_validation_result(
    plan: V2WorkflowPlan,
    result: FinalTargetValidationResult,
) -> V2WorkflowPlan:
    """Bind a deterministic FinalTargetValidationResult onto a VALIDATE plan.

    A FinalTargetValidationRequest must already have been emitted, and the
    result must reference exactly that request.  The resulting status is driven
    by the PASS/FAIL/UNRESOLVED verdict; only PASS opens the door to a final
    evidence record.
    """
    if plan.current_step != V2WorkflowStep.VALIDATE:
        raise ValueError("final validation result may only bind on the VALIDATE step")
    if plan.final_validation_request_sha256 is None:
        raise ValueError("final validation result requires an emitted request first")
    if result.request_sha256 != plan.final_validation_request_sha256:
        raise ValueError("final validation result does not reference the emitted request")

    status_map = {
        FinalValidationStatus.PASS: V2WorkflowStatus.VALIDATION_READY,
        FinalValidationStatus.FAIL: V2WorkflowStatus.EVIDENCE_INCOMPLETE,
        FinalValidationStatus.UNRESOLVED: V2WorkflowStatus.SCIENTIFIC_INPUT_REQUIRED,
    }
    update: dict[str, Any] = {
        "final_validation_result_sha256": result.content_sha256(),
        "final_validation_status": result.status,
        "status": status_map[result.status],
        "provenance": [*plan.provenance, result.content_sha256()],
    }
    if result.status != FinalValidationStatus.PASS:
        update["unresolved_reason"] = result.unresolved_reason or (
            f"final target validation returned {result.status.value}"
        )
    return plan.model_copy(update=update)


def record_final_analysis_evidence(
    plan: V2WorkflowPlan,
    *,
    final_analysis_evidence_sha256: str,
) -> V2WorkflowPlan:
    """Bind the deterministic FE-068 final analysis evidence onto the plan.

    Only meaningful once a PASS validation result is bound; the final evidence
    record cannot COMPLETE the workflow without it.
    """
    if plan.final_validation_status != FinalValidationStatus.PASS:
        raise ValueError(
            "final analysis evidence requires a PASS final validation result first"
        )
    return plan.model_copy(
        update={
            "final_analysis_evidence_sha256": final_analysis_evidence_sha256,
            "provenance": [*plan.provenance, final_analysis_evidence_sha256],
        }
    )


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
                known_existing_symbol="_exec_label_with_teacher",
                registered_action_type="label_with_teacher",
                requires_hpc_approval=True,
                approval_boundary="costly_teacher_labeling",
                status=ExecutorEndpointStatus.REUSABLE_VIA_THIN_ADAPTER,
                notes="adapters.acquisition.label_with_teacher; role=data-curator; HPC approval gated",
            ),
            V2ExecutorEndpoint(
                v2_request_type="TrainingDatasetUpdateRequest",
                expected_existing_executor_capability="build/update Student training dataset from prior train population + new labels",
                source_file="runtimes/pydantic_ai/executors.py",
                known_existing_symbol="_exec_generate_group_split",
                registered_action_type="generate_group_split",
                requires_hpc_approval=False,
                approval_boundary=None,
                status=ExecutorEndpointStatus.REUSABLE_VIA_THIN_ADAPTER,
                notes="workflow.steps.split_dataset + prepare_student_distillation_dataset; role=data-curator; deterministic",
            ),
            V2ExecutorEndpoint(
                v2_request_type="RedistillationRequest",
                expected_existing_executor_capability="Student committee training/redistillation",
                source_file="runtimes/pydantic_ai/executors.py",
                known_existing_symbol="_exec_train_committee",
                registered_action_type="train_committee",
                requires_hpc_approval=True,
                approval_boundary="costly_training",
                status=ExecutorEndpointStatus.REUSABLE_VIA_THIN_ADAPTER,
                notes="workflow.steps.train_committee; role=ml-trainer; HPC approval gated",
            ),
            V2ExecutorEndpoint(
                v2_request_type="NextEvaluationRequest",
                expected_existing_executor_capability="protected evaluation / Stage-8 style E-F comparison",
                source_file="runtimes/pydantic_ai/executors.py",
                known_existing_symbol="_exec_evaluate_committee",
                registered_action_type="evaluate_heldout_fidelity",
                requires_hpc_approval=True,
                approval_boundary="costly_training",
                status=ExecutorEndpointStatus.REUSABLE_VIA_THIN_ADAPTER,
                notes="workflow.steps.evaluate_committee / evaluate_multi_population (Stage-8 FE-062); role=ml-trainer; HPC approval gated",
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


class ExecutorDispatchProposal(ContractBase):
    """Non-executing binding of a V2 request to a real registered executor.

    This is a plan, not an execution.  ``executes_immediately`` is always
    False; a human authorizes the actual dispatch downstream.  The HPC-approval
    requirement is derived from the registered action descriptor, never
    self-granted.
    """

    v2_request_type: str
    registered_action_type: str
    executor_symbol: str
    source_file: str
    requires_human_approval: bool
    approval_boundary: str | None = None
    identity_provenance: list[str]
    executes_immediately: bool = False

    @model_validator(mode="after")
    def _non_executing(self):
        if self.executes_immediately:
            raise ValueError("V2 executor dispatch proposals never execute immediately")
        if not self.identity_provenance:
            raise ValueError("executor dispatch proposal requires identity provenance")
        return self


def resolve_executor_dispatch(
    adapter_map: V2ExecutorAdapterMap,
    v2_request_type: str,
    *,
    identity_provenance: list[str],
    action_registry: dict[str, Any] | None = None,
) -> ExecutorDispatchProposal:
    """Resolve a V2 request type to a non-executing dispatch proposal.

    ``framework_v2`` stays pure: the caller injects the real
    ``build_executor_registry()`` result as ``action_registry`` when it wants
    the approval boundary cross-checked against the live descriptor.  When
    injected, a mismatch between the adapter map and the registry fails closed.
    """
    endpoint = next(
        (e for e in adapter_map.endpoints if e.v2_request_type == v2_request_type),
        None,
    )
    if endpoint is None:
        raise ValueError(f"no executor endpoint mapped for {v2_request_type}")
    if endpoint.registered_action_type is None or endpoint.known_existing_symbol is None:
        raise ValueError(
            f"{v2_request_type} is not bound to a registered executor "
            f"(status={endpoint.status.value})"
        )

    requires_approval = endpoint.requires_hpc_approval
    approval_boundary = endpoint.approval_boundary
    if action_registry is not None:
        descriptor = action_registry.get(endpoint.registered_action_type)
        if descriptor is None:
            raise ValueError(
                f"registered action {endpoint.registered_action_type} absent from "
                "injected executor registry"
            )
        live_boundary = getattr(descriptor, "approval_boundary", None)
        live_requires = live_boundary is not None
        if live_requires != requires_approval or live_boundary != approval_boundary:
            raise ValueError(
                "executor adapter approval boundary disagrees with live registry for "
                f"{endpoint.registered_action_type}: adapter="
                f"({requires_approval},{approval_boundary}) registry="
                f"({live_requires},{live_boundary})"
            )

    return ExecutorDispatchProposal(
        v2_request_type=endpoint.v2_request_type,
        registered_action_type=endpoint.registered_action_type,
        executor_symbol=endpoint.known_existing_symbol,
        source_file=endpoint.source_file,
        requires_human_approval=requires_approval,
        approval_boundary=approval_boundary,
        identity_provenance=list(identity_provenance),
        executes_immediately=False,
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


def build_final_target_validation_result(
    *,
    result_id: str,
    request: FinalTargetValidationRequest,
    status: FinalValidationStatus,
    fe067_evidence_sha256: str,
    validation_provenance: list[str],
    per_region_status: dict[str, str] | None = None,
    unresolved_reason: str = "",
) -> FinalTargetValidationResult:
    """Bind a deterministic FE-067 validation outcome to its request.

    The final Student and protected-evaluation identities are pulled from the
    request so the verdict cannot be re-attributed to a different Student or
    evaluation population.
    """
    return FinalTargetValidationResult(
        result_id=result_id,
        campaign_id=request.campaign_id,
        request_sha256=request.content_sha256(),
        status=status,
        fe067_evidence_sha256=fe067_evidence_sha256,
        final_student_committee_sha256=request.final_student_committee_sha256,
        protected_evaluation_population_sha256=request.protected_evaluation_population_sha256,
        validation_provenance=validation_provenance,
        per_region_status=per_region_status or {},
        unresolved_reason=unresolved_reason,
    )


def build_v2_final_evidence_record(
    *,
    record_id: str,
    campaign_id: str,
    human_target_sha256: str,
    target_operationalization_sha256: str,
    final_validation_result: FinalTargetValidationResult,
    final_analysis_evidence_sha256: str,
    ledger: ErrorLedger,
    efficiency_bundle: EfficiencyEvidenceBundle,
    convergence_records: list[ConvergenceEvidenceRecord],
    recovery_history_sha256s: list[str],
    protected_population_sha256s: list[str],
    fe067_bridge_ref: str | None = None,
    fe068_bridge_ref: str | None = None,
) -> V2FinalEvidenceRecord:
    if final_validation_result.status != FinalValidationStatus.PASS:
        raise ValueError(
            "V2 final evidence requires a PASS final validation result; got "
            f"{final_validation_result.status.value}"
        )
    if not final_analysis_evidence_sha256.strip():
        raise ValueError("V2 final evidence requires deterministic FE-068 analysis evidence")
    return V2FinalEvidenceRecord(
        record_id=record_id,
        campaign_id=campaign_id,
        human_target_sha256=human_target_sha256,
        target_operationalization_sha256=target_operationalization_sha256,
        target_validation_request_sha256=final_validation_result.request_sha256,
        final_validation_result_sha256=final_validation_result.content_sha256(),
        final_validation_status=final_validation_result.status,
        final_analysis_evidence_sha256=final_analysis_evidence_sha256,
        final_student_committee_sha256=final_validation_result.final_student_committee_sha256,
        protected_evaluation_population_sha256=(
            final_validation_result.protected_evaluation_population_sha256
        ),
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
            final_validation_result.content_sha256(),
            final_validation_result.request_sha256,
            final_analysis_evidence_sha256,
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
    "ExecutorDispatchProposal",
    "ExecutorEndpointStatus",
    "FinalTargetValidationRequest",
    "FinalTargetValidationResult",
    "FinalValidationStatus",
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
    "build_final_target_validation_result",
    "build_v2_final_evidence_record",
    "build_v2_workflow_plan",
    "convergence_from_existing_artifact",
    "coverage_signals_from_records",
    "default_v2_executor_adapter_map",
    "latest_region_states",
    "pareto_records_from_ledger",
    "record_final_analysis_evidence",
    "record_final_validation_result",
    "resolve_executor_dispatch",
    "route_after_tracking",
]
