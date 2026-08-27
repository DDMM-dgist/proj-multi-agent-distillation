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


class RecoveryExecutionState(str, Enum):
    PLANNED = "PLANNED"
    LABELS_READY = "LABELS_READY"
    DATASET_READY = "DATASET_READY"
    STUDENT_READY = "STUDENT_READY"
    EVALUATION_READY = "EVALUATION_READY"


class TeacherLabelingRequest(ContractBase):
    request_id: str
    campaign_id: str
    iteration: int
    selected_candidate_ids: list[str]
    region_ids: list[str]
    teacher_identity_sha256: str
    candidate_population_sha256: str
    access_partition_sha256: str
    requires_hpc_approval: bool = True
    expected_output_artifact: str
    next_consumer: str = "TrainingDatasetUpdateRequest"

    @model_validator(mode="after")
    def _valid(self):
        if not self.selected_candidate_ids:
            raise ValueError("teacher labeling requires selected candidates")
        return self


class TrainingDatasetUpdateRequest(ContractBase):
    request_id: str
    campaign_id: str
    iteration: int
    prior_training_population_sha256: str
    teacher_label_artifact_sha256: str
    added_candidate_ids: list[str]
    region_ids: list[str]
    split_lineage_sha256: str
    expected_output_artifact: str
    next_consumer: str = "RedistillationRequest"


class RedistillationRequest(ContractBase):
    request_id: str
    campaign_id: str
    iteration: int
    updated_training_population_sha256: str
    student_recipe_sha256: str
    prior_student_committee_sha256: str | None = None
    requires_hpc_approval: bool = True
    teacher_retraining_allowed: bool = False
    new_dft_allowed: bool = False
    expected_output_artifact: str
    next_consumer: str = "NextEvaluationRequest"

    @model_validator(mode="after")
    def _frozen_teacher_no_new_dft(self):
        if self.teacher_retraining_allowed:
            raise ValueError("V2 redistillation must keep the Teacher frozen")
        if self.new_dft_allowed:
            raise ValueError("V2 redistillation must not request new DFT")
        return self


class NextEvaluationRequest(ContractBase):
    request_id: str
    campaign_id: str
    iteration: int
    student_committee_sha256: str
    protected_population_sha256: str
    evaluation_binding_sha256: str
    target_validation_contract_sha256: str
    expected_output_artifact: str
    next_consumer: str = "ErrorLedger"


class RecoveryExecutionBundle(ContractBase):
    bundle_id: str
    recovery_plan_sha256: str
    state: RecoveryExecutionState
    selected_candidate_ids: list[str]
    teacher_labeling_request: TeacherLabelingRequest
    dataset_update_request: TrainingDatasetUpdateRequest | None = None
    redistillation_request: RedistillationRequest | None = None
    next_evaluation_request: NextEvaluationRequest | None = None
    evaluation_artifact_sha256: str | None = None
    region_ids: list[str]
    iteration: int
    access_partition_sha256: str

    @model_validator(mode="after")
    def _state_consistent(self):
        if not self.selected_candidate_ids:
            raise ValueError("recovery bundle requires selected candidates")
        order = [
            RecoveryExecutionState.PLANNED,
            RecoveryExecutionState.LABELS_READY,
            RecoveryExecutionState.DATASET_READY,
            RecoveryExecutionState.STUDENT_READY,
            RecoveryExecutionState.EVALUATION_READY,
        ]
        reached = order.index(self.state)
        # a downstream request may exist only once its producing transition ran;
        # never pretend a future artifact already exists.
        expect = {
            "dataset_update_request": reached >= 1,
            "redistillation_request": reached >= 2,
            "next_evaluation_request": reached >= 3,
            "evaluation_artifact_sha256": reached >= 4,
        }
        for attr, should_exist in expect.items():
            present = getattr(self, attr) is not None
            if should_exist and not present:
                raise ValueError(f"{attr} required in state {self.state.value}")
            if not should_exist and present:
                raise ValueError(f"{attr} must not exist before its transition in state {self.state.value}")
        return self


def build_planned_recovery(
    plan: RegionRecoveryPlan,
    *,
    selected_candidate_ids: list[str],
    teacher_identity_sha256: str,
    candidate_population_sha256: str,
    access_partition_sha256: str,
    bundle_id: str,
    expected_label_artifact: str,
    labeling_request_id: str | None = None,
) -> RecoveryExecutionBundle:
    if not selected_candidate_ids:
        raise ValueError("recovery execution requires selected candidates")
    if set(selected_candidate_ids) & set(plan.protected_candidate_ids):
        raise ValueError("protected structures cannot enter recovery labeling")
    labeling = TeacherLabelingRequest(
        request_id=labeling_request_id or f"{plan.plan_id}_teacher_labeling",
        campaign_id=plan.campaign_id,
        iteration=plan.iteration,
        selected_candidate_ids=list(selected_candidate_ids),
        region_ids=list(plan.deficient_region_ids),
        teacher_identity_sha256=teacher_identity_sha256,
        candidate_population_sha256=candidate_population_sha256,
        access_partition_sha256=access_partition_sha256,
        expected_output_artifact=expected_label_artifact,
    )
    return RecoveryExecutionBundle(
        bundle_id=bundle_id,
        recovery_plan_sha256=plan.content_sha256(),
        state=RecoveryExecutionState.PLANNED,
        selected_candidate_ids=list(selected_candidate_ids),
        teacher_labeling_request=labeling,
        region_ids=list(plan.deficient_region_ids),
        iteration=plan.iteration,
        access_partition_sha256=access_partition_sha256,
    )


def _require_state(bundle: RecoveryExecutionBundle, expected: RecoveryExecutionState) -> None:
    if bundle.state != expected:
        raise ValueError(
            f"transition requires state {expected.value}, bundle is in {bundle.state.value}"
        )


def attach_label_artifact(
    bundle: RecoveryExecutionBundle,
    *,
    teacher_label_artifact_sha256: str,
    split_lineage_sha256: str,
    expected_output_artifact: str,
    prior_training_population_sha256: str,
) -> RecoveryExecutionBundle:
    _require_state(bundle, RecoveryExecutionState.PLANNED)
    dataset_request = TrainingDatasetUpdateRequest(
        request_id=f"{bundle.bundle_id}_dataset_update",
        campaign_id=bundle.teacher_labeling_request.campaign_id,
        iteration=bundle.iteration,
        prior_training_population_sha256=prior_training_population_sha256,
        teacher_label_artifact_sha256=teacher_label_artifact_sha256,
        added_candidate_ids=list(bundle.selected_candidate_ids),
        region_ids=list(bundle.region_ids),
        split_lineage_sha256=split_lineage_sha256,
        expected_output_artifact=expected_output_artifact,
    )
    return bundle.model_copy(
        update={
            "state": RecoveryExecutionState.LABELS_READY,
            "dataset_update_request": dataset_request,
        }
    )


def attach_updated_dataset(
    bundle: RecoveryExecutionBundle,
    *,
    updated_training_population_sha256: str,
    student_recipe_sha256: str,
    expected_output_artifact: str,
    prior_student_committee_sha256: str | None = None,
) -> RecoveryExecutionBundle:
    _require_state(bundle, RecoveryExecutionState.LABELS_READY)
    redistill = RedistillationRequest(
        request_id=f"{bundle.bundle_id}_redistillation",
        campaign_id=bundle.teacher_labeling_request.campaign_id,
        iteration=bundle.iteration,
        updated_training_population_sha256=updated_training_population_sha256,
        student_recipe_sha256=student_recipe_sha256,
        prior_student_committee_sha256=prior_student_committee_sha256,
        expected_output_artifact=expected_output_artifact,
    )
    return bundle.model_copy(
        update={
            "state": RecoveryExecutionState.DATASET_READY,
            "redistillation_request": redistill,
        }
    )


def attach_student_artifact(
    bundle: RecoveryExecutionBundle,
    *,
    student_committee_sha256: str,
    protected_population_sha256: str,
    evaluation_binding_sha256: str,
    target_validation_contract_sha256: str,
    expected_output_artifact: str,
) -> RecoveryExecutionBundle:
    _require_state(bundle, RecoveryExecutionState.DATASET_READY)
    next_eval = NextEvaluationRequest(
        request_id=f"{bundle.bundle_id}_next_evaluation",
        campaign_id=bundle.teacher_labeling_request.campaign_id,
        iteration=bundle.iteration,
        student_committee_sha256=student_committee_sha256,
        protected_population_sha256=protected_population_sha256,
        evaluation_binding_sha256=evaluation_binding_sha256,
        target_validation_contract_sha256=target_validation_contract_sha256,
        expected_output_artifact=expected_output_artifact,
    )
    return bundle.model_copy(
        update={
            "state": RecoveryExecutionState.STUDENT_READY,
            "next_evaluation_request": next_eval,
        }
    )


def attach_evaluation_artifact(
    bundle: RecoveryExecutionBundle,
    *,
    evaluation_artifact_sha256: str,
) -> RecoveryExecutionBundle:
    _require_state(bundle, RecoveryExecutionState.STUDENT_READY)
    return bundle.model_copy(
        update={
            "state": RecoveryExecutionState.EVALUATION_READY,
            "evaluation_artifact_sha256": evaluation_artifact_sha256,
        }
    )


__all__ = [
    "NextEvaluationRequest",
    "RecoveryAction",
    "RecoveryExecutionBundle",
    "RecoveryExecutionState",
    "RedistillationRequest",
    "RegionRecoveryPlan",
    "TeacherLabelingRequest",
    "TrainingDatasetUpdateRequest",
    "attach_evaluation_artifact",
    "attach_label_artifact",
    "attach_student_artifact",
    "attach_updated_dataset",
    "build_planned_recovery",
    "plan_region_recovery",
]
