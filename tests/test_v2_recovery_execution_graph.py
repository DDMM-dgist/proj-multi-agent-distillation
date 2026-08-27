"""V2-H05: staged region recovery execution graph.

A RECOVER region produces a concrete, ordered chain of authorized next requests
(teacher labeling -> dataset update -> redistillation -> next evaluation) without
pretending future artifacts already exist.  The Teacher stays frozen and no new
DFT is requested at any stage.
"""
import pytest

from framework_v2.error_tracking import ErrorLedger, RegionErrorRecord
from framework_v2.region_recovery import (
    NextEvaluationRequest,
    RecoveryExecutionState,
    RedistillationRequest,
    attach_evaluation_artifact,
    attach_label_artifact,
    attach_student_artifact,
    attach_updated_dataset,
    build_planned_recovery,
    plan_region_recovery,
)
from framework_v2.v2_sampling import RegionClosureState, SamplerKind


def _plan():
    ledger = ErrorLedger(
        ledger_id="l",
        campaign_id="c",
        records=[
            RegionErrorRecord(
                campaign_id="c",
                iteration=0,
                region_id="r_bad",
                region_membership_sha256="r_bad",
                state=RegionClosureState.RECOVER,
                failure_reason="force error high",
            )
        ],
    )
    return plan_region_recovery(
        ledger,
        iteration=0,
        eligible_training_candidate_ids=["c1", "c2", "c3"],
        protected_candidate_ids=["p1"],
        sampler=SamplerKind.DIRECT_LIKE,
        n_select=2,
        rationale="recover r_bad",
    )


def _planned_bundle(selected=("c1", "c2")):
    return build_planned_recovery(
        _plan(),
        selected_candidate_ids=list(selected),
        teacher_identity_sha256="teacher",
        candidate_population_sha256="pool",
        access_partition_sha256="access",
        bundle_id="bundle",
        expected_label_artifact="labels.json",
    )


def _full_bundle():
    bundle = _planned_bundle()
    bundle = attach_label_artifact(
        bundle,
        teacher_label_artifact_sha256="labels",
        split_lineage_sha256="split",
        expected_output_artifact="dataset.extxyz",
        prior_training_population_sha256="train0",
    )
    bundle = attach_updated_dataset(
        bundle,
        updated_training_population_sha256="train1",
        student_recipe_sha256="recipe",
        expected_output_artifact="committee.pt",
    )
    bundle = attach_student_artifact(
        bundle,
        student_committee_sha256="committee",
        protected_population_sha256="protected",
        evaluation_binding_sha256="binding",
        target_validation_contract_sha256="tv",
        expected_output_artifact="eval.json",
    )
    return attach_evaluation_artifact(bundle, evaluation_artifact_sha256="eval")


def test_planned_bundle_has_no_future_hash_requests():
    bundle = _planned_bundle()
    assert bundle.state == RecoveryExecutionState.PLANNED
    assert bundle.dataset_update_request is None
    assert bundle.redistillation_request is None
    assert bundle.next_evaluation_request is None
    assert bundle.evaluation_artifact_sha256 is None
    assert bundle.teacher_labeling_request.requires_hpc_approval is True


def test_recovery_transitions_require_correct_state():
    planned = _planned_bundle()
    with pytest.raises(ValueError, match="requires state LABELS_READY"):
        attach_updated_dataset(
            planned,
            updated_training_population_sha256="x",
            student_recipe_sha256="r",
            expected_output_artifact="c.pt",
        )
    with pytest.raises(ValueError, match="requires state DATASET_READY"):
        attach_student_artifact(
            planned,
            student_committee_sha256="x",
            protected_population_sha256="p",
            evaluation_binding_sha256="b",
            target_validation_contract_sha256="tv",
            expected_output_artifact="e.json",
        )
    with pytest.raises(ValueError, match="requires state STUDENT_READY"):
        attach_evaluation_artifact(planned, evaluation_artifact_sha256="e")


def test_full_chain_reaches_evaluation_ready_in_order():
    bundle = _full_bundle()
    assert bundle.state == RecoveryExecutionState.EVALUATION_READY
    assert bundle.dataset_update_request is not None
    assert bundle.redistillation_request is not None
    assert bundle.next_evaluation_request is not None
    assert bundle.evaluation_artifact_sha256 == "eval"
    # chain wiring
    assert bundle.teacher_labeling_request.next_consumer == "TrainingDatasetUpdateRequest"
    assert bundle.dataset_update_request.next_consumer == "RedistillationRequest"
    assert bundle.redistillation_request.next_consumer == "NextEvaluationRequest"
    assert bundle.next_evaluation_request.next_consumer == "ErrorLedger"


def test_redistillation_keeps_teacher_frozen_and_no_new_dft():
    with pytest.raises(ValueError, match="Teacher frozen"):
        RedistillationRequest(
            request_id="r",
            campaign_id="c",
            iteration=0,
            updated_training_population_sha256="t",
            student_recipe_sha256="s",
            teacher_retraining_allowed=True,
            expected_output_artifact="c.pt",
        )
    with pytest.raises(ValueError, match="new DFT"):
        RedistillationRequest(
            request_id="r",
            campaign_id="c",
            iteration=0,
            updated_training_population_sha256="t",
            student_recipe_sha256="s",
            new_dft_allowed=True,
            expected_output_artifact="c.pt",
        )


def test_protected_candidates_cannot_enter_recovery_labeling():
    with pytest.raises(ValueError, match="protected"):
        build_planned_recovery(
            _plan(),
            selected_candidate_ids=["c1", "p1"],  # p1 is protected
            teacher_identity_sha256="teacher",
            candidate_population_sha256="pool",
            access_partition_sha256="access",
            bundle_id="bundle",
            expected_label_artifact="labels.json",
        )


def test_empty_selection_rejected():
    with pytest.raises(ValueError, match="selected candidates"):
        build_planned_recovery(
            _plan(),
            selected_candidate_ids=[],
            teacher_identity_sha256="teacher",
            candidate_population_sha256="pool",
            access_partition_sha256="access",
            bundle_id="bundle",
            expected_label_artifact="labels.json",
        )
