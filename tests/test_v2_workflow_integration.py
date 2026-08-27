"""V2-H10: workflow integration adapters + evidence bridges.

These tests exercise the thin, non-executing glue in
``framework_v2.v2_workflow``: the paper-facing SPECIFY->...->VALIDATE plan
transitions, coverage/convergence/efficiency evidence records, the executor
adapter map, and the final-evidence bridge.  Nothing here runs Teacher
inference, Student training, MD, DFT, replay, or supercell jobs.
"""
import pytest

from framework_v2.contracts import PartitionRole
from framework_v2.error_tracking import (
    ErrorLedger,
    RawEfficiencyRecord,
    RegionErrorRecord,
)
from framework_v2.protected_eligibility import derive_training_eligible_candidates
from framework_v2.v2_sampling import RegionClosureState
from framework_v2.v2_workflow import (
    ConvergenceKind,
    CoverageEvidenceRecord,
    ExecutorEndpointStatus,
    FinalValidationStatus,
    V2WorkflowStatus,
    V2WorkflowStep,
    advance_v2_workflow_plan,
    build_efficiency_evidence_bundle,
    build_final_target_validation_request,
    build_final_target_validation_result,
    build_v2_final_evidence_record,
    build_v2_workflow_plan,
    convergence_from_existing_artifact,
    coverage_signals_from_records,
    default_v2_executor_adapter_map,
    latest_region_states,
    record_final_analysis_evidence,
    record_final_validation_result,
    resolve_executor_dispatch,
    route_after_tracking,
)


def _closed_record(region_id, iteration=0, state=RegionClosureState.CLOSED):
    return RegionErrorRecord(
        campaign_id="run",
        iteration=iteration,
        region_id=region_id,
        region_membership_sha256="membership",
        state=state,
        failure_reason=(
            "failed required criteria: force" if state == RegionClosureState.RECOVER else ""
        ),
        efficiency=RawEfficiencyRecord(
            teacher_evaluations=5,
            measurement_provenance={"teacher_evaluations": ["mock"]},
        ),
    )


# --- coverage evidence ------------------------------------------------------


def test_coverage_evidence_maps_to_namespaced_signal():
    record = CoverageEvidenceRecord(
        record_id="c",
        campaign_id="run",
        iteration=0,
        structural_region_manifest_sha256="regions",
        population_sha256="population",
        region_id="A",
        metric_name="descriptor_saturation",
        measured_value=0.82,
        definition="fraction of marginal novelty curve below bound knee",
        aggregation="per-region",
        provenance=["coverage_artifact"],
    )
    assert coverage_signals_from_records([record], region_id="A") == {
        "coverage.descriptor_saturation": 0.82
    }


def test_missing_coverage_value_stays_none_not_zero():
    record = CoverageEvidenceRecord(
        record_id="c",
        campaign_id="run",
        iteration=0,
        structural_region_manifest_sha256="regions",
        region_id="A",
        metric_name="descriptor_saturation",
        measured_value=None,
        definition="defined but not evaluated",
        aggregation="per-region",
        provenance=["coverage_artifact"],
    )
    signals = coverage_signals_from_records([record], region_id="A")
    assert signals["coverage.descriptor_saturation"] is None


# --- efficiency / pareto ----------------------------------------------------


def test_pareto_records_keep_raw_dimensions_without_scalar_cost():
    ledger = ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[_closed_record("A")],
    )
    bundle = build_efficiency_evidence_bundle(bundle_id="eff", ledger=ledger)
    row = bundle.pareto_records[0]
    assert row.teacher_evaluations is not None
    assert not hasattr(row, "total_cost")


def test_convergence_evidence_distinguishes_kinds():
    kinds = set()
    for kind in (
        ConvergenceKind.TRAINING,
        ConvergenceKind.RECOVERY_ITERATION,
        ConvergenceKind.CAMPAIGN_CLOSURE,
    ):
        rec = convergence_from_existing_artifact(
            record_id=f"cv_{kind.value}",
            campaign_id="run",
            iteration=0,
            kind=kind,
            artifact_sha256="artifact",
            epochs=100,
            continuation_rounds=0,
            stopping_criterion_sha256="policy",
            stopping_reason="patience reached",
            converged=True,
            provenance=["convergence_report"],
        )
        kinds.add(rec.kind)
    assert kinds == {
        ConvergenceKind.TRAINING,
        ConvergenceKind.RECOVERY_ITERATION,
        ConvergenceKind.CAMPAIGN_CLOSURE,
    }


def test_convergence_unknown_requires_reason_but_measured_zero_is_allowed():
    measured_zero = convergence_from_existing_artifact(
        record_id="cv0",
        campaign_id="run",
        iteration=0,
        kind=ConvergenceKind.TRAINING,
        artifact_sha256="artifact",
        epochs=0,
        continuation_rounds=0,
        stopping_criterion_sha256="policy",
        stopping_reason="already converged at init",
        converged=True,
        provenance=["convergence_report"],
    )
    assert measured_zero.epochs == 0
    assert measured_zero.unresolved_reason == ""

    unknown = convergence_from_existing_artifact(
        record_id="cv1",
        campaign_id="run",
        iteration=0,
        kind=ConvergenceKind.TRAINING,
        artifact_sha256="artifact",
        epochs=None,
        continuation_rounds=None,
        stopping_criterion_sha256=None,
        stopping_reason="",
        converged=None,
        provenance=["convergence_report"],
    )
    assert unknown.converged is None
    assert unknown.unresolved_reason


# --- workflow transitions ---------------------------------------------------


def test_workflow_cannot_advance_from_specify_if_operationalization_pending():
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="c", human_target_sha256="target"
    )
    next_plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TargetOperationalizationPending",
        unresolved_status=V2WorkflowStatus.SCIENTIFIC_INPUT_REQUIRED,
        unresolved_reason="broad target family pending observable selection",
    )
    assert next_plan.current_step == V2WorkflowStep.SPECIFY
    assert next_plan.status == V2WorkflowStatus.SCIENTIFIC_INPUT_REQUIRED


def test_workflow_cannot_advance_from_curate_if_selection_budget_insufficient():
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="c", human_target_sha256="target"
    ).model_copy(
        update={
            "current_step": V2WorkflowStep.CURATE,
            "status": V2WorkflowStatus.READY_TO_PLAN,
        }
    )
    next_plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="SelectionBudgetInsufficient",
        produced_artifact_sha256="selection",
    )
    assert next_plan.status == V2WorkflowStatus.EVIDENCE_INCOMPLETE
    assert next_plan.current_step == V2WorkflowStep.CURATE


def _distill_plan():
    return build_v2_workflow_plan(
        plan_id="p", campaign_id="c", human_target_sha256="target"
    ).model_copy(
        update={
            "current_step": V2WorkflowStep.DISTILL,
            "status": V2WorkflowStatus.READY_TO_EXECUTE_EXTERNAL_ACTION,
        }
    )


def test_distill_emitting_request_does_not_make_evaluation_possible():
    # Emitting a TeacherLabelingRequest must NOT advance to TRACK; it only puts
    # the plan into a waiting state on the DISTILL step.  Evaluation (TRACK) is
    # impossible until the real Student artifact exists.
    plan = _distill_plan()
    waiting = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TeacherLabelingRequest",
        produced_artifact_sha256="labeling",
    )
    assert waiting.current_step == V2WorkflowStep.DISTILL
    assert waiting.status == V2WorkflowStatus.WAITING_FOR_ARTIFACT
    assert waiting.execution_allowed is False
    assert waiting.teacher_labeling_request_sha256 == "labeling"
    assert waiting.student_committee_sha256 is None


def test_distill_waits_while_student_artifact_absent():
    # The whole intermediate chain (labels -> dataset) keeps the plan waiting on
    # DISTILL; only the Student committee artifact permits TRACK.
    plan = _distill_plan()
    after_request = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TeacherLabelingRequest",
        produced_artifact_sha256="labeling",
    )
    after_labels = advance_v2_workflow_plan(
        after_request,
        produced_artifact_type="TeacherLabelArtifact",
        produced_artifact_sha256="labels",
    )
    assert after_labels.current_step == V2WorkflowStep.DISTILL
    assert after_labels.status == V2WorkflowStatus.WAITING_FOR_ARTIFACT
    after_dataset = advance_v2_workflow_plan(
        after_labels,
        produced_artifact_type="TrainingDatasetArtifact",
        produced_artifact_sha256="dataset",
    )
    assert after_dataset.current_step == V2WorkflowStep.DISTILL
    assert after_dataset.status == V2WorkflowStatus.WAITING_FOR_ARTIFACT


def test_distill_advances_to_track_only_on_student_ready_artifact():
    plan = _distill_plan()
    p = advance_v2_workflow_plan(
        plan, produced_artifact_type="TeacherLabelingRequest", produced_artifact_sha256="labeling"
    )
    p = advance_v2_workflow_plan(
        p, produced_artifact_type="TeacherLabelArtifact", produced_artifact_sha256="labels"
    )
    p = advance_v2_workflow_plan(
        p, produced_artifact_type="TrainingDatasetArtifact", produced_artifact_sha256="dataset"
    )
    tracked = advance_v2_workflow_plan(
        p, produced_artifact_type="StudentCommitteeArtifact", produced_artifact_sha256="committee"
    )
    assert tracked.current_step == V2WorkflowStep.TRACK
    assert tracked.status == V2WorkflowStatus.READY_TO_EXECUTE_EXTERNAL_ACTION
    assert tracked.student_committee_sha256 == "committee"


def test_distill_out_of_order_artifacts_fail_closed():
    plan = _distill_plan()
    # dataset before labels
    with pytest.raises(ValueError, match="completed Teacher labels"):
        advance_v2_workflow_plan(
            plan,
            produced_artifact_type="TrainingDatasetArtifact",
            produced_artifact_sha256="dataset",
        )
    # student committee before dataset
    with pytest.raises(ValueError, match="updated training dataset"):
        advance_v2_workflow_plan(
            plan,
            produced_artifact_type="StudentCommitteeArtifact",
            produced_artifact_sha256="committee",
        )
    # label artifact before a labeling request
    with pytest.raises(ValueError, match="emitted labeling request"):
        advance_v2_workflow_plan(
            plan,
            produced_artifact_type="TeacherLabelArtifact",
            produced_artifact_sha256="labels",
        )


def test_recover_reentry_clears_intermediate_distill_artifacts():
    # A prior iteration's Student artifact must not leak forward; re-entering
    # DISTILL via RECOVER re-gates through the staged chain.
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="run", human_target_sha256="target"
    ).model_copy(
        update={
            "current_step": V2WorkflowStep.RECOVER,
            "status": V2WorkflowStatus.RECOVERY_REQUIRED,
            "teacher_labeling_request_sha256": "old_req",
            "teacher_label_artifact_sha256": "old_labels",
            "training_dataset_artifact_sha256": "old_dataset",
            "student_committee_sha256": "old_committee",
        }
    )
    reentered = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="RecoveryExecutionBundle",
        produced_artifact_sha256="bundle",
    )
    assert reentered.current_step == V2WorkflowStep.DISTILL
    assert reentered.teacher_labeling_request_sha256 is None
    assert reentered.teacher_label_artifact_sha256 is None
    assert reentered.training_dataset_artifact_sha256 is None
    assert reentered.student_committee_sha256 is None


def test_track_with_recover_region_routes_to_recover():
    ledger = ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[
            _closed_record("A"),
            _closed_record("B", state=RegionClosureState.RECOVER),
        ],
    )
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="run", human_target_sha256="target"
    ).model_copy(update={"current_step": V2WorkflowStep.TRACK})
    routed = route_after_tracking(plan, ledger=ledger, required_region_ids=["A", "B"])
    assert routed.current_step == V2WorkflowStep.RECOVER
    assert routed.status == V2WorkflowStatus.RECOVERY_REQUIRED


def test_evidence_not_evaluated_does_not_trigger_recovery():
    ledger = ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[
            _closed_record("A"),
            _closed_record("B", state=RegionClosureState.EVIDENCE_NOT_EVALUATED),
        ],
    )
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="run", human_target_sha256="target"
    ).model_copy(update={"current_step": V2WorkflowStep.TRACK})
    routed = route_after_tracking(plan, ledger=ledger, required_region_ids=["A", "B"])
    assert routed.current_step == V2WorkflowStep.TRACK
    assert routed.status == V2WorkflowStatus.EVIDENCE_INCOMPLETE


def test_recover_waits_for_staged_artifact_before_redistill():
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="run", human_target_sha256="target"
    ).model_copy(
        update={
            "current_step": V2WorkflowStep.RECOVER,
            "status": V2WorkflowStatus.RECOVERY_REQUIRED,
        }
    )
    with pytest.raises(ValueError, match="RECOVER expects RecoveryExecutionBundle"):
        advance_v2_workflow_plan(
            plan,
            produced_artifact_type="RedistillationRequest",
            produced_artifact_sha256="premature",
        )
    staged = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="RecoveryExecutionBundle",
        produced_artifact_sha256="bundle",
    )
    assert staged.current_step == V2WorkflowStep.DISTILL
    assert staged.recovery_bundle_sha256 == "bundle"


# --- final validation gate --------------------------------------------------


def test_validate_requires_all_latest_regions_closed_not_no_deficient_only():
    ledger = ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[
            _closed_record("A"),
            _closed_record("B", state=RegionClosureState.EVIDENCE_NOT_EVALUATED),
        ],
    )
    # deficient_regions() is empty (no RECOVER) yet validation must still fail.
    assert ledger.deficient_regions(0) == []
    with pytest.raises(ValueError, match="B=EVIDENCE_NOT_EVALUATED"):
        build_final_target_validation_request(
            request_id="req",
            campaign_id="run",
            ledger=ledger,
            required_region_ids=["A", "B"],
            target_validation_contract_sha256="tvc",
            final_student_committee_sha256="committee",
            protected_evaluation_population_sha256="protected",
            structural_region_manifest_sha256="regions",
            evaluation_binding_sha256="binding",
        )


def test_all_closed_permits_final_target_validation_request():
    ledger = ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[_closed_record("A"), _closed_record("B")],
    )
    req = build_final_target_validation_request(
        request_id="req",
        campaign_id="run",
        ledger=ledger,
        required_region_ids=["A", "B"],
        target_validation_contract_sha256="tvc",
        final_student_committee_sha256="committee",
        protected_evaluation_population_sha256="protected",
        structural_region_manifest_sha256="regions",
        evaluation_binding_sha256="binding",
    )
    assert req.error_ledger_sha256 == ledger.content_sha256()


def _closed_ledger():
    return ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[_closed_record("A"), _closed_record("B")],
    )


def _validation_request(ledger):
    return build_final_target_validation_request(
        request_id="req",
        campaign_id="run",
        ledger=ledger,
        required_region_ids=["A", "B"],
        target_validation_contract_sha256="tvc",
        final_student_committee_sha256="committee",
        protected_evaluation_population_sha256="protected",
        structural_region_manifest_sha256="regions",
        evaluation_binding_sha256="binding",
    )


def _pass_result(req):
    return build_final_target_validation_result(
        result_id="result",
        request=req,
        status=FinalValidationStatus.PASS,
        fe067_evidence_sha256="fe067_evidence",
        validation_provenance=["fe067_report"],
        per_region_status={"A": "PASS", "B": "PASS"},
    )


def test_final_validation_result_binds_request_and_identities():
    req = _validation_request(_closed_ledger())
    result = _pass_result(req)
    assert result.request_sha256 == req.content_sha256()
    assert result.final_student_committee_sha256 == "committee"
    assert result.protected_evaluation_population_sha256 == "protected"
    assert result.status == FinalValidationStatus.PASS


def test_final_evidence_binds_result_identity_and_invariants():
    ledger = _closed_ledger()
    req = _validation_request(ledger)
    result = _pass_result(req)
    bundle = build_efficiency_evidence_bundle(bundle_id="eff", ledger=ledger)
    convergence = [
        convergence_from_existing_artifact(
            record_id="cv",
            campaign_id="run",
            iteration=0,
            kind=ConvergenceKind.CAMPAIGN_CLOSURE,
            artifact_sha256="artifact",
            epochs=100,
            continuation_rounds=1,
            stopping_criterion_sha256="policy",
            stopping_reason="patience reached",
            converged=True,
            provenance=["convergence_report"],
        )
    ]
    final = build_v2_final_evidence_record(
        record_id="final",
        campaign_id="run",
        human_target_sha256="target",
        target_operationalization_sha256="operationalization",
        final_validation_result=result,
        final_analysis_evidence_sha256="fe068_final_analysis",
        ledger=ledger,
        efficiency_bundle=bundle,
        convergence_records=convergence,
        recovery_history_sha256s=["recovery_bundle"],
        protected_population_sha256s=["protected"],
        fe067_bridge_ref="evaluate_observable",
        fe068_bridge_ref="validate_run_summary_report",
    )
    assert final.teacher_frozen is True
    assert final.new_dft_performed is False
    # binds the RESULT identity, not only the request
    assert final.final_validation_result_sha256 == result.content_sha256()
    assert final.target_validation_request_sha256 == req.content_sha256()
    assert final.final_validation_status == FinalValidationStatus.PASS
    assert final.final_analysis_evidence_sha256 == "fe068_final_analysis"
    assert final.recovery_history_sha256s == ["recovery_bundle"]
    assert final.protected_population_sha256s == ["protected"]


def test_final_evidence_rejects_non_pass_result():
    req = _validation_request(_closed_ledger())
    fail = build_final_target_validation_result(
        result_id="result",
        request=req,
        status=FinalValidationStatus.FAIL,
        fe067_evidence_sha256="fe067_evidence",
        validation_provenance=["fe067_report"],
    )
    ledger = _closed_ledger()
    bundle = build_efficiency_evidence_bundle(bundle_id="eff", ledger=ledger)
    with pytest.raises(ValueError, match="PASS final validation result"):
        build_v2_final_evidence_record(
            record_id="final",
            campaign_id="run",
            human_target_sha256="target",
            target_operationalization_sha256="operationalization",
            final_validation_result=fail,
            final_analysis_evidence_sha256="fe068_final_analysis",
            ledger=ledger,
            efficiency_bundle=bundle,
            convergence_records=[],
            recovery_history_sha256s=[],
            protected_population_sha256s=["protected"],
        )


# --- final validation RESULT gate (workflow COMPLETE) -----------------------


def _validate_plan_with_request(req):
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="run", human_target_sha256="target"
    ).model_copy(
        update={
            "current_step": V2WorkflowStep.VALIDATE,
            "status": V2WorkflowStatus.VALIDATION_READY,
        }
    )
    return advance_v2_workflow_plan(
        plan,
        produced_artifact_type="FinalTargetValidationRequest",
        produced_artifact_sha256=req.content_sha256(),
    )


def test_validation_request_alone_does_not_complete():
    req = _validation_request(_closed_ledger())
    plan = _validate_plan_with_request(req)
    assert plan.status == V2WorkflowStatus.WAITING_FOR_ARTIFACT
    assert plan.current_step == V2WorkflowStep.VALIDATE
    # attempting to jump straight to a final evidence record must fail closed
    with pytest.raises(ValueError, match="PASS FinalTargetValidationResult"):
        advance_v2_workflow_plan(
            plan,
            produced_artifact_type="V2FinalEvidenceRecord",
            produced_artifact_sha256="evidence",
        )


def test_fail_result_does_not_complete():
    req = _validation_request(_closed_ledger())
    plan = _validate_plan_with_request(req)
    fail = build_final_target_validation_result(
        result_id="result",
        request=req,
        status=FinalValidationStatus.FAIL,
        fe067_evidence_sha256="fe067_evidence",
        validation_provenance=["fe067_report"],
    )
    after = record_final_validation_result(plan, fail)
    assert after.status == V2WorkflowStatus.EVIDENCE_INCOMPLETE
    assert after.final_validation_status == FinalValidationStatus.FAIL
    with pytest.raises(ValueError, match="PASS FinalTargetValidationResult"):
        advance_v2_workflow_plan(
            after,
            produced_artifact_type="V2FinalEvidenceRecord",
            produced_artifact_sha256="evidence",
        )


def test_unresolved_result_routes_to_human_input_not_complete():
    req = _validation_request(_closed_ledger())
    plan = _validate_plan_with_request(req)
    unresolved = build_final_target_validation_result(
        result_id="result",
        request=req,
        status=FinalValidationStatus.UNRESOLVED,
        fe067_evidence_sha256="fe067_evidence",
        validation_provenance=["fe067_report"],
        unresolved_reason="observable estimator returned NaN on region B",
    )
    after = record_final_validation_result(plan, unresolved)
    assert after.status == V2WorkflowStatus.SCIENTIFIC_INPUT_REQUIRED
    with pytest.raises(ValueError, match="PASS FinalTargetValidationResult"):
        advance_v2_workflow_plan(
            after,
            produced_artifact_type="V2FinalEvidenceRecord",
            produced_artifact_sha256="evidence",
        )


def test_pass_result_without_final_analysis_evidence_does_not_complete():
    req = _validation_request(_closed_ledger())
    plan = _validate_plan_with_request(req)
    after = record_final_validation_result(plan, _pass_result(req))
    assert after.status == V2WorkflowStatus.VALIDATION_READY
    # FE-068 deterministic analysis evidence is part of the completion contract
    with pytest.raises(ValueError, match="deterministic final analysis"):
        advance_v2_workflow_plan(
            after,
            produced_artifact_type="V2FinalEvidenceRecord",
            produced_artifact_sha256="evidence",
        )


def test_pass_result_plus_final_analysis_evidence_completes():
    req = _validation_request(_closed_ledger())
    plan = _validate_plan_with_request(req)
    after = record_final_validation_result(plan, _pass_result(req))
    after = record_final_analysis_evidence(
        after, final_analysis_evidence_sha256="fe068_final_analysis"
    )
    complete = advance_v2_workflow_plan(
        after,
        produced_artifact_type="V2FinalEvidenceRecord",
        produced_artifact_sha256="evidence",
    )
    assert complete.status == V2WorkflowStatus.COMPLETE
    assert complete.final_evidence_sha256 == "evidence"


def test_result_must_reference_emitted_request():
    req = _validation_request(_closed_ledger())
    plan = _validate_plan_with_request(req)
    # a result whose request_sha256 does not match the emitted request
    other_ledger = ErrorLedger(
        ledger_id="ledger2",
        campaign_id="run",
        records=[_closed_record("A"), _closed_record("B"), _closed_record("C")],
    )
    other_req = build_final_target_validation_request(
        request_id="req2",
        campaign_id="run",
        ledger=other_ledger,
        required_region_ids=["A", "B", "C"],
        target_validation_contract_sha256="tvc",
        final_student_committee_sha256="committee",
        protected_evaluation_population_sha256="protected",
        structural_region_manifest_sha256="regions",
        evaluation_binding_sha256="binding",
    )
    mismatched = _pass_result(other_req)
    with pytest.raises(ValueError, match="does not reference the emitted request"):
        record_final_validation_result(plan, mismatched)


# --- executor adapter map ---------------------------------------------------


def test_executor_adapter_map_reuses_confirmed_fe067_fe068_surfaces():
    amap = default_v2_executor_adapter_map()
    by_type = {e.v2_request_type: e for e in amap.endpoints}
    fe067 = by_type["FinalTargetValidationRequest"]
    fe068 = by_type["V2FinalEvidenceRecord"]
    assert fe067.status == ExecutorEndpointStatus.CONFIRMED_REUSABLE
    assert fe067.known_existing_symbol == "evaluate_observable"
    assert fe068.status == ExecutorEndpointStatus.CONFIRMED_REUSABLE
    assert fe068.known_existing_symbol == "validate_run_summary_report"


def test_execution_chain_endpoints_bound_to_real_symbols_not_none():
    # No endpoint may claim "no source confirmation needed" while leaving the
    # symbol None; every execution-chain request maps to a real action.
    amap = default_v2_executor_adapter_map()
    by_type = {e.v2_request_type: e for e in amap.endpoints}
    expected = {
        "TeacherLabelingRequest": ("_exec_label_with_teacher", "label_with_teacher", True),
        "TrainingDatasetUpdateRequest": ("_exec_generate_group_split", "generate_group_split", False),
        "RedistillationRequest": ("_exec_train_committee", "train_committee", True),
        "NextEvaluationRequest": ("_exec_evaluate_committee", "evaluate_heldout_fidelity", True),
    }
    for req_type, (symbol, action, approval) in expected.items():
        ep = by_type[req_type]
        assert ep.known_existing_symbol == symbol
        assert ep.registered_action_type == action
        assert ep.requires_hpc_approval is approval
        assert ep.status == ExecutorEndpointStatus.REUSABLE_VIA_THIN_ADAPTER
        assert ep.known_existing_symbol is not None


def test_resolve_executor_dispatch_is_non_executing_and_matches_live_registry():
    from runtimes.pydantic_ai.executors import build_executor_registry

    registry = build_executor_registry()
    amap = default_v2_executor_adapter_map()
    proposal = resolve_executor_dispatch(
        amap,
        "TeacherLabelingRequest",
        identity_provenance=["candidate_pop_sha", "teacher_sha"],
        action_registry=registry,
    )
    assert proposal.executes_immediately is False
    assert proposal.registered_action_type == "label_with_teacher"
    assert proposal.executor_symbol == "_exec_label_with_teacher"
    assert proposal.requires_human_approval is True
    assert proposal.approval_boundary == "costly_teacher_labeling"
    # the registry actually knows this action
    assert "label_with_teacher" in registry

    deterministic = resolve_executor_dispatch(
        amap,
        "TrainingDatasetUpdateRequest",
        identity_provenance=["prior_train_sha", "labels_sha"],
        action_registry=registry,
    )
    assert deterministic.requires_human_approval is False
    assert deterministic.approval_boundary is None


# --- protected identity never enters training (H06 bridge) ------------------


def test_no_protected_identity_appears_in_training_artifacts():
    # H06 authoritative eligibility: protected/test frames are rejected, never
    # silently dropped, so no training artifact can carry a protected id.
    with pytest.raises(ValueError, match="ineligible candidates"):
        derive_training_eligible_candidates(
            candidate_ids=["train_a", "protected_b"],
            candidate_roles={
                "train_a": PartitionRole.TRAIN,
                "protected_b": PartitionRole.BLIND_TEST,
            },
            protected_ids={"protected_b"},
            training_split_sha256="split",
            expected_training_split_sha256="split",
            protected_evidence_sha256="protected",
            expected_protected_evidence_sha256="protected",
        )
    clean = derive_training_eligible_candidates(
        candidate_ids=["train_a"],
        candidate_roles={"train_a": PartitionRole.TRAIN},
        protected_ids={"protected_b"},
        training_split_sha256="split",
        expected_training_split_sha256="split",
        protected_evidence_sha256="protected",
        expected_protected_evidence_sha256="protected",
    )
    assert clean.eligible_candidate_ids == ["train_a"]
    assert "protected_b" not in clean.eligible_candidate_ids


def test_latest_region_states_uses_most_recent_iteration():
    ledger = ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[
            _closed_record("A", iteration=0, state=RegionClosureState.RECOVER),
            _closed_record("A", iteration=1, state=RegionClosureState.CLOSED),
        ],
    )
    latest = latest_region_states(ledger, ["A", "B"])
    assert latest["A"] == RegionClosureState.CLOSED
    assert latest["B"] == RegionClosureState.EVIDENCE_NOT_EVALUATED
