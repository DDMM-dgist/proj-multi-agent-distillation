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
    V2WorkflowStatus,
    V2WorkflowStep,
    advance_v2_workflow_plan,
    build_efficiency_evidence_bundle,
    build_final_target_validation_request,
    build_v2_final_evidence_record,
    build_v2_workflow_plan,
    convergence_from_existing_artifact,
    coverage_signals_from_records,
    default_v2_executor_adapter_map,
    latest_region_states,
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


def test_distill_emits_external_request_but_does_not_execute():
    plan = build_v2_workflow_plan(
        plan_id="p", campaign_id="c", human_target_sha256="target"
    ).model_copy(
        update={
            "current_step": V2WorkflowStep.DISTILL,
            "status": V2WorkflowStatus.READY_TO_EXECUTE_EXTERNAL_ACTION,
        }
    )
    next_plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TeacherLabelingRequest",
        produced_artifact_sha256="labeling",
    )
    assert next_plan.current_step == V2WorkflowStep.TRACK
    assert next_plan.status == V2WorkflowStatus.WAITING_FOR_ARTIFACT
    assert next_plan.execution_allowed is False
    assert next_plan.teacher_labeling_request_sha256 == "labeling"


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


def test_final_evidence_includes_efficiency_convergence_recovery_and_invariants():
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
        final_validation_request=req,
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
    assert final.efficiency_bundle_sha256
    assert final.convergence_evidence_sha256s
    assert final.recovery_history_sha256s == ["recovery_bundle"]
    assert final.protected_population_sha256s == ["protected"]


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
