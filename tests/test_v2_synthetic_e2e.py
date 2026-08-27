"""V2-H08: synthetic property-guided control loop with mocks only.

Demonstrates framework cohesion end-to-end without touching a Teacher, Student,
DFT, or MD backend: human target -> operationalization -> validation contract ->
structural regions -> protected evaluation binding -> region metrics ->
error ledger -> region-directed recovery execution graph -> re-evaluation closes
the deficient region.  All hashes/artifacts are mock strings.

H10 upgrade: the same mock scenario now also drives a ``V2WorkflowPlan`` through
its paper-facing transitions
(SPECIFY->DISCOVER->CURATE->DISTILL->TRACK->RECOVER->DISTILL->TRACK->VALIDATE->
COMPLETE) using only the H10 transition helpers -- no test-only semantics and no
executor execution.
"""
from framework_v2.error_tracking import (
    ErrorLedger,
    RawEfficiencyRecord,
    build_error_ledger_iteration,
)
from framework_v2.property_targets import (
    HumanTargetPropertyContract,
    TargetPropertyFamily,
    build_target_validation_contract,
    default_observable_registry,
    operationalize_target_request,
)
from framework_v2.region_evaluation import (
    FrameEvaluationRecord,
    aggregate_region_metrics,
    bind_evaluation_population_to_regions,
)
from framework_v2.region_recovery import (
    RecoveryExecutionState,
    attach_evaluation_artifact,
    attach_label_artifact,
    attach_student_artifact,
    attach_updated_dataset,
    build_planned_recovery,
    plan_region_recovery,
)
from framework_v2.structural_regions import explicit_regions_from_membership
from framework_v2.structural_representation import (
    CompositionRepresentationAdapter,
    StructureRecord,
)
from framework_v2.v2_sampling import (
    CriterionBindingStatus,
    CriterionComparator,
    CriterionRole,
    RegionClosureState,
    RegionStoppingPolicy,
    SamplerKind,
    SamplerRequest,
    SignalCriterion,
    sample_candidates,
)
from framework_v2.v2_workflow import (
    ConvergenceKind,
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
    record_final_analysis_evidence,
    record_final_validation_result,
    route_after_tracking,
)


def test_property_guided_v2_mock_control_loop():
    target = HumanTargetPropertyContract(
        contract_id="target",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["coordination"],
    )
    op = operationalize_target_request(target)
    validation = build_target_validation_contract(op, default_observable_registry())

    assert validation.target_property_family == TargetPropertyFamily.STRUCTURAL
    assert [b.observable.name for b in validation.observables] == ["coordination"]

    # H10: start the paper-facing workflow plan at SPECIFY.
    plan = build_v2_workflow_plan(
        plan_id="plan",
        campaign_id="mock",
        human_target_sha256=target.content_sha256(),
    )
    assert plan.current_step == V2WorkflowStep.SPECIFY
    # SPECIFY -> DISCOVER once the target is operationalized.
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TargetOperationalizationResult",
        produced_artifact_sha256=op.content_sha256(),
    )
    assert plan.current_step == V2WorkflowStep.DISCOVER

    structures = [
        StructureRecord(structure_id="a_train", species_counts={"Si": 1, "O": 2}),
        StructureRecord(structure_id="a_eval", species_counts={"Si": 1, "O": 2}),
        StructureRecord(structure_id="b_train", species_counts={"Si": 1, "O": 1}),
        StructureRecord(structure_id="b_recovery", species_counts={"Si": 2, "O": 1}),
        StructureRecord(structure_id="b_eval", species_counts={"Si": 1, "O": 1}),
    ]
    rep = CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        structures, representation_id="rep"
    )
    regions = explicit_regions_from_membership(
        manifest_id="regions",
        frame_to_region={
            "a_train": "A",
            "a_eval": "A",
            "b_train": "B",
            "b_recovery": "B",
            "b_eval": "B",
        },
        source_sha256="source",
        membership_manifest_sha256="membership",
    )
    # DISCOVER -> CURATE once the structural region manifest exists.
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="StructuralRegionManifest",
        produced_artifact_sha256=regions.content_sha256(),
    )
    assert plan.current_step == V2WorkflowStep.CURATE

    # CURATE: real structural-stratified selection over the eligible training
    # candidates; protected eval frames are excluded from selection.
    sampler_result = sample_candidates(
        SamplerRequest(
            sampler=SamplerKind.FPS,
            candidate_ids=["a_train", "b_train", "b_recovery"],
            n_select=2,
            protected_candidate_ids=["a_eval", "b_eval"],
        ),
        representation=rep,
    )
    assert "a_eval" not in sampler_result.selected_ids
    assert "b_eval" not in sampler_result.selected_ids
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="SamplerResult",
        produced_artifact_sha256=sampler_result.content_sha256(),
    )
    assert plan.current_step == V2WorkflowStep.DISTILL

    # DISTILL emits an external Teacher-labeling request and *waits*; the plan
    # never executes Teacher inference itself and does not reach TRACK until the
    # real Student committee artifact exists.
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TeacherLabelingRequest",
        produced_artifact_sha256="labeling_request_0",
    )
    assert plan.current_step == V2WorkflowStep.DISTILL
    assert plan.status == V2WorkflowStatus.WAITING_FOR_ARTIFACT
    assert plan.execution_allowed is False
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TeacherLabelArtifact",
        produced_artifact_sha256="labels_0",
    )
    assert plan.current_step == V2WorkflowStep.DISTILL
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TrainingDatasetArtifact",
        produced_artifact_sha256="dataset_0",
    )
    assert plan.current_step == V2WorkflowStep.DISTILL
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="StudentCommitteeArtifact",
        produced_artifact_sha256="committee_0",
    )
    assert plan.current_step == V2WorkflowStep.TRACK
    assert plan.execution_allowed is False

    binding0 = bind_evaluation_population_to_regions(
        region_manifest=regions,
        evaluation_frame_ids=["a_eval", "b_eval"],
        evaluation_population_sha256="protected_eval",
        binding_id="binding0",
        required_region_ids=["A", "B"],
    )

    policy = RegionStoppingPolicy(
        policy_id="closure",
        criteria=[
            SignalCriterion(
                signal="force.component_rmse_eV_per_angstrom",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.BOUND,
                comparator=CriterionComparator.LE,
                value=0.30,
                units="eV/Angstrom",
                provenance=["synthetic_pre_result"],
            ),
            SignalCriterion(
                signal="target.coordination_error",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.BOUND,
                comparator=CriterionComparator.LE,
                value=0.10,
                units="fraction",
                provenance=["synthetic_pre_result"],
            ),
        ],
    )

    rev0 = aggregate_region_metrics(
        binding0,
        [
            FrameEvaluationRecord(
                frame_id="a_eval",
                n_atoms=3,
                reference_channel="student_vs_teacher",
                force_component_errors=[0.1, 0.1, 0.1],
                target_metrics={"coordination_error": 0.05},
            ),
            FrameEvaluationRecord(
                frame_id="b_eval",
                n_atoms=2,
                reference_channel="student_vs_teacher",
                force_component_errors=[0.1, 0.1, 0.1],
                target_metrics={"coordination_error": 0.20},
            ),
        ],
    )

    ledger0 = build_error_ledger_iteration(
        ledger=ErrorLedger(ledger_id="ledger", campaign_id="mock"),
        iteration=0,
        evaluation_binding=binding0,
        region_evaluations=rev0,
        closure_policy=policy,
        target_validation_sha256=validation.content_sha256(),
        training_population_sha256="train0",
        efficiency=RawEfficiencyRecord(
            selected_structures=2,
            cumulative_training_structures=2,
            measurement_provenance={
                "selected_structures": ["mock"],
                "cumulative_training_structures": ["mock"],
            },
        ),
    )
    assert ledger0.deficient_regions(0) == ["B"]

    # TRACK: region B is an evaluated RECOVER state -> plan routes to RECOVER.
    plan = route_after_tracking(
        plan, ledger=ledger0, required_region_ids=["A", "B"]
    )
    assert plan.current_step == V2WorkflowStep.RECOVER
    assert plan.status == V2WorkflowStatus.RECOVERY_REQUIRED

    recovery = plan_region_recovery(
        ledger0,
        iteration=0,
        eligible_training_candidate_ids=["b_recovery"],
        protected_candidate_ids=["a_eval", "b_eval"],
        sampler=SamplerKind.UNCERTAINTY_DIVERSITY,
        n_select=1,
        rationale="B failed required target coordination metric",
    )
    # Real API: selection is explicit; b_recovery is the only eligible candidate
    # and protected eval frames are excluded by the planner + bundle guard.
    planned = build_planned_recovery(
        recovery,
        selected_candidate_ids=["b_recovery"],
        teacher_identity_sha256="teacher",
        candidate_population_sha256="source",
        access_partition_sha256="partition",
        bundle_id="bundle0",
        expected_label_artifact="train_update.json",
    )
    assert planned.state == RecoveryExecutionState.PLANNED
    assert planned.selected_candidate_ids == ["b_recovery"]
    assert "a_eval" not in planned.selected_candidate_ids
    assert "b_eval" not in planned.selected_candidate_ids

    labels = attach_label_artifact(
        planned,
        teacher_label_artifact_sha256="labels1",
        split_lineage_sha256="split",
        expected_output_artifact="train_update.json",
        prior_training_population_sha256="train0",
    )
    dataset = attach_updated_dataset(
        labels,
        updated_training_population_sha256="train1",
        student_recipe_sha256="recipe",
        expected_output_artifact="student_committee.json",
    )
    student = attach_student_artifact(
        dataset,
        student_committee_sha256="student1",
        protected_population_sha256="protected_eval",
        evaluation_binding_sha256=binding0.content_sha256(),
        target_validation_contract_sha256=validation.content_sha256(),
        expected_output_artifact="eval1.json",
    )
    done = attach_evaluation_artifact(student, evaluation_artifact_sha256="eval1")
    assert done.state == RecoveryExecutionState.EVALUATION_READY

    # RECOVER waits for the staged recovery bundle; only then does it hand back
    # to DISTILL for redistillation.
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="RecoveryExecutionBundle",
        produced_artifact_sha256=done.content_sha256(),
    )
    assert plan.current_step == V2WorkflowStep.DISTILL
    assert plan.recovery_bundle_sha256 == done.content_sha256()
    # re-entering DISTILL cleared the prior iteration's intermediate artifacts
    assert plan.student_committee_sha256 is None
    # redistillation re-gates through the full artifact chain; only the new
    # Student committee artifact returns to TRACK.
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="RedistillationRequest",
        produced_artifact_sha256="redistill_request_1",
    )
    assert plan.current_step == V2WorkflowStep.DISTILL
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TeacherLabelArtifact",
        produced_artifact_sha256="labels_1",
    )
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="TrainingDatasetArtifact",
        produced_artifact_sha256="dataset_1",
    )
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="StudentCommitteeArtifact",
        produced_artifact_sha256="committee_1",
    )
    assert plan.current_step == V2WorkflowStep.TRACK

    rev1 = aggregate_region_metrics(
        binding0,
        [
            FrameEvaluationRecord(
                frame_id="a_eval",
                n_atoms=3,
                reference_channel="student_vs_teacher",
                force_component_errors=[0.1],
                target_metrics={"coordination_error": 0.05},
            ),
            FrameEvaluationRecord(
                frame_id="b_eval",
                n_atoms=2,
                reference_channel="student_vs_teacher",
                force_component_errors=[0.1],
                target_metrics={"coordination_error": 0.05},
            ),
        ],
    )
    ledger1 = build_error_ledger_iteration(
        ledger=ledger0,
        iteration=1,
        evaluation_binding=binding0,
        region_evaluations=rev1,
        closure_policy=policy,
        target_validation_sha256=validation.content_sha256(),
        training_population_sha256="train1",
        efficiency=RawEfficiencyRecord(
            added_structures=1,
            cumulative_training_structures=3,
            recovery_iterations=1,
            measurement_provenance={
                "added_structures": ["mock"],
                "cumulative_training_structures": ["mock"],
                "recovery_iterations": ["mock"],
            },
        ),
    )
    assert ledger1.deficient_regions(1) == []
    assert all(
        r.state == RegionClosureState.CLOSED for r in ledger1.records_for_iteration(1)
    )

    # TRACK: every latest required region is CLOSED -> route to VALIDATE.
    plan = route_after_tracking(
        plan, ledger=ledger1, required_region_ids=["A", "B"]
    )
    assert plan.current_step == V2WorkflowStep.VALIDATE
    assert plan.status == V2WorkflowStatus.VALIDATION_READY

    # VALIDATE: build the final target-validation request (gated on all-CLOSED,
    # not merely no-deficient) and record the plan is waiting for its evidence.
    final_request = build_final_target_validation_request(
        request_id="final_request",
        campaign_id="mock",
        ledger=ledger1,
        required_region_ids=["A", "B"],
        target_validation_contract_sha256=validation.content_sha256(),
        final_student_committee_sha256="student1",
        protected_evaluation_population_sha256="protected_eval",
        structural_region_manifest_sha256=regions.content_sha256(),
        evaluation_binding_sha256=binding0.content_sha256(),
    )
    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="FinalTargetValidationRequest",
        produced_artifact_sha256=final_request.content_sha256(),
    )
    assert plan.status == V2WorkflowStatus.WAITING_FOR_ARTIFACT

    # A request alone cannot COMPLETE; a deterministic FE-067 RESULT (PASS) plus
    # deterministic FE-068 final-analysis evidence are required first.
    final_result = build_final_target_validation_result(
        result_id="final_result",
        request=final_request,
        status=FinalValidationStatus.PASS,
        fe067_evidence_sha256="fe067_evidence",
        validation_provenance=[ledger1.content_sha256()],
        per_region_status={"A": "PASS", "B": "PASS"},
    )
    plan = record_final_validation_result(plan, final_result)
    assert plan.final_validation_status == FinalValidationStatus.PASS
    assert plan.status == V2WorkflowStatus.VALIDATION_READY
    plan = record_final_analysis_evidence(
        plan, final_analysis_evidence_sha256="fe068_final_analysis"
    )

    efficiency_bundle = build_efficiency_evidence_bundle(
        bundle_id="efficiency", ledger=ledger1
    )
    convergence = [
        convergence_from_existing_artifact(
            record_id="campaign_closure",
            campaign_id="mock",
            iteration=1,
            kind=ConvergenceKind.CAMPAIGN_CLOSURE,
            artifact_sha256=ledger1.content_sha256(),
            epochs=100,
            continuation_rounds=1,
            stopping_criterion_sha256="policy",
            stopping_reason="all required regions CLOSED",
            converged=True,
            provenance=[ledger1.content_sha256()],
        )
    ]
    final_evidence = build_v2_final_evidence_record(
        record_id="final_evidence",
        campaign_id="mock",
        human_target_sha256=target.content_sha256(),
        target_operationalization_sha256=op.content_sha256(),
        final_validation_result=final_result,
        final_analysis_evidence_sha256="fe068_final_analysis",
        ledger=ledger1,
        efficiency_bundle=efficiency_bundle,
        convergence_records=convergence,
        recovery_history_sha256s=[done.content_sha256()],
        protected_population_sha256s=["protected_eval"],
        fe067_bridge_ref="evaluate_observable",
        fe068_bridge_ref="validate_run_summary_report",
    )
    assert final_evidence.teacher_frozen is True
    assert final_evidence.new_dft_performed is False
    assert final_evidence.final_validation_result_sha256 == final_result.content_sha256()

    plan = advance_v2_workflow_plan(
        plan,
        produced_artifact_type="V2FinalEvidenceRecord",
        produced_artifact_sha256=final_evidence.content_sha256(),
    )
    assert plan.status == V2WorkflowStatus.COMPLETE
    assert plan.final_evidence_sha256 == final_evidence.content_sha256()
