"""V2-H08: synthetic property-guided control loop with mocks only.

Demonstrates framework cohesion end-to-end without touching a Teacher, Student,
DFT, or MD backend: human target -> operationalization -> validation contract ->
structural regions -> protected evaluation binding -> region metrics ->
error ledger -> region-directed recovery execution graph -> re-evaluation closes
the deficient region.  All hashes/artifacts are mock strings.
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
    SignalCriterion,
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
