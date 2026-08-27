"""V2-H04: evaluation-population -> region binding, per-region aggregation, and
automatic ErrorLedger population.

Region metrics are computed only over the bound evaluation population (training
frames are ignored), reference channels may not be mixed, and closure distinguishes
target failure / missing evidence / unbound criterion.  Efficiency numbers are
optional: a measured value must carry provenance (unknown != zero).
"""
import pytest

from framework_v2.error_tracking import (
    ErrorLedger,
    RawEfficiencyRecord,
    build_error_ledger_iteration,
)
from framework_v2.region_evaluation import (
    EvaluationPopulationRegionBinding,
    FrameEvaluationRecord,
    aggregate_region_metrics,
    bind_evaluation_population_to_regions,
)
from framework_v2.structural_regions import (
    StructuralRegion,
    StructuralRegionManifest,
    StructuralRegionProviderType,
)
from framework_v2.v2_sampling import (
    CriterionBindingStatus,
    CriterionComparator,
    CriterionRole,
    RegionClosureState,
    RegionStoppingPolicy,
    SignalCriterion,
)


def _manifest():
    regions = [
        StructuralRegion(
            region_id=r,
            provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
            membership_provenance=["m"],
            population_size=2,
            membership_manifest_sha256="m",
        )
        for r in ("r1", "r2")
    ]
    return StructuralRegionManifest(
        manifest_id="mani",
        provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
        regions=regions,
        frame_to_region={"e1": "r1", "e2": "r1", "e3": "r2", "e4": "r2"},
        source_sha256="src",
    )


def _binding(frame_ids=("e1", "e2", "e3", "e4")):
    return bind_evaluation_population_to_regions(
        region_manifest=_manifest(),
        evaluation_frame_ids=list(frame_ids),
        evaluation_population_sha256="evalpop",
        binding_id="bind",
    )


def _frame(fid, region_channel="student_vs_teacher", **kw):
    return FrameEvaluationRecord(frame_id=fid, n_atoms=10, reference_channel=region_channel, **kw)


def _bound(signal, value, comparator=CriterionComparator.LE):
    return SignalCriterion(
        signal=signal,
        role=CriterionRole.SCIENTIFIC_REQUIRED,
        binding_status=CriterionBindingStatus.BOUND,
        comparator=comparator,
        value=value,
        provenance=["human_contract"],
    )


def _eff():
    return RawEfficiencyRecord(
        added_structures=4,
        measurement_provenance={"added_structures": ["selection_log"]},
    )


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def test_aggregation_ignores_training_only_members():
    binding = _binding()
    records = [
        _frame("e1", energy_error_eV=0.1, force_component_errors=[0.1]),
        _frame("e2", energy_error_eV=0.1, force_component_errors=[0.1]),
        _frame("e3", energy_error_eV=0.1, force_component_errors=[0.1]),
        _frame("e4", energy_error_eV=0.1, force_component_errors=[0.1]),
        # training-only frame not in binding -> ignored
        _frame("train_only", energy_error_eV=99.0, force_component_errors=[99.0]),
    ]
    regions = aggregate_region_metrics(binding, records)
    assert {r.region_id for r in regions} == {"r1", "r2"}
    assert all(r.n_frames == 2 for r in regions)
    assert all(r.energy_rmse_meV_per_atom < 50 for r in regions)


def test_required_eval_frame_missing_fails():
    binding = _binding()
    with pytest.raises(ValueError, match="missing"):
        aggregate_region_metrics(binding, [_frame("e1", energy_error_eV=0.1)])


def test_required_region_no_evidence_fails():
    binding = bind_evaluation_population_to_regions(
        region_manifest=_manifest(),
        evaluation_frame_ids=["e1", "e2"],
        evaluation_population_sha256="evalpop",
        binding_id="bind",
        required_region_ids=["r1", "r2"],  # r2 declared required but no frames bound
    )
    with pytest.raises(ValueError, match="no evaluation evidence"):
        aggregate_region_metrics(
            binding,
            [_frame("e1", energy_error_eV=0.1), _frame("e2", energy_error_eV=0.1)],
        )


def test_mixed_reference_channel_fails():
    binding = _binding(frame_ids=("e1", "e2"))
    with pytest.raises(ValueError, match="mixed reference channels"):
        aggregate_region_metrics(
            binding,
            [
                _frame("e1", region_channel="student_vs_teacher", energy_error_eV=0.1),
                _frame("e2", region_channel="student_vs_dft", energy_error_eV=0.1),
            ],
        )


def test_binding_rejects_unassigned_evaluation_frame():
    with pytest.raises(ValueError, match="no structural-region assignment"):
        bind_evaluation_population_to_regions(
            region_manifest=_manifest(),
            evaluation_frame_ids=["e1", "unknown_frame"],
            evaluation_population_sha256="evalpop",
            binding_id="bind",
        )


# --------------------------------------------------------------------------
# ledger population + closure routing
# --------------------------------------------------------------------------
def test_target_failure_triggers_recover_even_when_energy_force_pass():
    binding = _binding()
    records = [
        _frame("e1", energy_error_eV=0.001, force_component_errors=[0.01], target_metrics={"target_error": 0.5}),
        _frame("e2", energy_error_eV=0.001, force_component_errors=[0.01], target_metrics={"target_error": 0.5}),
        _frame("e3", energy_error_eV=0.001, force_component_errors=[0.01], target_metrics={"target_error": 0.01}),
        _frame("e4", energy_error_eV=0.001, force_component_errors=[0.01], target_metrics={"target_error": 0.01}),
    ]
    regions = aggregate_region_metrics(binding, records)
    policy = RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            _bound("energy.rmse_meV_per_atom", 50.0),
            _bound("force.component_rmse_eV_per_angstrom", 1.0),
            _bound("target.target_error", 0.05),
        ],
    )
    ledger = build_error_ledger_iteration(
        ledger=ErrorLedger(ledger_id="l", campaign_id="c"),
        iteration=0,
        evaluation_binding=binding,
        region_evaluations=regions,
        closure_policy=policy,
        target_validation_sha256="tv",
        training_population_sha256="train",
        efficiency=_eff(),
    )
    states = {r.region_id: r.state for r in ledger.records_for_iteration(0)}
    assert states["r1"] == RegionClosureState.RECOVER  # target failed, E/F passed
    assert states["r2"] == RegionClosureState.CLOSED
    assert ledger.deficient_regions(0) == ["r1"]


def test_missing_measurement_gives_evidence_not_evaluated():
    binding = _binding()
    records = [
        _frame("e1", energy_error_eV=0.001, force_component_errors=[0.01]),
        _frame("e2", energy_error_eV=0.001, force_component_errors=[0.01]),
        _frame("e3", energy_error_eV=0.001, force_component_errors=[0.01]),
        _frame("e4", energy_error_eV=0.001, force_component_errors=[0.01]),
    ]
    regions = aggregate_region_metrics(binding, records)
    policy = RegionStoppingPolicy(
        policy_id="p",
        criteria=[_bound("target.target_error", 0.05)],  # never measured
    )
    ledger = build_error_ledger_iteration(
        ledger=ErrorLedger(ledger_id="l", campaign_id="c"),
        iteration=0,
        evaluation_binding=binding,
        region_evaluations=regions,
        closure_policy=policy,
        target_validation_sha256=None,
        training_population_sha256="train",
        efficiency=_eff(),
    )
    states = {r.state for r in ledger.records_for_iteration(0)}
    assert states == {RegionClosureState.EVIDENCE_NOT_EVALUATED}
    assert ledger.deficient_regions(0) == []  # missing evidence is not RECOVER


def test_unbound_criterion_gives_human_scientific_input_required():
    binding = _binding()
    records = [_frame(f, energy_error_eV=0.001) for f in ("e1", "e2", "e3", "e4")]
    regions = aggregate_region_metrics(binding, records)
    policy = RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            SignalCriterion(
                signal="target.target_error",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.UNBOUND,
                unbound_reason="human has not set target threshold",
            )
        ],
    )
    ledger = build_error_ledger_iteration(
        ledger=ErrorLedger(ledger_id="l", campaign_id="c"),
        iteration=0,
        evaluation_binding=binding,
        region_evaluations=regions,
        closure_policy=policy,
        target_validation_sha256=None,
        training_population_sha256="train",
        efficiency=_eff(),
    )
    states = {r.state for r in ledger.records_for_iteration(0)}
    assert states == {RegionClosureState.HUMAN_SCIENTIFIC_INPUT_REQUIRED}
    assert ledger.deficient_regions(0) == []


# --------------------------------------------------------------------------
# efficiency provenance
# --------------------------------------------------------------------------
def test_measured_efficiency_requires_provenance():
    with pytest.raises(ValueError, match="lacks provenance"):
        RawEfficiencyRecord(added_structures=2)  # measured but no provenance


def test_unknown_efficiency_is_not_zero():
    rec = RawEfficiencyRecord()
    assert rec.added_structures is None
    assert rec.wall_time_seconds is None
    # a fully-unknown record is valid: unknown != zero
    assert rec.measurement_provenance == {}
