"""V2-H10: efficiency evidence keeps raw dimensions; unknown != zero.

Focused checks on the Pareto/efficiency bundle adapter: per-iteration
incremental fields are preserved as recorded (never summed into a scalar),
cumulative fields are echoed as-measured, unknown stays ``None`` (distinct from
a measured ``0``), and no weighted total-cost scalar exists on any record.
"""
from framework_v2.error_tracking import (
    ErrorLedger,
    RawEfficiencyRecord,
    RegionErrorRecord,
)
from framework_v2.v2_sampling import RegionClosureState
from framework_v2.v2_workflow import (
    ConvergenceKind,
    build_efficiency_evidence_bundle,
    convergence_from_existing_artifact,
    pareto_records_from_ledger,
)


def _record(region_id, iteration, efficiency):
    return RegionErrorRecord(
        campaign_id="run",
        iteration=iteration,
        region_id=region_id,
        region_membership_sha256="membership",
        state=RegionClosureState.CLOSED,
        efficiency=efficiency,
    )


def _ledger():
    return ErrorLedger(
        ledger_id="ledger",
        campaign_id="run",
        records=[
            _record(
                "A",
                0,
                RawEfficiencyRecord(
                    added_structures=8,
                    cumulative_training_structures=8,
                    teacher_evaluations=8,
                    measurement_provenance={
                        "added_structures": ["iter0"],
                        "cumulative_training_structures": ["iter0"],
                        "teacher_evaluations": ["iter0"],
                    },
                ),
            ),
            _record(
                "A",
                1,
                RawEfficiencyRecord(
                    added_structures=4,
                    cumulative_training_structures=12,
                    teacher_evaluations=4,
                    measurement_provenance={
                        "added_structures": ["iter1"],
                        "cumulative_training_structures": ["iter1"],
                        "teacher_evaluations": ["iter1"],
                    },
                ),
            ),
        ],
    )


def test_incremental_fields_remain_per_iteration():
    rows = pareto_records_from_ledger(_ledger())
    assert [r.added_structures for r in rows] == [8, 4]


def test_cumulative_fields_are_not_summed():
    rows = pareto_records_from_ledger(_ledger())
    # cumulative echoes the measured cumulative, not sum-of-incrementals (which
    # would double-count to 8 + 12 = 20).
    assert [r.cumulative_training_structures for r in rows] == [8, 12]


def test_convergence_unknown_vs_measured_zero_epochs():
    measured_zero = convergence_from_existing_artifact(
        record_id="cv0",
        campaign_id="run",
        iteration=0,
        kind=ConvergenceKind.TRAINING,
        artifact_sha256="artifact",
        epochs=0,
        continuation_rounds=0,
        stopping_criterion_sha256="policy",
        stopping_reason="converged at init",
        converged=True,
        provenance=["report"],
    )
    unknown = convergence_from_existing_artifact(
        record_id="cv1",
        campaign_id="run",
        iteration=0,
        kind=ConvergenceKind.RECOVERY_ITERATION,
        artifact_sha256="artifact",
        epochs=None,
        continuation_rounds=None,
        stopping_criterion_sha256=None,
        stopping_reason="",
        converged=None,
        provenance=["report"],
    )
    assert measured_zero.epochs == 0
    assert unknown.epochs is None
    assert unknown.unresolved_reason


def test_no_scalar_cost_field_on_bundle_or_records():
    bundle = build_efficiency_evidence_bundle(bundle_id="eff", ledger=_ledger())
    assert not hasattr(bundle, "total_cost")
    assert not hasattr(bundle, "weighted_cost")
    for row in bundle.pareto_records:
        assert not hasattr(row, "total_cost")
        assert not hasattr(row, "weighted_cost")
