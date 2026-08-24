"""Regression tests for Framework V2 ConvergencePolicy classifier.

Covers Section 20 CASE C: max-epoch reached while validation is still
improving must classify as NOT_CONVERGED.
"""
from __future__ import annotations

import pytest

from framework_v2 import ConvergencePolicy, ProvenanceClass
from framework_v2.convergence import (
    CONVERGED_AT_MAX,
    CONVERGED_EARLY,
    INSUFFICIENT_DATA,
    NOT_CONVERGED,
    build_convergence_report,
    classify_seed_convergence,
    convergence_gate_ok,
)


def _policy(**overrides):
    kwargs = dict(
        policy_id="test-policy",
        trailing_window=50,
        projection_window=50,
        min_relative_improvement=0.05,
        boundary_tolerance=5,
        metrics=["valid_energy_rmse", "valid_force_rmse"],
        provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
        provenance_source="test fixture",
    )
    kwargs.update(overrides)
    return ConvergencePolicy(**kwargs)


def _fabricate_log(*, total_epoch: int, ep_series: list[tuple[float, float, float, float]],
                   best_at: int | None = None) -> str:
    """Fabricate a SIMPLE-NN-format LOG.

    ``ep_series`` is a list of ``(train_e, valid_e, train_f, valid_f)``
    tuples, one per emitted epoch (typically every 10 epochs in R31
    but the classifier accepts any spacing).
    """
    lines = [f"Total traning epoch: {total_epoch}"]
    if best_at is not None:
        lines.append(f"Best loss lammps potential written at {best_at} epoch")
    for i, (te, ve, tf, vf) in enumerate(ep_series):
        ep = (i + 1) * 10  # 10-epoch stride like R31
        lines.append(
            f"Epoch     {ep} E RMSE(T V) {te:.4e} {ve:.4e} "
            f"F RMSE(T V) {tf:.4e} {vf:.4e} learning_rate: 1.0000e-04"
        )
    return "\n".join(lines) + "\n"


class TestClassifierOnFabricatedLogs:
    def test_case_c_not_converged_r31_style(self):
        """CASE C: R31-style seed-202634 pattern -- best_epoch=200,
        epochs_requested=200, valid-E slope still strongly negative
        (~-0.02/epoch) at 200. Must classify NOT_CONVERGED.
        """
        # 20 epochs of "every 10" = up to epoch 200. Val-E declines
        # from 5.0 to 2.5 linearly; val-F declines from 5.0 to 2.2.
        series = [
            (te, ve, tf, vf)
            for te, ve, tf, vf in [
                (5.0 - 0.13*i, 5.0 - 0.13*i,   # train/valid E
                 5.0 - 0.14*i, 5.0 - 0.14*i)   # train/valid F
                for i in range(20)
            ]
        ]
        log = _fabricate_log(total_epoch=200, ep_series=series, best_at=200)
        report = classify_seed_convergence(log, _policy())
        assert report["status"] == NOT_CONVERGED, report
        assert report["at_boundary"] is True
        assert report["best_epoch"] == 200
        assert report["epochs_requested"] == 200
        # Both metrics should be flagged as meaningfully improving
        assert report["per_metric"]["valid_energy_rmse"]["meaningfully_improving"] is True
        assert report["per_metric"]["valid_force_rmse"]["meaningfully_improving"] is True

    def test_converged_at_max_when_slopes_are_flat(self):
        """Boundary reached, but trailing slope over last 50 epochs is
        essentially zero -> CONVERGED_AT_MAX."""
        series = []
        # First 15 points strongly descend; last 5 points nearly flat.
        for i in range(15):
            v = 5.0 - 0.28 * i  # from 5.0 down to ~0.8
            series.append((v, v, v + 0.5, v + 0.5))
        last = series[-1][1]
        for _ in range(5):
            series.append((last, last, last + 0.5, last + 0.5))
        log = _fabricate_log(total_epoch=200, ep_series=series, best_at=200)
        report = classify_seed_convergence(log, _policy())
        assert report["status"] == CONVERGED_AT_MAX, report
        assert report["at_boundary"] is True

    def test_converged_early_when_best_epoch_is_well_before_budget(self):
        """best_epoch < epochs_requested - boundary_tolerance -> CONVERGED_EARLY."""
        series = [(1.0, 1.0, 1.5, 1.5)] * 20  # 200 epochs of flat metric
        log = _fabricate_log(total_epoch=200, ep_series=series, best_at=50)
        report = classify_seed_convergence(log, _policy())
        assert report["status"] == CONVERGED_EARLY, report
        assert report["at_boundary"] is False
        assert report["best_epoch"] == 50

    def test_insufficient_data_when_log_has_no_epochs(self):
        log = "Total traning epoch: 200\nBest loss lammps potential written at 200 epoch\n"
        report = classify_seed_convergence(log, _policy())
        assert report["status"] == INSUFFICIENT_DATA

    def test_boundary_tolerance_respected(self):
        # best_epoch=196, requested=200, tolerance=5 -> at boundary
        series = [(v, v, v, v) for v in [5.0 - 0.02*i for i in range(20)]]
        log = _fabricate_log(total_epoch=200, ep_series=series, best_at=196)
        report = classify_seed_convergence(log, _policy(boundary_tolerance=5))
        assert report["at_boundary"] is True
        # With strict tolerance=1, best_epoch=196 is well before boundary
        # (196 < 200-1=199)
        report_strict = classify_seed_convergence(log, _policy(boundary_tolerance=1))
        assert report_strict["at_boundary"] is False


class TestCommitteeReport:
    def test_worst_seed_dominates_committee_status(self):
        # seed A: NOT_CONVERGED; seed B: CONVERGED_EARLY. Committee -> NOT_CONVERGED.
        series_a = [(v, v, v, v) for v in [5.0 - 0.13*i for i in range(20)]]
        series_b = [(1.0, 1.0, 1.5, 1.5)] * 20
        logs = {
            "seed-A": _fabricate_log(total_epoch=200, ep_series=series_a, best_at=200),
            "seed-B": _fabricate_log(total_epoch=200, ep_series=series_b, best_at=50),
        }
        report = build_convergence_report(_policy(), seed_logs=logs)
        assert report["committee_status"] == NOT_CONVERGED
        assert convergence_gate_ok(report) is False

    def test_all_early_converged_committee_passes(self):
        series = [(1.0, 1.0, 1.5, 1.5)] * 20
        logs = {
            "seed-A": _fabricate_log(total_epoch=200, ep_series=series, best_at=50),
            "seed-B": _fabricate_log(total_epoch=200, ep_series=series, best_at=60),
        }
        report = build_convergence_report(_policy(), seed_logs=logs)
        assert report["committee_status"] == CONVERGED_EARLY
        assert convergence_gate_ok(report) is True

    def test_report_echoes_policy(self):
        policy = _policy(trailing_window=42)
        logs = {"seed-X": _fabricate_log(
            total_epoch=200,
            ep_series=[(1.0, 1.0, 1.5, 1.5)] * 20,
            best_at=50,
        )}
        report = build_convergence_report(policy, seed_logs=logs)
        assert report["policy"]["trailing_window"] == 42
        assert report["policy_sha256"] == policy.content_sha256()
