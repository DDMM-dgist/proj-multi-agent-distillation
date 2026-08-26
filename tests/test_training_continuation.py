"""Regression tests for the autonomous *bounded* training-continuation layer
(:mod:`framework_v2.training_continuation`).

Covers the required behaviours:

  1. boundary + still-improving  -> automatic bounded continuation plan
  2. converged                   -> no recovery
  3. divergence / NaN            -> NOT routed to "train longer" (human)
  4. missing resumable checkpoint -> fail closed (human)
  5. mixed committee             -> only non-converged seeds continue
  6. repeated continuation stays bounded (rounds + cumulative epochs)
  7. budget exhaustion           -> RECOVERY_BUDGET_EXHAUSTED (human)
  8. no arbitrary human/LLM epoch number is ever required (quantum derived)
  9. the convergence gate is unchanged and cannot be bypassed

Plus: the policy is versioned + hash-bound, and its validators fail closed.
"""
from __future__ import annotations

import math

import pytest

from framework_v2 import ProvenanceClass
from framework_v2.convergence import (
    NOT_CONVERGED,
    build_convergence_report,
    convergence_gate_ok,
    default_training_convergence_policy,
)
from framework_v2.states import SemanticState
from framework_v2.training_continuation import (
    RECOVERY_BUDGET_EXHAUSTED_CODE,
    CheckpointInfo,
    ContinuationHistory,
    ContinuationOutcome,
    QuantumSource,
    SeedContinuationReason,
    TrainingContinuationPolicy,
    assess_seed_continuation,
    checkpoint_is_resumable,
    continuation_quantum,
    default_training_continuation_policy,
    detect_numerical_health,
    plan_committee_continuation,
)


# --------------------------------------------------------------------------
# Helpers: fabricate SIMPLE-NN-format LOGs and convergence reports.
# --------------------------------------------------------------------------
def _log(*, total_epoch: int, series, best_at=None) -> str:
    """``series`` = list of (train_e, valid_e, train_f, valid_f), one per
    emitted epoch on a 10-epoch stride (matching R31 LOGs)."""
    lines = [f"Total traning epoch: {total_epoch}"]
    if best_at is not None:
        lines.append(f"Best loss lammps potential written at {best_at} epoch")
    for i, (te, ve, tf, vf) in enumerate(series):
        ep = (i + 1) * 10
        lines.append(
            f"Epoch     {ep} E RMSE(T V) {te:.4e} {ve:.4e} "
            f"F RMSE(T V) {tf:.4e} {vf:.4e} learning_rate: 1.0000e-04"
        )
    return "\n".join(lines) + "\n"


def _improving_series(n=20, start=5.0, drop=0.13):
    """Strongly, linearly declining validation error (NOT_CONVERGED-shaped)."""
    return [(start - drop * i,) * 4 for i in range(n)]


def _flat_series(n=20, level=1.0):
    """Essentially flat validation error (CONVERGED_AT_MAX-shaped)."""
    return [(level, level, level, level) for _ in range(n)]


def _resumable_ckpt(seed=1, epoch=200):
    return CheckpointInfo(
        seed=seed, path=f"/run/seed-{seed}/checkpoint_latest.pth.tar",
        sha256=f"sha-{seed}", epoch=epoch, resumable=True)


def _report(seed_logs):
    conv_policy = default_training_convergence_policy()
    return build_convergence_report(conv_policy, seed_logs=seed_logs), conv_policy


# --------------------------------------------------------------------------
# Policy contract: versioned + hash-bound + fail-closed validators.
# --------------------------------------------------------------------------
class TestPolicyContract:
    def test_default_policy_is_versioned_and_hash_bound(self):
        p = default_training_continuation_policy()
        assert p.schema_version >= 1
        assert p.policy_id == "framework-default-training-continuation-v1"
        # Deterministic content hash, stable across rebuilds.
        assert p.content_sha256() == default_training_continuation_policy().content_sha256()

    def test_changing_a_bound_changes_the_hash(self):
        a = default_training_continuation_policy()
        b = a.model_copy(update={"max_continuation_rounds": a.max_continuation_rounds + 1})
        assert a.content_sha256() != b.content_sha256()

    def test_policy_is_frozen(self):
        p = default_training_continuation_policy()
        with pytest.raises(Exception):
            p.max_continuation_rounds = 99  # type: ignore[misc]

    def test_eligible_states_must_be_recovery_bearing(self):
        with pytest.raises(Exception):
            TrainingContinuationPolicy(
                policy_id="bad", max_continuation_rounds=1,
                max_cumulative_continuation_epochs=50,
                eligible_states=["PASS"],  # not recovery-bearing
                divergence_relative_tolerance=0.1,
                provenance_class=ProvenanceClass.FRAMEWORK_CONSTRAINT,
                provenance_source="x")

    def test_only_escalate_to_human_exhaustion_supported(self):
        with pytest.raises(Exception):
            TrainingContinuationPolicy(
                policy_id="bad", max_continuation_rounds=1,
                max_cumulative_continuation_epochs=50,
                eligible_states=[NOT_CONVERGED],
                divergence_relative_tolerance=0.1,
                exhaustion_behavior="silently_stop",
                provenance_class=ProvenanceClass.FRAMEWORK_CONSTRAINT,
                provenance_source="x")

    def test_budget_exhausted_failure_code_registered(self):
        from workflow.recovery_taxonomy import domain_of, resolve_failure_code
        fc = resolve_failure_code(RECOVERY_BUDGET_EXHAUSTED_CODE)
        assert fc.code == RECOVERY_BUDGET_EXHAUSTED_CODE
        assert domain_of(RECOVERY_BUDGET_EXHAUSTED_CODE) == "operational"


# --------------------------------------------------------------------------
# Continuation quantum is DERIVED, never authored (Requirement 8).
# --------------------------------------------------------------------------
class TestDerivedQuantum:
    def test_quantum_equals_projection_window(self):
        cont = default_training_continuation_policy()
        conv = default_training_convergence_policy()
        assert cont.quantum_source == QuantumSource.PROJECTION_WINDOW
        assert continuation_quantum(cont, conv) == conv.projection_window

    def test_quantum_from_trailing_window_when_selected(self):
        conv = default_training_convergence_policy()
        cont = default_training_continuation_policy().model_copy(
            update={"quantum_source": QuantumSource.TRAILING_WINDOW})
        assert continuation_quantum(cont, conv) == conv.trailing_window

    def test_policy_stores_no_raw_target_epoch_number(self):
        """The policy never encodes a campaign epoch count (250/300/...)."""
        dumped = default_training_continuation_policy().model_dump(mode="json")
        for v in dumped.values():
            assert v not in (250, 300, 350, 400)


# --------------------------------------------------------------------------
# Numerical health is a distinct, independent check.
# --------------------------------------------------------------------------
class TestNumericalHealth:
    def test_nan_is_unhealthy(self):
        log = _log(total_epoch=200,
                   series=[(1.0, float("nan"), 1.0, 1.0)] + _flat_series(19))
        ok, reason = detect_numerical_health(
            log, ["valid_energy_rmse"], divergence_relative_tolerance=0.1)
        assert ok is False and reason.startswith("nan_or_inf")

    def test_divergence_is_unhealthy(self):
        # valid-E: min ~1.0 early, blows up to 5.0 at the end (>10% over min).
        series = [(1.0, 1.0, 1.0, 1.0)] * 10 + [(1.0, 5.0, 1.0, 1.0)]
        log = _log(total_epoch=200, series=series)
        ok, reason = detect_numerical_health(
            log, ["valid_energy_rmse"], divergence_relative_tolerance=0.1)
        assert ok is False and reason.startswith("diverged")

    def test_healthy_improving_is_healthy(self):
        log = _log(total_epoch=200, series=_improving_series())
        ok, reason = detect_numerical_health(
            log, ["valid_energy_rmse", "valid_force_rmse"],
            divergence_relative_tolerance=0.1)
        assert ok is True and reason == "healthy"


# --------------------------------------------------------------------------
# checkpoint resumability (pure).
# --------------------------------------------------------------------------
class TestCheckpointResumable:
    def test_all_keys_present_is_resumable(self):
        ok, reason = checkpoint_is_resumable(
            ["model", "optimizer", "epoch", "scale_factor", "pca", "loss"])
        assert ok is True and reason == "resumable"

    def test_missing_optimizer_is_not_resumable(self):
        ok, reason = checkpoint_is_resumable(
            ["model", "epoch", "scale_factor", "pca"])
        assert ok is False and "optimizer" in reason


# --------------------------------------------------------------------------
# Committee-level decisions.
# --------------------------------------------------------------------------
class TestCommitteeContinuation:
    def _plan(self, seed_logs, checkpoints, *, cont=None, history=None):
        report, conv = _report(seed_logs)
        cont = cont or default_training_continuation_policy()
        history = history or ContinuationHistory(base_boundary=200, rounds_used=0)
        decision = plan_committee_continuation(
            report, seed_logs, checkpoints, cont, conv, history,
            triggering_convergence_report_sha256="conv-report-sha")
        return report, conv, decision

    # (1) boundary + improving -> automatic continuation plan
    def test_boundary_and_improving_yields_continue(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        report, conv, d = self._plan(logs, cks)

        assert report["committee_status"] == NOT_CONVERGED
        assert d.outcome == ContinuationOutcome.CONTINUE
        assert [dv.seed for dv in d.directives] == ["seed-1"]
        # Target is DERIVED: boundary(200) + projection_window(50).
        assert d.quantum_epochs == conv.projection_window
        assert d.target_epoch == 200 + conv.projection_window
        assert d.total_epoch_override == 200 + conv.projection_window
        dv = d.directives[0]
        assert dv.start_epoch == 200
        assert dv.source_checkpoint_sha256 == "sha-1"

    # (1) provenance is recorded for the round (Requirement 6)
    def test_continue_records_full_provenance(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        _, conv, d = self._plan(logs, cks)
        prov = d.provenance
        assert prov is not None
        assert prov.round_index == 1
        assert prov.continuation_policy_sha256 == default_training_continuation_policy().content_sha256()
        assert prov.convergence_policy_sha256 == conv.content_sha256()
        assert prov.triggering_convergence_report_sha256 == "conv-report-sha"
        assert prov.quantum_epochs == conv.projection_window
        assert prov.cumulative_continuation_epochs_before == 0
        assert prov.cumulative_continuation_epochs_after == conv.projection_window
        assert prov.per_seed[0]["source_checkpoint_sha256"] == "sha-1"

    # (1) the RecoveryPlan uses the canonical typed NOT_CONVERGED route
    def test_continue_builds_typed_not_converged_recovery_plan(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        _, _, d = self._plan(logs, cks)
        plan = d.recovery_plan
        assert plan is not None
        assert plan.failure_state == SemanticState.NOT_CONVERGED
        assert plan.failure_code == "training_instability"
        assert plan.failed_stage == "training"
        # The plan mandates re-running the UNCHANGED gate.
        assert any("convergence" in c.lower() for c in plan.revalidation_criteria)

    # (2) converged -> no recovery
    def test_converged_yields_no_recovery(self):
        logs = {"seed-1": _log(total_epoch=200, series=_flat_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        report, _, d = self._plan(logs, cks)
        assert report["committee_status"] != NOT_CONVERGED
        assert d.outcome == ContinuationOutcome.NO_RECOVERY_CONVERGED
        assert d.directives == []
        assert d.recovery_plan is None

    # (3) divergence/NaN -> NOT routed to train-longer
    def test_nan_forces_human_escalation_not_train_longer(self):
        # Improving trailing window (NOT_CONVERGED) but a NaN early in the LOG.
        series = ([(5.0, float("nan"), 5.0, 5.0)]
                  + [(5.0 - 0.13 * i,) * 4 for i in range(1, 20)])
        logs = {"seed-1": _log(total_epoch=200, series=series, best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        report, _, d = self._plan(logs, cks)
        assert report["committee_status"] == NOT_CONVERGED
        assert d.outcome == ContinuationOutcome.HUMAN_ESCALATION_REQUIRED
        assert d.directives == []
        assert d.per_seed["seed-1"]["reason"] == SeedContinuationReason.NUMERICAL_FAILURE.value

    # (4) missing resumable checkpoint -> fail closed
    def test_missing_checkpoint_forces_human_escalation(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        report, _, d = self._plan(logs, checkpoints={})  # no checkpoint at all
        assert report["committee_status"] == NOT_CONVERGED
        assert d.outcome == ContinuationOutcome.HUMAN_ESCALATION_REQUIRED
        assert d.per_seed["seed-1"]["reason"] == SeedContinuationReason.NO_RESUMABLE_CHECKPOINT.value

    def test_nonresumable_checkpoint_forces_human_escalation(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": CheckpointInfo(
            seed=1, path="/x", sha256="s", epoch=200,
            resumable=False, resumable_reason="missing_checkpoint_keys:optimizer")}
        _, _, d = self._plan(logs, cks)
        assert d.outcome == ContinuationOutcome.HUMAN_ESCALATION_REQUIRED
        assert d.per_seed["seed-1"]["reason"] == SeedContinuationReason.NO_RESUMABLE_CHECKPOINT.value

    # (5) mixed committee -> only non-converged seeds continue
    def test_mixed_committee_continues_only_non_converged(self):
        logs = {
            "seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200),
            "seed-2": _log(total_epoch=200, series=_flat_series(), best_at=200),
            "seed-3": _log(total_epoch=200, series=_improving_series(), best_at=200),
        }
        cks = {"seed-1": _resumable_ckpt(1), "seed-2": _resumable_ckpt(2),
               "seed-3": _resumable_ckpt(3)}
        _, conv, d = self._plan(logs, cks)
        assert d.outcome == ContinuationOutcome.CONTINUE
        assert sorted(dv.seed for dv in d.directives) == ["seed-1", "seed-3"]
        assert d.preserved_seeds == ["seed-2"]  # converged member untouched
        # A single override is valid because both continued seeds share a target.
        assert d.total_epoch_override == 200 + conv.projection_window

    # (6)+(7) repeated continuation stays bounded; exhaustion -> escalation
    def test_rounds_budget_exhausted(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        # rounds_used already at the cap (default max 3).
        history = ContinuationHistory(base_boundary=200, rounds_used=3)
        _, _, d = self._plan(logs, cks, history=history)
        assert d.outcome == ContinuationOutcome.RECOVERY_BUDGET_EXHAUSTED
        assert d.directives == []

    def test_last_round_within_rounds_budget_still_continues(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        history = ContinuationHistory(base_boundary=200, rounds_used=2)  # 2 < 3
        _, _, d = self._plan(logs, cks, history=history)
        assert d.outcome == ContinuationOutcome.CONTINUE

    def test_cumulative_epoch_budget_exhausted(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        # quantum is 50; cap the cumulative budget below one quantum.
        cont = default_training_continuation_policy().model_copy(
            update={"max_cumulative_continuation_epochs": 40})
        _, _, d = self._plan(logs, cks, cont=cont)
        assert d.outcome == ContinuationOutcome.RECOVERY_BUDGET_EXHAUSTED

    def test_repeated_continuation_is_bounded_across_rounds(self):
        """Simulate advancing boundaries: each round derives the next target and
        eventually the cumulative cap forces escalation -- never unbounded."""
        cont = default_training_continuation_policy()  # max_cum=150, quantum=50
        conv = default_training_convergence_policy()
        outcomes = []
        for boundary in (200, 250, 300, 350):
            logs = {"seed-1": _log(total_epoch=boundary,
                                   series=_improving_series(), best_at=boundary)}
            cks = {"seed-1": _resumable_ckpt(1, epoch=boundary)}
            report = build_convergence_report(conv, seed_logs=logs)
            rounds_used = (boundary - 200) // 50
            hist = ContinuationHistory(base_boundary=200, rounds_used=rounds_used)
            d = plan_committee_continuation(
                report, logs, cks, cont, conv, hist,
                triggering_convergence_report_sha256="s")
            outcomes.append(d.outcome)
        # 200->250, 250->300, 300->350 continue; 350->400 exceeds 150 cumulative.
        assert outcomes[:3] == [ContinuationOutcome.CONTINUE] * 3
        assert outcomes[3] == ContinuationOutcome.RECOVERY_BUDGET_EXHAUSTED

    # (9) the convergence gate is unchanged and cannot be bypassed
    def test_continuation_does_not_bypass_convergence_gate(self):
        logs = {"seed-1": _log(total_epoch=200, series=_improving_series(), best_at=200)}
        cks = {"seed-1": _resumable_ckpt(1)}
        report, _, d = self._plan(logs, cks)
        # A CONTINUE decision is issued precisely because the gate is still FAIL.
        assert d.outcome == ContinuationOutcome.CONTINUE
        assert convergence_gate_ok(report) is False


# --------------------------------------------------------------------------
# Per-seed assessment ordering (numerical health precedes "still improving").
# --------------------------------------------------------------------------
class TestSeedAssessmentOrdering:
    def test_converged_seed_is_preserved(self):
        logs = _log(total_epoch=200, series=_flat_series(), best_at=200)
        conv = default_training_convergence_policy()
        rep = build_convergence_report(conv, seed_logs={"seed-1": logs})["per_seed"]["seed-1"]
        d = assess_seed_continuation(
            "seed-1", rep, logs, _resumable_ckpt(1),
            default_training_continuation_policy())
        assert d.reason == SeedContinuationReason.CONVERGED
        assert d.eligible is False

    def test_eligible_seed_carries_checkpoint_provenance(self):
        logs = _log(total_epoch=200, series=_improving_series(), best_at=200)
        conv = default_training_convergence_policy()
        rep = build_convergence_report(conv, seed_logs={"seed-1": logs})["per_seed"]["seed-1"]
        d = assess_seed_continuation(
            "seed-1", rep, logs, _resumable_ckpt(1),
            default_training_continuation_policy())
        assert d.reason == SeedContinuationReason.ELIGIBLE
        assert d.eligible is True
        assert d.start_epoch == 200
        assert d.source_checkpoint_sha256 == "sha-1"
