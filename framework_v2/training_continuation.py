"""Framework V2 -- autonomous *bounded* training-continuation recovery layer.

The convergence classifier (:mod:`framework_v2.convergence`) can already say a
committee is ``NOT_CONVERGED`` because it hit the epoch boundary while the
validation trend was still meaningfully improving -- i.e. the training budget
was simply too small. Framework V2 already had every ingredient to *react* to
that (TRUE checkpoint resume, typed recovery routing back to TRAINING via the
``training_instability`` code, deterministic re-gating) EXCEPT the policy layer
that decides -- deterministically and without asking a human or an LLM to invent
an epoch number -- *whether* and *how far* to continue, and *when to stop and
escalate*.

This module is that layer. It is material-agnostic and model-agnostic: it never
encodes a chemistry, a model family, or a campaign-specific epoch count. The one
"scientific number" (how many more epochs one continuation round buys) is
**derived** from the bound :class:`ConvergencePolicy` evidence window, not
authored per material.

Design invariants:

  * The convergence gate is NOT modified and cannot be bypassed. This layer only
    *reads* a convergence report; a PASS still requires
    ``convergence.convergence_gate_ok`` on freshly-rebuilt evidence.
  * Continuation is allowed ONLY for the specific, evidenced failure mode
    "reached the boundary while still materially improving, numerically healthy,
    with a resumable checkpoint". Every other failure mode (numerical
    divergence/NaN, missing checkpoint, insufficient evidence) is routed to a
    human, never silently treated as "train longer".
  * The recovery is BOUNDED by a *global* compute ceiling: a versioned,
    hash-bound :class:`TrainingContinuationPolicy` caps both the number of rounds
    and the cumulative continuation epochs. The ceiling is a generic
    compute-safety backstop -- deliberately generous so an ordinarily
    under-trained but healthy, still-improving run resolves autonomously across
    however many evidence windows it needs -- while still guaranteeing
    termination. When the ceiling is reached the layer emits
    ``RECOVERY_BUDGET_EXHAUSTED`` and requires human escalation. Note the ceiling
    only ever fully binds a run that keeps improving by a hair forever: a healthy
    run normally *converges* (its trend flattens -> ``CONVERGED_AT_MAX`` ->
    gate PASS) long before the ceiling, which stops continuation immediately.
  * Everything is deterministic and provenance-bound: the source checkpoint SHA,
    start/target epochs, both policy SHAs, the triggering convergence-report SHA,
    and the cumulative budget consumption are recorded for every round.
"""
from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Mapping, Optional, Sequence

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, ProvenanceClass
from framework_v2.convergence import (
    CONVERGED_STATUSES,
    NOT_CONVERGED,
    INSUFFICIENT_DATA,
    _METRIC_EXTRACTORS,
    parse_all_epoch_points,
)
from framework_v2.recovery import RecoveryPlan
from framework_v2.states import SemanticState
# The human-escalation routing code emitted when the bounded recovery budget is
# spent is defined and registered canonically in the shared taxonomy at that
# module's import time (so the cached failure_category_enum always enumerates it,
# regardless of import order). We only re-export the constant here.
from workflow.recovery_taxonomy import RECOVERY_BUDGET_EXHAUSTED_CODE


# Which field of the bound ConvergencePolicy the continuation quantum is read
# from. The quantum is NEVER a free-standing material-specific number: one
# continuation round buys exactly one convergence *evidence window* of epochs,
# so the new evidence is directly comparable under the SAME classifier.
class QuantumSource(str, Enum):
    PROJECTION_WINDOW = "projection_window"
    TRAILING_WINDOW = "trailing_window"


# Per-seed eligibility outcome (scientific eligibility, budget applied later).
class SeedContinuationReason(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    CONVERGED = "CONVERGED"                      # preserve; not a problem
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"      # NaN / inf / divergence
    NOT_AT_BOUNDARY = "NOT_AT_BOUNDARY"
    NOT_IMPROVING = "NOT_IMPROVING"
    NO_RESUMABLE_CHECKPOINT = "NO_RESUMABLE_CHECKPOINT"


# Reasons that -- for a seed that IS in an eligible (NOT_CONVERGED) state -- mean
# the seed cannot be auto-continued and the situation must go to a human rather
# than be silently dropped or retried as "train longer".
_ESCALATION_REASONS = frozenset({
    SeedContinuationReason.INSUFFICIENT_EVIDENCE,
    SeedContinuationReason.NUMERICAL_FAILURE,
    SeedContinuationReason.NOT_AT_BOUNDARY,
    SeedContinuationReason.NOT_IMPROVING,
    SeedContinuationReason.NO_RESUMABLE_CHECKPOINT,
})


# Committee-level decision outcome.
class ContinuationOutcome(str, Enum):
    CONTINUE = "CONTINUE"                                # bounded plan issued
    NO_RECOVERY_CONVERGED = "NO_RECOVERY_CONVERGED"      # nothing to do
    RECOVERY_BUDGET_EXHAUSTED = "RECOVERY_BUDGET_EXHAUSTED"   # -> human
    HUMAN_ESCALATION_REQUIRED = "HUMAN_ESCALATION_REQUIRED"   # non-continuation fault -> human


# =====================================================================
# Policy contract (versioned + hash-bound)
# =====================================================================
class TrainingContinuationPolicy(ContractBase):
    """Bounds and rules for autonomous training continuation.

    ``continuation_quantum`` is NOT stored as a number: it is derived from the
    bound convergence policy's evidence window (``quantum_source``) at plan time,
    so it always tracks the classifier's own window and is never a material-
    specific authored value. Everything else is an explicit, hash-bound bound.
    """
    policy_id: str
    quantum_source: QuantumSource = QuantumSource.PROJECTION_WINDOW
    max_continuation_rounds: int = Field(ge=1)
    max_cumulative_continuation_epochs: int = Field(ge=1)
    eligible_states: list[str] = Field(min_length=1)
    require_resumable_checkpoint: bool = True
    divergence_relative_tolerance: float = Field(gt=0.0)
    exhaustion_behavior: str = "escalate_to_human"
    provenance_class: ProvenanceClass
    provenance_source: str

    @model_validator(mode="after")
    def _eligible_states_are_recovery_bearing(self):
        for s in self.eligible_states:
            try:
                st = SemanticState(s)
            except ValueError as exc:
                raise ValueError(f"eligible_states entry {s!r} is not a SemanticState") from exc
            from framework_v2.states import RECOVERY_BEARING_STATES
            if st not in RECOVERY_BEARING_STATES:
                raise ValueError(
                    f"eligible_states entry {s!r} is not a recovery-bearing state; "
                    f"autonomous continuation is only defensible for recovery-bearing "
                    f"training states (normally NOT_CONVERGED)")
        return self

    @model_validator(mode="after")
    def _exhaustion_behavior_known(self):
        if self.exhaustion_behavior != "escalate_to_human":
            raise ValueError(
                "the only supported exhaustion_behavior is 'escalate_to_human' "
                "(the framework never silently stops or silently continues past budget)")
        return self


DEFAULT_TRAINING_CONTINUATION_POLICY_ID = "framework-default-training-continuation-v2"

# Global compute-safety ceiling for autonomous continuation. These are generic
# FRAMEWORK_CONSTRAINT backstops, NOT targets and NOT tuned to any material or
# run: they are deliberately generous so that ordinary under-training (a healthy,
# still-improving committee that simply needs more epochs) resolves autonomously
# across as many evidence windows as it needs, while termination is still
# guaranteed. The two limits are coherent at the default 50-epoch window
# (24 rounds x 50 = 1200 epochs); whichever binds first stops continuation. A
# healthy run almost always CONVERGES (trend flattens) well before either limit;
# the ceiling only fully binds a pathological "improves by a hair forever" run.
DEFAULT_MAX_CONTINUATION_ROUNDS = 24
DEFAULT_MAX_CUMULATIVE_CONTINUATION_EPOCHS = 1200


def default_training_continuation_policy() -> TrainingContinuationPolicy:
    """The framework's own fail-closed bounded-continuation policy.

    Numbers are ``FRAMEWORK_CONSTRAINT`` global compute-safety ceilings, not
    campaign targets: at most :data:`DEFAULT_MAX_CONTINUATION_ROUNDS` automatic
    rounds and at most :data:`DEFAULT_MAX_CUMULATIVE_CONTINUATION_EPOCHS`
    cumulative continuation epochs before a human must decide. They are large
    enough that normal under-training resolves autonomously (the run typically
    converges first), yet still guarantee termination. A run may override by
    supplying its own contract; when it does not, this bounds the autonomy rather
    than leaving it unbounded (or absent)."""
    return TrainingContinuationPolicy(
        policy_id=DEFAULT_TRAINING_CONTINUATION_POLICY_ID,
        quantum_source=QuantumSource.PROJECTION_WINDOW,
        max_continuation_rounds=DEFAULT_MAX_CONTINUATION_ROUNDS,
        max_cumulative_continuation_epochs=DEFAULT_MAX_CUMULATIVE_CONTINUATION_EPOCHS,
        eligible_states=[NOT_CONVERGED],
        require_resumable_checkpoint=True,
        divergence_relative_tolerance=0.10,
        exhaustion_behavior="escalate_to_human",
        provenance_class=ProvenanceClass.FRAMEWORK_CONSTRAINT,
        provenance_source=(
            "framework_v2 built-in fail-closed bounded training-continuation "
            "policy: continuation quantum = convergence-policy projection window; "
            "generic global compute-safety ceiling of "
            f"<= {DEFAULT_MAX_CONTINUATION_ROUNDS} automatic rounds and "
            f"<= {DEFAULT_MAX_CUMULATIVE_CONTINUATION_EPOCHS} cumulative "
            "continuation epochs (deliberately generous so ordinary under-training "
            "resolves autonomously; a healthy run normally converges first) before "
            "mandatory human escalation"),
    )


# =====================================================================
# Runtime inputs (not hash-bound contracts -- observed environment facts)
# =====================================================================
# Keys a SIMPLE-NN-style resumable checkpoint must carry for a TRUE resume
# (restores model + optimizer + normalization state + epoch counter).
REQUIRED_RESUMABLE_KEYS: tuple[str, ...] = (
    "model", "optimizer", "epoch", "scale_factor", "pca")


def checkpoint_is_resumable(
    present_keys: Sequence[str],
    *,
    required_keys: Sequence[str] = REQUIRED_RESUMABLE_KEYS,
) -> tuple[bool, str]:
    """Pure check: does a checkpoint (described by the keys it contains) carry
    everything a TRUE resume needs? Returns (ok, reason)."""
    missing = [k for k in required_keys if k not in set(present_keys)]
    if missing:
        return False, f"missing_checkpoint_keys:{','.join(missing)}"
    return True, "resumable"


@dataclasses.dataclass(frozen=True)
class CheckpointInfo:
    """Observed facts about one seed's latest checkpoint."""
    seed: int
    path: str
    sha256: str
    epoch: int
    resumable: bool
    resumable_reason: str = "resumable"


@dataclasses.dataclass(frozen=True)
class ContinuationHistory:
    """How much of the bounded budget has already been spent for a run.

    ``base_boundary`` is the training epoch boundary BEFORE any continuation
    (e.g. 200); ``rounds_used`` counts continuation rounds already executed.
    Callers derive these from prior provenance records / run state.
    """
    base_boundary: int
    rounds_used: int = 0


# =====================================================================
# Numerical health (NaN / inf / divergence) -- independent of the gate
# =====================================================================
def detect_numerical_health(
    log_text: str,
    metrics: Sequence[str],
    *,
    divergence_relative_tolerance: float,
) -> tuple[bool, str]:
    """Deterministically decide whether a seed's training was numerically healthy.

    Unhealthy iff any requested metric contains a non-finite value (NaN/inf) or
    its last value exceeds its running minimum by more than
    ``divergence_relative_tolerance`` (a blow-up). A blow-up must NOT be treated
    as "train longer"; it is a distinct fault that needs a human. Returns
    (healthy, reason)."""
    points = parse_all_epoch_points(log_text)
    if not points:
        return False, "no_epoch_points"
    for name in metrics:
        extractor = _METRIC_EXTRACTORS.get(name)
        if extractor is None:
            # Unknown metric name: fail closed rather than ignore.
            return False, f"unknown_metric:{name}"
        series = [extractor(p) for p in points]
        if any((v is None or not math.isfinite(v)) for v in series):
            return False, f"nan_or_inf:{name}"
        vmin = min(series)
        vlast = series[-1]
        if vmin > 0 and vlast > vmin * (1.0 + divergence_relative_tolerance):
            return False, f"diverged:{name}"
    return True, "healthy"


def _eligible_metrics_hint(seed_report: Mapping) -> list[str]:
    """Metrics to health-check for a seed: exactly those the convergence policy
    classified on (surfaced in the per-seed report), so numerical health and the
    gate look at the same series. Falls back to all known metrics."""
    pm = seed_report.get("per_metric") or {}
    names = [n for n in pm.keys() if n in _METRIC_EXTRACTORS]
    return names or list(_METRIC_EXTRACTORS.keys())


# =====================================================================
# Per-seed eligibility (scientific; budget applied at committee level)
# =====================================================================
@dataclasses.dataclass(frozen=True)
class SeedContinuationDecision:
    seed: str
    reason: SeedContinuationReason
    eligible: bool
    detail: str
    start_epoch: Optional[int] = None
    source_checkpoint_sha256: Optional[str] = None


def assess_seed_continuation(
    seed_id: str,
    seed_report: Mapping,
    log_text: str,
    checkpoint: Optional[CheckpointInfo],
    cont_policy: TrainingContinuationPolicy,
) -> SeedContinuationDecision:
    """Scientific (budget-independent) continuation eligibility for one seed.

    Order matters: converged seeds are simply preserved; among seeds in an
    eligible failure state, numerical faults and missing checkpoints are surfaced
    as their own reasons (never collapsed into "train longer")."""
    status = seed_report.get("status")

    # A converged seed is preserved, not a problem, not eligible.
    if status in CONVERGED_STATUSES:
        return SeedContinuationDecision(
            seed_id, SeedContinuationReason.CONVERGED, False,
            f"seed status {status}: preserve as-is")

    if status == INSUFFICIENT_DATA:
        return SeedContinuationDecision(
            seed_id, SeedContinuationReason.INSUFFICIENT_EVIDENCE, False,
            "convergence evidence insufficient to classify; cannot auto-continue")

    # Only recovery-bearing training states are continuation-eligible.
    if status not in cont_policy.eligible_states:
        return SeedContinuationDecision(
            seed_id, SeedContinuationReason.INSUFFICIENT_EVIDENCE, False,
            f"seed status {status!r} is not a continuation-eligible state")

    # NUMERICAL health precedes the "still improving" test so a blow-up is never
    # mistaken for a continuation opportunity.
    healthy, health_reason = detect_numerical_health(
        log_text, _eligible_metrics_hint(seed_report),
        divergence_relative_tolerance=cont_policy.divergence_relative_tolerance)
    if not healthy:
        return SeedContinuationDecision(
            seed_id, SeedContinuationReason.NUMERICAL_FAILURE, False,
            f"numerical failure ({health_reason}); route to human, not train-longer")

    # Defensive re-checks of the two evidence facts that DEFINE NOT_CONVERGED, so
    # eligibility never rests on the status label alone.
    if seed_report.get("at_boundary") is not True:
        return SeedContinuationDecision(
            seed_id, SeedContinuationReason.NOT_AT_BOUNDARY, False,
            "not at the epoch boundary; more epochs are not the indicated fix")
    per_metric = seed_report.get("per_metric") or {}
    any_improving = any(m.get("meaningfully_improving") is True for m in per_metric.values())
    if not any_improving:
        return SeedContinuationDecision(
            seed_id, SeedContinuationReason.NOT_IMPROVING, False,
            "no metric is still meaningfully improving; more epochs unjustified")

    # A resumable checkpoint is mandatory for a TRUE resume.
    if cont_policy.require_resumable_checkpoint:
        if checkpoint is None:
            return SeedContinuationDecision(
                seed_id, SeedContinuationReason.NO_RESUMABLE_CHECKPOINT, False,
                "no checkpoint available for TRUE resume")
        if not checkpoint.resumable:
            return SeedContinuationDecision(
                seed_id, SeedContinuationReason.NO_RESUMABLE_CHECKPOINT, False,
                f"checkpoint not resumable ({checkpoint.resumable_reason})")

    return SeedContinuationDecision(
        seed_id, SeedContinuationReason.ELIGIBLE, True,
        "at boundary, still improving, numerically healthy, resumable",
        start_epoch=(checkpoint.epoch if checkpoint else None),
        source_checkpoint_sha256=(checkpoint.sha256 if checkpoint else None))


# =====================================================================
# Committee-level decision + provenance
# =====================================================================
@dataclasses.dataclass(frozen=True)
class SeedContinuationDirective:
    """A single eligible seed's concrete, deterministic continuation directive."""
    seed: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    start_epoch: int
    target_epoch: int


class ContinuationRoundProvenance(ContractBase):
    """Hash-bound provenance for one continuation round (Requirement 6)."""
    round_index: int
    continuation_policy_sha256: str
    convergence_policy_sha256: str
    triggering_convergence_report_sha256: str
    quantum_epochs: int
    target_epoch: int
    per_seed: list[dict]
    cumulative_continuation_epochs_before: int
    cumulative_continuation_epochs_after: int
    rounds_used_before: int
    resulting_convergence_report_sha256: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CommitteeContinuationDecision:
    outcome: ContinuationOutcome
    directives: list[SeedContinuationDirective]
    preserved_seeds: list[str]
    escalation_seeds: list[str]
    per_seed: dict
    detail: str
    quantum_epochs: int
    target_epoch: Optional[int] = None
    total_epoch_override: Optional[int] = None
    recovery_plan: Optional[RecoveryPlan] = None
    provenance: Optional[ContinuationRoundProvenance] = None


def continuation_quantum(cont_policy: TrainingContinuationPolicy, conv_policy) -> int:
    """Epochs one continuation round buys, DERIVED from the convergence window."""
    if cont_policy.quantum_source == QuantumSource.PROJECTION_WINDOW:
        q = int(conv_policy.projection_window)
    elif cont_policy.quantum_source == QuantumSource.TRAILING_WINDOW:
        q = int(conv_policy.trailing_window)
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unknown quantum_source {cont_policy.quantum_source!r}")
    if q < 1:
        raise ValueError(f"derived continuation quantum {q} < 1; convergence window invalid")
    return q


def plan_committee_continuation(
    convergence_report: Mapping,
    seed_logs: Mapping[str, str],
    checkpoints: Mapping[str, CheckpointInfo],
    cont_policy: TrainingContinuationPolicy,
    conv_policy,
    history: ContinuationHistory,
    *,
    triggering_convergence_report_sha256: str,
) -> CommitteeContinuationDecision:
    """Deterministically decide the next bounded continuation for a committee.

    * Only individually-eligible (NOT_CONVERGED + at boundary + improving +
      numerically healthy + resumable) seeds are continued.
    * Already-converged seeds are preserved (never retrained).
    * A NOT_CONVERGED seed that cannot be auto-continued (numerical fault,
      missing checkpoint, contradictory evidence) forces HUMAN escalation.
    * The round is issued only if the bounded budget (rounds AND cumulative
      epochs) still admits a full quantum; otherwise RECOVERY_BUDGET_EXHAUSTED.
    """
    quantum = continuation_quantum(cont_policy, conv_policy)
    per_seed_reports = convergence_report.get("per_seed") or {}

    per_seed_decisions: dict[str, SeedContinuationDecision] = {}
    for seed_id, rep in per_seed_reports.items():
        per_seed_decisions[seed_id] = assess_seed_continuation(
            seed_id, rep, seed_logs.get(seed_id, ""), checkpoints.get(seed_id), cont_policy)

    eligible = [d for d in per_seed_decisions.values()
                if d.reason == SeedContinuationReason.ELIGIBLE]
    preserved = [d.seed for d in per_seed_decisions.values()
                 if d.reason == SeedContinuationReason.CONVERGED]
    escalation = [d.seed for d in per_seed_decisions.values()
                  if d.reason in _ESCALATION_REASONS]

    per_seed_out = {
        d.seed: {"reason": d.reason.value, "eligible": d.eligible, "detail": d.detail}
        for d in per_seed_decisions.values()
    }

    # A NOT_CONVERGED seed that cannot be auto-continued for a non-budget reason
    # is a change of situation: hand to a human, never silently drop or retry.
    if escalation:
        return CommitteeContinuationDecision(
            ContinuationOutcome.HUMAN_ESCALATION_REQUIRED, [], preserved, escalation,
            per_seed_out,
            "at least one non-converged seed cannot be auto-continued "
            f"(seeds: {sorted(escalation)}); human decision required",
            quantum)

    if not eligible:
        return CommitteeContinuationDecision(
            ContinuationOutcome.NO_RECOVERY_CONVERGED, [], preserved, [],
            per_seed_out, "all seeds converged; no continuation recovery required",
            quantum)

    # --- bounded budget check (rounds AND cumulative epochs) ---
    # The round advances every eligible seed by one quantum from its own current
    # boundary. In the common case all eligible seeds share a boundary.
    current_boundary = max(int(per_seed_reports[d.seed]["epochs_requested"]) for d in eligible)
    target_epoch = current_boundary + quantum
    cumulative_before = current_boundary - history.base_boundary
    cumulative_after = target_epoch - history.base_boundary

    rounds_ok = history.rounds_used < cont_policy.max_continuation_rounds
    epochs_ok = cumulative_after <= cont_policy.max_cumulative_continuation_epochs
    if not (rounds_ok and epochs_ok):
        reason = []
        if not rounds_ok:
            reason.append(
                f"rounds_used={history.rounds_used} >= max={cont_policy.max_continuation_rounds}")
        if not epochs_ok:
            reason.append(
                f"cumulative_continuation_epochs would be {cumulative_after} > "
                f"max={cont_policy.max_cumulative_continuation_epochs}")
        return CommitteeContinuationDecision(
            ContinuationOutcome.RECOVERY_BUDGET_EXHAUSTED, [], preserved, [],
            per_seed_out,
            "bounded continuation budget exhausted (" + "; ".join(reason) +
            "); human escalation required", quantum,
            target_epoch=target_epoch)

    # --- build deterministic per-seed directives (only eligible seeds) ---
    directives: list[SeedContinuationDirective] = []
    for d in sorted(eligible, key=lambda x: x.seed):
        ck = checkpoints[d.seed]
        directives.append(SeedContinuationDirective(
            seed=d.seed,
            source_checkpoint_path=ck.path,
            source_checkpoint_sha256=ck.sha256,
            start_epoch=ck.epoch,
            target_epoch=int(per_seed_reports[d.seed]["epochs_requested"]) + quantum,
        ))

    # A single committee-wide total_epoch_override is well-defined only when every
    # eligible seed shares a target; otherwise callers must dispatch per-seed.
    distinct_targets = sorted({dv.target_epoch for dv in directives})
    total_epoch_override = distinct_targets[0] if len(distinct_targets) == 1 else None

    plan = _build_continuation_recovery_plan(
        run_id=str(convergence_report.get("run_id") or "run"),
        directives=directives, preserved=preserved, quantum=quantum,
        conv_policy=conv_policy, cont_policy=cont_policy,
        round_index=history.rounds_used + 1,
        triggering_convergence_report_sha256=triggering_convergence_report_sha256)

    prov = ContinuationRoundProvenance(
        round_index=history.rounds_used + 1,
        continuation_policy_sha256=cont_policy.content_sha256(),
        convergence_policy_sha256=conv_policy.content_sha256(),
        triggering_convergence_report_sha256=triggering_convergence_report_sha256,
        quantum_epochs=quantum,
        target_epoch=target_epoch,
        per_seed=[{"seed": dv.seed,
                   "source_checkpoint_sha256": dv.source_checkpoint_sha256,
                   "start_epoch": dv.start_epoch,
                   "target_epoch": dv.target_epoch} for dv in directives],
        cumulative_continuation_epochs_before=cumulative_before,
        cumulative_continuation_epochs_after=cumulative_after,
        rounds_used_before=history.rounds_used,
    )

    return CommitteeContinuationDecision(
        ContinuationOutcome.CONTINUE, directives, preserved, [], per_seed_out,
        f"continue {len(directives)} non-converged seed(s) by {quantum} epochs to "
        f"epoch {target_epoch} (round {history.rounds_used + 1} of "
        f"{cont_policy.max_continuation_rounds})",
        quantum, target_epoch=target_epoch, total_epoch_override=total_epoch_override,
        recovery_plan=plan, provenance=prov)


def _build_continuation_recovery_plan(
    *, run_id, directives, preserved, quantum, conv_policy, cont_policy,
    round_index, triggering_convergence_report_sha256) -> RecoveryPlan:
    """Construct the typed NOT_CONVERGED RecoveryPlan for an eligible round.

    Uses the canonical ``training_instability`` -> TRAINING route. The epoch
    target is DERIVED (start boundary + quantum); no human/LLM invents it."""
    seeds = ", ".join(dv.seed for dv in directives)
    return RecoveryPlan(
        plan_id=f"{run_id}-training-continuation-r{round_index}",
        run_id=run_id,
        failed_stage="training",
        failure_state=SemanticState.NOT_CONVERGED,
        failure_code="training_instability",
        responsible_stage="training",
        responsible_capability="model_fitting",
        objective=(
            "Autonomous bounded continuation: the committee reached the epoch "
            "boundary while still materially improving (NOT_CONVERGED). Continue "
            "only the non-converged, numerically-healthy, resumable members by one "
            "derived evidence window under the UNCHANGED convergence gate. More "
            "epochs alone never imply PASS."),
        required_changes=[
            f"TRUE-RESUME seeds [{seeds}] from checkpoint_latest (model+optimizer+"
            "scale+pca preserved); do NOT retrain from epoch 0.",
            f"Set each continued seed's total_epoch_override deterministically to "
            f"start_boundary + {quantum} (continuation quantum = convergence "
            f"{cont_policy.quantum_source.value}); no human/LLM chooses the number.",
            f"Preserve already-converged members untouched: [{', '.join(preserved) or 'none'}].",
        ],
        revalidation_criteria=[
            "Rebuild the convergence_report from the continued LOGs and re-run the "
            "UNCHANGED deterministic convergence gate (convergence_gate_ok).",
            "Committee status must be CONVERGED_* for PASS; NOT_CONVERGED stays a FAIL.",
            "If still eligible NOT_CONVERGED and budget remains, the next bounded "
            "round is constructed automatically; on budget exhaustion, escalate to "
            "a human (RECOVERY_BUDGET_EXHAUSTED).",
        ],
        superseded_packet_sha256=triggering_convergence_report_sha256,
    )


__all__ = [
    "RECOVERY_BUDGET_EXHAUSTED_CODE",
    "QuantumSource",
    "SeedContinuationReason",
    "ContinuationOutcome",
    "TrainingContinuationPolicy",
    "DEFAULT_TRAINING_CONTINUATION_POLICY_ID",
    "DEFAULT_MAX_CONTINUATION_ROUNDS",
    "DEFAULT_MAX_CUMULATIVE_CONTINUATION_EPOCHS",
    "default_training_continuation_policy",
    "REQUIRED_RESUMABLE_KEYS",
    "checkpoint_is_resumable",
    "CheckpointInfo",
    "ContinuationHistory",
    "detect_numerical_health",
    "SeedContinuationDecision",
    "assess_seed_continuation",
    "SeedContinuationDirective",
    "ContinuationRoundProvenance",
    "CommitteeContinuationDecision",
    "continuation_quantum",
    "plan_committee_continuation",
]
