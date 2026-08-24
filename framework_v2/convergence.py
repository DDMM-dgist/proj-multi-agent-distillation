"""Framework V2 -- deterministic convergence classifier for training.

R31 exposed that ``epochs_completed == epochs_requested`` was silently
accepted as "converged". Framework V2 rejects that: the classifier makes
the distinction explicit from evidence already present in every training
log:

  * ``CONVERGED_EARLY``    -- best_epoch lies well before the budget
    boundary; training naturally stopped improving.
  * ``CONVERGED_AT_MAX``   -- best_epoch is at/near the boundary, but
    the trailing validation slope is flat; the budget was fully used and
    the potential no longer improves meaningfully.
  * ``NOT_CONVERGED``      -- best_epoch is at/near the boundary AND at
    least one validation metric's trailing slope projects a meaningful
    further improvement. This is the R31 failure mode: the training
    budget was too small.
  * ``INSUFFICIENT_DATA``  -- the LOG is missing, empty, or has too few
    epoch lines to classify.

Config comes from a ``ConvergencePolicy`` contract; NO thresholds are
hard-coded here. The workflow config (or the StudentRecipePlan's
additional-parameters section) supplies the numbers, and every produced
report echoes the policy verbatim so an auditor can see exactly which
thresholds were used and where they came from.

Integration point: the training stage's gate MUST call
``build_convergence_report`` and refuse a PASS if
``convergence_gate_ok`` returns False (Section 10 / Section 13). That
makes convergence a DeterministicFact (see ``framework_v2.facts``) that
Judges cannot override.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from framework_v2.contracts import ConvergencePolicy, ConvergenceStatus


# Status label constants (re-exported for convenience; source of truth is
# the ConvergenceStatus enum)
CONVERGED_EARLY = ConvergenceStatus.CONVERGED_EARLY.value
CONVERGED_AT_MAX = ConvergenceStatus.CONVERGED_AT_MAX.value
NOT_CONVERGED = ConvergenceStatus.NOT_CONVERGED.value
INSUFFICIENT_DATA = ConvergenceStatus.INSUFFICIENT_DATA.value

CONVERGED_STATUSES = frozenset({CONVERGED_EARLY, CONVERGED_AT_MAX})
_STATUS_SEVERITY = {
    NOT_CONVERGED: 3,
    INSUFFICIENT_DATA: 2,
    CONVERGED_AT_MAX: 1,
    CONVERGED_EARLY: 0,
}

# --- LOG regexes (matching SIMPLE-NN's format; kept module-local so we
# do not couple to training_evidence.py's private names). If SIMPLE-NN
# changes format, both modules must be updated in lockstep.
_EPOCH_RE = re.compile(
    r"^\s*Epoch\s+(\d+)\s+E RMSE\(T V\)\s+(\S+)\s+(\S+)\s+"
    r"F RMSE\(T V\)\s+(\S+)\s+(\S+)\s+learning_rate:\s+(\S+)")
_TOTAL_EPOCH_RE = re.compile(r"Total\s+tran?ing epoch\s*:\s*(\d+)")
_BEST_EPOCH_RE = re.compile(r"Best loss.*written at\s+(\d+)\s+epoch")


@dataclasses.dataclass(frozen=True)
class EpochPoint:
    epoch: int
    train_e_rmse: float
    valid_e_rmse: float
    train_f_rmse: float
    valid_f_rmse: float
    learning_rate: float


_METRIC_EXTRACTORS: dict[str, Callable[[EpochPoint], float]] = {
    "valid_energy_rmse": lambda pt: pt.valid_e_rmse,
    "valid_force_rmse": lambda pt: pt.valid_f_rmse,
    "train_energy_rmse": lambda pt: pt.train_e_rmse,
    "train_force_rmse": lambda pt: pt.train_f_rmse,
}


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_all_epoch_points(log_text: str) -> list[EpochPoint]:
    """Every ``Epoch <N> ...`` line from a SIMPLE-NN LOG, in order.
    A line where any of the six numeric fields fails to parse is
    skipped (the LOG is external; one bad line shouldn't drop the
    series)."""
    pts: list[EpochPoint] = []
    for line in log_text.splitlines():
        m = _EPOCH_RE.match(line)
        if not m:
            continue
        vals = [_to_float(m.group(i)) for i in range(2, 7)]
        if any(v is None for v in vals):
            continue
        pts.append(EpochPoint(
            epoch=int(m.group(1)),
            train_e_rmse=vals[0], valid_e_rmse=vals[1],
            train_f_rmse=vals[2], valid_f_rmse=vals[3],
            learning_rate=vals[4],
        ))
    return pts


def find_best_epoch(log_text: str) -> int | None:
    for line in log_text.splitlines():
        m = _BEST_EPOCH_RE.search(line)
        if m:
            return int(m.group(1))
    return None


def find_epochs_requested(log_text: str) -> int | None:
    for line in log_text.splitlines():
        m = _TOTAL_EPOCH_RE.search(line)
        if m:
            return int(m.group(1))
    return None


def compute_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope of ``y`` vs ``x``. ``None`` if <2 points or
    degenerate (all x equal)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def compute_trailing_slope(
    points: Sequence[EpochPoint],
    window: int,
    extractor: Callable[[EpochPoint], float],
) -> tuple[float | None, float | None, int]:
    """Slope over the last ``window`` epochs. Returns
    (slope, last-value, n-points-used)."""
    if not points:
        return None, None, 0
    last_ep = points[-1].epoch
    sel = [(pt.epoch, extractor(pt)) for pt in points if pt.epoch >= last_ep - window]
    if len(sel) < 2:
        return None, extractor(points[-1]), len(sel)
    xs = [p[0] for p in sel]
    ys = [p[1] for p in sel]
    return compute_slope(xs, ys), ys[-1], len(sel)


def classify_seed_convergence(log_text: str, policy: ConvergencePolicy) -> dict:
    """Classify one seed's training against the supplied policy.

    A metric is "meaningfully improving" iff its trailing slope is
    negative AND ``abs(slope) * projection_window / current >=
    policy.min_relative_improvement``. A seed is NOT_CONVERGED iff it
    is at the boundary AND at least one requested metric is
    meaningfully improving.
    """
    points = parse_all_epoch_points(log_text)
    epochs_requested = find_epochs_requested(log_text)
    best_epoch = find_best_epoch(log_text)
    epochs_completed = points[-1].epoch if points else None

    if not points or len(points) < 2:
        return {
            "status": INSUFFICIENT_DATA,
            "epochs_completed": epochs_completed,
            "epochs_requested": epochs_requested,
            "best_epoch": best_epoch,
            "at_boundary": None,
            "per_metric": {},
            "rationale": "log missing or fewer than 2 epoch lines",
        }

    at_boundary = None
    if isinstance(epochs_requested, int) and isinstance(best_epoch, int):
        at_boundary = best_epoch >= epochs_requested - policy.boundary_tolerance
    elif isinstance(epochs_requested, int) and isinstance(epochs_completed, int):
        at_boundary = epochs_completed >= epochs_requested - policy.boundary_tolerance

    per_metric: dict[str, dict] = {}
    any_improving = False
    for name in policy.metrics:
        extractor = _METRIC_EXTRACTORS.get(name)
        if extractor is None:
            per_metric[name] = {
                "error": "unknown_metric",
                "supported": sorted(_METRIC_EXTRACTORS.keys()),
            }
            continue
        slope, current, n_pts = compute_trailing_slope(
            points, policy.trailing_window, extractor)
        if slope is None or current is None:
            per_metric[name] = {
                "slope_per_epoch": slope, "current": current,
                "trailing_points_used": n_pts,
                "meaningfully_improving": None,
                "reason": "insufficient_points_or_degenerate",
            }
            continue
        projected = abs(slope) * policy.projection_window
        rel = projected / abs(current) if current else (
            float("inf") if projected else 0.0)
        improving = (slope < 0) and (rel >= policy.min_relative_improvement)
        any_improving = any_improving or improving
        per_metric[name] = {
            "slope_per_epoch": slope,
            "current": current,
            "trailing_points_used": n_pts,
            "projected_absolute_improvement": projected,
            "projected_relative_improvement": rel,
            "meaningfully_improving": improving,
        }

    if at_boundary and any_improving:
        status = NOT_CONVERGED
        rationale = ("best/completed epoch is at the requested-epoch boundary "
                     "and at least one validation metric still projects a "
                     "meaningful further improvement")
    elif at_boundary is False:
        status = CONVERGED_EARLY
        rationale = ("best_epoch precedes epochs_requested by more than "
                     "boundary_tolerance")
    elif at_boundary and not any_improving:
        status = CONVERGED_AT_MAX
        rationale = ("best/completed epoch is at the requested-epoch boundary "
                     "and no requested metric projects a meaningful further "
                     "improvement (trailing slopes flat)")
    else:
        status = INSUFFICIENT_DATA
        rationale = ("epochs_requested or best_epoch not recorded; cannot "
                     "determine boundary")

    return {
        "status": status,
        "epochs_completed": epochs_completed,
        "epochs_requested": epochs_requested,
        "best_epoch": best_epoch,
        "at_boundary": at_boundary,
        "per_metric": per_metric,
        "rationale": rationale,
    }


def _committee_status(seed_statuses: Iterable[str]) -> str:
    """Committee status = worst per-seed status."""
    statuses = list(seed_statuses)
    if not statuses:
        return INSUFFICIENT_DATA
    return max(statuses, key=lambda s: _STATUS_SEVERITY.get(s, 99))


def build_convergence_report(
    policy: ConvergencePolicy,
    *,
    run_dir: str | Path | None = None,
    seed_logs: Mapping[str | int, str] | None = None,
) -> dict:
    """Committee-wide convergence report.

    Either provide ``seed_logs`` (a mapping seed_id -> log_text, used by
    tests and by callers that already loaded the logs) or ``run_dir``
    (the report reads ``artifacts/committee/seed-*/LOG`` from
    underneath it).

    The returned dict echoes ``policy`` verbatim under ``policy`` so an
    auditor sees exactly which thresholds produced the classification.
    """
    if seed_logs is None:
        if run_dir is None:
            raise ValueError("either run_dir or seed_logs must be provided")
        run_dir = Path(run_dir).resolve()
        committee = run_dir / "artifacts" / "committee"
        collected: dict[str, str] = {}
        if committee.is_dir():
            for seed_dir in sorted(committee.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed-"):
                    continue
                log_path = seed_dir / "LOG"
                if log_path.exists():
                    collected[seed_dir.name] = log_path.read_text(
                        encoding="utf-8", errors="replace")
        seed_logs = collected

    per_seed: dict[str, dict] = {}
    for seed_id, log_text in seed_logs.items():
        per_seed[str(seed_id)] = classify_seed_convergence(log_text, policy)
    committee_status = _committee_status(rep["status"] for rep in per_seed.values())

    return {
        "schema_version": 1,
        "profile": "training_convergence_report",
        "committee_status": committee_status,
        "committee_status_meaning": {
            NOT_CONVERGED: ("at least one seed reached the requested-epoch "
                            "boundary while still meaningfully improving; do "
                            "not accept as trained"),
            CONVERGED_AT_MAX: ("all seeds reached the boundary but validation "
                               "trends have flattened; training used its "
                               "budget"),
            CONVERGED_EARLY: ("all seeds' best_epoch precedes their boundary; "
                              "training naturally stopped improving"),
            INSUFFICIENT_DATA: ("logs missing or too short to classify at "
                                "least one seed"),
        }[committee_status],
        "per_seed": per_seed,
        "policy": policy.model_dump(mode="json"),
        "policy_sha256": policy.content_sha256(),
    }


def convergence_gate_ok(report: dict) -> bool:
    """HARD precondition on a training-stage PASS: True iff
    ``committee_status`` is one of the CONVERGED_* labels."""
    return report.get("committee_status") in CONVERGED_STATUSES


# Identifier for the framework's built-in training-convergence policy. A run
# whose config does not supply its own ConvergencePolicy falls back to this so
# the R31 "max-epoch == converged" gate is ALWAYS enforced (never silently
# skipped). Callers that need different thresholds supply their own contract.
DEFAULT_TRAINING_CONVERGENCE_POLICY_ID = "framework-default-training-convergence-v1"


def default_training_convergence_policy() -> ConvergencePolicy:
    """The framework's own training-convergence policy.

    Its numbers are stamped ``FRAMEWORK_CONSTRAINT`` (Section: ProvenanceClass)
    -- the framework itself imposes them as the fail-closed default that makes
    the R31 max-epoch-as-converged guard un-skippable. A run is free to override
    by supplying its own ``ConvergencePolicy`` (workflow.yaml / StudentRecipePlan
    ``additional``); when it does not, this is bound so the convergence gate is
    still enforced rather than silently absent (the demonstrated defect)."""
    from framework_v2.contracts import ProvenanceClass
    return ConvergencePolicy(
        policy_id=DEFAULT_TRAINING_CONVERGENCE_POLICY_ID,
        trailing_window=50,
        projection_window=50,
        min_relative_improvement=0.05,
        boundary_tolerance=5,
        metrics=["valid_energy_rmse", "valid_force_rmse"],
        provenance_class=ProvenanceClass.FRAMEWORK_CONSTRAINT,
        provenance_source=(
            "framework_v2 built-in fail-closed training-convergence gate "
            "(R31 max-epoch-as-converged guard); bound by bind-closure for any "
            "stage whose StageReviewSpec requires convergence_report evidence "
            "and whose run config supplies no ConvergencePolicy override"),
    )


__all__ = [
    "EpochPoint",
    "CONVERGED_EARLY", "CONVERGED_AT_MAX", "NOT_CONVERGED",
    "INSUFFICIENT_DATA", "CONVERGED_STATUSES",
    "parse_all_epoch_points", "find_best_epoch", "find_epochs_requested",
    "compute_slope", "compute_trailing_slope",
    "classify_seed_convergence",
    "build_convergence_report",
    "convergence_gate_ok",
    "DEFAULT_TRAINING_CONVERGENCE_POLICY_ID",
    "default_training_convergence_policy",
]
