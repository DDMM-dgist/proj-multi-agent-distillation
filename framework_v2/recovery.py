"""Framework V2 — first-class, material-agnostic RecoveryPlan + dependency-aware
invalidation (Sections L & M).

Section L requires that a Controller-native typed RecoveryPlan be *mandatory*
on any valid REVISE-class verdict: a stage may not advance from a REVISE by
re-judging; it must carry a typed plan naming the failing state, the responsible
re-entry stage, a concrete objective, what must change, and how the fix is
re-validated. Section M requires that when a stage's decision is revised, the
downstream stages that consumed its output (by SHA) are invalidated rather than
silently kept.

This module is the *generic core* home for both. It deliberately does NOT
duplicate the Controller's authority: ``workflow.controller.RunController``
remains the sole place a recovery is bound to a pending gate, and
``runtimes.pydantic_ai.recovery_bridge.RecoveryPlanDraft`` remains the runtime
projection onto the Controller's on-disk shape. What lives here is (a) the
typed, versioned, material-agnostic recovery *contract shape* and the rule that
a recovery is required for each recovery-bearing SemanticState, and (b) pure
functions computing downstream invalidation from a stage-dependency graph or
from recorded per-stage packet SHAs.

Nothing here encodes any material, model family, or campaign: the failure
states are the framework-invariant vocabulary from :mod:`framework_v2.states`,
the failure codes are resolved against the shared, extensible
:mod:`workflow.recovery_taxonomy` registry, and the stage identities are the
canonical stages from :mod:`framework_v2.stages`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Optional

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso
from framework_v2.stages import stage_index
from framework_v2.states import RECOVERY_BEARING_STATES, SemanticState


# Default registered failure_code for each recovery-bearing SemanticState. This
# is the ONE place the generic recovery vocabulary (states) is mapped to the
# shared failure-code registry, so a RecoveryPlan carrying a specialised state
# routes to a stable, registered code instead of an ad-hoc string. Plain REVISE
# / FAIL are intentionally NOT auto-mapped: they are generic and the producer
# must choose a registered code appropriate to the actual root cause.
DEFAULT_STATE_FAILURE_CODE: dict[SemanticState, str] = {
    SemanticState.REPRESENTATION_INSUFFICIENT: "representation_insufficient",
    SemanticState.REVISE_SPLIT: "split_unrepresentative",
    SemanticState.NOT_CONVERGED: "training_instability",
}


def recovery_required(state: SemanticState) -> bool:
    """True iff a valid outcome in ``state`` MUST carry a typed RecoveryPlan
    before the stage can advance (Section L). These are exactly the
    recovery-bearing states; PASS/NOT_APPLICABLE and the gate-blocking evidence
    states are handled by other paths (regenerate evidence / re-run review)."""
    return state in RECOVERY_BEARING_STATES


class RecoveryPlan(ContractBase):
    """The generic, typed recovery contract for one revised stage decision.

    Material-agnostic: it names *which* typed failure occurred, *where* recovery
    re-enters the canonical pipeline, *what* must change, and *how* the fix is
    re-validated — but never any material observable or model specific.
    """
    plan_id: str
    run_id: str
    failed_stage: str
    failure_state: SemanticState
    failure_code: str
    responsible_stage: str
    responsible_capability: str = ""
    objective: str
    required_changes: list[str] = Field(min_length=1)
    revalidation_criteria: list[str] = Field(min_length=1)
    invalidated_downstream_stages: list[str] = Field(default_factory=list)
    superseded_decision_sha256: Optional[str] = None
    superseded_packet_sha256: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _failure_state_is_recovery_bearing(self):
        if self.failure_state not in RECOVERY_BEARING_STATES:
            raise ValueError(
                f"RecoveryPlan.failure_state {self.failure_state.value!r} is not a "
                f"recovery-bearing state; a RecoveryPlan is only required/valid for "
                f"{sorted(s.value for s in RECOVERY_BEARING_STATES)}"
            )
        return self

    @model_validator(mode="after")
    def _stages_are_canonical(self):
        if stage_index(self.failed_stage) < 0:
            raise ValueError(f"failed_stage {self.failed_stage!r} is not a canonical stage")
        if stage_index(self.responsible_stage) < 0:
            raise ValueError(f"responsible_stage {self.responsible_stage!r} is not a canonical stage")
        return self

    @model_validator(mode="after")
    def _reentry_not_after_failure(self):
        """Recovery re-enters at or before the failed stage — a plan cannot fix a
        stage by re-running something strictly downstream of it."""
        if stage_index(self.responsible_stage) > stage_index(self.failed_stage):
            raise ValueError(
                f"responsible_stage {self.responsible_stage!r} is canonically AFTER "
                f"failed_stage {self.failed_stage!r}; recovery must re-enter at or "
                f"before the failed stage"
            )
        return self

    @model_validator(mode="after")
    def _failure_code_registered(self):
        from workflow.recovery_taxonomy import resolve_failure_code
        try:
            resolve_failure_code(self.failure_code)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _specialised_state_code_consistent(self):
        """If the failure_state has a canonical default code, the plan must use it
        (so a REPRESENTATION_INSUFFICIENT plan cannot be routed under an unrelated
        code and lose its typed meaning). Generic REVISE/FAIL are unconstrained."""
        expected = DEFAULT_STATE_FAILURE_CODE.get(self.failure_state)
        if expected is not None and self.failure_code != expected:
            raise ValueError(
                f"failure_state {self.failure_state.value!r} requires failure_code "
                f"{expected!r}, got {self.failure_code!r}"
            )
        return self


def default_failure_code_for(state: SemanticState) -> Optional[str]:
    """The registered failure_code a plan for ``state`` should use, or None if the
    state is generic (REVISE/FAIL) and the producer must choose a code."""
    return DEFAULT_STATE_FAILURE_CODE.get(state)


# =====================================================================
# DEPENDENCY-AWARE INVALIDATION (Section M)
# =====================================================================
def _build_dependents(dependencies: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    """Invert an upstream map (stage -> stages it consumed) into a dependents map
    (stage -> stages that directly consume it)."""
    dependents: dict[str, set[str]] = defaultdict(set)
    for stage, upstreams in dependencies.items():
        for up in upstreams:
            dependents[up].add(stage)
    return dependents


def transitive_downstream(
    changed_stages: Iterable[str],
    dependencies: Mapping[str, Iterable[str]],
) -> set[str]:
    """All stages that transitively depend on any changed stage.

    ``dependencies`` maps each stage to the upstream stages whose output it
    consumed (the natural declaration direction). The changed stages themselves
    are NOT included in the result — only what is downstream of them.
    """
    dependents = _build_dependents(dependencies)
    changed = set(changed_stages)
    result: set[str] = set()
    stack = list(changed)
    while stack:
        s = stack.pop()
        for dep in dependents.get(s, ()):  # stages consuming s
            if dep not in result and dep not in changed:
                result.add(dep)
                stack.append(dep)
    return result


def invalidate_downstream(
    changed_stages: Iterable[str],
    dependencies: Mapping[str, Iterable[str]],
) -> list[str]:
    """Downstream stages invalidated by a change, ordered by canonical position.

    A stage that is canonically *before* every changed stage can never be
    invalidated by that change; such an edge indicates a malformed dependency
    graph and is rejected fail-closed rather than silently producing a
    backwards invalidation."""
    changed = set(changed_stages)
    inv = transitive_downstream(changed, dependencies)
    earliest_change = min((stage_index(s) for s in changed), default=-1)
    for s in inv:
        if stage_index(s) >= 0 and 0 <= stage_index(s) < earliest_change:
            raise ValueError(
                f"stage {s!r} depends on a change at a canonically later stage; "
                f"dependency graph is not acyclic in canonical order"
            )
    return sorted(inv, key=stage_index)


def stale_downstream_by_sha(
    prior_sha_by_stage: Mapping[str, str],
    new_sha_by_stage: Mapping[str, str],
    dependencies: Mapping[str, Iterable[str]],
) -> tuple[list[str], list[str]]:
    """Compare recorded per-stage packet/decision SHAs and return
    ``(changed_stages, invalidated_downstream)``.

    A stage is *changed* if its SHA differs between the prior and new maps (a
    stage present in one map but absent in the other also counts as changed).
    Downstream invalidation is then the transitive dependents of the changed
    set. This is the SHA-driven form of Section M: a stage whose upstream SHA no
    longer matches what it was validated against is stale."""
    stages = set(prior_sha_by_stage) | set(new_sha_by_stage)
    changed = [
        s for s in stages
        if prior_sha_by_stage.get(s) != new_sha_by_stage.get(s)
    ]
    invalidated = invalidate_downstream(changed, dependencies)
    # a changed stage is itself stale, but "downstream invalidation" excludes the
    # changed stages; callers that want the full stale set union them.
    return sorted(changed, key=stage_index), invalidated


__all__ = [
    "DEFAULT_STATE_FAILURE_CODE",
    "recovery_required",
    "RecoveryPlan",
    "default_failure_code_for",
    "transitive_downstream",
    "invalidate_downstream",
    "stale_downstream_by_sha",
]
