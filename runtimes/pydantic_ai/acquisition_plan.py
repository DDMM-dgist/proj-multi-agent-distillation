"""Provenance-bound bridge: autonomous acquisition ADMISSIBLE DECISION SPACE -> typed
AcquisitionPlanProposal -> deterministically-validated ``framework_v2`` evidence chain.

    framework_v2.acquisition deterministic pipeline (objective -> inventory -> target-regime ->
        region-resolution -> coverage-gap -> strategy) + an admissible decision space
    -> PydanticAI producer -> typed AcquisitionPlanProposal (THIS MODULE)
    -> THIS MODULE's contextual validator + framework_v2 plan assembly + validate_acquisition_plan_v2
    -> orchestrator_bridge.propose_acquisition_plan -> RunController.bind_new_input (sole binder)

This mirrors ``runtimes.pydantic_ai.teacher_validation_plan`` exactly, for the same reason: the
low-level acquisition knobs (which parents, how many per structure, and the generation params
``T_K``/``beta``/``sigma_range_A``/``cell_sigma``/``seed``) are the ONE genuinely-scientific choice
the deterministic pipeline cannot make on its own -- the coverage analysis establishes only WHICH
regimes are under-covered and WHICH parents/sources are admissible, never the exact recipe. The
producer proposes that recipe; ``validate_acquisition_plan_proposal`` (contextual, fail-closed) plus
``framework_v2.acquisition.validators.validate_acquisition_plan_v2`` (the deterministic evidence-
chain gate) enforce it. Nothing here binds or executes anything: binding stays the frozen
``RunController.bind_new_input`` reached only through the audited orchestrator bridge, and the actual
Teacher-driven candidate generation stays the ACQUISITION stage's own approval-gated executor.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import NonEmptyStr


class AcquisitionPlanProposalValidationError(ValueError):
    pass


class AcquisitionPlanProposal(BaseModel):
    """A producer's typed reasoning output proposing the low-level acquisition recipe within the
    deterministically-derived admissible decision space. ``coverage_gap_sha256`` ties the proposal
    back to the exact coverage analysis it answers, so a recipe answering a stale/different coverage
    fails closed rather than being silently accepted -- exactly
    ``TeacherValidationPlanProposal.evidence_profile_sha256``'s role."""
    model_config = {"extra": "forbid"}
    run_id: NonEmptyStr
    coverage_gap_sha256: NonEmptyStr
    strategy_kind: NonEmptyStr
    selected_parent_ids: list[NonEmptyStr] = Field(min_length=1)
    selected_source_global_indices: list[int] = Field(default_factory=list)
    n_per_structure: int = Field(ge=1)
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: NonEmptyStr


def validate_acquisition_plan_proposal(
    proposal: AcquisitionPlanProposal, *, expected_run_id: str,
    expected_coverage_gap_sha256: str, admissible_strategy_kind: str,
    admissible_parent_ids, required_param_keys=(), param_bounds=None,
) -> AcquisitionPlanProposal:
    """Contextual, fail-closed validation beyond Pydantic shape (mirrors
    ``validate_teacher_validation_plan_proposal``).

    ``admissible_parent_ids`` is the deterministic pipeline's admissible parent pool -- the
    proposal's ``selected_parent_ids`` must be a non-empty SUBSET of it (proposing a parent the
    coverage analysis never admitted is rejected unconditionally). ``admissible_strategy_kind`` is
    the strategy the deterministic ``select_strategy`` chose from the coverage/inventory evidence;
    the proposal may not override it. ``required_param_keys`` must all be present in ``params``;
    ``param_bounds`` (optional ``{key: (lo, hi)}``) fail closed on any numeric param outside its
    inclusive admissible interval. No material-specific default is invented here -- every bound is
    supplied by the caller's admissible decision space."""
    if proposal.run_id != expected_run_id:
        raise AcquisitionPlanProposalValidationError(
            f"proposal targets run_id {proposal.run_id!r}, expected {expected_run_id!r}")
    if proposal.coverage_gap_sha256 != expected_coverage_gap_sha256:
        raise AcquisitionPlanProposalValidationError(
            "proposal's coverage_gap_sha256 does not match the coverage analysis it was given "
            "-- refusing to bind an acquisition plan to stale/different coverage evidence")
    if proposal.strategy_kind != admissible_strategy_kind:
        raise AcquisitionPlanProposalValidationError(
            f"proposal strategy_kind {proposal.strategy_kind!r} != the deterministically-selected "
            f"strategy {admissible_strategy_kind!r} -- the producer may not override strategy")
    admissible = set(admissible_parent_ids)
    selected = set(proposal.selected_parent_ids)
    unsupported = sorted(selected - admissible)
    if unsupported:
        raise AcquisitionPlanProposalValidationError(
            f"proposal selects parent(s) not admissible under this coverage analysis: "
            f"{unsupported}")
    if len(selected) != len(proposal.selected_parent_ids):
        raise AcquisitionPlanProposalValidationError("selected_parent_ids contains duplicates")
    missing = sorted(set(required_param_keys) - set(proposal.params))
    if missing:
        raise AcquisitionPlanProposalValidationError(
            f"proposal params missing required key(s): {missing}")
    for key, (lo, hi) in (param_bounds or {}).items():
        if key not in proposal.params:
            continue
        value = proposal.params[key]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise AcquisitionPlanProposalValidationError(
                f"param {key!r}={value!r} is not numeric but a bound was declared for it")
        if not (lo <= numeric <= hi):
            raise AcquisitionPlanProposalValidationError(
                f"param {key!r}={numeric} is outside its admissible interval [{lo}, {hi}]")
    return proposal
