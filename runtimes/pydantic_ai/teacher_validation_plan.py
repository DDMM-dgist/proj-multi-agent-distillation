"""Provenance-bound bridge: evidence-driven Teacher validation ADMISSIBLE DECISION SPACE ->
typed TeacherValidationPlan draft.

    inspect_teacher_evidence (deterministic fact-finder, validation/teacher_evidence_profile.py)
    -> TeacherEvidenceProfile
    -> derive_admissible_decision_space (deterministic, policy-only -- never a scientific choice)
    -> PydanticAI Orchestrator -> typed TeacherValidationPlanProposal (THIS MODULE)
    -> THIS MODULE -> typed TeacherValidationPlanDraft (JSON-serializable)
    -> workflow.controller.RunController.commit_teacher_validation_plan (sole authoritative
       validator)

This module mirrors ``runtimes.pydantic_ai.recovery_bridge``'s structure exactly, for the same
reason: it produces a DRAFT only. Nothing here commits, approves, or authorizes anything --
``commit_teacher_validation_plan`` remains the one place a Teacher validation plan is ever bound
to a run, and ``authorize_downstream_teacher_reliance`` remains the only place a human approves
costly downstream reliance (Teacher labeling / Student training) on a plan that lacks predictive-
fidelity evidence.

``selected_components`` is the ONE genuinely scientific choice this bridge cannot make
deterministically: which admissible component(s) a campaign actually USES (as opposed to what its
evidence merely makes possible -- ``derive_admissible_decision_space``'s job, never a mutually
exclusive strategy enum). ``validate_teacher_validation_plan_proposal`` enforces, fail-closed, that
the choice is a genuine (non-empty) subset of what the evidence profile admits, and that it
respects any objective(s) the run's own ``validation_profile.yaml`` declared
(``workflow.contracts.TEACHER_VALIDATION_OBJECTIVES``).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import NonEmptyStr


class TeacherValidationPlanValidationError(ValueError):
    pass


class TeacherValidationPlanProposal(BaseModel):
    """The Orchestrator's typed reasoning output proposing WHICH admissible Teacher-validation
    component(s) this campaign will actually use, and (when relevant) which reference kind /
    target split / source dataset role that selection implies. ``evidence_profile_sha256`` ties
    this proposal back to the exact evidence profile it was given, so a proposal answering a stale
    or different profile fails closed rather than being silently accepted -- exactly
    ``RecoveryPlanProposal.diagnosis_artifact_sha256``'s role."""
    model_config = {"extra": "forbid"}
    run_id: NonEmptyStr
    evidence_profile_sha256: NonEmptyStr
    selected_components: list[NonEmptyStr] = Field(min_length=1)
    reference_kind: Optional[str] = None
    target_split: Optional[str] = None
    source_dataset_role: Optional[str] = None
    rationale: NonEmptyStr


def validate_teacher_validation_plan_proposal(
    proposal: TeacherValidationPlanProposal, *, expected_run_id: str,
    expected_evidence_profile_sha256: str, admissible_components,
    validation_objectives=(),
) -> TeacherValidationPlanProposal:
    """Contextual, fail-closed validation beyond Pydantic shape (mirrors
    ``recovery_bridge.validate_recovery_plan_proposal``'s role for ``RecoveryPlanProposal``).

    ``admissible_components`` is the run's own ``derive_admissible_decision_space(...)
    ["admissible_components"]`` -- the proposal's ``selected_components`` must be a non-empty
    SUBSET of it; selecting anything else (an unsupported claim the evidence does not actually
    back) is rejected unconditionally, never overridable by human approval (see
    ``RunController.commit_teacher_validation_plan``'s docstring).

    ``validation_objectives`` (optional, from ``workflow.contracts.
    parse_teacher_validation_objectives``) are additional, run-declared requirements this proposal
    must also satisfy when the relevant evidence exists:
      * ``require_predictive_fidelity_when_evidence_supports_it``: if
        ``ORIGINAL_HELDOUT_FIDELITY`` or ``INDEPENDENT_REFERENCE_FIDELITY`` is admissible, at
        least one of them must be selected.
      * ``assess_deployment_applicability_when_domain_evidence_exists``: if
        ``DEPLOYMENT_APPLICABILITY`` is admissible, it must be selected.
      * ``prohibit_unsupported_generalization_claims`` is enforced unconditionally above (an
        unsupported claim is rejected regardless of whether this objective is declared) -- it is
        listed in the vocabulary for run-declared visibility/documentation, not because it adds a
        further check here.
    """
    if proposal.run_id != expected_run_id:
        raise TeacherValidationPlanValidationError(
            f"proposal targets run_id {proposal.run_id!r}, expected {expected_run_id!r}")
    if proposal.evidence_profile_sha256 != expected_evidence_profile_sha256:
        raise TeacherValidationPlanValidationError(
            "proposal's evidence_profile_sha256 does not match the evidence profile it was given "
            "-- refusing to bind a Teacher validation plan to the wrong (or stale) evidence")
    admissible = set(admissible_components)
    selected = set(proposal.selected_components)
    unsupported = sorted(selected - admissible)
    if unsupported:
        raise TeacherValidationPlanValidationError(
            f"proposal selects component(s) not admissible under this evidence profile: "
            f"{unsupported} -- admissible: {sorted(admissible)}")
    objectives = set(validation_objectives)
    if "require_predictive_fidelity_when_evidence_supports_it" in objectives:
        fidelity = {"ORIGINAL_HELDOUT_FIDELITY", "INDEPENDENT_REFERENCE_FIDELITY"}
        if (fidelity & admissible) and not (fidelity & selected):
            raise TeacherValidationPlanValidationError(
                "validation_profile objective 'require_predictive_fidelity_when_evidence_"
                "supports_it' requires selecting ORIGINAL_HELDOUT_FIDELITY or "
                "INDEPENDENT_REFERENCE_FIDELITY -- the evidence profile admits at least one of "
                "them but the proposal selected neither")
    if "assess_deployment_applicability_when_domain_evidence_exists" in objectives:
        if "DEPLOYMENT_APPLICABILITY" in admissible and "DEPLOYMENT_APPLICABILITY" not in selected:
            raise TeacherValidationPlanValidationError(
                "validation_profile objective 'assess_deployment_applicability_when_domain_"
                "evidence_exists' requires selecting DEPLOYMENT_APPLICABILITY -- the evidence "
                "profile admits it but the proposal did not select it")
    return proposal


class TeacherValidationPlanDraft(BaseModel):
    """Typed, JSON-serializable draft of a TeacherValidationPlan. ``.to_plan_json()`` produces
    exactly the dict shape ``RunController.commit_teacher_validation_plan`` expects/validates;
    that method re-validates every field authoritatively -- this model's own validation is a
    producer-side convenience, not a parallel authority."""
    model_config = {"extra": "forbid"}
    schema_version: int = 1
    run_id: NonEmptyStr
    evidence_profile_sha256: NonEmptyStr
    evidence_profile: dict[str, Any]
    admissible_components: list[str]
    selected_components: list[str] = Field(min_length=1)
    components: dict[str, Any]
    protected_data_restrictions: list[str] = Field(default_factory=list)
    approval_conditions: list[str] = Field(default_factory=list)
    reference_kind: Optional[str] = None
    target_split: Optional[str] = None
    source_dataset_role: Optional[str] = None
    rationale: NonEmptyStr
    proposed_by: Any
    validation_objectives: list[str] = Field(default_factory=list)

    def to_plan_json(self) -> dict:
        return self.model_dump()


def build_teacher_validation_plan_draft_from_proposal(
    proposal: TeacherValidationPlanProposal, *, decision_space: dict, evidence_profile: dict,
    proposed_by: Any, validation_objectives=(),
) -> TeacherValidationPlanDraft:
    """Re-project an already-validated ``TeacherValidationPlanProposal`` (the Orchestrator's
    scientific choice of which admissible component(s) to use) plus the deterministic decision
    space it was given into a typed, JSON-serializable draft -- so this reasoning-output path adds
    no second way to construct a plan, only a new, agent-driven SOURCE of the one genuinely
    scientific field (``selected_components``, plus the optional reference/target-split/source-
    role fields that follow from it)."""
    return TeacherValidationPlanDraft(
        run_id=proposal.run_id,
        evidence_profile_sha256=proposal.evidence_profile_sha256,
        evidence_profile=evidence_profile,
        admissible_components=list(decision_space["admissible_components"]),
        selected_components=list(proposal.selected_components),
        components=decision_space["components"],
        protected_data_restrictions=list(decision_space["protected_data_restrictions"]),
        approval_conditions=list(decision_space["approval_conditions"]),
        reference_kind=proposal.reference_kind,
        target_split=proposal.target_split,
        source_dataset_role=proposal.source_dataset_role,
        rationale=proposal.rationale,
        proposed_by=proposed_by,
        validation_objectives=list(validation_objectives),
    )
