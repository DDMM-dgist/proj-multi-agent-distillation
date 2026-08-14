"""Analyst reasoning output: RootCauseClassification + RecoveryRecommendation (Phase 6/2).

classify_root_cause is NOT a deterministic executor — it is the Analyst's typed reasoning output:

    deterministic validation/analysis artifacts
    -> PydanticAI Analyst -> typed RootCauseClassification
    -> deterministic contract/evidence validation -> RecoveryRecommendation

The classification is evidence-bound: every cited artifact must exist. It cannot change a Judge
verdict or mutate controller stage state (it carries no such fields and the validator touches no
controller state).

``failure_category`` is a registered ``failure_code`` from ``workflow.recovery_taxonomy`` — the
same shared registry ``workflow.controller.RECOVERY_CATEGORIES`` resolves against — rather than a
locally-declared ``Literal``. This is what lets diagnosis (this module) and recovery
(``workflow.controller.propose_recovery``) share one vocabulary instead of two independently
maintained category sets: an unregistered string fails closed here exactly as it does in the
controller. ``failure_domain`` is never supplied by a caller; it is always derived from
``failure_category`` via ``workflow.recovery_taxonomy.domain_of`` so the two can never disagree.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from workflow import recovery_taxonomy

from .models import EvidenceReference, NonEmptyStr

# A registered failure_code (see workflow.recovery_taxonomy) — validated dynamically against the
# shared registry rather than pinned to a fixed Literal, so new campaigns can register additional
# codes without editing this module.
FailureCategory = str


def _resolve_or_raise(value: str) -> str:
    try:
        recovery_taxonomy.resolve_failure_code(value)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    return value


class RootCauseClassification(BaseModel):
    model_config = {"extra": "forbid"}
    run_id: NonEmptyStr
    stage: NonEmptyStr
    failure_category: FailureCategory
    failure_domain: str = ""
    affected_channel: str = ""
    affected_artifact_refs: list[EvidenceReference] = Field(default_factory=list)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    evidence_summary: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)
    excluded_alternatives: list[str] = Field(default_factory=list)
    uncertainty_or_limitations: str = ""
    recommended_recovery_target: NonEmptyStr
    recommended_next_action: NonEmptyStr
    requires_human_approval: bool = True

    @field_validator("failure_category")
    @classmethod
    def _check_failure_category(cls, value):
        return _resolve_or_raise(value)

    @model_validator(mode="after")
    def _bind_failure_domain(self):
        # Always derived, never trusted from the caller: this is what makes failure_domain
        # incapable of drifting from the registered failure_category.
        self.failure_domain = recovery_taxonomy.domain_of(self.failure_category)
        return self


class RecoveryRecommendation(BaseModel):
    model_config = {"extra": "forbid"}
    run_id: NonEmptyStr
    target_stage: NonEmptyStr
    failure_category: FailureCategory
    failure_domain: str = ""
    actions: list[NonEmptyStr] = Field(default_factory=list)
    rationale: NonEmptyStr
    requires_human_approval: bool = True

    @field_validator("failure_category")
    @classmethod
    def _check_failure_category(cls, value):
        return _resolve_or_raise(value)

    @model_validator(mode="after")
    def _bind_failure_domain(self):
        self.failure_domain = recovery_taxonomy.domain_of(self.failure_category)
        return self


class RootCauseValidationError(ValueError):
    pass


def validate_root_cause_classification(classification: RootCauseClassification, *,
                                       available_artifacts, valid_recovery_targets):
    """Evidence-bind a classification. Raises if it cites a nonexistent artifact, has no
    evidence, or targets an unknown recovery stage. (failure_category is enforced by the model.)"""
    available = set(available_artifacts)
    cited = [r.path for r in classification.evidence_refs] + \
            [r.path for r in classification.affected_artifact_refs]
    missing = [p for p in cited if p not in available]
    if missing:
        raise RootCauseValidationError(f"cites nonexistent artifact(s): {missing}")
    if not classification.evidence_refs:
        raise RootCauseValidationError("classification has no evidence_refs (missing evidence)")
    if classification.recommended_recovery_target not in set(valid_recovery_targets):
        raise RootCauseValidationError(
            f"invalid recovery target: {classification.recommended_recovery_target}")
    return classification


def to_recovery_recommendation(classification: RootCauseClassification) -> RecoveryRecommendation:
    """Project a validated classification into a typed RecoveryRecommendation (no controller
    mutation; the controller records recoveries only via its own approved recovery API)."""
    return RecoveryRecommendation(
        run_id=classification.run_id, target_stage=classification.recommended_recovery_target,
        failure_category=classification.failure_category,
        actions=[classification.recommended_next_action],
        rationale=classification.evidence_summary,
        requires_human_approval=classification.requires_human_approval)
