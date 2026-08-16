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

``RootCauseClassification.failure_category`` is the Analyst's real LLM ``output_type`` field (see
``role_outputs.register_reasoning_output_model``), so it is typed as
``recovery_taxonomy.failure_category_enum()`` — a real Enum built FROM the registry — rather than
a plain ``str`` plus a hidden validator: the generated Pydantic JSON Schema exposes the literal
registered codes as an ``"enum": [...]``, so a provider enforcing strict structured output can
itself constrain the model to a registered code, not just have it rejected after the fact once the
model has already spent its structured-output retries guessing at an unpublished vocabulary.
``RecoveryRecommendation.failure_category`` stays a plain, validator-checked ``str``: it is never
an LLM output_type (always Python-constructed via ``to_recovery_recommendation`` below from an
already-validated classification), so it has no schema-visibility gap to close.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from workflow import recovery_taxonomy

from .models import EvidenceReference, NonEmptyStr

# RecoveryRecommendation.failure_category only (never an LLM output_type; see module docstring).
# RootCauseClassification.failure_category is typed directly as recovery_taxonomy's Enum below.
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
    # Schema-visible registered vocabulary (see module docstring) -- Pydantic itself rejects any
    # value outside the registry, so no separate field_validator is needed here (contrast
    # RecoveryRecommendation.failure_category below, which is not an LLM output_type).
    failure_category: recovery_taxonomy.failure_category_enum()
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


# affected_channel values that assert a Teacher-vs-DFT physical-accuracy comparison was actually
# made -- the exact assertion R20 demonstrated being made with no such comparison anywhere in the
# failed stage's evidence (teacher_baseline uses no DFT labels at all). Free-text (see root_cause.
# py's module docstring on why affected_channel is not itself a schema Enum), so matched loosely.
_DFT_CHANNEL_MARKERS = ("dft", "reference_disagreement")


def _asserts_dft_comparison(classification: "RootCauseClassification") -> bool:
    channel = (classification.affected_channel or "").strip().lower()
    # `failure_category` is a `str`-mixin Enum member (see recovery_taxonomy.failure_category_enum):
    # `str(member)` renders "FailureCategory.reference_disagreement", not the raw registered code,
    # so compare the member's own `.value` (equivalently, `==` against the plain string) rather than
    # its `str()` -- otherwise this half of the gate can never fire.
    category = classification.failure_category.value \
        if hasattr(classification.failure_category, "value") else str(classification.failure_category)
    return (any(marker in channel for marker in _DFT_CHANNEL_MARKERS)
            or category == "reference_disagreement")


def validate_root_cause_classification(classification: RootCauseClassification, *,
                                       available_artifacts, valid_recovery_targets,
                                       dft_comparison_evidence_present: bool = True):
    """Evidence-bind a classification. Raises if it cites a nonexistent artifact, has no
    evidence, or targets an unknown recovery stage. (failure_category is enforced by the model.)

    ``dft_comparison_evidence_present``: deterministically computed by the caller (see
    ``cli._stage_evidence_reveals_dft_comparison``) from the FAILED STAGE'S OWN evidence
    artifacts -- never trusted from the classification itself. When False, a classification
    asserting a Teacher-vs-DFT channel/disagreement (``affected_channel`` naming "dft", or
    ``failure_category == "reference_disagreement"``) is rejected: the failed stage made no such
    comparison, so no evidence supports that specific claim (R20 forensic finding: teacher_baseline
    was misclassified as a ``reference_disagreement``/``teacher_vs_dft`` failure although it uses
    no DFT labels at all). This does not block genuine Teacher-vs-DFT diagnoses where the evidence
    actually contains one (e.g. a reference_validation gate failure) -- only an unsupported one.
    """
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
    if not dft_comparison_evidence_present and _asserts_dft_comparison(classification):
        raise RootCauseValidationError(
            "classification asserts a Teacher-vs-DFT channel/disagreement (affected_channel="
            f"{classification.affected_channel!r}, failure_category="
            f"{classification.failure_category!r}), but the failed stage's own evidence contains "
            "no Teacher-vs-DFT comparison (no dft-labeled energy/forces, and no protected-"
            "reference/DFT usage recorded) -- this is unsupported by evidence; classify as an "
            "evidence/provenance gap instead (e.g. 'evidence_gap' or 'lineage_or_leakage')")
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
