"""Provenance-bound bridge: Analyst RootCauseClassification -> typed RecoveryPlan draft.

    deterministic validation/analysis artifacts -> PydanticAI Analyst
    -> typed RootCauseClassification (root_cause.py, evidence-bound)
    -> THIS MODULE -> typed RecoveryPlanDraft (JSON-serializable)
    -> workflow.controller.RunController.propose_recovery (sole authoritative validator)

This module produces a DRAFT only. Nothing here approves, executes, or authorizes anything:
``propose_recovery`` remains the one place a RecoveryPlan is ever bound to a pending gate, and
``approve_recovery``/``authorize_recovery_capabilities`` remain the only places human approval or
a costly-child-action authorization envelope is ever recorded. Building a draft with this module
does not touch controller state and cannot be mistaken for any of those steps.

``RootCauseClassification`` (diagnosis) and ``RecoveryPlan`` (recovery) stay distinct types: this
bridge reads the former and produces the latter's typed draft, it never subclasses or merges them,
so a caller can never confuse "the Analyst's interpretation" with "the approved recovery contract".
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from workflow import recovery_taxonomy

from .models import NonEmptyStr
from .root_cause import RootCauseClassification


class EvidenceHashRef(BaseModel):
    """A hash-bound pointer to one artifact this draft's provenance depends on."""
    model_config = {"extra": "forbid"}
    path: NonEmptyStr
    sha256: NonEmptyStr


class DiagnosisBinding(BaseModel):
    """Provenance binding a RecoveryPlan draft to the exact diagnosis artifact and evidence that
    produced it -- fail-closed at propose_recovery time if either has since changed or gone
    missing (see workflow.controller.RunController._validate_diagnosis_binding)."""
    model_config = {"extra": "forbid"}
    diagnosis_artifact_path: NonEmptyStr
    diagnosis_artifact_sha256: NonEmptyStr
    triggering_evidence: list[EvidenceHashRef] = Field(default_factory=list)


class RecoveryRouting(BaseModel):
    """Capability-based responsible-agent routing for this draft.

    ``capability`` is a registered capability name resolved, at propose_recovery time, against
    the run's own recovery_capability_roster (or DEFAULT_RECOVERY_CAPABILITY_ROSTER if the run
    declares none) -- never a hardcoded literal agent-name set baked into this bridge.
    """
    model_config = {"extra": "forbid"}
    capability: NonEmptyStr
    expected_role: Optional[str] = None


class RecoveryPlanDraft(BaseModel):
    """Typed, JSON-serializable draft of a RecoveryPlan. ``.to_plan_json()`` produces exactly the
    dict shape ``RunController.propose_recovery`` expects/validates; propose_recovery re-validates
    every field authoritatively -- this model's own validation is a producer-side convenience,
    not a parallel authority."""
    model_config = {"extra": "forbid"}

    schema_version: int = 1
    failed_stage: NonEmptyStr
    failure_category: NonEmptyStr
    failure_domain: str = ""
    root_cause: NonEmptyStr
    # A provenance-bound proposer identity: either a bare human display-name string, or a
    # structured {"actor_kind": "human"|"agent"|"system", "canonical_id": ...} mapping (see
    # workflow.actor_identity.normalize_actor_identity, propose_recovery's sole authoritative
    # validator for this field). Required -- an agent-driven draft must always declare who/what
    # is proposing it, never leave that to be inferred or added later by whoever writes the file.
    # NOT an authority claim in its own right: once this draft's rendered JSON reaches
    # propose_recovery through an agent-facing bridge that supplies its own trusted `proposer`
    # kwarg (e.g. orchestrator_bridge._exec_propose_recovery), that trusted identity -- not this
    # field -- is what gets recorded, and a mismatch fails closed. This field is only trusted
    # outright when propose_recovery is invoked directly with no trusted proposer (the
    # human-operated CLI's call shape).
    proposed_by: Any
    routing: RecoveryRouting
    return_stage: NonEmptyStr
    proposed_changes: list[dict[str, Any]] = Field(min_length=1)
    labeling: dict[str, bool]
    student_training: dict[str, Any]
    revalidation: dict[str, Any]
    estimated_cost: dict[str, Any] = Field(default_factory=dict)
    diagnosis_binding: Optional[DiagnosisBinding] = None
    diagnosis_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    diagnosis_status: NonEmptyStr = "proposed"
    excluded_alternatives: list[str] = Field(default_factory=list)
    required_input_artifact_roles: list[str] = Field(default_factory=list)
    expected_output_artifact_roles: list[str] = Field(default_factory=list)
    resource_request: dict[str, Any] = Field(default_factory=dict)
    policy_budget_request: dict[str, Any] = Field(default_factory=dict)
    requires_human_approval: bool = True
    recovery_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_failure_category_and_domain(self):
        try:
            resolved = recovery_taxonomy.resolve_failure_code(self.failure_category)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if self.failure_domain and self.failure_domain != resolved.domain:
            raise ValueError(
                f"failure_domain {self.failure_domain!r} does not match the registered "
                f"domain for failure_category {self.failure_category!r} ({resolved.domain!r})"
            )
        self.failure_domain = resolved.domain
        return self

    def to_plan_json(self) -> dict:
        """Render the exact JSON dict propose_recovery() expects on disk."""
        payload = {
            "schema_version": self.schema_version,
            "proposed_by": self.proposed_by,
            "failed_stage": self.failed_stage,
            "failure_category": self.failure_category,
            "failure_domain": self.failure_domain,
            "root_cause": self.root_cause,
            "responsible_capability": self.routing.capability,
            "return_stage": self.return_stage,
            "proposed_changes": self.proposed_changes,
            "labeling": self.labeling,
            "student_training": self.student_training,
            "revalidation": self.revalidation,
            "estimated_cost": self.estimated_cost,
            "required_input_artifact_roles": self.required_input_artifact_roles,
            "expected_output_artifact_roles": self.expected_output_artifact_roles,
            # Descriptive-only fields below: propose_recovery does not require or validate them,
            # they are preserved verbatim on the recovery record for audit/traceability.
            "diagnosis_confidence": self.diagnosis_confidence,
            "diagnosis_status": self.diagnosis_status,
            "excluded_alternatives": list(self.excluded_alternatives),
            "resource_request": self.resource_request,
            "policy_budget_request": self.policy_budget_request,
            "requires_human_approval": self.requires_human_approval,
            "recovery_context": self.recovery_context,
        }
        if self.routing.expected_role is not None:
            payload["responsible_agent"] = self.routing.expected_role
        if self.diagnosis_binding is not None:
            payload["diagnosis_binding"] = self.diagnosis_binding.model_dump()
        return payload


def build_recovery_plan_draft(
    classification: RootCauseClassification, *,
    proposed_by: Any,
    failed_stage: str,
    capability: str,
    return_stage: str,
    proposed_changes: list,
    labeling: dict,
    student_training: dict,
    revalidation: dict,
    estimated_cost: Optional[dict] = None,
    diagnosis_artifact_path: Optional[str] = None,
    diagnosis_artifact_sha256: Optional[str] = None,
    triggering_evidence: Optional[list] = None,
    expected_role: Optional[str] = None,
    required_input_artifact_roles: Optional[list] = None,
    expected_output_artifact_roles: Optional[list] = None,
    resource_request: Optional[dict] = None,
    policy_budget_request: Optional[dict] = None,
) -> RecoveryPlanDraft:
    """Bridge an evidence-bound RootCauseClassification into a typed RecoveryPlanDraft.

    The classification must already have passed
    ``root_cause.validate_root_cause_classification`` -- this function does not re-run that
    evidence binding, it only reprojects an already-validated diagnosis into recovery-plan shape.
    ``failed_stage``/``return_stage`` are supplied by the caller (typically ``return_stage`` is
    ``classification.recommended_recovery_target``) rather than inferred, so this bridge never
    silently overrides the controller's own pending-gate/stage-ordering checks -- those still run,
    authoritatively, inside propose_recovery.
    """
    diagnosis_binding = None
    if diagnosis_artifact_path is not None or diagnosis_artifact_sha256 is not None:
        diagnosis_binding = DiagnosisBinding(
            diagnosis_artifact_path=diagnosis_artifact_path,
            diagnosis_artifact_sha256=diagnosis_artifact_sha256,
            triggering_evidence=[
                EvidenceHashRef(path=ref.path, sha256=(ref.integrity or {}).get("sha256", ""))
                for ref in classification.evidence_refs
            ] if triggering_evidence is None else [
                EvidenceHashRef(**item) for item in triggering_evidence
            ],
        )
    return RecoveryPlanDraft(
        proposed_by=proposed_by,
        failed_stage=failed_stage,
        failure_category=classification.failure_category,
        failure_domain=classification.failure_domain,
        root_cause=classification.evidence_summary,
        routing=RecoveryRouting(capability=capability, expected_role=expected_role),
        return_stage=return_stage,
        proposed_changes=proposed_changes,
        labeling=labeling,
        student_training=student_training,
        revalidation=revalidation,
        estimated_cost=estimated_cost or {},
        diagnosis_binding=diagnosis_binding,
        diagnosis_confidence=classification.confidence,
        diagnosis_status="proposed",
        excluded_alternatives=list(classification.excluded_alternatives),
        required_input_artifact_roles=list(required_input_artifact_roles or []),
        expected_output_artifact_roles=list(expected_output_artifact_roles or []),
        resource_request=resource_request or {},
        policy_budget_request=policy_budget_request or {},
        requires_human_approval=classification.requires_human_approval,
        recovery_context={"run_id": classification.run_id, "diagnosed_stage": classification.stage},
    )
