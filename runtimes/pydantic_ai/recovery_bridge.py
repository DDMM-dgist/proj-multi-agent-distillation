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

``RecoveryPlanProposal`` (below) closes a second gap: ``RootCauseClassification`` alone cannot
supply ``build_recovery_plan_draft``'s scientific-choice fields (``capability``,
``proposed_changes``, ``labeling``, ``student_training``, ``revalidation``, and — since the
Analyst's suggested ``recommended_recovery_target`` is advisory, not binding — ``return_stage``).
Those choices are still genuinely scientific, so this module never fills them in deterministically:
instead ``RecoveryPlanProposal`` is a second, distinct typed reasoning output — produced by a live
PydanticAI Orchestrator role through the exact same generic ``typed_reasoning_output`` acceptance
path as ``RootCauseClassification`` (see ``role_outputs.register_reasoning_output_model``), evidence
-bound back to the diagnosis it was given (``diagnosis_artifact_sha256``), and validated (fail
-closed, contextually, via ``validate_recovery_plan_proposal``) before ``build_recovery_plan_draft_
from_proposal`` re-projects it into the existing, unchanged ``build_recovery_plan_draft`` call.
``propose_recovery`` remains the sole authoritative validator either way.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from workflow import recovery_taxonomy

from .actions import ROLE_ALLOWED_ACTIONS
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


class RecoveryPlanValidationError(ValueError):
    pass


class CorrectiveAction(BaseModel):
    """Schema-visible shape of an optional, already-registered corrective action a
    ``RecoveryPlanProposal`` may request. ``action_type`` cannot be a fixed ``Literal`` here (unlike
    ``actions.DataCuratorActionProposal`` etc.) because which role -- and therefore which action
    set -- applies depends on the ``capability`` the SAME proposal chooses; there is no role known
    at class-definition time to build a per-role enum from. Structural presence of `action_type`/
    `parameters` is enforced here (so the model schema, not just prose, says a corrective action is
    a typed dispatch request, not free-form commentary); registry MEMBERSHIP of `action_type` for
    the chosen capability's role is still resolved contextually, in
    ``validate_recovery_plan_proposal``, against the single authoritative ``actions.
    ROLE_ALLOWED_ACTIONS`` -- never a second, duplicated action list."""
    model_config = {"extra": "forbid"}
    action_type: NonEmptyStr
    parameters: dict[str, Any] = Field(default_factory=dict)


class ProposedChange(BaseModel):
    """Schema-visible shape of one ``proposed_changes`` entry. ``propose_recovery``
    (``workflow.controller.py``) requires every item be a dict with a non-empty string ``type`` --
    previously invisible in the Pydantic schema (bare ``dict[str, Any]``), the same defect class as
    ``corrective_action``. Deliberately ``extra: allow``: unlike ``corrective_action`` (a precise
    dispatch request), a proposed change is a free-form scientific description the controller never
    assumes a fixed shape for beyond ``type`` (see ``RunController._validate_protected_reference_
    roles``'s own docstring); rejecting the model's other descriptive keys (``id``,
    ``responsible_agent``, ``action``, ``acceptance_criteria``, ``artifact_roles``, ...) would only
    discard legitimate content the controller happily stores and a human reviews."""
    model_config = {"extra": "allow"}
    type: NonEmptyStr


class LabelingPlan(BaseModel):
    """``propose_recovery`` requires exactly these two boolean keys (previously a bare
    ``dict[str, bool]`` with no required-key guarantee)."""
    model_config = {"extra": "allow"}
    teacher_relabel: bool
    new_dft: bool


class StudentTrainingPlan(BaseModel):
    """``propose_recovery`` requires ``retrain``/``mode`` plus the cross-field rule ``retrain ==
    (mode != "none")`` (previously a bare ``dict[str, Any]`` exposing neither)."""
    model_config = {"extra": "allow"}
    retrain: bool
    mode: NonEmptyStr

    @model_validator(mode="after")
    def _check_retrain_mode_consistency(self):
        if self.retrain == (self.mode == "none"):
            raise ValueError(
                f"student_training.retrain ({self.retrain!r}) is inconsistent with mode "
                f"{self.mode!r} -- retrain must be true iff mode != 'none'")
        return self


class RevalidationPlan(BaseModel):
    """``propose_recovery`` requires ``reuse_profile`` plus a non-empty ``targets`` list of
    non-empty strings (previously a bare ``dict[str, Any]`` exposing neither)."""
    model_config = {"extra": "allow"}
    reuse_profile: bool
    targets: list[NonEmptyStr] = Field(min_length=1)


class RecoveryPlanProposal(BaseModel):
    """The Orchestrator's typed reasoning output proposing HOW to recover from a diagnosed
    failure -- distinct from ``RootCauseClassification`` (WHY it failed) and from the approved
    ``RecoveryPlan`` itself (this is still only a proposal; ``propose_recovery`` is what actually
    binds it to a pending gate). ``diagnosis_artifact_sha256`` ties this proposal back to the exact
    diagnosis it was asked to act on, so a proposal answering a stale or different diagnosis fails
    closed rather than being silently accepted."""
    model_config = {"extra": "forbid"}
    run_id: NonEmptyStr
    failed_stage: NonEmptyStr
    diagnosis_artifact_sha256: NonEmptyStr
    capability: NonEmptyStr
    return_stage: NonEmptyStr
    proposed_changes: list[ProposedChange] = Field(min_length=1)
    labeling: LabelingPlan
    student_training: StudentTrainingPlan
    revalidation: RevalidationPlan
    rationale: NonEmptyStr
    # Optional: the ONE concrete, already-registered action (role-appropriate for `capability`)
    # that a generic driver (run-campaign) may dispatch automatically, without any human/Claude
    # out-of-band step, to actually perform this recovery's corrective work once approved. A
    # proposal that leaves this unset relies entirely on the workflow's own declared stage graph
    # naturally re-running return_stage (and everything after it) once the recovery iteration
    # quarantines their prior outputs -- still valid, just with no NEW parameters to apply.
    corrective_action: Optional[CorrectiveAction] = None


def valid_corrective_actions_by_capability(capability_roster: dict) -> dict[str, list[str]]:
    """The exact ``corrective_action.action_type`` vocabulary available under each registered
    recovery capability, derived single-source from ``actions.ROLE_ALLOWED_ACTIONS`` (the same
    registry ``validate_recovery_plan_proposal`` enforces membership against below) -- never a
    second, hand-maintained action list. Callers (``cli._propose_recovery_via_reasoning_roles``)
    place this directly in the Orchestrator's task context, analogous to how ``valid_capabilities``
    and ``valid_stage_names`` are already surfaced, so the model sees the real, current registry
    slice for every capability it might choose rather than guessing at a structure or vocabulary
    the schema alone cannot express (capability is chosen by the same proposal, so no single
    role -- and thus no single allowed-action ``Literal`` -- can be fixed at class-definition time)."""
    return {capability: sorted(ROLE_ALLOWED_ACTIONS[role])
            for capability, role in capability_roster.items() if role in ROLE_ALLOWED_ACTIONS}


def validate_recovery_plan_proposal(proposal: RecoveryPlanProposal, *, expected_failed_stage: str,
                                    expected_diagnosis_sha256: str, capability_roster: dict,
                                    valid_stage_names,
                                    dft_comparison_evidence_present: bool = True,
                                    gate_alleges_accuracy_disagreement: bool = True
                                    ) -> RecoveryPlanProposal:
    """Contextual, fail-closed validation beyond Pydantic shape (mirrors
    ``root_cause.validate_root_cause_classification``'s role for ``RootCauseClassification``).

    ``capability_roster`` maps each registered recovery capability name to the role responsible
    for it (e.g. ``workflow.controller.DEFAULT_RECOVERY_CAPABILITY_ROSTER``) -- the same roster
    ``propose_recovery`` itself resolves ``responsible_capability`` against. It replaces a bare
    ``valid_capabilities`` name-set because ``corrective_action.action_type``, when supplied, must
    be validated against the actions THAT role is actually allowed to dispatch
    (``actions.ROLE_ALLOWED_ACTIONS``) -- a plain set of capability names carries no role.

    ``dft_comparison_evidence_present``: same deterministic, caller-computed signal
    (``cli._stage_evidence_reveals_dft_comparison``) as
    ``root_cause.validate_root_cause_classification`` takes -- computed from the FAILED STAGE'S
    OWN evidence, never trusted from the proposal. When False, a proposal authorizing fresh DFT
    (``labeling.new_dft``), a teacher relabel (``labeling.teacher_relabel``), or Student retraining
    (``student_training.retrain``) is rejected: those are exactly the costly scientific-compute
    actions R20 authorized off an evidence/provenance-gap diagnosis that had been mislabeled as a
    Teacher-vs-DFT disagreement -- none of the three is evidence-justified when the failed stage
    made no Teacher-vs-DFT comparison at all. This does not block a corrective_action requesting
    more evidence (e.g. "validate_teacher_reference" / extracting failing frames): only the three
    named authorizations above are gated, since those are the ones a genuine evidence gap can never
    itself justify.

    ``gate_alleges_accuracy_disagreement``: a second, independent signal
    (``cli._gate_alleges_accuracy_disagreement``), computed from the actual Judge vote bundle's
    ``rationale``/``required_fix`` text for THIS gate failure -- never inferred from the mere
    presence of a DFT comparison in the failed stage's evidence. A ``reference_validation``
    failure ALWAYS structurally contains a DFT comparison (it's the stage's whole purpose), so
    ``dft_comparison_evidence_present`` alone cannot catch a proposal that authorizes fresh DFT/
    teacher relabeling/retraining when no Judge actually alleged a Teacher-vs-DFT accuracy problem
    (R26 forensic finding: a REVISE driven entirely by evidence-exposure/lineage-mapping rationale
    was used to justify inferring "Teacher disagreement" and thus new DFT/retraining). When False,
    the same three actions are rejected here too, independently of
    ``dft_comparison_evidence_present``.
    """
    if proposal.failed_stage != expected_failed_stage:
        raise RecoveryPlanValidationError(
            f"proposal targets failed_stage {proposal.failed_stage!r}, expected "
            f"{expected_failed_stage!r}")
    if proposal.diagnosis_artifact_sha256 != expected_diagnosis_sha256:
        raise RecoveryPlanValidationError(
            "proposal's diagnosis_artifact_sha256 does not match the diagnosis it was given "
            "-- refusing to bind a recovery plan to the wrong (or stale) diagnosis")
    if proposal.capability not in capability_roster:
        raise RecoveryPlanValidationError(f"unregistered recovery capability: {proposal.capability!r}")
    if proposal.return_stage not in set(valid_stage_names):
        raise RecoveryPlanValidationError(f"invalid return stage: {proposal.return_stage!r}")
    if proposal.corrective_action is not None:
        role = capability_roster[proposal.capability]
        allowed = ROLE_ALLOWED_ACTIONS.get(role, set())
        if proposal.corrective_action.action_type not in allowed:
            raise RecoveryPlanValidationError(
                f"corrective_action.action_type {proposal.corrective_action.action_type!r} is not "
                f"an action {role!r} (the role responsible for capability {proposal.capability!r}) "
                f"may dispatch -- allowed: {sorted(allowed)}")
    if not (dft_comparison_evidence_present and gate_alleges_accuracy_disagreement):
        unsupported = []
        if proposal.labeling.new_dft:
            unsupported.append("labeling.new_dft")
        if proposal.labeling.teacher_relabel:
            unsupported.append("labeling.teacher_relabel")
        if proposal.student_training.retrain:
            unsupported.append("student_training.retrain")
        if unsupported:
            if not dft_comparison_evidence_present:
                reason = "the failed stage's own evidence contains no Teacher-vs-DFT comparison"
            else:
                reason = ("no Judge's required_fix/rationale for this gate failure actually "
                          "alleges a Teacher-vs-DFT accuracy/disagreement problem -- the stage's "
                          "own evidence containing a DFT comparison is not itself proof of a "
                          "disagreement")
            raise RecoveryPlanValidationError(
                f"proposal authorizes {unsupported} but {reason} -- fresh DFT, teacher relabeling, "
                "and Student retraining are not justified without it; propose an "
                "evidence-gathering corrective_action instead (e.g. extract failing frames / "
                "validate_teacher_reference) and re-diagnose once real Teacher-vs-DFT accuracy "
                "evidence exists")
    return proposal


def build_recovery_plan_draft_from_proposal(
    classification: RootCauseClassification, proposal: RecoveryPlanProposal, *,
    proposed_by: Any,
    diagnosis_artifact_path: str,
    diagnosis_artifact_sha256: str,
    expected_role: Optional[str] = None,
    estimated_cost: Optional[dict] = None,
    triggering_evidence: Optional[list] = None,
    artifact_sha256_lookup: Optional[dict] = None,
    required_input_artifact_roles: Optional[list] = None,
    expected_output_artifact_roles: Optional[list] = None,
    resource_request: Optional[dict] = None,
    policy_budget_request: Optional[dict] = None,
) -> RecoveryPlanDraft:
    """Re-project an already-validated ``RecoveryPlanProposal`` (the Orchestrator's scientific
    choices) plus the diagnosis it answers into the existing, unchanged
    ``build_recovery_plan_draft`` -- so this new reasoning-output path adds no second way to
    construct a ``RecoveryPlanDraft``, only a new, agent-driven SOURCE of its scientific fields."""
    return build_recovery_plan_draft(
        classification, proposed_by=proposed_by, failed_stage=proposal.failed_stage,
        capability=proposal.capability, return_stage=proposal.return_stage,
        proposed_changes=[change.model_dump() for change in proposal.proposed_changes],
        labeling=proposal.labeling.model_dump(),
        student_training=proposal.student_training.model_dump(),
        revalidation=proposal.revalidation.model_dump(),
        estimated_cost=estimated_cost, diagnosis_artifact_path=diagnosis_artifact_path,
        diagnosis_artifact_sha256=diagnosis_artifact_sha256, triggering_evidence=triggering_evidence,
        artifact_sha256_lookup=artifact_sha256_lookup,
        expected_role=expected_role, required_input_artifact_roles=required_input_artifact_roles,
        expected_output_artifact_roles=expected_output_artifact_roles,
        resource_request=resource_request, policy_budget_request=policy_budget_request,
        extra_recovery_context=(
            {"corrective_action": proposal.corrective_action.model_dump()}
            if proposal.corrective_action is not None else None))


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
    artifact_sha256_lookup: Optional[dict] = None,
    expected_role: Optional[str] = None,
    required_input_artifact_roles: Optional[list] = None,
    expected_output_artifact_roles: Optional[list] = None,
    resource_request: Optional[dict] = None,
    policy_budget_request: Optional[dict] = None,
    extra_recovery_context: Optional[dict] = None,
) -> RecoveryPlanDraft:
    """Bridge an evidence-bound RootCauseClassification into a typed RecoveryPlanDraft.

    The classification must already have passed
    ``root_cause.validate_root_cause_classification`` -- this function does not re-run that
    evidence binding, it only reprojects an already-validated diagnosis into recovery-plan shape.
    ``failed_stage``/``return_stage`` are supplied by the caller (typically ``return_stage`` is
    ``classification.recommended_recovery_target``) rather than inferred, so this bridge never
    silently overrides the controller's own pending-gate/stage-ordering checks -- those still run,
    authoritatively, inside propose_recovery.

    ``artifact_sha256_lookup`` (path -> sha256), when given, is the CONTROLLER's own
    already-registered hash for each artifact (``RunController.state["artifacts"]``) -- the same
    pattern as ``actions.ActionProposalBase.advisory_claimed_config_hashes`` (a model-claimed digest
    is prose audit only, never an authoritative integrity assertion). It is used in preference to
    whatever ``EvidenceReference.integrity`` the reasoning-output model self-reported, because
    ``RunController._validate_diagnosis_binding`` hash-verifies ``triggering_evidence`` against the
    real file: trusting a model's self-reported hash for that binding would let a wrong/omitted
    hash slip through the Orchestrator's own contextual acceptance only to fail later at
    ``propose_recovery``, or -- if a model ever guessed correctly by chance -- meant the binding
    was never actually anchored in controller-trusted state. Callers with no controller-registered
    artifacts to look up from (e.g. tests constructing a diagnosis by hand) may omit this and fall
    back to the model-reported ``integrity``, which the controller still fail-closed hash-verifies.
    """
    diagnosis_binding = None
    if diagnosis_artifact_path is not None or diagnosis_artifact_sha256 is not None:
        def _sha256_for(ref):
            if artifact_sha256_lookup is not None:
                return artifact_sha256_lookup.get(ref.path, "")
            return (ref.integrity or {}).get("sha256", "")
        diagnosis_binding = DiagnosisBinding(
            diagnosis_artifact_path=diagnosis_artifact_path,
            diagnosis_artifact_sha256=diagnosis_artifact_sha256,
            triggering_evidence=[
                EvidenceHashRef(path=ref.path, sha256=_sha256_for(ref))
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
        recovery_context={"run_id": classification.run_id, "diagnosed_stage": classification.stage,
                          **(extra_recovery_context or {})},
    )
