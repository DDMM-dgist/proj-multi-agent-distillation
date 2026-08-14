"""Typed, evidence-linked acquisition/repair targeting (Priority #3, requirement #6).

    coverage evidence (coverage/report.py) / root-cause evidence (root_cause.py)
    -> Analyst/Orchestrator reasoning
    -> AcquisitionTargetProposal | DataRepairProposal (THIS MODULE, evidence-bound,
       campaign-agnostic)
    -> a RecoveryPlanDraft.proposed_changes entry (see recovery_bridge.py)
    -> RunController.propose_recovery (sole authoritative validator)

Neither typed model here selects parent structure ids, perturbation counts, or any other
concrete AcquisitionPlan value -- that selection stays a separate, explicit, campaign-specific
step (see runtimes.pydantic_ai.executors._validate_acquisition_plan, and coverage/__init__.py's
own evidence-only scoping: "It does not choose a pass/fail threshold, an acquisition count, or a
parent-selection policy"). The concrete 36x2 "Option A" campaign design
(work/r11_acquisition_design/option_a_parent_manifest.json) is exactly one historical instance of
that later, human-approved step -- never a framework rule these types encode or assume.

Both proposals are evidence-bound: `evidence_refs` cannot be empty, so a targeting/repair
proposal citing zero evidence is rejected at construction time, mirroring
`recovery_bridge.RecoveryPlanDraft`'s diagnosis-provenance binding. `requested_capability` is a
plain registered-capability name, resolved later (at propose_recovery time) against the run's own
capability roster -- exactly like `recovery_bridge.RecoveryRouting.capability` -- never a
hardcoded literal agent name.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import NonEmptyStr
from .recovery_bridge import EvidenceHashRef


def _proposal_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AcquisitionTargetProposal(BaseModel):
    """Evidence-linked description of WHERE additional structures should be acquired to close a
    coverage gap. Feeds the existing generic AcquisitionPlan only via `to_proposed_change()`
    (a RecoveryPlanDraft.proposed_changes entry) and `binding_for_acquisition_plan()` (an
    additive, OPTIONAL provenance field a caller may attach to a concrete AcquisitionPlan dict
    before it is submitted) -- it never becomes, replaces, or auto-fills an AcquisitionPlan.
    """
    model_config = {"extra": "forbid"}

    change_kind: Literal["acquisition_target"] = "acquisition_target"
    target_population: NonEmptyStr
    target_direction: NonEmptyStr
    target_slice_labels: list[str] = Field(default_factory=list)
    rationale: NonEmptyStr
    evidence_refs: list[EvidenceHashRef] = Field(min_length=1)
    # Declares what role each of `evidence_refs` plays (e.g. "candidate_population",
    # "teacher_train_partition") -- never authoritative on its own, but read by
    # RunController._validate_protected_reference_roles (see workflow/controller.py) so a
    # proposed_change that routes a run-declared protected-reference role into this proposal
    # fails closed exactly like a top-level required_input_artifact_roles/
    # expected_output_artifact_roles declaration would.
    artifact_roles: list[str] = Field(default_factory=list)
    # Advisory only -- an AcquisitionPlan's own `eligible_source_categories` remains the
    # authoritative value checked by executors._validate_acquisition_plan.
    eligible_source_categories: list[str] = Field(default_factory=list)
    requested_capability: NonEmptyStr = "acquisition"
    resource_request: dict[str, Any] = Field(default_factory=dict)

    def to_proposed_change(self) -> dict:
        return {
            # `type` mirrors `change_kind` so this entry satisfies
            # RunController.propose_recovery's own required-field check on
            # proposed_changes[*].type (used for recovery_signature and
            # recovery_policy.allowed_action_types) without introducing a second,
            # independently-settable action-type vocabulary.
            "type": self.change_kind,
            "change_kind": self.change_kind,
            "target_population": self.target_population,
            "target_direction": self.target_direction,
            "target_slice_labels": list(self.target_slice_labels),
            "rationale": self.rationale,
            "evidence_refs": [ref.model_dump() for ref in self.evidence_refs],
            "artifact_roles": list(self.artifact_roles),
            "eligible_source_categories": list(self.eligible_source_categories),
            "requested_capability": self.requested_capability,
            "resource_request": self.resource_request,
        }

    def proposal_sha256(self) -> str:
        return _proposal_sha256(self.to_proposed_change())

    def binding_for_acquisition_plan(self) -> dict:
        return {
            "change_kind": self.change_kind,
            "target_proposal_sha256": self.proposal_sha256(),
            "target_population": self.target_population,
            "target_direction": self.target_direction,
            "evidence_refs": [ref.model_dump() for ref in self.evidence_refs],
        }


class DataRepairProposal(BaseModel):
    """Evidence-linked description of WHICH existing artifact/labels need repair (e.g.
    relabeling a corrupted subset, replacing a bad manifest). Feeds the same
    RecoveryPlanDraft.proposed_changes mechanism as AcquisitionTargetProposal -- it does not
    itself repair anything; propose_recovery and, later, a trusted executor remain the only
    places a repair actually executes.
    """
    model_config = {"extra": "forbid"}

    change_kind: Literal["data_repair"] = "data_repair"
    defect_description: NonEmptyStr
    affected_artifact_refs: list[EvidenceHashRef] = Field(min_length=1)
    rationale: NonEmptyStr
    evidence_refs: list[EvidenceHashRef] = Field(min_length=1)
    # See AcquisitionTargetProposal.artifact_roles -- same generic protected-reference-role
    # cross-check hook, declaring what role each of `affected_artifact_refs`/`evidence_refs` plays.
    artifact_roles: list[str] = Field(default_factory=list)
    requested_capability: NonEmptyStr = "data_repair"
    resource_request: dict[str, Any] = Field(default_factory=dict)

    def to_proposed_change(self) -> dict:
        return {
            # See AcquisitionTargetProposal.to_proposed_change -- same `type`/`change_kind`
            # mirroring for RunController.propose_recovery's required-field check.
            "type": self.change_kind,
            "change_kind": self.change_kind,
            "defect_description": self.defect_description,
            "affected_artifact_refs": [ref.model_dump() for ref in self.affected_artifact_refs],
            "rationale": self.rationale,
            "evidence_refs": [ref.model_dump() for ref in self.evidence_refs],
            "artifact_roles": list(self.artifact_roles),
            "requested_capability": self.requested_capability,
            "resource_request": self.resource_request,
        }

    def proposal_sha256(self) -> str:
        return _proposal_sha256(self.to_proposed_change())


def bind_acquisition_plan(plan: dict, target: AcquisitionTargetProposal) -> dict:
    """Return a NEW AcquisitionPlan dict with an additive `target_binding` provenance field.

    Never mutates `plan` in place, never adds/removes/overrides any of
    executors._REQUIRED_ACQUISITION_PLAN_FIELDS -- an unbound plan remains fully valid, so this
    binding is purely optional extra provenance a caller opts into.
    """
    bound = dict(plan)
    bound["target_binding"] = target.binding_for_acquisition_plan()
    return bound
