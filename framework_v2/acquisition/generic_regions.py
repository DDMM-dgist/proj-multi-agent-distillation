"""Framework V2 -- generic executable regions + target-relevance separation (FE-027 P2).

P1 gave the generic path a discriminative ``DomainRepresentation`` discovered deterministically
from raw structures. Two things are still missing before a brand-new material can reach a
``TargetRegimeModel`` without per-material authoring:

  * §4 -- EXECUTABLE region membership. A discovered regime is defined by axis-interval boxes
    (``DomainRegime.membership_rule``), and every pooled frame was already assigned to exactly
    one regime during discovery. This module turns that assignment into an executable classifier
    (frame -> regime_id) WITHOUT parsing the human-readable rule string -- it reads the structured
    membership evidence the discoverer recorded. Nothing here interprets a free-text predicate.

  * §5 -- separation of DISCOVERY (deterministic) from RELEVANCE (a scientific judgment). Which
    discovered regimes are the campaign's CORE_TARGET vs ADJACENT_PHYSICS / GENERATION_PATHWAY /
    BOUNDARY_GUARDRAIL / OUT_OF_TARGET_ACQUISITION is NOT derivable from raw structure alone -- it
    depends on the deployment claim. That assignment is therefore an Agent PROPOSAL (control
    plane: :class:`RegimeRelevanceProposal`), gated by a purely deterministic validator (evidence
    plane) and, downstream, the existing fixed 3-Judge committee. The core never invents roles and
    never asks a human for per-material region Python; it only checks the proposal is complete,
    well-formed, bound to the exact representation, and has a target.

The validator mirrors :mod:`framework_v2.acquisition.validators`: it returns a list of issue
strings (empty iff admissible) so the existing bounded semantic-correction retry loop can hand a
corrected proposal back through the same gate. :func:`assemble_target_regime_model` is the
fail-closed deterministic bind that refuses to build a model from an inadmissible proposal.
"""
from __future__ import annotations

from typing import Callable, Optional

from framework_v2.acquisition.contracts import (
    RelevanceRole,
    TargetRegime,
    TargetRegimeModel,
)
from framework_v2.contracts import ContractBase, DeploymentScopeContract, DomainRepresentation
from pydantic import Field

# The prefix the generic representation builder stamps onto a regime's membership evidence to
# record the pool it was discovered from (see ``build_representation``'s ``evidence_ref``). Member
# frame ids never carry it, so it cleanly separates the provenance ref from the member ids.
_POOL_EVIDENCE_PREFIX = "pool_manifest:"


class RegimeRelevanceAssignment(ContractBase):
    """One Agent-proposed target-relevance role for a single DISCOVERED regime.

    ``regime_id`` must name a regime that exists in the bound representation; ``relevance_role``
    is the target-relative role (roles are campaign-relative, never intrinsic to the region). The
    ``rationale`` is the Agent's scientific justification, reviewed by the 3-Judge committee -- it
    is never parsed or trusted by the deterministic validator."""
    regime_id: str
    relevance_role: RelevanceRole
    rationale: str = ""


class RegimeRelevanceProposal(ContractBase):
    """The control-plane artifact: an Agent's proposed relevance role for EVERY discovered regime.

    Bound by ``representation_sha256`` to the exact discovered representation it assigns roles over
    (so a proposal cannot silently be paired with a different discovery) and by
    ``scope_contract_sha256`` to the campaign's deployment scope. Region DISCOVERY produced the
    regimes deterministically; this proposal supplies only the scientific RELEVANCE judgment on top
    of them."""
    proposal_id: str
    representation_sha256: str
    scope_contract_sha256: str
    assignments: list[RegimeRelevanceAssignment] = Field(default_factory=list)


class RelevanceProposalInvalid(RuntimeError):
    """Fail-closed: a relevance proposal cannot be bound into a TargetRegimeModel.

    Carries the deterministic issue list so the surface refuses to silently proceed and the
    bounded semantic-correction retry loop can feed a corrected proposal back through the gate."""

    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues) or "invalid relevance proposal")
        self.issues = list(issues)


# --------------------------------------------------------------------------------------------
# §4 -- executable region membership (read from discovery, not parsed from prose)
# --------------------------------------------------------------------------------------------
def _regime_members(representation: DomainRepresentation) -> dict[str, tuple[str, ...]]:
    """Map each discovered regime_id to the tuple of member frame ids the discoverer recorded.

    The pool-provenance ref (``pool_manifest:...``) is stripped; only frame ids remain. This is
    the executable membership -- a deterministic lookup materialized by discovery, not a re-parse
    of the membership_rule string."""
    out: dict[str, tuple[str, ...]] = {}
    for regime in representation.regimes:
        members = tuple(
            ref for ref in regime.membership_evidence_refs
            if not ref.startswith(_POOL_EVIDENCE_PREFIX)
        )
        out[regime.regime_id] = members
    return out


def build_regime_membership(representation: DomainRepresentation) -> dict[str, tuple[str, ...]]:
    """Public accessor for the per-regime member-frame-id map (executable membership over pool)."""
    return _regime_members(representation)


def build_frame_regime_classifier(
    representation: DomainRepresentation,
) -> Callable[[str], Optional[str]]:
    """Return an executable classifier ``frame_id -> regime_id`` (or None if the frame is not in
    any discovered regime, i.e. it was unresolved during discovery).

    Deterministic and total over the pool; raises if the discovery is internally inconsistent
    (a frame claimed by two regimes) rather than silently picking one."""
    inverse: dict[str, str] = {}
    for regime_id, members in _regime_members(representation).items():
        for frame_id in members:
            if frame_id in inverse and inverse[frame_id] != regime_id:
                raise ValueError(
                    f"frame {frame_id!r} is claimed by two discovered regimes "
                    f"({inverse[frame_id]!r} and {regime_id!r}); the representation is "
                    "internally inconsistent")
            inverse[frame_id] = regime_id

    def classify(frame_id: str) -> Optional[str]:
        return inverse.get(frame_id)

    return classify


# --------------------------------------------------------------------------------------------
# §5 -- deterministic validation of the Agent-proposed relevance roles
# --------------------------------------------------------------------------------------------
def validate_relevance_proposal(
    representation: DomainRepresentation,
    proposal: RegimeRelevanceProposal,
    *,
    scope_contract: DeploymentScopeContract,
) -> list[str]:
    """Purely deterministic gate on a relevance proposal. Returns issues; empty iff admissible.

    Checks (all material-agnostic):
      1. the proposal is bound to the exact representation + scope contract it claims;
      2. every discovered regime is assigned exactly once -- no missing regime, no assignment to a
         regime that does not exist, no duplicate assignment (relevance is total over discovery);
      3. at least one regime is CORE_TARGET (a target campaign must have a target).
    Role-value validity is already guaranteed by the ``RelevanceRole`` enum on the contract.
    """
    issues: list[str] = []

    if proposal.representation_sha256 != representation.content_sha256():
        issues.append("proposal not bound to the provided representation")
    if proposal.scope_contract_sha256 != scope_contract.content_sha256():
        issues.append("proposal not bound to the provided scope contract")

    discovered_ids = [r.regime_id for r in representation.regimes]
    discovered = set(discovered_ids)
    assigned_ids = [a.regime_id for a in proposal.assignments]

    seen: set[str] = set()
    for rid in assigned_ids:
        if rid in seen:
            issues.append(f"regime assigned more than once: {rid}")
        seen.add(rid)
        if rid not in discovered:
            issues.append(f"assignment references an unknown regime: {rid}")

    for rid in discovered_ids:
        if rid not in seen:
            issues.append(f"discovered regime has no relevance assignment: {rid}")

    if not any(a.relevance_role == RelevanceRole.CORE_TARGET for a in proposal.assignments):
        issues.append("no regime is assigned CORE_TARGET (the objective must have a target)")

    return issues


def assemble_target_regime_model(
    representation: DomainRepresentation,
    proposal: RegimeRelevanceProposal,
    *,
    scope_contract: DeploymentScopeContract,
    objective_sha256: str,
    model_id: str,
) -> TargetRegimeModel:
    """Fail-closed deterministic bind: validated discovered regimes + proposed roles -> model.

    Re-runs :func:`validate_relevance_proposal` and raises :class:`RelevanceProposalInvalid` on any
    issue, so the function and the gate cannot drift. The produced ``TargetRegime.membership_rule``
    is the discovered regime's EXECUTABLE interval-box rule -- never a fresh free-text predicate.
    """
    issues = validate_relevance_proposal(
        representation, proposal, scope_contract=scope_contract)
    if issues:
        raise RelevanceProposalInvalid(issues)

    role_of = {a.regime_id: a.relevance_role for a in proposal.assignments}
    regimes = [
        TargetRegime(
            regime_id=regime.regime_id,
            label=regime.label,
            relevance_role=role_of[regime.regime_id],
            membership_rule=regime.membership_rule,
            evidence_refs=[f"representation:{representation.content_sha256()}"],
        )
        for regime in representation.regimes
    ]
    return TargetRegimeModel(
        model_id=model_id,
        objective_sha256=objective_sha256,
        descriptor=representation.descriptor,
        regimes=regimes,
    )


__all__ = [
    "RegimeRelevanceAssignment",
    "RegimeRelevanceProposal",
    "RelevanceProposalInvalid",
    "build_regime_membership",
    "build_frame_regime_classifier",
    "validate_relevance_proposal",
    "assemble_target_regime_model",
]
