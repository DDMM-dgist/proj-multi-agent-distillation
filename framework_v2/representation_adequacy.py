"""Framework V2 — representation-adequacy as a first-class decision (Section N).

The closure directive makes representation adequacy a scientific decision in its
own right: good *internal* coverage of a chosen configurational representation
does not prove that representation is adequate for the deployment claim. A
campaign must show the representation discriminates the physics the claim
depends on (and does so at least as well as a meaningful alternative). When it
cannot, the outcome is the typed state ``REPRESENTATION_INSUFFICIENT`` — never a
silent PASS on the strength of coverage alone, and never a generic REVISE.

This module is the authoritative home referenced by
:mod:`framework_v2.states` for that state. It is material-agnostic: the
representation, the descriptor, and the evidence are all campaign OUTPUTS bound
here by SHA/ref, never enumerated by the core.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso
from framework_v2.states import SemanticState


class RepresentationSpec(ContractBase):
    """A first-class, typed description of ONE candidate structural representation.

    A representation is a campaign OUTPUT, never enumerated by the core: the spec
    names the descriptor, the feature variables that span the descriptor space, and
    the deterministic ``provenance`` recipe that computes those features from raw
    structures (e.g. ``generic_raw_structure_v1``). It carries no material name,
    phase label, or hard-coded cutoff — only the shape of the representation and how
    it is computed — so the same contract describes a generic raw-structure
    representation and a specialized plugin's representation identically.

    Multiple specs are compared in :class:`RepresentationAdequacyAssessment` (adequacy
    is comparative), and the chosen spec's ``content_sha256`` is the representation
    identity bound downstream. ``feature_variables`` unions the continuous and
    categorical variable names so a reader sees the full descriptor-space axis set.
    """
    spec_id: str
    descriptor: str
    kind: Literal["continuous", "categorical", "hierarchical", "hybrid"]
    continuous_variables: list[str] = Field(default_factory=list)
    categorical_variables: list[str] = Field(default_factory=list)
    provenance: str
    scope_contract_sha256: str

    @model_validator(mode="after")
    def _has_at_least_one_variable(self):
        if not self.continuous_variables and not self.categorical_variables:
            raise ValueError(
                "a RepresentationSpec must declare at least one continuous or "
                "categorical feature variable (an empty descriptor space is not a "
                "representation)"
            )
        return self

    @property
    def feature_variables(self) -> list[str]:
        return list(self.continuous_variables) + list(self.categorical_variables)


class RepresentationAdequacyEvidence(ContractBase):
    """One piece of evidence bearing on whether the representation is adequate
    for the deployment claim.

    ``supports_adequacy`` records the direction of the evidence; ``fact_refs``
    binds it to authoritative :class:`~framework_v2.facts.DeterministicFact`
    ids so the assessment cannot rest on prose alone.
    """
    evidence_id: str
    kind: str            # campaign-chosen: e.g. discriminative_power, claim_correlation, sensitivity
    description: str
    supports_adequacy: bool
    fact_refs: list[str] = Field(default_factory=list)


class RepresentationAdequacyAssessment(ContractBase):
    """A first-class, versioned assessment of representation adequacy.

    The verdict is restricted to ``PASS`` or ``REPRESENTATION_INSUFFICIENT``.
    A PASS requires (a) at least one supporting piece of adequacy evidence and
    (b) that at least one meaningful alternative representation was considered —
    adequacy is comparative, so a representation accepted without any alternative
    ever weighed cannot be declared adequate here.
    """
    assessment_id: str
    representation_sha256: str
    scope_contract_sha256: str
    deployment_claim: str
    alternatives_considered: list[str] = Field(default_factory=list)
    adequacy_evidence: list[RepresentationAdequacyEvidence] = Field(default_factory=list)
    verdict: SemanticState
    rationale: str = ""
    assessed_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _verdict_in_allowed(self):
        allowed = {SemanticState.PASS, SemanticState.REPRESENTATION_INSUFFICIENT}
        if self.verdict not in allowed:
            raise ValueError(
                f"RepresentationAdequacyAssessment.verdict must be one of "
                f"{sorted(s.value for s in allowed)}, got {self.verdict.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _pass_requires_comparative_support(self):
        if self.verdict == SemanticState.PASS:
            if not any(e.supports_adequacy for e in self.adequacy_evidence):
                raise ValueError(
                    "a PASS adequacy verdict requires at least one piece of evidence "
                    "that supports adequacy (coverage alone is not adequacy)"
                )
            if not self.alternatives_considered:
                raise ValueError(
                    "a PASS adequacy verdict requires at least one alternative "
                    "representation to have been considered (adequacy is comparative)"
                )
        return self


def assess_representation_adequacy(
    *,
    assessment_id: str,
    representation_sha256: str,
    scope_contract_sha256: str,
    deployment_claim: str,
    adequacy_evidence: list[RepresentationAdequacyEvidence],
    alternatives_considered: Optional[list[str]] = None,
    rationale: str = "",
) -> RepresentationAdequacyAssessment:
    """Deterministically derive the adequacy verdict from the evidence.

    PASS iff there is at least one supporting piece of evidence AND at least one
    alternative was considered; otherwise ``REPRESENTATION_INSUFFICIENT``. The
    returned contract re-validates the same invariants, so the function and the
    contract cannot drift.
    """
    alts = list(alternatives_considered or [])
    has_support = any(e.supports_adequacy for e in adequacy_evidence)
    verdict = (
        SemanticState.PASS
        if (has_support and alts)
        else SemanticState.REPRESENTATION_INSUFFICIENT
    )
    return RepresentationAdequacyAssessment(
        assessment_id=assessment_id,
        representation_sha256=representation_sha256,
        scope_contract_sha256=scope_contract_sha256,
        deployment_claim=deployment_claim,
        alternatives_considered=alts,
        adequacy_evidence=list(adequacy_evidence),
        verdict=verdict,
        rationale=rationale,
    )


__all__ = [
    "RepresentationSpec",
    "RepresentationAdequacyEvidence",
    "RepresentationAdequacyAssessment",
    "assess_representation_adequacy",
]
