"""Framework V2 — Teacher-first baseline, Teacher-distribution coverage, and
replay/data-mixing as scientific decisions (Sections C, O, P).

Three generic, material-agnostic contracts the closure directive requires but
that had no typed home:

  * :class:`TeacherBaseline` (Section C) — the Teacher's reference agreement
    must be *established before* Student design. The baseline binds validated
    per-channel claims (against DFT/experiment/other references) to
    deterministic facts, so "Teacher-first" is a checkable contract, not a
    convention.

  * :class:`TeacherDistributionCoverage` (Section O) — a Data-curator decision:
    does the distillation data actually cover the Teacher's own training
    distribution? Distilling only where the Teacher is itself extrapolating is a
    silent failure; this makes the assessment explicit and yields a typed verdict
    routed under the ``teacher_distribution_coverage`` failure code.

  * :class:`ReplayStrategy` (Section P) — the replay / data-mixing choice is a
    scientific decision requiring comparative evidence over meaningful
    alternatives, not an unexamined default. An unjustified strategy routes under
    the ``replay_strategy_unjustified`` failure code.

Nothing here names a material, model family, or observable: channels, references,
descriptors, and mixing sources are all campaign OUTPUTS bound by id/ref/SHA.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso
from framework_v2.states import SemanticState


# =====================================================================
# TeacherBaseline (Section C — Teacher-first)
# =====================================================================
class TeacherBaselineClaim(ContractBase):
    """One validated agreement claim about the Teacher against a reference.

    ``channel_id`` links to a :class:`~framework_v2.validation_profile.ValidationChannel`.
    An ``established`` claim must cite the deterministic facts that establish it.
    """
    claim_id: str
    channel_id: str
    reference: str
    established: bool
    fact_refs: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _established_needs_facts(self):
        if self.established and not self.fact_refs:
            raise ValueError(
                f"TeacherBaselineClaim {self.claim_id!r} is marked established but cites "
                f"no deterministic fact_refs"
            )
        return self


class TeacherBaseline(ContractBase):
    """The Teacher's reference baseline, established before Student design.

    ``established_before_student_design`` records the Teacher-first invariant;
    the Controller is the authoritative enforcer of stage ordering, but the
    baseline itself carries the claim so an auditor sees it explicitly.
    """
    baseline_id: str
    teacher_id: str
    scope_contract_sha256: str
    validation_profile_sha256: str
    reference_claims: list[TeacherBaselineClaim] = Field(min_length=1)
    established_before_student_design: bool = True
    established_at: str = Field(default_factory=utc_now_iso)

    def claim(self, claim_id: str) -> Optional[TeacherBaselineClaim]:
        for c in self.reference_claims:
            if c.claim_id == claim_id:
                return c
        return None

    def established_claims(self) -> list[TeacherBaselineClaim]:
        return [c for c in self.reference_claims if c.established]

    @model_validator(mode="after")
    def _unique_claim_ids(self):
        ids = [c.claim_id for c in self.reference_claims]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate TeacherBaselineClaim.claim_id: {dupes}")
        return self


# =====================================================================
# TeacherDistributionCoverage (Section O — Data-curator)
# =====================================================================
class TeacherDistributionCoverage(ContractBase):
    """Whether the distillation data covers the Teacher's OWN training
    distribution (so the Student is distilled where the Teacher is reliable, not
    where the Teacher itself extrapolates).

    Verdict is ``PASS`` or ``REVISE`` (the latter routed under the
    ``teacher_distribution_coverage`` failure code by the recovery layer). An
    ``assessed`` False coverage can never PASS — unassessed is not adequate.
    """
    coverage_id: str
    teacher_id: str
    descriptor: str
    distance_metric: str
    assessed: bool
    in_distribution_fraction: Optional[float] = None
    out_of_distribution_regions: list[str] = Field(default_factory=list)
    fact_refs: list[str] = Field(default_factory=list)
    verdict: SemanticState
    rationale: str = ""
    assessed_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _verdict_allowed(self):
        allowed = {SemanticState.PASS, SemanticState.REVISE}
        if self.verdict not in allowed:
            raise ValueError(
                f"TeacherDistributionCoverage.verdict must be one of "
                f"{sorted(s.value for s in allowed)}, got {self.verdict.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _unassessed_cannot_pass(self):
        if self.verdict == SemanticState.PASS and not self.assessed:
            raise ValueError(
                "TeacherDistributionCoverage cannot PASS while assessed is False — "
                "unassessed coverage is not adequate coverage"
            )
        return self

    @model_validator(mode="after")
    def _fraction_range(self):
        if self.in_distribution_fraction is not None:
            if not 0.0 <= self.in_distribution_fraction <= 1.0:
                raise ValueError("in_distribution_fraction must be in [0, 1]")
        return self


# =====================================================================
# ReplayStrategy (Section P — data-mixing as a scientific decision)
# =====================================================================
class ReplayStrategy(ContractBase):
    """The replay / data-mixing decision.

    ``method`` ``"none"`` means no replay (no mixing of prior/other sources). For
    any non-``"none"`` method the choice is a scientific one and must carry a
    comparison against meaningful alternatives plus supporting evidence; without
    it the strategy is unjustified (``replay_strategy_unjustified``).
    """
    strategy_id: str
    method: str
    mixing_fractions: dict[str, float] = Field(default_factory=dict)
    alternatives_considered: list[str] = Field(default_factory=list)
    comparative_evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _nontrivial_requires_justification(self):
        if self.method != "none":
            if not self.alternatives_considered:
                raise ValueError(
                    f"replay method {self.method!r} requires alternatives_considered "
                    f"(a non-trivial data-mixing choice must be comparative)"
                )
            if not self.comparative_evidence_refs:
                raise ValueError(
                    f"replay method {self.method!r} requires comparative_evidence_refs "
                    f"(the choice must be evidence-based, not an unexamined default)"
                )
        return self

    @model_validator(mode="after")
    def _fractions_valid(self):
        for src, frac in self.mixing_fractions.items():
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"mixing fraction for {src!r} must be in [0, 1] (got {frac})")
        if self.mixing_fractions:
            total = sum(self.mixing_fractions.values())
            if not 0.999 <= total <= 1.001:
                raise ValueError(
                    f"ReplayStrategy.mixing_fractions must sum to 1.0 when present (got {total})"
                )
        return self


__all__ = [
    "TeacherBaselineClaim",
    "TeacherBaseline",
    "TeacherDistributionCoverage",
    "ReplayStrategy",
]
