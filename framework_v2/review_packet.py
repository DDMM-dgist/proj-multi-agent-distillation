"""Framework V2 — the ONE CanonicalReviewPacket compiler and the JudgeReview
vote contract (Sections H & J).

Section H requires that there be exactly one place that compiles the evidence a
Judge sees for a stage decision, that the compiled packet be serialized
canonically and receive a SHA, and that *all three* mutually-blind Judges
receive the *same* packet SHA. This module provides:

  * :class:`CanonicalReviewPacket` — a frozen, artifact-addressable contract
    carrying everything a Judge is allowed to reason over for one stage
    decision: the run/stage/attempt identity, the versioned ValidationProfile
    and StageReviewSpec (by version *and* SHA), the ScientificDecisionRecord
    under review (by id + SHA), upstream contract SHAs, the authoritative
    deterministic facts, bounded evidence/protocol references, the producer's
    rationale + alternatives, uncertainty/sensitivity summaries, and the
    declared downstream dependencies. The packet is a pure function of its
    inputs — it carries no wall-clock timestamp — so ``packet_sha256`` is
    stable across repeated builds of the same inputs. That single SHA is what
    every lens's Judge task is bound to.

  * :class:`CanonicalReviewPacketCompiler` — wraps the generic
    :class:`framework_v2.evidence_compiler.EvidenceCompiler` (same bounded,
    injected-summarizer facts pipeline) and returns exactly one packet.

Section J requires that, before a Judge result becomes a vote, it is validated
deterministically; a structurally invalid output is
``SemanticState.INVALID_JUDGE_OUTPUT`` — NOT a scientific REVISE — and it never
enters Gate aggregation. This module provides:

  * :class:`CriterionResult` / :class:`JudgeReview` — the typed vote a Judge
    returns for its single lens (per-criterion result carrying the fact ids and
    evidence refs relied on), and

  * :func:`validate_judge_review` — the deterministic validator that binds a
    review to the packet SHA + the lens's criteria and yields a
    :class:`JudgeReviewValidation` whose ``state`` is either the Judge's
    verdict class or ``INVALID_JUDGE_OUTPUT``.

Everything here is material-agnostic: no observable, material, model family, or
campaign constant appears. The packet transports whatever the campaign's
Producer put into the ValidationProfile / StageReviewSpec / facts.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase
from framework_v2.evidence_compiler import EvidenceCompiler
from framework_v2.facts import DeterministicFact, FactVerdict
from framework_v2.review_spec import ReviewCriterion, StageReviewSpec
from framework_v2.states import GateVerdict, SemanticState


# A criterion whose observed ``value_read`` normalizes to one of these tokens is asserting
# the required evidence could not be read / is absent -- an EMPTINESS claim -- rather than
# reporting a real observed value. This is what distinguishes a Judge that deterministically
# contradicts an authoritative "present" fact (Section 13) from a legitimate scientific REVISE
# that read a real value and found it wanting.
_EVIDENCE_ABSENCE_TOKENS = frozenset({
    "", "none", "null", "nil", "n/a", "na", "nan", "unknown", "missing", "absent",
    "unreadable", "not found", "not present", "not readable", "not available",
    "unavailable", "no value", "not provided", "not supplied",
})


def _asserts_evidence_absent(value_read: str) -> bool:
    """True iff a criterion's ``value_read`` asserts the required evidence is absent/unreadable
    (an emptiness claim) rather than reporting a real observed value."""
    return str(value_read).strip().lower() in _EVIDENCE_ABSENCE_TOKENS


# =====================================================================
# THE CANONICAL REVIEW PACKET (Section H)
# =====================================================================
class CanonicalReviewPacket(ContractBase):
    """The single, canonically-serialized, SHA-addressable bundle that every
    mutually-blind Judge for one stage decision receives.

    The packet deliberately carries NO wall-clock field: its identity
    (``content_sha256``) is a pure function of the decision + evidence, so all
    three lenses provably reason over the same bytes and a rebuild from the same
    inputs reproduces the same SHA.
    """
    packet_id: str
    run_id: str
    stage: str
    attempt: int = 1

    # versioned governance contracts (by version AND sha)
    validation_profile_id: str
    validation_profile_version: int
    validation_profile_sha256: str
    stage_review_spec_id: str
    stage_review_spec_version: int
    stage_review_spec_sha256: str

    # the decision under review
    decision_id: str
    decision_sha256: str

    # upstream identities this decision was derived from (name -> sha)
    upstream_contract_sha256: dict[str, str] = Field(default_factory=dict)
    scope_contract_sha256: Optional[str] = None

    # authoritative facts + bounded evidence (from the EvidenceCompiler)
    deterministic_facts: list[DeterministicFact] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    protocol_refs: list[dict[str, Any]] = Field(default_factory=list)

    # producer reasoning the Judges interpret (never authoritative on its own)
    producer_rationale: str = ""
    alternatives_considered: list[str] = Field(default_factory=list)
    uncertainty_summary: Optional[str] = None
    sensitivity_summary: Optional[str] = None

    # declared downstream dependencies (used for dependency-aware invalidation)
    downstream_dependencies: list[str] = Field(default_factory=list)

    def packet_sha256(self) -> str:
        """Alias for the content SHA — this is the value all 3 lenses bind to."""
        return self.content_sha256()

    def fact(self, fact_id: str) -> Optional[DeterministicFact]:
        for f in self.deterministic_facts:
            if f.fact_id == fact_id:
                return f
        return None

    @property
    def fact_ids(self) -> set[str]:
        return {f.fact_id for f in self.deterministic_facts}

    @model_validator(mode="after")
    def _unique_fact_ids(self):
        ids = [f.fact_id for f in self.deterministic_facts]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate DeterministicFact.fact_id in packet: {dupes}")
        return self


class CanonicalReviewPacketCompiler:
    """Compile exactly one :class:`CanonicalReviewPacket` for a stage decision.

    Wraps the generic :class:`EvidenceCompiler` so the bounded, injected-
    summarizer facts pipeline is shared with every other stage — there is no
    per-stage bespoke evidence hook. The compiler is deterministic: given the
    same inputs it returns a packet with the same ``packet_sha256``.
    """

    def __init__(self, *, evidence_compiler: Optional[EvidenceCompiler] = None):
        self._evidence = evidence_compiler or EvidenceCompiler()

    def compile(
        self,
        *,
        packet_id: str,
        run_id: str,
        stage: str,
        decision_id: str,
        decision_sha256: str,
        validation_profile_id: str,
        validation_profile_version: int,
        validation_profile_sha256: str,
        stage_review_spec: StageReviewSpec,
        attempt: int = 1,
        facts: Sequence[DeterministicFact] = (),
        artifacts: Sequence[str] = (),
        protocol_refs: Sequence[str] = (),
        upstream_contract_sha256: Optional[dict[str, str]] = None,
        scope_contract_sha256: Optional[str] = None,
        producer_rationale: str = "",
        alternatives_considered: Optional[Sequence[str]] = None,
        uncertainty_summary: Optional[str] = None,
        sensitivity_summary: Optional[str] = None,
        downstream_dependencies: Optional[Sequence[str]] = None,
    ) -> CanonicalReviewPacket:
        if stage_review_spec.stage != stage:
            raise ValueError(
                f"stage_review_spec.stage {stage_review_spec.stage!r} does not "
                f"match packet stage {stage!r}"
            )
        bundle = self._evidence.compile_stage_evidence(
            stage=stage,
            artifacts=artifacts,
            facts=facts,
            protocol_refs=protocol_refs,
            scope_contract_sha256=scope_contract_sha256,
        )
        return CanonicalReviewPacket(
            packet_id=packet_id,
            run_id=run_id,
            stage=stage,
            attempt=attempt,
            validation_profile_id=validation_profile_id,
            validation_profile_version=validation_profile_version,
            validation_profile_sha256=validation_profile_sha256,
            stage_review_spec_id=stage_review_spec.spec_id,
            stage_review_spec_version=stage_review_spec.spec_version,
            stage_review_spec_sha256=stage_review_spec.content_sha256(),
            decision_id=decision_id,
            decision_sha256=decision_sha256,
            upstream_contract_sha256=dict(upstream_contract_sha256 or {}),
            scope_contract_sha256=scope_contract_sha256,
            deterministic_facts=list(facts),
            evidence_refs=list(bundle.get("artifacts", [])),
            protocol_refs=list(bundle.get("protocol_refs", [])),
            producer_rationale=producer_rationale,
            alternatives_considered=list(alternatives_considered or []),
            uncertainty_summary=uncertainty_summary,
            sensitivity_summary=sensitivity_summary,
            downstream_dependencies=list(downstream_dependencies or []),
        )


# =====================================================================
# THE JUDGE VOTE CONTRACT + DETERMINISTIC VALIDATION (Section J)
# =====================================================================
class CriterionResult(ContractBase):
    """One Judge's per-criterion result for its single lens.

    ``fact_ids`` / ``evidence_refs`` are the packet identities the Judge relied
    on. The deterministic validator checks they exist in the packet — a Judge
    may not rely on a fact that is not in the packet it was given.
    """
    criterion_id: str
    lens_id: str
    ok: bool
    value_read: str
    fact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class JudgeReview(ContractBase):
    """The typed vote a single mutually-blind Judge returns for one lens.

    It is bound to the exact packet (``packet_sha256``) and stage review spec
    (``stage_review_spec_sha256``) it was asked to review, so a vote produced
    against a different/edited packet cannot silently be aggregated.
    """
    review_id: str
    run_id: str
    stage: str
    attempt: int = 1
    lens_id: str
    packet_sha256: str
    stage_review_spec_sha256: str
    verdict: GateVerdict
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    rationale: str
    required_fix: str = ""
    recovery_objective: Optional[str] = None

    @model_validator(mode="after")
    def _results_lens_consistent(self):
        for r in self.criteria_results:
            if r.lens_id != self.lens_id:
                raise ValueError(
                    f"CriterionResult.lens_id {r.lens_id!r} does not match "
                    f"JudgeReview.lens_id {self.lens_id!r}"
                )
        return self


class JudgeReviewValidation(ContractBase):
    """Outcome of deterministically validating a :class:`JudgeReview`.

    ``valid`` False means the output never becomes a vote: ``state`` is
    ``INVALID_JUDGE_OUTPUT`` and the Gate must not aggregate it. When
    ``valid`` is True, ``state`` is the Judge's verdict class and
    ``derived_failure_states`` lists the specialised SemanticStates implied by
    any failed *blocking* criteria (e.g. REPRESENTATION_INSUFFICIENT) so the
    Gate/recovery layer can preserve them instead of collapsing to plain REVISE.
    """
    valid: bool
    state: SemanticState
    errors: list[str] = Field(default_factory=list)
    derived_failure_states: list[SemanticState] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)


def validate_judge_review(
    review: JudgeReview,
    spec: StageReviewSpec,
    packet: CanonicalReviewPacket,
) -> JudgeReviewValidation:
    """Deterministically validate a Judge output before it can become a vote.

    Any structural failure yields ``INVALID_JUDGE_OUTPUT`` (Section J: not a
    scientific REVISE; never enters Gate aggregation). On success the returned
    state is the verdict class and ``derived_failure_states`` carries the typed
    failure semantics of any failed blocking criterion.
    """
    errors: list[str] = []

    def invalid() -> JudgeReviewValidation:
        return JudgeReviewValidation(
            valid=False,
            state=SemanticState.INVALID_JUDGE_OUTPUT,
            errors=errors,
        )

    # 1. binding: the review must be for this packet + this spec + this stage.
    if review.packet_sha256 != packet.packet_sha256():
        errors.append(
            "review.packet_sha256 does not match the canonical packet SHA "
            "(the Judge reviewed a different/edited packet)"
        )
    if review.stage_review_spec_sha256 != spec.content_sha256():
        errors.append("review.stage_review_spec_sha256 does not match the StageReviewSpec")
    if review.stage != spec.stage or review.stage != packet.stage:
        errors.append("review.stage does not match the spec/packet stage")

    # 2. lens must be one this spec declares.
    if review.lens_id not in spec.lens_ids:
        errors.append(f"review.lens_id {review.lens_id!r} is not a declared lens for this spec")
        # cannot resolve the lens's criteria -> stop here
        return invalid()

    lens_criteria: list[ReviewCriterion] = spec.criteria_for_lens(review.lens_id)
    expected_ids = [c.criterion_id for c in lens_criteria]
    got_ids = [r.criterion_id for r in review.criteria_results]

    # 3. cover exactly this lens's criteria, in order, no dupes/extras.
    if got_ids != expected_ids:
        errors.append(
            f"criteria_results {got_ids} do not match this lens's ordered "
            f"criteria {expected_ids}"
        )

    # 4. every relied-on fact id must be present in the packet.
    packet_fact_ids = packet.fact_ids
    for r in review.criteria_results:
        missing = [fid for fid in r.fact_ids if fid not in packet_fact_ids]
        if missing:
            errors.append(
                f"criterion {r.criterion_id!r} relies on fact ids not in the "
                f"packet: {sorted(missing)}"
            )

    # 5. verdict/severity consistency.
    all_ok = bool(review.criteria_results) and all(r.ok for r in review.criteria_results)
    if review.verdict == GateVerdict.PASS and not all_ok:
        errors.append("verdict PASS but not every criterion result is ok")
    if review.verdict != GateVerdict.PASS and not review.required_fix.strip():
        errors.append("REVISE/FAIL review requires a non-empty required_fix")

    # 6. deterministic contradiction with the packet's authoritative facts (Section 13).
    #    A DeterministicFact is authoritative; an LLM Judge may interpret it but may not negate
    #    it. A non-PASS verdict whose EVERY failed criterion asserts the required evidence is
    #    absent/unreadable -- while the packet's authoritative DeterministicFacts prove that
    #    evidence present and nothing authoritative failed -- deterministically contradicts the
    #    packet it is bound to. That is INVALID_JUDGE_OUTPUT (never a Gate vote), NOT a scientific
    #    REVISE. A REVISE that reports a real observed value and finds it scientifically wanting
    #    (any populated value_read) is untouched, as is any stage whose packet carries no
    #    authoritative PASS fact or carries a FAIL fact.
    if not errors and review.verdict != GateVerdict.PASS:
        failed = [r for r in review.criteria_results if not r.ok]
        asserts_absent = [r for r in failed if _asserts_evidence_absent(r.value_read)]
        facts = packet.deterministic_facts
        authoritative_present = (
            any(f.verdict == FactVerdict.PASS for f in facts)
            and not any(f.verdict == FactVerdict.FAIL for f in facts))
        if failed and len(asserts_absent) == len(failed) and authoritative_present:
            errors.append(
                "verdict contradicts the packet's authoritative facts: every failed criterion "
                "asserts the required evidence is absent/unreadable, yet the packet's "
                "DeterministicFacts prove it present and validated (deterministic contradiction, "
                "not a scientific REVISE)")

    if errors:
        return invalid()

    # valid: derive the typed failure semantics of any failed BLOCKING criterion.
    crit_by_id = {c.criterion_id: c for c in lens_criteria}
    failed_criteria: list[str] = []
    derived: list[SemanticState] = []
    for r in review.criteria_results:
        if r.ok:
            continue
        crit = crit_by_id[r.criterion_id]
        failed_criteria.append(r.criterion_id)
        if crit.severity == "blocking":
            derived.append(crit.failure_state)

    # de-duplicate while preserving order
    seen: set[SemanticState] = set()
    derived_unique = [s for s in derived if not (s in seen or seen.add(s))]

    return JudgeReviewValidation(
        valid=True,
        state=SemanticState(review.verdict.value),
        errors=[],
        derived_failure_states=derived_unique,
        failed_criteria=failed_criteria,
    )


__all__ = [
    "CanonicalReviewPacket",
    "CanonicalReviewPacketCompiler",
    "CriterionResult",
    "JudgeReview",
    "JudgeReviewValidation",
    "validate_judge_review",
]
