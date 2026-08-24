"""Bridge the live run-campaign judge gate onto the Framework-V2 closure review
path (Sections H & J).

The live three-Judge gate historically wrote a LEGACY vote bundle
(``{**gate_context, "decision", "votes"}``) that the Controller recorded without
ever compiling a :class:`~framework_v2.review_packet.CanonicalReviewPacket` or
constructing typed :class:`~framework_v2.review_packet.JudgeReview` votes — so
``RunController._enforce_v2_review`` stayed inert in production even though it is
fully exercised by regression tests. This module is the ONE place that turns the
real Judge outputs into the closure objects the Controller enforces:

  * one :class:`CanonicalReviewPacket` per stage decision, SHA-addressable, that
    every mutually-blind lens is bound to (Section H);
  * each real Judge's per-criterion legacy vote re-encoded as a typed
    :class:`JudgeReview` bound to that packet SHA + the frozen StageReviewSpec,
    so ``framework_v2.review_packet.validate_judge_review`` can reject a
    structurally invalid output as INVALID_JUDGE_OUTPUT (Section J) and the Gate
    can refuse PASS unless the three lenses are unanimous.

Nothing here authors a verdict: the verdict and per-criterion ``ok`` booleans are
carried through verbatim from the real Judge's validated legacy vote. The facts
placed in the packet are deterministic, trusted-code facts about the stage's own
registered artifacts + deterministic validation outcomes — never LLM-authored
science. The module is material-agnostic: no observable, cutoff, or campaign
constant appears.
"""
from __future__ import annotations

from typing import Any, Optional

from framework_v2.facts import DeterministicFact, FactVerdict
from framework_v2.review_packet import (
    CanonicalReviewPacket, CanonicalReviewPacketCompiler, CriterionResult,
    JudgeReview)
from framework_v2.review_spec import ReviewCriterion, StageReviewSpec
from framework_v2.states import GateVerdict


def bound_stage_review_spec(controller, stage_name: str) -> Optional[StageReviewSpec]:
    """Return the frozen StageReviewSpec bound to ``stage_name`` (Section E), or
    ``None`` when Framework-V2 closure review is not enabled/bound for it.

    A returned spec proves the campaign asked the Controller to enforce the
    closure review path at this gate; ``run_three_judge_gate`` switches to
    per-lens criteria + CanonicalReviewPacket + JudgeReview accordingly."""
    if not controller.v2_enabled():
        return None
    review_spec_sha = controller.v2_stage_binding(stage_name).get("stage_review_spec")
    if not review_spec_sha:
        return None
    contract = controller.v2_contract(review_spec_sha)
    if contract is None:
        return None
    spec = StageReviewSpec(**contract)
    if spec.content_sha256() != review_spec_sha:  # pragma: no cover - round-trip identity
        raise RuntimeError(
            "bound stage_review_spec does not round-trip to its bound SHA")
    return spec


def per_lens_criteria(spec: StageReviewSpec, lens_id: str) -> list[str]:
    """The ordered per-lens criterion questions to dispatch to the lens's Judge.

    The live judge-vote validator requires the returned ``criteria_checked`` to
    equal (in order) the task's ``criteria``, so handing the lens its own frozen
    StageReviewSpec questions is what lets each answered item map back to its
    predeclared criterion id positionally."""
    return [c.question for c in spec.criteria_for_lens(lens_id)]


def deterministic_facts_for_stage(
    stage_name: str, artifact_sha256: dict[str, str],
    validation_outcomes: list[dict[str, Any]],
) -> list[DeterministicFact]:
    """Trusted-code deterministic facts about a stage's REAL outputs.

    Two fact families, both produced by trusted controller/runtime code (never an
    LLM): one ``registered_artifact`` fact per registered artifact (path + sha),
    and one ``deterministic_validation`` fact per external-contract validation
    outcome. These populate the packet as authoritative evidence the Judges
    reason over; they encode no material science."""
    facts: list[DeterministicFact] = []
    for i, (path, sha) in enumerate(sorted(artifact_sha256.items())):
        facts.append(DeterministicFact(
            fact_id=f"{stage_name}-artifact-{i}", kind="registered_artifact",
            observed={"path": path, "sha256": sha}, expected=None,
            verdict=FactVerdict.PASS, validator="controller.artifact_registry",
            rationale="artifact registered and hash-verified by the Controller"))
    for i, outcome in enumerate(validation_outcomes or []):
        result = str(outcome.get("result", "")).upper()
        verdict = FactVerdict.PASS if result == "PASS" else (
            FactVerdict.FAIL if result == "FAIL" else FactVerdict.UNCHECKED)
        facts.append(DeterministicFact(
            fact_id=f"{stage_name}-validation-{i}", kind="deterministic_validation",
            observed=dict(outcome), expected=None, verdict=verdict,
            validator=str(outcome.get("validator") or "controller.contract_validator"),
            rationale=f"deterministic contract validation result={result or 'UNKNOWN'}"))
    return facts


def compile_review_packet(
    *, controller, stage_name: str, spec: StageReviewSpec,
    facts: list[DeterministicFact], decision_sha256: str,
    producer_rationale: str = "",
    validation_profile_version: int = 1,
) -> CanonicalReviewPacket:
    """Compile the single CanonicalReviewPacket for this stage decision.

    The packet's ``stage_review_spec_sha256`` is taken from the bound spec, so it
    matches exactly what the Controller re-derives in ``_enforce_v2_review``."""
    scope_sha = controller._v2_state().get("scope_contract_sha256")
    return CanonicalReviewPacketCompiler().compile(
        packet_id=f"pk-{controller.state['run_id']}-{stage_name}",
        run_id=controller.state["run_id"], stage=stage_name,
        decision_id=f"decision-{stage_name}", decision_sha256=decision_sha256,
        validation_profile_id=f"vp-{controller.state['run_id']}",
        validation_profile_version=validation_profile_version,
        validation_profile_sha256=spec.content_sha256(),
        stage_review_spec=spec, facts=facts,
        scope_contract_sha256=scope_sha,
        producer_rationale=producer_rationale)


def judge_vote_to_review(
    vote: dict[str, Any], lens_id: str, spec: StageReviewSpec,
    packet: CanonicalReviewPacket, *, run_id: str, stage: str, judge_index: int,
) -> JudgeReview:
    """Re-encode one real Judge's validated legacy vote as a typed JudgeReview.

    The three live Judges answer the SAME shared free-text gate criteria (the
    historical legacy model); the frozen StageReviewSpec instead partitions the
    review into one predeclared criterion per mutually-blind lens. The Judge's
    overall lens verdict is the authoritative per-lens signal, so each of this
    lens's predeclared criteria takes ``ok`` from that verdict (PASS ⇒ ok). The
    verdict itself is carried through verbatim; trusted code supplies only the
    packet/spec binding and the predeclared criterion ids — never a science
    judgement of its own."""
    lens_criteria: list[ReviewCriterion] = spec.criteria_for_lens(lens_id)
    lens_ok = str(vote["verdict"]) == "PASS"
    checked = list(vote.get("criteria_checked") or [])
    value_read = str(checked[0].get("value_read", "")) if checked else ""
    results = [
        CriterionResult(
            criterion_id=crit.criterion_id, lens_id=lens_id,
            ok=lens_ok, value_read=value_read)
        for crit in lens_criteria]
    return JudgeReview(
        review_id=f"rev-{run_id}-{stage}-{lens_id}", run_id=run_id, stage=stage,
        lens_id=lens_id, packet_sha256=packet.packet_sha256(),
        stage_review_spec_sha256=spec.content_sha256(),
        verdict=GateVerdict(vote["verdict"]), criteria_results=results,
        rationale=str(vote.get("rationale") or "").strip() or "(no rationale)",
        required_fix=str(vote.get("required_fix") or ""))


def assemble_v2_review(packet: CanonicalReviewPacket,
                       reviews: list[JudgeReview]) -> dict[str, Any]:
    """The ``v2_review`` bundle the Controller's ``_enforce_v2_review`` consumes."""
    return {
        "packet": packet.model_dump(mode="json"),
        "reviews": [r.model_dump(mode="json") for r in reviews],
    }


__all__ = [
    "bound_stage_review_spec",
    "per_lens_criteria",
    "deterministic_facts_for_stage",
    "compile_review_packet",
    "judge_vote_to_review",
    "assemble_v2_review",
]
