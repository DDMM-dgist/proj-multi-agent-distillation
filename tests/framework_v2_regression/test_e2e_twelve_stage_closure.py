"""Framework V2 closure — 12-stage E2E through the closure contracts (Section AH)
plus a real pydantic-ai multi-stage integration (Section AI).

The pre-closure ``test_e2e_synthetic.py`` threads the older V2 contracts through a
single chain but predates the closure review objects. This module drives all
twelve canonical stages end-to-end through the *closure* contracts:

  * for each stage, compile the ONE :class:`CanonicalReviewPacket` (Section H),
  * produce the three mutually-blind lens votes bound to that packet SHA +
    StageReviewSpec SHA (Section F/H),
  * validate each vote deterministically (:func:`validate_judge_review`,
    Section J) so a structurally invalid output becomes ``INVALID_JUDGE_OUTPUT``
    and never a vote,
  * apply the unanimous-3/3-valid-PASS gate rule (Section K — no naive
    majority), and
  * confirm the committee is L2-reproducible (Section AG): all three lenses
    provably reasoned over the identical packet + decision bytes.

Section AI is exercised by :class:`RealPydanticAiMultiStageIntegration`, which
runs a real ``pydantic_ai.Agent`` (with the network-free ``TestModel``) across
multiple stages, mapping each structured model output into the closure
:class:`JudgeReview` contract and validating it — the LLM supplies only the
verdict/criterion judgments while the SHAs are bound by trusted code.

No scientific compute, no provider, no live vLLM: TestModel is deterministic.
"""
from __future__ import annotations

import unittest

from framework_v2.review_spec import (
    default_stage_review_specs, CANONICAL_LENS_IDS)
from framework_v2.review_packet import (
    CanonicalReviewPacketCompiler, JudgeReview, CriterionResult,
    validate_judge_review)
from framework_v2.stages import CANONICAL_STAGE_ORDER, stage_index
from framework_v2.states import GateVerdict, SemanticState, NON_COLLAPSIBLE_STATES
from framework_v2.judge_reproducibility import verify_l2

try:
    import pydantic_ai  # noqa: F401
    _HAS_PYDANTIC_AI = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC_AI = False


# --------------------------------------------------------------------------
# Shared helpers — compile the ONE packet + the three lens votes for a stage.
# --------------------------------------------------------------------------
def _compile_packet(stage, spec, *, decision_sha256="dec-sha"):
    return CanonicalReviewPacketCompiler().compile(
        packet_id=f"pk-{stage}", run_id="e2e-12", stage=stage,
        decision_id=f"d-{stage}", decision_sha256=decision_sha256,
        validation_profile_id="vp", validation_profile_version=1,
        validation_profile_sha256="vp-sha", stage_review_spec=spec,
        producer_rationale=f"{stage} decision rests on cited evidence")


def _pass_review(stage, spec, packet, lens):
    crits = spec.criteria_for_lens(lens)
    return JudgeReview(
        review_id=f"rev-{stage}-{lens}", run_id="e2e-12", stage=stage,
        lens_id=lens, packet_sha256=packet.packet_sha256(),
        stage_review_spec_sha256=spec.content_sha256(),
        verdict=GateVerdict.PASS,
        criteria_results=[CriterionResult(
            criterion_id=c.criterion_id, lens_id=lens, ok=True,
            value_read="verified") for c in crits],
        rationale="all criteria satisfied")


def _committee(stage, spec, packet):
    return [_pass_review(stage, spec, packet, lens) for lens in spec.lens_ids]


def _provenance(stage, packet, lens, *, decision_sha256="dec-sha"):
    """A minimal provenance mapping carrying the four L1 repro fields."""
    return dict(packet_sha256=packet.packet_sha256(),
                decision_sha256=decision_sha256, temperature=0.0, seed=7)


def _gate(reviews, spec, packet):
    """Apply the Section-K gate rule: PASS iff every one of the three lens
    outputs is deterministically VALID *and* its verdict is PASS. Returns
    (verdict, validations)."""
    validations = [validate_judge_review(r, spec, packet) for r in reviews]
    all_valid = all(v.valid for v in validations)
    all_pass = all(v.valid and v.state == SemanticState.PASS for v in validations)
    if not all_valid:
        return SemanticState.INVALID_JUDGE_OUTPUT, validations
    return (SemanticState.PASS if all_pass else SemanticState.REVISE), validations


# --------------------------------------------------------------------------
class TwelveStageClosureE2E(unittest.TestCase):
    def test_all_twelve_stages_have_a_default_spec_in_canonical_order(self):
        specs = default_stage_review_specs()
        self.assertEqual(len(CANONICAL_STAGE_ORDER), 12)
        prev = -1
        for stage_enum in CANONICAL_STAGE_ORDER:
            stage = stage_enum.value
            self.assertIn(stage, specs, f"no default spec for {stage}")
            idx = stage_index(stage)
            self.assertGreater(idx, prev, "stages must be strictly ordered")
            prev = idx
            # every default spec covers exactly the three canonical lenses
            self.assertEqual(tuple(specs[stage].lens_ids), CANONICAL_LENS_IDS)

    def test_full_unanimous_pass_chain_through_all_twelve_stages(self):
        specs = default_stage_review_specs()
        for stage_enum in CANONICAL_STAGE_ORDER:
            stage = stage_enum.value
            spec = specs[stage]
            packet = _compile_packet(stage, spec)
            reviews = _committee(stage, spec, packet)

            verdict, validations = _gate(reviews, spec, packet)
            self.assertEqual(verdict, SemanticState.PASS,
                             f"stage {stage} did not reach unanimous PASS")
            self.assertTrue(all(v.valid for v in validations), stage)
            # Section H: all three lenses bound to the identical packet SHA.
            self.assertEqual({r.packet_sha256 for r in reviews},
                             {packet.packet_sha256()}, stage)

    def test_committee_is_l2_reproducible_at_every_stage(self):
        specs = default_stage_review_specs()
        for stage_enum in CANONICAL_STAGE_ORDER:
            stage = stage_enum.value
            spec = specs[stage]
            packet = _compile_packet(stage, spec)
            recs = {lens: _provenance(stage, packet, lens)
                    for lens in spec.lens_ids}
            r = verify_l2(recs, expected_lens_ids=CANONICAL_LENS_IDS)
            self.assertTrue(r.reproducible, f"{stage}: {r.errors}")
            self.assertEqual(r.shared_packet_sha256, packet.packet_sha256())

    def test_packet_sha_is_deterministic_across_rebuilds(self):
        specs = default_stage_review_specs()
        for stage_enum in CANONICAL_STAGE_ORDER:
            stage = stage_enum.value
            spec = specs[stage]
            a = _compile_packet(stage, spec).packet_sha256()
            b = _compile_packet(stage, spec).packet_sha256()
            self.assertEqual(a, b, f"{stage} packet SHA not reproducible")

    def test_representation_revise_preserves_non_collapsible_state(self):
        """A blocking failure of the acquisition representation criterion must
        surface as REPRESENTATION_INSUFFICIENT, not a generic REVISE."""
        specs = default_stage_review_specs()
        stage = "acquisition"
        spec = specs[stage]
        packet = _compile_packet(stage, spec)
        reviews = _committee(stage, spec, packet)

        # Flip the scientific_validity lens to a valid REVISE whose failing
        # criterion is the representation-adequacy one (failure_state
        # REPRESENTATION_INSUFFICIENT, severity blocking).
        v_lens = "scientific_validity"
        idx = spec.lens_ids.index(v_lens)
        crits = spec.criteria_for_lens(v_lens)
        reviews[idx] = JudgeReview(
            review_id=f"rev-{stage}-{v_lens}", run_id="e2e-12", stage=stage,
            lens_id=v_lens, packet_sha256=packet.packet_sha256(),
            stage_review_spec_sha256=spec.content_sha256(),
            verdict=GateVerdict.REVISE,
            criteria_results=[CriterionResult(
                criterion_id=c.criterion_id, lens_id=v_lens,
                ok=(c.criterion_id != "aq-representation"),
                value_read="assessed") for c in crits],
            rationale="representation adequacy not established",
            required_fix="justify the configurational representation")

        verdict, validations = _gate(reviews, spec, packet)
        self.assertEqual(verdict, SemanticState.REVISE)
        v = validations[idx]
        self.assertTrue(v.valid)
        self.assertIn(SemanticState.REPRESENTATION_INSUFFICIENT,
                      v.derived_failure_states)
        self.assertIn(SemanticState.REPRESENTATION_INSUFFICIENT,
                      NON_COLLAPSIBLE_STATES)

    def test_tampered_packet_sha_is_invalid_judge_output_not_a_vote(self):
        specs = default_stage_review_specs()
        stage = "training"
        spec = specs[stage]
        packet = _compile_packet(stage, spec)
        reviews = _committee(stage, spec, packet)
        reviews[0] = reviews[0].model_copy(update={"packet_sha256": "deadbeef"})

        verdict, validations = _gate(reviews, spec, packet)
        self.assertEqual(verdict, SemanticState.INVALID_JUDGE_OUTPUT)
        self.assertFalse(validations[0].valid)
        self.assertEqual(validations[0].state,
                         SemanticState.INVALID_JUDGE_OUTPUT)

    def test_pass_with_a_failed_criterion_is_invalid_judge_output(self):
        """A lens that returns PASS while a criterion result is not ok is a
        structurally invalid output (Section J), never a valid PASS vote."""
        specs = default_stage_review_specs()
        stage = "evaluation"
        spec = specs[stage]
        packet = _compile_packet(stage, spec)
        reviews = _committee(stage, spec, packet)
        first = reviews[0]
        broken_results = [
            cr.model_copy(update={"ok": False}) if i == 0 else cr
            for i, cr in enumerate(first.criteria_results)]
        reviews[0] = first.model_copy(update={"criteria_results": broken_results})

        verdict, validations = _gate(reviews, spec, packet)
        self.assertEqual(verdict, SemanticState.INVALID_JUDGE_OUTPUT)
        self.assertFalse(validations[0].valid)


# --------------------------------------------------------------------------
# Section AI — real pydantic-ai integration across multiple stages.
# --------------------------------------------------------------------------
@unittest.skipUnless(_HAS_PYDANTIC_AI, "pydantic_ai not installed")
class RealPydanticAiMultiStageIntegration(unittest.TestCase):
    """Drive a real ``pydantic_ai.Agent`` (network-free TestModel) across
    several stages; the model supplies verdict + per-criterion judgments and the
    SHAs are bound by trusted code into the closure JudgeReview contract."""

    def _vote_model(self):
        from typing import Literal
        from pydantic import BaseModel

        class _CritOut(BaseModel):
            model_config = {"extra": "forbid"}
            criterion_id: str
            ok: bool
            value_read: str

        class _VoteOut(BaseModel):
            model_config = {"extra": "forbid"}
            verdict: Literal["PASS", "REVISE", "FAIL"]
            criteria: list[_CritOut]
            required_fix: str = ""

        return _VoteOut, _CritOut

    def _agent_vote(self, spec, lens, VoteOut):
        """Run a real pydantic_ai.Agent+TestModel that emits a _VoteOut for the
        lens's ordered criteria (all PASS)."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        crits = spec.criteria_for_lens(lens)
        args = {"verdict": "PASS",
                "criteria": [{"criterion_id": c.criterion_id, "ok": True,
                              "value_read": "verified"} for c in crits],
                "required_fix": ""}
        agent = Agent(TestModel(custom_output_args=args), output_type=VoteOut)
        return agent.run_sync(f"review lens {lens}").output

    def _to_judge_review(self, stage, spec, packet, lens, vote):
        return JudgeReview(
            review_id=f"rev-{stage}-{lens}", run_id="e2e-ai", stage=stage,
            lens_id=lens, packet_sha256=packet.packet_sha256(),
            stage_review_spec_sha256=spec.content_sha256(),
            verdict=GateVerdict(vote.verdict),
            criteria_results=[CriterionResult(
                criterion_id=c.criterion_id, lens_id=lens, ok=c.ok,
                value_read=c.value_read) for c in vote.criteria],
            rationale="model-produced vote",
            required_fix=vote.required_fix)

    def test_real_agent_votes_pass_multiple_stages(self):
        VoteOut, _ = self._vote_model()
        specs = default_stage_review_specs()
        # a representative spread across the lifecycle
        for stage in ("teacher_baseline", "training", "physical_validation"):
            spec = specs[stage]
            packet = _compile_packet(stage, spec)
            reviews = []
            for lens in spec.lens_ids:
                vote = self._agent_vote(spec, lens, VoteOut)
                review = self._to_judge_review(stage, spec, packet, lens, vote)
                reviews.append(review)

            verdict, validations = _gate(reviews, spec, packet)
            self.assertEqual(verdict, SemanticState.PASS,
                             f"{stage}: real-agent committee not PASS")
            self.assertTrue(all(v.valid for v in validations), stage)
            # Section H byte-identity holds for model-produced votes too.
            self.assertEqual({r.packet_sha256 for r in reviews},
                             {packet.packet_sha256()}, stage)

    def test_real_agent_committee_is_l2_reproducible(self):
        VoteOut, _ = self._vote_model()
        specs = default_stage_review_specs()
        stage = "training"
        spec = specs[stage]
        packet = _compile_packet(stage, spec)
        recs = {}
        for lens in spec.lens_ids:
            vote = self._agent_vote(spec, lens, VoteOut)
            # bind reproducibility provenance for the attempt
            recs[lens] = dict(packet_sha256=packet.packet_sha256(),
                              decision_sha256="dec-sha",
                              temperature=0.0, seed=7,
                              _verdict=vote.verdict)
        r = verify_l2(recs, expected_lens_ids=CANONICAL_LENS_IDS)
        self.assertTrue(r.reproducible, r.errors)
        self.assertEqual(r.shared_packet_sha256, packet.packet_sha256())


if __name__ == "__main__":
    unittest.main()
