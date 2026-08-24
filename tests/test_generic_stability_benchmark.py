"""FE-027 P5 (§9/§13) -- repeated-live decision-stability benchmark.

The central FE-027 thesis is that a LIVE Agent's natural-language output legitimately VARIES from
run to run, but the framework's *typed* decisions do NOT -- because every governed decision is
derived deterministically from evidence (evidence plane), not parsed out of prose (control plane).
A material-agnostic acquisition planner is only trustworthy if repeating the same live decision
yields the SAME strategy class / typed roles / sizing / generation bounds / Judge verdicts /
recovery routing, even though the prose differs every time.

This benchmark simulates >=3 repeated live invocations by holding every EVIDENCE input fixed while
deliberately VARYING the prose each repeat carries (proposal rationale, Judge rationale, recovery
objective text). It then proves:

  * the prose genuinely differs across repeats (so we are measuring stability under variance, not
    trivial determinism);
  * the typed relevance roles (regime_id -> RelevanceRole) + executable membership rules are
    byte-stable across repeats, and the deterministic relevance validator accepts every repeat;
  * evidence-driven acquisition sizing (recommended_new / target_count) is identical across repeats;
  * the data/physics-derived protocol envelope bounds are identical across repeats;
  * the 3-Judge committee's typed verdicts (via the REAL validate_judge_review) are identical
    across repeats despite prose-varied rationales;
  * recovery routing (typed failure_state -> registered failure_code) is identical across repeats.

It uses only the deterministic FE-027 surfaces + the shared Judge/recovery machinery; it opens no
socket and needs no live model, so the stability property is proven hermetically and reproducibly.
"""
from __future__ import annotations

import unittest

try:
    import ase  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from test_generic_regions import _representation

_N_REPEATS = 3


def _varied_prose(kind: str, repeat: int) -> str:
    """A distinct natural-language string per repeat -- what a live LLM would legitimately vary."""
    flavors = [
        "the coverage gap in this regime clearly warrants target treatment",
        "on reflection this regime is central to the deployment objective",
        "I judge this regime core to the target given the farthest-point spread",
        "this regime anchors the primary deployment claim and must be covered",
    ]
    return f"[{kind}#run{repeat}] {flavors[repeat % len(flavors)]}"


def _typed_role_signature(model) -> tuple:
    """The prose-INDEPENDENT typed core of a TargetRegimeModel: sorted (regime, role, rule)."""
    return tuple(sorted(
        (tr.regime_id, tr.relevance_role.value, tr.membership_rule) for tr in model.regimes))


@unittest.skipUnless(_HAS_DEPS, "pydantic/ase not installed")
class GenericStabilityBenchmarkP5(unittest.TestCase):
    def test_repeated_live_decisions_are_typed_stable_under_prose_variance(self):
        from framework_v2.acquisition.contracts import RelevanceRole
        from framework_v2.acquisition.coverage_gap import build_coverage_gap_analysis
        from framework_v2.acquisition.contracts import AcquisitionPhase
        from framework_v2.acquisition.generic_coverage import (
            FrameworkSizingParams, compute_target_regime_coverage_inputs,
            recommend_acquisition_sizing)
        from framework_v2.acquisition.generic_protocol import (
            EnvelopeParams, build_md_envelope, build_perturbation_envelope)
        from framework_v2.acquisition.generic_regions import (
            RegimeRelevanceAssignment, RegimeRelevanceProposal,
            assemble_target_regime_model, validate_relevance_proposal)

        representation, pool, scope = _representation(discriminative=True)
        sizing_params = FrameworkSizingParams()
        env_params = EnvelopeParams()

        prose_seen: set[str] = set()
        role_sigs: set[tuple] = set()
        sizing_sigs: set[tuple] = set()
        pert_bound_sigs: set[tuple] = set()
        md_bound_sigs: set[tuple] = set()

        for r in range(_N_REPEATS):
            # A live Agent proposes the SAME typed roles (first regime CORE_TARGET, rest guardrail)
            # but with DIFFERENT prose rationale each repeat.
            assignments = []
            for i, regime in enumerate(representation.regimes):
                role = (RelevanceRole.CORE_TARGET if i == 0
                        else RelevanceRole.BOUNDARY_GUARDRAIL)
                rationale = _varied_prose("relevance", r)
                prose_seen.add(rationale)
                assignments.append(RegimeRelevanceAssignment(
                    regime_id=regime.regime_id, relevance_role=role, rationale=rationale))
            proposal = RegimeRelevanceProposal(
                proposal_id=f"p5-run{r}",
                representation_sha256=representation.content_sha256(),
                scope_contract_sha256=scope.content_sha256(),
                assignments=assignments)

            # Deterministic validator accepts every repeat (prose never affects admissibility).
            self.assertEqual(
                validate_relevance_proposal(representation, proposal, scope_contract=scope), [])

            model = assemble_target_regime_model(
                representation, proposal, scope_contract=scope,
                objective_sha256="p5-obj", model_id=f"p5-trm-{r}")
            role_sigs.add(_typed_role_signature(model))

            # Evidence-driven sizing is a pure function of pool geometry + roles, not prose.
            cov_inputs = compute_target_regime_coverage_inputs(
                pool, representation, model, params=sizing_params)
            coverage = build_coverage_gap_analysis(
                analysis_id=f"cga-{r}", phase=AcquisitionPhase.INITIAL,
                target_regime_model_sha256=model.content_sha256(),
                region_resolution_sha256="rr", regime_inputs=list(cov_inputs),
                saturation_threshold=sizing_params.saturation_threshold)
            sizing = recommend_acquisition_sizing(
                coverage, params=sizing_params, sizing_id=f"sz-{r}")
            sizing_sigs.add((
                tuple(sorted(sizing.recommended_new.items())),
                tuple(sorted(sizing.target_count.items()))))

            # Generation bounds derive from the pool's own nearest-neighbor scale + versioned knobs.
            pert = build_perturbation_envelope(pool, params=env_params, envelope_id=f"pe-{r}")
            md = build_md_envelope(pool, params=env_params, envelope_id=f"me-{r}")
            pert_bound_sigs.add((
                tuple(sorted(pert.param_bounds.items())),
                tuple(sorted(pert.output_admissibility.items()))))
            md_bound_sigs.add((
                tuple(sorted(md.param_bounds.items())),
                tuple(md.presence_required_keys),
                tuple(md.unbounded_from_raw_structure)))

        # Prose genuinely varied -> we measured stability under variance, not trivial determinism.
        self.assertGreater(len(prose_seen), 1)
        # Every typed decision axis collapsed to exactly ONE value across all repeats.
        self.assertEqual(len(role_sigs), 1, "relevance roles/rules not prose-stable")
        self.assertEqual(len(sizing_sigs), 1, "acquisition sizing not prose-stable")
        self.assertEqual(len(pert_bound_sigs), 1, "perturbation bounds not prose-stable")
        self.assertEqual(len(md_bound_sigs), 1, "MD envelope bounds not prose-stable")

    def test_three_judge_verdicts_are_stable_under_prose_variance(self):
        from framework_v2.review_packet import (
            CanonicalReviewPacketCompiler, CriterionResult, JudgeReview,
            validate_judge_review)
        from framework_v2.review_spec import default_stage_review_specs
        from framework_v2.stages import CanonicalStage
        from framework_v2.states import GateVerdict
        from framework_v2.facts import DeterministicFact, FactVerdict

        stage = CanonicalStage.ACQUISITION.value
        spec = default_stage_review_specs()[stage]
        facts = [DeterministicFact(
            fact_id=f, kind="test", observed=1.0, expected=1.0,
            verdict=FactVerdict.PASS, validator="unit", rationale="ok")
            for f in ("f1", "f2")]
        packet = CanonicalReviewPacketCompiler().compile(
            packet_id="p5-pk", run_id="run", stage=stage,
            decision_id="d1", decision_sha256="dsha",
            validation_profile_id="vp", validation_profile_version=1,
            validation_profile_sha256="vpsha", stage_review_spec=spec,
            facts=facts, producer_rationale="fixed decision under review")

        prose_seen: set[str] = set()
        committee_sigs: set[tuple] = set()

        for r in range(_N_REPEATS):
            per_lens_states: list[str] = []
            for lens in spec.lens_ids:
                rationale = _varied_prose("judge", r) + f" ({lens})"
                prose_seen.add(rationale)
                crits = spec.criteria_for_lens(lens)
                review = JudgeReview(
                    review_id=f"rev-{lens}-{r}", run_id="run", stage=stage, lens_id=lens,
                    packet_sha256=packet.packet_sha256(),
                    stage_review_spec_sha256=spec.content_sha256(),
                    verdict=GateVerdict.PASS, required_fix="",
                    criteria_results=[CriterionResult(
                        criterion_id=c.criterion_id, lens_id=lens, ok=True,
                        value_read="v", fact_ids=[]) for c in crits],
                    rationale=rationale)
                v = validate_judge_review(review, spec, packet)
                self.assertTrue(v.valid)
                per_lens_states.append(v.state.value)
            committee_sigs.add(tuple(per_lens_states))

        self.assertGreater(len(prose_seen), 1)
        self.assertEqual(len(committee_sigs), 1, "committee verdicts not prose-stable")
        # The single stable committee outcome is unanimous PASS.
        self.assertEqual(committee_sigs.pop(), (GateVerdict.PASS.value,) * 3)

    def test_recovery_routing_is_stable_under_prose_variance(self):
        from framework_v2.recovery import (
            DEFAULT_STATE_FAILURE_CODE, RecoveryPlan, default_failure_code_for)
        from framework_v2.stages import CanonicalStage
        from framework_v2.states import SemanticState

        state = SemanticState.REPRESENTATION_INSUFFICIENT
        failed = CanonicalStage.REFERENCE_VALIDATION.value
        code_seen: set[str] = set()
        prose_seen: set[str] = set()

        for r in range(_N_REPEATS):
            obj = _varied_prose("recovery", r)
            prose_seen.add(obj)
            plan = RecoveryPlan(
                plan_id=f"rp-{r}", run_id="run", failed_stage=failed,
                failure_state=state, failure_code=default_failure_code_for(state),
                responsible_stage=failed, objective=obj,
                required_changes=[f"broaden the representation ({r})"],
                revalidation_criteria=[f"re-assess adequacy ({r})"])
            code_seen.add(plan.failure_code)

        self.assertGreater(len(prose_seen), 1)
        # Typed routing is a pure function of the failure_state, never of the prose.
        self.assertEqual(code_seen, {DEFAULT_STATE_FAILURE_CODE[state]})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
