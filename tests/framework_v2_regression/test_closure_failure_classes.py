"""Framework V2 closure — deterministic regression matrix for every closure
failure class (directive Section AF).

Each test pins ONE typed failure mode of the closure contracts introduced by
tasks C1-C5 (semantic states, ValidationProfile, StageReviewSpec,
CanonicalReviewPacket + JudgeReview validation, first-class RecoveryPlan +
dependency invalidation, representation adequacy, Teacher baseline/coverage,
replay strategy, model adapters + material plugin registry). These are the
"scientifically-different failures must not collapse into a generic REVISE"
guarantees, verified in isolation and material-agnostically.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from framework_v2.stages import CanonicalStage
from framework_v2.states import (
    GateVerdict,
    SemanticState,
    NON_COLLAPSIBLE_STATES,
    RECOVERY_BEARING_STATES,
)
from framework_v2.recovery import (
    RecoveryPlan,
    recovery_required,
    default_failure_code_for,
    transitive_downstream,
    invalidate_downstream,
    stale_downstream_by_sha,
    DEFAULT_STATE_FAILURE_CODE,
)
from framework_v2.review_spec import (
    ReviewCriterion,
    StageReviewSpec,
    default_stage_review_specs,
    LENS_SCIENTIFIC_VALIDITY,
    LENS_EVIDENCE_PROVENANCE,
    LENS_REPRODUCIBILITY_DEPLOYMENT,
    CANONICAL_LENS_IDS,
)
from framework_v2.review_packet import (
    CanonicalReviewPacket,
    CanonicalReviewPacketCompiler,
    CriterionResult,
    JudgeReview,
    validate_judge_review,
)
from framework_v2.validation_profile import (
    ValidationChannel,
    ChannelKind,
    ValidationProfile,
)
from framework_v2.representation_adequacy import (
    RepresentationAdequacyEvidence,
    RepresentationAdequacyAssessment,
    assess_representation_adequacy,
)
from framework_v2.teacher import (
    TeacherBaselineClaim,
    TeacherBaseline,
    TeacherDistributionCoverage,
    ReplayStrategy,
)
from framework_v2.adapters import (
    ModelRole,
    ModelAdapterSpec,
    register_fact_producer,
    get_fact_producer,
    produce_material_facts,
)
from framework_v2.facts import DeterministicFact, FactVerdict


# =====================================================================
# helpers
# =====================================================================
def _fact(fid: str) -> DeterministicFact:
    return DeterministicFact(
        fact_id=fid, kind="test", observed=1.0, expected=1.0,
        verdict=FactVerdict.PASS, validator="unit", rationale="ok")


def _spec_and_packet(stage: str = CanonicalStage.ACQUISITION.value):
    spec = default_stage_review_specs()[stage]
    # inline the facts the acquisition criteria reference by id so lens reviews
    # can cite them (the ids below are arbitrary — only presence is validated).
    facts = [_fact("f1"), _fact("f2")]
    packet = CanonicalReviewPacketCompiler().compile(
        packet_id="pk", run_id="run", stage=stage,
        decision_id="d1", decision_sha256="dsha",
        validation_profile_id="vp", validation_profile_version=1,
        validation_profile_sha256="vpsha", stage_review_spec=spec,
        facts=facts, producer_rationale="r")
    return spec, packet


def _review_for_lens(spec, packet, lens, *, verdict=GateVerdict.PASS,
                     ok=True, required_fix="", fact_ids=()):
    crits = spec.criteria_for_lens(lens)
    return JudgeReview(
        review_id=f"rev-{lens}", run_id="run", stage=spec.stage, lens_id=lens,
        packet_sha256=packet.packet_sha256(),
        stage_review_spec_sha256=spec.content_sha256(),
        verdict=verdict, required_fix=required_fix,
        criteria_results=[CriterionResult(
            criterion_id=c.criterion_id, lens_id=lens, ok=ok,
            value_read="v", fact_ids=list(fact_ids)) for c in crits],
        rationale="because")


# =====================================================================
# C1 — SemanticState vocabulary invariants
# =====================================================================
def test_specialised_states_are_non_collapsible():
    for s in (SemanticState.REPRESENTATION_INSUFFICIENT, SemanticState.REVISE_SPLIT,
              SemanticState.NOT_CONVERGED, SemanticState.INVALID_JUDGE_OUTPUT,
              SemanticState.LINEAGE_LEAKAGE, SemanticState.BLIND_TEST_ACCESS_VIOLATION):
        assert s in NON_COLLAPSIBLE_STATES


def test_closure_states_registered_in_taxonomy():
    from workflow.recovery_taxonomy import resolve_failure_code
    for code in ("representation_insufficient", "split_unrepresentative",
                 "evidence_insufficient", "teacher_distribution_coverage",
                 "replay_strategy_unjustified"):
        assert resolve_failure_code(code) is not None


# =====================================================================
# C2 — StageReviewSpec / ReviewCriterion failure classes
# =====================================================================
def test_criterion_rejects_non_recovery_bearing_failure_state():
    with pytest.raises(ValidationError):
        ReviewCriterion(criterion_id="c", lens_id=LENS_SCIENTIFIC_VALIDITY,
                        question="q?", kind="qualitative",
                        failure_state=SemanticState.PASS, failure_code="dataset_coverage")


def test_criterion_rejects_unregistered_failure_code():
    with pytest.raises(ValidationError):
        ReviewCriterion(criterion_id="c", lens_id=LENS_SCIENTIFIC_VALIDITY,
                        question="q?", kind="qualitative",
                        failure_code="totally_unregistered_code_xyz")


def _crit(cid, lens, code="dataset_coverage"):
    return ReviewCriterion(criterion_id=cid, lens_id=lens, question="q?",
                           kind="qualitative", failure_code=code)


def test_spec_rejects_duplicate_criterion_id():
    with pytest.raises(ValidationError):
        StageReviewSpec(
            spec_id="s", stage=CanonicalStage.TRAINING.value,
            validation_profile_version=1,
            criteria=[_crit("dup", LENS_SCIENTIFIC_VALIDITY),
                      _crit("dup", LENS_EVIDENCE_PROVENANCE),
                      _crit("x", LENS_REPRODUCIBILITY_DEPLOYMENT)])


def test_spec_rejects_empty_lens():
    with pytest.raises(ValidationError):
        StageReviewSpec(
            spec_id="s", stage=CanonicalStage.TRAINING.value,
            validation_profile_version=1,
            criteria=[_crit("a", LENS_SCIENTIFIC_VALIDITY),
                      _crit("b", LENS_EVIDENCE_PROVENANCE)])  # R lens empty


def test_spec_rejects_undeclared_lens_reference():
    with pytest.raises(ValidationError):
        StageReviewSpec(
            spec_id="s", stage=CanonicalStage.TRAINING.value,
            validation_profile_version=1,
            criteria=[_crit("a", LENS_SCIENTIFIC_VALIDITY),
                      _crit("b", LENS_EVIDENCE_PROVENANCE),
                      _crit("c", "not_a_declared_lens")])


def test_default_specs_cover_all_twelve_stages_and_three_lenses():
    specs = default_stage_review_specs()
    assert set(specs) == {s.value for s in CanonicalStage}
    for spec in specs.values():
        for lens in CANONICAL_LENS_IDS:
            assert spec.criteria_for_lens(lens), f"{spec.stage} missing {lens}"


# =====================================================================
# C2 — ValidationProfile failure classes
# =====================================================================
def _channel(cid, obs):
    return ValidationChannel(channel_id=cid, observable=obs, kind=ChannelKind.FIDELITY)


def test_profile_rejects_duplicate_channel_id():
    with pytest.raises(ValidationError):
        ValidationProfile(profile_id="p", objective="o",
                          intended_deployment_claim="claim",
                          channels=[_channel("dup", "a"), _channel("dup", "b")])


def test_profile_rejects_empty_claim():
    with pytest.raises(ValidationError):
        ValidationProfile(profile_id="p", objective="o",
                          intended_deployment_claim="   ")


def test_profile_rejects_unsupported_property_also_a_channel():
    with pytest.raises(ValidationError):
        ValidationProfile(profile_id="p", objective="o",
                          intended_deployment_claim="claim",
                          channels=[_channel("c", "elastic_constant")],
                          unsupported_properties=["elastic_constant"])


# =====================================================================
# C3 — CanonicalReviewPacket + validate_judge_review
# =====================================================================
def test_packet_sha_is_stable_across_rebuilds():
    _, p1 = _spec_and_packet()
    _, p2 = _spec_and_packet()
    assert p1.packet_sha256() == p2.packet_sha256()


def test_compiler_rejects_stage_mismatch():
    spec = default_stage_review_specs()[CanonicalStage.TRAINING.value]
    with pytest.raises(ValueError):
        CanonicalReviewPacketCompiler().compile(
            packet_id="pk", run_id="run", stage=CanonicalStage.EVALUATION.value,
            decision_id="d", decision_sha256="s",
            validation_profile_id="vp", validation_profile_version=1,
            validation_profile_sha256="vpsha", stage_review_spec=spec)


def test_packet_rejects_duplicate_fact_ids():
    spec = default_stage_review_specs()[CanonicalStage.ACQUISITION.value]
    with pytest.raises((ValidationError, ValueError)):
        CanonicalReviewPacketCompiler().compile(
            packet_id="pk", run_id="run", stage=CanonicalStage.ACQUISITION.value,
            decision_id="d", decision_sha256="s",
            validation_profile_id="vp", validation_profile_version=1,
            validation_profile_sha256="vpsha", stage_review_spec=spec,
            facts=[_fact("same"), _fact("same")])


def test_judge_review_wrong_packet_sha_is_invalid_output():
    spec, packet = _spec_and_packet()
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY)
    tampered = review.model_copy(update={"packet_sha256": "deadbeef"})
    v = validate_judge_review(tampered, spec, packet)
    assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT


def test_judge_review_wrong_spec_sha_is_invalid_output():
    spec, packet = _spec_and_packet()
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY)
    tampered = review.model_copy(update={"stage_review_spec_sha256": "nope"})
    v = validate_judge_review(tampered, spec, packet)
    assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT


def test_judge_review_undeclared_lens_is_invalid_output():
    spec, packet = _spec_and_packet()
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY)
    # force an undeclared lens on the review + its results
    bad_results = [r.model_copy(update={"lens_id": "ghost_lens"})
                   for r in review.criteria_results]
    tampered = review.model_copy(update={"lens_id": "ghost_lens",
                                         "criteria_results": bad_results})
    v = validate_judge_review(tampered, spec, packet)
    assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT


def test_judge_review_criteria_mismatch_is_invalid_output():
    spec, packet = _spec_and_packet()
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY)
    # drop one criterion result -> coverage mismatch
    tampered = review.model_copy(update={"criteria_results": review.criteria_results[:-1]}) \
        if len(review.criteria_results) > 1 else review.model_copy(
            update={"criteria_results": review.criteria_results + [
                CriterionResult(criterion_id="extra", lens_id=LENS_SCIENTIFIC_VALIDITY,
                                ok=True, value_read="v")]})
    v = validate_judge_review(tampered, spec, packet)
    assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT


def test_judge_review_fact_not_in_packet_is_invalid_output():
    spec, packet = _spec_and_packet()
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY,
                              fact_ids=("fact_absent_from_packet",))
    v = validate_judge_review(review, spec, packet)
    assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT


def test_judge_review_pass_with_failed_criterion_is_invalid_output():
    spec, packet = _spec_and_packet()
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY,
                              verdict=GateVerdict.PASS, ok=False)
    v = validate_judge_review(review, spec, packet)
    assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT


def test_judge_review_revise_without_required_fix_is_invalid_output():
    spec, packet = _spec_and_packet()
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY,
                              verdict=GateVerdict.REVISE, ok=False, required_fix="")
    v = validate_judge_review(review, spec, packet)
    assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT


def test_valid_revise_derives_specialised_failure_state():
    # the acquisition scientific_validity lens owns aq-representation whose
    # failure_state is REPRESENTATION_INSUFFICIENT — a failed blocking criterion
    # must surface that typed state, not a plain REVISE.
    spec, packet = _spec_and_packet(CanonicalStage.ACQUISITION.value)
    review = _review_for_lens(spec, packet, LENS_SCIENTIFIC_VALIDITY,
                              verdict=GateVerdict.REVISE, ok=False,
                              required_fix="justify representation")
    v = validate_judge_review(review, spec, packet)
    assert v.valid and v.state == SemanticState.REVISE
    assert SemanticState.REPRESENTATION_INSUFFICIENT in v.derived_failure_states


# ---------------------------------------------------------------------
# C3 (Section 13) — a REVISE that contradicts the packet's authoritative facts is
# INVALID_JUDGE_OUTPUT, not a scientific REVISE. R31/C12 teacher_labeling forensic case:
# a lens returned REVISE with EVERY criterion asserting the required authoritative evidence
# is absent/unreadable (value_read None/empty, ok=False) while the SAME bound packet's
# DeterministicFacts (and the other two lenses) proved that evidence present and validated.
# ---------------------------------------------------------------------
def _revise_review(spec, packet, lens, value_read, *, required_fix="ensure the evidence is present"):
    crits = spec.criteria_for_lens(lens)
    return JudgeReview(
        review_id=f"rev-{lens}", run_id="run", stage=spec.stage, lens_id=lens,
        packet_sha256=packet.packet_sha256(),
        stage_review_spec_sha256=spec.content_sha256(),
        verdict=GateVerdict.REVISE, required_fix=required_fix,
        criteria_results=[CriterionResult(
            criterion_id=c.criterion_id, lens_id=lens, ok=False,
            value_read=value_read) for c in crits],
        rationale="the required evidence could not be found")


def test_revise_claiming_absent_evidence_contradicts_authoritative_facts_is_invalid_output():
    spec, packet = _spec_and_packet()  # packet carries PASS facts, no FAIL
    for token in ("None", "", "null", "n/a", "missing", "not found", "unreadable"):
        review = _revise_review(spec, packet, LENS_SCIENTIFIC_VALIDITY, token)
        v = validate_judge_review(review, spec, packet)
        assert not v.valid and v.state == SemanticState.INVALID_JUDGE_OUTPUT, token


def test_revise_reading_a_real_value_is_a_legitimate_revise_not_a_contradiction():
    # A REVISE that reports a REAL observed value (populated value_read) and finds it wanting is
    # untouched by the contradiction rule -- it stays a valid scientific REVISE even though the
    # packet carries authoritative PASS facts.
    spec, packet = _spec_and_packet()
    review = _revise_review(spec, packet, LENS_SCIENTIFIC_VALIDITY, "coverage_ratio=0.42")
    v = validate_judge_review(review, spec, packet)
    assert v.valid and v.state == SemanticState.REVISE


def test_absent_evidence_claim_is_valid_when_packet_has_no_authoritative_pass_fact():
    # Generality guard: the contradiction rule fires ONLY when the packet's authoritative facts
    # actually prove the evidence present (>=1 PASS and no FAIL). With no PASS fact, an "absent"
    # claim is not contradicted and remains a legitimate REVISE.
    stage = CanonicalStage.ACQUISITION.value
    spec = default_stage_review_specs()[stage]
    unchecked = DeterministicFact(
        fact_id="u1", kind="test", observed=None, expected=None,
        verdict=FactVerdict.UNCHECKED, validator="unit", rationale="not checked")
    packet = CanonicalReviewPacketCompiler().compile(
        packet_id="pk", run_id="run", stage=stage, decision_id="d1", decision_sha256="dsha",
        validation_profile_id="vp", validation_profile_version=1,
        validation_profile_sha256="vpsha", stage_review_spec=spec,
        facts=[unchecked], producer_rationale="r")
    review = _revise_review(spec, packet, LENS_SCIENTIFIC_VALIDITY, "None")
    v = validate_judge_review(review, spec, packet)
    assert v.valid and v.state == SemanticState.REVISE


def test_judge_review_contract_rejects_result_lens_mismatch():
    spec, packet = _spec_and_packet()
    crits = spec.criteria_for_lens(LENS_SCIENTIFIC_VALIDITY)
    with pytest.raises(ValidationError):
        JudgeReview(review_id="r", run_id="run", stage=spec.stage,
                    lens_id=LENS_SCIENTIFIC_VALIDITY,
                    packet_sha256=packet.packet_sha256(),
                    stage_review_spec_sha256=spec.content_sha256(),
                    verdict=GateVerdict.PASS, rationale="x",
                    criteria_results=[CriterionResult(
                        criterion_id=crits[0].criterion_id,
                        lens_id="different_lens", ok=True, value_read="v")])


# =====================================================================
# C4 — RecoveryPlan + dependency invalidation
# =====================================================================
def _plan(**overrides):
    base = dict(
        plan_id="p", run_id="run",
        failed_stage=CanonicalStage.TRAINING.value,
        failure_state=SemanticState.REVISE,
        failure_code="dataset_coverage",
        responsible_stage=CanonicalStage.TRAINING.value,
        objective="fix it",
        required_changes=["change x"],
        revalidation_criteria=["recheck y"])
    base.update(overrides)
    return RecoveryPlan(**base)


def test_recovery_plan_rejects_non_recovery_bearing_state():
    with pytest.raises(ValidationError):
        _plan(failure_state=SemanticState.EVIDENCE_INSUFFICIENT,
              failure_code="evidence_insufficient")


def test_recovery_plan_rejects_non_canonical_stage():
    with pytest.raises(ValidationError):
        _plan(failed_stage="not_a_stage")


def test_recovery_plan_rejects_reentry_after_failure():
    with pytest.raises(ValidationError):
        _plan(failed_stage=CanonicalStage.TRAINING.value,
              responsible_stage=CanonicalStage.EVALUATION.value)


def test_recovery_plan_rejects_unregistered_code():
    with pytest.raises(ValidationError):
        _plan(failure_code="nonexistent_code_qqq")


def test_recovery_plan_rejects_specialised_state_code_mismatch():
    # REPRESENTATION_INSUFFICIENT must route under representation_insufficient
    with pytest.raises(ValidationError):
        _plan(failed_stage=CanonicalStage.ACQUISITION.value,
              responsible_stage=CanonicalStage.ACQUISITION.value,
              failure_state=SemanticState.REPRESENTATION_INSUFFICIENT,
              failure_code="dataset_coverage")


def test_recovery_plan_accepts_specialised_state_with_correct_code():
    plan = _plan(failed_stage=CanonicalStage.ACQUISITION.value,
                 responsible_stage=CanonicalStage.ACQUISITION.value,
                 failure_state=SemanticState.REPRESENTATION_INSUFFICIENT,
                 failure_code="representation_insufficient")
    assert plan.failure_code == "representation_insufficient"
    assert default_failure_code_for(SemanticState.REPRESENTATION_INSUFFICIENT) == \
        "representation_insufficient"


def test_recovery_required_semantics():
    assert recovery_required(SemanticState.REVISE)
    assert recovery_required(SemanticState.NOT_CONVERGED)
    assert not recovery_required(SemanticState.EVIDENCE_INSUFFICIENT)
    assert not recovery_required(SemanticState.PASS)


def test_invalidate_downstream_transitive_and_ordered():
    S = CanonicalStage
    deps = {
        S.EVALUATION.value: [S.TRAINING.value],
        S.UNCERTAINTY.value: [S.EVALUATION.value],
        S.DEPLOYMENT_MD.value: [S.TRAINING.value],
    }
    inv = invalidate_downstream([S.TRAINING.value], deps)
    assert set(inv) == {S.EVALUATION.value, S.UNCERTAINTY.value, S.DEPLOYMENT_MD.value}
    # canonical ordering
    from framework_v2.stages import stage_index
    assert inv == sorted(inv, key=stage_index)


def test_invalidate_downstream_rejects_backwards_edge():
    S = CanonicalStage
    # declare an earlier stage as depending on a later change -> malformed
    deps = {S.ACQUISITION.value: [S.TRAINING.value]}
    with pytest.raises(ValueError):
        invalidate_downstream([S.TRAINING.value], deps)


def test_stale_downstream_by_sha_detects_change():
    S = CanonicalStage
    deps = {S.EVALUATION.value: [S.TRAINING.value]}
    prior = {S.TRAINING.value: "aaa", S.EVALUATION.value: "bbb"}
    new = {S.TRAINING.value: "ZZZ", S.EVALUATION.value: "bbb"}
    changed, invalidated = stale_downstream_by_sha(prior, new, deps)
    assert changed == [S.TRAINING.value]
    assert invalidated == [S.EVALUATION.value]


# =====================================================================
# C5 — representation adequacy (Section N)
# =====================================================================
def _adq_evidence(supports: bool):
    return RepresentationAdequacyEvidence(
        evidence_id="e", kind="discriminative_power",
        description="d", supports_adequacy=supports, fact_refs=["f1"])


def test_representation_no_support_is_insufficient():
    a = assess_representation_adequacy(
        assessment_id="a", representation_sha256="rs", scope_contract_sha256="sc",
        deployment_claim="claim", adequacy_evidence=[_adq_evidence(False)],
        alternatives_considered=["alt"])
    assert a.verdict == SemanticState.REPRESENTATION_INSUFFICIENT


def test_representation_no_alternative_is_insufficient():
    a = assess_representation_adequacy(
        assessment_id="a", representation_sha256="rs", scope_contract_sha256="sc",
        deployment_claim="claim", adequacy_evidence=[_adq_evidence(True)],
        alternatives_considered=[])
    assert a.verdict == SemanticState.REPRESENTATION_INSUFFICIENT


def test_representation_support_plus_alternative_passes():
    a = assess_representation_adequacy(
        assessment_id="a", representation_sha256="rs", scope_contract_sha256="sc",
        deployment_claim="claim", adequacy_evidence=[_adq_evidence(True)],
        alternatives_considered=["alt"])
    assert a.verdict == SemanticState.PASS


def test_representation_assessment_rejects_disallowed_verdict():
    with pytest.raises(ValidationError):
        RepresentationAdequacyAssessment(
            assessment_id="a", representation_sha256="rs", scope_contract_sha256="sc",
            deployment_claim="claim", verdict=SemanticState.FAIL)


def test_representation_pass_requires_comparative_support():
    with pytest.raises(ValidationError):
        RepresentationAdequacyAssessment(
            assessment_id="a", representation_sha256="rs", scope_contract_sha256="sc",
            deployment_claim="claim", verdict=SemanticState.PASS,
            adequacy_evidence=[_adq_evidence(True)], alternatives_considered=[])


# =====================================================================
# C5 — Teacher baseline / coverage / replay (Sections C, O, P)
# =====================================================================
def test_teacher_baseline_claim_established_needs_facts():
    with pytest.raises(ValidationError):
        TeacherBaselineClaim(claim_id="c", channel_id="ch", reference="DFT",
                             established=True, fact_refs=[])


def test_teacher_baseline_rejects_duplicate_claim_ids():
    good = TeacherBaselineClaim(claim_id="c", channel_id="ch", reference="DFT",
                               established=True, fact_refs=["f"])
    with pytest.raises(ValidationError):
        TeacherBaseline(baseline_id="b", teacher_id="t",
                        scope_contract_sha256="s", validation_profile_sha256="v",
                        reference_claims=[good, good])


def test_teacher_distribution_unassessed_cannot_pass():
    with pytest.raises(ValidationError):
        TeacherDistributionCoverage(
            coverage_id="c", teacher_id="t", descriptor="d", distance_metric="m",
            assessed=False, verdict=SemanticState.PASS)


def test_teacher_distribution_rejects_disallowed_verdict():
    with pytest.raises(ValidationError):
        TeacherDistributionCoverage(
            coverage_id="c", teacher_id="t", descriptor="d", distance_metric="m",
            assessed=True, verdict=SemanticState.FAIL)


def test_replay_nontrivial_requires_justification():
    with pytest.raises(ValidationError):
        ReplayStrategy(strategy_id="s", method="mix_prior",
                       alternatives_considered=[], comparative_evidence_refs=[])


def test_replay_fractions_must_sum_to_one():
    with pytest.raises(ValidationError):
        ReplayStrategy(strategy_id="s", method="none",
                       mixing_fractions={"a": 0.3, "b": 0.3})


def test_replay_none_method_is_valid_without_evidence():
    s = ReplayStrategy(strategy_id="s", method="none")
    assert s.method == "none"


# =====================================================================
# C5 — model adapters + material fact-producer plugin registry (generality)
# =====================================================================
def test_adapter_capability_negotiation():
    spec = ModelAdapterSpec(adapter_id="a", role=ModelRole.TEACHER,
                            model_family="anything", capabilities=["energy", "forces"])
    assert spec.supports("energy")
    assert not spec.supports("stress")


def test_fact_producer_fails_closed_on_duplicate():
    name = "test.dup_producer_xyz"
    register_fact_producer(name, lambda **k: [], replace=True)
    with pytest.raises(ValueError):
        register_fact_producer(name, lambda **k: [])


def test_produce_material_facts_rejects_non_fact():
    name = "test.bad_producer_xyz"
    register_fact_producer(name, lambda **k: ["not a fact"], replace=True)
    with pytest.raises(TypeError):
        produce_material_facts(name)


def test_produce_material_facts_accepts_facts():
    name = "test.good_producer_xyz"
    register_fact_producer(name, lambda **k: [_fact("mf1")], replace=True)
    facts = produce_material_facts(name)
    assert len(facts) == 1 and facts[0].fact_id == "mf1"


def test_unregistered_producer_fails_closed():
    with pytest.raises(KeyError):
        get_fact_producer("test.never_registered_zzz")
