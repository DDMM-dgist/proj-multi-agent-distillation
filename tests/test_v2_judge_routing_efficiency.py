"""V2-H07: deterministic-first Judge routing produces an accounted artifact.

Deterministic numeric gates plan zero Judge/LLM calls; only an allowed
scientific-ambiguity reason plans exactly one blind Judge call; a deterministic
failure always routes to the deterministic gate and can never be overridden by a
Judge.  Routing cost is surfaced as a provenance-carrying efficiency record.
"""
from framework_v2.error_tracking import (
    RawEfficiencyRecord,
    efficiency_from_judge_routing,
)
from framework_v2.v2_judge_policy import (
    DecisionMode,
    EvidenceSufficiency,
    JudgeRoutingDecision,
    V2JudgePolicy,
    route_v2_decision,
)


def _policy():
    return V2JudgePolicy(policy_id="judge")


def test_deterministic_force_gate_plans_zero_judge_calls():
    decision = route_v2_decision(
        _policy(),
        gate_id="force_rmse_gate",
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )
    assert isinstance(decision, JudgeRoutingDecision)
    assert decision.mode == DecisionMode.DETERMINISTIC_GATE
    assert decision.judge_calls_planned == 0
    assert decision.llm_calls_planned == 0
    assert decision.policy_sha256 == _policy().content_sha256()


def test_literature_ambiguity_plans_one_judge_call():
    decision = route_v2_decision(
        _policy(),
        gate_id="reference_retrieval_gate",
        evidence_sufficiency=EvidenceSufficiency.AMBIGUOUS,
        reason="scientific_ambiguity",
    )
    assert decision.mode == DecisionMode.JUDGE_ALLOWED
    assert decision.judge_calls_planned == 1
    assert decision.llm_calls_planned == 1


def test_deterministic_failure_cannot_be_judge_overridden():
    decision = route_v2_decision(
        _policy(),
        gate_id="force_rmse_gate",
        evidence_sufficiency=EvidenceSufficiency.AMBIGUOUS,
        deterministic_failure=True,
        reason="scientific_ambiguity",
    )
    assert decision.mode == DecisionMode.DETERMINISTIC_GATE
    assert decision.judge_calls_planned == 0
    assert decision.llm_calls_planned == 0


def test_efficiency_from_routing_carries_provenance():
    decision = route_v2_decision(
        _policy(),
        gate_id="reference_retrieval_gate",
        evidence_sufficiency=EvidenceSufficiency.AMBIGUOUS,
        reason="scientific_ambiguity",
    )
    eff = efficiency_from_judge_routing(decision)
    assert isinstance(eff, RawEfficiencyRecord)
    assert eff.judge_calls == 1
    assert eff.llm_calls == 1
    assert eff.measurement_provenance["judge_calls"] == [decision.content_sha256()]
    assert eff.measurement_provenance["llm_calls"] == [decision.content_sha256()]


def test_deterministic_routing_efficiency_is_zero_not_unknown():
    decision = route_v2_decision(
        _policy(),
        gate_id="force_rmse_gate",
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )
    eff = efficiency_from_judge_routing(decision)
    assert eff.judge_calls == 0
    assert eff.llm_calls == 0
