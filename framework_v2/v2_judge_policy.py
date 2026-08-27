"""Deterministic-first Judge policy for V2."""
from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase


class DecisionMode(str, Enum):
    DETERMINISTIC_GATE = "DETERMINISTIC_GATE"
    JUDGE_ALLOWED = "JUDGE_ALLOWED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"
    MISSING_SCIENTIFIC_BOUNDARY = "MISSING_SCIENTIFIC_BOUNDARY"


class V2JudgePolicy(ContractBase):
    policy_id: str
    deterministic_first: bool = True
    allowed_judge_reasons: list[str] = Field(
        default_factory=lambda: [
            "target_operationalization",
            "literature_reference_retrieval",
            "scientific_ambiguity",
            "strategy_recommendation",
            "explanation_summary",
        ]
    )
    multiple_blind_judges_required_for_numeric_gates: bool = False
    llm_cost_instrumentation_required: bool = True

    @model_validator(mode="after")
    def _v2_policy(self):
        if not self.deterministic_first:
            raise ValueError("V2 requires deterministic-first gate policy")
        if self.multiple_blind_judges_required_for_numeric_gates:
            raise ValueError("V2 must not require multiple blind Judges for routine numeric gates")
        return self


def choose_decision_mode(
    policy: V2JudgePolicy,
    *,
    evidence_sufficiency: EvidenceSufficiency,
    deterministic_failure: bool = False,
    reason: str = "",
) -> DecisionMode:
    if deterministic_failure:
        return DecisionMode.DETERMINISTIC_GATE
    if evidence_sufficiency == EvidenceSufficiency.SUFFICIENT:
        return DecisionMode.DETERMINISTIC_GATE
    if evidence_sufficiency == EvidenceSufficiency.MISSING_SCIENTIFIC_BOUNDARY:
        return DecisionMode.HUMAN_REQUIRED
    if reason in policy.allowed_judge_reasons:
        return DecisionMode.JUDGE_ALLOWED
    return DecisionMode.HUMAN_REQUIRED


class JudgeRoutingDecision(ContractBase):
    decision_id: str
    gate_id: str
    mode: DecisionMode
    reason: str
    deterministic_failure: bool = False
    judge_calls_planned: int = 0
    llm_calls_planned: int = 0
    policy_sha256: str


def route_v2_decision(
    policy: V2JudgePolicy,
    *,
    gate_id: str,
    evidence_sufficiency: EvidenceSufficiency,
    deterministic_failure: bool = False,
    reason: str = "",
) -> JudgeRoutingDecision:
    mode = choose_decision_mode(
        policy,
        evidence_sufficiency=evidence_sufficiency,
        deterministic_failure=deterministic_failure,
        reason=reason,
    )
    calls = 1 if mode == DecisionMode.JUDGE_ALLOWED else 0
    return JudgeRoutingDecision(
        decision_id=f"{gate_id}_routing",
        gate_id=gate_id,
        mode=mode,
        reason=reason,
        deterministic_failure=deterministic_failure,
        judge_calls_planned=calls,
        llm_calls_planned=calls,
        policy_sha256=policy.content_sha256(),
    )


__all__ = [
    "DecisionMode",
    "EvidenceSufficiency",
    "JudgeRoutingDecision",
    "V2JudgePolicy",
    "choose_decision_mode",
    "route_v2_decision",
]
