"""Framework V2 — DeterministicFact vs ScientificJudgment (Section 13).

R31 saw multiple cases where a REVISE verdict was carried by an LLM Judge
citing a claim that a deterministic validator had already contradicted
(e.g. Judge said "manifest missing" while their own value_read reported
observed == expected). Framework V2 makes this structurally impossible:

  * A ``DeterministicFact`` is produced by a deterministic validator
    against real artifact content. It carries the artifact SHA it was
    computed against, a machine-readable ``kind``, the observed value,
    optionally the expected value, and a ``verdict`` in
    {``PASS``, ``FAIL``, ``UNCHECKED``}.

  * A ``ScientificJudgment`` is a Judge's interpretation. It may cite
    facts by ``fact_id`` but MUST NOT contradict any cited fact's
    verdict. If it does, ``JudgeContradiction`` classifies it and the
    downstream gate treats the judgment as ``JUDGE_CONTRADICTION`` --
    not as scientific evidence.

Together these types enforce the invariant:
"deterministic facts are authoritative; LLM judgments interpret them."

The Judge-gate wiring reads:

    contradictions = detect_judge_contradictions(judgment, cited_facts)
    if contradictions:
        # This judgment cannot be used as REVISE/FAIL evidence
        judgment_status = "JUDGE_CONTRADICTION"
    else:
        judgment_status = "USABLE"

``JUDGE_CONTRADICTION`` is recorded in the DecisionLedger with the exact
contradicting claim so the auditor sees what happened; it never gets
laundered into a scientific REVISE.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Optional

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso


class FactVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCHECKED = "UNCHECKED"


class DeterministicFact(ContractBase):
    """A single deterministic check outcome computed from real artifact
    content. Authoritative; an LLM Judge cannot negate it.

    ``kind`` categorises the fact so downstream code can look it up
    without string-matching: e.g. ``"file_exists"``, ``"sha256_match"``,
    ``"frame_count"``, ``"lineage_overlap"``, ``"convergence_status"``,
    ``"registered_checkpoint"``, ``"membership_under_declared_rule"``.
    ``artifact_sha256`` binds the fact to a specific artifact identity
    so a later Judge citing it is citing that exact identity, not a
    replaced file with the same path.
    """
    fact_id: str
    kind: str
    artifact_sha256: Optional[str] = None
    observed: Any
    expected: Any = None
    verdict: FactVerdict
    computed_at: str = Field(default_factory=utc_now_iso)
    validator: str  # dotted module path of the validator that produced this
    rationale: str = ""

    def matches(self, claim: "JudgeClaim") -> bool:
        """True iff this fact's identity aligns with a Judge claim (same
        ``kind`` and optionally same ``artifact_sha256``)."""
        if claim.about_kind != self.kind:
            return False
        if claim.about_artifact_sha256 and claim.about_artifact_sha256 != self.artifact_sha256:
            return False
        return True


class JudgeClaim(ContractBase):
    """A single claim inside a Judge's judgment. If the claim contradicts
    a cited ``DeterministicFact``, it triggers ``JUDGE_CONTRADICTION``."""
    claim_id: str
    about_kind: str
    about_artifact_sha256: Optional[str] = None
    asserted_verdict: FactVerdict
    quote: str = ""  # verbatim excerpt from the Judge's message


class ScientificJudgment(ContractBase):
    """A Judge's interpretation, with structural links to the facts it
    cites and the claims it makes. This is *not* a rubber-stamp: it is
    the Judge's voice, but its scientific validity is validated against
    the cited facts."""
    judgment_id: str
    judge_role: str
    cited_fact_ids: list[str] = Field(default_factory=list)
    claims: list[JudgeClaim] = Field(default_factory=list)
    interpretation: str
    verdict_advice: Optional[str] = None  # "PASS", "REVISE", "FAIL", or None
    at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _claim_ids_unique(self):
        ids = [c.claim_id for c in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError("JudgeClaim.claim_id must be unique within a ScientificJudgment")
        return self


class JudgeContradiction(ContractBase):
    """Records a single contradiction: a Judge claim whose asserted verdict
    disagrees with a cited fact's actual verdict."""
    contradiction_id: str
    judgment_id: str
    claim_id: str
    fact_id: str
    fact_verdict: FactVerdict
    claimed_verdict: FactVerdict
    kind: str
    artifact_sha256: Optional[str] = None
    quote: str = ""


def detect_judge_contradictions(
    judgment: ScientificJudgment,
    cited_facts: dict[str, DeterministicFact],
) -> list[JudgeContradiction]:
    """For each claim in ``judgment``, find any cited fact of matching
    identity whose ``verdict`` differs from the claim's
    ``asserted_verdict``. Return one contradiction record per
    disagreement.

    ``cited_facts`` is a mapping from ``fact_id`` to the fact record --
    typically the DecisionLedger's fact table restricted to
    ``judgment.cited_fact_ids``. Facts not in this mapping are ignored;
    the contradiction detector does not chase claims about facts the
    judgment did not cite.
    """
    contradictions: list[JudgeContradiction] = []
    for claim in judgment.claims:
        # Find any cited fact whose identity matches this claim
        for fact_id in judgment.cited_fact_ids:
            fact = cited_facts.get(fact_id)
            if fact is None:
                continue
            if not fact.matches(claim):
                continue
            if fact.verdict != claim.asserted_verdict:
                cid = _contradiction_id(judgment.judgment_id, claim.claim_id, fact_id)
                contradictions.append(JudgeContradiction(
                    contradiction_id=cid,
                    judgment_id=judgment.judgment_id,
                    claim_id=claim.claim_id,
                    fact_id=fact.fact_id,
                    fact_verdict=fact.verdict,
                    claimed_verdict=claim.asserted_verdict,
                    kind=fact.kind,
                    artifact_sha256=fact.artifact_sha256,
                    quote=claim.quote,
                ))
    return contradictions


def _contradiction_id(judgment_id: str, claim_id: str, fact_id: str) -> str:
    h = hashlib.sha256(f"{judgment_id}|{claim_id}|{fact_id}".encode("utf-8"))
    return "jc-" + h.hexdigest()[:16]


def judgment_usability(
    judgment: ScientificJudgment,
    cited_facts: dict[str, DeterministicFact],
) -> tuple[str, list[JudgeContradiction]]:
    """Returns (``status``, ``contradictions``). ``status`` is
    ``"USABLE"`` if no contradictions, else ``"JUDGE_CONTRADICTION"``.
    Callers must NOT accept a ``JUDGE_CONTRADICTION`` judgment as
    REVISE/FAIL evidence; they must record it separately in the
    DecisionLedger and continue based only on the deterministic facts.
    """
    cs = detect_judge_contradictions(judgment, cited_facts)
    return ("JUDGE_CONTRADICTION" if cs else "USABLE"), cs


__all__ = [
    "FactVerdict",
    "DeterministicFact",
    "JudgeClaim",
    "ScientificJudgment",
    "JudgeContradiction",
    "detect_judge_contradictions",
    "judgment_usability",
]
