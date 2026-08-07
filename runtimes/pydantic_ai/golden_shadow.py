"""Golden-shadow comparison harness (Phase 7).

Compares a frozen Claude-baseline Judge vote against a PydanticAI Judge vote on the SAME frozen
inputs (artifact + hash, ordered criteria, assigned lens, role prompt/spec, validation profile,
gate context). String exact-match is NOT the metric; the metric schema below captures agreement
and, more importantly, safety signals (false-PASS, wrong-lens, nonexistent-artifact citation,
unauthorized tool request, provenance completeness, cost).

No actual provider is called here. Until a real provider run is executed (behind approval), the
harness status is HARNESS_READY_PROVIDER_RUN_PENDING and any metrics computed from fixtures/
TestModel MUST NOT be reported as an actual golden-shadow comparison.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

STATUS_HARNESS_READY = "HARNESS_READY_PROVIDER_RUN_PENDING"
STATUS_ACTUAL_COMPARISON = "ACTUAL_COMPARISON_COMPLETE"


class GoldenTask(BaseModel):
    """A frozen Judge task: everything that must be identical across both runtimes."""
    model_config = {"extra": "forbid"}
    task_id: str
    artifact_path: str
    artifact_sha256: str
    ordered_criteria: list[str]
    assigned_lens: str
    role_prompt_sha256: str
    validation_profile: str
    gate_context_sha256: str


class ShadowComparison(BaseModel):
    model_config = {"extra": "forbid"}
    task_id: str
    lens: str
    verdict_agreement: bool
    criterion_coverage: float                 # fraction of ordered criteria the candidate checked
    evidence_completeness: float
    unsupported_claim: bool
    false_pass: bool                          # candidate PASS where baseline did not
    malformed_output: bool
    wrong_lens_output: bool
    nonexistent_artifact_citation: bool
    unauthorized_tool_request: bool
    provider_failure: bool
    provenance_complete: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    estimated_cost: Optional[float] = None


def compare_votes(task: GoldenTask, baseline_vote: dict, candidate_vote: dict, *,
                  candidate_provenance=None, source: str = "fixture") -> ShadowComparison:
    """Compute the comparison metrics for one task. ``source`` records whether the candidate came
    from a fixture/TestModel or a real provider — a fixture result is never an actual comparison."""
    crit = list(task.ordered_criteria)
    checked = [c.get("criterion") for c in candidate_vote.get("criteria_checked", [])]
    coverage = (sum(1 for c in crit if c in checked) / len(crit)) if crit else 0.0
    b_verdict = baseline_vote.get("verdict")
    c_verdict = candidate_vote.get("verdict")
    wrong_lens = candidate_vote.get("review_lens") != task.assigned_lens
    false_pass = (c_verdict == "PASS" and b_verdict in ("REVISE", "FAIL"))
    prov = candidate_provenance
    return ShadowComparison(
        task_id=task.task_id, lens=task.assigned_lens,
        verdict_agreement=(b_verdict == c_verdict),
        criterion_coverage=coverage,
        evidence_completeness=1.0 if set(crit) <= set(checked) else coverage,
        unsupported_claim=bool(candidate_vote.get("_unsupported_claim", False)),
        false_pass=false_pass, malformed_output=False, wrong_lens_output=wrong_lens,
        nonexistent_artifact_citation=bool(candidate_vote.get("_nonexistent_citation", False)),
        unauthorized_tool_request=bool(getattr(prov, "tool_invocations", None) and
                                       any(not t.ok for t in prov.tool_invocations
                                           if getattr(t, "tool", "") == "__unauthorized__")),
        provider_failure=bool(getattr(prov, "failure_category", "")),
        provenance_complete=bool(prov is None or (getattr(prov, "attempt_id", "") and
                                                  getattr(prov, "prompt_sha256", ""))),
        prompt_tokens=getattr(prov, "prompt_tokens", 0) if prov else 0,
        completion_tokens=getattr(prov, "completion_tokens", 0) if prov else 0,
        latency_s=getattr(prov, "latency_s", 0.0) if prov else 0.0)


class ShadowRunReport(BaseModel):
    model_config = {"extra": "forbid"}
    status: str
    source: str
    n_tasks: int
    comparisons: list[ShadowComparison] = Field(default_factory=list)
    # aggregate acceptance signals (production-gate criteria)
    false_pass_count: int = 0
    wrong_lens_accepted: int = 0
    nonexistent_citation_count: int = 0
    provenance_incomplete_count: int = 0


def build_report(task_votes, *, source: str = "fixture") -> ShadowRunReport:
    """task_votes: list of (GoldenTask, baseline_vote, candidate_vote, provenance). With a fixture/
    TestModel source the status stays HARNESS_READY_PROVIDER_RUN_PENDING — NOT an actual comparison."""
    comps = [compare_votes(t, b, c, candidate_provenance=p, source=source)
             for (t, b, c, p) in task_votes]
    status = STATUS_ACTUAL_COMPARISON if source == "provider" else STATUS_HARNESS_READY
    return ShadowRunReport(
        status=status, source=source, n_tasks=len(comps), comparisons=comps,
        false_pass_count=sum(1 for c in comps if c.false_pass),
        wrong_lens_accepted=sum(1 for c in comps if c.wrong_lens_output),
        nonexistent_citation_count=sum(1 for c in comps if c.nonexistent_artifact_citation),
        provenance_incomplete_count=sum(1 for c in comps if not c.provenance_complete))
