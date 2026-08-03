"""Pydantic parsing/provenance models for the PydanticAI runtime.

Design (Plan B): the JSON Schemas under ``orchestration/schema/`` stay canonical.
These models mirror those contracts closely enough to give a PydanticAI agent a typed
``output_type``, but they are NOT the source of truth: after parsing, a result is
``.model_dump()``-ed and revalidated by ``orchestration.exchange.validate_agent_response``.
Pydantic type-validation and physics/domain validation stay distinct.

Requires ``pydantic`` (optional dependency). Not imported by any core package.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --- Contract mirrors (typed output for the LLM) --------------------------------

class EvidenceReference(BaseModel):
    """A pointer to an artifact the agent relied on. Mirrors the exchange 'reference'."""
    model_config = {"extra": "forbid"}
    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    integrity: Optional[dict[str, Any]] = None


class CriterionCheck(BaseModel):
    model_config = {"extra": "forbid"}
    criterion: str = Field(min_length=1)
    value_read: Any = None
    ok: bool


class JudgeVoteModel(BaseModel):
    """Typed mirror of judge_vote.schema.json (result_contract == 'JudgeVote')."""
    model_config = {"extra": "forbid"}
    review_lens: str = Field(min_length=1)
    verdict: Literal["PASS", "REVISE", "FAIL"]
    criteria_checked: list[CriterionCheck]
    rationale: str = Field(min_length=1)
    required_fix: str = ""


class RequestedApproval(BaseModel):
    model_config = {"extra": "forbid"}
    boundary: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class AgentResultModel(BaseModel):
    """Typed mirror of agent_result.schema.json (result_contract == 'AgentResult')."""
    model_config = {"extra": "forbid"}
    schema_version: int = 1
    task_id: str
    agent: str
    status: str
    summary: str = Field(min_length=1)
    artifacts: list[EvidenceReference] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    requested_approval: Optional[RequestedApproval] = None
    next_actions: list[str] = Field(default_factory=list)


# --- Provenance records (runtime-owned, not a scientific contract) ---------------

class ToolInvocationRecord(BaseModel):
    """One read-only tool call the agent made, for the audit trail."""
    model_config = {"extra": "forbid"}
    tool: str
    argument: str
    ok: bool
    detail: str = ""


class ValidationErrorRecord(BaseModel):
    """A validation failure, preserved rather than hidden."""
    model_config = {"extra": "forbid"}
    stage: Literal["pydantic_parse", "contract_validation", "artifact_validation"]
    message: str


class RuntimeContext(BaseModel):
    """Everything a runtime needs to run one task deterministically and auditably."""
    model_config = {"extra": "forbid"}
    exchange_dir: str
    repo_root: str
    provider: str = "mock"
    model_id: str = "mock"
    timeout_s: float = 120.0
    max_retries: int = 1
    # A read-only allow-list: absolute path prefixes the agent's tools may read.
    read_allow_prefixes: list[str] = Field(default_factory=list)


class RuntimeInvocationRecord(BaseModel):
    """The complete, hash-bound provenance of one agent invocation attempt.

    Preserves BOTH the raw model response and the parsed result, every hash needed to
    reproduce the attempt, tool calls, token usage, and any validation failure — so a
    failed attempt is auditable, not discarded.
    """
    model_config = {"extra": "forbid"}
    attempt_id: str
    # Retry lineage. parent_attempt_id links a retry to the attempt it followed; a fresh
    # invocation has None. retry_category names WHICH retry layer produced this attempt
    # (see runtimes/pydantic_ai/README.md — most layers are not implemented in the PoC).
    parent_attempt_id: Optional[str] = None
    retry_category: Literal[
        "none", "agent_invocation", "model_output", "provider", "controller_task",
        "scientific_recovery",
    ] = "none"
    task_id: str
    agent: str
    provider: str
    model_id: str
    runtime_version: str
    prompt_sha256: str
    input_artifacts_sha256: dict[str, str] = Field(default_factory=dict)
    tool_manifest_sha256: str
    raw_response: str
    parsed_result: Optional[dict[str, Any]] = None
    tool_invocations: list[ToolInvocationRecord] = Field(default_factory=list)
    validation_errors: list[ValidationErrorRecord] = Field(default_factory=list)
    # Token accounting. usage_source makes clear whether these numbers are real provider
    # usage, a test-model's synthetic count, an estimate, or unavailable — so mock counts
    # are never mistaken for billable usage.
    usage_source: Literal["mock", "test-model", "provider", "estimated", "unavailable"] = "unavailable"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # UTC ISO-8601 time the provenance record was built. Per-call start/end timing (to
    # measure provider latency) arrives with the real provider path (P4); this PoC records
    # only the record-build time.
    recorded_at: str = ""
    accepted: bool = False
