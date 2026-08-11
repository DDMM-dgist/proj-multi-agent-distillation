"""Pydantic parsing/provenance models for the PydanticAI runtime.

Design (Plan B): the JSON Schemas under ``orchestration/schema/`` stay canonical.
These models mirror those contracts closely enough to give a PydanticAI agent a typed
``output_type``, but they are NOT the source of truth: after parsing, a result is
``.model_dump()``-ed and revalidated by ``orchestration.exchange.validate_agent_response``.
Pydantic type-validation and physics/domain validation stay distinct.

Requires ``pydantic`` (optional dependency). Not imported by any core package.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field

# A non-empty string, mirroring the JSON Schema ``{"type": "string", "minLength": 1}``.
NonEmptyStr = Annotated[str, Field(min_length=1)]


# --- Contract mirrors (typed output for the LLM) --------------------------------

class EvidenceReference(BaseModel):
    """A pointer to an artifact the agent relied on. Mirrors the exchange 'reference'."""
    model_config = {"extra": "forbid"}
    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    integrity: Optional[dict[str, Any]] = None


# --- Typed input mirror (Phase 2 / D1) ------------------------------------------

class AgentTaskModel(BaseModel):
    """Typed mirror of ``orchestration/schema/agent_task.schema.json``.

    NOT authoritative. Parsing a task as this model does NOT make it valid: the canonical
    ``orchestration.exchange.validate_task`` MUST still run (it also enforces spec-specific
    rules the JSON Schema cannot express, e.g. a Judge task requiring ``context.review_lens``
    and ``context.review_focus``). This model exists only so a runtime can carry a task with
    typed field access and reject obviously-malformed packets early. Required/optional and
    ``extra='forbid'`` are kept in lockstep with the canonical schema by
    ``tests/test_pydantic_ai_schema_drift.py``.
    """
    model_config = {"extra": "forbid"}
    schema_version: Literal[1]
    task_id: NonEmptyStr
    agent: NonEmptyStr
    run_id: Optional[str] = None
    created_at: NonEmptyStr
    instruction: NonEmptyStr
    inputs: list[EvidenceReference]
    criteria: list[NonEmptyStr]
    constraints: list[NonEmptyStr]
    context: dict[str, Any]


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
    # Bounded-retry policy (Phase 2/D5). provider_retries = extra attempts after the first on a
    # RETRYABLE failure; max_total_calls caps total provider calls (cost guard); backoff is
    # exponential with jitter between attempts.
    provider_retries: int = 0
    structured_output_retries: int = 0
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    max_total_calls: int = 1
    # Deterministic per-invocation bound on the number of MODEL REQUESTS (pydantic_ai
    # UsageLimits.request_limit). Each tool round-trip consumes one request, so this fails a
    # runaway tool loop closed BEFORE the context window is exhausted. Legitimate current tasks
    # need <=2 requests (e.g. Judge: read_json + final vote); 6 leaves headroom for a few tool
    # round-trips while stopping a loop like Stage B attempt-1's 20x read_artifact_manifest.
    request_limit: int = 6
    correlation_id: str = ""


class ProviderConfiguration(BaseModel):
    """Typed provider/runtime configuration (Phase 2/D3), assembled by the CLI from env + flags.

    ``model_id`` and any credential come from the ENVIRONMENT, never from a committed config.
    """
    model_config = {"extra": "forbid"}
    provider: NonEmptyStr
    model_id: NonEmptyStr
    provider_sdk_version: str = ""
    timeout_s: float = 120.0
    provider_retries: int = 2
    structured_output_retries: int = 1
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    max_total_calls: int = 3
    usage_source: Literal["mock", "test-model", "provider", "estimated", "unavailable"] = "provider"
    correlation_id: str = ""


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
    # UTC ISO-8601 time the provenance record was built.
    recorded_at: str = ""
    accepted: bool = False
    # --- Phase 2/D4-D5 additive failure + timing + mode fields --------------------
    # A provider exception is recorded here (not lost): exception_message is ALWAYS redacted.
    failure_category: str = ""          # "" on success; else a failures.FailureCategory value
    exception_class: str = ""
    exception_message: str = ""         # redacted before storage
    retryable: bool = False
    started_at: str = ""
    finished_at: str = ""
    latency_s: float = 0.0
    correlation_id: str = ""
    # Execution mode + whether this attempt mutated controller-visible state.
    mode: Literal["", "primary", "shadow", "dry_run", "validate_only"] = ""
    controller_mutated: bool = False
    estimated_cost: Optional[float] = None
    # --- Deterministic-verdict ownership (Stage D-1 refactor) ----------------------
    # For an authoritative (fully deterministic) judge gate the ACCEPTED verdict is owned by the
    # deterministic policy and bound by trusted code; the LLM's proposed verdict is preserved for
    # audit but is not authoritative. accepted_verdict is the verdict actually accepted for the gate
    # (deterministic for authoritative gates; the LLM's verdict for advisory gates). Absent (None) on
    # older provenance -> consumers fall back to parsed_result.verdict.
    accepted_verdict: Optional[str] = None
    llm_proposed_verdict: Optional[str] = None
    verdict_overridden: bool = False
    criterion_contradictions: list[str] = Field(default_factory=list)
