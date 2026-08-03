"""Common runtime interface + the shared invocation path used by EVERY runtime.

A runtime produces a CANDIDATE result; it never mutates controller state. Acceptance is
the driver's job, gated on the existing validators. Both MockAgentRuntime and
PydanticAIRuntime differ ONLY in how ``raw_response`` and token usage are produced — the
tool set, parsing, hashing, provenance construction, and validation-error capture are all
built here by ``build_invocation`` so the mock path takes no shortcut around the real one.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .models import RuntimeContext, RuntimeInvocationRecord, ValidationErrorRecord
from .tool_registry import ReadOnlyToolset

# Shared runtime version string. Lives here (not in a concrete runtime) so both the mock
# and the real runtime import it from a common place.
RUNTIME_VERSION = "pydantic-ai-runtime/0.1.0"


@dataclass
class AgentInvocation:
    """What a runtime returns: the candidate result dict + full provenance record.

    The dict is a plain contract payload (AgentResult/JudgeVote shape) that the driver
    feeds to the EXISTING ``validate_agent_response`` — parsing success alone is never
    acceptance.
    """
    candidate: dict[str, Any]
    provenance: RuntimeInvocationRecord


class AgentRuntime(Protocol):
    """Provider-neutral agent execution. Implementations: MockAgentRuntime (no deps),
    PydanticAIRuntime (real provider). Both return an AgentInvocation and touch no
    controller state.
    """

    def run(self, task: dict, spec: Any, context: RuntimeContext) -> AgentInvocation:
        ...


def new_attempt_id() -> str:
    return f"attempt-{uuid.uuid4()}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_invocation(*, task, spec, context: RuntimeContext, toolset: ReadOnlyToolset,
                     raw_response: str, usage_source: str, prompt_tokens: int,
                     completion_tokens: int, runtime_version: str,
                     parent_attempt_id=None, retry_category="none") -> AgentInvocation:
    """The single provenance/parse path shared by the mock and real runtimes.

    Parses the raw response as JSON (capturing a pydantic_parse ValidationErrorRecord on
    failure instead of raising), hashes the prompt/inputs/tool surface, and assembles the
    RuntimeInvocationRecord. Never validates the *contract* — that is the driver's job via
    the existing ``validate_agent_response``.
    """
    prompt = getattr(spec, "prompt", "")
    candidate = None
    validation_errors = []
    try:
        candidate = json.loads(raw_response)
    except json.JSONDecodeError as error:
        validation_errors.append(ValidationErrorRecord(
            stage="pydantic_parse", message=f"raw response is not JSON: {error}"))

    record = RuntimeInvocationRecord(
        attempt_id=new_attempt_id(),
        parent_attempt_id=parent_attempt_id,
        retry_category=retry_category,
        task_id=task.get("task_id", ""),
        agent=getattr(spec, "name", task.get("agent", "")),
        provider=context.provider,
        model_id=context.model_id,
        runtime_version=runtime_version,
        prompt_sha256=sha256_text(prompt),
        input_artifacts_sha256={
            ref.get("role", ""): (ref.get("integrity", {}) or {}).get("sha256", "")
            for ref in task.get("inputs", []) if isinstance(ref, dict)
        },
        tool_manifest_sha256=toolset.tool_manifest_sha256(),
        raw_response=raw_response,
        parsed_result=candidate,
        tool_invocations=toolset.invocations,
        validation_errors=validation_errors,
        usage_source=usage_source,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        recorded_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        accepted=False,
    )
    return AgentInvocation(candidate=candidate if candidate is not None else {},
                           provenance=record)
