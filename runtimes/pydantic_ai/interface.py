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
import time as _time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from .failures import classify_failure
from .models import RuntimeContext, RuntimeInvocationRecord, ValidationErrorRecord
from .redaction import redact
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
    # Provenance of every earlier attempt in a retry chain (Phase 2/D5). The driver persists
    # these too, so a failed-then-succeeded (or exhausted) chain is fully auditable.
    prior_attempts: list = field(default_factory=list)


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


def _input_hashes(task) -> dict:
    return {
        ref.get("role", ""): (ref.get("integrity", {}) or {}).get("sha256", "")
        for ref in task.get("inputs", []) if isinstance(ref, dict)
    }


def build_invocation(*, task, spec, context: RuntimeContext, toolset: ReadOnlyToolset,
                     raw_response: str, usage_source: str, prompt_tokens: int,
                     completion_tokens: int, runtime_version: str,
                     parent_attempt_id=None, retry_category="none",
                     started_at="", finished_at="", latency_s=0.0) -> AgentInvocation:
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
        input_artifacts_sha256=_input_hashes(task),
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
        started_at=started_at,
        finished_at=finished_at,
        latency_s=latency_s,
        correlation_id=getattr(context, "correlation_id", "") or "",
    )
    return AgentInvocation(candidate=candidate if candidate is not None else {},
                           provenance=record)


def build_failure_record(*, task, spec, context: RuntimeContext, toolset: ReadOnlyToolset,
                         exc: BaseException, usage_source: str, runtime_version: str,
                         attempt_id: str, parent_attempt_id, started_at: str, finished_at: str,
                         latency_s: float, extra_secrets: Iterable[str] = ()) -> RuntimeInvocationRecord:
    """Build a provenance record for a provider attempt that raised BEFORE producing output.

    The attempt is preserved (never lost). The exception message is ALWAYS redacted, the failure
    is classified, and no raw/parsed output is stored (there was none).
    """
    category, retryable = classify_failure(exc)
    prompt = getattr(spec, "prompt", "")
    return RuntimeInvocationRecord(
        attempt_id=attempt_id,
        parent_attempt_id=parent_attempt_id,
        retry_category="provider",
        task_id=task.get("task_id", ""),
        agent=getattr(spec, "name", task.get("agent", "")),
        provider=context.provider,
        model_id=context.model_id,
        runtime_version=runtime_version,
        prompt_sha256=sha256_text(prompt),
        input_artifacts_sha256=_input_hashes(task),
        tool_manifest_sha256=toolset.tool_manifest_sha256(),
        raw_response="",
        parsed_result=None,
        tool_invocations=toolset.invocations,
        validation_errors=[],
        usage_source=usage_source,
        prompt_tokens=0,
        completion_tokens=0,
        recorded_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        accepted=False,
        failure_category=category,
        exception_class=type(exc).__name__,
        exception_message=redact(str(exc), extra_secrets),
        retryable=retryable,
        started_at=started_at,
        finished_at=finished_at,
        latency_s=latency_s,
        correlation_id=getattr(context, "correlation_id", "") or "",
    )


def _backoff_delay(attempt_index: int, context: RuntimeContext, task) -> float:
    """Exponential backoff with deterministic jitter (no RNG, so tests are reproducible)."""
    base = max(0.0, getattr(context, "backoff_base_s", 0.5))
    cap = max(base, getattr(context, "backoff_max_s", 8.0))
    raw = min(cap, base * (2 ** attempt_index))
    seed = f"{task.get('task_id','')}:{attempt_index}".encode()
    jitter_frac = int(hashlib.sha256(seed).hexdigest()[:4], 16) / 0xFFFF  # in [0,1]
    return min(cap, raw + jitter_frac * base)


def execute_with_retry(*, call: Callable[[], tuple], task, spec, context: RuntimeContext,
                       toolset: ReadOnlyToolset, usage_source: str, runtime_version: str,
                       clock: Callable[[], _dt.datetime] = None,
                       sleep: Callable[[float], None] = None,
                       extra_secrets: Iterable[str] = ()) -> AgentInvocation:
    """Run ``call`` with bounded retry, preserving provenance for EVERY attempt.

    ``call`` returns ``(raw_response, prompt_tokens, completion_tokens)`` or raises. Retries only
    on a RETRYABLE failure category, up to ``min(provider_retries+1, max_total_calls)`` attempts
    (a hard cost cap). Non-retryable failures are terminal on the first attempt. Backoff is
    exponential with deterministic jitter; ``clock``/``sleep`` are injectable for tests.
    """
    clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))
    sleep = sleep or _time.sleep
    provider_retries = max(0, int(getattr(context, "provider_retries", 0)))
    max_calls = max(1, int(getattr(context, "max_total_calls", 1)))
    limit = min(provider_retries + 1, max_calls)
    prior: list = []
    parent = None
    for i in range(limit):
        start = clock()
        try:
            raw, ptok, ctok = call()
        except BaseException as exc:  # noqa: BLE001 - classify + preserve, never leak/lose
            finish = clock()
            rec = build_failure_record(
                task=task, spec=spec, context=context, toolset=toolset, exc=exc,
                usage_source=usage_source, runtime_version=runtime_version,
                attempt_id=new_attempt_id(), parent_attempt_id=parent,
                started_at=start.isoformat(), finished_at=finish.isoformat(),
                latency_s=(finish - start).total_seconds(), extra_secrets=extra_secrets)
            parent = rec.attempt_id
            if rec.retryable and i < limit - 1:
                prior.append(rec)
                sleep(_backoff_delay(i, context, task))
                continue
            return AgentInvocation(candidate={}, provenance=rec, prior_attempts=list(prior))
        finish = clock()
        inv = build_invocation(
            task=task, spec=spec, context=context, toolset=toolset, raw_response=raw,
            usage_source=usage_source, prompt_tokens=ptok, completion_tokens=ctok,
            runtime_version=runtime_version, parent_attempt_id=parent,
            retry_category=("provider" if parent else "none"),
            started_at=start.isoformat(), finished_at=finish.isoformat(),
            latency_s=(finish - start).total_seconds())
        inv.prior_attempts = list(prior)
        return inv
    # Unreachable: the loop always returns. Guard for safety.
    raise RuntimeError("execute_with_retry exhausted without returning")
