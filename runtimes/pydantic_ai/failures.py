"""Provider/runtime failure classification (Phase 2/D4-D5).

Classifies an exception raised during a provider invocation into a stable failure category and
whether it is retryable. Classification is heuristic (by exception type name + message) so it
works across providers without importing any provider SDK. Non-retryable categories must never
be retried (they would only re-bill or re-fail).
"""
from __future__ import annotations

from typing import Literal, Tuple

FailureCategory = Literal[
    "authentication_failure",
    "timeout",
    "rate_limit",
    "provider_network_failure",
    "provider_internal_failure",
    "malformed_output",
    "structured_output_failure",
    "usage_limit_exceeded",
    "tool_failure",
    "tool_policy_refusal",
    "pydantic_parse_failure",
    "contract_failure",
    "artifact_validation_failure",
    "unsupported_action",
    "missing_artifact",
    "unknown",
]

# Categories safe to retry (transient / server-side). Everything else is terminal.
_RETRYABLE = {"timeout", "rate_limit", "provider_network_failure", "provider_internal_failure"}


def classify_failure(exc: BaseException) -> Tuple[str, bool]:
    """Return ``(failure_category, retryable)`` for an exception from a provider call."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    status = _status_code(exc)

    # Bounded tool-call / request budget tripped (pydantic_ai UsageLimitExceeded): a runaway
    # tool loop was stopped BEFORE context exhaustion. Terminal + operational — never retry.
    if "usagelimit" in name or "request_limit" in msg or "exceed the request" in msg:
        return "usage_limit_exceeded", False
    if "authentication" in name or "permission" in name or status in (401, 403) \
            or "invalid api key" in msg or "unauthorized" in msg or "authentication" in msg:
        return "authentication_failure", False
    if "timeout" in name or "timeout" in msg or status == 408:
        return "timeout", True
    if status == 429 or "rate limit" in msg or "ratelimit" in name or "too many requests" in msg:
        return "rate_limit", True
    if status in (500, 502, 503, 504) or "internalserver" in name or "overloaded" in msg \
            or "service unavailable" in msg:
        return "provider_internal_failure", True
    if "connection" in name or "connect" in msg or "network" in name or "socket" in msg:
        return "provider_network_failure", True
    if "structured" in msg or "output type" in msg or "tool call" in msg and "output" in msg:
        return "structured_output_failure", False
    if "json" in name or "jsondecode" in name:
        return "malformed_output", False
    return "unknown", False


def is_retryable(category: str) -> bool:
    return category in _RETRYABLE


def _status_code(exc: BaseException):
    """Best-effort HTTP status extraction across SDK exception shapes."""
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    resp = getattr(exc, "response", None)
    if resp is not None:
        value = getattr(resp, "status_code", None)
        if isinstance(value, int):
            return value
    return None
