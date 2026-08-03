"""MockAgentRuntime — deterministic, dependency-free agent execution.

Proves the full runtime pipeline WITHOUT any LLM provider or API key, so the PoC and its
tests run anywhere. It differs from the real PydanticAIRuntime ONLY in how the raw
response is produced (a caller-supplied ``responder`` test double instead of a model
call): the tool set, parsing, hashing, provenance, and validation-error capture are the
SAME shared ``build_invocation`` path, so the mock takes no shortcut around the real one.
Token counts are tagged ``usage_source='mock'`` so they are never mistaken for real usage.
"""
from __future__ import annotations

from typing import Any, Callable

from .interface import AgentInvocation, build_invocation
from .models import RuntimeContext
from .tool_registry import ReadOnlyToolset

RUNTIME_VERSION = "pydantic-ai-runtime/0.1.0"


class MockAgentRuntime:
    """A test-double runtime. ``responder(task, spec, toolset) -> (raw_text, (in, out))``
    stands in for the model call; the raw text is preserved and parsed exactly as a real
    response would be, through the shared build_invocation path.
    """

    def __init__(self, responder: Callable[[dict, Any, ReadOnlyToolset], tuple]):
        self._responder = responder

    def run(self, task: dict, spec: Any, context: RuntimeContext) -> AgentInvocation:
        toolset = ReadOnlyToolset(context.read_allow_prefixes)
        raw_text, tokens = self._responder(task, spec, toolset)
        prompt_tokens, completion_tokens = tokens
        return build_invocation(
            task=task, spec=spec, context=context, toolset=toolset,
            raw_response=raw_text, usage_source="mock",
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            runtime_version=f"{RUNTIME_VERSION}+mock",
        )
