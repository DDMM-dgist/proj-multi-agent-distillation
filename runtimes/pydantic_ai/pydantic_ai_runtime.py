"""PydanticAIRuntime — the real pydantic_ai-backed runtime.

Lazy-imports ``pydantic_ai`` so importing this module (and running the mock PoC/tests)
needs no provider or API key. It shares the exact provenance/parse path
(``interface.build_invocation``) with the mock runtime; only the raw response and token
usage come from a real ``pydantic_ai.Agent`` here.

The ``model`` argument accepts anything pydantic_ai's Agent accepts:
- a provider model id string (e.g. ``"anthropic:claude-..."``) for a real provider call
  (needs the provider's API key in the ENVIRONMENT — never committed), OR
- a ``pydantic_ai.models.test.TestModel`` / ``FunctionModel`` for network-free tests.

Enable with ``pip install -e .[pydantic-ai]``.
"""
from __future__ import annotations

import json

from .interface import RUNTIME_VERSION, AgentInvocation, build_invocation
from .models import AgentResultModel, JudgeVoteModel, RuntimeContext
from .tool_registry import ReadOnlyToolset, ToolAccessError

# Map the repo's result_contract to the typed output model pydantic_ai enforces.
_OUTPUT_MODELS = {"JudgeVote": JudgeVoteModel, "AgentResult": AgentResultModel}


class PydanticAIRuntime:
    """Runs one task on a real pydantic_ai Agent. Same AgentInvocation contract as the
    mock runtime. ``usage_source`` is 'test-model' for TestModel/FunctionModel, else
    'provider'."""

    def __init__(self, *, model=None, usage_source="provider"):
        self._model = model
        self._usage_source = usage_source

    def _build_agent(self, spec, toolset, output_model):
        from pydantic_ai import Agent  # lazy: absence must not break import/tests

        system_prompt = (getattr(spec, "prompt", "") + "\n\n" + toolset.context_note())
        agent = Agent(self._model, output_type=output_model, system_prompt=system_prompt)

        # Both read-only tools (matching tool_registry.EXPOSED_READ_TOOLS) are registered.
        # Any failure — blocked path, read error, bad encoding, or invalid JSON — is
        # returned to the model as an explicit, distinguishable refusal and recorded in the
        # audit trail, rather than crashing the run. Enforcement is at the tool, not the agent.
        @agent.tool_plain
        def read_text(path: str) -> str:
            try:
                return toolset.read_text(path)
            except ToolAccessError as error:
                return f"ACCESS DENIED: {error}"
            except UnicodeError as error:
                return f"INVALID ENCODING: {error}"
            except OSError as error:
                return f"READ ERROR: {error}"

        @agent.tool_plain
        def read_json(path: str):
            try:
                return toolset.read_json(path)
            except ToolAccessError as error:
                return {"ok": False, "error": f"ACCESS DENIED: {error}"}
            except UnicodeError as error:
                return {"ok": False, "error": f"INVALID ENCODING: {error}"}
            except json.JSONDecodeError as error:
                return {"ok": False, "error": f"INVALID JSON: {error}"}
            except OSError as error:
                return {"ok": False, "error": f"READ ERROR: {error}"}

        return agent

    def run(self, task: dict, spec, context: RuntimeContext) -> AgentInvocation:
        result_contract = getattr(spec, "result_contract", "AgentResult")
        output_model = _OUTPUT_MODELS.get(result_contract, AgentResultModel)
        toolset = ReadOnlyToolset(context.read_allow_prefixes)

        agent = self._build_agent(spec, toolset, output_model)
        run = agent.run_sync(json.dumps({"task": task}))

        raw_response = run.output.model_dump_json()
        usage = run.usage()
        return build_invocation(
            task=task, spec=spec, context=context, toolset=toolset,
            raw_response=raw_response, usage_source=self._usage_source,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            runtime_version=RUNTIME_VERSION,
        )
