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
import os

from .interface import RUNTIME_VERSION, AgentInvocation, execute_with_retry
from .models import AgentResultModel, JudgeVoteModel, RuntimeContext
from .redaction import secrets_from_env
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

    def _build_agent(self, spec, toolset, output_model, context=None):
        from pydantic_ai import Agent  # lazy: absence must not break import/tests

        system_prompt = (getattr(spec, "prompt", "") + "\n\n" + toolset.context_note())
        kwargs = {"output_type": output_model, "system_prompt": system_prompt}
        timeout = getattr(context, "timeout_s", None)
        if timeout:
            # Best-effort timeout; harmless for TestModel, applied for real providers.
            try:
                agent = Agent(self._model, model_settings={"timeout": float(timeout)}, **kwargs)
            except TypeError:
                agent = Agent(self._model, **kwargs)
        else:
            agent = Agent(self._model, **kwargs)

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

        @agent.tool_plain
        def read_csv_summary(path: str):
            try:
                return toolset.read_csv_summary(path)
            except ToolAccessError as error:
                return {"ok": False, "error": f"ACCESS DENIED: {error}"}
            except UnicodeError as error:
                return {"ok": False, "error": f"INVALID ENCODING: {error}"}
            except OSError as error:
                return {"ok": False, "error": f"READ ERROR: {error}"}
            except Exception as error:  # csv.Error etc. — refuse, never crash
                return {"ok": False, "error": f"CSV ERROR: {error}"}

        @agent.tool_plain
        def read_artifact_manifest(path: str):
            try:
                return toolset.read_artifact_manifest(path)
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
        # Each role's Agent emits its role-specific typed output (Judge -> JudgeVote, producers ->
        # role-scoped ActionProposal, Orchestrator/Literature -> typed plan/evidence); unknown
        # roles fall back to the generic AgentResult. See role_outputs.select_output_model.
        from .role_outputs import select_output_model
        output_model = select_output_model(spec)
        toolset = ReadOnlyToolset(context.read_allow_prefixes)
        agent = self._build_agent(spec, toolset, output_model, context)
        # Values that must never reach provenance/logs, masked wherever they appear.
        extra_secrets = secrets_from_env(dict(os.environ))

        # Deterministic per-invocation bound: request_limit caps model requests (each tool
        # round-trip is one), so an unproductive tool loop fails closed (UsageLimitExceeded ->
        # classified terminal, provenance keeps every attempted tool call) rather than running
        # until the context window is exhausted.
        from pydantic_ai.usage import UsageLimits  # lazy: only when a real agent runs
        limits = UsageLimits(request_limit=getattr(context, "request_limit", 6))

        def _call():
            run = agent.run_sync(json.dumps({"task": task}), usage_limits=limits)
            usage = run.usage()
            return (run.output.model_dump_json(),
                    getattr(usage, "input_tokens", 0) or 0,
                    getattr(usage, "output_tokens", 0) or 0)

        # A provider exception is classified + preserved as an attempt record, never lost;
        # retryable failures are retried within the bounded policy in context.
        return execute_with_retry(
            call=_call, task=task, spec=spec, context=context, toolset=toolset,
            usage_source=self._usage_source, runtime_version=RUNTIME_VERSION,
            extra_secrets=extra_secrets)
