"""Bounded tool-call guard: an unproductive tool loop must fail CLOSED at the request_limit
(pydantic_ai UsageLimits) instead of running until the context window is exhausted — the
production-hardening finding from Stage B attempt-1 (Orchestrator called read_artifact_manifest
20x -> 4096 context overflow -> HTTP 400). Network-free (a real Agent driven by FunctionModel).
Skips without the pydantic + pydantic_ai extras.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = str(ROOT / "agent_specs")

try:
    import pydantic  # noqa: F401
    import pydantic_ai  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS, "pydantic / pydantic_ai not installed")
class ToolBudgetTests(unittest.TestCase):
    def _loop_runtime(self):
        # A model that ALWAYS calls a tool -> an infinite tool loop unless bounded.
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        calls = {"n": 0}

        def fn(messages, info):
            calls["n"] += 1
            return ModelResponse(parts=[ToolCallPart(tool_name="read_json", args={"path": "x"})])

        return PydanticAIRuntime(model=FunctionModel(fn), usage_source="test-model"), calls

    def test_runaway_tool_loop_fails_closed_at_request_limit(self):
        from orchestration.specs import load_agent_specs
        from runtimes.pydantic_ai.models import RuntimeContext
        spec = load_agent_specs(SPECS)["judge"]
        rt, calls = self._loop_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            ex = Path(tmp) / "ex"; ex.mkdir()
            ctx = RuntimeContext(exchange_dir=str(ex), repo_root=str(ROOT),
                                 provider="anthropic", model_id="anthropic:test",
                                 read_allow_prefixes=[str(ex)], request_limit=3)
            task = {"schema_version": 1, "task_id": "loop1", "agent": "judge",
                    "created_at": "t", "instruction": "x", "inputs": [], "criteria": ["c1"],
                    "constraints": [], "context": {"review_lens": "evidence_provenance",
                                                   "review_focus": "f"}}
            inv = rt.run(task, spec, ctx)
        prov = inv.provenance
        # fails closed with the explicit bounded-budget category; terminal (never retried)
        self.assertEqual(prov.failure_category, "usage_limit_exceeded")
        self.assertFalse(prov.retryable)
        self.assertEqual(inv.candidate, {})                 # no accepted output
        self.assertFalse(prov.controller_mutated)           # no mutation
        # the loop was bounded (did NOT run away); every attempted tool call is preserved
        self.assertLessEqual(calls["n"], 3, "request_limit did not bound the loop")
        self.assertTrue(prov.tool_invocations, "attempted tool calls must be recorded")
        self.assertTrue(all(t.tool == "read_json" for t in prov.tool_invocations))

    def test_router_does_not_accept_or_mutate_on_budget_failure(self):
        from orchestration.specs import load_agent_specs
        from runtimes.pydantic_ai.models import RuntimeContext
        from runtimes.pydantic_ai.production_router import run_role
        from runtimes.pydantic_ai.executors import build_executor_registry
        spec = load_agent_specs(SPECS)["judge"]
        rt, _ = self._loop_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            ex = Path(tmp) / "ex"; ex.mkdir()
            ctx = RuntimeContext(exchange_dir=str(ex), repo_root=str(ROOT),
                                 read_allow_prefixes=[str(ex)], request_limit=3)
            task = {"task_id": "loop2", "agent": "judge", "inputs": [],
                    "context": {"review_lens": "evidence_provenance", "review_focus": "f"},
                    "criteria": ["c1"]}
            res = run_role(rt, task, spec, ctx, controller=None,
                           registry=build_executor_registry(), mode="shadow")
        self.assertFalse(res.accepted)
        self.assertFalse(res.controller_mutated)
        self.assertTrue(res.error)
        self.assertFalse((ex / "results").exists())         # nothing accepted into the exchange

    def test_default_request_limit_is_small(self):
        from runtimes.pydantic_ai.models import RuntimeContext
        ctx = RuntimeContext(exchange_dir="e", repo_root="r")
        self.assertEqual(ctx.request_limit, 6)              # bounded by default, not pydantic_ai's 50


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
