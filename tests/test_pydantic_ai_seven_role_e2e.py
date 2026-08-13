"""Phase 6/4: seven-role network-free end-to-end dry-run.

Runs all seven roles via real pydantic_ai.Agent + TestModel through a single controller sandbox:
Orchestrator/Literature/Analyst/producers emit their role-specific typed output; producer
ActionProposals go through the dispatch->controller enforcement (dry-run); three mutually-blind
Judge invocations each emit a typed JudgeVote. No Claude Code Agent tool, no manual JSON, no
provider, no shell, no scientific compute, no scheduler submission. Skips without pydantic_ai.
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


def _run(d: Path):
    state = {
        "schema_version": 7, "run_id": "r", "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
        "project_dir": "p", "inputs": [], "code_revision": "x", "events": [],
        "stages": [{"name": "data_curation", "status": "pending", "gate": "pending", "artifacts": []}],
        "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                        "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
        "recoveries": [], "pending_recovery": None,
        "runtime_attempts": [], "idempotency": {}, "action_approvals": {}, "scheduler_jobs": {},
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state))
    return d


def _proposal_args(role, action_type, key):
    return {"requested_by_role": role, "action_type": action_type, "schema_version": 1,
            "run_id": "r", "stage": "data_curation", "requested_at": "2026-08-07T00:00:00Z",
            "rationale": "sandbox dry-run", "idempotency_key": key, "parameters": {},
            "active_config_refs": [], "advisory_claimed_config_hashes": {}, "input_artifacts": [],
            "input_artifact_hashes": {}, "expected_outputs": [], "dry_run": True}


@unittest.skipUnless(_HAS, "pydantic / pydantic_ai not installed")
class SevenRoleE2ETests(unittest.TestCase):
    def setUp(self):
        from orchestration.specs import load_agent_specs
        self.specs = load_agent_specs(SPECS)

    def _runtime(self, output_args):
        from pydantic_ai.models.test import TestModel
        from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
        return PydanticAIRuntime(model=TestModel(custom_output_args=output_args),
                                 usage_source="test-model")

    def _ctx(self, exch):
        from runtimes.pydantic_ai.models import RuntimeContext
        return RuntimeContext(exchange_dir=str(exch), repo_root=str(ROOT),
                              provider="test", model_id="test-model",
                              read_allow_prefixes=[str(exch)], correlation_id="e2e")

    def _invoke(self, role, output_args, ctx):
        from orchestration.exchange import make_task
        spec = self.specs[role]
        context = ({"review_lens": "evidence_provenance", "review_focus": "prov"}
                   if role == "judge" else {})
        task = make_task(role, f"{role} sandbox task", criteria=["c1"], context=context)
        inv = self._runtime(output_args).run(task, spec, ctx)
        return task, inv

    def _assert_provenance(self, inv, role):
        rec = inv.provenance
        self.assertTrue(rec.attempt_id)
        self.assertEqual(rec.agent, role)
        self.assertTrue(rec.prompt_sha256)
        self.assertTrue(rec.tool_manifest_sha256)
        self.assertEqual(rec.usage_source, "test-model")
        self.assertIsNotNone(inv.candidate)

    def test_seven_role_end_to_end_dry_run(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        with tempfile.TemporaryDirectory() as tmp:
            controller = RunController(_run(Path(tmp) / "run"))
            exch = Path(tmp) / "exchange"; exch.mkdir()
            registry = build_executor_registry()

            # 1) Orchestrator typed RunPlan
            _, orch = self._invoke("orchestrator", {
                "run_id": "r", "current_stage": "data_curation", "rationale": "proceed",
                "summary": "plan", "proposed_tasks": [], "approval_requests": []}, self._ctx(exch))
            self._assert_provenance(orch, "orchestrator")
            controller.record_runtime_attempt(task_id="orch", attempt_id=orch.provenance.attempt_id,
                                              provenance_path="p", role="orchestrator")

            # 2) Literature evidence
            _, lit = self._invoke("literature", {
                "status": "completed", "sources": [], "evidence_gaps": [],
                "summary": "no external source needed"}, self._ctx(exch))
            self._assert_provenance(lit, "literature")

            # 3) producers -> role-scoped ActionProposal -> dispatch enforcement (dry-run)
            producer_actions = {"data-curator": "inspect_dataset",
                                "ml-trainer": "compute_committee_disagreement",
                                "simulation": "compute_nve_drift",
                                "analyst": "compare_force_errors"}
            for role, action in producer_actions.items():
                _, inv = self._invoke(role, _proposal_args(role, action, f"{role}-1"), self._ctx(exch))
                self._assert_provenance(inv, role)
                outcome = dispatch_via_controller(inv.candidate, controller=controller,
                                                  registry=registry, mode="dry_run")
                self.assertIn(outcome.status, ("DRY_RUN",), f"{role}:{outcome.status}")

            # 4) three mutually-blind Judge invocations, each a typed JudgeVote, distinct lenses
            lenses = ["evidence_provenance", "scientific_validity", "reproducibility_deployment"]
            votes = []
            for lens in lenses:
                spec = self.specs["judge"]
                from orchestration.exchange import make_task
                task = make_task("judge", "review", criteria=["c1"],
                                 context={"review_lens": lens, "review_focus": "f"})
                inv = self._runtime({"review_lens": lens, "verdict": "PASS",
                                     "criteria_checked": [{"criterion": "c1", "value_read": "ok",
                                                           "ok": True}],
                                     "rationale": "checked", "required_fix": ""}).run(
                    task, spec, self._ctx(exch))
                votes.append(inv.candidate)
                self._assert_provenance(inv, "judge")
            self.assertEqual([v["review_lens"] for v in votes], lenses)  # one blind vote per lens
            self.assertEqual({v["verdict"] for v in votes}, {"PASS"})

            # controller remains the sole durable-state owner; a runtime attempt was recorded
            self.assertGreaterEqual(len(controller.state["runtime_attempts"]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
