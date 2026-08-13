"""Phase 6/0: single production router auto-selects the acceptance strategy per role, reachable
from the CLI, with no manual per-role function selection. Network-free (mock runtime); skips
without the ``pydantic`` extra.
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
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _v7_run(d: Path):
    state = {
        "schema_version": 7, "run_id": "r", "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
        "project_dir": "p", "inputs": [], "code_revision": "x", "events": [],
        "stages": [{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}],
        "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                        "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
        "recoveries": [], "pending_recovery": None,
        "runtime_attempts": [], "idempotency": {}, "action_approvals": {}, "scheduler_jobs": {},
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state))
    return d


def _proposal(role, action, key="k"):
    return {"requested_by_role": role, "action_type": action, "schema_version": 1, "run_id": "r",
            "stage": "s", "requested_at": "t", "rationale": "why", "idempotency_key": key,
            "parameters": {}, "active_config_refs": [], "advisory_claimed_config_hashes": {},
            "input_artifacts": [], "input_artifact_hashes": {}, "expected_outputs": [],
            "dry_run": True}


TYPED_OUTPUT = {
    "orchestrator": {"run_id": "r", "current_stage": "s", "rationale": "go", "summary": "plan",
                     "proposed_tasks": [], "approval_requests": []},
    "literature": {"status": "completed", "sources": [], "evidence_gaps": [], "summary": "none"},
    "judge": {"review_lens": "evidence_provenance", "verdict": "PASS",
              "criteria_checked": [{"criterion": "c1", "value_read": "x", "ok": True}],
              "rationale": "ok", "required_fix": ""},
    "data-curator": _proposal("data-curator", "inspect_dataset"),
    "ml-trainer": _proposal("ml-trainer", "compute_committee_disagreement"),
    "simulation": _proposal("simulation", "compute_nve_drift"),
    "analyst": _proposal("analyst", "compare_force_errors"),
}
EXPECTED_STRATEGY = {
    "judge": "judge_gate", "orchestrator": "typed_result", "literature": "typed_result",
    "data-curator": "producer_dispatch", "ml-trainer": "producer_dispatch",
    "simulation": "producer_dispatch", "analyst": "producer_dispatch",
}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ProductionRouterTests(unittest.TestCase):
    def _cli(self, role, response, tmp, mode="shadow", run_dir=None):
        from runtimes.pydantic_ai import cli
        d = Path(tmp)
        task = {"schema_version": 1, "task_id": "t1", "agent": role,
                "created_at": "2026-08-07T00:00:00Z", "instruction": "do", "inputs": [],
                "criteria": ["c1"], "constraints": [],
                "context": ({"review_lens": "evidence_provenance", "review_focus": "f"}
                            if role == "judge" else {})}
        tp = d / "task.json"; tp.write_text(json.dumps(task))
        rp = d / "resp.json"; rp.write_text(json.dumps(response))
        argv = ["run-task", "--runtime", "mock", "--agent", role, "--agent-specs-dir", SPECS,
                "--task", str(tp), "--exchange-dir", str(d / "ex"), "--mock-response", str(rp),
                "--mode", mode]
        if run_dir:
            argv += ["--run-dir", str(run_dir)]
        return cli.main(argv)

    def test_router_strategy_per_role(self):
        from runtimes.pydantic_ai.production_router import acceptance_strategy
        from orchestration.specs import load_agent_specs
        specs = load_agent_specs(SPECS)
        for role, expected in EXPECTED_STRATEGY.items():
            self.assertEqual(acceptance_strategy(specs[role]), expected, role)

    def test_all_seven_roles_route_from_cli(self):
        from runtimes.pydantic_ai import cli
        for role in EXPECTED_STRATEGY:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = _v7_run(Path(tmp) / "run") if EXPECTED_STRATEGY[role] == "producer_dispatch" else None
                code = self._cli(role, TYPED_OUTPUT[role], tmp, mode="shadow", run_dir=run_dir)
                self.assertEqual(code, cli.EXIT_SUCCESS, f"{role} routing failed (exit {code})")

    def test_producer_primary_reaches_and_runs_trusted_executor(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _v7_run(Path(tmp) / "run")
            prop = _proposal("simulation", "compute_nve_drift", key="nve1")
            prop["parameters"] = {"energies": [0.0, 0.1, 0.2], "timestep_fs": 1.0, "n_atoms": 1}
            code = self._cli("simulation", prop, tmp, mode="primary", run_dir=run_dir)
            self.assertEqual(code, cli.EXIT_SUCCESS)  # primary -> dispatch -> real executor EXECUTED

    def test_wrong_role_payload_rejected(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _v7_run(Path(tmp) / "run")
            # a data-curator invocation whose output claims ml-trainer + a foreign action
            bad = _proposal("ml-trainer", "train_committee")
            code = self._cli("data-curator", bad, tmp, mode="primary", run_dir=run_dir)
            self.assertNotEqual(code, cli.EXIT_SUCCESS)  # fail-closed

    def test_shadow_mode_no_controller_mutation(self):
        from runtimes.pydantic_ai.production_router import run_role
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.models import RuntimeContext
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController
        from orchestration.specs import load_agent_specs
        with tempfile.TemporaryDirectory() as tmp:
            controller = RunController(_v7_run(Path(tmp) / "run"))
            ex = Path(tmp) / "ex"; ex.mkdir()
            spec = load_agent_specs(SPECS)["data-curator"]
            raw = json.dumps(_proposal("data-curator", "generate_group_split", key="gg"))
            ctx = RuntimeContext(exchange_dir=str(ex), repo_root=str(ROOT))
            res = run_role(MockAgentRuntime(lambda t, s, ts: (raw, (0, 0))),
                           {"task_id": "t", "agent": "data-curator", "inputs": []}, spec, ctx,
                           controller=controller, registry=build_executor_registry(), mode="shadow")
            self.assertFalse(res.controller_mutated)               # shadow never mutates
            self.assertNotIn("gg", controller.state.get("idempotency", {}))

    def test_judge_routes_to_gate_path(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            code = self._cli("judge", TYPED_OUTPUT["judge"], tmp, mode="validate-only")
            self.assertEqual(code, cli.EXIT_SUCCESS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
