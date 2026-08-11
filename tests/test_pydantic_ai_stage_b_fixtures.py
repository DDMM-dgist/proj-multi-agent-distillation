"""Stage B: network-free validation of the seven frozen role fixtures + producer dry-run proof.

NO provider, NO model, NO GPU. Confirms every Stage B task is well-formed and portable, that each
producer's intended action dry-run-dispatches to DRY_RUN with ZERO controller mutation, and that the
v7 controller manifest the runner writes actually loads. Skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "stage_b_validate", ROOT / "work" / "stage_b_validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The exact minimal v7 manifest the runner writes for producer --run-dir (mirrors the controller's
# initialize() shape + the v7 additive keys).
V7_MANIFEST = {
    "schema_version": 7, "run_id": "stageB-smoke",
    "created_at": "2026-08-07T00:00:00+00:00", "updated_at": "2026-08-07T00:00:00+00:00",
    "workflow_config": "w", "artifacts": [], "project_dir": "p", "inputs": [],
    "code_revision": "x", "events": [],
    "stages": [{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}],
    "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                    "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
    "recoveries": [], "pending_recovery": None,
    "runtime_attempts": [], "idempotency": {}, "action_approvals": {}, "scheduler_jobs": {},
}

# Minimal valid producer proposals (what a compliant model must emit); dry_run defaults True.
_PROD = {
    "data-curator": ("DataCuratorActionProposal", "inspect_dataset", "stageB-dc-inspect-0001"),
    "ml-trainer":   ("MLTrainerActionProposal", "compute_committee_disagreement", "stageB-mlt-cd-0001"),
    "simulation":   ("SimulationActionProposal", "compute_nve_drift", "stageB-sim-nve-0001"),
    "analyst":      ("AnalystActionProposal", "compare_force_errors", "stageB-an-cfe-0001"),
}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class StageBFixtureTests(unittest.TestCase):
    def test_all_seven_fixtures_valid_network_free(self):
        mod = _load_validator()
        ok, msgs = mod.validate_all(str(ROOT))
        self.assertTrue(ok, "Stage B fixture validation failed:\n" + "\n".join(msgs))
        # all seven roles are covered
        self.assertEqual(set(mod.STAGE_B_TASKS), {
            "orchestrator", "literature", "data-curator", "ml-trainer",
            "simulation", "analyst", "judge"})

    def test_v7_manifest_written_by_runner_loads(self):
        from workflow.controller import RunController, SCHEMA_VERSION
        self.assertEqual(SCHEMA_VERSION, 7)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "manifest.json").write_text(json.dumps(V7_MANIFEST))
            ctrl = RunController(str(d))
            self.assertEqual(ctrl.state["schema_version"], 7)

    def test_each_producer_action_dry_run_dispatches_with_zero_mutation(self):
        from runtimes.pydantic_ai import actions as A
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController
        reg = build_executor_registry()
        for role, (model_name, action, key) in _PROD.items():
            with tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                (d / "manifest.json").write_text(json.dumps(V7_MANIFEST))
                ctrl = RunController(str(d))
                Model = getattr(A, model_name)
                proposal = Model(action_type=action, run_id="stageB-smoke", stage="s",
                                 requested_at="2026-08-07T00:00:00Z",
                                 rationale="stage B smoke dry-run", idempotency_key=key).model_dump()
                self.assertEqual(proposal["requested_by_role"], role)  # role-scoped default
                self.assertTrue(proposal["dry_run"])                    # default dry-run
                outcome = dispatch_via_controller(proposal, controller=ctrl, registry=reg,
                                                  mode="dry_run")
                self.assertEqual(outcome.status, "DRY_RUN", f"{role}/{action}: {outcome.status}")
                # zero controller mutation: idempotency key never consumed on a dry run
                self.assertNotIn(key, ctrl.state.get("idempotency", {}), role)

    def test_orchestrator_fixture_is_plan_only(self):
        # attempt-2 fix 1: the orchestrator smoke is plan-only — no artifact dependency and an
        # explicit "call NO tools" instruction (so it cannot enter a read-tool loop).
        task = json.loads((ROOT / "examples/stage_b_smoke/orchestrator.json").read_text())
        self.assertEqual(task["inputs"], [])
        instr = task["instruction"].lower()
        self.assertIn("plan-only", instr)
        self.assertIn("not call any tool", instr)

    def test_judge_fixture_two_ordered_criteria_and_strict_contract(self):
        # attempt-2 fix 2: the compound criterion is split into two atomic ORDERED criteria; a
        # vote mirroring both passes canonical validation, while a mismatched vote still fails
        # (the contract is NOT relaxed).
        from orchestration.specs import load_agent_specs
        from orchestration.exchange import validate_agent_response
        task = json.loads((ROOT / "examples/stage_b_smoke/judge.json").read_text())
        self.assertEqual(len(task["criteria"]), 2, "judge task must have two atomic criteria")
        spec = load_agent_specs(str(ROOT / "agent_specs"))["judge"]
        good = {"review_lens": "evidence_provenance", "verdict": "PASS",
                "criteria_checked": [{"criterion": task["criteria"][0], "value_read": 12, "ok": True},
                                     {"criterion": task["criteria"][1], "value_read": "passed", "ok": True}],
                "rationale": "structure_count 12 and validation_status passed", "required_fix": ""}
        validate_agent_response(good, spec, task)  # must not raise
        # a vote that collapses the two criteria into one still fails the ordered-match contract
        bad = dict(good, criteria_checked=[
            {"criterion": "evidence.json has structure_count == 12 and validation_status == 'passed'",
             "value_read": True, "ok": True}])
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validate_agent_response(bad, spec, task)

    def test_producer_action_authorization_matrix(self):
        # Each chosen action is allowed for its role, not approval-gated, not out-of-scope.
        from runtimes.pydantic_ai.actions import (
            ROLE_ALLOWED_ACTIONS, APPROVAL_GATED_ACTIONS, CAPABILITY_REGISTRY)
        for role, (_m, action, _k) in _PROD.items():
            self.assertIn(action, ROLE_ALLOWED_ACTIONS[role], role)
            self.assertNotIn(action, APPROVAL_GATED_ACTIONS, action)
            self.assertNotIn(action, CAPABILITY_REGISTRY, action)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
