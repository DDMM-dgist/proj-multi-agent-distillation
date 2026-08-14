"""Priority #3 requirement #8: cumulative loop-safety policy, explicit and human-configured.

Proves each OPTIONAL recovery_policy limit (max_recovery_attempts, allowed_action_types,
cumulative_budget, max_repeated_signature) is enforced by RunController.propose_recovery exactly
at its declared boundary when a run configures it, is completely unenforced (no invented default)
when a run does not configure it, and -- for max_repeated_signature -- can only be bypassed by a
separate, explicit plan.escalation_acknowledged + plan.escalation_rationale, never silently.

Each "next attempt" is produced through the REAL deterministic lifecycle (approve_recovery ->
start_iteration -> re-fail the same gate), not a shortcut, since these policy limits are keyed off
`self.state["recoveries"]`, which only grows through that lifecycle.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController


class _RecoveryPolicyFixture(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _controller(self, root, *, recovery_policy=None):
        cfg_dict = {"run_id": "loop-safety", "stages": [{
            "name": "validation", "command": None, "outputs": ["artifacts/result.txt"],
            "gate": {"criteria": [self.GATE_CRITERION]},
        }]}
        if recovery_policy is not None:
            cfg_dict["recovery_policy"] = recovery_policy
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump(cfg_dict))
        controller = RunController.initialize(cfg, root / "run")
        self._fail_gate(controller)
        return controller

    def _pass_gate(self, controller, stage):
        artifacts = {a["path"]: a["sha256"] for a in controller.stage_artifacts(stage)}
        vote_path = controller.run_dir / "gates" / f"{stage}.votes.json"
        criteria = controller.stage(stage).get("gate_criteria")
        lenses = controller.stage(stage).get("gate_review_lenses")
        def vote(judge_id, lens):
            return {"judge_id": judge_id, "review_lens": lens["id"], "verdict": "PASS",
                    "criteria_checked": [{"criterion": criterion, "value_read": "verified",
                                          "ok": True} for criterion in criteria],
                    "rationale": "ok", "required_fix": ""}
        vote_path.write_text(json.dumps({
            "stage": stage, "criteria": criteria, "review_lenses": lenses,
            "artifact_sha256": artifacts, "decision": "PASS",
            "votes": [vote(f"judge-{index}", lens) for index, lens in enumerate(lenses, 1)],
        }))
        controller.record_gate(stage, votes_path=vote_path)

    def _fail_gate(self, controller, content="same-evidence"):
        result = controller.run_dir / "artifacts/result.txt"
        result.write_text(content)
        controller.complete_external_stage("validation", [result])
        controller.record_gate("validation", "REVISE")

    def _approve_and_reenter(self, controller, content="same-evidence"):
        """Complete the real lifecycle for the current pending recovery so the SAME gate is
        pending again -- this is the only way `recoveries` grows, since there is no
        reject/withdraw shortcut in the framework."""
        controller.approve_recovery("researcher", "approved retry")
        controller.start_iteration()
        self._fail_gate(controller, content)

    def _base_plan(self, **overrides):
        plan = {
            "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
            "failure_category": "dataset_coverage", "root_cause": "low support in slice",
            "responsible_agent": "data-curator", "return_stage": "validation",
            "proposed_changes": [{"type": "add_deployment_frames"}],
            "labeling": {"teacher_relabel": False, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["validation"]},
            "estimated_cost": {},
        }
        plan.update(overrides)
        return plan

    def _propose(self, root, controller, plan, name="plan.json"):
        plan_path = root / name
        plan_path.write_text(json.dumps(plan))
        return controller.propose_recovery(plan_path)


class MaxRecoveryAttemptsTests(_RecoveryPolicyFixture):
    def test_no_policy_means_unlimited_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            for i in range(3):
                recovery = self._propose(root, controller, self._base_plan(), name=f"p{i}.json")
                self.assertEqual(recovery["status"], "proposed")
                self._approve_and_reenter(controller)

    def test_attempt_within_limit_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, recovery_policy={"max_recovery_attempts": 2})
            recovery = self._propose(root, controller, self._base_plan())
            self.assertEqual(recovery["status"], "proposed")

    def test_attempt_exceeding_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root, recovery_policy={"max_recovery_attempts": 1})
            self._propose(root, controller, self._base_plan())
            self._approve_and_reenter(controller)
            with self.assertRaisesRegex(ValueError, "max_recovery_attempts"):
                self._propose(root, controller, self._base_plan(), name="p2.json")


class AllowedActionTypesTests(_RecoveryPolicyFixture):
    def test_disallowed_action_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"allowed_action_types": ["add_deployment_frames"]})
            plan = self._base_plan(proposed_changes=[{"type": "fix_protocol"}])
            with self.assertRaisesRegex(ValueError, "not permitted by"):
                self._propose(root, controller, plan)

    def test_allowed_action_type_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"allowed_action_types": ["add_deployment_frames"]})
            recovery = self._propose(root, controller, self._base_plan())
            self.assertEqual(recovery["status"], "proposed")


class CumulativeBudgetTests(_RecoveryPolicyFixture):
    def test_cumulative_cost_within_budget_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"cumulative_budget": {"gpu_hours": 10}})
            plan = self._base_plan(estimated_cost={"gpu_hours": 4})
            recovery = self._propose(root, controller, plan)
            self.assertEqual(recovery["status"], "proposed")

    def test_cumulative_cost_across_attempts_exceeding_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"cumulative_budget": {"gpu_hours": 5}})
            self._propose(root, controller, self._base_plan(estimated_cost={"gpu_hours": 4}))
            self._approve_and_reenter(controller)
            plan = self._base_plan(estimated_cost={"gpu_hours": 4})
            with self.assertRaisesRegex(ValueError, "cumulative_budget"):
                self._propose(root, controller, plan, name="p2.json")


class StagnationEscalationTests(_RecoveryPolicyFixture):
    def test_repeated_signature_under_the_limit_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"max_repeated_signature": 2})
            self._propose(root, controller, self._base_plan())
            self._approve_and_reenter(controller)
            recovery = self._propose(root, controller, self._base_plan(), name="p2.json")
            self.assertEqual(recovery["status"], "proposed")

    def test_repeated_signature_at_limit_is_rejected_without_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"max_repeated_signature": 1})
            self._propose(root, controller, self._base_plan())
            self._approve_and_reenter(controller)
            with self.assertRaisesRegex(ValueError, "recovery_signature repeats"):
                self._propose(root, controller, self._base_plan(), name="p2.json")

    def test_acknowledged_escalation_bypasses_the_stagnation_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"max_repeated_signature": 1})
            self._propose(root, controller, self._base_plan())
            self._approve_and_reenter(controller)
            plan = self._base_plan(escalation_acknowledged=True,
                                   escalation_rationale="human reviewed, proceeding anyway")
            recovery = self._propose(root, controller, plan, name="p2.json")
            self.assertEqual(recovery["status"], "proposed")

    def test_escalation_flag_without_rationale_does_not_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_policy={"max_repeated_signature": 1})
            self._propose(root, controller, self._base_plan())
            self._approve_and_reenter(controller)
            plan = self._base_plan(escalation_acknowledged=True)
            with self.assertRaisesRegex(ValueError, "recovery_signature repeats"):
                self._propose(root, controller, plan, name="p2.json")

    def test_different_return_stage_changes_the_signature_and_is_not_flagged(self):
        # A materially different corrective plan (different return_stage) must NOT be treated as
        # a repeat -- the signature is deliberately content-sensitive, not attempt-count-only.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({
                "run_id": "loop-safety-2",
                "recovery_policy": {"max_repeated_signature": 1},
                "stages": [
                    {"name": "data", "command": None, "outputs": ["artifacts/data.txt"],
                     "gate": {"criteria": [self.GATE_CRITERION]}},
                    {"name": "validation", "command": None, "outputs": ["artifacts/result.txt"],
                     "gate": {"criteria": [self.GATE_CRITERION]}},
                ],
            }))
            controller = RunController.initialize(cfg, root / "run")
            data = controller.run_dir / "artifacts/data.txt"
            data.write_text("data")
            controller.complete_external_stage("data", [data])
            self._pass_gate(controller, "data")
            self._fail_gate(controller)
            self._propose(root, controller, self._base_plan())
            self._approve_and_reenter(controller)
            plan = self._base_plan(return_stage="data")
            recovery = self._propose(root, controller, plan, name="p2.json")
            self.assertEqual(recovery["status"], "proposed")


if __name__ == "__main__":
    unittest.main()
