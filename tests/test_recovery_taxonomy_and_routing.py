"""Priority #3 requirements #1 and #4, exercised through propose_recovery itself.

Requirement #1 (unified diagnosis/recovery vocabulary): propose_recovery resolves
failure_category against the SAME workflow.recovery_taxonomy registry used by the Analyst's
RootCauseClassification -- an unregistered category must be rejected, and a declared
failure_domain that disagrees with the category's registered domain must be rejected too.

Requirement #4 (capability-based responsible-agent routing): a plan may route via a registered
responsible_capability (resolved against this run's own recovery_capability_roster, or the
built-in default roster if the run declares none) instead of a hardcoded literal agent name; an
unregistered capability, or a responsible_agent/responsible_capability mismatch, is rejected.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController


class _RecoveryPlanFixture(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _controller(self, root, *, recovery_capability_roster=None):
        cfg_dict = {"run_id": "taxonomy-routing", "stages": [{
            "name": "validation", "command": None, "outputs": ["artifacts/result.txt"],
            "gate": {"criteria": [self.GATE_CRITERION]},
        }]}
        if recovery_capability_roster is not None:
            cfg_dict["recovery_capability_roster"] = recovery_capability_roster
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump(cfg_dict))
        controller = RunController.initialize(cfg, root / "run")
        result = controller.run_dir / "artifacts/result.txt"
        result.write_text("result")
        controller.complete_external_stage("validation", [result])
        controller.record_gate("validation", "REVISE")
        return controller

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

    def _propose(self, root, controller, plan):
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        return controller.propose_recovery(plan_path)


class TaxonomyValidationTests(_RecoveryPlanFixture):
    def test_valid_registered_category_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            recovery = self._propose(root, controller, self._base_plan())
            self.assertEqual(recovery["status"], "proposed")

    def test_unregistered_failure_category_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan = self._base_plan(failure_category="not_a_real_category_xyz")
            with self.assertRaisesRegex(ValueError, "invalid failure_category"):
                self._propose(root, controller, plan)

    def test_mismatched_failure_domain_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            # dataset_coverage is registered under a data-domain, not "student_training" --
            # a plan asserting the wrong domain for a real category must fail closed.
            plan = self._base_plan(failure_domain="student_training")
            with self.assertRaisesRegex(ValueError, "failure_domain"):
                self._propose(root, controller, plan)

    def test_declared_domain_matching_the_registry_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            from workflow import recovery_taxonomy
            root = Path(tmp)
            controller = self._controller(root)
            resolved = recovery_taxonomy.resolve_failure_code("dataset_coverage")
            plan = self._base_plan(failure_domain=resolved.domain)
            recovery = self._propose(root, controller, plan)
            self.assertEqual(recovery["status"], "proposed")


class CapabilityRoutingTests(_RecoveryPlanFixture):
    def test_legacy_responsible_agent_path_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            recovery = self._propose(root, controller, self._base_plan())
            self.assertEqual(recovery["resolved_responsible_agent"], "data-curator")
            self.assertIsNone(recovery["resolved_responsible_capability"])

    def test_responsible_agent_not_in_roster_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan = self._base_plan(responsible_agent="totally-unregistered-role")
            with self.assertRaisesRegex(ValueError, "not a registered recovery role"):
                self._propose(root, controller, plan)

    def test_responsible_capability_resolves_via_default_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan = self._base_plan(responsible_capability="data_repair")
            del plan["responsible_agent"]
            recovery = self._propose(root, controller, plan)
            self.assertEqual(recovery["resolved_responsible_capability"], "data_repair")
            self.assertEqual(recovery["resolved_responsible_agent"], "data-curator")

    def test_responsible_capability_resolves_via_run_declared_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(
                root, recovery_capability_roster={"custom_capability": "custom-role"})
            plan = self._base_plan(responsible_capability="custom_capability",
                                   responsible_agent="custom-role")
            recovery = self._propose(root, controller, plan)
            self.assertEqual(recovery["resolved_responsible_agent"], "custom-role")
            self.assertEqual(recovery["resolved_responsible_capability"], "custom_capability")

    def test_unregistered_capability_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan = self._base_plan(responsible_capability="not_a_real_capability")
            del plan["responsible_agent"]
            with self.assertRaisesRegex(ValueError, "not registered in this run's roster"):
                self._propose(root, controller, plan)

    def test_responsible_agent_inconsistent_with_capability_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._controller(root)
            plan = self._base_plan(responsible_capability="data_repair",
                                   responsible_agent="some-unrelated-role")
            with self.assertRaisesRegex(ValueError, "does not match responsible_capability"):
                self._propose(root, controller, plan)


if __name__ == "__main__":
    unittest.main()
