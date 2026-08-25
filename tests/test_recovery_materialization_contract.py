"""Recovery-materialization contract -- the generic fix for the ffv4m RECOVERY_EXECUTION_UNVERIFIED
dead-loop (docs/postmortems/ffv4m_recovery_execution_unverified.md).

The defect: a data_repair RecoveryPlan dispatched a corrective ``action_type`` equal to its return
stage's OWN deterministic route action (``build_data_coverage_report``). Re-running that executor on
unchanged inputs re-emitted a byte-identical artifact (``DUPLICATE``), which
``verify_recovery_execution`` rejects as unchanged -- an unbreakable ``DUPLICATE ->
RECOVERY_EXECUTION_UNVERIFIED`` loop. There was no pre-acceptance check that the chosen corrective
action could actually MATERIALIZE a changed artifact at/downstream of the return stage.

Two layers are exercised here:

* the STATIC compatibility invariant: every registered corrective capability declares the
  materializing transitions its executor can produce, every declaration is non-empty and drawn from
  the shared MATERIALIZING_TRANSITIONS vocabulary (no "notes-only" corrective capability), and the
  pure classifier reports a provable deterministic no-op as ``None``;

* the end-to-end acceptance gate in ``RunController.propose_recovery`` (the sole authoritative
  deterministic validator), reproducing the exact ffv4m class on a run whose return stage's route
  action is KNOWN: the UNSUPPORTED same-route repair is refused BEFORE approval (no dispatch, no
  loop), while a SUPPORTED repair dispatching a DISTINCT corrective action is accepted and records
  the materializing transition. A stage with no route metadata (unknown route) is never rejected --
  ``verify_recovery_execution`` stays the backstop there.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import (
    DEFAULT_RECOVERY_CAPABILITY_ROSTER, RECOVERY_CAPABILITY_MATERIALIZATION, RunController,
)
from workflow.recovery_taxonomy import (
    MATERIALIZING_TRANSITIONS, classify_recovery_materialization,
)

_CRITERION = "artifact is complete and internally consistent"
_ROUTE_ACTION = "build_data_coverage_report"
_SUPPORT_ONLY_CAPABILITIES = frozenset(
    {"root_cause_analysis", "orchestration", "evidence_repair"})


class MaterializationStaticInvariantTests(unittest.TestCase):
    """Build-time compatibility invariant -- no code path is driven, only the declared contract."""

    def test_every_declaration_is_a_nonempty_subset_of_the_shared_vocabulary(self):
        # "No notes-only capability is allowed": a corrective capability that declares no
        # materializing transition could never satisfy the hash-change verification invariant.
        for capability, transitions in RECOVERY_CAPABILITY_MATERIALIZATION.items():
            self.assertIsInstance(transitions, frozenset)
            self.assertTrue(transitions, f"{capability!r} declares no materializing transition")
            self.assertLessEqual(
                transitions, MATERIALIZING_TRANSITIONS,
                f"{capability!r} declares a transition outside MATERIALIZING_TRANSITIONS")

    def test_corrective_capabilities_are_registered_roster_capabilities(self):
        for capability in RECOVERY_CAPABILITY_MATERIALIZATION:
            self.assertIn(capability, DEFAULT_RECOVERY_CAPABILITY_ROSTER)

    def test_support_only_roles_are_not_declared_corrective(self):
        # Support-only roles materialize nothing on their own; they must not be mistaken for a
        # corrective capability by appearing in the materialization contract.
        for capability in _SUPPORT_ONLY_CAPABILITIES:
            self.assertIn(capability, DEFAULT_RECOVERY_CAPABILITY_ROSTER)
            self.assertNotIn(capability, RECOVERY_CAPABILITY_MATERIALIZATION)

    def test_classifier_reports_a_same_route_rerun_as_a_provable_no_op(self):
        self.assertIsNone(classify_recovery_materialization(
            return_stage_route_action=_ROUTE_ACTION,
            corrective_action_type=_ROUTE_ACTION,
            return_stage_supersedes_inputs=False,
            authorizes_scientific_recompute=False))
        # An absent corrective action (a bare forward re-run of the route action) is the same no-op.
        self.assertIsNone(classify_recovery_materialization(
            return_stage_route_action=_ROUTE_ACTION,
            corrective_action_type=None,
            return_stage_supersedes_inputs=False,
            authorizes_scientific_recompute=False))

    def test_classifier_names_each_materializing_transition(self):
        self.assertEqual("scientific_recompute", classify_recovery_materialization(
            return_stage_route_action=_ROUTE_ACTION, corrective_action_type=_ROUTE_ACTION,
            return_stage_supersedes_inputs=True, authorizes_scientific_recompute=True))
        self.assertEqual("input_supersession_replan", classify_recovery_materialization(
            return_stage_route_action=_ROUTE_ACTION, corrective_action_type=_ROUTE_ACTION,
            return_stage_supersedes_inputs=True, authorizes_scientific_recompute=False))
        self.assertEqual("distinct_evidence_artifact", classify_recovery_materialization(
            return_stage_route_action=_ROUTE_ACTION,
            corrective_action_type="build_coverage_evidence_addendum",
            return_stage_supersedes_inputs=False, authorizes_scientific_recompute=False))
        # Every value the classifier can return is a member of the shared vocabulary (or None).
        for transition in ("scientific_recompute", "input_supersession_replan",
                           "distinct_evidence_artifact"):
            self.assertIn(transition, MATERIALIZING_TRANSITIONS)


class Ffv4mDeadLoopAcceptanceGateTests(unittest.TestCase):
    """Drive real ``propose_recovery`` on a run whose data_coverage stage declares a KNOWN route
    action, reproducing the exact ffv4m failure class at the acceptance boundary."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _controller(self, *, routed=True):
        stage = {"name": "data_coverage", "command": None,
                 "outputs": ["artifacts/data_coverage.json"],
                 "gate": {"criteria": [_CRITERION]}}
        if routed:
            stage["pydantic_ai"] = {"action": _ROUTE_ACTION}
        cfg = self.root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "ffv4m-class", "stages": [stage]}))
        controller = RunController.initialize(cfg, self.root / "run")
        artifact = controller.run_dir / "artifacts/data_coverage.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        self._artifact = artifact
        artifact.write_text(json.dumps({"coverage_status": "PARTIAL"}))
        controller.complete_external_stage("data_coverage", [artifact])
        controller.record_gate("data_coverage", "REVISE")
        return controller

    def _plan(self, *, corrective_action_type=None):
        plan = {
            "schema_version": 1, "proposed_by": "automation",
            "failed_stage": "data_coverage", "failure_category": "dataset_coverage",
            "root_cause": "coverage evidence was not surfaced at criterion granularity",
            "responsible_capability": "data_repair", "return_stage": "data_coverage",
            "proposed_changes": [{"type": "surface_coverage_evidence"}],
            "labeling": {"teacher_relabel": False, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["data_coverage"]},
            "estimated_cost": {},
        }
        if corrective_action_type is not None:
            plan["recovery_context"] = {
                "corrective_action": {"action_type": corrective_action_type, "parameters": {}}}
        return plan

    def _propose(self, controller, plan):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        return controller.propose_recovery(plan_path)

    def test_unsupported_same_route_repair_is_refused_before_approval(self):
        # The exact ffv4m defect: the corrective action equals the return stage's own route action,
        # so re-running it on unchanged inputs can only ever DUPLICATE.
        controller = self._controller(routed=True)
        with self.assertRaisesRegex(ValueError, "RECOVERY_EXECUTION_UNVERIFIED"):
            self._propose(controller, self._plan(corrective_action_type=_ROUTE_ACTION))
        # Nothing was bound: the gate is still waiting, no recovery entered the dead-loop path.
        self.assertEqual(controller.state["pending_recovery"]["status"], "required")
        self.assertEqual(controller.state.get("recoveries", []), [])

    def test_bare_forward_rerun_with_no_corrective_action_is_refused(self):
        # A recovery with no corrective_action at all is a bare re-run of the route action -- the
        # same provable no-op, refused identically.
        controller = self._controller(routed=True)
        with self.assertRaisesRegex(ValueError, "RECOVERY_EXECUTION_UNVERIFIED"):
            self._propose(controller, self._plan(corrective_action_type=None))
        self.assertEqual(controller.state.get("recoveries", []), [])

    def test_supported_distinct_corrective_action_is_accepted_and_records_the_transition(self):
        # A corrective action that dispatches a DISTINCT executor materializes a distinct evidence
        # artifact, so it is accepted and the recovery record carries the transition.
        controller = self._controller(routed=True)
        recovery = self._propose(
            controller, self._plan(corrective_action_type="build_coverage_evidence_addendum"))
        self.assertEqual(recovery["status"], "proposed")
        self.assertEqual(recovery["materialization_transition"], "distinct_evidence_artifact")

    def test_unknown_route_stage_is_not_rejected_and_leaves_verification_as_backstop(self):
        # When the return stage declares no route metadata the route is unknown; even a bare
        # re-run (the provable no-op shape) is NOT rejected -- the guard is SOUND to skip (never a
        # false rejection) and verify_recovery_execution stays the backstop.
        controller = self._controller(routed=False)
        recovery = self._propose(controller, self._plan(corrective_action_type=None))
        self.assertEqual(recovery["status"], "proposed")
        self.assertIsNone(recovery["materialization_transition"])


if __name__ == "__main__":
    unittest.main()
