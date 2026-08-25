"""FE-044 -- recovery corrective ``acquire_structures`` must resolve its approval boundary through the
SAME canonical typed-effect binding the forward acquisition path uses
(``run_production_stage`` -> ``_bind_acquisition_plan_for_stage``), so a geometry-only / existing-pool
reacquisition is NOT spuriously gated behind ``costly_teacher_labeling`` while a genuinely
Teacher-driven acquisition recipe still is.

Background (the live ffv4t regression): FE-043 correctly bound a fresh superseding existing-pool
AcquisitionPlan for a ``return_stage=acquisition`` recovery, but ``_dispatch_recovery_corrective_action``
built its proposal WITHOUT running it through ``_bind_acquisition_plan_for_stage``. So the framework-
authoritative ``performs_teacher_inference`` effect flag was never injected, ``actions.
resolve_action_approval_boundary`` fail-closed, and the geometry-only corrective ``acquire_structures``
tripped ``APPROVAL_REQUIRED: costly_teacher_labeling`` (exit 2) -- even though the byte-identical
forward dispatch of the SAME bound plan ran with the boundary relaxed to ``None``.

These tests drive the REAL production entry points (``run_campaign`` / ``_run_campaign_loop`` ->
``_dispatch_recovery_corrective_action`` -> ``dispatch.authorize_and_execute`` ->
``resolve_action_approval_boundary``). Only the (expensive, geometry-heavy) canonical planner and the
(expensive) ``acquire_structures`` executor are replaced by observable fixtures; the approval-boundary
resolution and the ``_bind_acquisition_plan_for_stage`` effect classification are the REAL code paths
under test. Crucially -- unlike the FE-043 wiring tests -- the corrective registry keeps the REAL
``costly_teacher_labeling`` default boundary (never overridden to ``None``), so the relaxation proved
here is the framework's typed-effect relaxation, not a test shortcut.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtimes.pydantic_ai import cli
from runtimes.pydantic_ai.actions import resolve_action_approval_boundary
from runtimes.pydantic_ai.cli import (
    _bind_acquisition_plan_for_stage, _proposal_from_stage, _stage_config,
)
from runtimes.pydantic_ai.default_acquisition_provider import _acquisition_plan_already_bound
from runtimes.pydantic_ai.dispatch import ActionDescriptor
from runtimes.pydantic_ai.executors import build_executor_registry
from workflow.controller import RunController
from workflow.integrity import sha256_file

from test_fe043_recovery_reacquisition_wiring import (
    ROOT, FFV4S_UNSUPPORTED_CLASSES, _Fe043Fixture, _write_plan,
)

# The action_type the acquisition stage routes to (the real per-action default boundary for it is
# ``costly_teacher_labeling`` -- APPROVAL_GATED_ACTIONS["acquire_structures"]).
ACQUIRE = "acquire_structures"


def _write_perturbation_plan(path: Path, *, seed: int) -> Path:
    """A VALID perturbation-projection AcquisitionPlan (the ``augment-atoms`` generation path). Passes
    ``_validate_acquisition_plan`` with no file I/O; paired with an ``augment-atoms`` acquisition_config
    it is classified as PERFORMING Teacher inference (the executor binds the Teacher calculator during
    generation), so its approval boundary must STAY ``costly_teacher_labeling``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "eligible_source_categories": ["amorphous_bulk_SiO2"],
        "selected_parent_structure_ids": [f"p-{seed}-0", f"p-{seed}-1"],
        "selected_source_global_indices": [seed * 10, seed * 10 + 1],
        "n_parents": 2, "n_per_structure": 3, "expected_output_count": 6,
        "T_K": 300.0, "beta": 1.0, "sigma_range_A": [0.02, 0.08], "cell_sigma": 0.0,
        "seed": seed, "duplicate_handling": "reject_duplicates",
        "protected_reference_exclusion_report": {
            "status": "PASS", "dft_labels_used_as_selection_scores": False},
    }))
    return path


class _Fe044Fixture(_Fe043Fixture):
    """Reuses the FE-043 drive-to-approved-recovery machinery but keeps the REAL costly boundary on the
    corrective ``acquire_structures`` executor, so the approval decision is made by the framework's
    typed-effect classification (FE-044), not by a test-provided ``approval_boundary=None`` override."""

    def _corrective_registry(self, *, executor=None):
        test = self
        run_dir = self.controller.run_dir

        def default_executor(proposal):
            params = proposal.get("parameters") or {}
            test.dispatch_saw_plan_path = params.get("acquisition_plan_path")
            test.dispatch_saw_performs_teacher = params.get("performs_teacher_inference")
            test.calls.append("dispatch")
            out = run_dir / "artifacts/acquired.json"
            out.write_text(json.dumps({"role": "acquired", "revision": 2, "n_frames": 61}))
            return {"path": str(out), "manifest": {"n_frames": 61}, "sha256": sha256_file(out)}

        registry = build_executor_registry()
        # Keep the REAL costly_teacher_labeling default boundary (the whole point of FE-044): only the
        # executor body is a fixture; the boundary the framework must relax/keep is untouched.
        registry[ACQUIRE] = ActionDescriptor(
            action_type=ACQUIRE, role="data-curator",
            approval_boundary="costly_teacher_labeling",
            executor=executor or default_executor)
        return registry


class Fe044ApprovalRoutingTests(_Fe044Fixture):
    def setUp(self):
        super().setUp()
        self.controller = self._staged_controller()
        self._drive_to_approved_recovery(self.controller)

    # 1 -- forward and recovery dispatch resolve IDENTICAL approval semantics for the SAME bound plan:
    #      both run _bind_acquisition_plan_for_stage, both classify the existing-pool plan geometry-only
    #      (performs_teacher_inference=False), both resolve the boundary to None.
    def test_1_forward_and_recovery_identical_boundary_for_same_plan(self):
        # Drive the recovery corrective dispatch (which supersedes the stale plan and binds the fresh
        # superseding one), capturing the performs_teacher_inference the corrective proposal carried.
        self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        rec_boundary = resolve_action_approval_boundary(
            ACQUIRE, "costly_teacher_labeling",
            {"performs_teacher_inference": self.dispatch_saw_performs_teacher})
        self.assertIs(self.dispatch_saw_performs_teacher, False)

        # Now reconstruct the FORWARD proposal for the SAME single active plan (stale is superseded) and
        # resolve its boundary through the identical binder -> both paths must agree, and both relax.
        c = RunController(self.controller.run_dir)
        fwd_proposal, _role = _proposal_from_stage(c, "acquisition", _stage_config(c, "acquisition"))
        fwd_proposal = _bind_acquisition_plan_for_stage(c, fwd_proposal)
        fwd_boundary = resolve_action_approval_boundary(
            ACQUIRE, "costly_teacher_labeling", fwd_proposal["parameters"])
        self.assertIs(fwd_proposal["parameters"]["performs_teacher_inference"], False)
        self.assertIsNone(fwd_boundary)
        self.assertEqual(fwd_boundary, rec_boundary)

    # 2 -- existing-pool geometry-only recovery: the corrective acquire_structures dispatches WITHOUT a
    #      costly_teacher_labeling approval (the boundary is relaxed by the injected effect flag), even
    #      though the registry descriptor carries the REAL costly default and NO approval was granted.
    def test_2_existing_pool_recovery_no_costly_approval(self):
        result = self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        self.assertNotIn("APPROVAL_REQUIRED", result.message)
        self.assertIn("dispatch", self.calls)  # the executor was actually reached
        self.assertEqual(self.forward_stage_reached, "acquisition")
        c = RunController(self.controller.run_dir)
        self.assertEqual(c.stage("acquisition")["status"], "completed")

    # 3 -- a genuinely Teacher-driven recovery recipe (augment-atoms perturbation generation) still
    #      requires costly_teacher_labeling: the SAME binding classifies it performs_teacher_inference=True
    #      and the boundary is NOT relaxed -> fail-closed APPROVAL_REQUIRED (no silent Teacher compute).
    def test_3_teacher_driven_recovery_still_costly(self):
        # A readable augment-atoms acquisition_config carried on the approved corrective action, plus a
        # perturbation (non-existing-pool) plan bound by the planner -> classified Teacher-driving.
        cfg = self.root / "augment.yaml"
        cfg.write_text("kind: augment-atoms\n")
        recovery = self.controller.state["recoveries"][-1]
        recovery["plan"]["recovery_context"]["corrective_action"]["parameters"] = {
            "acquisition_config": str(cfg.resolve())}
        self.controller.save()

        def _bind_perturbation(controller, **_kwargs):
            self.calls.append("planner")
            if not _acquisition_plan_already_bound(controller):
                controller.bind_new_input(
                    _write_perturbation_plan(self.root / "reacq" / "acquisition_plan.json", seed=5))
            return None

        result = self._run_after_recovery(planner=_bind_perturbation)
        self.assertEqual(result.outcome, cli.CAMPAIGN_FAILED, result.message)
        self.assertIn("APPROVAL_REQUIRED", result.message)
        self.assertIn("costly_teacher_labeling", result.message)
        # The costly executor body must never have run (fail-closed before execution).
        self.assertNotIn("dispatch", self.calls)

    # 4 -- exact ffv4t shape: a fresh superseding existing-pool plan is already bound by FE-043, the
    #      coverage-gap classes reach the planner, and the corrective acquire_structures executes with
    #      NO spurious APPROVAL_REQUIRED -- the end-to-end regression ffv4t hit is closed.
    def test_4_ffv4t_fixture_corrective_acquire_executes(self):
        result = self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        self.assertNotIn("APPROVAL_REQUIRED", result.message)
        self.assertEqual(self.planner_saw_classes, FFV4S_UNSUPPORTED_CLASSES)
        self.assertIsNotNone(self.dispatch_saw_plan_path)
        c = RunController(self.controller.run_dir)
        self.assertEqual(c.stage("acquisition")["status"], "completed")
        acquired = json.loads((c.run_dir / "artifacts/acquired.json").read_text())
        self.assertEqual(acquired["revision"], 2)

    # 5 -- no approval leakage: relaxing acquire_structures does NOT record any costly_teacher_labeling
    #      approval, and a genuine downstream Teacher-labeling action (inherent-costly) is never relaxed
    #      by the same effect flag -- so later Teacher labeling still stops at its own costly boundary.
    def test_5_no_leakage_to_downstream_teacher_labeling(self):
        self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        c = RunController(self.controller.run_dir)
        # The geometry-only relaxation is NOT an approval: no costly_teacher_labeling grant was recorded
        # in the controller's action_approvals store (a dict keyed by boundary).
        self.assertNotIn("costly_teacher_labeling", c.state.get("action_approvals", {}))
        # An inherent Teacher-labeling action can never be relaxed by a performs_teacher_inference=False
        # flag (the executor binds the Teacher regardless); its boundary stays costly.
        self.assertEqual(
            resolve_action_approval_boundary(
                "label_with_teacher", "costly_teacher_labeling",
                {"performs_teacher_inference": False}),
            "costly_teacher_labeling")


if __name__ == "__main__":
    unittest.main()
