"""Phase 6/3 (updated for R3 production readiness): verify READY_HPC_APPROVAL_GATED actions
have a REAL backing (not just a label).

As of the "Close production readiness for R3 scientific workflow" change, these actions carry
a real inline executor in ``build_executor_registry()`` — Teacher/Student compute runs
in-process on the local research GPU workstation rather than through a separate remote-HPC
bridge, so there is no external scheduler step to hand the real callable to. The safety
invariant that matters is therefore no longer "no executor is registered"; it is that the
dispatch pipeline in ``dispatch.authorize_and_execute`` never lets that executor run except in
mode="primary" with a recorded human approval (see ``test_pydantic_ai_executors.py::
test_hpc_action_requires_approval_then_reaches_real_train_wrapper`` for the original instance
of this pattern). For each of the 9 HPC actions: the backing module.function exists and is
importable/callable; the action is approval-gated; a proposal for it is never EXECUTED in
dry-run mode or without a granted approval; status is READY_HPC_APPROVAL_GATED. If a backing
turns out not to exist, this test FAILS (per the stop rule).
Network-free (imports only; nothing is executed). Skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

HPC_ACTIONS = {
    # build_teacher_baseline/validate_teacher_reference compose several real calls (see their
    # BINDINGS.backing strings); label_with_teacher is their first, most expensive real step
    # (fresh Teacher inference), so it is the representative callable checked here.
    "build_teacher_baseline": "adapters.acquisition.label_with_teacher",
    "validate_teacher_reference": "adapters.acquisition.label_with_teacher",
    "build_teacher_physical_validation_target": "validation.teacher_physical_validation.compute_teacher_validation_target",
    "acquire_structures": "adapters.acquisition.acquire",
    "label_with_teacher": "adapters.acquisition.label_with_teacher",
    "train_committee": "workflow.steps.train_committee",
    "evaluate_heldout_fidelity": "workflow.steps.evaluate_committee",
    "run_teacher_md": "adapters.acquisition.run_teacher_md",
    "run_student_md": "workflow.steps.run_md",
}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class HpcBackingVerificationTests(unittest.TestCase):
    def test_backing_functions_exist_and_are_callable(self):
        for action, dotted in HPC_ACTIONS.items():
            module_name, func_name = dotted.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name, None)
            self.assertTrue(callable(fn), f"{action}: {dotted} is not callable/importable")

    def test_hpc_actions_have_real_backing_gated_by_dispatch(self):
        # The core safety invariant is no longer "no executor is registered" — R3 wired real
        # in-process Teacher/Student executors for these actions. What must still hold: the
        # backing is a real module.function, the action is approval-gated, and
        # dispatch.authorize_and_execute never reaches that executor except in mode="primary"
        # with a recorded human approval (verified behaviorally below, not just by status flags).
        from runtimes.pydantic_ai.executors import BINDINGS, build_executor_registry
        reg = build_executor_registry()
        for action in HPC_ACTIONS:
            b = BINDINGS[action]
            self.assertEqual(b.status, "READY_HPC_APPROVAL_GATED", action)
            self.assertTrue(b.real_execution_required_later, action)
            self.assertIsNotNone(reg[action].executor, action)       # real backing, per R3
            self.assertIn(HPC_ACTIONS[action].split(".")[0], b.backing, action)

    def test_hpc_actions_never_execute_without_dry_run_or_approval(self):
        # Behavioral proof of the actual safety invariant: even with a real executor registered,
        # authorize_and_execute must never reach EXECUTED unless mode="primary" AND a human
        # approval for the action's boundary has been granted on the run.
        from runtimes.pydantic_ai.actions import APPROVAL_GATED_ACTIONS
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController
        import json

        def _run(d: Path):
            state = {
                "schema_version": 7, "run_id": "r", "created_at": "2026-08-07T00:00:00+00:00",
                "updated_at": "2026-08-07T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
                "project_dir": "p", "inputs": [], "code_revision": "x", "events": [],
                "stages": [{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}],
                "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                                "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
                "recoveries": [], "pending_recovery": None,
                "runtime_attempts": [], "idempotency": {}, "action_approvals": {},
            }
            d.mkdir(parents=True, exist_ok=True)
            (d / "manifest.json").write_text(json.dumps(state))
            return d

        reg = build_executor_registry()
        for i, action in enumerate(HPC_ACTIONS):
            self.assertIn(action, APPROVAL_GATED_ACTIONS, action)
            with tempfile.TemporaryDirectory() as tmp:
                c = RunController(_run(Path(tmp) / "run"))
                role = reg[action].role
                prop = {"requested_by_role": role, "action_type": action,
                       "idempotency_key": f"k{i}", "run_id": "r", "stage": "s",
                       "requested_at": "t", "rationale": "sandbox", "parameters": {}}
                dry = dispatch_via_controller(prop, controller=c, registry=reg, mode="dry_run")
                self.assertNotEqual(dry.status, "EXECUTED", action)
                unapproved = dispatch_via_controller(prop, controller=c, registry=reg, mode="primary")
                self.assertNotEqual(unapproved.status, "EXECUTED", action)
                # acquire_structures fails closed on a missing AcquisitionPlan before the
                # approval check even runs (dispatch.py checks plan_sha256 first for this
                # action); every other action reaches the approval-boundary check directly.
                if action == "acquire_structures":
                    self.assertEqual(unapproved.status, "INVALID", action)
                else:
                    self.assertEqual(unapproved.status, "APPROVAL_REQUIRED", action)

    def test_costly_hpc_actions_are_approval_gated(self):
        # The costly, side-effecting HPC actions require an approval record before execution.
        from runtimes.pydantic_ai.actions import APPROVAL_GATED_ACTIONS
        for action in ("label_with_teacher", "train_committee", "build_teacher_physical_validation_target",
                       "run_teacher_md", "run_student_md"):
            self.assertIn(action, APPROVAL_GATED_ACTIONS, action)

    def test_hpc_backing_matrix_is_complete(self):
        from runtimes.pydantic_ai.executors import BINDINGS
        hpc = {a for a, b in BINDINGS.items() if b.status == "READY_HPC_APPROVAL_GATED"}
        self.assertEqual(hpc, set(HPC_ACTIONS))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
