"""Phase 6: producer trusted-executor integration + honest backing matrix.

Proves AVAILABLE actions reach their EXISTING scientific implementation through the SAME
registry/controller/validator path (sandbox-primary, synthetic zero-cost input), and that
NOT_IMPLEMENTED / AVAILABLE_HPC actions are never faked as executed. Network-free; skips
without the ``pydantic`` extra.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


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


def _prop(role, action, key, parameters):
    return {"requested_by_role": role, "action_type": action, "idempotency_key": key,
            "run_id": "r", "stage": "s", "requested_at": "t", "rationale": "sandbox",
            "parameters": parameters}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ExecutorIntegrationTests(unittest.TestCase):
    def _dispatch(self, tmp, prop, mode="primary"):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        c = RunController(_run(Path(tmp) / "run"))
        return c, dispatch_via_controller(prop, controller=c, registry=build_executor_registry(),
                                          mode=mode)

    def test_compute_nve_drift_reaches_real_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, o = self._dispatch(tmp, _prop("simulation", "compute_nve_drift", "n1",
                                             {"energies": [0.0, 0.1, 0.2], "timestep_fs": 1.0,
                                              "n_atoms": 1}))
            self.assertEqual(o.status, "EXECUTED")
            self.assertIn("nve_drift", o.artifact["metrics"])

    def test_committee_disagreement_reaches_real_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            forces = [[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]]  # 2 seeds, 1 atom
            _, o = self._dispatch(tmp, _prop("ml-trainer", "compute_committee_disagreement", "c1",
                                             {"forces_per_seed": forces, "aggregate": "mean"}))
            self.assertEqual(o.status, "EXECUTED")
            self.assertIn("u_frame", o.artifact["metrics"])

    def test_compute_rdf_reaches_real_executor_with_synthetic_frames(self):
        from ase import Atoms
        from ase.io import write
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "cu.extxyz"
            write(str(frames), Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]],
                                     cell=[20, 20, 20], pbc=True))
            _, o = self._dispatch(tmp, _prop("simulation", "compute_rdf", "r1",
                                             {"frames_path": str(frames), "elements": ["Cu"],
                                              "r_max": 6.0, "nbins": 50}))
            self.assertEqual(o.status, "EXECUTED")
            self.assertIn("Cu-Cu", o.artifact["metrics"]["rdf_peaks"])

    def test_not_implemented_action_is_never_faked_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, o = self._dispatch(tmp, _prop("data-curator", "sample_seed_pool", "s1", {}))
            self.assertEqual(o.status, "DRY_RUN")   # no executor -> never EXECUTED, never mocked
            self.assertFalse(o.executed)

    def test_hpc_action_is_gated_and_not_executed(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        with tempfile.TemporaryDirectory() as tmp:
            c = RunController(_run(Path(tmp) / "run"))
            reg = build_executor_registry()
            p = _prop("ml-trainer", "train_committee", "t1", {})
            o1 = dispatch_via_controller(p, controller=c, registry=reg, mode="primary")
            self.assertEqual(o1.status, "APPROVAL_REQUIRED")     # gated
            c.grant_action_approval("costly_training")
            o2 = dispatch_via_controller(p, controller=c, registry=reg, mode="primary")
            self.assertEqual(o2.status, "DRY_RUN")               # approved but NOT run in-process
            self.assertFalse(o2.executed)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class BackingMatrixTests(unittest.TestCase):
    def test_every_role_action_has_a_binding(self):
        from runtimes.pydantic_ai.actions import ROLE_ALLOWED_ACTIONS
        from runtimes.pydantic_ai.executors import BINDINGS
        all_actions = set().union(*ROLE_ALLOWED_ACTIONS.values())
        self.assertEqual(set(BINDINGS), all_actions)

    def test_available_have_executor_others_do_not(self):
        from runtimes.pydantic_ai.executors import BINDINGS, build_executor_registry
        reg = build_executor_registry()
        for action, b in BINDINGS.items():
            if b.status == "AVAILABLE":
                self.assertIsNotNone(b.fn, action)
                self.assertIsNotNone(reg[action].executor, action)
            else:  # AVAILABLE_HPC / NOT_IMPLEMENTED never carry an inline executor
                self.assertIsNone(b.fn, action)
                self.assertIsNone(reg[action].executor, action)

    def test_status_counts_are_honest(self):
        from collections import Counter
        from runtimes.pydantic_ai.executors import BINDINGS
        counts = Counter(b.status for b in BINDINGS.values())
        self.assertEqual(counts["AVAILABLE"], 12)
        self.assertEqual(counts["AVAILABLE_HPC"], 5)
        self.assertEqual(counts["NOT_IMPLEMENTED"], 20)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
