"""Phase 5: trusted executor / controller bridge — controller-backed approval + idempotency.

Network-free; skips without the ``pydantic`` extra (dispatch needs it).
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
        "stages": [{"name": "data_curation", "status": "pending", "gate": "pending", "artifacts": []}],
        "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                        "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
        "recoveries": [], "pending_recovery": None,
        "runtime_attempts": [], "idempotency": {}, "action_approvals": {},
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state))
    return d


def _prop(action="label_with_teacher", role="data-curator", key="lab1"):
    return {"requested_by_role": role, "action_type": action, "idempotency_key": key,
            "run_id": "r", "stage": "data_curation", "requested_at": "t", "rationale": "cover gap"}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ControllerBridgeTests(unittest.TestCase):
    def _setup(self, tmp):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.dispatch import default_registry, ActionDescriptor
        c = RunController(_run(Path(tmp) / "run"))
        reg = default_registry()
        reg["label_with_teacher"] = ActionDescriptor(
            action_type="label_with_teacher", role="data-curator",
            approval_boundary="costly_teacher_labeling",
            executor=lambda p: {"path": "runs/r/labeled.xyz", "sha256": "hh"})
        return c, reg

    def test_costly_action_blocked_without_controller_approval(self):
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        with tempfile.TemporaryDirectory() as tmp:
            c, reg = self._setup(tmp)
            o = dispatch_via_controller(_prop(), controller=c, registry=reg, mode="primary")
            self.assertEqual(o.status, "APPROVAL_REQUIRED")

    def test_granted_approval_allows_execution_and_persists_idempotency(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        with tempfile.TemporaryDirectory() as tmp:
            c, reg = self._setup(tmp)
            c.grant_action_approval("costly_teacher_labeling", note="human ok")
            o = dispatch_via_controller(_prop(), controller=c, registry=reg, mode="primary")
            self.assertEqual(o.status, "EXECUTED")
            self.assertEqual(o.artifact["sha256"], "hh")
            # idempotency persisted to the controller manifest
            self.assertTrue(c.action_seen("lab1"))
            reloaded = RunController(c.run_dir)
            self.assertTrue(reloaded.action_seen("lab1"))
            # a second dispatch with the same key does NOT re-execute
            o2 = dispatch_via_controller(_prop(), controller=reloaded, registry=reg, mode="primary")
            self.assertEqual(o2.status, "DUPLICATE")

    def test_wrong_role_still_denied_through_bridge(self):
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        with tempfile.TemporaryDirectory() as tmp:
            c, reg = self._setup(tmp)
            o = dispatch_via_controller(_prop(action="train_committee", role="judge"),
                                        controller=c, registry=reg, mode="primary")
            self.assertEqual(o.status, "DENIED")

    def test_runtime_attempt_reference_recorded(self):
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            c = RunController(_run(Path(tmp) / "run"))
            c.record_runtime_attempt(task_id="t1", attempt_id="a1",
                                     provenance_path="exchange/provenance/t1.a1.json",
                                     role="judge", stage="data_curation", correlation_id="c1")
            self.assertEqual(RunController(c.run_dir).state["runtime_attempts"][0]["attempt_id"], "a1")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
