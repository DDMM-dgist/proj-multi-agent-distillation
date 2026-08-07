"""Phase 6/1: typed scheduler interface + controller pending->collect->resume lifecycle.

No real HPC backend; the sandbox adapter uses the SAME typed contracts + controller path.
Network-free; skips without the ``pydantic`` extra.
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
        "runtime_attempts": [], "idempotency": {}, "action_approvals": {}, "scheduler_jobs": {},
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state))
    return d


def _proposal(key="job1"):
    from runtimes.pydantic_ai.scheduler import SchedulerSubmissionProposal
    return SchedulerSubmissionProposal(
        run_id="r", stage="s", protocol_ref="md.yaml", protocol_hash="ph",
        config_ref="student.yaml", config_hash="ch", idempotency_key=key)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class SchedulerInterfaceTests(unittest.TestCase):
    def _adapter(self, tmp):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import ControllerApprovalStore
        from runtimes.pydantic_ai.scheduler import SandboxSchedulerAdapter
        c = RunController(_run(Path(tmp) / "run"))
        return c, SandboxSchedulerAdapter(c, ControllerApprovalStore(c))

    def test_submit_denied_without_approval(self):
        from runtimes.pydantic_ai.scheduler import SchedulerError
        with tempfile.TemporaryDirectory() as tmp:
            _, adapter = self._adapter(tmp)
            with self.assertRaises(SchedulerError) as cm:
                adapter.submit(_proposal())
            self.assertIn("APPROVAL_REQUIRED", str(cm.exception))

    def test_submit_records_pending_and_query_binds_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, adapter = self._adapter(tmp)
            c.grant_action_approval("scheduler_submission")
            identity = adapter.submit(_proposal())
            self.assertEqual(identity.backend, "sandbox")
            job = c.get_scheduler_job(identity.external_job_id)
            self.assertEqual(job["state"], "PENDING")               # controller records pending
            status = adapter.query("r", identity.external_job_id)
            self.assertEqual(status.external_job_id, identity.external_job_id)
            self.assertIn(status.state, ("PENDING", "RUNNING"))

    def test_duplicate_submit_rejected(self):
        from runtimes.pydantic_ai.scheduler import SchedulerError
        with tempfile.TemporaryDirectory() as tmp:
            c, adapter = self._adapter(tmp)
            c.grant_action_approval("scheduler_submission")
            adapter.submit(_proposal(key="dup"))
            with self.assertRaises(SchedulerError) as cm:
                adapter.submit(_proposal(key="dup"))
            self.assertIn("DUPLICATE", str(cm.exception))

    def test_wrong_run_or_job_identity_rejected(self):
        from runtimes.pydantic_ai.scheduler import SchedulerError, SchedulerCollectionRequest
        with tempfile.TemporaryDirectory() as tmp:
            c, adapter = self._adapter(tmp)
            c.grant_action_approval("scheduler_submission")
            identity = adapter.submit(_proposal())
            with self.assertRaises(SchedulerError):
                adapter.query("WRONG-RUN", identity.external_job_id)
            with self.assertRaises(SchedulerError):
                adapter.collect(SchedulerCollectionRequest(run_id="r", external_job_id="nope"))

    def test_cannot_collect_before_completion(self):
        from runtimes.pydantic_ai.scheduler import SchedulerError, SchedulerCollectionRequest
        with tempfile.TemporaryDirectory() as tmp:
            c, adapter = self._adapter(tmp)
            c.grant_action_approval("scheduler_submission")
            identity = adapter.submit(_proposal())
            with self.assertRaises(SchedulerError) as cm:
                adapter.collect(SchedulerCollectionRequest(run_id="r",
                                                           external_job_id=identity.external_job_id))
            self.assertIn("not complete", str(cm.exception))

    def test_collect_after_completion_binds_artifact_and_enables_resume(self):
        from runtimes.pydantic_ai.scheduler import SchedulerCollectionRequest
        with tempfile.TemporaryDirectory() as tmp:
            c, adapter = self._adapter(tmp)
            c.grant_action_approval("scheduler_submission")
            identity = adapter.submit(_proposal())
            adapter.simulate_external_completion(identity.external_job_id,
                                                 "runs/r/md/traj.extxyz", "deadbeef")
            collected = adapter.collect(SchedulerCollectionRequest(
                run_id="r", external_job_id=identity.external_job_id))
            self.assertEqual(collected["state"], "COLLECTED")
            self.assertEqual(collected["artifact_sha256"], "deadbeef")
            # resume is only valid once COLLECTED
            job = c.get_scheduler_job(identity.external_job_id)
            self.assertEqual(job["state"], "COLLECTED")

    def test_no_fake_execution_without_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, adapter = self._adapter(tmp)
            c.grant_action_approval("scheduler_submission")
            identity = adapter.submit(_proposal())
            status = adapter.query("r", identity.external_job_id)
            # never EXECUTED/COMPLETED without a real backend + observed completion
            self.assertNotIn(status.state, ("EXECUTED", "COMPLETED"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
