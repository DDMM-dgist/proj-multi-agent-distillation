"""R17 forensic-audit regression (Part C, Path 3): a FORWARD stage's own primary action can raise
``dispatch.ExternalActionPending`` (e.g. a scheduler-bridge action still queued externally) just as
readily as an approved recovery's corrective action can -- but before this fix,
``run_production_stage`` collapsed that into the generic ``status not in {"EXECUTED","DUPLICATE"}``
branch, producing a terminal ``DISPATCH_REJECTED``/``CAMPAIGN_FAILED`` outcome instead of the
resumable pause ``dispatch.py`` itself documents ("not a failure ... the SAME idempotency key can
be dispatched again later to re-check"). This mirrors the exact resumable-pause treatment already
proven for the recovery corrective-action path in ``tests/test_run_campaign_recovery.py``'s
``test_corrective_action_pending_pauses_campaign_without_completing_stage``, just for a stage's own
forward dispatch.

Network-free (mock runtime only); drives the real ``run_production_stage``/``run_campaign``
production path with a monkeypatched executor -- no parallel/bypass implementation.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


def _dataset(path: Path) -> Path:
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    write(str(path), [atoms])
    return path


def _one_stage_workflow(root: Path) -> Path:
    dataset = root / "dataset.extxyz"
    _dataset(dataset)
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "external-pending-test",
        "inputs": [str(dataset)],
        "stages": [{
            "name": "stage_a", "command": None, "outputs": ["artifacts/manifest.json"],
            "gate": {"criteria": ["dataset manifest is complete"]},
            "pydantic_ai": {
                "role": "data-curator", "action": "build_dataset_manifest",
                "idempotency_key": "external-pending-test:stage_a:001",
                "parameters": {"dataset": str(dataset),
                              "manifest_path": "{artifacts_dir}/manifest.json"},
            },
        }],
    }))
    return workflow


class _FlakyOnce:
    """Raises ExternalActionPending exactly once, then delegates to the real executor -- proving
    a resumed dispatch genuinely re-checks rather than fabricating success."""

    def __init__(self, real_executor):
        self._real = real_executor
        self.calls = 0
        self.__name__ = "flaky_once_" + real_executor.__name__

    def __call__(self, proposal):
        from runtimes.pydantic_ai.dispatch import ExternalActionPending
        self.calls += 1
        if self.calls == 1:
            raise ExternalActionPending("scheduler job still queued")
        return self._real(proposal)


class ForwardStageExternalActionPendingTests(unittest.TestCase):
    def _patched_registry(self, flaky_holder):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        from runtimes.pydantic_ai.executors import build_executor_registry

        real_registry = build_executor_registry()
        flaky = _FlakyOnce(real_registry["build_dataset_manifest"].executor)
        flaky_holder.append(flaky)

        def _patched():
            reg = build_executor_registry()
            reg["build_dataset_manifest"] = ActionDescriptor(
                action_type="build_dataset_manifest", role="data-curator", executor=flaky)
            return reg
        return _patched

    def test_pending_forward_action_pauses_campaign_without_failing_or_completing_stage(self):
        import runtimes.pydantic_ai.executors as executors_mod
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        original = executors_mod.build_executor_registry
        flaky_holder = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                workflow = _one_stage_workflow(root)
                run_dir = root / "run"
                RunController.initialize(workflow, run_dir)
                c = RunController(run_dir)

                executors_mod.build_executor_registry = self._patched_registry(flaky_holder)

                result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                          auto_mock_judges=True, max_iterations=5)
                self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_EXTERNAL_ACTION,
                                 result.message)
                self.assertEqual(result.exit_code, cli.EXIT_EXTERNAL_ACTION_PENDING)

                c = RunController(run_dir)
                self.assertEqual(c.stage("stage_a")["status"], "pending")
                self.assertIsNone(c.state.get("pending_recovery"))

                # The idempotency key must NOT have been consumed by the pending attempt: the
                # SAME key resumes and completes cleanly, exactly once for real.
                result2 = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                           auto_mock_judges=True, max_iterations=5)
                self.assertEqual(result2.outcome, cli.CAMPAIGN_COMPLETED, result2.message)
                self.assertEqual(result2.exit_code, cli.EXIT_SUCCESS)

                c = RunController(run_dir)
                self.assertEqual(c.stage("stage_a")["status"], "completed")
                self.assertEqual(c.stage("stage_a")["gate"], "PASS")
                self.assertEqual(flaky_holder[0].calls, 2)
                completed_events = [e for e in c.state["events"]
                                   if e.get("type") == "external_stage_completed" and
                                   e.get("stage") == "stage_a"]
                self.assertEqual(len(completed_events), 1)
        finally:
            executors_mod.build_executor_registry = original

    def test_still_pending_on_resume_pauses_again_without_state_corruption(self):
        import runtimes.pydantic_ai.executors as executors_mod
        from runtimes.pydantic_ai.dispatch import ActionDescriptor, ExternalActionPending
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        original = executors_mod.build_executor_registry

        def _always_pending_executor(_proposal):
            raise ExternalActionPending("scheduler job still queued")

        def _patched():
            reg = original()
            reg["build_dataset_manifest"] = ActionDescriptor(
                action_type="build_dataset_manifest", role="data-curator",
                executor=_always_pending_executor)
            return reg

        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                workflow = _one_stage_workflow(root)
                run_dir = root / "run"
                RunController.initialize(workflow, run_dir)
                c = RunController(run_dir)
                executors_mod.build_executor_registry = _patched

                for _ in range(3):
                    result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                              auto_mock_judges=True, max_iterations=5)
                    self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_EXTERNAL_ACTION,
                                     result.message)
                    c = RunController(run_dir)
                    self.assertEqual(c.stage("stage_a")["status"], "pending")
                    self.assertIsNone(c.state.get("pending_recovery"))
        finally:
            executors_mod.build_executor_registry = original


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
