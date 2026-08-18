"""R28 forensic-defect regression suite: a Controller-dispatched external executor (the
acquisition subprocess) that never returned could not be recorded as attempted (``attempts``
stayed 0) or judged (``record_gate`` required ``status == "completed"``), so the only durable
manifest state was pending/attempts=0 -- unrepresentable, requiring an operator to kill the
process from outside the Controller lifecycle to close the run (see
``runs/sio2-sox-allegro-simplenn-r28/artifacts/r28_workflow_failure_report.json``, preserved
verbatim and never modified by this suite).

These tests are deterministic and network-free: no multi-hour run is required to prove the fix.
Each test drives the REAL production code path (``workflow.controller.RunController``,
``runtimes.pydantic_ai.dispatch.authorize_and_execute``, ``runtimes.pydantic_ai.cli.
run_production_stage``/``run_campaign``, ``workflow.subprocess_runner.run_bounded``,
``adapters.acquisition.check_acquisition_feasibility``) with a monkeypatched executor or a real
short-lived subprocess standing in for the (potentially hours-long) external process -- never a
parallel/bypass reimplementation of the lifecycle under test.

Lettered tests A-H below correspond to the governing task's required deterministic coverage;
``R28RegressionTopologyTests`` at the bottom reproduces R28's own failure topology end-to-end and
asserts it now lands on GATE_FAIL, never pending/attempts=0.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import tempfile
import time
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


def _one_stage_workflow(root: Path, *, run_id: str) -> Path:
    dataset = root / "dataset.extxyz"
    _dataset(dataset)
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": run_id,
        "inputs": [str(dataset)],
        "stages": [{
            "name": "stage_a", "command": None, "outputs": ["artifacts/manifest.json"],
            "gate": {"criteria": ["dataset manifest is complete"]},
            "pydantic_ai": {
                "role": "data-curator", "action": "build_dataset_manifest",
                "idempotency_key": f"{run_id}:stage_a:001",
                "parameters": {"dataset": str(dataset),
                              "manifest_path": "{artifacts_dir}/manifest.json"},
            },
        }],
    }))
    return workflow


class _PatchedRegistry:
    """Context manager: swap in a registry whose ``build_dataset_manifest`` executor is
    ``executor``, restoring the real registry factory afterward -- same monkeypatch shape as
    ``tests/test_run_campaign_external_pending.py``, never a parallel dispatch implementation."""

    def __init__(self, executor):
        self._executor = executor

    def __enter__(self):
        import runtimes.pydantic_ai.executors as executors_mod
        self._mod = executors_mod
        self._original = executors_mod.build_executor_registry
        executor = self._executor

        def _patched():
            from runtimes.pydantic_ai.dispatch import ActionDescriptor
            reg = self._original()
            reg["build_dataset_manifest"] = ActionDescriptor(
                action_type="build_dataset_manifest", role="data-curator", executor=executor)
            return reg

        executors_mod.build_executor_registry = _patched
        return self

    def __exit__(self, *exc):
        self._mod.build_executor_registry = self._original


def _events(c, *, type_):
    return [e for e in c.state["events"] if e.get("type") == type_]


# --- Test A: attempt + stage_execution_started event recorded AT DISPATCH TIME, before the ------
# --- executor returns (the exact R28 defect: previously only recorded on return) -----------------

class AttemptRecordedAtDispatchTests(unittest.TestCase):
    def test_attempts_and_event_are_durable_before_executor_returns(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.deterministic_executors import build_dataset_manifest
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="dispatch-time-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            observed = {}

            def _observing_executor(proposal):
                # Read a FRESH controller view from disk: proves the attempt was durably
                # persisted before this executor call returns, not merely held in memory.
                snapshot = RunController(run_dir)
                observed["status"] = snapshot.stage("stage_a")["status"]
                observed["attempts"] = snapshot.stage("stage_a")["attempts"]
                observed["started_events"] = len(_events(snapshot, type_="stage_execution_started"))
                return build_dataset_manifest(proposal)

            with _PatchedRegistry(_observing_executor):
                result = cli.run_production_stage(c, "stage_a", runtime="mock",
                                                  repo_root=str(ROOT), auto_mock_judges=True)
                self.assertEqual(result.reason, "SUCCESS", result.message)

            self.assertEqual(observed["status"], "running")
            self.assertEqual(observed["attempts"], 1)
            self.assertEqual(observed["started_events"], 1)

            c = RunController(run_dir)
            started = _events(c, type_="stage_execution_started")
            self.assertEqual(len(started), 1)
            self.assertEqual(started[0]["attempt"], 1)
            self.assertEqual(started[0]["stage"], "stage_a")
            self.assertEqual(started[0]["executor"], "data-curator:build_dataset_manifest")
            self.assertIn("pid", started[0])
            self.assertIn("plan_sha256", started[0])


# --- Test B: normal completion ---------------------------------------------------------------

class NormalCompletionTests(unittest.TestCase):
    def test_normal_completion_records_one_attempt_and_passes_gate(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="normal-completion-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=5)
            self.assertEqual(result.outcome, cli.CAMPAIGN_COMPLETED, result.message)
            self.assertEqual(result.exit_code, cli.EXIT_SUCCESS)

            c = RunController(run_dir)
            stage = c.stage("stage_a")
            self.assertEqual(stage["status"], "completed")
            self.assertEqual(stage["gate"], "PASS")
            self.assertEqual(stage["attempts"], 1)
            self.assertEqual(len(_events(c, type_="stage_execution_started")), 1)
            self.assertIsNone(c.state.get("pending_recovery"))


# --- Test C: ordinary (non-timeout) executor exception defers to pending, attempts preserved --

class OrdinaryExecutorExceptionTests(unittest.TestCase):
    def test_ordinary_exception_defers_to_pending_without_losing_the_attempt(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.deterministic_executors import build_dataset_manifest
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="ordinary-exception-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            def _raising_executor(proposal):
                raise RuntimeError("simulated ordinary executor failure")

            with _PatchedRegistry(_raising_executor):
                result = cli.run_production_stage(c, "stage_a", runtime="mock",
                                                  repo_root=str(ROOT), auto_mock_judges=True)
            self.assertEqual(result.reason, "DISPATCH_REJECTED", result.message)
            self.assertEqual(result.exit_code, cli.EXIT_VALIDATION_REJECTED)

            c = RunController(run_dir)
            stage = c.stage("stage_a")
            # THE R28 DEFECT this proves fixed: pre-fix, this stage would show
            # status="pending", attempts=0 -- an unrepresentable "nothing happened" record
            # despite a real dispatched attempt. Post-fix: attempts is durably 1.
            self.assertEqual(stage["status"], "pending")
            self.assertEqual(stage["attempts"], 1)
            self.assertIsNone(stage.get("runner"))
            self.assertIsNone(c.state.get("pending_recovery"))
            self.assertEqual(len(_events(c, type_="stage_execution_started")), 1)
            deferred = _events(c, type_="stage_execution_deferred")
            self.assertEqual(len(deferred), 1)
            self.assertEqual(deferred[0]["attempt"], 1)

            # The SAME idempotency key must still be resumable (fast-path deferral is not a
            # terminal failure) -- fixing the underlying input and retrying works exactly as
            # the pre-existing ad-hoc run-stage retry contract requires.
            with _PatchedRegistry(build_dataset_manifest):
                result2 = cli.run_production_stage(c, "stage_a", runtime="mock",
                                                    repo_root=str(ROOT), auto_mock_judges=True)
            self.assertEqual(result2.reason, "SUCCESS", result2.message)
            c = RunController(run_dir)
            self.assertEqual(c.stage("stage_a")["status"], "completed")
            self.assertEqual(c.stage("stage_a")["attempts"], 2)


# --- Test D: wall-time budget actually terminates ONLY its own process group -------------------

class SubprocessRunnerBoundedTests(unittest.TestCase):
    def test_timeout_kills_only_the_owned_process_group_and_reports_timed_out(self):
        from workflow.subprocess_runner import run_bounded

        started_pid = {}
        result = run_bounded(
            ["sleep", "30"], cwd=None, env=os.environ.copy(), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout_s=0.3, grace_s=0.5, poll_interval_s=0.05,
            on_start=lambda pid: started_pid.setdefault("pid", pid),
        )
        self.assertTrue(result.timed_out)
        self.assertIsNotNone(result.pid)
        self.assertEqual(started_pid["pid"], result.pid)
        self.assertLess(result.elapsed_s, 5.0)
        # The process (and its process group leader) must actually be gone afterward.
        time.sleep(0.1)
        with self.assertRaises(ProcessLookupError):
            os.kill(result.pid, 0)

    def test_heartbeat_fires_while_running_and_stops_reporting_timeout_on_normal_exit(self):
        from workflow.subprocess_runner import run_bounded

        heartbeats = []
        result = run_bounded(
            ["sleep", "0.6"], cwd=None, env=os.environ.copy(), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout_s=5.0, heartbeat_interval_s=0.1,
            poll_interval_s=0.05, heartbeat_cb=lambda: heartbeats.append(time.monotonic()),
        )
        self.assertFalse(result.timed_out)
        self.assertEqual(result.returncode, 0)
        self.assertGreaterEqual(len(heartbeats), 2)


# --- Test E: record_gate accepts REVISE/FAIL against a timed_out/failed/cancelled stage, --------
# --- but PASS still strictly requires status == "completed" ------------------------------------

class TimeoutGateRoutingTests(unittest.TestCase):
    def test_timeout_stage_execution_then_fail_gate_is_representable(self):
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="timeout-gate-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            c.begin_stage_execution("stage_a", runner_id="test", executor="test:exec")
            self.assertEqual(c.stage("stage_a")["attempts"], 1)
            c.timeout_stage_execution("stage_a", elapsed_s=12.3, timeout_s=10.0)
            self.assertEqual(c.stage("stage_a")["status"], "timed_out")
            self.assertEqual(len(_events(c, type_="stage_execution_timed_out")), 1)

            c.record_gate("stage_a", "FAIL", evidence="wall-time budget exceeded")
            self.assertEqual(c.stage("stage_a")["gate"], "FAIL")
            self.assertIsNotNone(c.state.get("pending_recovery"))
            self.assertEqual(c.state["pending_recovery"]["failed_stage"], "stage_a")
            self.assertEqual(c.state["pending_recovery"]["artifact_sha256"], {})

    def test_pass_still_requires_completed_status(self):
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="pass-requires-completed-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            c.begin_stage_execution("stage_a", runner_id="test", executor="test:exec")
            c.timeout_stage_execution("stage_a")
            # record_gate rejects a votes-less PASS before it even reaches the completed-status
            # check (ValueError); either way, a timed_out stage can never be PASSed.
            with self.assertRaises((ValueError, RuntimeError)):
                c.record_gate("stage_a", "PASS")

    def test_revise_or_fail_against_a_never_started_pending_stage_is_rejected(self):
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="pending-gate-rejected-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            self.assertEqual(c.stage("stage_a")["status"], "pending")
            with self.assertRaises(RuntimeError):
                c.record_gate("stage_a", "FAIL", evidence="never dispatched")


# --- Test F: heartbeat/progress events ----------------------------------------------------------

class HeartbeatEventTests(unittest.TestCase):
    def test_executor_progress_calls_produce_durable_heartbeat_events(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.deterministic_executors import build_dataset_manifest
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="heartbeat-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            def _reporting_executor(proposal, progress_cb=None):
                if progress_cb is not None:
                    progress_cb({"pid": os.getpid(), "n": 1})
                    progress_cb({"pid": os.getpid(), "n": 2})
                return build_dataset_manifest(proposal)

            with _PatchedRegistry(_reporting_executor):
                result = cli.run_production_stage(c, "stage_a", runtime="mock",
                                                  repo_root=str(ROOT), auto_mock_judges=True)
            self.assertEqual(result.reason, "SUCCESS", result.message)

            c = RunController(run_dir)
            heartbeats = _events(c, type_="executor_heartbeat")
            self.assertGreaterEqual(len(heartbeats), 2)
            for hb in heartbeats:
                self.assertEqual(hb["stage"], "stage_a")
                self.assertEqual(hb["attempt"], 1)
                self.assertEqual(hb["pid"], os.getpid())
                self.assertIsInstance(hb["elapsed_s"], float)
                self.assertGreaterEqual(hb["elapsed_s"], 0.0)
                self.assertIn("n", hb["progress"])


# --- Test G: acquisition feasibility pre-check reproduces R28's own forensic formula ------------

class AcquisitionFeasibilityTests(unittest.TestCase):
    def test_r28_approved_parameters_are_flagged_infeasible(self):
        from adapters.acquisition import AcquisitionFeasibilityError, check_acquisition_feasibility

        # Exact parameters from runs/sio2-sox-allegro-simplenn-r28's approved acquisition plan
        # (see artifacts/r28_workflow_failure_report.json's algorithmic_pathology analysis).
        with self.assertRaises(AcquisitionFeasibilityError):
            check_acquisition_feasibility({
                "sigma_range": [0.005, 0.03], "similarity_threshold": 0.10,
                "max_relax_steps": 20,
            })

    def test_r27_operator_amended_parameters_pass(self):
        from adapters.acquisition import check_acquisition_feasibility

        reach = check_acquisition_feasibility({
            "sigma_range": [0.005, 0.03], "similarity_threshold": 0.02,
            "max_relax_steps": 20,
        })
        self.assertIsNotNone(reach)
        self.assertGreater(reach, 0.02 * 1.5)

    def test_missing_parameters_are_a_no_op(self):
        from adapters.acquisition import check_acquisition_feasibility

        self.assertIsNone(check_acquisition_feasibility({}))
        self.assertIsNone(check_acquisition_feasibility({"sigma_range": [0.01, 0.02]}))

    def test_run_augment_atoms_rejects_before_any_subprocess_is_dispatched(self):
        import adapters.acquisition as acquisition_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "native_config.yaml"
            config_path.write_text(yaml.safe_dump({
                "config": {"sigma_range": [0.005, 0.03], "similarity_threshold": 0.10,
                          "max_relax_steps": 20},
            }))
            seed_path = root / "seed.extxyz"
            _dataset(seed_path)
            out_path = root / "out.extxyz"

            def _run_bounded_must_not_be_called(*a, **kw):
                raise AssertionError("run_bounded must not be called: feasibility check must "
                                     "reject before any subprocess is dispatched")

            original = acquisition_mod.run_bounded
            acquisition_mod.run_bounded = _run_bounded_must_not_be_called
            try:
                with self.assertRaises(acquisition_mod.AcquisitionFeasibilityError):
                    acquisition_mod.run_augment_atoms(
                        {"config_path": str(config_path), "command": ["true"]},
                        seed_path, out_path)
            finally:
                acquisition_mod.run_bounded = original


# --- Test H: the pre-fix defect (pending/attempts=0 after ANY unresolved attempt) can no --------
# --- longer occur, regardless of whether the terminal shape is a deferral or a timeout ----------

class OldDefectCanNoLongerOccurTests(unittest.TestCase):
    def test_ordinary_failure_never_leaves_pending_with_zero_attempts(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="old-defect-ordinary-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            def _raising_executor(proposal):
                raise RuntimeError("no output produced, no progress reported")

            with _PatchedRegistry(_raising_executor):
                cli.run_production_stage(c, "stage_a", runtime="mock", repo_root=str(ROOT),
                                         auto_mock_judges=True)
            c = RunController(run_dir)
            stage = c.stage("stage_a")
            self.assertFalse(stage["status"] == "pending" and stage["attempts"] == 0)
            self.assertEqual(stage["attempts"], 1)

    def test_timeout_failure_never_leaves_pending_with_zero_attempts(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="old-defect-timeout-test")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            class _SimulatedHangTimeoutError(TimeoutError):
                pass

            def _timing_out_executor(proposal):
                raise _SimulatedHangTimeoutError(
                    "augment-atoms command timed out after 3600s (pid=99999)")

            with _PatchedRegistry(_timing_out_executor):
                result = cli.run_production_stage(c, "stage_a", runtime="mock",
                                                  repo_root=str(ROOT), auto_mock_judges=True)
            self.assertEqual(result.reason, "GATE_FAIL", result.message)

            c = RunController(run_dir)
            stage = c.stage("stage_a")
            self.assertFalse(stage["status"] == "pending" and stage["attempts"] == 0)
            self.assertEqual(stage["attempts"], 1)
            self.assertEqual(stage["status"], "timed_out")
            self.assertEqual(stage["gate"], "FAIL")


# --- R28 regression topology: reproduce R28's own failure shape end-to-end and prove it now -----
# --- lands on GATE_FAIL, never pending/attempts=0 ------------------------------------------------

class R28RegressionTopologyTests(unittest.TestCase):
    """Mirrors runs/sio2-sox-allegro-simplenn-r28's actual failure: an acquisition stage's
    trusted executor is genuinely invoked (past every enforcement check) and then never returns
    within its wall-time budget. R28 itself could only leave this at pending/attempts=0 with
    3 real dispatch attempts unaccounted for -- final_state=R28_CONTROLLER_WORKFLOW_NEEDS_FIX,
    completed_stages=2/12, campaign_outcome=INCOMPLETE_BLOCKED_AT_ACQUISITION (see
    runs/sio2-sox-allegro-simplenn-r28/manifest.json and campaign_events.jsonl, never modified by
    this test). This test raises the real ``adapters.acquisition.AcquisitionTimeoutError`` (the
    exception type ``run_augment_atoms`` now raises when ``workflow.subprocess_runner.run_bounded``
    reports ``timed_out``) from a stand-in executor -- exercising the exact generic
    TimeoutError-name detection in ``runtimes.pydantic_ai.cli.run_production_stage`` without
    requiring an actual multi-hour hang.
    """

    def test_r28_topology_now_lands_on_gate_fail_not_pending_attempts_zero(self):
        from adapters.acquisition import AcquisitionTimeoutError
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _one_stage_workflow(root, run_id="r28-topology-regression")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            def _hanging_acquisition_executor(proposal):
                raise AcquisitionTimeoutError(
                    "augment-atoms command timed out after 10800s (pid=424242)")

            with _PatchedRegistry(_hanging_acquisition_executor):
                result = cli.run_production_stage(c, "stage_a", runtime="mock",
                                                  repo_root=str(ROOT), auto_mock_judges=True)

            self.assertEqual(result.reason, "GATE_FAIL", result.message)
            self.assertEqual(result.gate_decision, "FAIL")
            self.assertEqual(result.exit_code, cli.EXIT_VALIDATION_REJECTED)

            c = RunController(run_dir)
            stage = c.stage("stage_a")
            # The R28 defect, made unrepresentable: never pending/attempts=0.
            self.assertNotEqual(stage["status"], "pending")
            self.assertGreaterEqual(stage["attempts"], 1)
            # The fixed, representable terminal shape instead:
            self.assertEqual(stage["status"], "timed_out")
            self.assertEqual(stage["attempts"], 1)
            self.assertEqual(stage["gate"], "FAIL")
            self.assertIsNotNone(c.state.get("pending_recovery"))
            self.assertEqual(c.state["pending_recovery"]["failed_stage"], "stage_a")

            started = _events(c, type_="stage_execution_started")
            timed_out = _events(c, type_="stage_execution_timed_out")
            gates = _events(c, type_="gate")
            self.assertEqual(len(started), 1)
            self.assertEqual(len(timed_out), 1)
            self.assertEqual(len(gates), 1)
            self.assertEqual(gates[0]["verdict"], "FAIL")

            # A materially unchanged resume attempt must not silently re-approve or auto-retry a
            # pathological plan: pending_recovery blocks run-campaign from proceeding without a
            # genuine Analyst diagnosis -- runtime="mock" deliberately refuses to fabricate one.
            with self.assertRaises(ValueError):
                cli.run_campaign(c, runtime="mock", repo_root=str(ROOT), auto_mock_judges=True,
                                 max_iterations=1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
