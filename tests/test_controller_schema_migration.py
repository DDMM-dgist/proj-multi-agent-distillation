"""Phase 4-5: controller schema v6->v7 additive migration + stale-running + idempotency.

Core (no pydantic). Proves: v6 read unchanged, additive v7 fields, copy-only migration with
source preserved + failure rollback, stale-running reconcile (operational, not scientific
recovery), and v7 round-trip.
"""
from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from workflow.controller import RunController, SCHEMA_VERSION
from workflow.manifest_migration import migrate_run_manifest

UTC = dt.timezone.utc


def _v6_state(stages=None):
    stages = stages or [{"name": "teacher_baseline", "status": "pending", "gate": "pending",
                         "artifacts": []}]
    return {
        "schema_version": 6, "run_id": "r", "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
        "project_dir": "p", "inputs": [], "code_revision": "x", "events": [], "stages": stages,
        "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                        "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
        "recoveries": [], "pending_recovery": None,
    }


def _write_run(d: Path, state) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state, indent=2))
    return d


class SchemaVersionTests(unittest.TestCase):
    def test_new_runs_target_is_seven(self):
        self.assertEqual(SCHEMA_VERSION, 7)

    def test_v6_manifest_loads_and_stays_v6_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run(Path(tmp) / "run", _v6_state())
            c = RunController(run)
            self.assertEqual(c.state["schema_version"], 6)      # not bumped in place
            self.assertFalse(c.action_seen("nope"))             # defaults when field absent
            on_disk = json.loads((run / "manifest.json").read_text())
            self.assertEqual(on_disk["schema_version"], 6)
            self.assertNotIn("runtime_attempts", on_disk)       # untouched by mere load

    def test_v7_features_are_additive_and_keep_v6_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run(Path(tmp) / "run", _v6_state())
            c = RunController(run)
            c.record_runtime_attempt(task_id="t1", attempt_id="a1", provenance_path="p.json",
                                     role="judge", stage="teacher_baseline")
            c.record_action("k1", action_type="inspect_dataset", status="EXECUTED")
            reloaded = json.loads((run / "manifest.json").read_text())
            self.assertEqual(reloaded["schema_version"], 6)     # additive, version NOT bumped
            self.assertEqual(len(reloaded["runtime_attempts"]), 1)
            self.assertIn("k1", reloaded["idempotency"])
            self.assertTrue(RunController(run).action_seen("k1"))


class StaleRunningTests(unittest.TestCase):
    def _controller(self, tmp, status="running", last_update=None, pid=None):
        stage = {"name": "student_training", "status": status, "gate": "pending", "artifacts": []}
        if last_update is not None or pid is not None:
            stage["runner"] = {"pid": pid, "runner_id": "job-1",
                               "started_at": "2026-08-07T00:00:00+00:00",
                               "last_update": last_update or "2026-08-07T00:00:00+00:00"}
        run = _write_run(Path(tmp) / "run", _v6_state(stages=[stage]))
        return RunController(run)

    def test_old_heartbeat_is_reconciled_to_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(tmp, last_update="2026-08-07T00:00:00+00:00")
            reconciled = c.reconcile_stale_stages(
                threshold_s=60, current_time=dt.datetime(2026, 8, 7, 1, 0, 0, tzinfo=UTC))
            self.assertEqual(reconciled, ["student_training"])
            self.assertEqual(c.stage("student_training")["status"], "interrupted")
            self.assertTrue(any(e["type"] == "stale_running_recovered" for e in c.state["events"]))

    def test_recent_heartbeat_is_not_reconciled(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(tmp, last_update="2026-08-07T00:59:30+00:00")
            reconciled = c.reconcile_stale_stages(
                threshold_s=60, current_time=dt.datetime(2026, 8, 7, 1, 0, 0, tzinfo=UTC))
            self.assertEqual(reconciled, [])
            self.assertEqual(c.stage("student_training")["status"], "running")

    def test_dead_pid_is_reconciled_even_if_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(tmp, last_update="2026-08-07T00:59:59+00:00", pid=4242)
            reconciled = c.reconcile_stale_stages(
                threshold_s=600, current_time=dt.datetime(2026, 8, 7, 1, 0, 0, tzinfo=UTC),
                is_pid_alive=lambda pid: False)
            self.assertEqual(reconciled, ["student_training"])

    def test_stage_without_runner_metadata_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(tmp)  # no runner
            reconciled = c.reconcile_stale_stages(
                threshold_s=1, current_time=dt.datetime(2027, 1, 1, tzinfo=UTC))
            self.assertEqual(reconciled, [])
            self.assertEqual(c.stage("student_training")["status"], "running")

    def test_begin_and_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run(Path(tmp) / "run", _v6_state(
                stages=[{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}]))
            c = RunController(run)
            c.begin_stage_execution("s", pid=123, runner_id="job-9")
            self.assertEqual(c.stage("s")["status"], "running")
            self.assertEqual(c.stage("s")["runner"]["pid"], 123)
            c.heartbeat_stage("s")
            self.assertIn("last_update", c.stage("s")["runner"])


class MigrationTests(unittest.TestCase):
    def test_migrate_copy_becomes_v7_and_source_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_run(Path(tmp) / "src", _v6_state())
            dst = Path(tmp) / "dst"
            manifest = migrate_run_manifest(src, dst)
            migrated = json.loads(manifest.read_text())
            self.assertEqual(migrated["schema_version"], 7)
            self.assertIn("runtime_attempts", migrated)
            self.assertIn("idempotency", migrated)
            self.assertTrue(any(e["type"] == "schema_migrated" for e in migrated["events"]))
            # source is untouched
            original = json.loads((src / "manifest.json").read_text())
            self.assertEqual(original["schema_version"], 6)
            self.assertNotIn("runtime_attempts", original)

    def test_migrate_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_run(Path(tmp) / "src", _v6_state())
            dst = _write_run(Path(tmp) / "dst", _v6_state())
            with self.assertRaises(FileExistsError):
                migrate_run_manifest(src, dst)

    def test_migration_failure_removes_dst_and_preserves_src(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = _v6_state()
            del bad["schema_version"]  # invalid: no schema_version
            src = _write_run(Path(tmp) / "src", bad)
            dst = Path(tmp) / "dst"
            with self.assertRaises(ValueError):
                migrate_run_manifest(src, dst)
            self.assertFalse(dst.exists())                    # partial copy removed
            self.assertTrue((src / "manifest.json").exists())  # source preserved

    def test_v7_round_trip_after_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_run(Path(tmp) / "src", _v6_state())
            dst = Path(tmp) / "dst"
            migrate_run_manifest(src, dst)
            c = RunController(dst)
            self.assertEqual(c.state["schema_version"], 7)
            c.record_runtime_attempt(task_id="t", attempt_id="a", provenance_path="p")
            again = RunController(dst)
            self.assertEqual(again.state["schema_version"], 7)  # stays v7 across save/load
            self.assertEqual(len(again.state["runtime_attempts"]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
