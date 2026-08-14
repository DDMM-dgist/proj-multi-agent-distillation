"""Priority #3 requirement #11 (implicit): a historical run manifest that predates the v9
recovery-taxonomy/capability-roster/protected-reference/loop-safety fields must still complete
the full recovery lifecycle unchanged -- every new field is read with `.get(..., default)`, never
assumed present, so an old on-disk manifest degrades gracefully to "no policy enforced, no
protected roles, default capability roster, legacy responsible_agent routing" rather than
crashing or silently changing behavior.

This mirrors tests/test_controller_schema_migration.py's v6-manifest-loads-unchanged pattern, but
drives it through propose_recovery/approve_recovery/start_iteration end-to-end rather than just
checking field presence.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflow.controller import RunController

_GATE_CRITERION = "artifact is complete and internally consistent"


def _v6_state(run_id="legacy-run"):
    return {
        "schema_version": 6, "run_id": run_id, "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
        "project_dir": "p", "inputs": [], "code_revision": {"available": False}, "events": [],
        "stages": [{"name": "validation", "status": "pending", "gate": "pending",
                    "artifacts": [], "outputs": ["artifacts/result.txt"], "attempts": 0,
                    "started_at": None, "gate_criteria": [_GATE_CRITERION],
                    "gate_review_lenses": [], "contract": None}],
        "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                        "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
        "recoveries": [], "pending_recovery": None,
    }


def _write_run(d: Path, state) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state, indent=2))
    return d


class LegacyManifestRecoveryLifecycleTests(unittest.TestCase):
    def test_v6_manifest_has_none_of_the_v9_recovery_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run(Path(tmp) / "run", _v6_state())
            controller = RunController(run)
            for key in ("recovery_capability_roster", "recovery_policy",
                        "protected_reference_roles"):
                self.assertNotIn(key, controller.state)

    def test_legacy_manifest_completes_the_full_recovery_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run(Path(tmp) / "run", _v6_state())
            controller = RunController(run)
            result = run / "artifacts/result.txt"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text("legacy result")
            controller.complete_external_stage("validation", [result])
            controller.record_gate("validation", "REVISE")
            self.assertEqual(controller.state["pending_recovery"]["status"], "required")

            plan = run.parent / "plan.json"
            plan.write_text(json.dumps({
                "schema_version": 1, "proposed_by": "automation", "failed_stage": "validation",
                "failure_category": "dataset_coverage", "root_cause": "legacy diagnosis",
                "responsible_agent": "data-curator", "return_stage": "validation",
                "proposed_changes": [{"type": "add_deployment_frames"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["validation"]},
                "estimated_cost": {},
            }))
            recovery = controller.propose_recovery(plan)
            self.assertEqual(recovery["status"], "proposed")
            self.assertEqual(recovery["resolved_responsible_agent"], "data-curator")
            self.assertIsNone(recovery["resolved_responsible_capability"])

            controller.approve_recovery("researcher", "approved on legacy manifest")
            controller.start_iteration()
            self.assertIsNone(controller.state["pending_recovery"])
            self.assertEqual(controller.state["recoveries"][-1]["status"], "activated")
            # schema_version stays whatever it was on disk -- migration is additive, not in-place.
            self.assertEqual(controller.state["schema_version"], 6)


if __name__ == "__main__":
    unittest.main()
