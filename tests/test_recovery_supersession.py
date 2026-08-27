from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflow.controller import RunController
from workflow.integrity import sha256_file


def _state():
    return {
        "schema_version": 6,
        "run_id": "supersession-test",
        "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00",
        "workflow_config": "w",
        "artifacts": [],
        "project_dir": ".",
        "inputs": [],
        "code_revision": {"available": False},
        "events": [],
        "stages": [
            {
                "name": "training",
                "status": "completed",
                "gate": "FAIL",
                "artifacts": [],
                "outputs": ["artifacts/result.txt"],
                "attempts": 1,
                "started_at": "2026-08-07T00:00:00+00:00",
                "completed_at": "2026-08-07T00:01:00+00:00",
                "gate_criteria": ["criterion"],
                "gate_review_lenses": [],
                "contract": None,
            },
            {
                "name": "evaluation",
                "status": "pending",
                "gate": "pending",
                "artifacts": [],
                "outputs": [],
                "attempts": 0,
                "started_at": None,
                "gate_criteria": [],
                "gate_review_lenses": [],
                "contract": None,
            },
        ],
        "iterations": [
            {
                "id": 1,
                "parent_iteration": None,
                "status": "active",
                "started_at": "2026-08-07T00:00:00+00:00",
                "trigger": None,
            }
        ],
        "recoveries": [],
        "pending_recovery": None,
    }


def _write_run(root: Path) -> Path:
    run = root / "run"
    run.mkdir()
    (run / "inputs").mkdir()
    (run / "manifest.json").write_text(json.dumps(_state(), indent=2) + "\n")
    return run


def _plan(path: Path) -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "proposed_by": {"actor_kind": "system", "canonical_id": "orchestrator"},
        "failed_stage": "training",
        "failure_category": "lineage_or_leakage",
        "failure_domain": "data_coverage",
        "root_cause": "old training population lineage trigger",
        "responsible_capability": "data_repair",
        "return_stage": "training",
        "proposed_changes": [{"type": "lineage_evidence_gathering"}],
        "labeling": {"teacher_relabel": False, "new_dft": False},
        "student_training": {"retrain": False, "mode": "none"},
        "revalidation": {"reuse_profile": True, "targets": ["training lineage"]},
        "estimated_cost": {},
    }, indent=2) + "\n")
    return path


class RecoverySupersessionTests(unittest.TestCase):
    def _controller_with_proposed_recovery(self, root: Path):
        run = _write_run(root)
        c = RunController(run)
        c.state["pending_recovery"] = {
            "status": "required",
            "failed_stage": "training",
            "verdict": "FAIL",
            "gate_recorded_at": "2026-08-07T00:02:00+00:00",
            "artifact_sha256": {},
        }
        c.save()
        recovery = c.propose_recovery(_plan(root / "plan.json"))
        evidence = root / "evidence.json"
        evidence.write_text(json.dumps({"obsolete": True}) + "\n")
        supersession = {
            "reason_code": "triggering_condition_resolved_by_superseding_framework_fix",
            "rationale": "later framework evidence resolved the old trigger",
            "superseding_code_revision": {"available": True, "git_commit": "abc"},
            "evidence": [{"path": str(evidence), "sha256": sha256_file(evidence), "role": "obsolete-trigger-evidence"}],
        }
        return c, recovery, supersession

    def test_proposed_recovery_can_be_superseded_with_durable_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, recovery, supersession = self._controller_with_proposed_recovery(Path(tmp))
            out = c.supersede_recovery(recovery["id"], supersession)
            self.assertEqual(out["status"], "superseded")
            self.assertIsNone(c.state["pending_recovery"])
            self.assertEqual(out["human_approval"], None)
            self.assertNotIn("execution", out)
            self.assertEqual(out["supersession"]["previous_status"], "proposed")
            self.assertEqual(out["supersession"]["new_status"], "superseded")
            self.assertTrue(any(e["type"] == "recovery_superseded" for e in c.state["events"]))
            reloaded = RunController(c.run_dir)
            self.assertEqual(reloaded.state["recoveries"][0]["status"], "superseded")

    def test_supersession_requires_structured_reason_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, recovery, supersession = self._controller_with_proposed_recovery(Path(tmp))
            bad = dict(supersession)
            bad.pop("reason_code")
            with self.assertRaisesRegex(ValueError, "reason_code"):
                c.supersede_recovery(recovery["id"], bad)
            bad = dict(supersession)
            bad["evidence"] = []
            with self.assertRaisesRegex(ValueError, "evidence"):
                c.supersede_recovery(recovery["id"], bad)

    def test_wrong_or_non_proposed_recovery_cannot_be_superseded(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, recovery, supersession = self._controller_with_proposed_recovery(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "current proposed pending recovery"):
                c.supersede_recovery(recovery["id"] + 1, supersession)
            c.approve_recovery("researcher")
            with self.assertRaisesRegex(RuntimeError, "current proposed pending recovery"):
                c.supersede_recovery(recovery["id"], supersession)

    def test_repeated_supersession_is_rejected_and_history_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, recovery, supersession = self._controller_with_proposed_recovery(Path(tmp))
            original_plan = json.dumps(recovery["plan"], sort_keys=True)
            c.supersede_recovery(recovery["id"], supersession)
            with self.assertRaisesRegex(RuntimeError, "current proposed pending recovery"):
                c.supersede_recovery(recovery["id"], supersession)
            self.assertEqual(json.dumps(c.state["recoveries"][0]["plan"], sort_keys=True), original_plan)

    def test_superseded_recovery_no_longer_blocks_rebind_but_proposed_and_approved_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, recovery, supersession = self._controller_with_proposed_recovery(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "recovery is pending"):
                c.rebind_inputs()
            c.supersede_recovery(recovery["id"], supersession)
            c.rebind_inputs()

        with tempfile.TemporaryDirectory() as tmp:
            c, _recovery, _supersession = self._controller_with_proposed_recovery(Path(tmp))
            c.approve_recovery("researcher")
            with self.assertRaisesRegex(RuntimeError, "recovery is pending"):
                c.rebind_inputs()


if __name__ == "__main__":
    unittest.main()
