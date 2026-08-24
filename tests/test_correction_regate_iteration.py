"""RunController.open_correction_iteration: an audited re-gate iteration that runs NO recovery.

This is the mechanism that lets a stage which already recorded a (defective) gate be RE-GATED after
an audited evidence-surfacing/framework correction -- the case the recovery path cannot serve
because start_iteration only bumps the iteration by activating an approved recovery (corrective
compute). The corrected re-gate needs a distinct iteration id so its iteration-scoped Judge task
identity cannot collide with the prior attempt's immutable packets. This proves: (1) it bumps the
iteration and supersedes the prior one without touching artifacts or opening a recovery; (2) its
trigger is a non-recovery marker so the gate path never treats it as a recovery iteration; (3) it
fails closed while a recovery is still pending.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController


class OpenCorrectionIterationTests(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _controller(self, root):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "correction-regate", "stages": [{
            "name": "evaluation", "command": None, "outputs": ["artifacts/report.json"],
            "gate": {"criteria": [self.GATE_CRITERION]},
        }]}))
        controller = RunController.initialize(cfg, root / "run")
        report = controller.run_dir / "artifacts/report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}")
        controller.complete_external_stage("evaluation", [report])
        return controller

    def test_bumps_iteration_without_recovery_or_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root)
            artifacts_before = [dict(a) for a in c.state["artifacts"]]
            recoveries_before = len(c.state.get("recoveries", []))
            old_id = c._current_iteration()["id"]

            new_id = c.open_correction_iteration(
                reason="recovery-004 evidence-surfacing correction re-gate",
                authorized_by="human operator (test)", regate_stage="evaluation")

            c2 = RunController(c.run_dir)
            self.assertEqual(new_id, old_id + 1)
            self.assertEqual(c2._current_iteration()["id"], new_id)
            # prior iteration superseded, no recovery opened, artifacts untouched
            prior = next(it for it in c2.state["iterations"] if it["id"] == old_id)
            self.assertEqual(prior["status"], "superseded")
            self.assertEqual(len(c2.state.get("recoveries", [])), recoveries_before)
            self.assertEqual([dict(a) for a in c2.state["artifacts"]], artifacts_before)
            # non-recovery trigger so the gate path never treats it as a recovery iteration
            trigger = c2._current_iteration()["trigger"]
            self.assertEqual(trigger["kind"], "evidence_surfacing_correction")
            self.assertIsNone(trigger["failed_stage"])
            self.assertEqual(c2._current_iteration()["recovery_execution"]["status"],
                             "not_applicable")
            self.assertTrue(any(e.get("type") == "correction_iteration_started"
                                for e in c2.state["events"]))

    def test_fails_closed_while_recovery_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root)
            c.record_gate("evaluation", "REVISE")  # binds a pending recovery
            with self.assertRaises(RuntimeError):
                c.open_correction_iteration(reason="x", authorized_by="y")

    def test_requires_nonempty_reason_and_authorizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller(root)
            with self.assertRaises(ValueError):
                c.open_correction_iteration(reason="  ", authorized_by="y")
            with self.assertRaises(ValueError):
                c.open_correction_iteration(reason="x", authorized_by="")


if __name__ == "__main__":
    unittest.main()
