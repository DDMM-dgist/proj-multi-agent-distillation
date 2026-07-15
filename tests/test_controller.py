import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController


class RunControllerTests(unittest.TestCase):
    def test_gate_blocks_next_stage_and_state_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({
                "run_id": "test-run",
                "stages": [
                    {"name": "data", "command": [sys.executable, "-c",
                     "from pathlib import Path; Path('artifacts/data.txt').write_text('ok')"],
                     "outputs": ["artifacts/data.txt"]},
                    {"name": "train", "command": [sys.executable, "-c",
                     "from pathlib import Path; Path('artifacts/model.txt').write_text('ok')"],
                     "outputs": ["artifacts/model.txt"]},
                ],
            }))
            run_dir = root / "run"
            controller = RunController.initialize(cfg, run_dir)
            controller.run_stage("data")
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                controller.run_stage("train")
            controller.record_gate("data", "PASS", "3/3 judges")
            RunController(run_dir).run_stage("train")
            resumed = RunController(run_dir)
            self.assertEqual(resumed.stage("train")["status"], "completed")
            self.assertEqual(len(resumed.state["artifacts"]), 2)

    def test_gate_rejects_unfinished_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{"name": "data", "command": ["false"]}]}))
            controller = RunController.initialize(cfg, root / "run")
            with self.assertRaisesRegex(RuntimeError, "completed"):
                controller.record_gate("data", "PASS")

    def test_external_agent_artifact_can_complete_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "x", "stages": [{"name": "analysis", "command": None}]}))
            controller = RunController.initialize(cfg, root / "run")
            report = root / "run" / "artifacts" / "report.json"
            report.write_text("{}")
            controller.complete_external_stage("analysis", ["artifacts/report.json"])
            self.assertEqual(controller.stage("analysis")["status"], "completed")
            self.assertEqual(len(controller.state["artifacts"]), 1)


if __name__ == "__main__":
    unittest.main()
