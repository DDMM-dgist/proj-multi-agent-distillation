"""Stage D-2 C1 executed-run provenance + closure regression tests (network-free; no execution).

Verifies the committed C1 run artifacts + the provenance-closure corrections: scientific artifacts are
byte-identical to what execution recorded (never altered), the execution-wrapper caveat is recorded
honestly (snapshot present, sha matches, committed_at_execution=false), the transition is
READY_FOR_ADVISORY_JUDGE (not ADVANCE), the Axis-A authoritative verdict is PASS, and the prepared
advisory Judge task is read-only/advisory. Does NOT run MSD or the Judge.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "runs" / "stage_d2" / "d2c1-posthoc-msd-random_x006"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@unittest.skipUnless(RD.exists(), "C1 run dir not present")
class StageD2C1RunProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.prov = json.loads((RD / "provenance.json").read_text())

    def test_scientific_artifacts_unaltered(self):
        for name in ("msd.csv", "msd_summary.json", "criterion_results.json", "approval.json"):
            self.assertEqual(_sha(RD / name), self.prov["artifacts_sha256"][name], name)
        # the exact scientific numbers preserved
        self.assertTrue((RD / "msd.csv").read_text().startswith(
            "frame_index,timestep,time_ps,msd_all,msd_type1,msd_type2\n0,0,0.0,0.0,0.0,0.0\n"))

    def test_execution_wrapper_caveat_recorded(self):
        ew = self.prov["execution_wrapper"]
        self.assertFalse(ew["wrapper_committed_at_execution"])
        self.assertEqual(ew["created_after_approved_head"], "b5762a1d57fad9dd16fe557702bb311117c38786")
        snap = RD / "execution_wrapper_snapshot.py"
        self.assertTrue(snap.exists())
        self.assertEqual(_sha(snap), ew["sha256"])                 # snapshot sha matches recorded
        self.assertEqual(self.prov["STAGE_D2_C1_EXECUTION_ATTEMPT_1"],
                         "AXIS_A_PASS_WITH_EXECUTION_WRAPPER_PROVENANCE_CAVEAT")

    def test_transition_not_advance_yet(self):
        self.assertEqual(self.prov["STAGE_D2_C1_AXIS_A"], "PASS")
        self.assertEqual(self.prov["STAGE_D2_C1_SEMANTIC_JUDGE"], "PENDING")
        self.assertEqual(self.prov["STAGE_D2_C1_TRANSITION"], "READY_FOR_ADVISORY_JUDGE")
        self.assertEqual(self.prov["transition"], "READY_FOR_ADVISORY_JUDGE")
        self.assertEqual(json.loads((RD / "run_manifest.json").read_text())["transition"],
                         "READY_FOR_ADVISORY_JUDGE")

    def test_axis_a_authoritative_pass_and_pbc_not_guaranteed(self):
        cr = json.loads((RD / "criterion_results.json").read_text())
        self.assertTrue(cr["deterministic_authoritative"])
        self.assertEqual(cr["authoritative_verdict"], "PASS")
        self.assertTrue(all(r["result"] for r in cr["criterion_results"]))
        summ = json.loads((RD / "msd_summary.json").read_text())
        self.assertFalse(summ["pbc"]["pbc_hard_guarantee"])        # wrapped-only: no math guarantee
        self.assertEqual(summ["pbc"]["pbc_method"], "minimum_image_continuity")
        self.assertIn("apparent_D_estimate_under_continuity_assumption", summ["apparent_D"])

    def test_isolation_recorded(self):
        self.assertTrue(self.prov["source_unchanged"])
        self.assertTrue(self.prov["stage_d1_unchanged"])
        self.assertTrue(self.prov["writes_under_run_dir_only"])
        self.assertFalse(self.prov["gpu"]); self.assertFalse(self.prov["network"])
        self.assertFalse(self.prov["scheduler"])

    def test_advisory_judge_prep_is_readonly_advisory(self):
        task = json.loads((ROOT / "examples/stage_d2/judge_interpretation_task.json").read_text())
        self.assertIs(task["context"]["deterministic_authoritative"], False)
        self.assertEqual(len(task["criteria"]), 4)
        self.assertIn("pbc_hard_guarantee", " ".join(task["criteria"]).lower() + task["instruction"].lower())
        self.assertIn("read-only", " ".join(task["constraints"]).lower())
        rn = (ROOT / "work" / "stage_d2_judge_run.sh").read_text()
        self.assertIn('--read-allow "$RUN_DIR"', rn)         # reads only the run dir
        self.assertIn("NO scheduler", rn)                    # documented: no scheduler
        self.assertIn("qwen2.5-7b-instruct", rn)
        self.assertIn("--mode shadow", rn)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
