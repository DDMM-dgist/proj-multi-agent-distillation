"""Stage D-2 C1 advisory-Judge APPEND-ONLY packaging tests (network-free; Judge NOT run).

Proves the append-only contract + semantic mapping before any real Judge run: the historical deferred
judge_interpretation.json is never overwritten, attempt filenames must be fresh, preserved scientific
artifacts remain byte-identical, the advisory verdict is NOT deterministically rebound to Axis-A, and
PASS/REVISE/FAIL map to ADVANCE/REVISE/FAIL_STOP. Also checks the runner is loopback-only / local /
retries=0. No GPU, no model.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))

try:
    import stage_d2_judge_map as M  # noqa: E402
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


def _seed_run(d):
    """Create a temp run dir with the preserve-set files (dummy but stable) + a deferred interp."""
    rd = Path(d)
    (rd / "msd.csv").write_text("frame_index,msd_all\n0,0.0\n")
    (rd / "msd_summary.json").write_text('{"pbc":{"pbc_hard_guarantee":false}}\n')
    (rd / "criterion_results.json").write_text('{"authoritative_verdict":"PASS"}\n')
    (rd / "approval.json").write_text('{"approved":true}\n')
    (rd / "execution_wrapper_snapshot.py").write_text("# wrapper\n")
    (rd / "judge_interpretation.json").write_text('{"status":"DEFERRED"}\n')
    return rd


def _prov(verdict, crit=("a", "b", "c", "d")):
    return {"provider": "local-openai", "model_id": "qwen2.5-7b-instruct", "usage_source": "provider",
            "validation_errors": [], "attempt_id": "x", "prompt_sha256": "p", "retry_category": "none",
            "parent_attempt_id": None, "criterion_contradictions": [],
            "parsed_result": {"verdict": verdict, "rationale": "r",
                              "criteria_checked": [{"criterion": c, "value_read": "v", "ok": True} for c in crit]}}


@unittest.skipUnless(_HAS, "stage_d2_judge_map not importable")
class StageD2JudgePrepTests(unittest.TestCase):
    def test_semantic_mapping(self):
        self.assertEqual(M.semantic_transition("PASS"), "ADVANCE")
        self.assertEqual(M.semantic_transition("REVISE"), "REVISE")
        self.assertEqual(M.semantic_transition("FAIL"), "FAIL_STOP")
        self.assertEqual(M.semantic_transition(None), "FAIL_STOP")   # conservative

    def test_advisory_verdict_not_rebound(self):
        # a FAIL advisory verdict stays FAIL even though Axis-A is PASS (no rebinding)
        interp, prov, sem = M.build_attempt_records(_prov("FAIL"), axis_a_verdict="PASS")
        self.assertEqual(interp["advisory_verdict"], "FAIL")
        self.assertIn("not rebound", interp["axis_a_authoritative_verdict"].lower())
        self.assertEqual(sem["STAGE_D2_C1_AXIS_A"], "PASS")
        self.assertEqual(sem["STAGE_D2_C1_TRANSITION"], "FAIL_STOP")
        # REVISE -> REVISE
        _, _, sem2 = M.build_attempt_records(_prov("REVISE"))
        self.assertEqual(sem2["STAGE_D2_C1_TRANSITION"], "REVISE")

    def test_append_only_fresh_and_preserves(self):
        with tempfile.TemporaryDirectory() as d:
            rd = _seed_run(d)
            before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in rd.iterdir()}
            sem = M.write_attempt_records(rd, _prov("PASS"))
            self.assertEqual(sem["STAGE_D2_C1_TRANSITION"], "ADVANCE")
            for f in M.ATTEMPT1:
                self.assertTrue((rd / f).exists())                 # fresh attempt files written
            self.assertTrue((rd / "run_manifest.after_judge.json").exists())
            # preserved artifacts byte-identical (deferred interp + scientific + wrapper)
            for name in M.PRESERVE_BYTE_IDENTICAL:
                self.assertEqual(hashlib.sha256((rd / name).read_bytes()).hexdigest(), before[name], name)
            # a SECOND attempt must be refused (fresh filenames only)
            with self.assertRaises(FileExistsError):
                M.write_attempt_records(rd, _prov("PASS"))

    def test_deferred_missing_refused(self):
        with tempfile.TemporaryDirectory() as d:
            rd = Path(d); (rd / "msd.csv").write_text("x\n")   # no deferred interp
            with self.assertRaises(FileNotFoundError):
                M.assert_appendonly(rd)

    def test_preserved_drift_detected(self):
        with tempfile.TemporaryDirectory() as d:
            rd = _seed_run(d)
            before = M.snapshot_hashes(rd)
            (rd / "msd.csv").write_text("TAMPERED\n")           # simulate history rewrite
            with self.assertRaises(AssertionError):
                M.assert_preserved(rd, before)

    def test_runner_is_loopback_local_retries0_appendonly(self):
        rn = (ROOT / "work" / "stage_d2_judge_run.sh").read_text()
        self.assertIn("127.0.0.1:8000/v1", rn)                  # local loopback endpoint
        self.assertIn("PYDANTIC_AI_PROVIDER=local-openai", rn)
        self.assertIn("retries=0", rn)
        self.assertIn("write_attempt_records", rn)              # append-only writer
        self.assertIn("assert_appendonly", rn)                  # fresh-name precondition
        self.assertNotIn('"$RUN_DIR/judge_interpretation.json"', rn)  # never overwrites the deferred file
        for sched in ("sbatch", "qsub", "squeue", "srun"):        # no scheduler submission
            self.assertNotIn(sched, rn)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
