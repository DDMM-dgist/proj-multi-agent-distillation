"""Stage D-2 C1 Judge path/config fix — reproduces the attempt-1 failure mode and proves the fix
(network-free; no LLM, no GPU). Attempt 1 failed with READ_ALLOW_PATH_RESOLUTION_LOOP: bare evidence
filenames resolved against repo-root CWD and fell outside the run-dir read allow-list. These tests
prove: a bare filename is refused, the repo-relative path is allowed, the task no longer requests a
manifest, request_limit stays 6, retries stay 0, attempt-2 names are fresh append-only, the failed
attempt-1 exchange provenance is preserved, and the scientific artifacts stay byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_REL = "tests/fixtures/stage_d2/d2c1-posthoc-msd-random_x006"
RD = ROOT / RUN_REL
sys.path.insert(0, str(ROOT / "tests" / "harness"))

try:
    import pydantic  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS and RD.exists(), "pydantic / C1 run dir required")
class StageD2JudgePathFixTests(unittest.TestCase):
    # A + B + C: bare filename refused; repo-relative path allowed — via the SAME toolset layer
    def test_bare_filename_refused_repo_relative_allowed(self):
        from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset, ToolAccessError
        allow = [str(RD)]
        cwd = os.getcwd()
        try:
            os.chdir(ROOT)                                  # cwd = repo root (as the Judge CLI ran)
            # A. bare filename resolves to <repo>/msd_summary.json -> OUTSIDE the run-dir allow-list
            with self.assertRaises(ToolAccessError):
                ReadOnlyToolset(allow).read_json("msd_summary.json")
            with self.assertRaises(ToolAccessError):
                ReadOnlyToolset(allow).read_csv_summary("msd.csv")
            # B + C. full repo-relative paths resolve INSIDE the allow-list and succeed
            summ = ReadOnlyToolset(allow).read_json(f"{RUN_REL}/msd_summary.json")
            self.assertIn("pbc", summ)
            csv = ReadOnlyToolset(allow).read_csv_summary(f"{RUN_REL}/msd.csv")
            self.assertEqual(csv["columns"][:4], ["frame_index", "timestep", "time_ps", "msd_all"])
        finally:
            os.chdir(cwd)

    # preflight (section 5): fail-closed check over the same layer, all required paths inside allow-list
    def test_preflight_passes_and_fails_closed(self):
        import stage_d2_judge_preflight as PF
        ok, checks = PF.preflight()
        self.assertTrue(ok)
        req = [c for c in checks if c["required"]]
        self.assertEqual(len(req), 2)
        for c in req:
            self.assertTrue(c["readable"] and c["inside_allow_list"], c)
        # fail closed if a required evidence path is missing (simulate via a wrong repo root)
        ok2, _ = PF.preflight(repo_root="/tmp")
        self.assertFalse(ok2)

    # D: task requests no manifest / read_artifact_manifest, uses repo-relative paths, keeps 4 questions
    def test_task_prohibits_manifest_and_uses_repo_relative_paths(self):
        t = json.loads((ROOT / "tests/fixtures/stage_d2/judge_interpretation_task.json").read_text())
        blob = (t["instruction"] + " " + " ".join(t["constraints"])).lower()
        # the task must explicitly PROHIBIT the spurious manifest read that wasted budget in attempt 1
        self.assertIn("do not call read_artifact_manifest", blob)
        self.assertIn("manifest.json", blob)                # only inside a "do NOT read manifest.json"
        self.assertIn("do not", blob.split("manifest.json")[0][-40:])   # prohibition precedes it
        # repo-relative evidence paths (the actual fix), and bare filenames explicitly forbidden
        self.assertIn(f"{RUN_REL}/msd_summary.json", t["instruction"])
        self.assertIn(f"{RUN_REL}/msd.csv", t["instruction"])
        self.assertIn("do not use bare filenames", t["instruction"].lower())
        self.assertEqual(len(t["criteria"]), 4)
        self.assertIs(t["context"]["deterministic_authoritative"], False)

    # E + F: request_limit stays 6 (global runtime unchanged); runner keeps retries=0
    def test_request_limit_6_and_retries_0_unchanged(self):
        from runtimes.pydantic_ai.models import RuntimeContext
        self.assertEqual(RuntimeContext.model_fields["request_limit"].default, 6)
        self.assertEqual(RuntimeContext.model_fields["provider_retries"].default, 0)
        rn = (ROOT / "tests" / "harness" / "stage_d2_judge_run.sh").read_text()
        self.assertIn("retries=0", rn)
        self.assertIn("stage_d2_judge_preflight.py", rn)     # preflight wired before launch
        self.assertIn("assert_appendonly('$RUN_DIR', $ATTEMPT)", rn)

    # G: append-only attempt naming (attempt 1 already occurred and FAILED; attempt 2 ran + was closed)
    def test_attempt2_names_and_no_successful_attempt1(self):
        import stage_d2_judge_map as M
        names = M.attempt_names(2)
        self.assertTrue(all("attempt2" in n for n in names))
        # a successful attempt-1 interpretation must NEVER exist (attempt 1 failed, stays failed)
        self.assertFalse((RD / "judge_interpretation_attempt1.json").exists())
        # attempt-3 names would be fresh (no attempt 3 has occurred)
        for n in M.attempt_names(3):
            self.assertFalse((RD / n).exists(), n)

    # H + I: failed attempt-1 exchange provenance preserved; scientific artifacts byte-identical
    def test_attempt1_provenance_and_scientific_artifacts_preserved(self):
        import glob
        prov = glob.glob(str(RD / "judge_exchange/exchange/provenance/*.json"))
        # the failed attempt-1 exchange provenance is retained (if the run dir carries it).
        # glob order is filesystem-dependent, so assert the usage_limit_exceeded failure is
        # present among the retained provenance files rather than assuming it is prov[0].
        if prov:
            cats = [json.loads(Path(x).read_text()).get("failure_category") for x in prov]
            self.assertIn("usage_limit_exceeded", cats)
        # the historical deferred interpretation is still DEFERRED
        self.assertEqual(json.loads((RD / "judge_interpretation.json").read_text())["status"], "DEFERRED")
        # scientific artifacts unchanged vs their recorded hashes
        recorded = json.loads((RD / "provenance.json").read_text())["artifacts_sha256"]
        for name in ("msd.csv", "msd_summary.json", "criterion_results.json", "approval.json"):
            self.assertEqual(hashlib.sha256((RD / name).read_bytes()).hexdigest(), recorded[name], name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
