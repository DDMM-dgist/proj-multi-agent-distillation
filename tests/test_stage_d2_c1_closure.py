"""Stage D-2 C1 Attempt-2 closure regression tests (network-free; no LLM, no inference).

Proves the deterministic closure of the append-only Attempt-2 records: the verdict stays REVISE and the
transition stays REVISE, copied from the preserved exchange provenance (not reconstructed); the append-
only files are fresh and re-generation is refused; Attempt-1 + the DEFERRED record + both exchange
provenances + all Axis-A scientific artifacts remain byte-identical.
"""
from __future__ import annotations

import glob
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "tests" / "fixtures" / "stage_d2" / "d2c1-posthoc-msd-random_x006"
A2 = ("judge_interpretation_attempt2.json", "judge_provenance_attempt2.json",
      "semantic_transition_attempt2.json", "run_manifest.after_judge_attempt2.json")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _exchange(token):
    hits = [p for p in glob.glob(str(RD / "judge_exchange/exchange/provenance/*.json")) if token in p]
    return json.loads(Path(hits[0]).read_text()) if hits else None


@unittest.skipUnless(RD.exists() and (RD / A2[0]).exists(), "closed C1 run dir required")
class StageD2C1ClosureTests(unittest.TestCase):
    def test_closure_used_no_llm(self):
        src = (ROOT / "tests" / "harness" / "stage_d2_close_attempt2.py").read_text().lower()
        for banned in ("openai", "vllm", "run-task", "base_url", "provider_model", "agent.run"):
            self.assertNotIn(banned, src)      # deterministic closure: no inference/provider call

    def test_verdict_and_transition_revise_copied_exactly(self):
        jp = _exchange("02d531f6"); pr = jp["parsed_result"]
        interp = json.loads((RD / "judge_interpretation_attempt2.json").read_text())
        self.assertEqual(interp["advisory_verdict"], "REVISE")
        self.assertEqual(interp["advisory_verdict"], pr["verdict"])
        self.assertEqual(interp["criteria_checked"], pr["criteria_checked"])   # exact copy
        self.assertEqual(interp["rationale"], pr["rationale"])                 # exact copy
        self.assertEqual(interp["required_fix"], pr["required_fix"])           # exact copy
        self.assertIn("not rebound", interp["axis_a_authoritative_verdict"].lower())
        sem = json.loads((RD / "semantic_transition_attempt2.json").read_text())
        self.assertEqual(sem["axis_a_verdict"], "PASS")
        self.assertEqual(sem["semantic_judge_verdict"], "REVISE")
        self.assertEqual(sem["final_transition"], "REVISE")
        self.assertIn("2fe2fc26", sem["closure"]["attempt1_exchange_provenance"])
        self.assertIn("02d531f6", sem["closure"]["attempt2_exchange_provenance"])

    def test_provenance_fields_copied(self):
        jp = _exchange("02d531f6")
        pv = json.loads((RD / "judge_provenance_attempt2.json").read_text())
        for k in ("provider", "model_id", "usage_source", "prompt_tokens", "completion_tokens", "attempt_id"):
            self.assertEqual(pv[k], jp.get(k), k)
        self.assertEqual(pv["provider"], "local-openai")
        self.assertEqual(pv["model_id"], "qwen2.5-7b-instruct")

    def test_append_only_fresh_regeneration_refused(self):
        import sys
        sys.path.insert(0, str(ROOT / "tests" / "harness"))
        from stage_d2_judge_map import assert_appendonly
        with self.assertRaises(FileExistsError):   # attempt-2 files now exist -> re-close refused
            assert_appendonly(RD, 2)

    def test_history_preserved_byte_identical(self):
        recorded = json.loads((RD / "provenance.json").read_text())["artifacts_sha256"]
        for name in ("msd.csv", "msd_summary.json", "criterion_results.json", "approval.json"):
            self.assertEqual(_sha(RD / name), recorded[name], name)
        self.assertEqual(json.loads((RD / "judge_interpretation.json").read_text())["status"], "DEFERRED")
        # both exchange provenances present + intact
        self.assertIsNotNone(_exchange("2fe2fc26"))
        self.assertEqual(_exchange("2fe2fc26").get("failure_category"), "usage_limit_exceeded")
        self.assertIsNotNone(_exchange("02d531f6"))
        self.assertEqual((_exchange("02d531f6").get("parsed_result") or {}).get("verdict"), "REVISE")
        # no successful attempt-1 interpretation file was ever created
        self.assertFalse((RD / "judge_interpretation_attempt1.json").exists())

    def test_original_run_manifest_not_overwritten(self):
        rm = json.loads((RD / "run_manifest.json").read_text())
        self.assertEqual(rm["transition"], "READY_FOR_ADVISORY_JUDGE")   # the pre-Judge manifest, intact
        after = json.loads((RD / "run_manifest.after_judge_attempt2.json").read_text())
        self.assertEqual(after["STAGE_D2_C1_TRANSITION"], "REVISE")      # the new consolidated one


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
