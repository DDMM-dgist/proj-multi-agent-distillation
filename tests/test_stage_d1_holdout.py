"""STAGE D-1 HOLDOUT packaging regression tests (network-free; no inference).

Verifies the holdout package is well-formed and represents all 8 cases WITHOUT any new architecture
semantics: fixtures validate; specs use only frozen operators + generic keys; evidence is
metrics-only (verdict never leaked); the authoritative block attaches and equals a fresh evaluation;
recorded deterministic predictions match; and the holdout evaluator (delegating to the FROZEN
per-checkpoint semantics) scores a synthetic consistent vote as pass and a false-scientific-PASS as a
hard-gate failure. Also checks the runner wiring. No live model is contacted.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tests" / "fixtures" / "stage_d1_holdout"
sys.path.insert(0, str(ROOT / "tests" / "harness"))

try:
    import pydantic  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS, "pydantic not installed")
class HoldoutPackagingTests(unittest.TestCase):
    def test_validator_passes(self):
        from stage_d1_holdout_validate import validate_all
        ok, msgs = validate_all(str(ROOT))
        self.assertTrue(ok, "\n".join(m for m in msgs if m.startswith("FAIL")))

    def test_eight_cases_and_predictions_match_fresh_eval(self):
        from runtimes.pydantic_ai.criterion_eval import derive_severity, evaluate_criteria
        gold = json.loads((BASE / "golden_decisions.json").read_text())
        preds = json.loads((BASE / "DETERMINISTIC_PREDICTIONS.json").read_text())
        self.assertEqual(len(gold), 8)
        for cid in gold:
            ev = json.loads((BASE / "evidence" / f"{cid}.json").read_text())
            spec = json.loads((BASE / "criteria" / f"{cid}.json").read_text())
            sev = derive_severity(evaluate_criteria(ev, spec))
            self.assertEqual(preds[cid]["deterministic_suggested_severity"], sev, cid)

    def test_specs_use_only_frozen_operators_and_generic_keys(self):
        from runtimes.pydantic_ai.criterion_eval import _OPERATORS
        allowed_keys = {"criterion", "operator", "lhs", "rhs", "invalidating", "all", "any"}

        def ops(item, acc):
            for comb in ("all", "any"):
                for sub in item.get(comb, []):
                    ops(sub, acc)
            if "operator" in item:
                acc.add(item["operator"])
        for f in (BASE / "criteria").glob("*.json"):
            for item in json.loads(f.read_text()):
                self.assertFalse(set(item) - allowed_keys, f"{f.name}: non-generic keys")
                acc = set(); ops(item, acc)
                self.assertFalse(acc - _OPERATORS, f"{f.name}: new operator {acc - _OPERATORS}")

    def test_evidence_metrics_only_no_verdict_leak(self):
        leak = {"verdict", "judge_decision", "decision", "historical_verdict", "gate_decision", "pass"}
        for f in (BASE / "evidence").glob("*.json"):
            ev = json.loads(f.read_text())
            self.assertFalse(leak & {str(k).lower() for k in ev}, f"{f.name}: leaked verdict key")

    def test_authoritative_block_attached_and_matches(self):
        from runtimes.pydantic_ai.criterion_eval import evaluate_criteria
        for f in (BASE / "tasks").glob("*.json"):
            task = json.loads(f.read_text()); cid = f.stem
            ctx = task["context"]
            self.assertIs(ctx["deterministic_authoritative"], True, cid)
            block = ctx["deterministic_criterion_results"]
            fresh = evaluate_criteria(json.loads((BASE / "evidence" / f"{cid}.json").read_text()),
                                      json.loads((BASE / "criteria" / f"{cid}.json").read_text()))
            self.assertEqual([b["result"] for b in block], [r.result for r in fresh], cid)

    def test_no_holdout_case_special_casing_in_helpers(self):
        for rel in ("tests/harness/stage_d1_holdout_gen_fixtures.py", "tests/harness/stage_d1_holdout_validate.py",
                    "tests/harness/stage_d1_holdout_evaluate.py"):
            src = (ROOT / rel).read_text()
            # the evaluator/validator must be generic: no per-case verdict branching
            self.assertNotIn('if cid ==', src, rel)
            self.assertNotIn('== "hd-', src, rel)

    def test_holdout_evaluator_delegates_and_scores(self):
        # synthetic provenance (no inference): a consistent vote passes; a false-scientific PASS fails.
        from stage_d1_evaluate import evaluate_checkpoint
        gold = json.loads((BASE / "golden_decisions.json").read_text())
        # a PASS case scored with a consistent PASS vote + grounded read
        cid = "hd-dft-cell_016"; exp = dict(gold[cid], _id=cid)
        crit = exp["ordered_criteria"]
        good = {"provider": "local-openai", "model_id": "qwen2.5-7b-instruct", "usage_source": "provider",
                "validation_errors": [], "controller_mutated": False,
                "tool_invocations": [{"tool": "read_json", "ok": True}],
                "attempt_id": "a", "prompt_sha256": "p", "tool_manifest_sha256": "t",
                "recorded_at": "2026-08-09T00:00:00Z",
                "parsed_result": {"verdict": "PASS",
                                  "criteria_checked": [{"criterion": c, "value_read": "v", "ok": True} for c in crit]}}
        r = evaluate_checkpoint(exp, good, {})
        self.assertTrue(r["semantic_pass"]); self.assertEqual(r["comparison"], "AGREE")
        self.assertEqual(r["false_scientific_pass"], 0)
        # a must_not_pass REVISE case scored with a PASS vote -> false scientific pass
        cid = "hd-committee-v3final"; exp = dict(gold[cid], _id=cid)
        crit = exp["ordered_criteria"]
        bad = dict(good, parsed_result={"verdict": "PASS",
                   "criteria_checked": [{"criterion": c, "value_read": "v", "ok": True} for c in crit]})
        r = evaluate_checkpoint(exp, bad, {})
        self.assertEqual(r["false_scientific_pass"], 1); self.assertFalse(r["semantic_pass"])

    def test_runner_wiring(self):
        rn = (ROOT / "tests" / "harness" / "stage_d1_holdout_replay.sh").read_text()
        self.assertIn("tests/fixtures/stage_d1_holdout", rn)
        self.assertIn("stage_d1_holdout_validate.py", rn)
        self.assertIn('STAGE_D1_GPU_MEM_UTIL:-0.36', rn)
        self.assertIn('STAGE_D1_MIN_FREE_MIB:-18000', rn)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", rn)
        self.assertIn("--mode shadow", rn)
        self.assertNotIn("pkill", rn)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
