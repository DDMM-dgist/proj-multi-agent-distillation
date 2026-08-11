"""STAGE D-1 HOLDOUT V2 packaging regression tests (network-free; no inference).

Verifies: the selection is deterministic + reproducible + matches the committed manifest (no verdict
input); fixtures validate; specs use only frozen operators; evidence is metrics-only; the authoritative
block binds an adversarial vote to the deterministic verdict (axis A); and the evaluator treats
criterion_contradictions>0 as a hard interpretation failure (axis B) while verdict_overridden alone is
descriptive. No new architecture semantics are introduced.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tests" / "fixtures" / "stage_d1_holdout_v2"
sys.path.insert(0, str(ROOT / "tests" / "harness"))

try:
    import pydantic  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS, "pydantic not installed")
class HoldoutV2Tests(unittest.TestCase):
    def test_selection_is_deterministic_and_matches_manifest(self):
        from stage_d1_holdout_v2_select import select
        committed = json.loads((BASE / "SELECTION_MANIFEST.json").read_text())
        self.assertEqual(select(), committed)            # reproducible from the sha256 rule
        self.assertEqual(len(committed), 7)
        # manifest carries ONLY selection metadata — never a verdict
        for m in committed:
            self.assertEqual(set(m), {"target", "gate", "gate_family", "selection_hash",
                                      "selection_rank", "source"})

    def test_validator_passes(self):
        from stage_d1_holdout_v2_validate import validate_all
        ok, msgs = validate_all(str(ROOT))
        self.assertTrue(ok, "\n".join(m for m in msgs if m.startswith("FAIL")))

    def test_predictions_match_fresh_eval_and_only_frozen_operators(self):
        from runtimes.pydantic_ai.criterion_eval import _OPERATORS, derive_severity, evaluate_criteria
        gold = json.loads((BASE / "golden_decisions.json").read_text())
        preds = json.loads((BASE / "DETERMINISTIC_PREDICTIONS.json").read_text())
        self.assertEqual(len(gold), 7)

        def ops(item, acc):
            for comb in ("all", "any"):
                for sub in item.get(comb, []):
                    ops(sub, acc)
            if "operator" in item:
                acc.add(item["operator"])
        for cid in gold:
            ev = json.loads((BASE / "evidence" / f"{cid}.json").read_text())
            spec = json.loads((BASE / "criteria" / f"{cid}.json").read_text())
            self.assertEqual(preds[cid]["deterministic_suggested_severity"],
                             derive_severity(evaluate_criteria(ev, spec)), cid)
            acc = set()
            for item in spec:
                ops(item, acc)
            self.assertFalse(acc - _OPERATORS, f"{cid}: new operator {acc - _OPERATORS}")

    def test_axis_a_adversarial_vote_binds(self):
        from orchestration.exchange import validate_agent_response
        from orchestration.specs import load_agent_specs
        spec = load_agent_specs(ROOT / "agent_specs", root=ROOT)["judge"]
        wrong = {"PASS": "FAIL", "REVISE": "FAIL", "FAIL": "PASS"}
        for tf in (BASE / "tasks").glob("*.json"):
            task = json.loads(tf.read_text()); ctx = task["context"]
            sev = ctx["deterministic_suggested_severity"]
            det_ok = [b["result"] for b in ctx["deterministic_criterion_results"]]
            adv = {"review_lens": ctx["review_lens"], "verdict": wrong[sev],
                   "criteria_checked": [{"criterion": c, "value_read": "a", "ok": (not det_ok[i])}
                                        for i, c in enumerate(task["criteria"])],
                   "rationale": "adv.", "required_fix": "adv."}
            self.assertEqual(validate_agent_response(adv, spec, task)["verdict"], sev, tf.name)

    def _prov(self, **over):
        base = {"provider": "local-openai", "model_id": "qwen2.5-7b-instruct", "usage_source": "provider",
                "validation_errors": [], "controller_mutated": False,
                "tool_invocations": [{"tool": "read_json", "ok": True}],
                "attempt_id": "a", "prompt_sha256": "p", "tool_manifest_sha256": "t",
                "recorded_at": "2026-08-09T00:00:00Z", "accepted_verdict": None,
                "parsed_result": {"verdict": "REVISE", "criteria_checked": []}}
        base.update(over); return base

    def test_axis_b_criterion_contradictions_is_hard_failure(self):
        from stage_d1_evaluate import evaluate_checkpoint
        gold = json.loads((BASE / "golden_decisions.json").read_text())
        cid = "hv2-er-finetune"; exp = dict(gold[cid], _id=cid)
        crit = exp["ordered_criteria"]
        cc = [{"criterion": c, "value_read": "v", "ok": False} for c in crit]
        good = self._prov(accepted_verdict="REVISE",
                          parsed_result={"verdict": "REVISE", "criteria_checked": cc})
        # no contradictions -> semantic pass
        r = evaluate_checkpoint(exp, good, {})
        self.assertTrue(r["semantic_pass"]); self.assertEqual(r["criterion_contradictions"], 0)
        # a flagged contradiction -> HARD interpretation failure
        bad = self._prov(accepted_verdict="REVISE", criterion_contradictions=[crit[0]],
                         parsed_result={"verdict": "REVISE", "criteria_checked": cc})
        r = evaluate_checkpoint(exp, bad, {})
        self.assertEqual(r["criterion_contradictions"], 1); self.assertFalse(r["semantic_pass"])

    def test_verdict_overridden_alone_is_descriptive_not_failure(self):
        from stage_d1_evaluate import evaluate_checkpoint
        gold = json.loads((BASE / "golden_decisions.json").read_text())
        cid = "hv2-er-finetune"; exp = dict(gold[cid], _id=cid)
        crit = exp["ordered_criteria"]
        cc = [{"criterion": c, "value_read": "v", "ok": False} for c in crit]
        prov = self._prov(accepted_verdict="REVISE", verdict_overridden=True,
                          llm_proposed_verdict="FAIL", criterion_contradictions=[],
                          parsed_result={"verdict": "FAIL", "criteria_checked": cc})
        r = evaluate_checkpoint(exp, prov, {})
        self.assertEqual(r["verdict_overridden"], 1)       # reported
        self.assertTrue(r["semantic_pass"])                # but NOT a failure by itself
        self.assertEqual(r["verdict"], "REVISE")           # accepted verdict is the bound one

    def test_runner_wiring(self):
        rn = (ROOT / "tests" / "harness" / "stage_d1_holdout_v2_replay.sh").read_text()
        self.assertIn("tests/fixtures/stage_d1_holdout_v2", rn)
        self.assertIn("stage_d1_holdout_v2_validate.py", rn)
        self.assertIn('STAGE_D1_GPU_MEM_UTIL:-0.36', rn)
        self.assertIn('STAGE_D1_MIN_FREE_MIB:-18000', rn)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", rn)
        self.assertIn("--mode shadow", rn); self.assertNotIn("pkill", rn)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
