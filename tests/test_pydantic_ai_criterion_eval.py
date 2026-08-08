"""Deterministic criterion evaluator: scientific numeric predicates are evaluated by Python, not by
LLM free-form arithmetic (which produced the Stage D-1 `0.339 > 0.376` error). Network-free; skips
without the ``pydantic`` extra. Includes the required cases: 0.339<=0.376 -> True; a reversed
comparison cannot occur; a missing value is handled deterministically; compound criteria; plus the
general severity policy and the D1 checkpoint specs reproducing the historical severities.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "examples/stage_d1_replay"

try:
    import pydantic  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS, "pydantic not installed")
class CriterionEvalTests(unittest.TestCase):
    def _m(self):
        import importlib
        return importlib.import_module("runtimes.pydantic_ai.criterion_eval")

    def test_le_true_and_not_reversible(self):
        ce = self._m()
        r = ce.evaluate_criterion({"a": 0.339, "b": 0.376},
                                  {"criterion": "a<=b", "operator": "le",
                                   "lhs": {"field": "a"}, "rhs": {"field": "b"}})
        self.assertTrue(r.result)                      # 0.339 <= 0.376 -> True, deterministically
        self.assertEqual(r.lhs, 0.339); self.assertEqual(r.rhs, 0.376)
        self.assertIn("0.339 <= 0.376 => True", r.provenance)
        # the SAME evidence can never yield False: no LLM path can flip the computed boolean
        for _ in range(3):
            self.assertTrue(ce.evaluate_criterion({"a": 0.339, "b": 0.376},
                            {"operator": "le", "lhs": {"field": "a"}, "rhs": {"field": "b"}}).result)
        # and the strictly-wrong claim gt is deterministically False
        self.assertFalse(ce.evaluate_criterion({"a": 0.339, "b": 0.376},
                         {"operator": "gt", "lhs": {"field": "a"}, "rhs": {"field": "b"}}).result)

    def test_missing_value_is_deterministic_false(self):
        ce = self._m()
        r = ce.evaluate_criterion({}, {"operator": "le", "lhs": {"field": "x"}, "rhs": {"const": 1}})
        self.assertFalse(r.result); self.assertIn("MISSING_FIELD:", r.provenance)
        self.assertTrue(ce.evaluate_criterion({"x": 1}, {"operator": "exists", "lhs": {"field": "x"}}).result)
        self.assertTrue(ce.evaluate_criterion({}, {"operator": "not_exists", "lhs": {"field": "x"}}).result)

    def test_compound_all_any(self):
        ce = self._m()
        allf = ce.evaluate_criterion({"a": 1, "b": 3}, {"all": [
            {"operator": "lt", "lhs": {"field": "a"}, "rhs": {"const": 2}},
            {"operator": "lt", "lhs": {"field": "b"}, "rhs": {"const": 2}}]})
        self.assertFalse(allf.result)                  # 1<2 True AND 3<2 False -> False
        anyt = ce.evaluate_criterion({"a": 1, "b": 3}, {"any": [
            {"operator": "lt", "lhs": {"field": "a"}, "rhs": {"const": 2}},
            {"operator": "lt", "lhs": {"field": "b"}, "rhs": {"const": 2}}]})
        self.assertTrue(anyt.result)

    def test_severity_policy_general(self):
        ce = self._m()
        R = ce.CriterionResult
        # a failed INVALIDATING criterion -> FAIL (invalid/unphysical blocks)
        self.assertEqual(ce.derive_severity([R(criterion="phys", operator="le", result=False,
                                               invalidating=True, provenance="p")]), "FAIL")
        # all met -> PASS
        self.assertEqual(ce.derive_severity([R(criterion="c", operator="le", result=True, provenance="p")]), "PASS")
        # a non-invalidating criterion unmet -> REVISE
        self.assertEqual(ce.derive_severity([R(criterion="c", operator="le", result=False, provenance="p")]), "REVISE")

    def test_in_range_and_approx(self):
        ce = self._m()
        self.assertTrue(ce.evaluate_criterion({"e": -9.7}, {"operator": "in_range",
                        "lhs": {"field": "e"}, "rhs": {"low": -11, "high": -8}}).result)
        self.assertFalse(ce.evaluate_criterion({"e": 17.29}, {"operator": "in_range",
                         "lhs": {"field": "e"}, "rhs": {"low": -11, "high": -8}}).result)
        self.assertTrue(ce.evaluate_criterion({"p": 1.610}, {"operator": "approx",
                        "lhs": {"field": "p"}, "rhs": {"value": 1.61, "tol": 0.05}}).result)

    def test_d1_specs_reproduce_historical_severity_generically(self):
        # The GENERIC specs + evidence deterministically reproduce every historical severity —
        # including the two the 7B judge got wrong (v5 arithmetic; cc001 FAIL-severity). No per-task
        # answer is encoded: this is field-ref predicates + the invalidating-severity policy.
        ce = self._m()
        gold = json.loads((ROOT / BASE / "golden_decisions.json").read_text())
        for cid, exp in gold.items():
            ev = json.loads((ROOT / BASE / "evidence" / f"{cid}.json").read_text())
            spec = json.loads((ROOT / BASE / "criteria" / f"{cid}.json").read_text())
            res = ce.evaluate_criteria(ev, spec)
            self.assertEqual(len(res), len(exp["ordered_criteria"]), cid)     # aligned 1:1
            self.assertEqual(ce.derive_severity(res), exp["historical_verdict"],
                             f"{cid}: deterministic severity != historical")
        # the specific regressions the layer fixes:
        v5 = ce.evaluate_criteria(json.loads((ROOT / BASE / "evidence/d1-committee-v5.json").read_text()),
                                  json.loads((ROOT / BASE / "criteria/d1-committee-v5.json").read_text()))
        self.assertTrue(v5[0].result)          # 0.339 <= 0.376 -> True (not the LLM's false "> ")
        self.assertEqual(ce.derive_severity(v5), "PASS")
        cc = ce.evaluate_criteria(json.loads((ROOT / BASE / "evidence/d1-dft-cc001.json").read_text()),
                                  json.loads((ROOT / BASE / "criteria/d1-dft-cc001.json").read_text()))
        self.assertEqual(ce.derive_severity(cc), "FAIL")   # invalidating physical predicates fail

    def test_attach_to_task_injects_authoritative_context(self):
        # integration point: results go into task.context (which the runtime serializes to the model)
        ce = self._m()
        results = ce.evaluate_criteria({"a": 0.339, "b": 0.376},
            [{"criterion": "a<=b", "operator": "le", "lhs": {"field": "a"}, "rhs": {"field": "b"}}])
        task = {"task_id": "t", "criteria": ["a<=b"], "context": {"review_lens": "scientific_validity"}}
        out = ce.attach_to_task(task, results)
        self.assertEqual(out["context"]["review_lens"], "scientific_validity")   # preserved
        self.assertEqual(out["context"]["deterministic_suggested_severity"], "PASS")
        blk = out["context"]["deterministic_criterion_results"]
        self.assertEqual(len(blk), 1)
        self.assertTrue(blk[0]["result"]); self.assertIn("0.339 <= 0.376 => True", blk[0]["provenance"])
        self.assertIn("authoritative", out["context"]["deterministic_note"].lower())
        self.assertIsNot(out["context"], task["context"])   # original not mutated

    def test_committed_d1_tasks_carry_correct_authoritative_block(self):
        # the frozen replay tasks were attached with the deterministic block; booleans + severity
        # must equal a fresh evaluation and the historical severity (proves the Judge would receive
        # 0.339<=0.376=True for v5 and FAIL-severity for cc001, upstream of any LLM arithmetic).
        ce = self._m()
        gold = json.loads((ROOT / BASE / "golden_decisions.json").read_text())
        for cid, exp in gold.items():
            task = json.loads((ROOT / BASE / "tasks" / f"{cid}.json").read_text())
            block = task["context"]["deterministic_criterion_results"]
            self.assertEqual(len(block), len(task["criteria"]), cid)      # one per ordered criterion
            fresh = ce.evaluate_criteria(
                json.loads((ROOT / BASE / "evidence" / f"{cid}.json").read_text()),
                json.loads((ROOT / BASE / "criteria" / f"{cid}.json").read_text()))
            self.assertEqual([b["result"] for b in block], [r.result for r in fresh], cid)
            self.assertEqual(task["context"]["deterministic_suggested_severity"],
                             exp["historical_verdict"], cid)

    def test_judge_spec_states_results_are_authoritative(self):
        md = (ROOT / "agents" / "judge.md").read_text().lower()
        self.assertIn("deterministic_criterion_results", md)
        self.assertIn("authoritative", md)
        self.assertIn("never recompute or reverse", md)

    def test_no_task_id_special_casing(self):
        # the evaluator and every criterion spec must be generic (no per-checkpoint answer key)
        src = (ROOT / "runtimes/pydantic_ai/criterion_eval.py").read_text()
        self.assertNotIn("d1-", src); self.assertNotIn("task_id", src)
        for f in (ROOT / BASE / "criteria").glob("*.json"):
            spec = json.loads(f.read_text())
            for c in spec:
                keys = set(c) - {"criterion", "operator", "lhs", "rhs", "invalidating", "all", "any"}
                self.assertFalse(keys, f"{f.name}: unexpected spec keys {keys}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
