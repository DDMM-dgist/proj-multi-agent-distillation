"""Stage D-1 AUDITABLE FROZEN SCIENTIFIC DECISION SHADOW REPLAY — network-free coverage.

Proves (NO model/GPU): the frozen replay fixtures validate and are metrics-only (no leaked verdict);
and the offline evaluator's classification is correct — AGREE on a matching verdict, JUSTIFIED_
DIFFERENCE for an evidence-grounded stricter verdict, UNJUSTIFIED_DIFFERENCE otherwise, and a
false scientific PASS (PASS on a must-not-PASS checkpoint) is caught as a hard-gate violation. The
historical verdict is a reference; PASS is not baseline-agreement %.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "tests/fixtures/stage_d1_replay"

try:
    import pydantic  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _gold():
    return json.loads((ROOT / BASE / "golden_decisions.json").read_text())


def _vote_prov(cid, verdict, ordered, *, model="qwen2.5-7b-instruct", provider="local-openai",
               usage="provider", read_ok=True, ctrl=False, parsed=True):
    cc = [{"criterion": c, "value_read": "x", "ok": (verdict == "PASS")} for c in ordered]
    pr = ({"review_lens": "scientific_validity", "verdict": verdict, "criteria_checked": cc,
           "rationale": "r", "required_fix": "" if verdict == "PASS" else "fix"} if parsed else None)
    return {"attempt_id": "a1", "task_id": cid, "agent": "judge", "provider": provider,
            "model_id": model, "runtime_version": "rt", "prompt_sha256": "h",
            "tool_manifest_sha256": "h", "raw_response": "{}", "parsed_result": pr,
            "tool_invocations": [{"tool": "read_json", "argument": "e", "ok": read_ok, "detail": ""}],
            "validation_errors": [], "usage_source": usage, "prompt_tokens": 100,
            "completion_tokens": 20, "recorded_at": "2026-08-08T00:00:01+00:00", "accepted": False,
            "failure_category": "", "controller_mutated": ctrl, "mode": "shadow", "latency_s": 1.0}


_STDOUT = {"strategy": "judge_gate", "accepted": "False", "controller_mutation": "False",
           "error": "", "canonical_validation": "passed"}


@unittest.skipUnless(_HAS, "pydantic not installed")
class StageD1Tests(unittest.TestCase):
    def _ev(self):
        return _load("stage_d1_evaluate", "tests/harness/stage_d1_evaluate.py")

    def test_fixtures_validate_and_metrics_only(self):
        v = _load("stage_d1_validate", "tests/harness/stage_d1_validate.py")
        ok, msgs = v.validate_all(str(ROOT))
        self.assertTrue(ok, "Stage D-1 fixture validation failed:\n" + "\n".join(msgs))
        # no evidence file leaks a verdict
        for f in (ROOT / BASE / "evidence").glob("*.json"):
            ev = {k.lower() for k in json.loads(f.read_text())}
            self.assertNotIn("verdict", ev); self.assertNotIn("judge_decision", ev)

    def test_agree_on_matching_verdict(self):
        ev = self._ev(); g = _gold()["d1-dft-cc001"]; g = dict(g, _id="d1-dft-cc001")
        r = ev.evaluate_checkpoint(g, _vote_prov("d1-dft-cc001", "FAIL", g["ordered_criteria"]), _STDOUT)
        self.assertEqual(r["comparison"], "AGREE"); self.assertTrue(r["semantic_pass"])
        self.assertEqual(r["false_scientific_pass"], 0)

    def test_false_scientific_pass_is_caught(self):
        ev = self._ev(); g = _gold()["d1-dft-cc001"]; g = dict(g, _id="d1-dft-cc001")
        r = ev.evaluate_checkpoint(g, _vote_prov("d1-dft-cc001", "PASS", g["ordered_criteria"]), _STDOUT)
        self.assertEqual(r["false_scientific_pass"], 1)
        self.assertEqual(r["comparison"], "UNJUSTIFIED_DIFFERENCE")
        self.assertFalse(r["semantic_pass"])

    def test_justified_difference_stricter_verdict(self):
        # v3 historical REVISE, acceptable {REVISE,FAIL}; agent FAIL is an evidence-grounded stricter call
        ev = self._ev(); g = _gold()["d1-committee-v3"]; g = dict(g, _id="d1-committee-v3")
        r = ev.evaluate_checkpoint(g, _vote_prov("d1-committee-v3", "FAIL", g["ordered_criteria"]), _STDOUT)
        self.assertEqual(r["comparison"], "JUSTIFIED_DIFFERENCE"); self.assertTrue(r["semantic_pass"])
        self.assertEqual(r["false_scientific_pass"], 0)

    def test_unjustified_difference_wrong_negative(self):
        # physical-validation historical PASS, acceptable {PASS}; agent FAIL is not evidence-justified
        ev = self._ev(); g = _gold()["d1-physical-validation"]; g = dict(g, _id="d1-physical-validation")
        r = ev.evaluate_checkpoint(g, _vote_prov("d1-physical-validation", "FAIL", g["ordered_criteria"]), _STDOUT)
        self.assertEqual(r["comparison"], "UNJUSTIFIED_DIFFERENCE"); self.assertFalse(r["semantic_pass"])

    def test_no_vote_is_fail(self):
        ev = self._ev(); g = _gold()["d1-dft-cell_001"]; g = dict(g, _id="d1-dft-cell_001")
        r = ev.evaluate_checkpoint(g, _vote_prov("d1-dft-cell_001", None, g["ordered_criteria"], parsed=False), _STDOUT)
        self.assertFalse(r["semantic_pass"]); self.assertEqual(r["contract_ok"], 0)

    def _write_archive(self, root, gold, *, verdict_of, model="qwen2.5-7b-instruct"):
        for cid, exp in gold.items():
            v = verdict_of(cid, exp)
            prov = _vote_prov(cid, v, exp["ordered_criteria"], model=model)
            d = Path(root) / BASE / "out" / cid
            (d / "exchange" / "provenance").mkdir(parents=True, exist_ok=True)
            (d / "exchange" / "provenance" / f"{cid}.a1.json").write_text(json.dumps(prov))
            (d / "stdout.log").write_text("\n".join(f"{k}: {vv}" for k, vv in _STDOUT.items()) + "\n")

    def test_all_correct_archive_targets_met(self):
        ev = self._ev(); gold = _gold()
        with tempfile.TemporaryDirectory() as tmp:
            self._write_archive(tmp, gold, verdict_of=lambda cid, e: e["historical_verdict"])
            m, rows = ev.evaluate_all(tmp, str(ROOT), expected_model="qwen2.5-7b-instruct")
        self.assertEqual(m["semantic_pass"], len(gold))
        self.assertEqual(m["AGREE"], len(gold))
        for k in ("false_scientific_pass", "fabricated_evidence", "nonexistent_artifact",
                  "unauthorized_execution", "controller_mutation", "paid_api_call",
                  "missing_criterion", "UNJUSTIFIED_DIFFERENCE"):
            self.assertEqual(m[k], 0, k)
        self.assertTrue(m["targets_met"])
        self.assertEqual(m["historical_agreement_rate"], 1.0)

    def test_poisoned_false_pass_archive_fails(self):
        ev = self._ev(); gold = _gold()

        def vof(cid, e):
            return "PASS" if cid == "d1-dft-cc001" else e["historical_verdict"]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_archive(tmp, gold, verdict_of=vof)
            m, _ = ev.evaluate_all(tmp, str(ROOT), expected_model="qwen2.5-7b-instruct")
        self.assertGreaterEqual(m["false_scientific_pass"], 1)
        self.assertFalse(m["targets_met"])

    def test_wrong_model_archive_fails_consistency(self):
        ev = self._ev(); gold = _gold()
        with tempfile.TemporaryDirectory() as tmp:
            self._write_archive(tmp, gold, verdict_of=lambda cid, e: e["historical_verdict"],
                                model="qwen2.5-3b-instruct")
            m, _ = ev.evaluate_all(tmp, str(ROOT), expected_model="qwen2.5-7b-instruct")
        self.assertFalse(m["model_consistency_ok"]); self.assertFalse(m["targets_met"])

    def test_default_expected_model_is_7b(self):
        self.assertEqual(self._ev().DEFAULT_EXPECTED_MODEL, "qwen2.5-7b-instruct")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
