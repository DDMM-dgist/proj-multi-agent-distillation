"""Network-free deterministic-verdict binding over the 15-case regression corpus (7 development +
8 consumed-holdout auditable decisions). For every authoritative gate, an ADVERSARIAL LLM vote (a
deliberately wrong verdict + flipped booleans) is bound by the canonical acceptance path to the
deterministic policy verdict — proving the refactor makes the accepted decision independent of the
LLM, across the whole corpus, without any model call. No live inference.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPORA = ("tests/fixtures/stage_d1_replay", "tests/fixtures/stage_d1_holdout")
_WRONG = {"PASS": "FAIL", "REVISE": "FAIL", "FAIL": "PASS"}   # a deliberately wrong LLM verdict

try:
    import pydantic  # noqa: F401
    from orchestration.exchange import validate_agent_response
    from orchestration.specs import load_agent_specs
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS, "pydantic/orchestration not importable")
class BindingCorpusTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_agent_specs(ROOT / "agent_specs", root=ROOT)["judge"]

    def test_all_15_cases_bind_to_deterministic_verdict(self):
        total = 0
        for base in CORPORA:
            for tf in sorted((ROOT / base / "tasks").glob("*.json")):
                task = json.loads(tf.read_text())
                ctx = task["context"]
                det_sev = ctx["deterministic_suggested_severity"]
                self.assertIs(ctx["deterministic_authoritative"], True, tf.name)
                crit = task["criteria"]
                # adversarial LLM vote: wrong verdict, every boolean flipped from the deterministic one
                det_ok = [b["result"] for b in ctx["deterministic_criterion_results"]]
                vote = {"review_lens": ctx["review_lens"], "verdict": _WRONG[det_sev],
                        "criteria_checked": [{"criterion": c, "value_read": "adversarial",
                                              "ok": (not det_ok[i])} for i, c in enumerate(crit)],
                        "rationale": "adversarial interpretation.", "required_fix": "adversarial."}
                out = validate_agent_response(vote, self.spec, task)      # must not raise
                self.assertEqual(out["verdict"], det_sev, f"{tf.name}: accepted verdict != policy")
                self.assertEqual([c["ok"] for c in out["criteria_checked"]], det_ok, tf.name)
                total += 1
        self.assertEqual(total, 15, f"expected 15 corpus cases, saw {total}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
