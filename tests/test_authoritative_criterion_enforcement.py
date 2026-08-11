"""Deterministic-verdict OWNERSHIP (Stage D-1 refactor) enforced by the canonical acceptance path.

The Stage D-1 holdout exposed an LLM_VERDICT_REGENERATION_CONSISTENCY_FAILURE: asking the LLM to
reproduce the deterministic severity meant a single verdict-copy error failed the whole case. The
architecture now BINDS the accepted verdict + per-criterion booleans from the deterministic policy for
a fully deterministic (authoritative) gate — the LLM owns only interpretation. These tests prove, via
the real orchestration.exchange path, that for authoritative gates the accepted verdict is the
deterministic one regardless of what the LLM emits, that a contradictory criterion commentary is
overridden and flagged (never accepted), that a final verdict is always produced, and that advisory
gates still take a genuine LLM verdict. Network-free.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import pydantic  # noqa: F401
    from orchestration.exchange import (bind_authoritative_judge_vote, make_task,
                                         validate_agent_response)
    from orchestration.specs import load_agent_specs
    from runtimes.pydantic_ai.criterion_eval import attach_to_task, evaluate_criteria
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


def _vote(criteria, oks, verdict, lens="scientific_validity"):
    return {"review_lens": lens, "verdict": verdict,
            "criteria_checked": [{"criterion": c, "value_read": "v", "ok": ok}
                                 for c, ok in zip(criteria, oks)],
            "rationale": "llm interpretation text.",
            "required_fix": "" if verdict == "PASS" else "llm-proposed fix."}


@unittest.skipUnless(_HAS, "pydantic/orchestration not importable")
class DeterministicVerdictOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_agent_specs(ROOT / "agent_specs", root=ROOT)["judge"]

    def _task(self, criteria, evidence, specs, *, authoritative=True):
        task = make_task("judge", "Review the gate.", criteria=criteria,
                         context={"review_lens": "scientific_validity",
                                  "review_focus": "Audit the numeric/physical criteria."})
        return attach_to_task(task, evaluate_criteria(evidence, specs), authoritative=authoritative)

    def _pass_gate(self):
        crit = ["m <= 2 (invalidating)", "n <= 2"]
        specs = [{"criterion": crit[0], "operator": "le", "lhs": {"field": "m"}, "rhs": {"const": 2}, "invalidating": True},
                 {"criterion": crit[1], "operator": "le", "lhs": {"field": "n"}, "rhs": {"const": 2}}]
        return crit, {"m": 1, "n": 1}, specs                # both true -> PASS

    def _fail_gate(self):
        crit = ["m <= 2 (invalidating)", "n <= 2"]
        specs = [{"criterion": crit[0], "operator": "le", "lhs": {"field": "m"}, "rhs": {"const": 2}, "invalidating": True},
                 {"criterion": crit[1], "operator": "le", "lhs": {"field": "n"}, "rhs": {"const": 2}}]
        return crit, {"m": 5, "n": 1}, specs                # invalidating false -> FAIL

    def _revise_gate(self):
        crit = ["m <= 2", "n <= 2"]                         # non-invalidating
        specs = [{"criterion": crit[0], "operator": "le", "lhs": {"field": "m"}, "rhs": {"const": 2}},
                 {"criterion": crit[1], "operator": "le", "lhs": {"field": "n"}, "rhs": {"const": 2}}]
        return crit, {"m": 5, "n": 1}, specs                # one false, none invalidating -> REVISE

    # 1. authoritative REVISE cannot become FAIL
    def test_authoritative_revise_cannot_become_fail(self):
        crit, ev, specs = self._revise_gate()
        task = self._task(crit, ev, specs)
        for llm in ("FAIL", "PASS", "REVISE"):
            out = validate_agent_response(_vote(crit, [True, True], llm), self.spec, task)
            self.assertEqual(out["verdict"], "REVISE")      # policy owns it, not the LLM

    # 2. authoritative FAIL cannot become REVISE/PASS
    def test_authoritative_fail_cannot_become_revise_or_pass(self):
        crit, ev, specs = self._fail_gate()
        task = self._task(crit, ev, specs)
        for llm in ("REVISE", "PASS", "FAIL"):
            out = validate_agent_response(_vote(crit, [True, True], llm), self.spec, task)
            self.assertEqual(out["verdict"], "FAIL")

    # 3. authoritative PASS cannot become REVISE/FAIL
    def test_authoritative_pass_cannot_become_revise_or_fail(self):
        crit, ev, specs = self._pass_gate()
        task = self._task(crit, ev, specs)
        for llm in ("REVISE", "FAIL", "PASS"):
            out = validate_agent_response(_vote(crit, [False, False], llm), self.spec, task)
            self.assertEqual(out["verdict"], "PASS")
            self.assertTrue(all(c["ok"] for c in out["criteria_checked"]))   # booleans bound too

    # 4. a final verdict is produced even if the LLM wording differs (or is malformed structurally)
    def test_final_verdict_always_produced(self):
        crit, ev, specs = self._revise_gate()
        task = self._task(crit, ev, specs)
        # LLM drops a criterion + mislabels booleans; binding rebuilds from ordered criteria
        vote = {"review_lens": "scientific_validity", "verdict": "PASS",
                "criteria_checked": [{"criterion": crit[0], "value_read": "v", "ok": True}],
                "rationale": "partial", "required_fix": ""}
        out = validate_agent_response(vote, self.spec, task)
        self.assertEqual(out["verdict"], "REVISE")
        self.assertEqual([c["criterion"] for c in out["criteria_checked"]], crit)   # rebuilt in order
        self.assertTrue(out["required_fix"].strip())        # filled for a non-PASS verdict

    # 5. contradictory criterion commentary is overridden + flagged (never accepted)
    def test_contradictory_criterion_commentary_flagged_and_overridden(self):
        crit, ev, specs = self._pass_gate()                 # deterministic booleans both True
        task = self._task(crit, ev, specs)
        bound, rec = bind_authoritative_judge_vote(_vote(crit, [False, True], "REVISE"), task)
        self.assertEqual([c["ok"] for c in bound["criteria_checked"]], [True, True])   # overridden
        self.assertIn(crit[0], rec["criterion_contradictions"])                        # flagged
        self.assertTrue(rec["verdict_overridden"])
        self.assertEqual(rec["llm_proposed_verdict"], "REVISE")
        self.assertEqual(rec["authoritative_verdict"], "PASS")
        # a missing-value deterministic False likewise cannot be flipped positive by the LLM
        mv_crit = ["required metric present and <= 2 (invalidating)"]
        mv_specs = [{"criterion": mv_crit[0], "operator": "le", "lhs": {"field": "absent"},
                     "rhs": {"const": 2}, "invalidating": True}]
        mv_task = self._task(mv_crit, {}, mv_specs)
        out = validate_agent_response(_vote(mv_crit, [True], "PASS"), self.spec, mv_task)
        self.assertEqual(out["verdict"], "FAIL")
        self.assertFalse(out["criteria_checked"][0]["ok"])

    # 6. advisory semantic gates still take a genuine LLM verdict
    def test_advisory_gate_keeps_llm_verdict(self):
        crit, ev, specs = self._pass_gate()
        task = self._task(crit, ev, specs, authoritative=False)   # deterministic severity would be PASS
        out = validate_agent_response(_vote(crit, [True, True], "REVISE"), self.spec, task)
        self.assertEqual(out["verdict"], "REVISE")          # advisory -> the LLM decides
        # and an advisory gate still rejects a malformed criteria set (LLM owns the structure here)
        bad = _vote(crit, [True, True], "PASS"); bad["criteria_checked"].pop()
        with self.assertRaises(ValueError):
            validate_agent_response(bad, self.spec, task)

    def test_no_task_id_special_casing(self):
        src = (ROOT / "orchestration" / "exchange.py").read_text()
        self.assertNotIn("d1-", src); self.assertNotIn("hd-", src)
        helper = src[src.index("def bind_authoritative_judge_vote"):src.index("def validate_agent_response")]
        self.assertNotIn("task_id", helper)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
