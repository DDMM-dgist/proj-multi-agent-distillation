"""Structural enforcement of the deterministic criterion block by the CANONICAL post-model
validator (orchestration.exchange.validate_agent_response -> validate_judge_vote -> _enforce_
deterministic). Proves the authoritative block is enforced in the acceptance path, not merely
instructed in judge.md: a JudgeVote that reverses a computed boolean, drops/duplicates a
deterministic criterion, converts a missing value into a positive, or (for a fully deterministic
gate) contradicts the deterministic severity is REJECTED. Network-free.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import pydantic  # noqa: F401
    from orchestration.exchange import make_task, validate_agent_response
    from orchestration.specs import load_agent_specs
    from runtimes.pydantic_ai.criterion_eval import attach_to_task, evaluate_criteria
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


def _vote(criteria, oks, verdict, lens="scientific_validity"):
    return {"review_lens": lens, "verdict": verdict,
            "criteria_checked": [{"criterion": c, "value_read": "v", "ok": ok}
                                 for c, ok in zip(criteria, oks)],
            "rationale": "reasoned from the authoritative deterministic results.",
            "required_fix": "" if verdict == "PASS" else "fix the failing criterion."}


@unittest.skipUnless(_HAS, "pydantic/orchestration not importable")
class AuthoritativeEnforcementTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_agent_specs(ROOT / "agent_specs", root=ROOT)["judge"]

    def _task(self, criteria, evidence, specs, *, authoritative=True):
        # a real (non-D1) judge task carrying the deterministic block, built generically
        task = make_task("judge", "Review the gate.", criteria=criteria,
                         context={"review_lens": "scientific_validity",
                                  "review_focus": "Audit the numeric/physical criteria."})
        results = evaluate_criteria(evidence, specs)
        return attach_to_task(task, results, authoritative=authoritative), results

    # a PASS gate (both criteria true) and a FAIL gate (an invalidating criterion false)
    def _pass_gate(self):
        crit = ["m <= 2 (invalidating physical range)", "n <= 2"]
        specs = [{"criterion": crit[0], "operator": "le", "lhs": {"field": "m"},
                  "rhs": {"const": 2}, "invalidating": True},
                 {"criterion": crit[1], "operator": "le", "lhs": {"field": "n"}, "rhs": {"const": 2}}]
        return crit, {"m": 1, "n": 1}, specs

    def _fail_gate(self):
        crit = ["m <= 2 (invalidating physical range)", "n <= 2"]
        specs = [{"criterion": crit[0], "operator": "le", "lhs": {"field": "m"},
                  "rhs": {"const": 2}, "invalidating": True},
                 {"criterion": crit[1], "operator": "le", "lhs": {"field": "n"}, "rhs": {"const": 2}}]
        return crit, {"m": 5, "n": 1}, specs   # m=5 -> invalidating criterion false -> FAIL

    def _revise_gate(self):
        crit = ["m <= 2", "n <= 2"]           # non-invalidating; one false -> REVISE
        specs = [{"criterion": crit[0], "operator": "le", "lhs": {"field": "m"}, "rhs": {"const": 2}},
                 {"criterion": crit[1], "operator": "le", "lhs": {"field": "n"}, "rhs": {"const": 2}}]
        return crit, {"m": 5, "n": 1}, specs

    # 1. deterministic true + Judge false -> rejected
    def test_deterministic_true_but_vote_false_rejected(self):
        crit, ev, specs = self._pass_gate()
        task, res = self._task(crit, ev, specs)
        self.assertTrue(res[0].result)                         # deterministic true
        vote = _vote(crit, [False, True], "REVISE")            # LLM reverses criterion 0 to false
        with self.assertRaisesRegex(ValueError, "contradicts the authoritative deterministic result"):
            validate_agent_response(vote, self.spec, task)

    # 2. deterministic false + Judge true -> rejected
    def test_deterministic_false_but_vote_true_rejected(self):
        crit, ev, specs = self._fail_gate()
        task, res = self._task(crit, ev, specs)
        self.assertFalse(res[0].result)                        # deterministic false
        vote = _vote(crit, [True, True], "PASS")               # LLM flips false -> true
        with self.assertRaisesRegex(ValueError, "contradicts the authoritative deterministic result"):
            validate_agent_response(vote, self.spec, task)

    # 3. deterministic FAIL + Judge REVISE/PASS -> rejected (fully deterministic gate)
    def test_deterministic_fail_but_vote_revise_or_pass_rejected(self):
        crit, ev, specs = self._fail_gate()
        task, _ = self._task(crit, ev, specs)
        # criteria_checked bools MUST match (ok0=False), but verdict claims REVISE/PASS
        for verdict in ("REVISE", "PASS"):
            with self.assertRaises(ValueError):
                validate_agent_response(_vote(crit, [False, True], verdict), self.spec, task)

    # 4. deterministic PASS + Judge REVISE/FAIL -> rejected (fully deterministic gate)
    def test_deterministic_pass_but_vote_revise_or_fail_rejected(self):
        crit, ev, specs = self._pass_gate()
        task, _ = self._task(crit, ev, specs)
        for verdict in ("REVISE", "FAIL"):
            with self.assertRaisesRegex(ValueError, "contradicts the authoritative deterministic severity"):
                validate_agent_response(_vote(crit, [True, True], verdict), self.spec, task)

    # 5. fully consistent JudgeVote -> accepted
    def test_consistent_vote_accepted(self):
        for gate, oks, verdict in ((self._pass_gate(), [True, True], "PASS"),
                                   (self._fail_gate(), [False, True], "FAIL"),
                                   (self._revise_gate(), [False, True], "REVISE")):
            crit, ev, specs = gate
            task, _ = self._task(crit, ev, specs)
            out = validate_agent_response(_vote(crit, oks, verdict), self.spec, task)
            self.assertEqual(out["verdict"], verdict)

    # missing-value cannot be converted into an unsupported positive
    def test_missing_value_cannot_be_flipped_positive(self):
        crit = ["required metric present and <= 2 (invalidating)"]
        specs = [{"criterion": crit[0], "operator": "le", "lhs": {"field": "absent"},
                  "rhs": {"const": 2}, "invalidating": True}]
        task, res = self._task(crit, {}, specs)                # field absent -> False (MISSING_FIELD)
        self.assertIn("MISSING_FIELD", res[0].provenance)
        with self.assertRaisesRegex(ValueError, "contradicts the authoritative deterministic result"):
            validate_agent_response(_vote(crit, [True], "PASS"), self.spec, task)
        # the only consistent vote is ok=False + FAIL (invalidating)
        self.assertEqual(validate_agent_response(_vote(crit, [False], "FAIL"),
                                                 self.spec, task)["verdict"], "FAIL")

    # missing / extra deterministic criterion -> rejected
    def test_missing_or_extra_criterion_rejected(self):
        crit, ev, specs = self._pass_gate()
        task, _ = self._task(crit, ev, specs)
        # drop a criterion from the vote (criteria_checked no longer matches ordered criteria)
        with self.assertRaises(ValueError):
            validate_agent_response(_vote(crit[:1], [True], "PASS"), self.spec, task)
        # add an extra contradictory criterion
        bad = _vote(crit, [True, True], "PASS")
        bad["criteria_checked"].append({"criterion": "surprise", "value_read": "x", "ok": False})
        with self.assertRaises(ValueError):
            validate_agent_response(bad, self.spec, task)

    # advisory block (semantic gate) is reference-only: verdict not bound
    def test_advisory_block_is_not_verdict_binding(self):
        crit, ev, specs = self._pass_gate()
        task, _ = self._task(crit, ev, specs, authoritative=False)
        # deterministic severity would be PASS, but advisory -> a REVISE verdict is allowed
        out = validate_agent_response(_vote(crit, [True, True], "REVISE"), self.spec, task)
        self.assertEqual(out["verdict"], "REVISE")

    # 6. no D1 task-ID special casing: enforcement fires on a generic task id, and the source has
    #    no task-id branching.
    def test_no_task_id_special_casing(self):
        crit, ev, specs = self._fail_gate()
        task, _ = self._task(crit, ev, specs)
        self.assertNotIn("d1-", task["task_id"])               # a normal make_task id, not a D1 id
        with self.assertRaises(ValueError):                    # still enforced -> generic
            validate_agent_response(_vote(crit, [False, True], "PASS"), self.spec, task)
        src = (ROOT / "orchestration" / "exchange.py").read_text()
        self.assertNotIn("d1-", src)
        # the enforcement helper keys off criterion order/identity + the mode flag, not task_id
        helper = src[src.index("def _enforce_deterministic"):src.index("def validate_judge_vote")]
        self.assertNotIn("task_id", helper)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
