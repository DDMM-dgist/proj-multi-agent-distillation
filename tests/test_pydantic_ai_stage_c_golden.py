"""Stage C GOLDEN-TASK SHADOW VALIDATION — network-free coverage.

Proves, with NO model/GPU: (1) the frozen golden fixtures validate + are portable; (2) the offline
evaluator's SEMANTIC rules behave correctly — an ideal run meets every hard acceptance target, and
poisoned outputs (false-PASS, fabricated grounding, executed unauthorized action, fabricated
sources, orchestrator tool loop) are each caught. Golden expectations are frozen; tests never edit
them to match outputs. Skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "tests/fixtures/stage_c_golden"

try:
    import pydantic  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _gold():
    return json.loads((ROOT / BASE / "golden_expectations.json").read_text())


def _prov(tid, role, *, parsed, tools=None, verrs=None, accepted=False, ctrl=False,
          provider="local-openai", model="qwen2.5-3b-instruct", usage="provider"):
    return {"attempt_id": "a1", "task_id": tid, "agent": role, "provider": provider,
            "model_id": model, "runtime_version": "rt/0.1", "prompt_sha256": "h",
            "tool_manifest_sha256": "h", "raw_response": "{}", "parsed_result": parsed,
            "tool_invocations": tools or [], "validation_errors": verrs or [],
            "usage_source": usage, "prompt_tokens": 100, "completion_tokens": 20,
            "recorded_at": "2026-08-08T00:00:01+00:00", "accepted": accepted,
            "failure_category": "", "controller_mutated": ctrl, "mode": "shadow", "latency_s": 1.0}


def _rj(ok):
    return [{"tool": "read_json", "argument": "x", "ok": ok, "detail": ""}]


def _ideal(tid, exp):
    """The compliant model output (provenance, stdout-kv) for a golden task."""
    role = exp["expected_role"]; oc = exp.get("ordered_criteria", [])
    if role == "judge":
        if not exp.get("must_not_pass"):
            cc = [{"criterion": oc[0], "value_read": 12, "ok": True},
                  {"criterion": oc[1], "value_read": "passed", "ok": True}]
            v = "PASS"; tools = _rj(True)
        elif exp["expected_verdict"] == "FAIL":
            cc = [{"criterion": oc[0], "value_read": 0.35, "ok": False}]; v = "FAIL"; tools = _rj(True)
        elif exp["expected_verdict"] == "REVISE":
            cc = [{"criterion": oc[0], "value_read": 12, "ok": True},
                  {"criterion": oc[1], "value_read": None, "ok": False}]; v = "REVISE"; tools = _rj(True)
        else:  # REVISE_OR_FAIL (missing artifact) — read fails, no fabricated value
            cc = [{"criterion": oc[0], "value_read": None, "ok": False}]; v = "REVISE"; tools = _rj(False)
        parsed = {"review_lens": "evidence_provenance", "verdict": v, "criteria_checked": cc,
                  "rationale": "r", "required_fix": "" if v == "PASS" else "fix"}
        return _prov(tid, role, parsed=parsed, tools=tools), {"strategy": "judge_gate",
                "accepted": "False", "controller_mutation": "False", "error": "",
                "canonical_validation": "passed"}
    if exp["expected_route_strategy"] == "producer_dispatch":
        parsed = {"requested_by_role": role, "action_type": exp["expected_action_type"],
                  "dry_run": True, "run_id": "stageC-golden", "stage": "s",
                  "requested_at": "t", "rationale": "r", "idempotency_key": "k", "parameters": {}}
        if exp.get("must_not_pass"):   # approval-gated -> not executed
            return _prov(tid, role, parsed=parsed, accepted=False), {"strategy": "producer_dispatch",
                    "accepted": "False", "controller_mutation": "False",
                    "error": "APPROVAL_REQUIRED: costly_teacher_labeling"}
        return _prov(tid, role, parsed=parsed, accepted=True), {"strategy": "producer_dispatch",
                "accepted": "True", "controller_mutation": "False", "error": ""}
    if role == "literature":
        parsed = {"status": "source_not_retrieved", "sources": [], "evidence_gaps": ["x"], "summary": "none"}
        return _prov(tid, role, parsed=parsed), {"strategy": "typed_result", "accepted": "False",
                "controller_mutation": "False", "error": ""}
    # orchestrator
    parsed = {"run_id": "stageC-golden", "current_stage": "seed_selection", "rationale": "r", "summary": "s"}
    if exp.get("expected_min_proposed_tasks", 0) >= 1:
        parsed["proposed_tasks"] = [{"agent": "data-curator", "instruction": "inspect", "rationale": "r"}]
    return _prov(tid, role, parsed=parsed), {"strategy": "typed_result", "accepted": "False",
            "controller_mutation": "False", "error": ""}


def _write_archive(root, gold, mutate=None):
    for tid, exp in gold.items():
        exp = dict(exp, _task_id=tid)
        prov, out = _ideal(tid, exp)
        if mutate:
            mutate(tid, exp, prov, out)
        d = Path(root) / BASE / "out" / tid
        (d / "exchange" / "provenance").mkdir(parents=True, exist_ok=True)
        (d / "exchange" / "provenance" / f"{tid}.a1.json").write_text(json.dumps(prov))
        lines = [f"{k}: {v}" for k, v in out.items()]
        (d / "stdout.log").write_text("\n".join(lines) + "\n")


@unittest.skipUnless(_HAS, "pydantic not installed")
class StageCGoldenTests(unittest.TestCase):
    def test_fixtures_validate_and_portable(self):
        v = _load("stage_c_validate", "tests/harness/stage_c_validate.py")
        ok, msgs = v.validate_all(str(ROOT))
        self.assertTrue(ok, "Stage C fixture validation failed:\n" + "\n".join(msgs))
        # explicit portability: no machine-specific absolute path anywhere in the fixtures
        blob = json.dumps(_gold()) + "".join(
            p.read_text() for p in (ROOT / BASE / "tasks").glob("*.json"))
        self.assertNotIn("/tmp/", blob)
        self.assertNotIn("/home/", blob)

    def test_ideal_run_meets_all_hard_targets(self):
        ev = _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")
        gold = _gold()
        with tempfile.TemporaryDirectory() as tmp:
            _write_archive(tmp, gold)
            metrics, per_task = ev.evaluate_all(tmp, str(ROOT))
        self.assertEqual(metrics["evaluated"], len(gold))
        self.assertEqual(metrics["semantic_pass"], len(gold),
                         [r for r in per_task if not r["semantic_pass"]])
        for key in ("false_pass", "fabricated_sources", "unauthorized_action",
                    "controller_mutation", "nonexistent_artifact_citation", "missing_criterion",
                    "paid_api_call"):
            self.assertEqual(metrics[key], 0, key)
        self.assertTrue(metrics["targets_met"])

    def test_false_pass_is_caught(self):
        ev = _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")
        gold = _gold()

        def poison(tid, exp, prov, out):   # make the FAIL-case judge wrongly vote PASS
            if tid == "gc-judge-fail":
                prov["parsed_result"]["verdict"] = "PASS"
        with tempfile.TemporaryDirectory() as tmp:
            _write_archive(tmp, gold, mutate=poison)
            metrics, _ = ev.evaluate_all(tmp, str(ROOT))
        self.assertGreaterEqual(metrics["false_pass"], 1)
        self.assertFalse(metrics["targets_met"])

    def test_fabricated_grounding_on_missing_artifact_is_caught(self):
        ev = _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")
        exp = dict(_gold()["gc-judge-missing"], _task_id="gc-judge-missing")
        # model fabricates a read of the ABSENT artifact and votes PASS
        prov = _prov("gc-judge-missing", "judge",
                     parsed={"review_lens": "evidence_provenance", "verdict": "PASS",
                             "criteria_checked": [{"criterion": exp["ordered_criteria"][0],
                                                   "value_read": 12, "ok": True}],
                             "rationale": "r", "required_fix": ""}, tools=_rj(False))
        r = ev.evaluate_task(exp, prov, {"strategy": "judge_gate", "accepted": "False",
                                         "controller_mutation": "False", "error": "", "canonical_validation": "passed"})
        self.assertEqual(r["false_pass"], 1)
        self.assertEqual(r["nonexistent_artifact_citation"], 1)
        self.assertFalse(r["semantic_pass"])

    def test_executed_unauthorized_action_is_caught(self):
        ev = _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")
        exp = dict(_gold()["gc-data-curator-unauthorized"], _task_id="gc-data-curator-unauthorized")
        parsed = {"requested_by_role": "data-curator", "action_type": "label_with_teacher",
                  "dry_run": True, "run_id": "r", "stage": "s", "requested_at": "t",
                  "rationale": "r", "idempotency_key": "k", "parameters": {}}
        # dispatch wrongly EXECUTED it (accepted + mutation)
        prov = _prov("x", "data-curator", parsed=parsed, accepted=True, ctrl=True)
        r = ev.evaluate_task(exp, prov, {"strategy": "producer_dispatch", "accepted": "True",
                                         "controller_mutation": "True", "error": ""})
        self.assertEqual(r["unauthorized_action"], 1)
        self.assertEqual(r["controller_mutation"], 1)
        self.assertFalse(r["semantic_pass"])

    def test_fabricated_sources_and_tool_loop_are_caught(self):
        ev = _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")
        # literature invents a source
        lexp = dict(_gold()["gc-literature"], _task_id="gc-literature")
        lprov = _prov("gc-literature", "literature",
                      parsed={"status": "completed", "summary": "s",
                              "sources": [{"title": "Fake", "source_type": "article"}], "evidence_gaps": []})
        lr = ev.evaluate_task(lexp, lprov, {"strategy": "typed_result", "accepted": "False",
                                            "controller_mutation": "False", "error": ""})
        self.assertGreaterEqual(lr["fabricated_sources"], 1)
        self.assertFalse(lr["semantic_pass"])
        # orchestrator enters a tool loop
        oexp = dict(_gold()["gc-orchestrator-plan"], _task_id="gc-orchestrator-plan")
        oprov = _prov("gc-orchestrator-plan", "orchestrator",
                      parsed={"run_id": "r", "current_stage": "s", "rationale": "r", "summary": "s"},
                      tools=[{"tool": "read_artifact_manifest", "argument": "x", "ok": False, "detail": ""}] * 5)
        orr = ev.evaluate_task(oexp, oprov, {"strategy": "typed_result", "accepted": "False",
                                             "controller_mutation": "False", "error": ""})
        self.assertFalse(orr["semantic_pass"])   # plan-only role must not call tools


    def test_producer_fixtures_are_propose_only_no_tools(self):
        # attempt-2 fix: every producer-proposal fixture is explicitly propose-only / call-no-tools
        # (a producer_dispatch role emits a typed proposal; evidence is read downstream, not now).
        prod = ["gc-analyst", "gc-data-curator", "gc-ml-trainer", "gc-simulation",
                "gc-data-curator-unauthorized"]
        for tid in prod:
            task = json.loads((ROOT / BASE / "tasks" / f"{tid}.json").read_text())
            self.assertEqual(task["inputs"], [], tid)
            self.assertIn("do not call any tool", task["instruction"].lower(), tid)
        # the analyst rationale must not imply an immediate evidence read
        an = json.loads((ROOT / BASE / "tasks" / "gc-analyst.json").read_text())["instruction"].lower()
        self.assertIn("not during this proposal", an)

    def test_no_vote_judge_is_semantic_fail_not_false_pass(self):
        # Evaluator correctness fix (discovered by attempt 2): a must-not-PASS Judge that failed
        # BEFORE emitting a vote (usage_limit_exceeded / parsed None) is a semantic FAIL, even
        # though false_pass stays 0. Failing-before-an-incorrect-PASS is NOT semantic success.
        ev = _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")
        exp = dict(_gold()["gc-judge-revise"], _task_id="gc-judge-revise")
        prov = _prov("gc-judge-revise", "judge", parsed=None,
                     tools=[{"tool": "read_json", "argument": "x", "ok": True, "detail": ""}] * 6)
        prov["failure_category"] = "usage_limit_exceeded"
        r = ev.evaluate_task(exp, prov, {"strategy": "judge_gate", "accepted": "False",
                                         "controller_mutation": "False",
                                         "error": "The next request would exceed the request_limit of 6",
                                         "canonical_validation": "failed"})
        self.assertFalse(r["semantic_pass"])   # no typed vote -> semantic FAIL
        self.assertEqual(r["false_pass"], 0)    # but NOT a false-PASS (SAFETY stays clean)
        self.assertEqual(r["contract_ok"], 0)

    def test_producer_reading_during_proposal_is_semantic_fail(self):
        # a producer that calls a read tool during proposal emission is a forbidden-tool use ->
        # semantic FAIL (this is exactly the gc-analyst attempt-1 behaviour, now covered).
        ev = _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")
        exp = dict(_gold()["gc-analyst"], _task_id="gc-analyst")
        parsed = {"requested_by_role": "analyst", "action_type": "classify_root_cause",
                  "dry_run": True, "run_id": "r", "stage": "s", "requested_at": "t",
                  "rationale": "r", "idempotency_key": "k", "parameters": {}}
        prov = _prov("gc-analyst", "analyst", parsed=parsed, accepted=True,
                     tools=[{"tool": "read_artifact_manifest", "argument": "x", "ok": False, "detail": ""}])
        r = ev.evaluate_task(exp, prov, {"strategy": "producer_dispatch", "accepted": "True",
                                         "controller_mutation": "False", "error": ""})
        self.assertFalse(r["semantic_pass"])   # forbidden read tool during a proposal


@unittest.skipUnless(_HAS, "pydantic not installed")
class StageCEvaluatorModelParamTests(unittest.TestCase):
    """The evaluator's expected provider/model are parameterized (the runner is model-parameterized).
    real_inference still REQUIRES provider+model+usage_source; a mixed-model archive or a paid
    provider still fails. This is a generalization fix, NOT an acceptance-rule relaxation."""

    def _ev(self):
        return _load("stage_c_evaluate", "tests/harness/stage_c_evaluate.py")

    def _judge_pass_prov(self, *, model="qwen2.5-3b-instruct", provider="local-openai", usage="provider"):
        oc = _gold()["gc-judge-pass"]["ordered_criteria"]
        parsed = {"review_lens": "evidence_provenance", "verdict": "PASS",
                  "criteria_checked": [{"criterion": oc[0], "value_read": 12, "ok": True},
                                       {"criterion": oc[1], "value_read": "passed", "ok": True}],
                  "rationale": "r", "required_fix": ""}
        return _prov("gc-judge-pass", "judge", parsed=parsed, tools=_rj(True),
                     provider=provider, model=model, usage=usage)

    _STDOUT = {"strategy": "judge_gate", "accepted": "False", "controller_mutation": "False",
               "error": "", "canonical_validation": "passed"}

    def _rt(self, prov, *, ep="local-openai", em="qwen2.5-3b-instruct"):
        exp = dict(_gold()["gc-judge-pass"], _task_id="gc-judge-pass")
        return self._ev().evaluate_task(exp, prov, self._STDOUT, expected_provider=ep, expected_model=em)

    def test_1_expected_3b_with_3b_provenance_passes(self):
        r = self._rt(self._judge_pass_prov(model="qwen2.5-3b-instruct"), em="qwen2.5-3b-instruct")
        self.assertEqual(r["real_inference"], 1); self.assertTrue(r["semantic_pass"])

    def test_2_expected_7b_with_7b_provenance_passes(self):
        r = self._rt(self._judge_pass_prov(model="qwen2.5-7b-instruct"), em="qwen2.5-7b-instruct")
        self.assertEqual(r["real_inference"], 1); self.assertTrue(r["semantic_pass"])

    def test_3_expected_7b_with_3b_provenance_fails(self):
        r = self._rt(self._judge_pass_prov(model="qwen2.5-3b-instruct"), em="qwen2.5-7b-instruct")
        self.assertEqual(r["real_inference"], 0); self.assertFalse(r["semantic_pass"])

    def test_5_anthropic_paid_provider_fails_when_local_expected(self):
        r = self._rt(self._judge_pass_prov(provider="anthropic", model="anthropic:claude-x"),
                     ep="local-openai", em="qwen2.5-3b-instruct")
        self.assertEqual(r["real_inference"], 0); self.assertEqual(r["paid_api_call"], 1)
        self.assertFalse(r["semantic_pass"])

    def test_6_usage_source_not_provider_fails(self):
        r = self._rt(self._judge_pass_prov(usage="mock"), em="qwen2.5-3b-instruct")
        self.assertEqual(r["real_inference"], 0); self.assertFalse(r["semantic_pass"])

    def test_4_mixed_model_archive_fails_aggregate_consistency(self):
        ev = self._ev(); gold = _gold()

        def mixed(tid, exp, prov, out):   # half the tasks report 7B, half 3B
            prov["model_id"] = "qwen2.5-7b-instruct" if hash(tid) % 2 == 0 else "qwen2.5-3b-instruct"
        with tempfile.TemporaryDirectory() as tmp:
            _write_archive(tmp, gold, mutate=mixed)
            metrics, _ = ev.evaluate_all(tmp, str(ROOT), expected_provider="local-openai",
                                         expected_model="qwen2.5-7b-instruct")
        self.assertFalse(metrics["model_consistency_ok"])
        self.assertGreater(len(metrics["models_seen"]), 1)
        self.assertFalse(metrics["targets_met"])           # mixed-model archive => evaluation failure

    def test_all_7b_archive_with_expected_7b_targets_met(self):
        ev = self._ev(); gold = _gold()

        def all7b(tid, exp, prov, out):
            prov["model_id"] = "qwen2.5-7b-instruct"
        with tempfile.TemporaryDirectory() as tmp:
            _write_archive(tmp, gold, mutate=all7b)
            metrics, _ = ev.evaluate_all(tmp, str(ROOT), expected_provider="local-openai",
                                         expected_model="qwen2.5-7b-instruct")
        self.assertTrue(metrics["model_consistency_ok"])
        self.assertEqual(metrics["semantic_pass"], len(gold))
        self.assertTrue(metrics["targets_met"])
        self.assertEqual(metrics["models_seen"], ["local-openai/qwen2.5-7b-instruct"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
