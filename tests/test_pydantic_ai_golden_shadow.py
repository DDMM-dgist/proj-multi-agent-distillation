"""Phase 7: golden-shadow harness (no provider run). Fixture/TestModel results are NEVER
reported as an actual comparison. Network-free; skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import unittest

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _task():
    from runtimes.pydantic_ai.golden_shadow import GoldenTask
    return GoldenTask(task_id="jt1", artifact_path="runs/b/stage1.json", artifact_sha256="abc",
                      ordered_criteria=["c1", "c2"], assigned_lens="evidence_provenance",
                      role_prompt_sha256="ph", validation_profile="sio2_v1", gate_context_sha256="gc")


def _vote(lens="evidence_provenance", verdict="PASS", checked=("c1", "c2")):
    return {"review_lens": lens, "verdict": verdict,
            "criteria_checked": [{"criterion": c, "value_read": "x", "ok": verdict == "PASS"}
                                 for c in checked],
            "rationale": "r", "required_fix": "" if verdict == "PASS" else "fix"}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class GoldenShadowHarnessTests(unittest.TestCase):
    def test_fixture_run_is_pending_not_actual(self):
        from runtimes.pydantic_ai.golden_shadow import build_report, STATUS_HARNESS_READY
        report = build_report([(_task(), _vote(), _vote(), None)], source="fixture")
        self.assertEqual(report.status, STATUS_HARNESS_READY)  # never "actual" from a fixture
        self.assertEqual(report.n_tasks, 1)

    def test_agreement_and_coverage_metrics(self):
        from runtimes.pydantic_ai.golden_shadow import compare_votes
        c = compare_votes(_task(), _vote(), _vote())
        self.assertTrue(c.verdict_agreement)
        self.assertEqual(c.criterion_coverage, 1.0)

    def test_detects_false_pass(self):
        from runtimes.pydantic_ai.golden_shadow import compare_votes
        c = compare_votes(_task(), _vote(verdict="FAIL"), _vote(verdict="PASS"))
        self.assertTrue(c.false_pass)

    def test_detects_wrong_lens(self):
        from runtimes.pydantic_ai.golden_shadow import compare_votes
        c = compare_votes(_task(), _vote(), _vote(lens="scientific_validity"))
        self.assertTrue(c.wrong_lens_output)

    def test_partial_criterion_coverage(self):
        from runtimes.pydantic_ai.golden_shadow import compare_votes
        c = compare_votes(_task(), _vote(), _vote(checked=("c1",)))
        self.assertEqual(c.criterion_coverage, 0.5)

    def test_report_aggregates_safety_signals(self):
        from runtimes.pydantic_ai.golden_shadow import build_report
        report = build_report([
            (_task(), _vote(verdict="FAIL"), _vote(verdict="PASS"), None),   # false pass
            (_task(), _vote(), _vote(lens="scientific_validity"), None),      # wrong lens
        ], source="fixture")
        self.assertEqual(report.false_pass_count, 1)
        self.assertEqual(report.wrong_lens_accepted, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
