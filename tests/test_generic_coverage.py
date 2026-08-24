"""FE-027 P3 -- two coverage axes, evidence-driven sizing, and replay mixing validation.

On the same synthetic non-SiO2 pool: proves target-regime coverage is computed from the real
farthest-point plateau curve (saturated when members collapse, unsaturated when they spread),
Teacher-distribution coverage is a SEPARATE axis that is honestly UNKNOWN with no evidence,
acquisition sizing is an OUTPUT (target_count derived, clamped by a compute ceiling), and the
replay mixing plan validator fails closed on leakage / bad provenance / silent inheritance.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import ase  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from test_generic_regions import _representation, _valid_proposal


def _target_regime_model():
    from framework_v2.acquisition.generic_regions import assemble_target_regime_model
    representation, pool, scope = _representation(discriminative=True)
    proposal = _valid_proposal(representation, scope)
    model = assemble_target_regime_model(
        representation, proposal, scope_contract=scope,
        objective_sha256="obj-sha", model_id="p3-trm")
    return representation, pool, model


@unittest.skipUnless(_HAS_DEPS, "pydantic/ase not installed")
class GenericCoverageP3(unittest.TestCase):
    def test_target_regime_coverage_inputs_from_real_pool(self):
        from framework_v2.acquisition.generic_coverage import (
            FrameworkSizingParams, compute_target_regime_coverage_inputs)

        representation, pool, model = _target_regime_model()
        params = FrameworkSizingParams()
        inputs = compute_target_regime_coverage_inputs(
            pool, representation, model, params=params)

        self.assertEqual(len(inputs), len(model.regimes))
        # Counts partition the resolved pool frames; saturation/headroom are in [0, 1].
        self.assertEqual(sum(i.current_count for i in inputs), pool.total_frames)
        for i in inputs:
            self.assertGreaterEqual(i.saturation, 0.0)
            self.assertLessEqual(i.saturation, 1.0)
            self.assertGreaterEqual(i.novelty_headroom, 0.0)

    def test_saturation_extremes_from_curve(self):
        from framework_v2.acquisition.generic_coverage import (
            FrameworkSizingParams, compute_saturation)

        params = FrameworkSizingParams()
        axes = ["x"]
        scales = {"x": 1.0}
        # Identical points -> zero novelty -> fully saturated.
        identical = [{"x": 0.5} for _ in range(6)]
        sat, head = compute_saturation(identical, axes, scales, params)
        self.assertEqual(sat, 1.0)
        self.assertEqual(head, 0.0)
        # Well-spread points -> the curve does not plateau -> low saturation.
        spread = [{"x": float(i)} for i in range(6)]
        sat2, head2 = compute_saturation(spread, axes, scales, params)
        self.assertLess(sat2, 0.5)
        self.assertGreater(head2, 0.5)
        # Too few members -> honestly non-saturating, never imputed.
        sat3, head3 = compute_saturation([{"x": 0.0}, {"x": 1.0}], axes, scales,
                                         FrameworkSizingParams(min_frames_for_curve=3))
        self.assertEqual(sat3, 0.0)

    def test_teacher_distribution_coverage_is_separate_and_unknown_without_evidence(self):
        from framework_v2.acquisition.generic_coverage import (
            TeacherCoverageStatus, assess_teacher_distribution_coverage)

        cov = assess_teacher_distribution_coverage(
            coverage_id="tdc", target_regime_model_sha256="trm-sha",
            teacher_distribution_evidence=None)
        self.assertEqual(cov.status, TeacherCoverageStatus.UNKNOWN)
        self.assertEqual(cov.per_regime_overlap, {})  # never fabricated

        partial = assess_teacher_distribution_coverage(
            coverage_id="tdc2", target_regime_model_sha256="trm-sha",
            teacher_distribution_evidence={"per_regime_overlap": {"r0": 0.4}, "complete": False})
        self.assertEqual(partial.status, TeacherCoverageStatus.PARTIALLY_KNOWN)
        full = assess_teacher_distribution_coverage(
            coverage_id="tdc3", target_regime_model_sha256="trm-sha",
            teacher_distribution_evidence={"per_regime_overlap": {"r0": 1.0}, "complete": True})
        self.assertEqual(full.status, TeacherCoverageStatus.KNOWN)

    def test_sizing_is_output_and_clamped_by_ceiling(self):
        from framework_v2.acquisition.contracts import (
            AcquisitionPhase, ComputeCeiling, RelevanceRole)
        from framework_v2.acquisition.coverage_gap import (
            RegimeCoverageInput, build_coverage_gap_analysis)
        from framework_v2.acquisition.generic_coverage import (
            FrameworkSizingParams, recommend_acquisition_sizing)

        params = FrameworkSizingParams(batch_growth_factor=1.0, saturation_threshold=0.8)
        # One unsaturated core regime (count 10) and one saturated boundary regime.
        inputs = [
            RegimeCoverageInput("core", RelevanceRole.CORE_TARGET, current_count=10,
                                saturation=0.1, novelty_headroom=0.9),
            RegimeCoverageInput("bnd", RelevanceRole.BOUNDARY_GUARDRAIL, current_count=5,
                                saturation=0.95, novelty_headroom=0.05),
        ]
        coverage = build_coverage_gap_analysis(
            analysis_id="cga", phase=AcquisitionPhase.INITIAL,
            target_regime_model_sha256="trm", region_resolution_sha256="rr",
            regime_inputs=inputs, saturation_threshold=0.8)

        sizing = recommend_acquisition_sizing(coverage, params=params, sizing_id="sz")
        self.assertEqual(sizing.recommended_new["bnd"], 0)          # saturated -> no acquisition
        self.assertEqual(sizing.recommended_new["core"], 10)        # 1.0 * current_count
        self.assertEqual(sizing.target_count["core"], 20)           # OUTPUT: current + new
        self.assertFalse(sizing.ceiling_clamped)

        clamped = recommend_acquisition_sizing(
            coverage, params=params, sizing_id="sz2",
            compute_ceiling=ComputeCeiling(max_candidates_generated=4))
        self.assertTrue(clamped.ceiling_clamped)
        self.assertLessEqual(sum(clamped.recommended_new.values()), 4)

    def test_replay_plan_validator_fails_closed(self):
        from framework_v2.acquisition.generic_replay import (
            ReplayDataMixingPlan, ReplaySourceRef, validate_replay_mixing_plan)

        good = ReplayDataMixingPlan(
            plan_id="rp", target_regime_model_sha256="trm", inherited=False,
            replay_sources=[ReplaySourceRef(
                source_ref="prior_round", n_frames=2, provenance_class="prior_round_labeled",
                frame_ids=["a", "b"])])
        issues = validate_replay_mixing_plan(
            good, protected_ids={"p"}, blind_test_ids={"z"},
            allowed_source_refs={"prior_round"})
        self.assertEqual(issues, [])

        bad = ReplayDataMixingPlan(
            plan_id="rp2", target_regime_model_sha256="trm", inherited=True,
            replay_sources=[
                ReplaySourceRef(source_ref="unknown_src", n_frames=3, provenance_class="x",
                                frame_ids=["a", "a", "p"])])  # dup + leak + count ok(3)
        issues = validate_replay_mixing_plan(
            bad, protected_ids={"p"}, blind_test_ids={"z"},
            allowed_source_refs={"prior_round"})
        self.assertTrue(any("not inherited" in i for i in issues))
        self.assertTrue(any("unaccounted provenance" in i for i in issues))
        self.assertTrue(any("replayed more than once" in i for i in issues))
        self.assertTrue(any("protected-reference" in i for i in issues))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
