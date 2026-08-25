"""FE-038 closure regression: the typed data-coverage assessability contract and the
capability-aware recovery-plan acceptance layer. Reproduces the exact ffv4p Stage-4 / recovery
case (retired in runs/sio2-sox-allegro-simplenn-ffv4p/DIAGNOSTIC_TERMINATION_NOTE.md) plus the
synthetic COVERAGE_SUFFICIENT / COVERAGE_INSUFFICIENT / NOT_ASSESSABLE cases.

Two invariants under test:
  * absence of a criterion is NOT_ASSESSABLE, never a false FAIL/insufficiency;
  * presence of no threshold is not a PASS -- a PASS must name a real, met criterion.

And the recovery-acceptance invariant: a plan whose corrective action provably materializes no
changed artifact (the ffv4p build_data_coverage_report-with-opaque-requirements case) is rejected
at ACCEPTANCE, before dispatch -- while the no-op guard is preserved unchanged as the backstop.

Network-free; the recovery half skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import unittest

from validation.coverage_assessment import (
    aggregate_assessment_status, build_coverage_assessment, make_dimension,
    validate_coverage_assessment)

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _lineage(equal=True):
    return {
        "acquisition_manifest_path": "runs/r/acquisition/acquisition.manifest.json",
        "acquisition_manifest_sha256": "a" * 64,
        "expected_identity": "a" * 64,
        "observed_identity": ("a" * 64) if equal else ("b" * 64),
        "equality_result": "PASS" if equal else "FAIL",
    }


def _protection(overlap=0):
    return {
        "reference_id": "reconstructed-teacher-test-1142",
        "protected_candidate_count": 1143,
        "protected_excluded_count": 15,
        "eligible_population_after_exclusion": 93,
        "post_selection_overlap_count": overlap,
        "result": "PASS" if overlap == 0 else "FAIL",
    }


class CoverageAssessmentInvariantTests(unittest.TestCase):
    def test_absent_criterion_is_not_assessable_never_fail(self):
        dim = make_dimension(
            dimension_id="config_type_coverage", declared_target={"config_types": ["a"]},
            metric="frame_count_by_config_type", criterion_provenance="absent",
            observed_support={"counts": {"a": 3}},
            reason="no frozen coverage_requirement exists")
        self.assertEqual(dim["assessment_status"], "NOT_ASSESSABLE")
        self.assertIsNone(dim["met"])
        self.assertEqual(aggregate_assessment_status([dim]), "NOT_ASSESSABLE")

    def test_absent_provenance_cannot_be_forced_to_fail(self):
        dim = make_dimension(
            dimension_id="d", declared_target={}, metric="m", criterion_provenance="absent",
            observed_support={}, reason="r")
        dim["assessment_status"] = "FAIL"  # a caller trying to fabricate insufficiency
        with self.assertRaises(ValueError):
            validate_coverage_assessment(build_coverage_assessment(
                teacher_training_data_access="representative", teacher_access_limitations=[],
                dimensions=[dim], acquisition_lineage=_lineage(),
                protected_reference_exclusion=_protection()))

    def test_pass_requires_real_met_criterion(self):
        dim = make_dimension(
            dimension_id="d", declared_target={}, metric="m",
            criterion_provenance="frozen_deployment_domain",
            criterion={"min_frames_by_config_type": {"a": 2}, "met": True},
            observed_support={"counts": {"a": 3}}, reason="met")
        self.assertEqual(dim["assessment_status"], "PASS")
        self.assertEqual(aggregate_assessment_status([dim]), "COVERAGE_SUFFICIENT")

    def test_unmet_criterion_is_fail_insufficient(self):
        dim = make_dimension(
            dimension_id="d", declared_target={}, metric="m",
            criterion_provenance="frozen_deployment_domain",
            criterion={"min_frames_by_config_type": {"a": 5}, "met": False},
            observed_support={"counts": {"a": 3}}, reason="unmet")
        self.assertEqual(dim["assessment_status"], "FAIL")
        self.assertEqual(aggregate_assessment_status([dim]), "COVERAGE_INSUFFICIENT")

    def test_non_absent_provenance_without_criterion_rejected(self):
        with self.assertRaises(ValueError):
            make_dimension(
                dimension_id="d", declared_target={}, metric="m",
                criterion_provenance="frozen_deployment_domain",
                observed_support={}, reason="r")

    def test_empty_dimensions_is_not_assessable(self):
        self.assertEqual(aggregate_assessment_status([]), "NOT_ASSESSABLE")

    def test_lineage_equality_inconsistency_rejected(self):
        dim = make_dimension(
            dimension_id="d", declared_target={}, metric="m", criterion_provenance="absent",
            observed_support={}, reason="r")
        bad = build_coverage_assessment(
            teacher_training_data_access="representative", teacher_access_limitations=[],
            dimensions=[dim], acquisition_lineage=_lineage(equal=True),
            protected_reference_exclusion=_protection())
        bad["acquisition_lineage"]["equality_result"] = "FAIL"  # PASS identities, FAIL result
        with self.assertRaises(ValueError):
            validate_coverage_assessment(bad)

    def test_protected_overlap_must_be_fail(self):
        dim = make_dimension(
            dimension_id="d", declared_target={}, metric="m", criterion_provenance="absent",
            observed_support={}, reason="r")
        leaked = build_coverage_assessment(
            teacher_training_data_access="representative", teacher_access_limitations=[],
            dimensions=[dim], acquisition_lineage=_lineage(),
            protected_reference_exclusion=_protection(overlap=2))
        self.assertEqual(leaked["protected_reference_exclusion"]["result"], "FAIL")
        validate_coverage_assessment(leaked)  # FAIL with overlap>0 is CONSISTENT
        leaked["protected_reference_exclusion"]["result"] = "PASS"  # PASS with overlap>0 is not
        with self.assertRaises(ValueError):
            validate_coverage_assessment(leaked)


class Ffv4pStage4RegressionTests(unittest.TestCase):
    """The exact ffv4p Stage-4 shape: no frozen quantitative coverage criterion exists for SiO2-x,
    so config_type_coverage and every declared structure class are NOT_ASSESSABLE -- the report is
    NOT_ASSESSABLE, NOT a false COVERAGE_INSUFFICIENT; FE-037 protection and lineage are surfaced
    explicitly; teacher access is geometry-only."""

    def _ffv4p_assessment(self):
        dims = [make_dimension(
            dimension_id="config_type_coverage", declared_target={"config_types": ["amorphous"]},
            metric="frame_count_by_config_type", criterion_provenance="absent",
            observed_support={"counts": {"amorphous": 8}},
            reason="no frozen coverage_requirement.min_frames_by_config_type in the locked domain")]
        for sc in ("silicon_crystalline_main", "sio2_quartz", "suboxide_interface"):
            dims.append(make_dimension(
                dimension_id=f"structure_class:{sc}", declared_target={"structure_class": sc},
                metric="frame_support_by_structure_class", criterion_provenance="absent",
                observed_support={"note": "no per-structure-class support metric"},
                reason="declared deployment structure class carries no frozen/derivable criterion"))
        return build_coverage_assessment(
            teacher_training_data_access="representative",
            teacher_access_limitations=["Teacher training distribution is geometry-only; labels "
                                        "not independently re-verified in this run"],
            dimensions=dims, acquisition_lineage=_lineage(),
            protected_reference_exclusion=_protection())

    def test_ffv4p_is_not_assessable_not_a_false_insufficiency(self):
        a = validate_coverage_assessment(self._ffv4p_assessment())
        self.assertEqual(a["assessment_status"], "NOT_ASSESSABLE")
        self.assertNotEqual(a["assessment_status"], "COVERAGE_INSUFFICIENT")
        self.assertTrue(all(d["assessment_status"] == "NOT_ASSESSABLE" for d in a["dimensions"]))

    def test_ffv4p_surfaces_fe037_protection_and_lineage(self):
        a = self._ffv4p_assessment()
        self.assertEqual(a["protected_reference_exclusion"]["result"], "PASS")
        self.assertEqual(a["protected_reference_exclusion"]["post_selection_overlap_count"], 0)
        self.assertEqual(a["acquisition_lineage"]["equality_result"], "PASS")
        self.assertEqual(a["teacher_training_data_access"]["mode"], "representative")
        self.assertTrue(a["teacher_training_data_access"]["limitations"])

    def test_sufficient_case_with_frozen_criterion(self):
        dim = make_dimension(
            dimension_id="config_type_coverage", declared_target={"config_types": ["a", "b"]},
            metric="frame_count_by_config_type", criterion_provenance="frozen_deployment_domain",
            criterion={"min_frames_by_config_type": {"a": 2, "b": 1}, "unmet": {}, "met": True},
            observed_support={"counts": {"a": 4, "b": 3}}, reason="all frozen minimums met")
        a = validate_coverage_assessment(build_coverage_assessment(
            teacher_training_data_access="full", teacher_access_limitations=[], dimensions=[dim],
            acquisition_lineage=_lineage(), protected_reference_exclusion=_protection()))
        self.assertEqual(a["assessment_status"], "COVERAGE_SUFFICIENT")

    def test_insufficient_case_with_frozen_criterion(self):
        dim = make_dimension(
            dimension_id="config_type_coverage", declared_target={"config_types": ["a"]},
            metric="frame_count_by_config_type", criterion_provenance="frozen_deployment_domain",
            criterion={"min_frames_by_config_type": {"a": 10},
                       "unmet": {"a": {"required_min_frames": 10, "observed_frames": 3}},
                       "met": False},
            observed_support={"counts": {"a": 3}}, reason="config_types below frozen minimum: ['a']")
        a = validate_coverage_assessment(build_coverage_assessment(
            teacher_training_data_access="full", teacher_access_limitations=[], dimensions=[dim],
            acquisition_lineage=_lineage(), protected_reference_exclusion=_protection()))
        self.assertEqual(a["assessment_status"], "COVERAGE_INSUFFICIENT")


_ROSTER = {"data_repair": "data-curator", "orchestration": "orchestrator"}
_STAGES = {"acquisition", "data_coverage"}


def _proposal(**over):
    from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
    base = dict(
        run_id="r", failed_stage="data_coverage", diagnosis_artifact_sha256="d" * 64,
        capability="data_repair", return_stage="data_coverage",
        proposed_changes=[{"type": "surface_coverage_evidence"}],
        labeling={"teacher_relabel": False, "new_dft": False},
        student_training={"retrain": False, "mode": "none"},
        revalidation={"reuse_profile": True, "targets": ["data_coverage"]},
        rationale="surface coverage assessability evidence")
    base.update(over)
    return RecoveryPlanProposal(**base)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class RecoveryCapabilityAwareAcceptanceTests(unittest.TestCase):
    """The ffv4p recovery case: corrective_action == the return stage's OWN route action, carrying
    only opaque free-text `requirements` that supersede no declared typed route parameter, no
    scientific recompute, and data_coverage does not replan -> provable no-op. It must be rejected
    at ACCEPTANCE, before dispatch, not left to the propose_recovery exit-2 backstop."""

    # data_coverage's declared typed route params (subset), from the ffv4p workflow.yaml.
    _ROUTE_PARAMS = {"candidate_dataset": "runs/r/acq/candidates.xyz",
                     "acquisition_manifest": "runs/r/acq/acquisition.manifest.json",
                     "report_path": "runs/r/data_coverage/report.json"}

    def _validate(self, p, **over):
        from runtimes.pydantic_ai.recovery_bridge import validate_recovery_plan_proposal
        kwargs = dict(expected_failed_stage="data_coverage", expected_diagnosis_sha256="d" * 64,
                      capability_roster=_ROSTER, valid_stage_names=_STAGES,
                      return_stage_route_action="build_data_coverage_report",
                      return_stage_route_parameters=self._ROUTE_PARAMS,
                      return_stage_replans=False)
        kwargs.update(over)
        return validate_recovery_plan_proposal(p, **kwargs)

    def test_ffv4p_opaque_requirements_rejected_before_dispatch(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        p = _proposal(corrective_action={
            "action_type": "build_data_coverage_report",
            "parameters": {"requirements": ["surface protected-reference provenance",
                                            "surface acquisition-lineage equality"]}})
        with self.assertRaises(RecoveryPlanValidationError) as ctx:
            self._validate(p)
        self.assertIn("not materializable", str(ctx.exception))
        self.assertIn("requirements", str(ctx.exception))

    def test_bare_forward_rerun_no_corrective_rejected(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        with self.assertRaises(RecoveryPlanValidationError):
            self._validate(_proposal(corrective_action=None))

    def test_distinct_evidence_action_accepted(self):
        # A distinct evidence-producing executor (!= the return stage's own route action) materializes
        # a new artifact, so it is accepted at the materialization axis.
        p = _proposal(corrective_action={
            "action_type": "compare_deployment_coverage", "parameters": {}})
        self.assertIsNotNone(self._validate(p))

    def test_typed_input_override_accepted(self):
        p = _proposal(corrective_action={
            "action_type": "build_data_coverage_report",
            "parameters": {"candidate_dataset": "runs/r/acq/candidates_v2.xyz"}})
        self.assertIsNotNone(self._validate(p))

    def test_reroute_to_replanning_acquisition_accepted(self):
        p = _proposal(return_stage="acquisition",
                      revalidation={"reuse_profile": False,
                                    "targets": ["acquisition", "data_coverage"]})
        self.assertIsNotNone(self._validate(
            p, return_stage_route_action="acquire_structures",
            return_stage_route_parameters={}, return_stage_replans=True))

    def test_no_route_facts_supplied_leaves_backstop(self):
        # Omitting route facts must NOT reject (controller guard remains the backstop).
        p = _proposal(corrective_action={
            "action_type": "build_data_coverage_report",
            "parameters": {"requirements": ["x"]}})
        self.assertIsNotNone(self._validate(
            p, return_stage_route_action=None, return_stage_route_parameters=None,
            return_stage_replans=None))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
