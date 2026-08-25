"""FE-039 regression: the gap-based Stage-4 configuration-space adequacy gate and its targeted
autonomous-reacquisition routing.

Stage 3 proposes an INITIAL acquisition population (the generic FPS/marginal-novelty sizing
heuristic; its N is a starting proposal, never an accepted adequacy result). Stage 4 is the
INDEPENDENT adequacy gate: does the ACQUIRED population structurally SUPPORT each declared
deployment structure class of the FROZEN scope? The only support signal is DECLARED-CLASS
OCCUPANCY resolved through the frozen, human-authored ``config_type -> canonical structure-class``
``label_map`` (criterion provenance ``frozen_deployment_domain``). A declared class with ZERO
acquired representatives is UNSUPPORTED -- a definitional presence/absence fact, NOT an invented
minimum-N / per-class quota / percentage / distance threshold.

The primary fixture is the ACTUAL, IMMUTABLE ffv4p evidence: N=8 acquired frames, all
``config_type == SiOx_crystal_amorphous_interfaces`` (indices [0,100,55,26,10,64,94,89]), assessed
against the frozen deployment scope's real ``label_map`` + ``primary_domains``. No N=8 failure is
hard-coded: the failure is DERIVED from that evidence (7 of 8 declared classes have zero occupancy).

Network-free; the recovery-acceptance half skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from validation.coverage_assessment import (
    aggregate_assessment_status, build_coverage_assessment, validate_coverage_assessment)
from validation.coverage_gap_assessment import (
    build_label_index, build_structure_class_dimensions, compute_structure_class_occupancy,
    derive_reacquisition_targets, resolve_config_type_domain, unsupported_structure_classes)

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

_PROJECT = Path(__file__).resolve().parent.parent
_FROZEN_SCOPE = (_PROJECT / "docs"
                 / "FRESH_CAMPAIGN_FROZEN_POLICIES_v2_fresh_frozen_framework_validation"
                 / "02_deployment_scope_v2.json")

# The IMMUTABLE ffv4p acquisition outcome: 8 frames, every one config_type
# SiOx_crystal_amorphous_interfaces (selected_source_global_indices [0,100,55,26,10,64,94,89]).
_FFV4P_ACQUIRED_COUNTS = {"SiOx_crystal_amorphous_interfaces": 8}


def _frozen_scope():
    return json.loads(_FROZEN_SCOPE.read_text())


def _lineage():
    return {
        "acquisition_manifest_path": "runs/r/acquisition/acquisition.manifest.json",
        "acquisition_manifest_sha256": "a" * 64, "expected_identity": "a" * 64,
        "observed_identity": "a" * 64, "equality_result": "PASS",
    }


def _protection(overlap=0):
    return {
        "reference_id": "reconstructed-teacher-test-1142", "protected_candidate_count": 1143,
        "protected_excluded_count": 15, "eligible_population_after_exclusion": 93,
        "post_selection_overlap_count": overlap, "result": "PASS" if overlap == 0 else "FAIL",
    }


# --- minimal synthetic frozen label_map (no material constant, no real class name) ---------------
_SYN_LABEL_MAP = [
    {"raw_label": "ct_a1", "canonical_domain": "class_A", "claim_role": "primary_claim",
     "rationale": "syn"},
    {"raw_label": "ct_a2", "canonical_domain": "class_A", "claim_role": "supporting_evidence",
     "rationale": "syn"},
    {"raw_label": "ct_b1", "canonical_domain": "class_B", "claim_role": "primary_claim",
     "rationale": "syn"},
    {"raw_label": "ct_oos", "canonical_domain": "OUT_OF_SCOPE_DIAGNOSTIC",
     "claim_role": "out_of_scope", "rationale": "syn"},
    {"raw_label": "ct_amb", "canonical_domain": "AMBIGUOUS", "claim_role": "ambiguous",
     "rationale": "syn"},
]
_SYN_DECLARED = ["class_A", "class_B"]


class Ffv4pRegressionTests(unittest.TestCase):
    """The Section-8 primary regression, driven by the ACTUAL ffv4p evidence + frozen label_map."""

    def setUp(self):
        if not _FROZEN_SCOPE.is_file():
            self.skipTest("frozen deployment scope v2 artifact not present")
        self.scope = _frozen_scope()
        self.declared = self.scope["primary_domains"]
        self.label_map = self.scope["label_map"]

    def test_ffv4p_seven_declared_classes_are_unsupported(self):
        dims = build_structure_class_dimensions(self.declared, _FFV4P_ACQUIRED_COUNTS, self.label_map)
        self.assertEqual(len(dims), len(self.declared))
        self.assertEqual(aggregate_assessment_status(dims), "COVERAGE_INSUFFICIENT")
        unsup = unsupported_structure_classes(dims)
        # exactly the one occupied class (via SiOx_crystal_amorphous_interfaces) is supported.
        supported = [d for d in dims if d["assessment_status"] == "PASS"]
        self.assertEqual(len(supported), 1)
        self.assertEqual(supported[0]["declared_target"]["structure_class"],
                         "amorphous_SiOx_sub_stoichiometric")
        self.assertEqual(supported[0]["criterion"]["observed_occupancy"], 8)
        self.assertEqual(set(unsup), set(self.declared) - {"amorphous_SiOx_sub_stoichiometric"})
        self.assertEqual(len(unsup), 7)

    def test_ffv4p_gate_uses_frozen_deployment_domain_provenance(self):
        dims = build_structure_class_dimensions(self.declared, _FFV4P_ACQUIRED_COUNTS, self.label_map)
        for d in dims:
            self.assertEqual(d["criterion_provenance"], "frozen_deployment_domain")
            self.assertEqual(d["criterion"]["kind"], "structural_presence")
            self.assertEqual(d["criterion"]["min_occupancy"], 1)

    def test_ffv4p_full_typed_block_validates_under_fe038_invariants(self):
        dims = build_structure_class_dimensions(self.declared, _FFV4P_ACQUIRED_COUNTS, self.label_map)
        block = build_coverage_assessment(
            teacher_training_data_access="representative", teacher_access_limitations=[],
            dimensions=dims, acquisition_lineage=_lineage(),
            protected_reference_exclusion=_protection())
        validate_coverage_assessment(block)
        self.assertEqual(block["assessment_status"], "COVERAGE_INSUFFICIENT")

    def test_ffv4p_recovery_targets_widen_and_materialize(self):
        unsup = unsupported_structure_classes(
            build_structure_class_dimensions(self.declared, _FFV4P_ACQUIRED_COUNTS, self.label_map))
        # a pool that has candidates for every raw label declared in the frozen map.
        pool = {m["raw_label"]: 5 for m in self.label_map}
        out = derive_reacquisition_targets(
            unsup, self.label_map, pool,
            already_eligible_source_categories=["SiOx_crystal_amorphous_interfaces"])
        self.assertTrue(out["materializable"])
        self.assertEqual(out["unremediable_classes"], [])
        self.assertTrue(out["widened_eligible_source_categories"])
        # the already-eligible category is never re-listed as a NEW widening.
        self.assertNotIn("SiOx_crystal_amorphous_interfaces",
                         out["widened_eligible_source_categories"])

    def test_no_false_failure_hardcoded_when_all_classes_occupied(self):
        # Section-8 guard: do NOT hard-code that N=8 fails. If every declared class DID have a
        # representative, the SAME gate must return SUFFICIENT with no gap.
        idx = build_label_index(self.label_map)
        # pick one primary_claim raw label per declared class, give each a representative.
        occupied = {}
        for cls in self.declared:
            for raw, entry in idx.items():
                if entry["canonical_domain"] == cls and entry.get("claim_role") == "primary_claim":
                    occupied[raw] = 3
                    break
        dims = build_structure_class_dimensions(self.declared, occupied, self.label_map)
        self.assertEqual(aggregate_assessment_status(dims), "COVERAGE_SUFFICIENT")
        self.assertEqual(unsupported_structure_classes(dims), [])


class GapGateSemanticsTests(unittest.TestCase):
    """Section-9 semantics on a synthetic frozen map (no material constant, no real class name)."""

    def test_zero_occupancy_required_class_is_insufficient(self):
        dims = build_structure_class_dimensions(_SYN_DECLARED, {"ct_a1": 4}, _SYN_LABEL_MAP)
        self.assertEqual(aggregate_assessment_status(dims), "COVERAGE_INSUFFICIENT")
        self.assertEqual(unsupported_structure_classes(dims), ["class_B"])

    def test_all_required_supported_no_false_failure(self):
        dims = build_structure_class_dimensions(_SYN_DECLARED, {"ct_a2": 1, "ct_b1": 1}, _SYN_LABEL_MAP)
        self.assertEqual(aggregate_assessment_status(dims), "COVERAGE_SUFFICIENT")
        self.assertEqual(unsupported_structure_classes(dims), [])

    def test_absent_label_map_is_not_assessable_never_silent_pass(self):
        dims = build_structure_class_dimensions(_SYN_DECLARED, {"ct_a1": 4}, None)
        self.assertEqual(aggregate_assessment_status(dims), "NOT_ASSESSABLE")
        # NOT_ASSESSABLE is neither SUFFICIENT nor a FAIL: it must not appear as an unsupported class.
        self.assertEqual(unsupported_structure_classes(dims), [])
        for d in dims:
            self.assertEqual(d["assessment_status"], "NOT_ASSESSABLE")
            self.assertEqual(d["criterion_provenance"], "absent")

    def test_out_of_scope_and_ambiguous_never_credit_a_class(self):
        occ = compute_structure_class_occupancy({"ct_oos": 9, "ct_amb": 9}, _SYN_DECLARED,
                                                _SYN_LABEL_MAP)
        self.assertEqual(occ["occupancy"], {"class_A": 0, "class_B": 0})
        self.assertEqual(occ["out_of_scope"], {"ct_oos": 9})
        self.assertEqual(occ["ambiguous"], {"ct_amb": 9})
        # a frame that only maps to OOS/AMBIGUOUS leaves both required classes unsupported.
        dims = build_structure_class_dimensions(_SYN_DECLARED, {"ct_oos": 9, "ct_amb": 9},
                                                _SYN_LABEL_MAP)
        self.assertEqual(set(unsupported_structure_classes(dims)), {"class_A", "class_B"})

    def test_unmapped_config_type_never_credits_a_class(self):
        occ = compute_structure_class_occupancy({"totally_unknown_ct": 7}, _SYN_DECLARED,
                                                _SYN_LABEL_MAP)
        self.assertEqual(occ["occupancy"], {"class_A": 0, "class_B": 0})
        self.assertEqual(occ["unmapped"], {"totally_unknown_ct": 7})

    def test_resolve_config_type_domain_fail_closed_reasons(self):
        idx = build_label_index(_SYN_LABEL_MAP)
        self.assertEqual(resolve_config_type_domain("ct_a1", idx), ("class_A", "primary_claim"))
        self.assertEqual(resolve_config_type_domain("ct_oos", idx), (None, "out_of_scope_diagnostic"))
        self.assertEqual(resolve_config_type_domain("ct_amb", idx), (None, "ambiguous"))
        self.assertEqual(resolve_config_type_domain("nope", idx), (None, "unmapped"))

    def test_inconsistent_frozen_label_map_fails_closed(self):
        bad = _SYN_LABEL_MAP + [
            {"raw_label": "ct_a1", "canonical_domain": "class_B", "claim_role": "primary_claim",
             "rationale": "conflict"}]
        with self.assertRaises(ValueError):
            build_label_index(bad)

    def test_cumulative_population_flips_unsupported_class_to_supported(self):
        # Section-5/8: Stage 4 reassesses the CUMULATIVE population. iter0 leaves class_B unsupported;
        # iter1 adds a class_B representative; the union must now support class_B.
        iter0 = {"ct_a1": 4}
        self.assertEqual(unsupported_structure_classes(
            build_structure_class_dimensions(_SYN_DECLARED, iter0, _SYN_LABEL_MAP)), ["class_B"])
        cumulative = dict(iter0)
        cumulative["ct_b1"] = cumulative.get("ct_b1", 0) + 2  # accepted iter1 frames added
        dims = build_structure_class_dimensions(_SYN_DECLARED, cumulative, _SYN_LABEL_MAP)
        self.assertEqual(aggregate_assessment_status(dims), "COVERAGE_SUFFICIENT")
        self.assertEqual(unsupported_structure_classes(dims), [])
        # iter0's accepted class_A frames are preserved in the cumulative occupancy.
        occ = compute_structure_class_occupancy(cumulative, _SYN_DECLARED, _SYN_LABEL_MAP)
        self.assertEqual(occ["occupancy"]["class_A"], 4)


class ReacquisitionTargetingTests(unittest.TestCase):
    """Section-4/9: deterministic reacquisition targeting invents no size and selects no identity."""

    def test_targets_only_name_families_never_frame_identities(self):
        out = derive_reacquisition_targets(
            ["class_B"], _SYN_LABEL_MAP, {"ct_b1": 5},
            already_eligible_source_categories=["ct_a1"])
        # only config_type family names -- no frame index, no size, no per-class quota.
        self.assertEqual(out["target_config_types_by_class"], {"class_B": ["ct_b1"]})
        self.assertEqual(set(out.keys()), {"target_config_types_by_class", "unremediable_classes",
                                           "widened_eligible_source_categories", "materializable"})
        # no numeric size / count key leaked anywhere.
        self.assertNotIn("n_additional", out)
        self.assertNotIn("additional_size", out)

    def test_gap_with_no_pool_candidate_is_unremediable_boundary(self):
        # class_B's only family (ct_b1) has zero remaining pool candidates -> genuine boundary.
        out = derive_reacquisition_targets(["class_B"], _SYN_LABEL_MAP, {"ct_a1": 5})
        self.assertEqual(out["unremediable_classes"], ["class_B"])
        self.assertFalse(out["materializable"])
        self.assertEqual(out["widened_eligible_source_categories"], [])

    def test_all_families_already_eligible_cannot_materialize_byte_identical(self):
        # every target family is ALREADY eligible -> no NEW eligibility -> a reacquisition here would
        # be byte-identical; the deriver reports it as NOT materializable (fail-closed).
        out = derive_reacquisition_targets(
            ["class_B"], _SYN_LABEL_MAP, {"ct_b1": 5},
            already_eligible_source_categories=["ct_b1"])
        self.assertFalse(out["materializable"])
        self.assertEqual(out["widened_eligible_source_categories"], [])

    def test_widening_supersedes_prior_plan_when_new_family_added(self):
        out = derive_reacquisition_targets(
            ["class_A", "class_B"], _SYN_LABEL_MAP, {"ct_a1": 3, "ct_b1": 3},
            already_eligible_source_categories=["ct_a1"])
        self.assertTrue(out["materializable"])
        # ct_b1 is genuinely new eligibility; ct_a1 is not re-widened.
        self.assertIn("ct_b1", out["widened_eligible_source_categories"])
        self.assertNotIn("ct_a1", out["widened_eligible_source_categories"])

    def test_primary_claim_only_restricts_target_families(self):
        # ct_a2 is supporting_evidence; primary_claim_only must not widen to it.
        out = derive_reacquisition_targets(
            ["class_A"], _SYN_LABEL_MAP, {"ct_a1": 2, "ct_a2": 2}, primary_claim_only=True)
        self.assertEqual(out["target_config_types_by_class"], {"class_A": ["ct_a1"]})
        out_all = derive_reacquisition_targets(
            ["class_A"], _SYN_LABEL_MAP, {"ct_a1": 2, "ct_a2": 2}, primary_claim_only=False)
        self.assertEqual(out_all["target_config_types_by_class"]["class_A"], ["ct_a1", "ct_a2"])


_ROSTER = {"data_repair": "data-curator", "orchestration": "orchestrator"}
_STAGES = {"acquisition", "data_coverage"}


def _proposal(**over):
    from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
    base = dict(
        run_id="r", failed_stage="data_coverage", diagnosis_artifact_sha256="d" * 64,
        capability="data_repair", return_stage="data_coverage",
        proposed_changes=[{"type": "reacquire_to_cover_unsupported_classes"}],
        labeling={"teacher_relabel": False, "new_dft": False},
        student_training={"retrain": False, "mode": "none"},
        revalidation={"reuse_profile": True, "targets": ["data_coverage"]},
        rationale="targeted reacquisition for zero-occupancy declared classes")
    base.update(over)
    return RecoveryPlanProposal(**base)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ReacquisitionRecoveryAcceptanceTests(unittest.TestCase):
    """Section-4/9: a COVERAGE_INSUFFICIENT gap routes to return_stage=acquisition that REPLANS
    (supersedes the prior plan) -> materializable -> accepted; a byte-identical no-op is rejected
    at ACCEPTANCE, with the FE-035 controller no-op guard preserved unchanged as the backstop."""

    def _validate(self, p, **over):
        from runtimes.pydantic_ai.recovery_bridge import validate_recovery_plan_proposal
        kwargs = dict(expected_failed_stage="data_coverage", expected_diagnosis_sha256="d" * 64,
                      capability_roster=_ROSTER, valid_stage_names=_STAGES,
                      return_stage_route_action="acquire_structures",
                      return_stage_route_parameters={}, return_stage_replans=True)
        kwargs.update(over)
        return validate_recovery_plan_proposal(p, **kwargs)

    def test_reroute_to_reacquisition_replan_accepted(self):
        p = _proposal(return_stage="acquisition",
                      revalidation={"reuse_profile": False,
                                    "targets": ["acquisition", "data_coverage"]})
        self.assertIsNotNone(self._validate(p))

    def test_no_op_data_coverage_rerun_rejected_before_dispatch(self):
        from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanValidationError
        p = _proposal(corrective_action={
            "action_type": "build_data_coverage_report",
            "parameters": {"requirements": ["re-count structure classes"]}})
        with self.assertRaises(RecoveryPlanValidationError):
            self._validate(
                p, return_stage_route_action="build_data_coverage_report",
                return_stage_route_parameters={"report_path": "runs/r/data_coverage/report.json"},
                return_stage_replans=False)

    def test_no_route_facts_leaves_controller_backstop(self):
        p = _proposal(return_stage="acquisition",
                      revalidation={"reuse_profile": False,
                                    "targets": ["acquisition", "data_coverage"]})
        self.assertIsNotNone(self._validate(
            p, return_stage_route_action=None, return_stage_route_parameters=None,
            return_stage_replans=None))


if __name__ == "__main__":
    unittest.main()
