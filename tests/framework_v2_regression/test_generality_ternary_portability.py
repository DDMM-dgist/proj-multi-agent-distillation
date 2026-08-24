"""Framework V2 closure — cross-material generality audit (generality addendum).

The hard acceptance requirement: the generic scientific core must drive a
chemically DIFFERENT campaign (a multicomponent / ternary system) through the
SAME code path without any per-chemistry rewrite, and the portability audit must
return GENERALITY_AUDIT = PASS.

This test defines a synthetic ternary campaign (a Cu-Zn-Sn kesterite-style
system — deliberately sharing no elements with the SiO2 campaign) whose material
science enters ONLY through a registered fact-producer plugin and
capability-advertising model adapters. It then runs the generic
:func:`run_portability_audit` and asserts PASS, including the source scan that
proves the generic core contains none of this campaign's material tokens.

A negative test confirms the audit is not tautological: a broken campaign (a
fact producer that emits nothing) yields FAIL.
"""
from __future__ import annotations

import unittest

from framework_v2.adapters import (
    ModelAdapterSpec, ModelRole, register_fact_producer)
from framework_v2.facts import DeterministicFact, FactVerdict
from framework_v2.generality_audit import (
    PortabilityCampaign, run_portability_audit)


# --- campaign plugin: material science lives ENTIRELY here, not in the core ---
_TERNARY_TOKENS = ("cu_zn_sn_kesterite", "czts_ternary_v1")


def _czts_phase_facts(*, cell_id: str = "czts-cell-0", **_ctx):
    """A registered ternary fact producer. Returns typed DeterministicFacts
    describing a synthetic Cu-Zn-Sn phase decision — no SiO2, no core edits."""
    return [
        DeterministicFact(
            fact_id="czts-phase-fraction", kind="ternary_phase_fraction",
            observed={"kesterite": 0.82, "stannite": 0.18}, expected=None,
            verdict=FactVerdict.PASS, validator="czts.phase_descriptor",
            rationale="single-phase-dominant kesterite cell"),
        DeterministicFact(
            fact_id="czts-cation-order", kind="cation_ordering_parameter",
            observed=0.91, expected=0.85, verdict=FactVerdict.PASS,
            validator="czts.phase_descriptor",
            rationale="Cu/Zn ordering above threshold"),
        DeterministicFact(
            fact_id="czts-composition", kind="composition_within_scope",
            observed={"Cu": 2, "Zn": 1, "Sn": 1}, expected=None,
            verdict=FactVerdict.PASS, validator="czts.phase_descriptor",
            rationale="stoichiometric Cu2ZnSn cell"),
    ]


def _register_once(name, producer):
    try:
        register_fact_producer(name, producer)
    except ValueError:
        register_fact_producer(name, producer, replace=True)


def _ternary_campaign(**over):
    _register_once("czts.phase_descriptor", _czts_phase_facts)
    teacher = ModelAdapterSpec(
        adapter_id="czts-teacher", role=ModelRole.TEACHER,
        model_family="dft_pbe_ternary",
        capabilities=["teacher.energy", "teacher.forces",
                      "teacher.applicability_domain"])
    student = ModelAdapterSpec(
        adapter_id="czts-student", role=ModelRole.STUDENT,
        model_family="mace_ternary",
        capabilities=["student.energy", "student.forces",
                      "acquisition.per_parent_augmentation_count"])
    base = dict(
        campaign_id="czts-portability-v1",
        material_tokens=list(_TERNARY_TOKENS),
        elements=["Cu", "Zn", "Sn"],
        fact_producer_name="czts.phase_descriptor",
        teacher_adapter=teacher, student_adapter=student,
        negotiated_capabilities=["teacher.energy",
                                 "acquisition.per_parent_augmentation_count"],
        decision_sha256="czts-decision-sha")
    base.update(over)
    return PortabilityCampaign(**base)


class TernaryPortabilityAuditTests(unittest.TestCase):
    def test_generality_audit_pass_on_ternary_campaign(self):
        campaign = _ternary_campaign()
        result = run_portability_audit(campaign, core_roots=("framework_v2",))
        self.assertEqual(result.verdict, "PASS",
                         f"failed checks: {[(c.check_id, c.detail) for c in result.failed_checks()]}")
        self.assertEqual(len(result.stages_audited), 12)
        # every declared check must be present and passing
        ids = {c.check_id for c in result.checks}
        self.assertIn("material_facts_are_plugin_typed", ids)
        self.assertIn("twelve_stage_unanimous_pass_with_generic_specs", ids)
        self.assertIn("every_stage_committee_l2_reproducible", ids)
        self.assertIn("generic_core_free_of_material_tokens", ids)
        self.assertTrue(all(c.passed for c in result.checks))

    def test_core_contains_no_ternary_material_tokens(self):
        campaign = _ternary_campaign()
        result = run_portability_audit(campaign, core_roots=("framework_v2",))
        scan = next(c for c in result.checks
                    if c.check_id == "generic_core_free_of_material_tokens")
        self.assertTrue(scan.passed, scan.detail)

    def test_uses_the_same_generic_specs_as_every_campaign(self):
        """No campaign-specific StageReviewSpec is passed — the audit uses the
        generic default specs, proving the review core is not rewritten."""
        from framework_v2.review_spec import default_stage_review_specs
        campaign = _ternary_campaign()
        # Explicitly hand the GENERIC specs; result must still PASS.
        result = run_portability_audit(
            campaign, specs=default_stage_review_specs(),
            core_roots=("framework_v2",))
        self.assertEqual(result.verdict, "PASS")

    def test_audit_is_not_tautological_broken_producer_fails(self):
        _register_once("czts.empty", lambda **_: [])
        campaign = _ternary_campaign(fact_producer_name="czts.empty")
        result = run_portability_audit(campaign, core_roots=("framework_v2",))
        self.assertEqual(result.verdict, "FAIL")
        failed = {c.check_id for c in result.failed_checks()}
        self.assertIn("material_facts_are_plugin_typed", failed)

    def test_audit_fails_when_material_token_would_leak_into_core(self):
        """If a campaign's material token appears in the generic core, the scan
        must fail — proving the check has teeth. We simulate a leak by declaring
        a token that genuinely exists in the core source (a generic word)."""
        campaign = _ternary_campaign(
            material_tokens=["DeterministicFact"])  # a core symbol
        result = run_portability_audit(campaign, core_roots=("framework_v2",))
        scan = next(c for c in result.checks
                    if c.check_id == "generic_core_free_of_material_tokens")
        self.assertFalse(scan.passed)
        self.assertEqual(result.verdict, "FAIL")


if __name__ == "__main__":
    unittest.main()
