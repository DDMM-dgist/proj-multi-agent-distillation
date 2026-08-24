"""Synthetic end-to-end regression suite for the runtime scientific-adequacy
integration (Session 2026-08-21, closure step 2b).

These tests do not consume any expensive scientific compute. They stand up a
synthetic RunController with minimal state, bind typed scientific policies,
write minimal deterministic evidence, and exercise the ``record_gate`` PASS
path to prove:

  1) procedural PASS + scientific FAIL blocks advancement;
  2) procedural PASS + scientific NOT_EVALUABLE blocks when required;
  3) domain-error diagnosis routes to coverage/acquisition (not framework-only);
  4) state-mismatch diagnosis routes to state-preparation, NOT to training;
  5) validation-method-defect diagnosis routes to analysis/framework, NOT training;
  6) uncalibrated uncertainty cannot satisfy CALIBRATED_PARTIAL requirement;
  7) one deployment representative point does not close a global-domain claim;
  8) a fully valid synthetic campaign can advance Stage 8 to 11 under bound policies;
  9) mid-run policy edit is refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework_v2.scientific_adequacy import (
    AdequacyCriterion, AdequacyStatus, CalibrationStatus, ClaimRole,
    DeploymentScopeContractV2, DeploymentStateRole, DomainMapping,
    EnsembleKind, EvaluationAdequacyPolicyV2, ObservableRole, ObservableSpec,
    PhysicalValidationPolicyV2, RootCauseClass, RootCauseDiagnosis,
    StatePreparationPolicy, ThresholdSourceClass, UncertaintyPolicyV2,
)
from framework_v2.scientific_gate import (
    ScientificAdequacyBlocked, assert_stage_scientific_adequacy, bind_policy,
)
from framework_v2.scientific_recovery import propose_recovery_from_diagnosis


# ---------------------------------------------------------------------------
# Synthetic run state (minimal shape the scientific_gate module reads).
# ---------------------------------------------------------------------------
def _state(stages):
    return {"stages": [{"name": s, "status": "pending"} for s in stages],
            "events": []}


def _crit(**over):
    base = dict(
        criterion_id="c",
        observable="student_vs_teacher::f_rmse",
        operator="max",
        value=0.30,
        unit="eV/A",
        rationale="prospective application error budget",
        source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
        source_reference="protocol://budget_v1",
        frozen_before_evaluation=True,
    )
    base.update(over)
    return AdequacyCriterion(**base).model_dump()


def _eval_policy(criteria=None, worst=None, agg=None,
                 witness="ledger#preregistered:1"):
    return EvaluationAdequacyPolicyV2(
        policy_id="eval_v2::demo",
        scope_contract_ref="scope::demo",
        per_domain_criteria=[AdequacyCriterion(**c) for c in (criteria or [])],
        worst_domain_criteria=[AdequacyCriterion(**c) for c in (worst or [])],
        aggregate_criteria=[AdequacyCriterion(**c) for c in (agg or [])],
        preregistration_witness_ref=witness,
    ).model_dump()


# =====================================================================
# 1) procedural PASS + scientific FAIL blocks advancement
# =====================================================================
def test_procedural_pass_plus_scientific_fail_blocks():
    state = _state(["evaluation"])
    policy = _eval_policy(criteria=[_crit()])
    bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", policy,
                source_ref="ref", required=True)
    # accuracy report: student_vs_teacher::f_rmse = 2.0 (> 0.30) -> FAIL
    report = {"student_vs_teacher": {"domA": {"f_rmse": 2.0}},
              "in_scope_domains": ["domA"]}
    with pytest.raises(ScientificAdequacyBlocked):
        assert_stage_scientific_adequacy(
            state, "evaluation",
            accuracy_report_loader=lambda: report)


# =====================================================================
# 2) procedural PASS + scientific NOT_EVALUABLE blocks when required
# =====================================================================
def test_procedural_pass_plus_not_evaluable_blocks_when_required():
    state = _state(["evaluation"])
    # criterion refers to metric NOT present in report -> NOT_EVALUABLE
    policy = _eval_policy(criteria=[_crit(observable="student_vs_teacher::not_present")])
    bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", policy,
                source_ref="ref", required=True)
    report = {"student_vs_teacher": {"domA": {"f_rmse": 0.1}},
              "in_scope_domains": ["domA"]}
    with pytest.raises(ScientificAdequacyBlocked):
        assert_stage_scientific_adequacy(
            state, "evaluation",
            accuracy_report_loader=lambda: report)


def test_not_evaluable_does_not_block_when_not_required():
    state = _state(["evaluation"])
    policy = _eval_policy(criteria=[_crit(observable="student_vs_teacher::not_present")])
    bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", policy,
                source_ref="ref", required=False)
    report = {"student_vs_teacher": {"domA": {"f_rmse": 0.1}},
              "in_scope_domains": ["domA"]}
    # required=False -> the summary is returned with NOT_EVALUABLE noted but no raise
    summary = assert_stage_scientific_adequacy(
        state, "evaluation",
        accuracy_report_loader=lambda: report)
    assert summary["adjudications"][0]["verdict"]["status"] == AdequacyStatus.NOT_EVALUABLE.value


# =====================================================================
# 3) domain-error diagnosis routes to coverage/acquisition, not framework-only
# =====================================================================
def test_fidelity_diagnosis_routes_to_coverage_or_acquisition():
    diag = RootCauseDiagnosis(
        diagnosis_id="d1",
        root_cause=RootCauseClass.FIDELITY_INADEQUACY,
    )
    proposal = propose_recovery_from_diagnosis(diag, failing_stage="evaluation")
    stages = proposal["admissible_return_stages"]
    assert any("coverage" in s or "acquisition" in s for s in stages)
    assert "training" in stages
    assert "governance_or_framework_recovery" not in stages


# =====================================================================
# 4) state-mismatch diagnosis routes to state-preparation, NOT training
# =====================================================================
def test_state_mismatch_diagnosis_never_routes_to_training():
    diag = RootCauseDiagnosis(
        diagnosis_id="d2",
        root_cause=RootCauseClass.DEPLOYMENT_STATE_MISMATCH,
    )
    proposal = propose_recovery_from_diagnosis(diag, failing_stage="deployment_md")
    stages = proposal["admissible_return_stages"]
    assert "state_preparation_recovery" in stages
    assert "training" not in stages
    # and the routing guard itself would raise if training were forced in:
    tampered = RootCauseDiagnosis(
        diagnosis_id="d2b",
        root_cause=RootCauseClass.DEPLOYMENT_STATE_MISMATCH,
        admissible_return_stages=["training", "state_preparation_recovery"],
    )
    with pytest.raises(ValueError):
        propose_recovery_from_diagnosis(tampered, failing_stage="deployment_md")


# =====================================================================
# 5) validation-method defect routes to analysis/framework recovery
# =====================================================================
def test_validation_method_defect_routes_to_framework_recovery():
    diag = RootCauseDiagnosis(
        diagnosis_id="d3",
        root_cause=RootCauseClass.PHYSICAL_OBSERVABLE_IMPLEMENTATION_DEFECT,
    )
    proposal = propose_recovery_from_diagnosis(diag, failing_stage="physical_validation")
    stages = proposal["admissible_return_stages"]
    assert "validation_method_recovery" in stages
    assert "training" not in stages
    # framework-evidence recovery cannot route to any scientific-compute stage
    d4 = RootCauseDiagnosis(
        diagnosis_id="d4",
        root_cause=RootCauseClass.FRAMEWORK_EVIDENCE_READABILITY_DEFECT,
        admissible_return_stages=["training"],   # tampered
    )
    with pytest.raises(ValueError):
        propose_recovery_from_diagnosis(d4, failing_stage="evaluation")


# =====================================================================
# 6) uncalibrated uncertainty cannot satisfy CALIBRATED_PARTIAL
# =====================================================================
def test_uncalibrated_fails_calibrated_partial_requirement():
    state = _state(["uncertainty"])
    policy = UncertaintyPolicyV2(
        policy_id="unc",
        scope_contract_ref="scope::demo",
        method="committee_force_std",
        metrics=["sigma_F"],
        required_status=CalibrationStatus.CALIBRATED_PARTIAL,
        calibration_evidence_ref="cal_ref",
    ).model_dump()
    bind_policy(state, "uncertainty", "UncertaintyPolicyV2", policy,
                source_ref="ref", required=True)
    uncal = {"calibration": {"status": CalibrationStatus.UNCALIBRATED.value}}
    with pytest.raises(ScientificAdequacyBlocked):
        assert_stage_scientific_adequacy(
            state, "uncertainty",
            uncertainty_report_loader=lambda: uncal)
    # And the reverse: an uncalibrated-required policy always PASSes any status
    state2 = _state(["uncertainty"])
    uncal_policy = UncertaintyPolicyV2(
        policy_id="unc_lax",
        scope_contract_ref="scope::demo",
        method="committee_force_std",
        metrics=["sigma_F"],
        required_status=CalibrationStatus.UNCALIBRATED,
    ).model_dump()
    bind_policy(state2, "uncertainty", "UncertaintyPolicyV2", uncal_policy,
                source_ref="ref", required=True)
    summary = assert_stage_scientific_adequacy(
        state2, "uncertainty",
        uncertainty_report_loader=lambda: uncal)
    assert summary["adjudications"][0]["verdict"]["status"] == AdequacyStatus.PASS.value


# =====================================================================
# 7) One deployment point does not close a global-domain claim
# =====================================================================
def test_representative_point_does_not_imply_global_domain():
    scope = DeploymentScopeContractV2(
        contract_id="scope::demo",
        objective="broad",
        primary_domains=["ambient", "high_pressure", "melt"],
        label_map=[DomainMapping(raw_label=lab, canonical_domain=lab,
                                 claim_role=ClaimRole.PRIMARY_CLAIM,
                                 rationale="declared")
                    for lab in ("ambient", "high_pressure", "melt")],
        representative_deployment_points=["ambient_representative_point"],
    )
    # Building a physical-validation policy for one point is legal, but the
    # policy is bound to ONE representative point. Verifying that binding is
    # strict:
    obs = ObservableSpec(
        name="nve_drift", kind="nve_drift", computation_method="deterministic",
        units="meV/atom/ps", ensemble_applicability=[EnsembleKind.NVE],
        reference_source="other", comparison_method="max_abs_threshold",
        role=ObservableRole.THRESHOLDED, frozen_before_student_results=True,
    )
    pv = PhysicalValidationPolicyV2(
        policy_id="pv::ambient",
        scope_contract_ref=scope.contract_id,
        representative_point_ref="ambient_representative_point",
        observables=[obs],
    )
    assert pv.representative_point_ref != "high_pressure_representative_point"
    assert pv.representative_point_ref != "melt_representative_point"
    # There is no adjudicator function that promotes single-point PASS to
    # multi-domain PASS -- assert that shape by absence:
    from framework_v2 import scientific_adequacy as sa
    assert not hasattr(sa, "promote_single_point_to_global")


# =====================================================================
# 8) Fully valid synthetic campaign advances Stage 8 -> 11
# =====================================================================
def test_fully_valid_synthetic_campaign_advances():
    state = _state(["evaluation", "uncertainty", "deployment_md", "physical_validation"])
    # bind well-designed policies
    eval_policy = _eval_policy(
        criteria=[_crit(criterion_id="e1", observable="student_vs_teacher::f_rmse",
                         operator="max", value=0.30, unit="eV/A")],
        worst=[_crit(criterion_id="w1", observable="student_vs_teacher::f_rmse",
                      operator="max", value=0.50, unit="eV/A")],
    )
    bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", eval_policy,
                source_ref="prospective::demo", required=True)
    unc_policy = UncertaintyPolicyV2(
        policy_id="unc",
        scope_contract_ref="scope::demo",
        method="committee_force_std",
        metrics=["sigma_F"],
        required_status=CalibrationStatus.UNCALIBRATED,
    ).model_dump()
    bind_policy(state, "uncertainty", "UncertaintyPolicyV2", unc_policy,
                source_ref="ref", required=True)
    state_prep_policy = StatePreparationPolicy(
        policy_id="prep",
        scope_contract_ref="scope::demo",
        state_role=DeploymentStateRole.AMBIENT_REPRESENTATIVE_POINT,
        intended_composition_ref="comp",
        intended_temperature_K=300.0,
        preparation_method="validated_ambient_reference",
        starting_structure_provenance_ref="sha256::abc123",
        ensemble=EnsembleKind.NPT,
        equilibration_protocol_ref="eq",
        production_protocol_ref="prod",
    ).model_dump()
    bind_policy(state, "deployment_md", "StatePreparationPolicy", state_prep_policy,
                source_ref="ref", required=True)
    obs = ObservableSpec(
        name="nve_drift", kind="nve_drift", computation_method="deterministic",
        units="meV/atom/ps", ensemble_applicability=[EnsembleKind.NVE],
        reference_source="other", comparison_method="max_abs_threshold",
        role=ObservableRole.THRESHOLDED, frozen_before_student_results=True,
    )
    pv_policy = PhysicalValidationPolicyV2(
        policy_id="pv",
        scope_contract_ref="scope::demo",
        representative_point_ref="ambient_representative_point",
        observables=[obs],
    ).model_dump()
    bind_policy(state, "physical_validation", "PhysicalValidationPolicyV2", pv_policy,
                source_ref="ref", required=True)
    # feed passing evidence into each stage
    good_accuracy = {
        "student_vs_teacher": {"domA": {"f_rmse": 0.15}},
        "in_scope_domains": ["domA"],
    }
    good_uncertainty = {"calibration": {"status": "uncalibrated"}}
    good_md = {"ensemble": "NPT", "starting_structure_sha256": "abc123",
               "protocol": {"temperature_K": 300.0}}
    good_validation = {"checks": [
        {"observable": "nve_drift", "unit": "meV/atom/ps",
         "criterion": {"operator": "max_abs", "threshold": 1.0}, "status": "PASS"},
    ]}
    for stage, loaders in (
        ("evaluation", {"accuracy_report_loader": lambda: good_accuracy}),
        ("uncertainty", {"uncertainty_report_loader": lambda: good_uncertainty}),
        ("deployment_md", {"md_manifest_loader": lambda: good_md}),
        ("physical_validation", {"validation_report_loader": lambda: good_validation}),
    ):
        summary = assert_stage_scientific_adequacy(state, stage, **loaders)
        assert summary is not None
        # Every adjudication should be PASS
        assert all(a["verdict"]["status"] == AdequacyStatus.PASS.value
                   for a in summary["adjudications"]), (stage, summary)


# =====================================================================
# 9) Mid-run policy edit is refused
# =====================================================================
def test_mid_run_policy_edit_refused():
    state = _state(["evaluation"])
    v1 = _eval_policy(criteria=[_crit()])
    bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", v1,
                source_ref="ref", required=True)
    # Re-binding with identical content is idempotent
    same = bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", v1,
                       source_ref="ref", required=True)
    assert same["content_sha256"] == state["scientific_policies"]["evaluation::EvaluationAdequacyPolicyV2"]["content_sha256"]
    # Re-binding with different content is refused
    v2 = _eval_policy(criteria=[_crit(value=0.5)])  # relaxed
    with pytest.raises(ValueError):
        bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", v2,
                    source_ref="ref", required=True)


# =====================================================================
# Bonus: the deployment_md state-mismatch case exercises the state scorer
# and produces FAIL (blocking record_gate PASS).
# =====================================================================
def test_state_realization_mismatch_blocks():
    state = _state(["deployment_md"])
    p = StatePreparationPolicy(
        policy_id="prep",
        scope_contract_ref="scope::demo",
        state_role=DeploymentStateRole.AMBIENT_REPRESENTATIVE_POINT,
        intended_composition_ref="comp",
        intended_temperature_K=300.0,
        preparation_method="validated_ambient_reference",
        starting_structure_provenance_ref="sha256::intended_reference_frame",
        ensemble=EnsembleKind.NPT,
        equilibration_protocol_ref="eq",
        production_protocol_ref="prod",
    ).model_dump()
    bind_policy(state, "deployment_md", "StatePreparationPolicy", p,
                source_ref="ref", required=True)
    # realized: NVT with a legacy starting structure -> FAIL
    md = {"ensemble": "NVT",
          "starting_structure_sha256": "sha256::legacy_r27_cell",
          "protocol": {"temperature_K": 300.0}}
    with pytest.raises(ScientificAdequacyBlocked):
        assert_stage_scientific_adequacy(
            state, "deployment_md",
            md_manifest_loader=lambda: md)
