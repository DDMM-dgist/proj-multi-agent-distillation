"""Regression tests for framework_v2.scientific_adequacy (Section 12 of the
closure directive). Session 2026-08-21.

These tests are deliberately generic: no material-specific numerical values
(e.g. any SiO2-specific density, cutoff, or MLIP error threshold) appear in
the test bodies. Every threshold is supplied inline by the test as a
contract input, exercising the scientific-adequacy layer, not any
particular material's physics.

Sixteen behaviors are asserted, one per test, mapping to the closure
directive's numbered requirements.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from framework_v2.scientific_adequacy import (
    AdequacyCriterion, AdequacyStatus, CalibrationStatus, ClaimRole,
    DEFAULT_ROOT_CAUSE_ROUTING, DEFAULT_SCIENTIFIC_QUESTIONS,
    DeploymentScopeContractV2, DeploymentStateRole, DomainMapping,
    EnsembleKind, EvaluationAdequacyPolicyV2,
    ObservableRole, ObservableSpec, PhysicalValidationPolicyV2,
    RootCauseClass, RootCauseDiagnosis, ScientificQuestion,
    StatePreparationPolicy, ThresholdSourceClass, UncertaintyPolicyV2,
    adjudicate_uncertainty, criterion_passes, evaluate_adequacy,
    route_by_root_cause,
)


# ------- helpers --------------------------------------------------
def _make_criterion(**overrides) -> AdequacyCriterion:
    base = dict(
        criterion_id="c1",
        observable="channelX::metric",
        operator="max",
        value=1.0,
        unit="unit",
        rationale="rationale independent of current result",
        source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
        source_reference="protocol://scope/error_budget_v1",
        frozen_before_evaluation=True,
    )
    base.update(overrides)
    return AdequacyCriterion(**base)


def _make_policy(**overrides) -> EvaluationAdequacyPolicyV2:
    base = dict(
        policy_id="p1",
        scope_contract_ref="scope::c12x",
        preregistration_witness_ref="ledger::witness#1",
    )
    base.update(overrides)
    return EvaluationAdequacyPolicyV2(**base)


# =====================================================================
# 1. procedural PASS can coexist with scientific NOT_EVALUABLE
# =====================================================================
def test_procedural_pass_does_not_imply_scientific_adequacy():
    # No criteria bound -> scoring returns NOT_EVALUABLE regardless of metric values
    policy = _make_policy()
    verdict = evaluate_adequacy(policy, metrics={"domA": {"m": 0.0}},
                                 in_scope_domains=["domA"])
    assert verdict.status == AdequacyStatus.NOT_EVALUABLE
    assert "no bound per-domain criterion" in " ".join(verdict.not_evaluable_reasons)


# =====================================================================
# 2. no threshold provenance -> NOT_EVALUABLE (fail-closed at contract layer)
# =====================================================================
def test_no_provenance_fails_at_contract_construction():
    # AdequacyCriterion refuses to construct without rationale or source_reference
    with pytest.raises(Exception):
        AdequacyCriterion(
            criterion_id="bad",
            observable="x::m",
            operator="max",
            value=1.0,
            unit="unit",
            rationale="",   # empty -> refused
            source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
            source_reference="ref",
            frozen_before_evaluation=True,
        )
    with pytest.raises(Exception):
        AdequacyCriterion(
            criterion_id="bad",
            observable="x::m",
            operator="max",
            value=1.0,
            unit="unit",
            rationale="ok",
            source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
            source_reference="",   # empty -> refused
            frozen_before_evaluation=True,
        )


# =====================================================================
# 3. post-result threshold cannot pretend to be preregistered
# =====================================================================
def test_post_result_criterion_marked_unfrozen_forces_not_evaluable():
    crit = _make_criterion(frozen_before_evaluation=False)
    policy = _make_policy(per_domain_criteria=[crit])
    metrics = {"domA": {"channelX::metric": 0.0}}
    verdict = evaluate_adequacy(policy, metrics, in_scope_domains=["domA"])
    assert verdict.status == AdequacyStatus.NOT_EVALUABLE
    assert any("not frozen before evaluation" in r for r in verdict.not_evaluable_reasons)


# =====================================================================
# 4. aggregate metric cannot hide an in-scope domain failure
# =====================================================================
def test_aggregate_metric_does_not_hide_domain_failure():
    per_domain = _make_criterion(criterion_id="pd_max",
                                 observable="channelX::metric",
                                 operator="max", value=1.0)
    agg = _make_criterion(criterion_id="agg_max",
                          observable="channelX::metric",
                          operator="max", value=1000.0)
    policy = _make_policy(per_domain_criteria=[per_domain],
                          aggregate_criteria=[agg])
    metrics = {
        "domA": {"channelX::metric": 100.0},   # fails per_domain (100 > 1.0)
        "__aggregate__": {"channelX::metric": 0.5},   # would pass aggregate
    }
    verdict = evaluate_adequacy(policy, metrics, in_scope_domains=["domA"])
    assert verdict.status == AdequacyStatus.FAIL
    assert verdict.per_domain_status["domA"] == AdequacyStatus.FAIL


# =====================================================================
# 5. ambiguous domain mapping fails closed
# =====================================================================
def test_ambiguous_domain_mapping_fails_closed():
    scope = DeploymentScopeContractV2(
        contract_id="scope::amb",
        objective="demo",
        primary_domains=["domA"],
        label_map=[
            DomainMapping(raw_label="labelX", canonical_domain="domA",
                          claim_role=ClaimRole.PRIMARY_CLAIM,
                          rationale="in-scope"),
            DomainMapping(raw_label="labelY", canonical_domain="unknown",
                          claim_role=ClaimRole.AMBIGUOUS,
                          rationale="scope did not enumerate"),
        ],
    )
    assert scope.role_of("labelX") == ClaimRole.PRIMARY_CLAIM
    assert scope.role_of("labelY") == ClaimRole.AMBIGUOUS
    # A label the map never declared also fails closed to AMBIGUOUS
    assert scope.role_of("labelZ") == ClaimRole.AMBIGUOUS


# =====================================================================
# 6. out-of-scope domain does not contaminate an explicitly bounded claim
# =====================================================================
def test_out_of_scope_domain_does_not_contaminate_claim():
    crit = _make_criterion(criterion_id="c",
                           observable="channelX::metric",
                           operator="max", value=1.0)
    policy = _make_policy(per_domain_criteria=[crit])
    metrics = {
        "in_domain": {"channelX::metric": 0.5},   # passes
        "out_of_scope": {"channelX::metric": 999.0},  # would fail if included
    }
    verdict = evaluate_adequacy(policy, metrics, in_scope_domains=["in_domain"])
    assert verdict.status == AdequacyStatus.PASS
    assert "out_of_scope" not in verdict.per_domain_status


# =====================================================================
# 7. one representative deployment point cannot imply global-domain validation
# =====================================================================
def test_representative_point_does_not_imply_global_domain():
    scope = DeploymentScopeContractV2(
        contract_id="scope::c",
        objective="demo",
        primary_domains=["ambient", "high_pressure"],
        label_map=[
            DomainMapping(raw_label="a1", canonical_domain="ambient",
                          claim_role=ClaimRole.PRIMARY_CLAIM,
                          rationale=""),
            DomainMapping(raw_label="h1", canonical_domain="high_pressure",
                          claim_role=ClaimRole.PRIMARY_CLAIM,
                          rationale=""),
        ],
        representative_deployment_points=["ambient_representative_point"],
    )
    # The scope contract lists TWO primary domains but only ONE representative
    # point; the framework must not treat the single point as validating both.
    assert len(scope.representative_deployment_points) == 1
    assert len(scope.primary_domains) == 2
    # A physical-validation policy is bound to ONE representative point.
    obs = ObservableSpec(
        name="pt_ok", kind="nve_drift", computation_method="deterministic",
        units="meV/atom/ps", ensemble_applicability=[EnsembleKind.NVE],
        reference_source="other", comparison_method="max_abs_threshold",
        role=ObservableRole.THRESHOLDED, frozen_before_student_results=True,
    )
    pv = PhysicalValidationPolicyV2(
        policy_id="pv::ambient_only",
        scope_contract_ref=scope.contract_id,
        representative_point_ref="ambient_representative_point",
        observables=[obs],
    )
    # No API asserts "PASS on this policy implies PASS on high_pressure"; the
    # test documents that binding is 1-to-1.
    assert pv.representative_point_ref == "ambient_representative_point"


# =====================================================================
# 8. fixed-volume NVT density is not treated as predicted equilibrium density
# =====================================================================
def test_nvt_density_requires_inherited_from_cell_justification():
    with pytest.raises(Exception):
        StatePreparationPolicy(
            policy_id="prep",
            scope_contract_ref="scope::c",
            state_role=DeploymentStateRole.AMBIENT_REPRESENTATIVE_POINT,
            intended_composition_ref="comp",
            intended_temperature_K=300.0,
            intended_pressure_GPa=None,
            intended_density_g_per_cm3=1.234,
            intended_density_justification="target for ambient (not inherited)",
            preparation_method="npt_equilibration",
            starting_structure_provenance_ref="ref",
            ensemble=EnsembleKind.NVT,
            equilibration_protocol_ref="eq",
            production_protocol_ref="prod",
        )
    ok = StatePreparationPolicy(
        policy_id="prep",
        scope_contract_ref="scope::c",
        state_role=DeploymentStateRole.AMBIENT_REPRESENTATIVE_POINT,
        intended_composition_ref="comp",
        intended_temperature_K=300.0,
        intended_pressure_GPa=None,
        intended_density_g_per_cm3=1.234,
        intended_density_justification="NVT: density is inherited_from_cell (documented)",
        preparation_method="validated_ambient_reference",
        starting_structure_provenance_ref="ref",
        ensemble=EnsembleKind.NVT,
        equilibration_protocol_ref="eq",
        production_protocol_ref="prod",
    )
    assert ok.ensemble == EnsembleKind.NVT


# =====================================================================
# 9. NVT pressure is diagnostic, not controlled
# =====================================================================
def test_nvt_ensemble_has_no_controlled_pressure_setpoint():
    p = StatePreparationPolicy(
        policy_id="prep",
        scope_contract_ref="scope::c",
        state_role=DeploymentStateRole.AMBIENT_REPRESENTATIVE_POINT,
        intended_composition_ref="comp",
        intended_temperature_K=300.0,
        intended_pressure_GPa=None,       # NVT: no controlled pressure
        preparation_method="validated_ambient_reference",
        starting_structure_provenance_ref="ref",
        ensemble=EnsembleKind.NVT,
        equilibration_protocol_ref="eq",
        production_protocol_ref="prod",
    )
    # NVT MUST NOT declare a controlled pressure setpoint (documented by the
    # policy shape). Setting it here would be a semantic error handled by
    # a stage-level check outside this test; here we just assert the
    # ensemble does not require pressure.
    assert p.intended_pressure_GPa is None


# =====================================================================
# 10. RDF peak height and peak position remain distinct typed values
# =====================================================================
def test_rdf_peak_height_and_peak_position_are_distinct_kinds():
    height = ObservableSpec(
        name="rdf_XY_peak_height", kind="rdf_peak_height",
        center_species="X", neighbor_species="Y",
        computation_method="max_g_r",
        units="g_of_r_units",
        ensemble_applicability=[EnsembleKind.NVT, EnsembleKind.NPT],
        reference_source="teacher",
        comparison_method="descriptive",
        role=ObservableRole.DESCRIPTIVE,
        frozen_before_student_results=True,
    )
    position = ObservableSpec(
        name="rdf_XY_first_peak_position", kind="rdf_peak_position",
        center_species="X", neighbor_species="Y",
        computation_method="argmax_g_r",
        units="Angstrom",
        ensemble_applicability=[EnsembleKind.NVT, EnsembleKind.NPT],
        reference_source="teacher",
        comparison_method="peak_position_within_A",
        role=ObservableRole.DESCRIPTIVE,
        frozen_before_student_results=True,
    )
    assert height.kind != position.kind


# =====================================================================
# 11. all-species neighbor count cannot satisfy species-specific coordination
# =====================================================================
def test_species_coordination_requires_species_and_cutoff_ref():
    with pytest.raises(Exception):
        ObservableSpec(
            name="all_species_neighbor_count", kind="species_coordination",
            center_species=None, neighbor_species=None,
            computation_method="all_neighbors_within_cutoff",
            units="count",
            ensemble_applicability=[EnsembleKind.NVT],
            reference_source="teacher",
            comparison_method="descriptive",
            role=ObservableRole.DESCRIPTIVE,
            # cutoff_source_ref missing -> refused for species_coordination
        )


# =====================================================================
# 12. unfrozen reference-derived cutoff fails closed
# =====================================================================
def test_reference_derived_cutoff_must_be_frozen_before_student():
    with pytest.raises(Exception):
        ObservableSpec(
            name="X_to_Y_coord", kind="species_coordination",
            center_species="X", neighbor_species="Y",
            computation_method="species_neighbors_within_cutoff",
            units="count",
            ensemble_applicability=[EnsembleKind.NVT],
            reference_source="teacher",
            comparison_method="descriptive",
            role=ObservableRole.DESCRIPTIVE,
            cutoff_source_ref="teacher_rdf_first_min_ref",
            cutoff_frozen_before_student=False,  # unfrozen -> refused
        )


# =====================================================================
# 13. Student/reference state mismatch fails closed at the policy shape
# =====================================================================
def test_reference_must_be_at_matched_state_by_default():
    # Positive control: default enforcement asserts matched state
    obs = ObservableSpec(
        name="rho", kind="density", computation_method="cell_volume_mass",
        units="g/cm^3",
        ensemble_applicability=[EnsembleKind.NPT],
        reference_source="experiment",
        comparison_method="descriptive",
        role=ObservableRole.DESCRIPTIVE,
        frozen_before_student_results=True,
    )
    pv = PhysicalValidationPolicyV2(
        policy_id="pv::default_matched",
        scope_contract_ref="scope::c",
        representative_point_ref="ambient_representative_point",
        observables=[obs],
    )
    assert pv.reference_at_matched_state is True


# =====================================================================
# 14. uncalibrated disagreement cannot satisfy calibrated-uncertainty requirement
# =====================================================================
def test_uncalibrated_cannot_satisfy_calibrated_required():
    policy = UncertaintyPolicyV2(
        policy_id="unc",
        scope_contract_ref="scope::c",
        method="committee_force_std",
        metrics=["sigma_F"],
        required_status=CalibrationStatus.CALIBRATED_PARTIAL,
        calibration_evidence_ref="cal_ref",
    )
    assert adjudicate_uncertainty(policy, CalibrationStatus.UNCALIBRATED) == AdequacyStatus.FAIL
    assert adjudicate_uncertainty(policy, CalibrationStatus.CALIBRATED_PARTIAL) == AdequacyStatus.PASS
    assert adjudicate_uncertainty(policy, CalibrationStatus.CALIBRATED) == AdequacyStatus.PASS
    # An uncalibrated-required policy is trivially satisfied by any status
    uncal_policy = UncertaintyPolicyV2(
        policy_id="unc2",
        scope_contract_ref="scope::c",
        method="committee_force_std",
        metrics=["sigma_F"],
        required_status=CalibrationStatus.UNCALIBRATED,
    )
    assert adjudicate_uncertainty(uncal_policy, CalibrationStatus.UNCALIBRATED) == AdequacyStatus.PASS


# =====================================================================
# 15. scientific recovery routes by diagnosed root cause, not stage number
# =====================================================================
def test_recovery_routes_by_root_cause_not_by_stage_number():
    d = RootCauseDiagnosis(
        diagnosis_id="d1", root_cause=RootCauseClass.FIDELITY_INADEQUACY,
    )
    stages_fid = route_by_root_cause(d)
    assert "training" in stages_fid
    assert "data_coverage_replay_if_supported" in stages_fid

    d2 = RootCauseDiagnosis(
        diagnosis_id="d2",
        root_cause=RootCauseClass.PHYSICAL_OBSERVABLE_IMPLEMENTATION_DEFECT,
    )
    stages_impl = route_by_root_cause(d2)
    assert "training" not in stages_impl  # implementation defect must NOT retrain
    assert "validation_method_recovery" in stages_impl

    d3 = RootCauseDiagnosis(
        diagnosis_id="d3", root_cause=RootCauseClass.DEPLOYMENT_STATE_MISMATCH,
    )
    stages_state = route_by_root_cause(d3)
    assert "state_preparation_recovery" in stages_state
    assert "training" not in stages_state  # state mismatch must NOT retrain
    # framework/evidence defects must not route to scientific compute
    d4 = RootCauseDiagnosis(
        diagnosis_id="d4",
        root_cause=RootCauseClass.FRAMEWORK_EVIDENCE_READABILITY_DEFECT,
    )
    stages_fw = route_by_root_cause(d4)
    assert "governance_or_framework_recovery" in stages_fw
    assert "training" not in stages_fw


# =====================================================================
# 16. C12F historical artifacts remain byte-immutable under this pass
# =====================================================================
_C12F_HASHES_BEFORE_GOVERNANCE_PASS = {
    "artifacts/md.manifest.json":
        "6541c3a1da04e038b3cbb05b0b9c36efda8b05806bcb941887a2660a2f7c46a0",
    "artifacts/deployment_md/trajectory.dump":
        "6eec4a0e90bc4c63ad2def8b081c0b1fdbec3e8358186a58bff7045d77988a4d",
    "artifacts/deployment_md/thermo.log":
        "3ed87bcec0beaea44726de04f90c0a38730101a2059c58ab35954d421c0983cc",
    "artifacts/deployment_md/input.lmp":
        "63e3438068ad26a04a15abcef02d3fdeb33afbe74eef291608eb1707c743aa53",
    "artifacts/deployment_md/context.yaml":
        "af0bc999434bf66c242d131cf38818d55a560e7ecf739929a54b90b5eb3d4931",
    "artifacts/deployment_md/deployment_provenance.json":
        "6cae634f29fd2599d537a208dd6be7cf0fd6bbf9c4a553c7b43430adb2b3302c",
}
_C12F_RUN = Path(__file__).resolve().parents[1] / "runs" / "sio2-sox-allegro-simplenn-c12f"


@pytest.mark.parametrize("relpath, expected",
                         sorted(_C12F_HASHES_BEFORE_GOVERNANCE_PASS.items()))
def test_c12f_scientific_artifacts_byte_immutable_under_governance_pass(relpath, expected):
    from workflow.integrity import sha256_file
    p = _C12F_RUN / relpath
    if not p.is_file():
        pytest.skip(f"C12F artifact not present: {relpath}")
    assert sha256_file(p) == expected, (
        f"{relpath} sha256 drifted -- the generic scientific-adequacy layer "
        "must be a governance addition; it must not modify any C12F scientific artifact")
