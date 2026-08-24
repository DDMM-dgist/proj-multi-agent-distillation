"""Final-closure regression suite covering Blockers 1-5 and the ledger.

Session 2026-08-21 final closure pass. Uses the SAME authoritative
scientific_gate / scientific_recovery / scientific_stage_entry choke points
that the Controller uses. No expensive scientific compute; no external APIs.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import numpy as np


# =====================================================================
# Blocker 1 close — typed physical observables
# =====================================================================
class _MiniAtoms:
    """Minimal ASE-like atoms fixture: enough for compute_species_coordination
    (which calls get_all_distances(mic=True) and get_chemical_symbols) and for
    compute_density (which calls get_masses().sum() and get_volume())."""
    def __init__(self, symbols, positions, cell=None, masses=None):
        self.symbols = list(symbols)
        self.pos = np.asarray(positions, dtype=float)
        self.cell_matrix = np.asarray(cell, dtype=float) if cell is not None else np.eye(3) * 100
        self.mass_arr = np.asarray(masses, dtype=float) if masses is not None else np.ones(len(symbols))

    def get_chemical_symbols(self):
        return list(self.symbols)

    def get_all_distances(self, mic=True):
        n = len(self.symbols)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                delta = self.pos[j] - self.pos[i]
                if mic:
                    # simple orthorhombic MIC using diagonal-only cell for the mini fixture
                    box = np.diag(self.cell_matrix)
                    delta = delta - box * np.round(delta / box)
                d[i, j] = np.linalg.norm(delta)
        return d

    def get_masses(self):
        return self.mass_arr

    def get_volume(self):
        return float(np.linalg.det(self.cell_matrix))

    def __len__(self):
        return len(self.symbols)


def test_compute_species_coordination_refuses_missing_cutoff():
    from validation.structure_dynamics import compute_species_coordination
    frame = _MiniAtoms(["Si", "O", "O"], [[0,0,0],[1.6,0,0],[0,1.6,0]])
    with pytest.raises(ValueError):
        compute_species_coordination([frame], "Si", "O", cutoff_A=None)
    with pytest.raises(ValueError):
        compute_species_coordination([frame], "Si", "O", cutoff_A=0.0)


def test_compute_species_coordination_emits_histogram_and_provenance():
    from validation.structure_dynamics import compute_species_coordination
    # 1 Si with 4 O neighbors within 2.0 A; no O-O within 2.0 A
    frame = _MiniAtoms(
        ["Si", "O", "O", "O", "O"],
        [[0,0,0], [1.6,0,0], [-1.6,0,0], [0,1.6,0], [0,-1.6,0]])
    result = compute_species_coordination(
        [frame], "Si", "O", cutoff_A=2.0,
        cutoff_source_ref="teacher_rdf_first_min",
        cutoff_frozen_before_student=True)
    assert result["cutoff_A"] == 2.0
    assert result["cutoff_source_ref"] == "teacher_rdf_first_min"
    assert result["cutoff_frozen_before_student"] is True
    assert result["aggregate_mean_coordination"] == 4.0
    # histogram: one Si with count=4
    assert result["coordination_histogram"][4] == 1
    assert result["coordination_fractions"] == {4: 1.0}


def test_rdf_first_peak_and_minimum_deterministic():
    from validation.structure_dynamics import rdf_first_peak_and_minimum
    r = np.linspace(0.05, 6.0, 200)
    # synthesize a smooth g(r) with a peak at r=1.6 and a minimum at r=3.0
    g = 1.0 + 3.0 * np.exp(-((r - 1.6) / 0.15) ** 2) - 0.5 * np.exp(-((r - 3.0) / 0.4) ** 2)
    out = rdf_first_peak_and_minimum(r.tolist(), g.tolist(), smoothing_window=5)
    assert abs(out["r_first_peak_A"] - 1.6) < 0.05
    assert abs(out["r_first_min_A"] - 3.0) < 0.10
    assert out["smoothing_window_bins"] == 5


def test_rdf_first_peak_smoothing_window_must_be_odd():
    from validation.structure_dynamics import rdf_first_peak_and_minimum
    r = np.linspace(0.05, 6.0, 200); g = np.ones(200)
    with pytest.raises(ValueError):
        rdf_first_peak_and_minimum(r.tolist(), g.tolist(), smoothing_window=4)


# =====================================================================
# Blocker 3 close — stage-entry policy requirement
# =====================================================================
def test_stage_entry_assertion_opt_in_default_off():
    from framework_v2.scientific_stage_entry import (
        assert_stage_entry_policies_bound, ScientificPolicyMissingAtStageEntry)
    state_off = {"scientific_policies": {}}
    # opt-in flag NOT set -> assertion is a no-op (C12F backward compat)
    assert_stage_entry_policies_bound(state_off, "evaluation")


def test_stage_entry_assertion_blocks_missing_policy_when_opted_in():
    from framework_v2.scientific_stage_entry import (
        assert_stage_entry_policies_bound, ScientificPolicyMissingAtStageEntry)
    state = {"scientific_stage_entry_enforcement": True,
             "scientific_policies": {}}
    with pytest.raises(ScientificPolicyMissingAtStageEntry):
        assert_stage_entry_policies_bound(state, "evaluation")
    with pytest.raises(ScientificPolicyMissingAtStageEntry):
        assert_stage_entry_policies_bound(state, "physical_validation")


def test_stage_entry_assertion_passes_when_all_required_policies_bound():
    from framework_v2.scientific_stage_entry import assert_stage_entry_policies_bound
    state = {
        "scientific_stage_entry_enforcement": True,
        "scientific_policies": {
            "evaluation::EvaluationAdequacyPolicyV2": {"kind": "X", "required": True},
        },
    }
    assert_stage_entry_policies_bound(state, "evaluation")  # no raise


def test_stage_entry_assertion_refuses_required_false_binding():
    from framework_v2.scientific_stage_entry import (
        assert_stage_entry_policies_bound, ScientificPolicyMissingAtStageEntry)
    state = {
        "scientific_stage_entry_enforcement": True,
        "scientific_policies": {
            "evaluation::EvaluationAdequacyPolicyV2": {"kind": "X", "required": False},
        },
    }
    with pytest.raises(ScientificPolicyMissingAtStageEntry):
        assert_stage_entry_policies_bound(state, "evaluation")


# =====================================================================
# Blocker 2 partial close — judge-packet extension helper
# =====================================================================
def _seed_state_with_eval_policy():
    from framework_v2.scientific_gate import bind_policy
    from framework_v2.scientific_adequacy import (
        AdequacyCriterion, EvaluationAdequacyPolicyV2, ThresholdSourceClass)
    state = {"stages": [{"name": "evaluation", "status": "pending"}], "events": []}
    policy = EvaluationAdequacyPolicyV2(
        policy_id="p", scope_contract_ref="s",
        preregistration_witness_ref="w",
        per_domain_criteria=[AdequacyCriterion(
            criterion_id="c", observable="student_vs_teacher::f_rmse",
            operator="max", value=0.3, unit="eV/A",
            rationale="synthetic", source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
            source_reference="synth", frozen_before_evaluation=True)],
    ).model_dump()
    bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", policy,
                source_ref="synth", required=True)
    return state


def test_judge_packet_extension_produces_scientific_block():
    from framework_v2.judge_packet_extension import build_scientific_extension_block
    state = _seed_state_with_eval_policy()
    block = build_scientific_extension_block(state, "evaluation",
                                              observed_evidence_values={"student_vs_teacher::f_rmse": 0.15})
    assert block["scientific_layer_active"] is True
    assert "PROCEDURAL_VALIDITY" in block["layer_separation_note"]
    assert "SCIENTIFIC_ADEQUACY" in block["layer_separation_note"]
    assert len(block["bound_policies"]) == 1
    bp = block["bound_policies"][0]
    assert bp["kind"] == "EvaluationAdequacyPolicyV2"
    assert bp["content_sha256"]  # non-empty
    assert bp["preregistration_witness_ref"] == "w"
    assert bp["criteria_summary"][0]["criterion_id"] == "c"


def test_judge_packet_extension_identical_across_three_judges():
    """Three mutually-blind Judges must see identical frozen policy content."""
    from framework_v2.judge_packet_extension import build_scientific_extension_block
    state = _seed_state_with_eval_policy()
    blocks = [build_scientific_extension_block(state, "evaluation",
                                                observed_evidence_values={"student_vs_teacher::f_rmse": 0.15})
              for _ in range(3)]
    hashes = [b["bound_policies"][0]["content_sha256"] for b in blocks]
    assert hashes[0] == hashes[1] == hashes[2]


def test_judge_packet_extension_policy_version_change_yields_new_hash():
    """Rebinding with different content is refused; but a NEW run-init with a
    different policy content produces a distinct hash (prospective-only)."""
    from framework_v2.scientific_gate import bind_policy
    from framework_v2.scientific_adequacy import (
        AdequacyCriterion, EvaluationAdequacyPolicyV2, ThresholdSourceClass)
    from framework_v2.judge_packet_extension import build_scientific_extension_block

    def make_state(threshold_value):
        state = {"stages": [{"name": "evaluation", "status": "pending"}], "events": []}
        policy = EvaluationAdequacyPolicyV2(
            policy_id="p", scope_contract_ref="s",
            preregistration_witness_ref="w",
            per_domain_criteria=[AdequacyCriterion(
                criterion_id="c", observable="student_vs_teacher::f_rmse",
                operator="max", value=threshold_value, unit="eV/A",
                rationale="synthetic", source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
                source_reference="synth", frozen_before_evaluation=True)],
        ).model_dump()
        bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", policy,
                    source_ref="synth", required=True)
        return state

    b1 = build_scientific_extension_block(make_state(0.3), "evaluation")
    b2 = build_scientific_extension_block(make_state(0.5), "evaluation")
    assert b1["bound_policies"][0]["content_sha256"] != b2["bound_policies"][0]["content_sha256"]


def test_judge_packet_extension_is_noop_when_no_policy():
    from framework_v2.judge_packet_extension import build_scientific_extension_block
    state = {"stages": [{"name": "evaluation", "status": "pending"}], "events": []}
    block = build_scientific_extension_block(state, "evaluation")
    assert block["scientific_layer_active"] is False


# =====================================================================
# Blocker 4 close — canonical scientific routing is authoritative
# =====================================================================
def test_canonical_router_is_the_authoritative_source():
    """The scientific-recovery router is the single source of truth for
    scientific stages; workflow.recovery_taxonomy remains importable but is
    not permitted to contradict it."""
    from framework_v2.scientific_recovery import propose_recovery_from_diagnosis
    from framework_v2.scientific_adequacy import RootCauseClass, RootCauseDiagnosis
    d = RootCauseDiagnosis(diagnosis_id="d", root_cause=RootCauseClass.FIDELITY_INADEQUACY)
    p = propose_recovery_from_diagnosis(d, failing_stage="evaluation")
    assert "training" in p["admissible_return_stages"]
    # recovery_taxonomy still importable but no longer authoritative for scientific stages
    import workflow.recovery_taxonomy as tax  # noqa: F401


def test_canonical_router_never_routes_state_defect_to_training():
    from framework_v2.scientific_recovery import propose_recovery_from_diagnosis
    from framework_v2.scientific_adequacy import RootCauseClass, RootCauseDiagnosis
    for cause in (RootCauseClass.DEPLOYMENT_STATE_MISMATCH,
                  RootCauseClass.PHYSICAL_OBSERVABLE_IMPLEMENTATION_DEFECT):
        d = RootCauseDiagnosis(diagnosis_id="d", root_cause=cause)
        p = propose_recovery_from_diagnosis(d, failing_stage="s")
        assert "training" not in p["admissible_return_stages"], cause


# =====================================================================
# Blocker 5 partial close — synthetic Paths A/B/C/D/E/F/G via authoritative
# scoring choke point. (True Controller-lifecycle E2E remains a separate
# integration; this suite exercises the same choke functions the Controller
# calls.)
# =====================================================================
def _fully_bind(state):
    from framework_v2.scientific_gate import bind_policy
    from framework_v2.scientific_adequacy import (
        AdequacyCriterion, CalibrationStatus, DeploymentStateRole,
        EnsembleKind, EvaluationAdequacyPolicyV2, ObservableRole, ObservableSpec,
        PhysicalValidationPolicyV2, StatePreparationPolicy, ThresholdSourceClass,
        UncertaintyPolicyV2, DeploymentScopeContractV2, DomainMapping, ClaimRole)
    scope = DeploymentScopeContractV2(
        contract_id="s", objective="o",
        primary_domains=["dA"],
        label_map=[DomainMapping(raw_label="dA", canonical_domain="dA",
                                  claim_role=ClaimRole.PRIMARY_CLAIM, rationale="")],
        representative_deployment_points=["pt"]
    ).model_dump()
    bind_policy(state, "deployment_md", "DeploymentScopeContractV2", scope,
                source_ref="synth", required=True)
    ep = EvaluationAdequacyPolicyV2(
        policy_id="e", scope_contract_ref="s", preregistration_witness_ref="w",
        per_domain_criteria=[AdequacyCriterion(
            criterion_id="c", observable="student_vs_teacher::f_rmse",
            operator="max", value=0.3, unit="eV/A", rationale="synth",
            source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
            source_reference="synth", frozen_before_evaluation=True)],
    ).model_dump()
    bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", ep,
                source_ref="synth", required=True)
    up = UncertaintyPolicyV2(policy_id="u", scope_contract_ref="s",
                             method="m", metrics=["m"],
                             required_status=CalibrationStatus.UNCALIBRATED).model_dump()
    bind_policy(state, "uncertainty", "UncertaintyPolicyV2", up,
                source_ref="synth", required=True)
    sp = StatePreparationPolicy(
        policy_id="p", scope_contract_ref="s",
        state_role=DeploymentStateRole.AMBIENT_REPRESENTATIVE_POINT,
        intended_composition_ref="c", intended_temperature_K=300.0,
        preparation_method="validated_ambient_reference",
        starting_structure_provenance_ref="sha::intended",
        ensemble=EnsembleKind.NPT,
        equilibration_protocol_ref="eq", production_protocol_ref="prod",
    ).model_dump()
    bind_policy(state, "deployment_md", "StatePreparationPolicy", sp,
                source_ref="synth", required=True)
    obs = ObservableSpec(name="nve_drift", kind="nve_drift",
                          computation_method="linear_fit",
                          units="meV/atom/ps",
                          ensemble_applicability=[EnsembleKind.NVE],
                          reference_source="other",
                          comparison_method="max_abs_threshold",
                          role=ObservableRole.THRESHOLDED,
                          frozen_before_student_results=True)
    pv = PhysicalValidationPolicyV2(policy_id="pv", scope_contract_ref="s",
                                     representative_point_ref="pt",
                                     observables=[obs]).model_dump()
    bind_policy(state, "physical_validation", "PhysicalValidationPolicyV2", pv,
                source_ref="synth", required=True)


def _mkstate():
    return {"stages": [{"name": s, "status": "pending"} for s in
                       ("evaluation", "uncertainty", "deployment_md", "physical_validation")],
            "events": [], "scientific_stage_entry_enforcement": True}


def test_path_F_stage_entry_missing_policy_blocks():
    from framework_v2.scientific_stage_entry import (
        assert_stage_entry_policies_bound, ScientificPolicyMissingAtStageEntry)
    state = _mkstate()
    # No binding done -> stage entry must block
    with pytest.raises(ScientificPolicyMissingAtStageEntry):
        assert_stage_entry_policies_bound(state, "evaluation")
    _fully_bind(state)
    # Now stage entry passes
    assert_stage_entry_policies_bound(state, "evaluation")
    assert_stage_entry_policies_bound(state, "uncertainty")
    assert_stage_entry_policies_bound(state, "deployment_md")
    assert_stage_entry_policies_bound(state, "physical_validation")


def test_path_G_mid_run_policy_mutation_refused():
    from framework_v2.scientific_gate import bind_policy
    from framework_v2.scientific_adequacy import (
        AdequacyCriterion, EvaluationAdequacyPolicyV2, ThresholdSourceClass)
    state = _mkstate()
    _fully_bind(state)
    hash1 = state["scientific_policies"]["evaluation::EvaluationAdequacyPolicyV2"]["content_sha256"]
    p2 = EvaluationAdequacyPolicyV2(
        policy_id="e", scope_contract_ref="s", preregistration_witness_ref="w",
        per_domain_criteria=[AdequacyCriterion(
            criterion_id="c", observable="student_vs_teacher::f_rmse",
            operator="max", value=1.5,   # relaxed threshold
            unit="eV/A", rationale="attempted mid-run relaxation",
            source_class=ThresholdSourceClass.APPLICATION_ERROR_BUDGET,
            source_reference="synth", frozen_before_evaluation=True)],
    ).model_dump()
    with pytest.raises(ValueError):
        bind_policy(state, "evaluation", "EvaluationAdequacyPolicyV2", p2,
                    source_ref="synth", required=True)
    # Hash unchanged
    assert state["scientific_policies"]["evaluation::EvaluationAdequacyPolicyV2"]["content_sha256"] == hash1


# =====================================================================
# Ledger consistency
# =====================================================================
def test_framework_evolution_ledger_exists_and_covers_all_changes():
    """The ledger must reference every framework file modified during this
    closure sequence at least once. This test is a coarse check."""
    ledger = Path("docs/FRAMEWORK_EVOLUTION_LEDGER.md")
    assert ledger.is_file(), "ledger must exist"
    txt = ledger.read_text()
    # Every framework file touched in this session should be mentioned somewhere.
    required_mentions = [
        "framework_v2/scientific_adequacy.py",
        "framework_v2/scientific_gate.py",
        "framework_v2/scientific_recovery.py",
        "workflow/controller.py",
        "validation/structure_dynamics.py",
        "runtimes/pydantic_ai/executors.py",
        "runtimes/pydantic_ai/cli.py",
    ]
    missing = [m for m in required_mentions if m not in txt]
    assert not missing, f"ledger does not reference: {missing}"
    # Every change_id must have a status marked
    import re
    ids = re.findall(r"^## (FE-\d+)", txt, re.MULTILINE)
    assert len(ids) >= 10, f"expected >=10 change_ids, found {ids}"
    statuses = re.findall(r"^- \*\*status:\*\*", txt, re.MULTILINE)
    # Every FE-* has a status
    assert len(statuses) >= len(ids)


# =====================================================================
# C12F immutability re-verified at the END of the final closure
# =====================================================================
_C12F_HASHES = {
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


@pytest.mark.parametrize("relpath, expected", sorted(_C12F_HASHES.items()))
def test_c12f_artifacts_still_byte_immutable_at_final_closure(relpath, expected):
    from workflow.integrity import sha256_file
    p = _C12F_RUN / relpath
    if not p.is_file():
        pytest.skip(f"absent: {relpath}")
    assert sha256_file(p) == expected
