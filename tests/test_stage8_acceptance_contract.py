import numpy as np
import pytest

from framework_v2.stage8_acceptance import (
    DomainAssignment,
    ElementalAlignmentPolicy,
    Stage8AcceptancePolicy,
    Stage8PrimaryPopulationPolicy,
    Stage8Role,
    aligned_energy_residuals_meV,
    build_population_domain_manifest,
    evaluate_stage8_acceptance,
    force_component_rmse,
    per_domain_aligned_energy_rmse,
)


PRIMARY = [
    "amorphous_bulk_SiO2",
    "crystalline_SiO2",
    "high_pressure_SiO2",
    "liquid_or_melt_SiO2",
    "surface_SiO2",
    "oxygen_vacancy_SiO2",
    "amorphous_SiOx_sub_stoichiometric",
    "condensed_pure_Si_boundary",
]


def _policy():
    categories = [
        ("bulk_amo", "amorphous_bulk_SiO2"),
        ("bulk_cryst", "crystalline_SiO2"),
        ("bulk_cryst_hp", "high_pressure_SiO2"),
        ("liquid", "liquid_or_melt_SiO2"),
        ("surfaces", "surface_SiO2"),
        ("vacancy", "oxygen_vacancy_SiO2"),
        ("SiOx_int_AL", "amorphous_SiOx_sub_stoichiometric"),
        ("silicon_defects", "condensed_pure_Si_boundary"),
    ]
    return Stage8PrimaryPopulationPolicy(
        policy_id="stage8-primary-test",
        primary_domains=PRIMARY,
        assignments=[
            DomainAssignment(source_category=c, domain=d, role=Stage8Role.PRIMARY_CLAIM)
            for c, d in categories
        ]
        + [
            DomainAssignment(
                source_category="cluster",
                domain="OUT_OF_SCOPE_DIAGNOSTIC",
                role=Stage8Role.DIAGNOSTIC_ONLY,
            ),
            DomainAssignment(
                source_category="silicon_others",
                domain="condensed_pure_Si_boundary",
                role=Stage8Role.DIAGNOSTIC_ONLY,
            ),
            DomainAssignment(
                source_category="interstitial",
                domain="AMBIGUOUS",
                role=Stage8Role.AMBIGUOUS,
            ),
        ],
        diagnostic_domains=["OUT_OF_SCOPE_DIAGNOSTIC", "condensed_pure_Si_boundary"],
    )


def _frame(source_category, symbols=("Si", "O", "O"), structure_id=None):
    return {
        "source_category": source_category,
        "structure_id": structure_id or source_category,
        "symbols": list(symbols),
    }


def _primary_frames():
    cats = [
        "bulk_amo",
        "bulk_cryst",
        "bulk_cryst_hp",
        "liquid",
        "surfaces",
        "vacancy",
        "SiOx_int_AL",
        "silicon_defects",
    ]
    return [_frame(cat, structure_id=f"p{i}") for i, cat in enumerate(cats)]


def test_all_primary_domains_required_and_diagnostic_excluded_from_pass():
    frames = _primary_frames() + [_frame("cluster"), _frame("silicon_others")]
    manifest = build_population_domain_manifest(
        manifest_id="m",
        population_id="dft-primary",
        source_population_path="holdout.xyz",
        source_population_sha256="a" * 64,
        policy=_policy(),
        frames=frames,
    )

    assert manifest.primary_frame_count == 8
    assert manifest.diagnostic_frame_count == 2
    assert manifest.primary_counts_by_domain == {d: 1 for d in PRIMARY}
    assert manifest.diagnostic_counts_by_domain == {
        "OUT_OF_SCOPE_DIAGNOSTIC": 1,
        "condensed_pure_Si_boundary": 1,
    }
    assert manifest.primary_indices == list(range(8))
    assert manifest.diagnostic_indices == [8, 9]


def test_missing_primary_domain_fails_closed():
    with pytest.raises(ValueError, match="lacks required domain"):
        build_population_domain_manifest(
            manifest_id="m",
            population_id="dft-primary",
            source_population_path="holdout.xyz",
            source_population_sha256="a" * 64,
            policy=_policy(),
            frames=_primary_frames()[:-1],
        )


def test_authoritative_source_category_mapping_and_unmapped_or_ambiguous_fail_closed():
    # Raw config_type is deliberately ignored; source_category is authoritative.
    frames = _primary_frames()
    frames[1]["config_type"] = "interstitial"
    manifest = build_population_domain_manifest(
        manifest_id="m",
        population_id="dft-primary",
        source_population_path="holdout.xyz",
        source_population_sha256="a" * 64,
        policy=_policy(),
        frames=frames,
    )
    assert manifest.frame_records[1].domain == "crystalline_SiO2"

    with pytest.raises(ValueError, match="unmapped"):
        build_population_domain_manifest(
            manifest_id="m",
            population_id="dft-primary",
            source_population_path="holdout.xyz",
            source_population_sha256="a" * 64,
            policy=_policy(),
            frames=_primary_frames() + [_frame("not_declared")],
        )
    with pytest.raises(ValueError, match="ambiguous"):
        build_population_domain_manifest(
            manifest_id="m",
            population_id="dft-primary",
            source_population_path="holdout.xyz",
            source_population_sha256="a" * 64,
            policy=_policy(),
            frames=_primary_frames() + [_frame("interstitial")],
        )


def test_species_wise_composition_alignment_and_same_subspace_linearity():
    alignment = ElementalAlignmentPolicy(policy_id="align", species=["Si", "O"])
    frames = [
        _frame("bulk_amo", ("Si", "O", "O")),
        _frame("bulk_cryst", ("Si", "Si", "O", "O")),
        _frame("liquid", ("Si", "Si", "Si", "O")),
        _frame("vacancy", ("Si", "O")),
    ]
    dft = np.array([-10.0, -20.0, -30.0, -12.0])
    teacher = dft + np.array([0.30, 0.20, -0.10, 0.04])
    student = teacher + np.array([0.03, -0.02, 0.04, -0.01])

    st, st_meta = aligned_energy_residuals_meV(teacher, student, frames, alignment)
    td, td_meta = aligned_energy_residuals_meV(dft, teacher, frames, alignment)
    sd, sd_meta = aligned_energy_residuals_meV(dft, student, frames, alignment)

    assert st_meta["design_matrix_shape"] == [4, 2]
    assert td_meta["design_matrix_shape"] == [4, 2]
    assert sd_meta["design_matrix_shape"] == [4, 2]
    assert st_meta["alignment_policy_sha256"] == td_meta["alignment_policy_sha256"]
    assert st_meta["alignment_policy_sha256"] == sd_meta["alignment_policy_sha256"]
    np.testing.assert_allclose(sd, st + td, atol=1e-10)


def test_per_domain_metrics_after_common_alignment():
    alignment = ElementalAlignmentPolicy(policy_id="align", species=["Si", "O"])
    frames = _primary_frames()
    teacher = np.linspace(-10.0, -17.0, 8)
    student = teacher + np.linspace(0.01, 0.08, 8)
    residuals, _meta = aligned_energy_residuals_meV(teacher, student, frames, alignment)
    manifest = build_population_domain_manifest(
        manifest_id="m",
        population_id="dft-primary",
        source_population_path="holdout.xyz",
        source_population_sha256="a" * 64,
        policy=_policy(),
        frames=frames,
    )

    per_domain = per_domain_aligned_energy_rmse(
        residuals, manifest.frame_records, manifest.primary_domains
    )
    assert set(per_domain) == set(PRIMARY)
    assert all(value >= 0.0 for value in per_domain.values())


def test_force_component_rmse_uses_vectorized_components():
    ref = [np.zeros((2, 3)), np.ones((1, 3))]
    pred = [np.ones((2, 3)), np.ones((1, 3)) * 3]
    expected = np.sqrt(np.mean(np.array([1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=float) ** 2))
    assert force_component_rmse(ref, pred) == pytest.approx(expected)


def test_frozen_caps_teacher_evidence_only_and_triangle_envelope_no_margin():
    pop_policy = _policy()
    align = ElementalAlignmentPolicy(policy_id="align", species=["Si", "O"])
    policy = Stage8AcceptancePolicy(
        policy_id="accept",
        primary_population_policy_sha256=pop_policy.content_sha256(),
        alignment_policy_sha256=align.content_sha256(),
        required_primary_domains=PRIMARY,
    )
    svt_e = {d: 24.0 for d in PRIMARY}
    svt_f = {d: 0.29 for d in PRIMARY}
    tdf_e = {d: 10.0 for d in PRIMARY}
    tdf_f = {d: 0.10 for d in PRIMARY}
    sdf_e = {d: 34.0 for d in PRIMARY}
    sdf_f = {d: 0.39 for d in PRIMARY}

    result = evaluate_stage8_acceptance(
        policy=policy,
        student_vs_teacher_energy_rmse=svt_e,
        student_vs_teacher_force_rmse=svt_f,
        teacher_vs_dft_energy_rmse=tdf_e,
        teacher_vs_dft_force_rmse=tdf_f,
        student_vs_dft_energy_rmse=sdf_e,
        student_vs_dft_force_rmse=sdf_f,
    )
    assert result["overall_pass"] is True
    assert result["teacher_vs_dft"]["semantics"] == "required_reference_evidence_only"
    assert result["no_empirical_margin"] is True
    assert result["student_vs_dft_consistency"][PRIMARY[0]]["energy_bound_meV_per_atom"] == 35.0
    assert result["student_vs_dft_consistency"][PRIMARY[0]]["force_bound_eV_per_angstrom"] == 0.4

    with pytest.raises(ValueError, match="permits no empirical margin"):
        Stage8AcceptancePolicy(
            policy_id="accept",
            primary_population_policy_sha256=pop_policy.content_sha256(),
            alignment_policy_sha256=align.content_sha256(),
            required_primary_domains=PRIMARY,
            student_vs_dft_envelope_margin=0.25,
        )


def test_acceptance_fails_when_caps_or_triangle_envelope_fail():
    pop_policy = _policy()
    align = ElementalAlignmentPolicy(policy_id="align", species=["Si", "O"])
    policy = Stage8AcceptancePolicy(
        policy_id="accept",
        primary_population_policy_sha256=pop_policy.content_sha256(),
        alignment_policy_sha256=align.content_sha256(),
        required_primary_domains=PRIMARY,
    )
    svt_e = {d: 24.0 for d in PRIMARY}
    svt_f = {d: 0.29 for d in PRIMARY}
    svt_e[PRIMARY[0]] = 26.0
    tdf_e = {d: 10.0 for d in PRIMARY}
    tdf_f = {d: 0.10 for d in PRIMARY}
    sdf_e = {d: 34.0 for d in PRIMARY}
    sdf_f = {d: 0.39 for d in PRIMARY}
    sdf_f[PRIMARY[1]] = 0.41

    result = evaluate_stage8_acceptance(
        policy=policy,
        student_vs_teacher_energy_rmse=svt_e,
        student_vs_teacher_force_rmse=svt_f,
        teacher_vs_dft_energy_rmse=tdf_e,
        teacher_vs_dft_force_rmse=tdf_f,
        student_vs_dft_energy_rmse=sdf_e,
        student_vs_dft_force_rmse=sdf_f,
    )
    assert result["overall_pass"] is False
    assert result["student_vs_teacher"][PRIMARY[0]]["energy_pass"] is False
    assert result["student_vs_dft_consistency"][PRIMARY[1]]["force_pass"] is False


def test_provenance_hash_binding_is_stable():
    pop_policy = _policy()
    align = ElementalAlignmentPolicy(policy_id="align", species=["Si", "O"])
    accept = Stage8AcceptancePolicy(
        policy_id="accept",
        primary_population_policy_sha256=pop_policy.content_sha256(),
        alignment_policy_sha256=align.content_sha256(),
        required_primary_domains=PRIMARY,
    )
    assert pop_policy.content_sha256() == _policy().content_sha256()
    assert accept.primary_population_policy_sha256 == pop_policy.content_sha256()
    assert accept.alignment_policy_sha256 == align.content_sha256()
