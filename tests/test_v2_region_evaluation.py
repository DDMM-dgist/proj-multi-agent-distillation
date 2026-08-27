from framework_v2.region_evaluation import structural_regions_from_stage8_manifest
from framework_v2.stage8_acceptance import (
    FrameDomainRecord,
    Stage8PopulationDomainManifest,
    Stage8Role,
)


def test_stage8_primary_domains_adapt_without_diagnostic_promotion():
    stage8 = Stage8PopulationDomainManifest(
        manifest_id="stage8",
        population_id="primary",
        source_population_path="primary.extxyz",
        source_population_sha256="source",
        policy_sha256="policy",
        primary_domains=["oxygen_vacancy_SiO2"],
        frame_records=[
            FrameDomainRecord(
                frame_index=0,
                frame_id="primary_frame",
                source_category="vacancy",
                domain="oxygen_vacancy_SiO2",
                role=Stage8Role.PRIMARY_CLAIM,
            ),
            FrameDomainRecord(
                frame_index=1,
                frame_id="diagnostic_frame",
                source_category="cluster",
                domain="cluster",
                role=Stage8Role.DIAGNOSTIC_ONLY,
            ),
        ],
        primary_frame_count=1,
        diagnostic_frame_count=1,
        primary_counts_by_domain={"oxygen_vacancy_SiO2": 1},
        diagnostic_counts_by_domain={"cluster": 1},
    )

    manifest = structural_regions_from_stage8_manifest(stage8, manifest_id="v2")
    assert manifest.region_for_frame("primary_frame").region_id == "oxygen_vacancy_SiO2"
    assert manifest.region_for_frame("diagnostic_frame") is None
