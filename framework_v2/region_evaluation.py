"""Region-aware V2 evaluation compatibility helpers."""
from __future__ import annotations

from framework_v2.stage8_acceptance import Stage8PopulationDomainManifest, Stage8Role
from framework_v2.structural_regions import (
    StructuralRegion,
    StructuralRegionManifest,
    StructuralRegionProviderType,
)


def structural_regions_from_stage8_manifest(
    manifest: Stage8PopulationDomainManifest,
    *,
    manifest_id: str,
) -> StructuralRegionManifest:
    """Adapt historical explicit-domain Stage-8 evidence to V2 regions.

    PRIMARY_CLAIM and DIAGNOSTIC_ONLY semantics remain in the historical
    manifest.  The V2 surface exposes provider-neutral regions without
    promoting diagnostic frames into primary pass/fail logic.
    """

    frame_to_region: dict[str, str] = {}
    counts: dict[str, int] = {}
    for record in manifest.frame_records:
        if record.role != Stage8Role.PRIMARY_CLAIM:
            continue
        frame_to_region[record.frame_id] = record.domain
        counts[record.domain] = counts.get(record.domain, 0) + 1

    regions = [
        StructuralRegion(
            region_id=domain,
            provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
            membership_provenance=[manifest.content_sha256(), manifest.policy_sha256],
            population_size=count,
            semantic_annotation=domain,
            membership_manifest_sha256=manifest.content_sha256(),
        )
        for domain, count in sorted(counts.items())
    ]
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
        regions=regions,
        frame_to_region=frame_to_region,
        source_sha256=manifest.source_population_sha256,
    )


__all__ = ["structural_regions_from_stage8_manifest"]
