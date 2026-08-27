"""Public V2 structural-region abstraction.

The historical framework uses ``DomainRepresentation`` and explicit domain
labels in several places.  V2 exposes a provider-neutral
``StructuralRegion`` surface over that machinery without rewriting historical
schemas.  Downstream curation, error tracking, and recovery consume only this
surface, so explicit semantic labels and discovered clusters are equivalent
for workflow purposes.
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, DomainRepresentation
from framework_v2.acquisition.contracts import TargetRegimeModel


class StructuralRegionProviderType(str, Enum):
    EXPLICIT_METADATA = "EXPLICIT_METADATA"
    STRUCTURAL_DISCOVERY = "STRUCTURAL_DISCOVERY"
    HYBRID = "HYBRID"


class StructuralRegion(ContractBase):
    region_id: str
    provider_type: StructuralRegionProviderType
    membership_provenance: list[str]
    population_size: int
    membership_manifest_sha256: str
    semantic_annotation: str | None = None
    descriptor_provenance: list[str] = Field(default_factory=list)
    representation_sha256: str | None = None
    parent_region_id: str | None = None
    cluster_is_physical_phase: bool = False

    @model_validator(mode="after")
    def _well_formed(self):
        if self.population_size < 0:
            raise ValueError("StructuralRegion population_size must be non-negative")
        if not self.membership_provenance:
            raise ValueError("StructuralRegion requires membership provenance")
        if self.provider_type == StructuralRegionProviderType.STRUCTURAL_DISCOVERY:
            if self.semantic_annotation and self.cluster_is_physical_phase:
                raise ValueError("a discovered cluster must not be asserted as a physical phase")
            if not self.representation_sha256:
                raise ValueError("discovered regions require representation_sha256")
        if self.provider_type == StructuralRegionProviderType.HYBRID and not self.parent_region_id:
            raise ValueError("hybrid subregions require parent_region_id")
        return self


class StructuralRegionManifest(ContractBase):
    manifest_id: str
    provider_type: StructuralRegionProviderType
    regions: list[StructuralRegion]
    frame_to_region: dict[str, str]
    source_sha256: str

    @model_validator(mode="after")
    def _consistent(self):
        ids = [region.region_id for region in self.regions]
        if len(set(ids)) != len(ids):
            raise ValueError("StructuralRegionManifest contains duplicate region_id")
        known = set(ids)
        missing = sorted({rid for rid in self.frame_to_region.values() if rid not in known})
        if missing:
            raise ValueError("frame_to_region references unknown regions: " + ", ".join(missing))
        return self

    def region_for_frame(self, frame_id: str) -> StructuralRegion | None:
        region_id = self.frame_to_region.get(frame_id)
        if region_id is None:
            return None
        return next(region for region in self.regions if region.region_id == region_id)


def explicit_regions_from_membership(
    *,
    manifest_id: str,
    frame_to_region: Mapping[str, str],
    source_sha256: str,
    membership_manifest_sha256: str,
    semantic_annotations: Mapping[str, str] | None = None,
) -> StructuralRegionManifest:
    counts: dict[str, int] = {}
    for region_id in frame_to_region.values():
        counts[region_id] = counts.get(region_id, 0) + 1
    annotations = dict(semantic_annotations or {})
    regions = [
        StructuralRegion(
            region_id=region_id,
            provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
            membership_provenance=[membership_manifest_sha256],
            population_size=count,
            semantic_annotation=annotations.get(region_id, region_id),
            membership_manifest_sha256=membership_manifest_sha256,
        )
        for region_id, count in sorted(counts.items())
    ]
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
        regions=regions,
        frame_to_region=dict(frame_to_region),
        source_sha256=source_sha256,
    )


def regions_from_domain_representation(
    representation: DomainRepresentation,
    *,
    manifest_id: str,
    source_sha256: str,
) -> StructuralRegionManifest:
    frame_to_region: dict[str, str] = {}
    regions: list[StructuralRegion] = []
    rep_sha = representation.content_sha256()
    for regime in representation.regimes:
        members = [
            ref
            for ref in regime.membership_evidence_refs
            if not ref.startswith("pool_manifest:")
        ]
        for frame_id in members:
            if frame_id in frame_to_region:
                raise ValueError(f"frame {frame_id!r} appears in multiple regions")
            frame_to_region[frame_id] = regime.regime_id
        regions.append(
            StructuralRegion(
                region_id=regime.regime_id,
                provider_type=StructuralRegionProviderType.STRUCTURAL_DISCOVERY,
                membership_provenance=list(regime.membership_evidence_refs),
                population_size=len(members),
                semantic_annotation=regime.label,
                descriptor_provenance=[representation.descriptor],
                representation_sha256=rep_sha,
                membership_manifest_sha256=rep_sha,
            )
        )
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.STRUCTURAL_DISCOVERY,
        regions=regions,
        frame_to_region=frame_to_region,
        source_sha256=source_sha256,
    )


def hybrid_regions_from_parent_and_subregions(
    *,
    manifest_id: str,
    parent_manifest: StructuralRegionManifest,
    subregion_manifest: StructuralRegionManifest,
    source_sha256: str,
) -> StructuralRegionManifest:
    regions: list[StructuralRegion] = list(parent_manifest.regions)
    for region in subregion_manifest.regions:
        parent = parent_manifest.region_for_frame(
            next(
                frame
                for frame, rid in subregion_manifest.frame_to_region.items()
                if rid == region.region_id
            )
        )
        if parent is None:
            raise ValueError(f"hybrid subregion {region.region_id!r} has no explicit parent")
        regions.append(
            StructuralRegion(
                region_id=region.region_id,
                provider_type=StructuralRegionProviderType.HYBRID,
                membership_provenance=region.membership_provenance,
                population_size=region.population_size,
                semantic_annotation=region.semantic_annotation,
                descriptor_provenance=region.descriptor_provenance,
                representation_sha256=region.representation_sha256,
                parent_region_id=parent.region_id,
                membership_manifest_sha256=region.membership_manifest_sha256,
            )
        )
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.HYBRID,
        regions=regions,
        frame_to_region=dict(subregion_manifest.frame_to_region),
        source_sha256=source_sha256,
    )


def structural_regions_from_target_regime_model(
    model: TargetRegimeModel,
    *,
    manifest_id: str,
    source_sha256: str,
) -> StructuralRegionManifest:
    """Compatibility adapter for existing target-regime contracts."""

    frame_to_region: dict[str, str] = {}
    regions: list[StructuralRegion] = []
    model_sha = model.content_sha256()
    for regime in model.regimes:
        members = [ref for ref in regime.evidence_refs if ref.startswith("frame:")]
        member_ids = [ref.removeprefix("frame:") for ref in members]
        for frame_id in member_ids:
            frame_to_region[frame_id] = regime.regime_id
        regions.append(
            StructuralRegion(
                region_id=regime.regime_id,
                provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
                membership_provenance=list(regime.evidence_refs) or [model_sha],
                population_size=len(member_ids),
                semantic_annotation=regime.label,
                membership_manifest_sha256=model_sha,
            )
        )
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
        regions=regions,
        frame_to_region=frame_to_region,
        source_sha256=source_sha256,
    )


__all__ = [
    "StructuralRegion",
    "StructuralRegionManifest",
    "StructuralRegionProviderType",
    "explicit_regions_from_membership",
    "hybrid_regions_from_parent_and_subregions",
    "regions_from_domain_representation",
    "structural_regions_from_target_regime_model",
]
