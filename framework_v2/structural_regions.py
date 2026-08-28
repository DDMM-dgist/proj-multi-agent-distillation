"""Public V2 structural-region abstraction.

The historical framework uses ``DomainRepresentation`` and explicit domain
labels in several places.  V2 exposes a provider-neutral
``StructuralRegion`` surface over that machinery without rewriting historical
schemas.  Downstream curation, error tracking, and recovery consume only this
surface, so explicit semantic labels and discovered clusters are equivalent
for workflow purposes.
"""
from __future__ import annotations

import hashlib
import json
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
    # Explicit discovery-tree depth (root discovery = 0). Downstream code must
    # read this field for ancestry depth; it must never parse it out of the
    # region_id string. Default 0 keeps historical flat regions backward
    # compatible without a migration.
    discovery_depth: int = 0
    cluster_is_physical_phase: bool = False

    @model_validator(mode="after")
    def _well_formed(self):
        if self.population_size < 0:
            raise ValueError("StructuralRegion population_size must be non-negative")
        if not self.membership_provenance:
            raise ValueError("StructuralRegion requires membership provenance")
        if self.discovery_depth < 0:
            raise ValueError("StructuralRegion discovery_depth must be non-negative")
        if self.discovery_depth > 0 and not self.parent_region_id:
            raise ValueError("a region at discovery_depth > 0 requires parent_region_id")
        if self.parent_region_id is not None and self.parent_region_id == self.region_id:
            raise ValueError("StructuralRegion cannot be its own parent")
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

    def region(self, region_id: str) -> StructuralRegion | None:
        return next((r for r in self.regions if r.region_id == region_id), None)

    def _refined_parent_ids(self) -> set[str]:
        """IDs of regions that have been superseded by children in the tree.

        A region is "refined" (no longer an active leaf) iff some other region
        declares it as its parent. This is derived from ancestry FIELDS only --
        never from parsing region_id strings."""
        return {
            r.parent_region_id
            for r in self.regions
            if r.parent_region_id is not None
        }

    def active_leaf_regions(self) -> list[StructuralRegion]:
        """The regions that own the current population: every region that is
        not the parent of any other region. CURATE / TRACK / ErrorLedger /
        RECOVER are authoritative over these, not over refined parents."""
        refined = self._refined_parent_ids()
        return [r for r in self.regions if r.region_id not in refined]

    def active_leaf_region_ids(self) -> list[str]:
        return [r.region_id for r in self.active_leaf_regions()]

    def active_leaf_frame_to_region(self) -> dict[str, str]:
        """Frame -> active-leaf assignment. By construction of
        ``refine_region_partition`` every frame already maps to an active leaf;
        this validates that invariant and fails closed otherwise."""
        leaves = set(self.active_leaf_region_ids())
        mapping: dict[str, str] = {}
        for frame_id, region_id in self.frame_to_region.items():
            if region_id not in leaves:
                raise ValueError(
                    f"frame {frame_id!r} maps to non-leaf region {region_id!r}; "
                    "active-leaf partition is inconsistent"
                )
            mapping[frame_id] = region_id
        return mapping

    def active_leaf_partition_sha256(self) -> str:
        """Order-independent identity of the active-leaf partition.

        Changes iff a frame's active-leaf assignment changes. Independent of
        region-list or dict ordering."""
        payload = sorted(self.active_leaf_frame_to_region().items())
        return _canonical_sha256({"active_leaf_frame_to_region": payload})

    def region_tree_sha256(self) -> str:
        """Order-independent identity of the ancestry tree (id, parent, depth,
        population). Changes iff a parent-child relation or population changes."""
        nodes = sorted(
            (r.region_id, r.parent_region_id, r.discovery_depth, r.population_size)
            for r in self.regions
        )
        return _canonical_sha256({"region_tree": nodes})


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


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def child_region_id(parent_region_id: str, child_index: int) -> str:
    """Canonical, deterministic, collision-free child identity under a parent.

    The returned string is OPAQUE to downstream code: ancestry is carried in
    ``parent_region_id`` / ``discovery_depth`` FIELDS, never by parsing this
    string. The scheme is independent of any physical interpretation and of any
    material system.
    """
    if not parent_region_id:
        raise ValueError("child_region_id requires a non-empty parent_region_id")
    if child_index < 0:
        raise ValueError("child_region_id child_index must be non-negative")
    return f"{parent_region_id}.{child_index}"


def mint_child_regions(
    *,
    parent: StructuralRegion,
    flat_subregion_manifest: StructuralRegionManifest,
    representation_sha256: str | None = None,
    extra_membership_provenance: list[str] | None = None,
) -> StructuralRegionManifest:
    """Re-key a flat sub-discovery manifest into deterministic child identities
    bound to ``parent``.

    ``flat_subregion_manifest`` is the raw output of running the generic
    discovery backend over the parent's member subset (region ids like
    ``structural_region_001``). Child ids are minted as
    ``child_region_id(parent.region_id, i)`` in sorted flat-id order so the
    mapping is deterministic and collision-free. The returned manifest's
    ``frame_to_region`` covers ONLY the parent's members (it is the typed
    child-discovery artifact consumed by :func:`refine_region_partition`)."""
    flat_ids = sorted(r.region_id for r in flat_subregion_manifest.regions)
    id_map = {
        flat_id: child_region_id(parent.region_id, index)
        for index, flat_id in enumerate(flat_ids)
    }
    depth = parent.discovery_depth + 1
    extra = list(extra_membership_provenance or [])
    children: list[StructuralRegion] = []
    for region in sorted(flat_subregion_manifest.regions, key=lambda r: r.region_id):
        children.append(
            StructuralRegion(
                region_id=id_map[region.region_id],
                provider_type=StructuralRegionProviderType.HYBRID,
                membership_provenance=[*region.membership_provenance, parent.region_id, *extra],
                population_size=region.population_size,
                semantic_annotation=None,
                descriptor_provenance=list(region.descriptor_provenance),
                representation_sha256=representation_sha256 or region.representation_sha256,
                parent_region_id=parent.region_id,
                discovery_depth=depth,
                membership_manifest_sha256=region.membership_manifest_sha256,
            )
        )
    frame_to_region = {
        frame_id: id_map[rid]
        for frame_id, rid in flat_subregion_manifest.frame_to_region.items()
    }
    return StructuralRegionManifest(
        manifest_id=f"{flat_subregion_manifest.manifest_id}::children_of::{parent.region_id}",
        provider_type=StructuralRegionProviderType.HYBRID,
        regions=children,
        frame_to_region=frame_to_region,
        source_sha256=flat_subregion_manifest.source_sha256,
    )


def refine_region_partition(
    *,
    manifest_id: str,
    current_manifest: StructuralRegionManifest,
    parent_region_id: str,
    child_manifest: StructuralRegionManifest,
    source_sha256: str | None = None,
) -> StructuralRegionManifest:
    """Partition-preserving replacement of one active-leaf region by its
    children.

    Unlike :func:`hybrid_regions_from_parent_and_subregions` (retained for
    backward compatibility), this preserves the COMPLETE active-leaf partition:
    every unaffected leaf and every frame outside the refined parent is kept
    byte-for-byte. The refined parent stays in the region list for ancestry /
    provenance but ceases to be an active leaf.

    Fails closed on any partition violation."""
    parent = current_manifest.region(parent_region_id)
    if parent is None:
        raise ValueError(f"parent region {parent_region_id!r} not in current manifest")
    if parent_region_id in current_manifest._refined_parent_ids():
        raise ValueError(f"region {parent_region_id!r} is already refined")

    parent_members = {
        frame_id
        for frame_id, rid in current_manifest.frame_to_region.items()
        if rid == parent_region_id
    }
    if not parent_members:
        raise ValueError(f"parent region {parent_region_id!r} has no member frames to refine")

    existing_ids = {r.region_id for r in current_manifest.regions}
    child_members = set(child_manifest.frame_to_region)

    # coverage: children exactly repartition the parent (subset + complete)
    outside = child_members - parent_members
    if outside:
        raise ValueError(
            "child frames fall outside parent membership: " + ", ".join(sorted(outside))
        )
    uncovered = parent_members - child_members
    if uncovered:
        raise ValueError(
            "child coverage is incomplete; unassigned parent frames: "
            + ", ".join(sorted(uncovered))
        )

    for child in child_manifest.regions:
        if child.region_id in existing_ids:
            raise ValueError(f"child region_id {child.region_id!r} collides with an existing region")
        if child.parent_region_id != parent_region_id:
            raise ValueError(
                f"child {child.region_id!r} parent_region_id "
                f"{child.parent_region_id!r} != {parent_region_id!r}"
            )
        if child.discovery_depth != parent.discovery_depth + 1:
            raise ValueError(
                f"child {child.region_id!r} discovery_depth {child.discovery_depth} "
                f"!= parent depth + 1 ({parent.discovery_depth + 1})"
            )

    regions = [*current_manifest.regions, *child_manifest.regions]

    frame_to_region = dict(current_manifest.frame_to_region)
    for frame_id in parent_members:
        frame_to_region[frame_id] = child_manifest.frame_to_region[frame_id]

    if len(frame_to_region) != len(current_manifest.frame_to_region):
        raise ValueError("refinement changed the total frame population")

    refined = StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.HYBRID,
        regions=regions,
        frame_to_region=frame_to_region,
        source_sha256=source_sha256 or current_manifest.source_sha256,
    )
    # Post-condition: the composed manifest is a complete active-leaf partition
    # and the refined parent is no longer an active leaf. Fails closed here.
    refined.active_leaf_frame_to_region()
    if parent_region_id in set(refined.active_leaf_region_ids()):
        raise ValueError("refined parent unexpectedly remained an active leaf")
    return refined


__all__ = [
    "StructuralRegion",
    "StructuralRegionManifest",
    "StructuralRegionProviderType",
    "child_region_id",
    "explicit_regions_from_membership",
    "hybrid_regions_from_parent_and_subregions",
    "mint_child_regions",
    "refine_region_partition",
    "regions_from_domain_representation",
    "structural_regions_from_target_regime_model",
]
