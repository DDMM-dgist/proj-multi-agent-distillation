"""Deterministic structural-region discovery for V2."""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase
from framework_v2.structural_regions import (
    StructuralRegion,
    StructuralRegionManifest,
    StructuralRegionProviderType,
    mint_child_regions,
)
from framework_v2.structural_representation import StructuralRepresentation


class DiscoveryMethod(str, Enum):
    FARTHEST_CENTROID = "FARTHEST_CENTROID"
    PLUGGABLE = "PLUGGABLE"


class RegionResolutionStatus(str, Enum):
    """Discovery-resolution decision for a single region.

    This axis is about STRUCTURAL resolution only; target relevance is a
    separate axis (:class:`TargetRelevanceStatus`) and must not be collapsed
    into this one.
    """
    RESOLVED = "RESOLVED"                          # adequately resolved; no refinement
    REFINE_SUPPORTED = "REFINE_SUPPORTED"          # evidence supports recursive discovery
    UNRESOLVED = "UNRESOLVED"                       # under-resolved but refinement NOT justified/possible
    REFINEMENT_NOT_SUPPORTED = "REFINEMENT_NOT_SUPPORTED"  # no stable substructure to discover


class TargetRelevanceStatus(str, Enum):
    """Target-relevance status of a region, kept orthogonal to discovery
    resolution. A region can be structurally RESOLVED yet
    TARGET_RELEVANCE_AMBIGUOUS, or structurally REFINE_SUPPORTED before its
    target relevance is decided."""
    TARGET_PRIMARY = "TARGET_PRIMARY"
    TARGET_BOUNDARY_SUPPORT = "TARGET_BOUNDARY_SUPPORT"
    OUT_OF_TARGET = "OUT_OF_TARGET"
    TARGET_RELEVANCE_AMBIGUOUS = "TARGET_RELEVANCE_AMBIGUOUS"
    TARGET_RELEVANCE_UNRESOLVED = "TARGET_RELEVANCE_UNRESOLVED"


class RegionDiscoveryConfig(ContractBase):
    method: DiscoveryMethod
    n_regions: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    reduction: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _explicit_params(self):
        if self.method == DiscoveryMethod.FARTHEST_CENTROID:
            if self.n_regions is None or self.n_regions <= 0:
                raise ValueError("FARTHEST_CENTROID requires positive n_regions")
        return self


class RegionRefinementEvidence(ContractBase):
    """One typed piece of evidence bearing on whether a region should be
    recursively refined. No numerical threshold is encoded here; the channel
    names are generic and material-agnostic."""
    channel: str  # e.g. discovery_stability / parameter_sensitivity /
                  # reproducible_substructure / occupancy_pathology /
                  # target_relevance / characterization_ambiguity
    supports_refinement: bool
    detail: str = ""
    refs: list[str] = Field(default_factory=list)


class RegionRefinementAssessment(ContractBase):
    """A first-class, typed assessment of whether a region is adequately
    resolved. It represents and ROUTES the decision; it deliberately does NOT
    invent a universal numerical heterogeneity threshold.

    ``resolution_status`` (structural) and ``target_relevance_status`` are kept
    as separate axes (a region may be well-resolved yet target-ambiguous)."""
    region_id: str
    manifest_sha256: str
    representation_sha256: str
    resolution_status: RegionResolutionStatus
    target_relevance_status: TargetRelevanceStatus = (
        TargetRelevanceStatus.TARGET_RELEVANCE_UNRESOLVED
    )
    evidence: list[RegionRefinementEvidence] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _supported_requires_evidence(self):
        if self.resolution_status == RegionResolutionStatus.REFINE_SUPPORTED:
            if not any(e.supports_refinement for e in self.evidence):
                raise ValueError(
                    "REFINE_SUPPORTED requires at least one supporting evidence channel"
                )
        return self


class RegionRefinementRequest(ContractBase):
    """A typed, authorized instruction to recursively refine a region.

    By construction this object can only exist for a ``REFINE_SUPPORTED``
    assessment: the validator rejects any other resolution status. This is the
    deterministic gate that stops an UNRESOLVED / REFINEMENT_NOT_SUPPORTED /
    RESOLVED region from being auto-refined -- an LLM cannot override failed
    deterministic evidence by hand-constructing a request."""
    parent_region_id: str
    current_manifest_sha256: str
    representation_sha256: str
    resolution_status: RegionResolutionStatus
    refinement_reasons: list[str]
    discovery_config: RegionDiscoveryConfig
    evidence_provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _only_refine_supported(self):
        if self.resolution_status != RegionResolutionStatus.REFINE_SUPPORTED:
            raise ValueError(
                "RegionRefinementRequest may only be constructed for a "
                f"REFINE_SUPPORTED resolution (got {self.resolution_status})"
            )
        if not self.refinement_reasons:
            raise ValueError("RegionRefinementRequest requires at least one refinement reason")
        return self


def refinement_is_authorized(assessment: RegionRefinementAssessment) -> bool:
    """Deterministic gate: only a REFINE_SUPPORTED assessment authorizes
    automatic recursive discovery."""
    return assessment.resolution_status == RegionResolutionStatus.REFINE_SUPPORTED


def authorize_refinement(
    assessment: RegionRefinementAssessment,
    *,
    discovery_config: "RegionDiscoveryConfig",
) -> RegionRefinementRequest:
    """Build an authorized refinement request from an assessment. Raises if the
    assessment does not authorize refinement (UNRESOLVED / RESOLVED /
    REFINEMENT_NOT_SUPPORTED all fail closed)."""
    if not refinement_is_authorized(assessment):
        raise ValueError(
            "refinement not authorized: resolution_status="
            f"{assessment.resolution_status}"
        )
    reasons = list(assessment.reasons) or [
        e.channel for e in assessment.evidence if e.supports_refinement
    ]
    return RegionRefinementRequest(
        parent_region_id=assessment.region_id,
        current_manifest_sha256=assessment.manifest_sha256,
        representation_sha256=assessment.representation_sha256,
        resolution_status=assessment.resolution_status,
        refinement_reasons=reasons,
        discovery_config=discovery_config,
        evidence_provenance=[ref for e in assessment.evidence for ref in e.refs],
    )


def _farthest_centers(matrix: np.ndarray, k: int) -> list[int]:
    n = matrix.shape[0]
    if k > n:
        raise ValueError("n_regions cannot exceed number of structures")
    centers = [0]
    while len(centers) < k:
        dists = []
        for i in range(n):
            if i in centers:
                dists.append(-1.0)
                continue
            d = min(float(np.linalg.norm(matrix[i] - matrix[c])) for c in centers)
            dists.append(d)
        centers.append(max(range(n), key=lambda i: (dists[i], -i)))
    return centers


def discover_structural_regions(
    representation: StructuralRepresentation,
    config: RegionDiscoveryConfig,
    *,
    manifest_id: str,
    source_sha256: str,
) -> StructuralRegionManifest:
    if config.method != DiscoveryMethod.FARTHEST_CENTROID:
        raise NotImplementedError("only FARTHEST_CENTROID is implemented in core V2")
    matrix = representation.as_array()
    centers = _farthest_centers(matrix, int(config.n_regions))
    assignments: dict[str, str] = {}
    members: dict[str, list[str]] = {
        f"structural_region_{i + 1:03d}": [] for i in range(len(centers))
    }
    region_ids = list(members)
    for row_index, structure_id in enumerate(representation.structure_ids):
        nearest = min(
            range(len(centers)),
            key=lambda c: (float(np.linalg.norm(matrix[row_index] - matrix[centers[c]])), c),
        )
        region_id = region_ids[nearest]
        assignments[structure_id] = region_id
        members[region_id].append(structure_id)

    rep_sha = representation.content_sha256()
    config_sha = config.content_sha256()
    regions = [
        StructuralRegion(
            region_id=region_id,
            provider_type=StructuralRegionProviderType.STRUCTURAL_DISCOVERY,
            membership_provenance=[rep_sha, config_sha],
            population_size=len(frame_ids),
            semantic_annotation=f"discovered cluster {region_id}",
            descriptor_provenance=[representation.backend.value, rep_sha],
            representation_sha256=rep_sha,
            membership_manifest_sha256=config_sha,
        )
        for region_id, frame_ids in members.items()
    ]
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.STRUCTURAL_DISCOVERY,
        regions=regions,
        frame_to_region=assignments,
        source_sha256=source_sha256,
    )


def _subset_representation(
    representation: StructuralRepresentation, member_ids: set[str]
) -> StructuralRepresentation:
    """Deterministic row subset of a representation, preserving original order.

    Fails closed if any requested member lacks a representation row (FPS/knn
    over a subset must not silently drop frames)."""
    index = {sid: row for sid, row in zip(representation.structure_ids, representation.matrix)}
    missing = sorted(mid for mid in member_ids if mid not in index)
    if missing:
        raise ValueError(
            "representation is missing rows for parent members: " + ", ".join(missing)
        )
    ids = [sid for sid in representation.structure_ids if sid in member_ids]
    matrix = [index[sid] for sid in ids]
    return StructuralRepresentation(
        representation_id=f"{representation.representation_id}::subset",
        backend=representation.backend,
        parameters=dict(representation.parameters),
        software=dict(representation.software),
        species=list(representation.species),
        structure_ids=ids,
        matrix=matrix,
        descriptor_dimension=representation.descriptor_dimension,
        pooling=representation.pooling,
        structure_ordering=representation.structure_ordering,
    )


def discover_structural_subregions(
    *,
    parent_manifest: StructuralRegionManifest,
    parent_region_id: str,
    representation: StructuralRepresentation,
    config: RegionDiscoveryConfig,
    manifest_id: str,
    source_sha256: str,
    authorization: RegionRefinementRequest | None = None,
) -> StructuralRegionManifest:
    """First-class recursive discovery: run the generic discovery backend over
    ONLY the members of one active-leaf region and mint deterministic children
    under it.

    Material-agnostic: the same call refines C0, C2, R17, a ternary-system
    region, or a child-of-child region. No caller-written, system-specific
    filtering is required. Returns the typed child-discovery artifact (a
    manifest whose ``frame_to_region`` covers only the parent's members),
    consumed by :func:`framework_v2.structural_regions.refine_region_partition`.
    """
    parent = parent_manifest.region(parent_region_id)
    if parent is None:
        raise ValueError(f"parent region {parent_region_id!r} not in manifest")
    if parent_region_id not in set(parent_manifest.active_leaf_region_ids()):
        raise ValueError(f"region {parent_region_id!r} is not an active leaf; cannot refine")

    if authorization is not None:
        if authorization.parent_region_id != parent_region_id:
            raise ValueError("authorization.parent_region_id does not match parent_region_id")
        if authorization.current_manifest_sha256 != parent_manifest.content_sha256():
            raise ValueError("authorization.current_manifest_sha256 does not match parent_manifest")
        if authorization.representation_sha256 != representation.content_sha256():
            raise ValueError("authorization.representation_sha256 does not match representation")

    member_ids = {
        frame_id
        for frame_id, rid in parent_manifest.frame_to_region.items()
        if rid == parent_region_id
    }
    if not member_ids:
        raise ValueError(f"parent region {parent_region_id!r} has no member frames")

    subset = _subset_representation(representation, member_ids)
    flat = discover_structural_regions(
        subset,
        config,
        manifest_id=f"{manifest_id}::flat",
        source_sha256=source_sha256,
    )
    children = mint_child_regions(
        parent=parent,
        flat_subregion_manifest=flat,
        representation_sha256=representation.content_sha256(),
        extra_membership_provenance=[
            f"parent_manifest:{parent_manifest.content_sha256()}",
            f"full_representation:{representation.content_sha256()}",
        ],
    )
    # Re-key the returned artifact's manifest_id/source to the caller's request
    # while preserving the deterministic child regions + partition.
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.HYBRID,
        regions=children.regions,
        frame_to_region=children.frame_to_region,
        source_sha256=source_sha256,
    )


__all__ = [
    "DiscoveryMethod",
    "RegionDiscoveryConfig",
    "RegionRefinementAssessment",
    "RegionRefinementEvidence",
    "RegionRefinementRequest",
    "RegionResolutionStatus",
    "TargetRelevanceStatus",
    "authorize_refinement",
    "discover_structural_regions",
    "discover_structural_subregions",
    "refinement_is_authorized",
]
