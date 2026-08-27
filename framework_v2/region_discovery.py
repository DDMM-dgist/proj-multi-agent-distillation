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
)
from framework_v2.structural_representation import StructuralRepresentation


class DiscoveryMethod(str, Enum):
    FARTHEST_CENTROID = "FARTHEST_CENTROID"
    PLUGGABLE = "PLUGGABLE"


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


__all__ = [
    "DiscoveryMethod",
    "RegionDiscoveryConfig",
    "discover_structural_regions",
]
