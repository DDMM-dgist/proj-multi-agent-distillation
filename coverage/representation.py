"""Generic local-environment coverage-representation interface.

This is the seam that keeps `coverage.reference_pool`, `coverage.nn_distance`,
`coverage.aggregate`, and `coverage.report` fully representation-agnostic. SOAP
(`coverage.soap_representation.SoapCoverageRepresentation`) is ONE implementation
of `CoverageRepresentation`, not the architecture: a future Teacher-latent-space
representation, or any other local-environment descriptor, can implement the same
protocol and be dropped in without changing Data Curator, Controller, or
DataCoverageReport semantics, and without generic coverage code ever importing
`dscribe` or assuming Euclidean-on-raw-vectors is a valid distance.

`CoverageRepresentation` owns ONLY scientific representation construction:
computing per-environment vectors and identity, and serializing full
provenance/a content hash. It owns no nearest-neighbor search responsibility
at all -- see `coverage.distance_policy.DistancePolicy` for comparison
semantics (normalization, metric, compatibility-rule declarations) and
`coverage.search_backend.SearchBackend` for the index-building/query
mechanics (e.g. `coverage.exact_kdtree_backend.ExactKDTreeBackend`). This
three-way split is what allows, without any change to
`coverage.reference_pool`/`coverage.nn_distance`/`coverage.report`: SOAP +
exact cKDTree; SOAP + a future validated approximate backend; a Teacher-latent
representation + exact backend; or a Teacher-latent representation + a future
approximate backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RepresentationBatch:
    """Per-environment representation vectors plus structure/environment identity.

    `structure_id` identifies the source structure (caller-supplied, e.g. a frame
    index or a stable frame id -- generic code never assumes it is a contiguous
    range). `environment_index` is the position of this environment (e.g. atom)
    within its structure.

    `compatibility_key` is an opaque, representation-defined, per-environment
    hashable tag (e.g. a SOAP representation may derive it from central atomic
    species and/or structure periodicity, depending on its `DistancePolicy`'s
    declared compatibility rules -- see coverage.soap_representation). Generic
    code (coverage.reference_pool, coverage.nn_distance) never interprets what
    this key means; it only groups environments that share an identical key
    into the same search index, and treats "no reference environment anywhere
    shares this key" as quantitative "unmatched" evidence, never an error.
    """

    vectors: "object"  # backend-defined array, shape (n_environments, n_features)
    structure_id: tuple
    environment_index: tuple
    compatibility_key: tuple

    def __post_init__(self):
        n = len(self.structure_id)
        if not (len(self.environment_index) == n and len(self.compatibility_key) == n):
            raise ValueError(
                "RepresentationBatch.structure_id, environment_index, and "
                "compatibility_key must all have the same length as the number of "
                "environments"
            )
        if len(self.vectors) != n:
            raise ValueError("RepresentationBatch.vectors must have one row per environment")

    def __len__(self) -> int:
        return len(self.structure_id)

    def environment_ids(self) -> tuple:
        """Canonical per-environment identity: `(structure_id, environment_index)`
        pairs, unique within this batch. This is the single source-of-truth key
        every reference pool / search-backend index uses to refer back to an
        environment's vector -- slice/domain membership always references these
        ids (or, equivalently, their positions within one canonical batch) rather
        than requiring a duplicated copy of the vector itself (see
        coverage.reference_pool).
        """
        return tuple(zip(self.structure_id, self.environment_index))


class CoverageRepresentation(Protocol):
    """Protocol every local-environment coverage representation must implement.

    Responsibilities, all representation-owned:
    * compute a RepresentationBatch (vectors + identity + compatibility_key)
      from structures;
    * serialize full method provenance (parameters, library version, a content
      hash covering every scientific choice).

    Explicitly NOT this protocol's responsibility: building or querying a
    nearest-neighbor search index (see coverage.search_backend.SearchBackend),
    or defining comparison semantics such as normalization/metric/compatibility
    rules (see coverage.distance_policy.DistancePolicy).
    """

    def compute(self, structures: Sequence, structure_ids: Sequence = None) -> RepresentationBatch:
        """Compute per-environment representation vectors for `structures`.

        `structure_ids` defaults to range(len(structures)) if not given.
        """
        ...

    def provenance(self) -> dict:
        """Full, JSON-serializable record of every representation choice."""
        ...

    def representation_hash(self) -> str:
        """Content hash covering every scientific parameter this representation uses.

        Two representation instances with the same hash must be guaranteed to
        produce numerically identical vectors for the same input structures --
        this is part of what licenses reusing a cached reference pool across
        runs (see coverage.reference_pool).
        """
        ...
