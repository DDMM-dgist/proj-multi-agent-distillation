"""Generic scientifically-meaningful comparison-semantics interface.

Split out from `coverage.representation` (vector construction) and
`coverage.search_backend` (index/search mechanics): a `DistancePolicy` owns
ONLY how two already-computed representation vectors are compared --
normalization applied before comparison, and the distance/kernel metric name
-- plus full provenance of both. It never computes vectors and never builds
or queries a search index.

`coverage.soap_distance_policy.SoapDistancePolicy` is ONE implementation, not
the architecture: a future Teacher-latent-space representation can pair with
its own `DistancePolicy` implementation (or reuse this one, if its comparison
semantics genuinely match) without any change to `coverage.reference_pool`,
`coverage.nn_distance`, `coverage.aggregate`, or `coverage.report`.

Compatibility rules (e.g. SOAP's central-species matching, or a periodicity
consistency requirement) are declared as explicit policy fields on the
concrete implementation, but the actual per-environment compatibility data
(e.g. a structure's chemical species, its periodicity) is representation-
specific structure introspection -- so a `CoverageRepresentation` computes
the opaque `compatibility_key` per environment (see
`coverage.representation.RepresentationBatch`) by consulting its own
`DistancePolicy`'s declared rules; `coverage.reference_pool` and
`coverage.nn_distance` only ever group and look up by that already-computed,
opaque key, never interpreting what it means.
"""
from __future__ import annotations

from typing import Protocol


class DistancePolicy(Protocol):
    """Protocol every campaign-specific comparison-semantics policy must implement."""

    def normalize(self, vectors):
        """Return `vectors` transformed for comparison under this policy (e.g.
        L2-normalized so a Euclidean-backed search backend realizes cosine
        distance), or unchanged if no transform applies. Pure function of
        `vectors` and this policy's own fields -- no side effects, no lookups
        into a search index.
        """
        ...

    def provenance(self) -> dict:
        """Full, JSON-serializable record of every comparison-semantics choice."""
        ...

    def content_hash(self) -> str:
        """Content hash covering every field this policy uses.

        Two policy instances with the same hash must produce numerically
        identical `normalize()` output for the same input vectors -- this is
        part of what licenses reusing a cached reference pool across runs
        (see coverage.reference_pool).
        """
        ...
