"""Generic nearest-neighbor search-backend interface.

Split out from `coverage.representation` (vector construction) and
`coverage.distance_policy` (comparison semantics): a `SearchBackend` owns
ONLY how a set of already-computed, already-distance-policy-normalized
vectors is indexed and queried for nearest neighbors. It has no notion of
SOAP, chemistry, material system, Teacher/Student, or any other campaign
concept -- it operates purely on raw vectors plus caller-supplied opaque
`environment_keys` (returned verbatim on a hit), declares whether its search
is exact or approximate, and serializes its own backend/library/version/
parameter provenance.

`coverage.exact_kdtree_backend.ExactKDTreeBackend` (scipy.spatial.cKDTree) is
ONE implementation, always exact. A future approximate backend (e.g. an
ANN/FAISS index) can implement this same protocol; nothing in
`coverage.reference_pool`, `coverage.nn_distance`, `coverage.aggregate`, or
`coverage.report` needs to change to use it. An approximate backend must
report `is_exact() == False`, and must never be silently substituted for an
exact backend without an explicit, separately-approved accuracy-validation
policy (see coverage/PRODUCTION_COST_ESTIMATE.md) -- that policy does not
exist yet, and this module does not implement or propose one.

Whether a query environment has ANY eligible reference environment at all
(e.g. no reference environment anywhere shares its compatibility_key) is
decided ONE layer up, by `coverage.reference_pool`/`coverage.nn_distance` --
a `SearchBackend` is only ever asked to query a non-empty index it was
already given, and never makes an "unsupported" determination itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SearchResult:
    """One query vector's nearest-neighbor hit against a `SearchIndex`."""

    distance: float
    reference_key: object


class SearchIndex(Protocol):
    """Opaque backend-defined nearest-neighbor index.

    Generic code never inspects the contents of a SearchIndex; it only passes
    it back into the same backend's `query_nearest`.
    """


class SearchBackend(Protocol):
    """Protocol every nearest-neighbor search backend must implement."""

    def build_index(self, vectors, environment_keys: Sequence) -> SearchIndex:
        """Build an index over `vectors` (shape (n, d)).

        `environment_keys[i]` is the opaque identity returned as
        `SearchResult.reference_key` on a hit against row `i` -- callers use
        this to map a hit back to canonical environment identity (see
        `coverage.representation.RepresentationBatch.environment_ids`)
        without the backend ever knowing what a "structure" or
        "environment" is. Must raise on an empty `vectors` (there is no such
        thing as an index over zero reference environments; the "no eligible
        reference at all" case is handled by the caller never building or
        querying an index for an empty group).
        """
        ...

    def query_nearest(self, index: SearchIndex, query_vectors) -> list:
        """Return one `SearchResult` per row of `query_vectors`, aligned
        index-for-index with the input rows.
        """
        ...

    def is_exact(self) -> bool:
        """True iff `query_nearest` returns the mathematically exact nearest
        neighbor for every query vector; False for any approximate backend.
        """
        ...

    def provenance(self) -> dict:
        """Backend name, library, version, and search parameters (e.g.
        exactness, worker/thread count) -- JSON-serializable.
        """
        ...
