"""Exact nearest-neighbor `SearchBackend` backed by `scipy.spatial.cKDTree`.

This is the promoted, backend-only form of what used to be embedded directly
in `coverage.soap_representation`. `cKDTree` is a generic exact
nearest-neighbor search data structure with no SOAP-, SiO2-, Allegro-, or
Student-specific logic whatsoever, so it now implements
`coverage.search_backend.SearchBackend` directly and can back ANY
`CoverageRepresentation`'s vectors, not only SOAP's.

Always exact (`is_exact() == True`) -- this module does not implement or
propose an approximate backend; see `coverage.search_backend`'s module
docstring for the accuracy-validation requirement any future approximate
backend must satisfy before it may replace this one in production use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from coverage.search_backend import SearchResult


@dataclass(frozen=True)
class KDTreeIndex:
    """Opaque `SearchIndex` for `ExactKDTreeBackend`."""

    tree: object
    reference_keys: tuple


class ExactKDTreeBackend:
    """Exact NN search via `scipy.spatial.cKDTree`."""

    def __init__(self, workers: int = -1):
        if not isinstance(workers, int) or isinstance(workers, bool):
            raise ValueError("ExactKDTreeBackend.workers must be an int")
        self._workers = workers

    def build_index(self, vectors, environment_keys: Sequence) -> KDTreeIndex:
        from scipy.spatial import cKDTree

        vectors = np.asarray(vectors, dtype=np.float64)
        environment_keys = tuple(environment_keys)
        if vectors.shape[0] == 0:
            raise ValueError("ExactKDTreeBackend.build_index requires at least one reference vector")
        if vectors.shape[0] != len(environment_keys):
            raise ValueError(
                "ExactKDTreeBackend.build_index requires exactly one environment_key per vector row"
            )
        return KDTreeIndex(tree=cKDTree(vectors), reference_keys=environment_keys)

    def query_nearest(self, index: KDTreeIndex, query_vectors) -> list:
        query_vectors = np.atleast_2d(np.asarray(query_vectors, dtype=np.float64))
        if query_vectors.shape[0] == 0:
            return []
        distances, positions = index.tree.query(query_vectors, k=1, workers=self._workers)
        distances = np.atleast_1d(distances)
        positions = np.atleast_1d(positions)
        return [
            SearchResult(distance=float(d), reference_key=index.reference_keys[int(p)])
            for d, p in zip(distances, positions)
        ]

    def is_exact(self) -> bool:
        return True

    def provenance(self) -> dict:
        import scipy

        return {
            "backend": "exact_kdtree",
            "library": "scipy.spatial.cKDTree",
            "scipy_version": scipy.__version__,
            "is_exact": True,
            "workers": self._workers,
        }
