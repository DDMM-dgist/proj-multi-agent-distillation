"""Generic directed nearest-neighbor distance evidence.

For every query environment, queries the reference pool's mandatory global
compatibility-key indices (see coverage.reference_pool) plus every per-slice
sub-index that actually exists in that reference pool, and keeps the full
per-slice distance vector alongside the global distance. This is what lets
both an environment-pooled view (the global distance) and any slice/domain-
balanced view (an equal- or custom-weighted combination across whatever
slices exist) be reconstructed later from the same stored raw evidence,
without rebuilding the reference pool or re-running any NN search. Which view
to act on is a later, human-approved policy step (see coverage.aggregate).

Every record is tagged with an explicit `direction`, `query_population`, and
`reference_population` so the caller (coverage.report) can never accidentally
reverse which side was queried against which -- see the module docstrings for
coverage.representation and coverage.reference_pool for why this is required
rather than inferred. This module never reads `config_type` or any other
fixed metadata field directly; query-side slice labels are supplied externally
via `query_slice_labels`.

This module has no dependency on any `CoverageRepresentation` implementation
or the `dscribe`/`scipy` packages -- it only consumes the generic
`DistancePolicy` (to normalize query vectors the same way reference vectors
were normalized) and `SearchBackend` (to actually query each already-built
index) protocols, plus the reference pool's own compatibility_key-bucketed
indices.

Unmatched/incompatible evidence: a query environment is "unmatched" against a
given level (global, or a particular reference slice) exactly when that level
has NO search index at all for the query environment's compatibility_key
(e.g. no reference environment anywhere shares its central species, under a
central-species-matching policy -- or, symmetrically, no reference environment
anywhere shares its periodicity, under a periodicity-consistency policy). This
is real, quantitative evidence (see coverage.aggregate's `n_unmatched`/
`unmatched_fraction`), never a silently-dropped record, and -- as of this
refactor -- never a hard crash either: a periodicity mismatch and a
central-species mismatch are now surfaced identically, via the same generic
compatibility_key bucketing mechanism, rather than one being an unhandled
error and the other being ordinary unmatched evidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from coverage.distance_policy import DistancePolicy
from coverage.reference_pool import ReferencePool
from coverage.representation import RepresentationBatch
from coverage.search_backend import SearchBackend


@dataclass(frozen=True)
class EnvironmentDistanceRecord:
    """One query environment's nearest-neighbor distance evidence for one directed
    query_population -> reference_population coverage question."""

    direction: str
    query_population: str
    reference_population: str
    query_structure_id: object
    query_environment_index: object
    query_slice_labels: tuple
    global_distance: "float | None"
    global_matched: bool
    slice_distances: dict  # slice_name -> float | None (only for slices present in reference_pool)
    slice_matched: dict  # slice_name -> bool


def _query_indices_by_key(
    search_backend: SearchBackend, indices_by_key: dict, normalized_vectors, compatibility_keys: tuple
) -> tuple:
    """Query every row of `normalized_vectors` against `indices_by_key` (a
    compatibility_key -> SearchIndex mapping from one reference-pool level),
    grouping by compatibility_key so each distinct key is queried against its
    matching index in one batched call. A row whose key has no matching index
    at this level is unmatched (distance=None) rather than erroring or being
    silently dropped.
    """
    positions_by_key = defaultdict(list)
    for i, key in enumerate(compatibility_keys):
        positions_by_key[key].append(i)

    n = len(compatibility_keys)
    distance = [None] * n
    matched = [False] * n
    for key, positions in positions_by_key.items():
        index = indices_by_key.get(key)
        if index is None:
            continue
        results = search_backend.query_nearest(index, normalized_vectors[positions])
        for pos, result in zip(positions, results):
            distance[pos] = result.distance
            matched[pos] = True
    return tuple(distance), tuple(matched)


def compute_environment_distances(
    direction: str,
    query_population: str,
    reference_pool: ReferencePool,
    distance_policy: DistancePolicy,
    search_backend: SearchBackend,
    query_batch: RepresentationBatch,
    query_slice_labels: dict = None,
) -> list:
    """Query `reference_pool`'s global indices and every existing slice index for
    every environment in `query_batch`.

    `query_slice_labels`, if given, maps a query structure_id to a (possibly
    empty) collection of slice-label strings for reporting/grouping on the
    query side (see coverage.aggregate) -- it never filters or restricts which
    reference index is queried; every query environment is always queried
    against the full global indices and every reference slice's indices that
    exist.
    """
    if not isinstance(direction, str) or not direction.strip():
        raise ValueError("direction must be a non-empty string")
    if not isinstance(query_population, str) or not query_population.strip():
        raise ValueError("query_population must be a non-empty string")
    if query_slice_labels is None:
        query_slice_labels = {}

    normalized_vectors = np.asarray(distance_policy.normalize(query_batch.vectors), dtype=np.float64)

    global_distance, global_matched = _query_indices_by_key(
        search_backend,
        reference_pool.global_indices_by_compatibility_key,
        normalized_vectors,
        query_batch.compatibility_key,
    )

    slice_results = {
        slice_name: _query_indices_by_key(
            search_backend, pool.indices_by_compatibility_key, normalized_vectors, query_batch.compatibility_key
        )
        for slice_name, pool in reference_pool.slices.items()
    }

    records = []
    for i in range(len(query_batch)):
        sid = query_batch.structure_id[i]
        records.append(
            EnvironmentDistanceRecord(
                direction=direction,
                query_population=query_population,
                reference_population=reference_pool.population_role,
                query_structure_id=sid,
                query_environment_index=query_batch.environment_index[i],
                query_slice_labels=tuple(query_slice_labels.get(sid, ())),
                global_distance=global_distance[i],
                global_matched=global_matched[i],
                slice_distances={name: dist[i] for name, (dist, _) in slice_results.items()},
                slice_matched={name: mat[i] for name, (_, mat) in slice_results.items()},
            )
        )
    return records
