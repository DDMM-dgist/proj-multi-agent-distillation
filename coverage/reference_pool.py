"""Generic reference pools for nearest-neighbor coverage evidence.

A reference pool is built from ONE explicit reference structure set (e.g. the
Teacher-train partition, the frozen deployment-target population, or the
current Student/distillation dataset) via an injected `CoverageRepresentation`
(see coverage.representation), `DistancePolicy` (see coverage.distance_policy),
and `SearchBackend` (see coverage.search_backend). This module never imports
`dscribe`, `scipy`, or any representation-/backend-specific module, and never
assumes SOAP is the only descriptor or cKDTree is the only search backend.

Canonical vector storage: every reference pool stores each environment's
representation vector EXACTLY ONCE, in `canonical_batch` (already transformed
by `distance_policy.normalize()`, since that is the form every index actually
searches over). Slice membership never duplicates vectors -- a `SlicePool`
records only which canonical environment POSITIONS belong to that slice
(`environment_positions`), so overlapping, multi-membership slices cost extra
index-building/search time and storage (see PRODUCTION_COST_ESTIMATE.md), not
extra copies of the underlying scientific evidence. A search backend's own
index structure (e.g. a cKDTree's internal copy of the vectors it was built
from) is a derived/cache artifact of that index, not the source of truth.

Distance-policy-driven compatibility grouping (e.g. SOAP central-species
matching) is handled generically here via each environment's opaque
`compatibility_key` (see coverage.representation.RepresentationBatch) -- this
module never interprets what a compatibility_key means, it only groups
environments that share an identical key into the same search index, at both
the mandatory global (population-wide) level and within every per-slice
sub-pool.

Every reference pool always has a mandatory global set of search indices
(one per distinct compatibility_key present in the whole population) built
from the entire reference structure set. It may additionally have per-slice
sub-pools, one per caller-supplied slice label. Slice membership is supplied
externally via `slice_membership` (a parallel, per-structure sequence of
zero/one/many labels) -- this module has no notion of `config_type` or any
other fixed metadata field; a structure may belong to zero, one, or several
slices, decided entirely by whatever campaign-specific domain/slice adapter
the caller used to build `slice_membership` (see coverage.adapters).

Building both a global set of indices and per-slice sub-indices from the same
canonical batch (rather than only ever querying per-slice indices) is
deliberate: with overlapping, non-partitioning slices, no combination of
per-slice indices is guaranteed to reconstruct "distance to the nearest
reference environment overall," so the global level is not optional.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from coverage.distance_policy import DistancePolicy
from coverage.representation import CoverageRepresentation, RepresentationBatch
from coverage.search_backend import SearchBackend


@dataclass(frozen=True)
class SlicePool:
    """One slice label's reference environments.

    `environment_positions` are index positions into the reference pool's
    `canonical_batch` -- no vector duplication. `indices_by_compatibility_key`
    holds one backend-built `SearchIndex` per distinct compatibility_key
    present among this slice's environments.
    """

    slice_name: str
    environment_positions: tuple
    n_atoms: int
    n_frames: int
    indices_by_compatibility_key: dict  # compatibility_key -> SearchIndex


@dataclass(frozen=True)
class ReferencePool:
    """A complete reference pool: canonical per-environment vector storage,
    plus a mandatory global (population-wide) set of compatibility-key search
    indices, plus optional per-slice sub-pools."""

    population_role: str
    representation_hash: str
    representation_provenance: dict
    search_backend_provenance: dict
    reference_manifest_sha256: str
    canonical_batch: RepresentationBatch
    global_indices_by_compatibility_key: dict  # compatibility_key -> SearchIndex
    slices: dict  # slice_name -> SlicePool
    total_atoms: int
    total_frames: int

    def slice_names(self) -> tuple:
        return tuple(sorted(self.slices))

    def slice_counts(self) -> dict:
        return {
            name: {"n_atoms": pool.n_atoms, "n_frames": pool.n_frames}
            for name, pool in self.slices.items()
        }


def _build_indices_by_compatibility_key(
    search_backend: SearchBackend, batch: RepresentationBatch, positions: Sequence[int]
) -> dict:
    buckets = defaultdict(list)
    for pos in positions:
        buckets[batch.compatibility_key[pos]].append(pos)
    environment_ids = batch.environment_ids()
    indices: dict = {}
    for key, bucket_positions in buckets.items():
        vectors = batch.vectors[bucket_positions]
        keys = [environment_ids[pos] for pos in bucket_positions]
        indices[key] = search_backend.build_index(vectors, keys)
    return indices


def build_reference_pool(
    population_role: str,
    representation: CoverageRepresentation,
    distance_policy: DistancePolicy,
    search_backend: SearchBackend,
    structures: Sequence,
    reference_manifest_sha256: str,
    structure_ids: Sequence = None,
    slice_membership: Sequence = None,
) -> ReferencePool:
    """Build a reference pool from `structures`.

    `population_role` is a free-form, campaign-supplied label (e.g.
    "teacher_train_partition", "deployment_target_population",
    "student_training_dataset", "candidate_population") -- this module does not
    restrict it to a fixed enum; conditional validation rules (such as which
    role requires validation/test exclusion) live in validation/data_coverage.py,
    not here.

    `structures` must already be the exact, pre-filtered reference structure set
    for `population_role` -- this function does not itself know which frames
    belong to which partition; that membership decision is made by the caller
    and recorded via `reference_manifest_sha256`.

    `slice_membership`, if given, must have the same length as `structures`,
    with each entry being a (possibly empty) collection of slice-label strings
    for that structure. A `SlicePool` is built for every label that appears at
    least once, in addition to the always-built global indices.
    """
    if not isinstance(population_role, str) or not population_role.strip():
        raise ValueError("population_role must be a non-empty string")
    if not structures:
        raise ValueError("build_reference_pool requires at least one reference structure")
    if not isinstance(reference_manifest_sha256, str) or not reference_manifest_sha256.strip():
        raise ValueError("reference_manifest_sha256 must be a non-empty string")

    if structure_ids is None:
        structure_ids = list(range(len(structures)))
    else:
        structure_ids = list(structure_ids)
        if len(structure_ids) != len(structures):
            raise ValueError("structure_ids must have the same length as structures")

    if slice_membership is None:
        slice_membership = [() for _ in structures]
    else:
        slice_membership = list(slice_membership)
        if len(slice_membership) != len(structures):
            raise ValueError(
                "slice_membership must have the same length as structures (one entry, "
                "possibly empty, per structure)"
            )

    slice_labels_by_structure_id = {
        sid: tuple(labels) for sid, labels in zip(structure_ids, slice_membership)
    }

    raw_batch = representation.compute(structures, structure_ids=structure_ids)
    normalized_vectors = distance_policy.normalize(raw_batch.vectors)
    # The canonical batch stores distance-policy-normalized vectors, since that
    # is the single form every search index (global and every slice) actually
    # indexes and searches over -- there is exactly one array of vectors this
    # pool ever holds, never a per-slice or per-key copy.
    batch = RepresentationBatch(
        vectors=normalized_vectors,
        structure_id=raw_batch.structure_id,
        environment_index=raw_batch.environment_index,
        compatibility_key=raw_batch.compatibility_key,
    )

    all_positions = list(range(len(batch)))
    global_indices = _build_indices_by_compatibility_key(search_backend, batch, all_positions)

    labels_by_position = [slice_labels_by_structure_id[sid] for sid in batch.structure_id]
    all_labels = sorted({label for labels in labels_by_position for label in labels})

    slices = {}
    for label in all_labels:
        positions = [i for i, labels in enumerate(labels_by_position) if label in labels]
        n_frames = len({batch.structure_id[i] for i in positions})
        slices[label] = SlicePool(
            slice_name=label,
            environment_positions=tuple(positions),
            n_atoms=len(positions),
            n_frames=n_frames,
            indices_by_compatibility_key=_build_indices_by_compatibility_key(
                search_backend, batch, positions
            ),
        )

    return ReferencePool(
        population_role=population_role,
        representation_hash=representation.representation_hash(),
        representation_provenance=representation.provenance(),
        search_backend_provenance=search_backend.provenance(),
        reference_manifest_sha256=reference_manifest_sha256,
        canonical_batch=batch,
        global_indices_by_compatibility_key=global_indices,
        slices=slices,
        total_atoms=len(batch),
        total_frames=len(structures),
    )
