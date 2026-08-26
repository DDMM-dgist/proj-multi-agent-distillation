"""Framework V2 -- GENERIC, material-agnostic raw-structure representation (FE-027 P1).

FE-026 shipped the full autonomous-acquisition pipeline but every path still needed a
material-specific ``StructuralDescriptorProvider`` plugin to turn raw structures into a
``DescriptorSpaceEvidence`` bundle. A brand-new material therefore could not run without a
human first authoring that plugin -- the ``OBJECTIVE_TO_EVIDENCE_SYNTHESIS_AUTONOMY_GAP``.

This module closes the keystone of that gap: a deterministic representation that is computed
ENTIRELY from raw structural facts -- atomic species, Cartesian positions, the cell, and the
periodic-boundary flags -- with NO material name, NO phase label, NO hard-coded element-pair
cutoff (e.g. no "Si-O 2.0 A"), and NO dataset-specific category semantics ("bulk_amo",
"liquid", "vacancy" are treated as opaque source-category strings, never interpreted). The
same code produces a usable descriptor space for SiO2-x and for an arbitrary synthetic
material, which is exactly what the portability requirement demands.

What is generic here:

  * The RAW POOL is located from the run's own frozen inputs by *schema detection* of a
    sanitized-pool manifest (a JSON enumerating source-category files + frame counts + SHAs).
    The manifest's ``category`` strings are opaque provenance labels; this module never reads
    meaning into them.
  * Per-frame FEATURES are four always-computable, unit-carrying continuous scalars:
    ``n_atoms``, ``number_density_atoms_per_A3`` (only when a finite periodic cell volume
    exists), ``mean_min_neighbor_distance_A`` (mean over atoms of the minimum image distance to
    the nearest other atom), and ``max_species_fraction`` (the composition fraction of the most
    abundant species, a scalar in (0, 1]). None of these names an element or a phase.
  * REGIMES are discovered by the generic gap-splitting discoverer
    (:func:`framework_v2.domain_discovery.discover_domain`); nothing here fixes a regime count.

Representation adequacy is treated comparatively (Section 3): a primary representation and a
deliberately coarser alternative are both built; if the primary does not *discriminate* the
pool (it collapses everything into a single regime) the caller fails closed with
``REPRESENTATION_INSUFFICIENT`` and recovers to the alternative rather than asking a human for
a descriptor.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Optional

from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
from framework_v2.contracts import DeploymentScopeContract, DomainRepresentation
from framework_v2.domain_discovery import DiscoveryConfig, SourceItem, discover_domain
from framework_v2.representation_adequacy import (
    RepresentationAdequacyEvidence,
    RepresentationSpec,
    assess_representation_adequacy,
)
from framework_v2.states import SemanticState

# The generic descriptor-space axes. Deliberately small, always computable, unit-carrying, and
# free of any element name / phase / cutoff. ``number_density`` may be missing for a
# non-periodic frame (recorded as an evidence gap by the discoverer, never imputed).
PRIMARY_CONTINUOUS_VARIABLES = (
    "n_atoms",
    "number_density_atoms_per_A3",
    "mean_min_neighbor_distance_A",
    "max_species_fraction",
)
# The coarser fallback representation: composition + size only (drops the two geometry axes).
# Used as the comparative alternative and as the recovery target when the primary is
# non-discriminative.
ALTERNATIVE_CONTINUOUS_VARIABLES = (
    "n_atoms",
    "max_species_fraction",
)

GENERIC_PROVENANCE = "generic_raw_structure_v1"


@dataclasses.dataclass(frozen=True)
class PoolFrame:
    """One raw structure drawn from the source pool, with its generic features already computed.

    ``category`` is the opaque source-category label the manifest recorded; ``frame_index`` is
    the frame's position within that category file. ``item_id`` is ``"{category}#{index}"`` --
    stable and content-independent so selection/provenance downstream is reproducible.
    """
    item_id: str
    category: str
    frame_index: int
    n_atoms: int
    features: dict


@dataclasses.dataclass(frozen=True)
class LoadedPool:
    """The raw pool loaded from the run's frozen inputs, plus its provenance."""
    manifest_path: str
    manifest_sha256: str
    total_frames: int
    frames: tuple[PoolFrame, ...]
    per_category_counts: dict

    def source_items(self, continuous_variables) -> list[SourceItem]:
        wanted = set(continuous_variables)
        return [
            SourceItem(item_id=f.item_id,
                       features={k: v for k, v in f.features.items() if k in wanted})
            for f in self.frames
        ]


# --------------------------------------------------------------------------------------------
# Raw-pool location (schema detection, never filename/material detection)
# --------------------------------------------------------------------------------------------
def _looks_like_pool_manifest(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("total_frames"), int):
        return False
    cats = obj.get("categories")
    if not isinstance(cats, list) or not cats:
        return False
    for c in cats:
        if not isinstance(c, dict):
            return False
        if not isinstance(c.get("sanitized_file"), str):
            return False
        if not isinstance(c.get("n_frames"), int):
            return False
    return True


def _iter_input_paths(controller):
    for rec in controller.state.get("inputs", []) or []:
        if not isinstance(rec, dict):
            continue
        # Prefer the run-bound snapshot when present; else the (hash-bound) source in place.
        for key in ("snapshot", "source"):
            p = rec.get(key)
            if p:
                yield Path(p)
                break


def locate_pool_manifest(controller, *, pool_manifest_path=None):
    """Find the single source-pool manifest among the run's frozen inputs by SCHEMA.

    Returns ``(manifest_path, manifest_dict)``. Fails closed with a typed
    ``AcquisitionCapabilityGap`` when zero inputs parse as a pool manifest (no admissible raw
    pool -> the framework cannot synthesize a representation and will NOT ask a human for a
    descriptor) or when more than one does (ambiguous -- refuses to silently pick).

    ``pool_manifest_path`` (post-split augmentation only) pins the pool to one EXPLICIT
    schema-valid manifest rather than scanning the run's frozen inputs. This is how the SAME
    generic descriptor pipeline is rebound from the Stage-3 acquisition candidate pool onto the
    frozen Stage-6 TRAIN-parent population without colliding with the already-bound acquisition
    pool manifest (which would otherwise trip the AMBIGUOUS_SOURCE_POOL guard). The pinned file
    must itself be a schema-valid pool manifest; it is validated exactly like a discovered one."""
    if pool_manifest_path is not None:
        path = Path(pool_manifest_path)
        if not path.is_file():
            raise AcquisitionCapabilityGap(
                f"the pinned pool manifest does not exist: {path}",
                gap_kind="NO_SOURCE_POOL")
        try:
            obj = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            raise AcquisitionCapabilityGap(
                f"the pinned pool manifest is unreadable/not JSON: {path} ({exc})",
                gap_kind="NO_SOURCE_POOL")
        if not _looks_like_pool_manifest(obj):
            raise AcquisitionCapabilityGap(
                f"the pinned pool manifest does not match the source-pool schema: {path}",
                gap_kind="NO_SOURCE_POOL")
        return path, obj
    found = []
    for path in _iter_input_paths(controller):
        if path.suffix.lower() != ".json" or not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if _looks_like_pool_manifest(obj):
            found.append((path, obj))
    if not found:
        raise AcquisitionCapabilityGap(
            "no source-pool manifest is present among the run's frozen inputs; the generic "
            "representation path cannot locate raw structures to build a descriptor space from",
            gap_kind="NO_SOURCE_POOL")
    if len(found) > 1:
        raise AcquisitionCapabilityGap(
            f"more than one source-pool manifest is present ({[str(p) for p, _ in found]}); "
            "refusing to silently disambiguate the raw pool",
            gap_kind="AMBIGUOUS_SOURCE_POOL")
    return found[0]


# --------------------------------------------------------------------------------------------
# Generic per-frame feature computation (from species / positions / cell / pbc ONLY)
# --------------------------------------------------------------------------------------------
def _shortest_periodic_image_distance(atoms) -> Optional[float]:
    """Length of the shortest non-zero lattice translation over the PERIODIC directions (angstrom).

    This is the nearest-image distance of any atom to its OWN periodic copies. For a one-atom
    periodic cell -- a crystalline primitive cell holds a single basis atom but periodicity
    guarantees images -- this is the physically correct nearest-neighbour distance; without it the
    descriptor would be spuriously undefined. Computed generically from the cell and PBC flags via a
    deterministic bounded integer-combination search over the periodic lattice vectors (no element,
    no phase, no cutoff). Returns None when no direction is periodic. Only directions flagged
    periodic contribute, so partial PBC (e.g. a slab) is handled correctly.
    """
    import itertools

    import numpy as np

    pbc = np.asarray(atoms.pbc, dtype=bool)
    if not pbc.any():
        return None
    cell = np.asarray(atoms.cell.array if hasattr(atoms.cell, "array") else atoms.cell, dtype=float)
    dims = [i for i in range(3) if pbc[i]]
    # A +-3 window over each periodic lattice direction robustly contains the shortest lattice
    # vector for any well-formed (near-reduced) cell; a degenerate zero-length lattice vector simply
    # contributes no admissible (non-zero) translation. Non-periodic directions are pinned to 0.
    ranges = [range(-3, 4) if i in dims else (0,) for i in range(3)]
    best = math.inf
    for combo in itertools.product(*ranges):
        if not any(combo):
            continue
        length = float(np.linalg.norm(np.asarray(combo, dtype=float) @ cell))
        if 0.0 < length < best:
            best = length
    return best if math.isfinite(best) else None


def _mean_min_neighbor_distance(atoms) -> Optional[float]:
    """Mean over atoms of the minimum-image distance to the nearest NEIGHBOUR (angstrom).

    The nearest neighbour of an atom is the closest of (a) any OTHER atom under the minimum-image
    convention and (b) -- for a periodic cell -- the atom's own nearest periodic image. Including
    (b) makes the descriptor physically correct for a one-atom periodic cell. No cutoff, no element
    pair -- purely geometric. Returns None only when the distance is genuinely undefined: an
    isolated NON-periodic single atom (no other atom and no periodic image), which the caller
    records as an evidence gap, never imputed.
    """
    import numpy as np

    n = len(atoms)
    if n == 0:
        return None
    periodic = bool(np.asarray(atoms.pbc, dtype=bool).any())

    nearest = np.full(n, np.inf)
    if n >= 2:
        d = atoms.get_all_distances(mic=periodic)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)

    if periodic:
        self_image = _shortest_periodic_image_distance(atoms)
        if self_image is not None:
            nearest = np.minimum(nearest, self_image)

    if not np.isfinite(nearest).all():
        return None
    return float(nearest.mean())


def _number_density(atoms) -> Optional[float]:
    """Atoms per cubic angstrom, only when a finite positive periodic cell volume exists."""
    import numpy as np

    if not bool(getattr(atoms, "pbc", None) is not None and atoms.pbc.any()):
        return None
    vol = float(atoms.get_volume())
    if not math.isfinite(vol) or vol <= 0.0:
        return None
    return float(len(atoms)) / vol


def _max_species_fraction(atoms) -> float:
    """Composition fraction of the most abundant species -- a scalar in (0, 1]."""
    from collections import Counter

    symbols = list(atoms.get_chemical_symbols())
    counts = Counter(symbols)
    return max(counts.values()) / float(len(symbols))


def compute_frame_features(atoms) -> dict:
    """The generic feature vector for one frame. Missing (uncomputable) axes are simply absent
    -- never imputed -- so the discoverer records them as evidence gaps."""
    feats: dict = {"n_atoms": int(len(atoms)), "max_species_fraction": _max_species_fraction(atoms)}
    dens = _number_density(atoms)
    if dens is not None:
        feats["number_density_atoms_per_A3"] = dens
    nn = _mean_min_neighbor_distance(atoms)
    if nn is not None:
        feats["mean_min_neighbor_distance_A"] = nn
    return feats


# --------------------------------------------------------------------------------------------
# Pool loading + representation building
# --------------------------------------------------------------------------------------------
def load_pool(controller, *, max_frames_per_category: Optional[int] = None,
              pool_manifest_path=None) -> LoadedPool:
    """Load raw frames from the schema-detected pool manifest and compute generic features.

    ``max_frames_per_category`` optionally caps how many frames per category are read (a
    deterministic head-slice) so planning-time descriptor computation stays bounded on very
    large pools; None reads all. The cap is a bounded-compute knob, not a scientific choice.
    ``pool_manifest_path`` pins the pool to one explicit schema-valid manifest (post-split
    TRAIN-parent augmentation); see ``locate_pool_manifest``."""
    from ase.io import read as ase_read

    manifest_path, manifest = locate_pool_manifest(
        controller, pool_manifest_path=pool_manifest_path)
    manifest_dir = manifest_path.parent
    manifest_sha = manifest.get("sanitized_pool_manifest_sha256") or ""

    frames: list[PoolFrame] = []
    per_category_counts: dict = {}
    for cat in manifest["categories"]:
        category = str(cat["category"])
        rel = str(cat["sanitized_file"])
        fpath = (manifest_dir / rel)
        if not fpath.is_file():
            raise AcquisitionCapabilityGap(
                f"source-pool manifest references a structure file that is missing: {fpath}",
                gap_kind="POOL_FILE_MISSING")
        index_spec = ":" if max_frames_per_category is None else f":{int(max_frames_per_category)}"
        atoms_list = ase_read(str(fpath), index=index_spec)
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        per_category_counts[category] = per_category_counts.get(category, 0) + len(atoms_list)
        for idx, atoms in enumerate(atoms_list):
            frames.append(PoolFrame(
                item_id=f"{category}#{idx}", category=category, frame_index=idx,
                n_atoms=int(len(atoms)), features=compute_frame_features(atoms)))
    if not frames:
        raise AcquisitionCapabilityGap(
            "the source-pool manifest enumerated zero readable structures; cannot build a "
            "representation from an empty pool",
            gap_kind="EMPTY_SOURCE_POOL")
    return LoadedPool(
        manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
        total_frames=len(frames), frames=tuple(frames),
        per_category_counts=per_category_counts)


def _discovery_config(continuous_variables) -> DiscoveryConfig:
    # Versioned framework parameters (not per-material magic): the split cap and gap
    # significance are portability-stable defaults documented as framework knobs.
    return DiscoveryConfig(
        continuous_variables=tuple(continuous_variables),
        categorical_variables=(),
        max_intervals_per_continuous_axis=4,
        gap_significance=0.15,
        min_items_per_regime=1,
    )


def build_representation(
    pool: LoadedPool, *, representation_id: str, descriptor: str, continuous_variables,
    scope_contract: DeploymentScopeContract, region_classifier=None,
) -> DomainRepresentation:
    """Discover a ``DomainRepresentation`` over the pool for the given continuous axes."""
    config = _discovery_config(continuous_variables)
    items = pool.source_items(continuous_variables)
    return discover_domain(
        representation_id=representation_id, descriptor=descriptor, items=items,
        config=config, scope_contract=scope_contract, region_classifier=region_classifier,
        evidence_ref=f"pool_manifest:{pool.manifest_sha256}")


def _spec_for(representation: DomainRepresentation, *, spec_id: str, continuous_variables,
              scope_contract: DeploymentScopeContract) -> RepresentationSpec:
    return RepresentationSpec(
        spec_id=spec_id, descriptor=representation.descriptor, kind=representation.kind,
        continuous_variables=list(continuous_variables), categorical_variables=[],
        provenance=GENERIC_PROVENANCE,
        scope_contract_sha256=scope_contract.content_sha256())


@dataclasses.dataclass(frozen=True)
class GenericRepresentationResult:
    """The chosen representation + its first-class adequacy assessment + provenance."""
    representation: DomainRepresentation
    spec: RepresentationSpec
    adequacy: object          # RepresentationAdequacyAssessment
    pool: LoadedPool
    recovered_from_primary: bool


def build_adequate_representation(
    pool: LoadedPool, *, id_prefix: str, scope_contract: DeploymentScopeContract,
    deployment_claim: str, region_classifier=None,
) -> GenericRepresentationResult:
    """Build the primary generic representation, assess adequacy COMPARATIVELY against a coarser
    alternative, and -- if the primary is non-discriminative -- recover to the alternative.

    Fails closed with ``AcquisitionCapabilityGap(REPRESENTATION_INSUFFICIENT)`` only if NEITHER
    representation discriminates the pool. It never asks a human for a descriptor."""
    primary = build_representation(
        pool, representation_id=f"{id_prefix}-repr-primary",
        descriptor="generic raw-structure descriptor space (size/density/geometry/composition)",
        continuous_variables=PRIMARY_CONTINUOUS_VARIABLES, scope_contract=scope_contract,
        region_classifier=region_classifier)
    alternative = build_representation(
        pool, representation_id=f"{id_prefix}-repr-alt",
        descriptor="generic coarse raw-structure descriptor space (size/composition only)",
        continuous_variables=ALTERNATIVE_CONTINUOUS_VARIABLES, scope_contract=scope_contract,
        region_classifier=region_classifier)

    primary_spec = _spec_for(primary, spec_id=f"{id_prefix}-spec-primary",
                             continuous_variables=PRIMARY_CONTINUOUS_VARIABLES,
                             scope_contract=scope_contract)
    alt_spec = _spec_for(alternative, spec_id=f"{id_prefix}-spec-alt",
                         continuous_variables=ALTERNATIVE_CONTINUOUS_VARIABLES,
                         scope_contract=scope_contract)

    def _discriminative(rep) -> bool:
        # A representation that collapses the whole pool into ONE regime discriminates nothing;
        # more than one discovered regime is deterministic evidence of discriminative power.
        return len(rep.regimes) > 1

    def _assess(rep, spec, alt_id):
        evidence = [RepresentationAdequacyEvidence(
            evidence_id=f"{spec.spec_id}-discrimination",
            kind="discriminative_power",
            description=(f"the representation discovered {len(rep.regimes)} distinct regimes over "
                         f"{pool.total_frames} pooled structures"),
            supports_adequacy=_discriminative(rep),
            fact_refs=[f"pool_manifest:{pool.manifest_sha256}"])]
        return assess_representation_adequacy(
            assessment_id=f"{spec.spec_id}-adequacy",
            representation_sha256=rep.content_sha256(),
            scope_contract_sha256=scope_contract.content_sha256(),
            deployment_claim=deployment_claim,
            adequacy_evidence=evidence,
            alternatives_considered=[alt_id],
            rationale=("discriminative power measured as the number of regimes the generic "
                       "gap-splitting discoverer resolves over the raw pool"))

    primary_adequacy = _assess(primary, primary_spec, alt_spec.spec_id)
    if primary_adequacy.verdict == SemanticState.PASS:
        return GenericRepresentationResult(
            representation=primary, spec=primary_spec, adequacy=primary_adequacy,
            pool=pool, recovered_from_primary=False)

    # Primary insufficient -> recover to the coarser alternative (never ask a human).
    alt_adequacy = _assess(alternative, alt_spec, primary_spec.spec_id)
    if alt_adequacy.verdict == SemanticState.PASS:
        return GenericRepresentationResult(
            representation=alternative, spec=alt_spec, adequacy=alt_adequacy,
            pool=pool, recovered_from_primary=True)

    raise AcquisitionCapabilityGap(
        "neither the primary nor the coarser generic representation discriminates the raw pool "
        "(both collapse it into a single regime); the generic representation path is insufficient "
        "for this pool and recovers to no admissible alternative",
        gap_kind="REPRESENTATION_INSUFFICIENT")


# --------------------------------------------------------------------------------------------
# FE-029 -- representation/sizing missing-axis compatibility (no frame dropping, no imputation)
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SizingRepresentation:
    """The representation the deterministic FPS labeling-population sizing runs on.

    ``axes`` are the descriptor axes present on EVERY eligible frame, so a dense sizing matrix is
    built with NO imputed value and NO dropped frame -- every eligible structure participates.
    ``adequacy`` is the first-class :class:`RepresentationAdequacyAssessment` for this (possibly
    axis-reduced) representation; sizing runs only when it PASSes. ``reduced`` records whether axes
    were dropped from the candidate because they were not universally available across the pool,
    and ``per_axis_available_count`` is the availability audit that justified the choice."""
    spec: object              # RepresentationSpec
    adequacy: object          # RepresentationAdequacyAssessment
    axes: tuple
    reduced: bool
    per_axis_available_count: dict


def audit_axis_availability(pool: LoadedPool, continuous_variables) -> dict:
    """Deterministic per-axis count of eligible frames that carry that axis.

    An axis is universally available iff its count equals ``pool.total_frames``. This is the audit
    that lets the sizing step choose an admissible dense axis set WITHOUT silently dropping frames
    or imputing missing values."""
    counts = {ax: 0 for ax in continuous_variables}
    for frame in pool.frames:
        for ax in continuous_variables:
            if ax in frame.features:
                counts[ax] += 1
    return counts


def derive_admissible_sizing_representation(
    pool: LoadedPool, *, candidate_spec, scope_contract: DeploymentScopeContract,
    deployment_claim: str, id_prefix: str, region_classifier=None,
) -> SizingRepresentation:
    """Derive the representation the deterministic FPS sizing will use, compatibly and safely.

    The FPS sizing needs a DENSE descriptor matrix, but a candidate representation axis (e.g. a
    geometry axis) may be genuinely uncomputable for some eligible frame (recorded as an evidence
    gap, never imputed). Rather than fail closed on the first such frame, drop the frame, or drop an
    axis silently, this derives -- deterministically -- the axis set that keeps EVERY eligible frame
    while remaining scientifically adequate:

      1. Axis-availability audit over ALL eligible frames.
      2. Ordered, de-duplicated trial axis-sets, most-faithful first: the FULL candidate axes (only
         if universally available), then the universally-available subset, then the coarse
         ALTERNATIVE axes intersected with availability.
      3. For each trial set that is non-empty and present on EVERY frame, build the representation
         and run RepresentationAdequacyEvidence; accept the first whose verdict is PASS.
      4. If none passes, fail closed with a typed
         ``AcquisitionCapabilityGap(REPRESENTATION_INSUFFICIENT)``.

    Because every accepted axis is present on every frame, the dense sizing matrix contains ALL
    eligible frames -- axis incompleteness never erases a structure from the scientific population
    (coverage / provenance / protected-reference exclusion continue to account for every frame).
    An axis-reduced representation is admitted ONLY when adequacy still holds; this is never a
    silent removal, and it is never a request for a per-material human descriptor choice."""
    counts = audit_axis_availability(pool, candidate_spec.continuous_variables)
    n = pool.total_frames

    trials: list[tuple] = []

    def _add(axes) -> None:
        axes = tuple(axes)
        if axes and axes not in trials and all(counts.get(ax, 0) == n for ax in axes):
            trials.append(axes)

    _add(candidate_spec.continuous_variables)  # full candidate (added only if universally present)
    _add(tuple(ax for ax in candidate_spec.continuous_variables if counts.get(ax, 0) == n))
    _add(tuple(ax for ax in ALTERNATIVE_CONTINUOUS_VARIABLES if counts.get(ax, 0) == n))

    for axes in trials:
        rep = build_representation(
            pool, representation_id=f"{id_prefix}-sizing-repr",
            descriptor=candidate_spec.descriptor,
            continuous_variables=axes, scope_contract=scope_contract,
            region_classifier=region_classifier)
        spec = _spec_for(rep, spec_id=f"{id_prefix}-sizing-spec",
                         continuous_variables=axes, scope_contract=scope_contract)
        n_regimes = len(rep.regimes)
        evidence = [RepresentationAdequacyEvidence(
            evidence_id=f"{spec.spec_id}-discrimination",
            kind="discriminative_power",
            description=(f"the sizing representation over axes {list(axes)} discovered {n_regimes} "
                         f"distinct regimes across all {n} eligible structures with no imputed "
                         f"value and no dropped frame"),
            supports_adequacy=n_regimes > 1,
            fact_refs=[f"pool_manifest:{pool.manifest_sha256}"])]
        adequacy = assess_representation_adequacy(
            assessment_id=f"{spec.spec_id}-adequacy",
            representation_sha256=rep.content_sha256(),
            scope_contract_sha256=scope_contract.content_sha256(),
            deployment_claim=deployment_claim,
            adequacy_evidence=evidence,
            alternatives_considered=[f"candidate:{candidate_spec.spec_id}"],
            rationale=("axis-availability-audited sizing representation; axes are reduced only when "
                       "not universally available, and a reduced set is admitted only when its "
                       "discriminative adequacy still holds"))
        if adequacy.verdict == SemanticState.PASS:
            return SizingRepresentation(
                spec=spec, adequacy=adequacy, axes=axes,
                reduced=(axes != tuple(candidate_spec.continuous_variables)),
                per_axis_available_count=dict(counts))

    raise AcquisitionCapabilityGap(
        "no axis subset of the candidate representation is both universally available across every "
        "eligible frame and scientifically adequate (discriminative) for labeling-population "
        "sizing; the generic representation is insufficient for this pool and recovers to no "
        "admissible alternative (no frame was dropped and no value was imputed)",
        gap_kind="REPRESENTATION_INSUFFICIENT")
