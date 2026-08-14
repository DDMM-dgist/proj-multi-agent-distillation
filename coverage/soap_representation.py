"""SOAP implementation of the generic `CoverageRepresentation` protocol.

Wraps the optional `dscribe` package plus the explicit, no-default
`SoapDescriptorConfig` (coverage.descriptor_config) and `SoapDistancePolicy`
(coverage.soap_distance_policy). This is ONE implementation of
`coverage.representation.CoverageRepresentation` -- generic coverage code
(reference_pool, nn_distance, aggregate, report) never imports this module or
`dscribe` directly; it only holds a `CoverageRepresentation` instance and calls
the protocol methods. Callers that don't have `dscribe` installed get a clear
`ModuleNotFoundError` at call time, not at import time, matching this repo's
existing optional-dependency convention.

This module owns ONLY representation construction: computing SOAP vectors,
provenance, and the representation hash. It contains no cKDTree, no
index-building, and no search responsibility at all -- that mechanic lives in
`coverage.search_backend`/`coverage.exact_kdtree_backend`, fully decoupled
from SOAP. `compute()` is a pure function of its inputs and this instance's
own (immutable) config/policy -- it holds no cross-call mutable state, so the
same `SoapCoverageRepresentation` instance may be reused freely across many
independent reference/query batches.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from coverage.descriptor_config import SoapDescriptorConfig
from coverage.representation import RepresentationBatch
from coverage.soap_distance_policy import SoapDistancePolicy


def dscribe_version() -> str:
    """Return the installed dscribe version, for provenance recording.

    dscribe does not expose `__version__` on the package itself (verified
    against dscribe 2.1.2), so this reads installed package metadata instead.
    Raises ModuleNotFoundError (not caught) if dscribe is not installed.
    """
    import importlib.metadata

    import dscribe  # noqa: F401 -- import error must surface before the metadata lookup

    return importlib.metadata.version("dscribe")


class SoapCoverageRepresentation:
    """SOAP-descriptor `CoverageRepresentation` with explicit, fail-closed distance policy."""

    def __init__(self, descriptor_config: SoapDescriptorConfig, distance_policy: SoapDistancePolicy):
        if not isinstance(descriptor_config, SoapDescriptorConfig):
            raise ValueError("descriptor_config must be a SoapDescriptorConfig instance")
        if not isinstance(distance_policy, SoapDistancePolicy):
            raise ValueError("distance_policy must be a SoapDistancePolicy instance")
        self._descriptor_config = descriptor_config
        self._distance_policy = distance_policy

    def compute(self, structures: Sequence, structure_ids: Sequence = None) -> RepresentationBatch:
        from dscribe.descriptors import SOAP

        if structure_ids is None:
            structure_ids = range(len(structures))
        structure_ids = list(structure_ids)
        if len(structure_ids) != len(structures):
            raise ValueError("structure_ids must have the same length as structures")

        config = self._descriptor_config
        species_set = set(config.species)
        for atoms in structures:
            present = set(atoms.get_chemical_symbols())
            missing = present - species_set
            if missing:
                raise ValueError(
                    f"SoapDescriptorConfig.species {sorted(species_set)} does not cover "
                    f"chemical symbols present in the input structures: {sorted(missing)}"
                )

        soap = SOAP(
            r_cut=config.r_cut,
            n_max=config.n_max,
            l_max=config.l_max,
            sigma=config.sigma,
            rbf=config.rbf,
            species=list(config.species),
            periodic=config.periodic,
            sparse=False,
        )

        match_species = self._distance_policy.central_species_matching
        match_periodicity = self._distance_policy.periodic_consistency_required

        all_vectors = []
        all_structure_id = []
        all_environment_index = []
        all_compatibility_key = []
        for sid, atoms in zip(structure_ids, structures):
            # A structure counts as "periodic" only if every cell direction is
            # periodic -- this is a scientific judgment call (mixed-pbc slabs
            # are treated as non-periodic), made fresh from the structure itself
            # every call, not cached across calls.
            is_periodic = bool(all(atoms.pbc)) if len(atoms.pbc) else False

            vectors = np.asarray(soap.create(atoms), dtype=np.float64)
            symbols = atoms.get_chemical_symbols()
            for env_index in range(len(atoms)):
                # Compatibility-key parts are tagged (not bare values) so a
                # species literally named "True"/"False" can never collide
                # with a periodicity flag -- both dimensions are opaque to
                # every module downstream of this one.
                key_parts = []
                if match_species:
                    key_parts.append(("species", symbols[env_index]))
                if match_periodicity:
                    key_parts.append(("periodic", is_periodic))

                all_vectors.append(vectors[env_index])
                all_structure_id.append(sid)
                all_environment_index.append(env_index)
                all_compatibility_key.append(tuple(key_parts))

        vectors_array = np.stack(all_vectors) if all_vectors else np.zeros((0, 0), dtype=np.float64)
        return RepresentationBatch(
            vectors=vectors_array,
            structure_id=tuple(all_structure_id),
            environment_index=tuple(all_environment_index),
            compatibility_key=tuple(all_compatibility_key),
        )

    def provenance(self) -> dict:
        return {
            "representation": "soap",
            "descriptor_config": _asdict_with_species(self._descriptor_config),
            "distance_policy": self._distance_policy.provenance(),
            "dscribe_version": dscribe_version(),
            "representation_hash": self.representation_hash(),
        }

    def representation_hash(self) -> str:
        import hashlib

        payload = "|".join([
            self._descriptor_config.content_hash(),
            self._distance_policy.content_hash(),
            dscribe_version(),
        ])
        return hashlib.sha256(payload.encode()).hexdigest()


def _asdict_with_species(config: SoapDescriptorConfig) -> dict:
    from dataclasses import asdict

    payload = asdict(config)
    payload["species"] = list(config.species)
    return payload
