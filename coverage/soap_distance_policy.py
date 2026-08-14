"""Explicit, no-default campaign policy for what "SOAP distance" means.

Raw dscribe SOAP vectors are not, by themselves, a distance. Whether two SOAP
vectors are compared via unnormalized Euclidean distance or via a normalized
(cosine/dot-product) kernel is a scientific choice with real consequences for
coverage evidence -- the common "SOAP kernel" in the literature is a normalized
dot-product kernel, not raw Euclidean distance on dscribe's default output.
Likewise, whether nearest-neighbor matching is restricted to environments with
the same central atomic species, and whether periodic and non-periodic
structures may be compared at all, change the resulting evidence. None of these
may be picked implicitly by this infrastructure -- this is deliberate, matching
coverage.descriptor_config's fail-closed convention for descriptor parameters.
Production configuration must fail closed if any of these are absent; only test
fixtures may supply explicit values chosen for fast, exact unit tests.

This is ONE implementation of the generic `coverage.distance_policy.DistancePolicy`
protocol -- `normalize()` and `provenance()` below are what let
`coverage.reference_pool`/`coverage.nn_distance` apply this policy's
normalization without any of those generic modules knowing this is SOAP or
importing this module directly. `central_species_matching` and
`periodic_consistency_required` are declarative compatibility-rule flags only:
the actual per-environment compatibility data (species, periodicity) is
computed by `coverage.soap_representation.SoapCoverageRepresentation`, which
consults this policy's flags to decide what to fold into each environment's
opaque `compatibility_key` (see coverage.representation.RepresentationBatch).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


_REQUIRED_FIELDS = ("normalization", "metric", "central_species_matching",
                    "periodic_consistency_required")
_NORMALIZATIONS = ("none", "l2")
_METRICS = ("euclidean", "cosine")


@dataclass(frozen=True)
class SoapDistancePolicy:
    normalization: str = None
    metric: str = None
    central_species_matching: bool = None
    periodic_consistency_required: bool = None

    def __post_init__(self):
        missing = [name for name in _REQUIRED_FIELDS if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "SoapDistancePolicy is missing required scientific policy fields "
                f"(no defaults are assumed for any of them): {missing}"
            )
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError(f"SoapDistancePolicy.normalization must be one of {_NORMALIZATIONS}")
        if self.metric not in _METRICS:
            raise ValueError(f"SoapDistancePolicy.metric must be one of {_METRICS}")
        if not isinstance(self.central_species_matching, bool):
            raise ValueError("SoapDistancePolicy.central_species_matching must be an explicit bool")
        if not isinstance(self.periodic_consistency_required, bool):
            raise ValueError(
                "SoapDistancePolicy.periodic_consistency_required must be an explicit bool"
            )
        if self.metric == "cosine" and self.normalization != "l2":
            raise ValueError(
                "SoapDistancePolicy.metric='cosine' requires normalization='l2' -- cosine "
                "distance is implemented as Euclidean distance between L2-normalized vectors "
                "(mathematically equivalent for a metric-tree backend), so an un-normalized "
                "vector paired with a cosine metric is an inconsistent policy, not a valid choice"
            )

    def normalize(self, vectors):
        """Return `vectors` transformed for comparison under this policy.

        `normalization="l2"` L2-normalizes each row (used to realize cosine
        distance as Euclidean distance between normalized vectors, since an
        exact-search backend such as scipy's cKDTree has no native cosine
        metric); `normalization="none"` returns `vectors` unchanged.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if self.normalization != "l2" or vectors.shape[0] == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError(
                "SoapDistancePolicy.normalization='l2' cannot normalize a zero-norm "
                "descriptor vector -- this indicates a degenerate environment, not a "
                "value this policy may silently pass through"
            )
        return vectors / norms[:, None]

    def provenance(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict) -> "SoapDistancePolicy":
        if not isinstance(payload, dict):
            raise ValueError("SoapDistancePolicy config payload must be an object")
        unknown = set(payload) - set(_REQUIRED_FIELDS)
        if unknown:
            raise ValueError(f"SoapDistancePolicy config has unknown fields: {sorted(unknown)}")
        try:
            return cls(**payload)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"SoapDistancePolicy config is malformed: {exc}") from exc
