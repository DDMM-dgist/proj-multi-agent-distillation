"""Explicit, no-default configuration for local-environment (SOAP) descriptors.

Every scientific parameter is required. A missing or empty field is a configuration
error raised at construction time, never a silently-assumed default -- this is
deliberate: cutoff radius, radial/angular basis truncation, and smoothing width all
change the resulting coverage evidence, and none of them may be picked implicitly by
this infrastructure. Production configuration must fail closed if any of these are
absent; only test fixtures may supply explicit small values chosen for fast, exact
unit tests.

This module intentionally does NOT include an aggregation-policy or weighting-policy
field. Frame-level summaries are always computed as a fixed, comprehensive set of
descriptive statistics (see coverage.aggregate.SUMMARY_STATS) rather than a single
policy-selected number, and reference weighting is always preserved as a full
per-reference-category distance matrix (see coverage.nn_distance) rather than
collapsed into one weighting choice up front -- both atom-pooled and
category-balanced views are always reconstructable from the same raw evidence
without recomputing anything. Deciding which view or statistic to act on is a later,
human-approved policy step, not a descriptor-computation concern.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


_REQUIRED_FIELDS = ("r_cut", "n_max", "l_max", "sigma", "species", "rbf", "periodic")


@dataclass(frozen=True)
class SoapDescriptorConfig:
    r_cut: float = None
    n_max: int = None
    l_max: int = None
    sigma: float = None
    species: tuple = None
    rbf: str = None
    periodic: bool = None

    def __post_init__(self):
        missing = [name for name in _REQUIRED_FIELDS if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "SoapDescriptorConfig is missing required scientific parameters "
                f"(no defaults are assumed for any of them): {missing}"
            )
        if not isinstance(self.species, tuple) or len(self.species) == 0:
            raise ValueError("SoapDescriptorConfig.species must be a non-empty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.species):
            raise ValueError("SoapDescriptorConfig.species entries must be non-empty strings")
        if not isinstance(self.r_cut, (int, float)) or isinstance(self.r_cut, bool) or self.r_cut <= 0:
            raise ValueError("SoapDescriptorConfig.r_cut must be a positive number")
        if not isinstance(self.n_max, int) or isinstance(self.n_max, bool) or self.n_max < 1:
            raise ValueError("SoapDescriptorConfig.n_max must be a positive integer")
        if not isinstance(self.l_max, int) or isinstance(self.l_max, bool) or self.l_max < 0:
            raise ValueError("SoapDescriptorConfig.l_max must be a non-negative integer")
        if not isinstance(self.sigma, (int, float)) or isinstance(self.sigma, bool) or self.sigma <= 0:
            raise ValueError("SoapDescriptorConfig.sigma must be a positive number")
        if not isinstance(self.rbf, str) or self.rbf not in ("gto", "polynomial"):
            raise ValueError("SoapDescriptorConfig.rbf must be one of 'gto', 'polynomial'")
        if not isinstance(self.periodic, bool):
            raise ValueError("SoapDescriptorConfig.periodic must be an explicit bool")

    def content_hash(self) -> str:
        payload = asdict(self)
        payload["species"] = list(self.species)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict) -> "SoapDescriptorConfig":
        if not isinstance(payload, dict):
            raise ValueError("SoapDescriptorConfig config payload must be an object")
        payload = dict(payload)
        if "species" in payload and payload["species"] is not None:
            payload["species"] = tuple(payload["species"])
        unknown = set(payload) - set(_REQUIRED_FIELDS)
        if unknown:
            raise ValueError(f"SoapDescriptorConfig config has unknown fields: {sorted(unknown)}")
        try:
            return cls(**payload)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"SoapDescriptorConfig config is malformed: {exc}") from exc
