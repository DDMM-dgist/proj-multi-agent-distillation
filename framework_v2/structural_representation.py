"""Generic structural representation adapters for V2.

The core contract is intentionally small: stable structure ordering, a numeric
matrix, and provenance/hash identity.  SOAP is exposed as a backend when DScribe
is installed; a deterministic built-in composition/count descriptor keeps unit
tests and dependency-light workflows executable without pretending to be SOAP.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Protocol

import numpy as np
from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase


class RepresentationBackend(str, Enum):
    COMPOSITION = "COMPOSITION"
    SOAP = "SOAP"
    PRETRAINED_MLIP = "PRETRAINED_MLIP"
    MACE_EMBEDDING = "MACE_EMBEDDING"


class StructureRecord(ContractBase):
    structure_id: str
    species_counts: dict[str, int]
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuralRepresentation(ContractBase):
    representation_id: str
    backend: RepresentationBackend
    parameters: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, str] = Field(default_factory=dict)
    species: list[str]
    structure_ids: list[str]
    matrix: list[list[float]]
    structure_ordering: str = "input_order"

    @model_validator(mode="after")
    def _shape(self):
        if len(set(self.structure_ids)) != len(self.structure_ids):
            raise ValueError("StructuralRepresentation structure_ids must be unique")
        if len(self.matrix) != len(self.structure_ids):
            raise ValueError("matrix row count must match structure_ids")
        for row in self.matrix:
            if len(row) != len(self.species):
                raise ValueError("matrix column count must match species for this backend")
        return self

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)


class RepresentationAdapter(Protocol):
    backend: RepresentationBackend

    def compute(
        self,
        structures: Iterable[StructureRecord],
        *,
        representation_id: str,
    ) -> StructuralRepresentation:
        ...


class CompositionRepresentationAdapter:
    """Dependency-light deterministic descriptor: elemental fractions."""

    backend = RepresentationBackend.COMPOSITION

    def __init__(self, *, species: list[str] | None = None) -> None:
        self.species = list(species or [])

    def compute(
        self,
        structures: Iterable[StructureRecord],
        *,
        representation_id: str,
    ) -> StructuralRepresentation:
        rows = list(structures)
        species = self.species or sorted({s for row in rows for s in row.species_counts})
        matrix: list[list[float]] = []
        for row in rows:
            total = sum(row.species_counts.values()) or 1
            matrix.append([row.species_counts.get(s, 0) / total for s in species])
        return StructuralRepresentation(
            representation_id=representation_id,
            backend=self.backend,
            parameters={"descriptor": "elemental_fraction"},
            software={"numpy": np.__version__},
            species=species,
            structure_ids=[row.structure_id for row in rows],
            matrix=matrix,
        )


class SoapRepresentationAdapter:
    """SOAP hook. Requires DScribe at runtime; no fallback is hidden."""

    backend = RepresentationBackend.SOAP

    def __init__(self, **parameters: Any) -> None:
        self.parameters = dict(parameters)

    def compute(
        self,
        structures: Iterable[StructureRecord],
        *,
        representation_id: str,
    ) -> StructuralRepresentation:
        try:
            import dscribe  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise RuntimeError("SOAP representation requires the dscribe package") from exc
        raise NotImplementedError(
            "SOAP backend hook is registered; ASE/DScribe structure plumbing must be supplied "
            "by the campaign adapter"
        )


__all__ = [
    "CompositionRepresentationAdapter",
    "RepresentationAdapter",
    "RepresentationBackend",
    "SoapRepresentationAdapter",
    "StructuralRepresentation",
    "StructureRecord",
]
