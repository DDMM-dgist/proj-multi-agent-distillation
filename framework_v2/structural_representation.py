"""Generic structural representation adapters for V2.

The core contract is intentionally small: stable structure ordering, a numeric
matrix, and provenance/hash identity.  SOAP is exposed as a real backend when
DScribe + ASE are installed; a deterministic built-in composition/count
descriptor keeps unit tests and dependency-light workflows executable without
pretending to be SOAP.  Descriptor width is explicit (``descriptor_dimension``)
and never assumed equal to the species count, so a real per-species SOAP power
spectrum is representable.
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


class PoolingMode(str, Enum):
    NONE = "NONE"
    MEAN = "MEAN"
    SUM = "SUM"


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
    descriptor_dimension: int | None = None
    pooling: PoolingMode = PoolingMode.NONE
    structure_ordering: str = "input_order"

    @model_validator(mode="before")
    @classmethod
    def _infer_descriptor_dimension(cls, data):
        # Backward compatibility: older callers/serialized records omit the
        # explicit descriptor width.  Infer it from the matrix so the width
        # invariant below still runs, without assuming len(species).
        if isinstance(data, dict) and data.get("descriptor_dimension") is None:
            matrix = data.get("matrix")
            if matrix:
                data = {**data, "descriptor_dimension": len(matrix[0])}
        return data

    @model_validator(mode="after")
    def _shape(self):
        if len(set(self.structure_ids)) != len(self.structure_ids):
            raise ValueError("StructuralRepresentation structure_ids must be unique")
        if len(self.matrix) != len(self.structure_ids):
            raise ValueError("matrix row count must match structure_ids")
        if self.descriptor_dimension is None or self.descriptor_dimension <= 0:
            raise ValueError("descriptor_dimension must be a positive integer")
        for row in self.matrix:
            if len(row) != self.descriptor_dimension:
                raise ValueError("matrix row width must match descriptor_dimension")
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
            descriptor_dimension=len(species),
            pooling=PoolingMode.NONE,
        )


class SoapDependencyError(RuntimeError):
    """Raised when the SOAP backend is requested but DScribe/ASE are absent."""


class SoapRepresentationAdapter:
    """Real SOAP descriptor via DScribe. No hidden composition fallback.

    Consumes ASE ``Atoms`` at runtime through :meth:`compute_ase` (geometry is
    required for SOAP, so the composition-count ``compute`` entrypoint is not
    applicable).  A local per-atom SOAP power spectrum is pooled to a fixed-size
    per-structure vector via ``MEAN``/``SUM``.  ASE ``Atoms`` are never stored
    inside the persisted contract.
    """

    backend = RepresentationBackend.SOAP

    def __init__(
        self,
        *,
        species: list[str],
        r_cut: float,
        n_max: int,
        l_max: int,
        sigma: float = 1.0,
        periodic: bool = False,
        pooling: PoolingMode = PoolingMode.MEAN,
        sparse: bool = False,
        **parameters: Any,
    ) -> None:
        self.species = list(species)
        self.parameters = {
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "sigma": sigma,
            "periodic": periodic,
            "sparse": sparse,
            **parameters,
        }
        self.pooling = pooling

    def compute_ase(
        self,
        structures: Iterable[tuple[str, Any]],
        *,
        representation_id: str,
    ) -> StructuralRepresentation:
        try:
            import dscribe  # type: ignore
            from dscribe.descriptors import SOAP  # type: ignore
            import ase  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise SoapDependencyError(
                "SOAP representation requires the dscribe and ase packages; install/"
                "provide them or select a different representation backend"
            ) from exc

        rows = list(structures)
        if not rows:
            raise ValueError("SOAP representation requires at least one structure")
        ids = [sid for sid, _atoms in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("SOAP structure IDs must be unique")
        if self.pooling == PoolingMode.NONE:
            raise ValueError(
                "SOAP local output requires MEAN or SUM pooling for a fixed-size "
                "per-structure representation"
            )

        soap = SOAP(
            species=self.species,
            r_cut=float(self.parameters["r_cut"]),
            n_max=int(self.parameters["n_max"]),
            l_max=int(self.parameters["l_max"]),
            sigma=float(self.parameters["sigma"]),
            periodic=bool(self.parameters["periodic"]),
            sparse=bool(self.parameters.get("sparse", False)),
        )

        matrix: list[list[float]] = []
        for _sid, atoms in rows:
            values = soap.create(atoms)
            arr = np.asarray(
                values.todense() if hasattr(values, "todense") else values, dtype=float
            )
            if arr.ndim == 2:
                if self.pooling == PoolingMode.MEAN:
                    arr = arr.mean(axis=0)
                elif self.pooling == PoolingMode.SUM:
                    arr = arr.sum(axis=0)
            matrix.append(arr.reshape(-1).astype(float).tolist())

        descriptor_dimension = len(matrix[0])
        for row in matrix:
            if len(row) != descriptor_dimension:
                raise ValueError("SOAP produced inconsistent descriptor widths across structures")

        import ase as _ase  # local, for version stamp only
        return StructuralRepresentation(
            representation_id=representation_id,
            backend=self.backend,
            parameters=self.parameters,
            software={
                "dscribe": getattr(dscribe, "__version__", "unknown"),
                "ase": getattr(_ase, "__version__", "unknown"),
                "numpy": np.__version__,
            },
            species=self.species,
            structure_ids=ids,
            matrix=matrix,
            descriptor_dimension=descriptor_dimension,
            pooling=self.pooling,
        )


__all__ = [
    "CompositionRepresentationAdapter",
    "PoolingMode",
    "RepresentationAdapter",
    "RepresentationBackend",
    "SoapDependencyError",
    "SoapRepresentationAdapter",
    "StructuralRepresentation",
    "StructureRecord",
]
