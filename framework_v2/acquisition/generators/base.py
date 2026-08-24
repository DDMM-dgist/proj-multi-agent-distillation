"""Framework V2 -- CandidateGenerator interface + registry (generic).

Every candidate-generation backend (local perturbation, Teacher-driven MD,
de-novo structure generation, ...) implements the same generic interface so the
planner can treat them uniformly and pick one from evidence. A backend must:

  * advertise feasibility via a deterministic capability probe (``probe``) --
    return ``feasible=False`` with a reason rather than raising when a
    dependency is missing (absence is evidence the planner must see);
  * accept a typed, content-addressable ``GenerationProtocol`` (the params the
    planner/agent proposed) and validate it deterministically
    (``validate_protocol``) -- returning a list of issues, empty iff valid;
  * execute the protocol (``generate``) into a workdir, returning a
    provenance-separated ``CandidateGenerationResult`` (generation provenance is
    NOT a training label).

The registry is a plain name->backend map so new backends are added without
editing the core.
"""
from __future__ import annotations

import abc
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import Field

from framework_v2.acquisition.contracts import (
    AcquisitionStrategyKind,
    BackendCapabilityRecord,
    CandidateGenerationResult,
)
from framework_v2.contracts import ContractBase


class GenerationProtocol(ContractBase):
    """The concrete, content-addressable generation protocol a backend
    executes. ``params`` is backend-specific but always fully specified (the
    planner leaves nothing implicit); its content-SHA is recorded as each
    candidate's ``generation_params_sha256`` for provenance."""
    protocol_id: str
    backend_id: str
    strategy_kind: AcquisitionStrategyKind
    strategy_sha256: str
    n_requested: int
    target_regime_ids: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class TeacherCalculatorProvider(Protocol):
    """Minimal frozen-Teacher interface a backend needs.

    ``identity_sha256`` pins the frozen Teacher's identity (recorded in
    provenance). ``make_ase_calculator`` returns an ASE-compatible Calculator
    driving the Teacher PES -- used both to relax perturbed structures and to
    drive dynamics. Kept minimal so a fake Teacher can satisfy it in tests."""
    @property
    def identity_sha256(self) -> str: ...
    def make_ase_calculator(self) -> Any: ...


class CandidateGenerator(abc.ABC):
    """Generic base every backend subclasses."""

    @property
    @abc.abstractmethod
    def backend_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def strategy_kind(self) -> AcquisitionStrategyKind: ...

    @abc.abstractmethod
    def probe(self) -> BackendCapabilityRecord:
        """Deterministic feasibility probe. Never raises for a missing
        dependency -- records ``feasible=False`` with a reason instead."""

    @abc.abstractmethod
    def validate_protocol(self, protocol: GenerationProtocol) -> list[str]:
        """Return a list of deterministic issues; empty iff the protocol is
        executable by this backend. Must not mutate the protocol."""

    @abc.abstractmethod
    def generate(
        self,
        protocol: GenerationProtocol,
        *,
        workdir: str,
        teacher: Optional[TeacherCalculatorProvider] = None,
    ) -> CandidateGenerationResult:
        """Execute the protocol; write structures under ``workdir``; return a
        provenance-separated result. Generation provenance (incl. any
        exploration PES) is NOT a training label."""


class CandidateGeneratorRegistry:
    """A simple name->backend registry."""

    def __init__(self) -> None:
        self._backends: dict[str, CandidateGenerator] = {}

    def register(self, backend: CandidateGenerator) -> None:
        if backend.backend_id in self._backends:
            raise ValueError(f"backend already registered: {backend.backend_id}")
        self._backends[backend.backend_id] = backend

    def get(self, backend_id: str) -> CandidateGenerator:
        if backend_id not in self._backends:
            raise KeyError(f"no such backend: {backend_id}")
        return self._backends[backend_id]

    def all(self) -> list[CandidateGenerator]:
        return list(self._backends.values())

    def probe_all(self) -> list[BackendCapabilityRecord]:
        return [b.probe() for b in self._backends.values()]
