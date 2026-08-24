"""Framework V2 -- SourceAndCapabilityInventory builder (generic).

The inventory answers a single question deterministically: *what source
material and what generation backends actually exist for this campaign, and
what can the frozen Teacher do?* It records what exists -- never what to use.
The strategy planner (a separate module) decides what to use from this
evidence.

The core owns only the assembly + fail-closed invariants. All material- and
environment-specific probing is delegated to caller-supplied plugins that
conform to the ``SourceProbe`` / ``BackendProbe`` / ``TeacherProbe`` protocols.
This keeps the inventory generic: a new backend or a new source universe is
introduced by writing a probe, never by editing this module.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from framework_v2.acquisition.contracts import (
    BackendCapabilityRecord,
    SourceAndCapabilityInventory,
    SourceCategoryRecord,
    TeacherCapabilityRecord,
)


@runtime_checkable
class SourceProbe(Protocol):
    """Reports the source categories that exist for a campaign."""
    def probe_sources(self) -> Sequence[SourceCategoryRecord]: ...


@runtime_checkable
class BackendProbe(Protocol):
    """Reports one generation backend's feasibility in this environment.

    A probe must return a record with ``feasible=False`` (and a reason)
    rather than raising when its library/dependency is missing -- absence is
    evidence the planner must see, not a crash."""
    def probe_backend(self) -> BackendCapabilityRecord: ...


@runtime_checkable
class TeacherProbe(Protocol):
    """Reports the frozen Teacher's capabilities."""
    def probe_teacher(self) -> TeacherCapabilityRecord: ...


def build_inventory(
    *,
    inventory_id: str,
    objective_sha256: str,
    source_probes: Sequence[SourceProbe],
    backend_probes: Sequence[BackendProbe],
    teacher_probe: TeacherProbe,
) -> SourceAndCapabilityInventory:
    """Assemble a content-addressable inventory from probes.

    Deterministic given deterministic probes. The inventory's own
    ``model_validator`` fails closed if the Teacher cannot label."""
    sources: list[SourceCategoryRecord] = []
    for sp in source_probes:
        sources.extend(sp.probe_sources())

    backends: list[BackendCapabilityRecord] = [bp.probe_backend() for bp in backend_probes]

    teacher = teacher_probe.probe_teacher()

    return SourceAndCapabilityInventory(
        inventory_id=inventory_id,
        objective_sha256=objective_sha256,
        sources=list(sources),
        backends=list(backends),
        teacher=teacher,
    )
