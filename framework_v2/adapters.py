"""Framework V2 — model adapters + the material-plugin interface (generality).

The cross-material generality addendum requires that the generic scientific core
never name a model family or a material's deterministic science. Two seams make
that possible:

  * :class:`ModelAdapterSpec` — a Teacher or Student is introduced to the core
    only through a typed adapter that advertises capability names (the same open
    namespace as :mod:`framework_v2.capability`). The core negotiates against
    those capabilities; it never imports or hard-codes Allegro / NequIP /
    SIMPLE-NN / any family.

  * A material-fact-producer registry — deterministic *material* science (e.g. a
    silica connectivity descriptor, a ternary phase membership rule) lives in a
    campaign plugin that registers a callable producing
    :class:`~framework_v2.facts.DeterministicFact` records under a namespaced
    key. The core invokes producers by key to obtain facts for a packet, so a
    new chemistry is added by registering a plugin, never by editing the core.

Layering: this module depends only on ``framework_v2`` foundations. Campaign
plugins import this module to register; the core imports it to look producers up.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Sequence

from pydantic import Field

from framework_v2.contracts import ContractBase
from framework_v2.facts import DeterministicFact


class ModelRole(str, Enum):
    TEACHER = "teacher"
    STUDENT = "student"


class ModelAdapterSpec(ContractBase):
    """How a Teacher or Student model is introduced to the generic core.

    ``model_family`` is a free campaign-chosen label (audit only). ``capabilities``
    are stable capability names the core negotiates against — the core never
    branches on ``model_family``.
    """
    adapter_id: str
    role: ModelRole
    model_family: str
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


# =====================================================================
# MATERIAL FACT-PRODUCER PLUGIN REGISTRY
# =====================================================================
# A producer takes an arbitrary campaign context mapping and returns a list of
# DeterministicFact. The core treats it as an opaque, deterministic source of
# material facts; all chemistry lives inside the plugin.
MaterialFactProducer = Callable[..., Sequence[DeterministicFact]]

_FACT_PRODUCERS: dict[str, MaterialFactProducer] = {}


def register_fact_producer(
    name: str, producer: MaterialFactProducer, *, replace: bool = False
) -> None:
    """Register a material deterministic-fact producer under a namespaced ``name``
    (convention ``<material>.<producer>``, e.g. ``sio2.structural_descriptors``).

    Fails closed on a duplicate name unless ``replace`` is set — a campaign
    cannot silently shadow another campaign's producer.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("fact producer name must be a non-empty string")
    if not callable(producer):
        raise ValueError("fact producer must be callable")
    if name in _FACT_PRODUCERS and not replace:
        raise ValueError(
            f"material fact producer {name!r} is already registered; pass replace=True "
            f"to override deliberately"
        )
    _FACT_PRODUCERS[name] = producer


def get_fact_producer(name: str) -> MaterialFactProducer:
    """Resolve a registered producer, failing closed on an unregistered name."""
    try:
        return _FACT_PRODUCERS[name]
    except KeyError as exc:
        raise KeyError(f"unregistered material fact producer: {name!r}") from exc


def registered_fact_producers() -> tuple[str, ...]:
    return tuple(sorted(_FACT_PRODUCERS))


def produce_material_facts(name: str, **context: Any) -> list[DeterministicFact]:
    """Invoke a registered producer and validate that it returned
    :class:`DeterministicFact` records (the core will not accept untyped
    material output masquerading as authoritative facts)."""
    producer = get_fact_producer(name)
    facts = list(producer(**context))
    for f in facts:
        if not isinstance(f, DeterministicFact):
            raise TypeError(
                f"material fact producer {name!r} returned a non-DeterministicFact "
                f"object: {type(f).__name__}"
            )
    return facts


__all__ = [
    "ModelRole",
    "ModelAdapterSpec",
    "MaterialFactProducer",
    "register_fact_producer",
    "get_fact_producer",
    "registered_fact_producers",
    "produce_material_facts",
]
