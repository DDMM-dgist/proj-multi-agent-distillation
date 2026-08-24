"""Framework V2 -- registered descriptor/domain plugins for AUTO-MATERIALIZED coverage evidence.

The framework-default acquisition planner must not require the user to hand-author a per-regime
coverage-evidence artifact, and must not embed material-specific descriptor extraction in the
planner/core. Instead the deterministic acquisition pipeline is fed by REGISTERED PLUGINS:

    raw source structures
      -> a registered ``StructuralDescriptorProvider`` (material-specific descriptor-space work,
         e.g. the SiO2 coordination/angle/SOAP descriptors)
      -> a ``DescriptorSpaceEvidence`` bundle (pure descriptor-space values + lazy representation
         builders)
      -> the framework's generic evidence materializer (inventory / target-regime / region /
         coverage / strategy) freezes it into a typed, content-addressed evidence artifact.

The framework core depends ONLY on this interface + the evidence contracts. Material feature
extraction stays entirely inside plugins. A future material either reuses a compatible generic
descriptor provider, registers its own, or -- if no admissible provider exists and coverage
evidence cannot be generated -- the framework FAILS CLOSED with a typed
``AcquisitionCapabilityGap`` (it never falls back to asking a human for n_parents/percentages/
sigma, and never fabricates descriptor-space values).

Both the DISCOVERED path (no trusted metadata -> discover regions from descriptors) and the
DECLARED path (trusted supplied metadata consumed after the mandatory lightweight consistency
audit, without forcing full descriptor discovery) are supported: a plugin advertises
``metadata_present`` and, when True, supplies the auditor + declared-representation builder that
``region_resolution.resolve_regions`` consumes.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from framework_v2.acquisition.coverage_gap import RegimeCoverageInput
from framework_v2.acquisition.contracts import SourceCategoryRecord
from framework_v2.acquisition.region_resolution import MetadataAuditor
from framework_v2.acquisition.strategy import StrategyEvidence
from framework_v2.contracts import DeploymentScopeContract, DomainRepresentation


@dataclasses.dataclass(frozen=True)
class DescriptorSpaceEvidence:
    """Everything a material-specific descriptor plugin contributes to the acquisition pipeline.

    This is the ONLY place material-specific descriptor-space work enters the framework. The
    framework composes the generic pipeline (inventory / target-regime / region / coverage /
    strategy) around these values and freezes the result; nothing here is a scientific *recipe*
    choice (that stays the dispatched Agent's job), only descriptor-space FACTS + admissible bounds.

    * ``descriptor`` -- human-readable descriptor string recorded on the TargetRegimeModel.
    * ``source_records`` -- the source-category inventory for this campaign's pool.
    * ``discovered_representation_builder`` -- lazily builds the DISCOVERED-path DomainRepresentation
      (typically wraps ``framework_v2.domain_discovery.discover_domain`` over the plugin's
      descriptor-space SourceItems + region_classifier). Called only when the DISCOVERED path runs.
    * ``regime_coverage_inputs`` -- per-target-regime coverage evidence (counts/saturation/novelty),
      keyed by the scope contract's region ids. Computed in descriptor space by the plugin.
    * ``strategy_evidence`` -- the deterministic descriptor-space signals the strategy planner reasons
      over (pool-covers-gaps / parents-reach-gaps / needs-new-configs / seeds-exist / mixed).
    * ``admissible_parent_ids`` -- the eligible parent-structure pool the Agent may select FROM
      (never which to use). Empty means no admissible parents -> the materializer fails closed.
    * ``required_param_keys`` / ``param_bounds`` -- the admissible generation-recipe decision space
      the Agent's proposal must satisfy (physics-bounded; supplied by the plugin, never invented by
      the core).
    * ``eligible_source_categories`` / ``selected_source_global_indices`` -- carried into the legacy
      14-field projection for the existing ACQUISITION executor.
    * ``metadata_present`` / ``metadata_auditor`` / ``declared_representation_builder`` -- DECLARED
      path support (see module docstring). When ``metadata_present`` is True BOTH the auditor and the
      declared-representation builder must be supplied (enforced by ``resolve_regions``).
    """
    descriptor: str
    source_records: tuple[SourceCategoryRecord, ...]
    discovered_representation_builder: Callable[[], DomainRepresentation]
    regime_coverage_inputs: tuple[RegimeCoverageInput, ...]
    strategy_evidence: StrategyEvidence
    admissible_parent_ids: tuple[str, ...]
    required_param_keys: tuple[str, ...]
    param_bounds: dict[str, tuple[float, float]]
    eligible_source_categories: tuple[str, ...]
    selected_source_global_indices: tuple[int, ...]
    duplicate_handling: str = "reject"
    saturation_threshold: float = 0.8
    available_source_coverage: Optional[dict[str, int]] = None
    metadata_present: bool = False
    metadata_auditor: Optional[MetadataAuditor] = None
    declared_representation_builder: Optional[Callable[[], DomainRepresentation]] = None


@runtime_checkable
class StructuralDescriptorProvider(Protocol):
    """A registered, material-specific descriptor plugin.

    ``applies`` gates admissibility from the run's own frozen inputs (objective + bound scope
    contract), deterministically and without side effects. ``build_descriptor_space_evidence``
    performs the material-specific descriptor-space computation and returns the bundle the framework
    materializer composes the generic pipeline around."""

    @property
    def material_id(self) -> str: ...

    def applies(self, *, controller, objective, scope_contract: DeploymentScopeContract) -> bool: ...

    def build_descriptor_space_evidence(
        self, *, controller, objective, scope_contract: DeploymentScopeContract,
    ) -> DescriptorSpaceEvidence: ...


class AcquisitionCapabilityGap(RuntimeError):
    """A typed, explainable fail-closed: the framework cannot autonomously materialize acquisition
    coverage evidence because no admissible descriptor/domain capability exists for this campaign.

    Raised (never swallowed as a silent no-op) when: no registered descriptor provider ``applies``;
    more than one applies ambiguously; a plugin yields an empty admissible parent pool; or the
    strategy planner finds no admissible backend. The framework surfaces this to a human as an
    irreducible capability gap -- it never falls back to prompting for low-level acquisition knobs
    and never fabricates descriptor-space values."""

    def __init__(self, message: str, *, gap_kind: str) -> None:
        super().__init__(message)
        self.gap_kind = gap_kind


_DESCRIPTOR_PROVIDERS: list[StructuralDescriptorProvider] = []
# The generic fallback lives in a SEPARATE slot from specialized plugins so it never competes
# for ambiguity with them: a specialized plugin that ``applies`` always wins, and the generic
# fallback is consulted ONLY when no specialized plugin applies (FE-027 priority resolution).
# This keeps a "works from raw structures by default" backend available without perturbing runs
# whose specialized plugin (or human-supplied plan) should take precedence.
_GENERIC_FALLBACK_PROVIDER: Optional[StructuralDescriptorProvider] = None


def register_descriptor_provider(provider: StructuralDescriptorProvider) -> None:
    """Register a SPECIALIZED descriptor plugin. Idempotent per ``material_id`` (re-registering
    the same material replaces the prior instance, so auto-registration at run-campaign startup is
    safe to call more than once)."""
    global _DESCRIPTOR_PROVIDERS
    _DESCRIPTOR_PROVIDERS = [
        p for p in _DESCRIPTOR_PROVIDERS if p.material_id != provider.material_id
    ]
    _DESCRIPTOR_PROVIDERS.append(provider)


def register_generic_descriptor_provider(provider: StructuralDescriptorProvider) -> None:
    """Register THE generic fallback provider (a single slot; re-registering replaces it).

    The generic fallback is the framework-provided, material-agnostic provider that builds a
    representation from raw species/positions/cell/pbc. It is deliberately NOT placed in the
    specialized-plugin list, so it can ``applies`` broadly without ever colliding ambiguously with
    a specialized plugin (specialized always wins; see ``resolve_descriptor_provider``)."""
    global _GENERIC_FALLBACK_PROVIDER
    _GENERIC_FALLBACK_PROVIDER = provider


def clear_descriptor_providers() -> None:
    """Remove all registered descriptor plugins AND the generic fallback (tests inject fakes;
    production auto-registers). Clearing both keeps a test's ``setUp`` a clean slate regardless of
    what a prior run/import registered."""
    global _GENERIC_FALLBACK_PROVIDER
    _DESCRIPTOR_PROVIDERS.clear()
    _GENERIC_FALLBACK_PROVIDER = None


def descriptor_providers() -> tuple[StructuralDescriptorProvider, ...]:
    return tuple(_DESCRIPTOR_PROVIDERS)


def generic_fallback_provider() -> Optional[StructuralDescriptorProvider]:
    return _GENERIC_FALLBACK_PROVIDER


def resolve_descriptor_provider(
    *, controller, objective, scope_contract: DeploymentScopeContract,
) -> StructuralDescriptorProvider:
    """Return the single admissible descriptor provider for this campaign under FE-027 priority.

    Priority: a SPECIALIZED plugin that ``applies`` always wins. Exactly one specialized applies
    -> return it; more than one -> fail closed AMBIGUOUS (the framework refuses to silently pick).
    ZERO specialized apply -> consult the generic fallback; if it applies, return it; otherwise
    fail closed with a typed ``AcquisitionCapabilityGap`` (never prompt for low-level knobs)."""
    applicable = [
        p for p in _DESCRIPTOR_PROVIDERS
        if p.applies(controller=controller, objective=objective, scope_contract=scope_contract)
    ]
    if len(applicable) > 1:
        raise AcquisitionCapabilityGap(
            "more than one registered StructuralDescriptorProvider applies "
            f"({sorted(p.material_id for p in applicable)}); refusing to silently disambiguate",
            gap_kind="AMBIGUOUS_DESCRIPTOR_PROVIDER",
        )
    if len(applicable) == 1:
        return applicable[0]

    fallback = _GENERIC_FALLBACK_PROVIDER
    if fallback is not None and fallback.applies(
            controller=controller, objective=objective, scope_contract=scope_contract):
        return fallback

    registered = sorted(p.material_id for p in _DESCRIPTOR_PROVIDERS)
    raise AcquisitionCapabilityGap(
        "no registered StructuralDescriptorProvider applies to this campaign "
        f"(specialized materials: {registered or 'none'}; generic fallback: "
        f"{'registered' if fallback is not None else 'none'}); the framework cannot autonomously "
        "materialize acquisition coverage evidence and will not prompt for low-level knobs",
        gap_kind="NO_ADMISSIBLE_DESCRIPTOR_PROVIDER",
    )
