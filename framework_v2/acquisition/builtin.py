"""Framework V2 -- shipped (built-in) descriptor/domain plugin registration.

The framework-default acquisition provider is material-AGNOSTIC: it composes the generic pipeline
around whatever ``StructuralDescriptorProvider`` plugin ``applies`` to a campaign. Those plugins are
the ONLY place material-specific descriptor-space work lives. A new user must not have to write
run-specific Python to get autonomous acquisition planning, so the framework auto-registers every
plugin it *ships* at run-campaign startup through this single entry point.

Registration is idempotent per ``material_id`` (see
``descriptor_plugins.register_descriptor_provider``), so calling this more than once -- e.g. once per
run-campaign invocation -- is safe. When the framework ships no descriptor plugin (or none is added
yet), this registers nothing: the default provider then simply never ``applies`` to any campaign, so
runs behave exactly as before this evolution (no perturbation to the broad regression suite).

To ship a new material's plugin, import its ``StructuralDescriptorProvider`` here and append it to
``_BUILTIN_PROVIDERS``. The heavy, material-specific descriptor code must stay behind a lazy import
inside the plugin so importing this module never drags a material library into every run.
"""
from __future__ import annotations

from framework_v2.acquisition.descriptor_plugins import (
    StructuralDescriptorProvider,
    register_descriptor_provider,
    register_generic_descriptor_provider,
)


def _builtin_providers() -> tuple[StructuralDescriptorProvider, ...]:
    """The descriptor plugins the framework ships. Built lazily so a material plugin's heavy
    dependencies are only imported when the framework actually has one to register."""
    providers: list[StructuralDescriptorProvider] = []
    # Shipped material plugins are appended here as they are added, e.g.:
    #     from framework_v2.acquisition.plugins.sio2 import SiO2DescriptorProvider
    #     providers.append(SiO2DescriptorProvider())
    return tuple(providers)


def register_builtin_descriptor_providers() -> tuple[str, ...]:
    """Register every shipped descriptor plugin; return the registered ``material_id``s.

    Idempotent (registration replaces per ``material_id``). An empty return means the framework
    ships no SPECIALIZED descriptor plugin -- but see ``register_builtin_generic_fallback``: the
    framework still ships a material-agnostic generic fallback, so autonomous acquisition planning
    remains available even with no specialized plugin."""
    registered: list[str] = []
    for provider in _builtin_providers():
        register_descriptor_provider(provider)
        registered.append(provider.material_id)
    return tuple(registered)


def register_builtin_generic_fallback() -> str:
    """Register the framework-shipped material-AGNOSTIC generic fallback provider (FE-027).

    This is the descriptor provider that makes autonomous acquisition planning work for a brand-new
    material with NO per-material Python: it builds its representation from raw structural facts and
    lives in the SEPARATE generic-fallback slot, so a specialized plugin always wins when one
    applies (see ``resolve_descriptor_provider``). Idempotent (the single fallback slot is
    replaced). The heavy generic descriptor code stays behind this lazy import so importing this
    module never drags the acquisition representation machinery into every run.

    Returns the fallback provider's ``material_id`` (a fixed generic identifier naming no material)."""
    from framework_v2.acquisition.generic_provider import GenericStructuralDescriptorProvider

    provider = GenericStructuralDescriptorProvider()
    register_generic_descriptor_provider(provider)
    return provider.material_id
