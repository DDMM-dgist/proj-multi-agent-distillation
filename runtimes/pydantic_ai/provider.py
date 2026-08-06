"""Provider adapter + credential preflight (Phase 2/D3).

NOTHING here calls a provider at import or preflight time. ``preflight_credentials`` inspects
the environment ONLY and returns a typed status, so a caller can SKIP/BLOCK cleanly when no
credential is present — a missing key is never reported as a successful provider run.

The real provider model is constructed lazily via ``build_provider_model`` (which imports
``pydantic_ai`` and, for Anthropic, checks the ``anthropic`` SDK is installed). A live call only
happens later, behind explicit human approval.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .models import ProviderConfiguration

MODEL_ENV = "PYDANTIC_AI_MODEL"

# provider name -> (credential env var, optional SDK import name for the [anthropic]-style extra)
_PROVIDER_KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
    "openai": ("OPENAI_API_KEY", "openai"),
    "google-gla": ("GEMINI_API_KEY", None),
    "groq": ("GROQ_API_KEY", None),
    "mistral": ("MISTRAL_API_KEY", None),
}

PreflightStatus = str  # one of: READY | SKIPPED | BLOCKED | NOT_CONFIGURED


@dataclass
class PreflightResult:
    status: PreflightStatus
    reason: str
    provider: str = ""
    model_id: str = ""
    key_present: bool = False
    sdk_present: bool = False


def _provider_of(model_id: str) -> str:
    return model_id.split(":", 1)[0] if ":" in model_id else "anthropic"


def provider_config_from_env(env: Optional[dict] = None) -> Optional[ProviderConfiguration]:
    """Build a ProviderConfiguration from the environment, or None if no model is configured."""
    env = os.environ if env is None else env
    model_id = env.get(MODEL_ENV)
    if not model_id:
        return None
    return ProviderConfiguration(provider=_provider_of(model_id), model_id=model_id)


def preflight_credentials(env: Optional[dict] = None) -> PreflightResult:
    """Inspect env only (no network). READY only when a model AND its credential are present."""
    env = os.environ if env is None else env
    model_id = env.get(MODEL_ENV)
    if not model_id:
        return PreflightResult("NOT_CONFIGURED",
                               f"{MODEL_ENV} is not set; no provider/model chosen")
    provider = _provider_of(model_id)
    entry = _PROVIDER_KEY_ENV.get(provider)
    if entry is None:
        return PreflightResult("BLOCKED",
                               f"unknown provider '{provider}'; no known credential env var",
                               provider=provider, model_id=model_id)
    key_env, sdk_name = entry
    key_present = bool(env.get(key_env))
    sdk_present = _sdk_available(sdk_name)
    if not key_present:
        return PreflightResult("SKIPPED",
                               f"{key_env} is not set; no provider will be called",
                               provider=provider, model_id=model_id, key_present=False,
                               sdk_present=sdk_present)
    if sdk_name and not sdk_present:
        return PreflightResult("BLOCKED",
                               f"credential present but the '{sdk_name}' SDK is not installed "
                               f"(pip install -e '.[pydantic-ai,{sdk_name}]')",
                               provider=provider, model_id=model_id, key_present=True,
                               sdk_present=False)
    return PreflightResult("READY",
                           "credential + SDK present; a live call is permitted AFTER approval",
                           provider=provider, model_id=model_id, key_present=True,
                           sdk_present=True)


def _sdk_available(sdk_name: Optional[str]) -> bool:
    if not sdk_name:
        return True
    import importlib.util
    return importlib.util.find_spec(sdk_name) is not None


def build_provider_model(model_id: str):
    """Return the model handle pydantic_ai's Agent accepts (the ``provider:model`` string).

    Lazy-imports pydantic_ai; does NOT contact the provider. Raises RuntimeError if the runtime
    deps are missing so callers fail loudly rather than silently degrade.
    """
    try:
        import pydantic_ai  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "pydantic_ai is not installed; pip install -e '.[pydantic-ai]' (and the provider "
            "SDK extra, e.g. '.[anthropic]') to use a real provider") from exc
    return model_id
