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
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from .models import ProviderConfiguration

MODEL_ENV = "PYDANTIC_AI_MODEL"
# Local/OpenAI-compatible backend selection (Phase L1). PROVIDER_ENV names the backend kind
# explicitly; BASE_URL_ENV points at a locally served OpenAI-compatible endpoint (e.g. vLLM's
# ``http://127.0.0.1:8000/v1`` or Ollama's ``http://127.0.0.1:11434/v1``). A local backend
# requires NO Anthropic credential and NO real API key — pydantic_ai's OpenAI/Ollama providers
# inject a non-secret placeholder key for locally served models.
PROVIDER_ENV = "PYDANTIC_AI_PROVIDER"
BASE_URL_ENV = "PYDANTIC_AI_BASE_URL"

# Provider kinds the runtime distinguishes. "test" = TestModel/FunctionModel (network-free, used
# only by tests). "local-openai" = any OpenAI-compatible local server (vLLM first). "ollama" =
# a local Ollama server. "anthropic"/"openai" = optional hosted backends (kept, not required).
PROVIDER_KINDS = ("test", "local-openai", "ollama", "anthropic", "openai")
LOCAL_KINDS = ("local-openai", "ollama")
# Non-secret placeholder credential for local OpenAI-compatible servers. A local endpoint needs no
# real key; this is passed explicitly so the openai SDK never falls back to a real OPENAI_API_KEY.
LOCAL_PLACEHOLDER_API_KEY = "api-key-not-set"
# Hosted (billable, credential-gated) backends: routed through preflight_credentials + the
# generic pydantic_ai "provider:model" string, never a direct provider SDK call from our code.
HOSTED_KINDS = ("anthropic", "openai")

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


def preflight_credentials(env: Optional[dict] = None, *,
                          provider: Optional[str] = None) -> PreflightResult:
    """Inspect env only (no network). READY only when a model AND its credential are present.

    ``provider`` should be the already-resolved kind (e.g. from ``select_provider_kind()``,
    which honors the explicit ``PYDANTIC_AI_PROVIDER``) so a bare model id like ``gpt-4o-mini``
    (no ``provider:`` prefix) is never misattributed by guessing from the model string. When
    omitted, falls back to parsing the ``provider:model`` prefix out of ``PYDANTIC_AI_MODEL``
    (legacy back-compat, defaults to "anthropic" for an unprefixed name).
    """
    env = os.environ if env is None else env
    model_id = env.get(MODEL_ENV)
    if not model_id:
        return PreflightResult("NOT_CONFIGURED",
                               f"{MODEL_ENV} is not set; no provider/model chosen")
    resolved_provider = provider or _provider_of(model_id)
    entry = _PROVIDER_KEY_ENV.get(resolved_provider)
    if entry is None:
        return PreflightResult("BLOCKED",
                               f"unknown provider '{resolved_provider}'; no known credential env var",
                               provider=resolved_provider, model_id=model_id)
    # Qualify the model id with its provider prefix so build_provider_model() always hands
    # pydantic_ai an explicit "provider:model" string rather than relying on its own
    # name-prefix guessing (e.g. "gpt-..." -> openai) for models that don't match that heuristic.
    qualified_model_id = model_id if ":" in model_id else f"{resolved_provider}:{model_id}"
    key_env, sdk_name = entry
    key_present = bool(env.get(key_env))
    sdk_present = _sdk_available(sdk_name)
    if not key_present:
        return PreflightResult("SKIPPED",
                               f"{key_env} is not set; no provider will be called",
                               provider=resolved_provider, model_id=qualified_model_id,
                               key_present=False, sdk_present=sdk_present)
    if sdk_name and not sdk_present:
        return PreflightResult("BLOCKED",
                               f"credential present but the '{sdk_name}' SDK is not installed "
                               f"(pip install -e '.[pydantic-ai,{sdk_name}]')",
                               provider=resolved_provider, model_id=qualified_model_id,
                               key_present=True, sdk_present=False)
    return PreflightResult("READY",
                           "credential + SDK present; a live call is permitted AFTER approval",
                           provider=resolved_provider, model_id=qualified_model_id,
                           key_present=True, sdk_present=True)


def _sdk_available(sdk_name: Optional[str]) -> bool:
    if not sdk_name:
        return True
    import importlib.util
    return importlib.util.find_spec(sdk_name) is not None


def build_provider_model(model_id: str):
    """Return the model handle pydantic_ai's Agent accepts (the ``provider:model`` string).

    Lazy-imports pydantic_ai; does NOT contact the provider. Raises RuntimeError if the runtime
    deps are missing so callers fail loudly rather than silently degrade. This is the ANTHROPIC/
    hosted path (kept optional); the local path is ``build_local_model``.
    """
    try:
        import pydantic_ai  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "pydantic_ai is not installed; pip install -e '.[pydantic-ai]' (and the provider "
            "SDK extra, e.g. '.[anthropic]') to use a real provider") from exc
    return model_id


# --- Phase L1: provider-kind selection (backend-neutral) ------------------------

def select_provider_kind(env: Optional[dict] = None) -> str:
    """Return the selected provider kind (one of PROVIDER_KINDS) or "" if not configured.

    Explicit PYDANTIC_AI_PROVIDER wins. Otherwise a legacy ``anthropic:<model>`` in
    PYDANTIC_AI_MODEL infers "anthropic" (back-compat). Local kinds are NOT inferred from the
    model string (local ids like ``qwen2.5:7b`` collide with the ``provider:model`` form), so a
    local backend must be selected explicitly via PYDANTIC_AI_PROVIDER.
    """
    env = os.environ if env is None else env
    explicit = (env.get(PROVIDER_ENV) or "").strip().lower()
    if explicit:
        return explicit
    model_id = env.get(MODEL_ENV) or ""
    if model_id.startswith("anthropic:"):
        return "anthropic"
    return ""


# --- Phase L2: fail-closed LOCAL preflight (no provider call) --------------------

# Operational statuses (NOT scientific/runtime failures): a not-running server is an
# operational condition the caller reports and exits on, never a validation failure.
LOCAL_READY = "LOCAL_PROVIDER_READY"
LOCAL_NOT_SELECTED = "LOCAL_PROVIDER_NOT_SELECTED"
LOCAL_MODEL_NOT_CONFIGURED = "LOCAL_MODEL_NOT_CONFIGURED"
LOCAL_BASE_URL_NOT_CONFIGURED = "LOCAL_BASE_URL_NOT_CONFIGURED"
LOCAL_SDK_NOT_INSTALLED = "LOCAL_SDK_NOT_INSTALLED"
LOCAL_NOT_CONSTRUCTIBLE = "LOCAL_PROVIDER_NOT_CONSTRUCTIBLE"
LOCAL_NOT_RUNNING = "LOCAL_PROVIDER_NOT_RUNNING"


@dataclass
class LocalPreflightResult:
    status: str
    reason: str
    kind: str = ""
    model_id: str = ""
    base_url: str = ""
    sdk_present: bool = False
    constructible: bool = False
    server_probed: bool = False
    server_reachable: bool = False
    # Effective bounded-call policy (a cost/robustness guard even for a free local backend).
    timeout_s: float = 120.0
    provider_retries: int = 0
    structured_output_retries: int = 0
    max_total_calls: int = 1
    # A local backend must never depend on an Anthropic credential.
    anthropic_key_required: bool = False


def _host_port(base_url: str):
    parts = urlsplit(base_url)
    if not parts.hostname:
        return None, None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.hostname, port


def _server_reachable(base_url: str, timeout_s: float) -> bool:
    """Bounded TCP connect to the server host:port. Opens a socket only — sends no request,
    performs NO inference, and is never a paid call. Used solely to distinguish a not-running
    server (an operational state) from a runtime failure."""
    host, port = _host_port(base_url)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _openai_sdk_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("openai") is not None


def preflight_local(env: Optional[dict] = None, *, probe: bool = False,
                    connect_timeout_s: float = 0.75) -> LocalPreflightResult:
    """Inspect env + (optionally) probe the local server. NEVER calls the model / runs inference.

    READY requires: a local kind selected, a model id, a base URL, the ``openai`` SDK installed,
    and a constructible pydantic_ai provider/model object. When ``probe`` is set, the server must
    also be reachable (a plain TCP connect); otherwise the server check is deferred to run time.
    A local backend requires NO Anthropic credential and NO real API key.
    """
    env = os.environ if env is None else env
    kind = select_provider_kind(env)
    if kind not in LOCAL_KINDS:
        return LocalPreflightResult(LOCAL_NOT_SELECTED,
                                    f"{PROVIDER_ENV} is not a local kind (got {kind!r}); "
                                    f"set it to one of {LOCAL_KINDS}", kind=kind)
    model_id = (env.get(MODEL_ENV) or "").strip()
    base_url = (env.get(BASE_URL_ENV) or env.get("OLLAMA_BASE_URL") or "").strip()
    cfg = ProviderConfiguration(provider=kind, model_id=model_id or f"{kind}:<unset>")
    common = dict(kind=kind, model_id=model_id, base_url=base_url,
                  timeout_s=cfg.timeout_s, provider_retries=0, structured_output_retries=0,
                  max_total_calls=1, anthropic_key_required=False)
    if not model_id:
        return LocalPreflightResult(LOCAL_MODEL_NOT_CONFIGURED,
                                    f"{MODEL_ENV} is not set (local model id required)", **common)
    if not base_url:
        return LocalPreflightResult(LOCAL_BASE_URL_NOT_CONFIGURED,
                                    f"{BASE_URL_ENV} (or OLLAMA_BASE_URL) is not set", **common)
    sdk = _openai_sdk_available()
    common["sdk_present"] = sdk
    if not sdk:
        return LocalPreflightResult(LOCAL_SDK_NOT_INSTALLED,
                                    "the 'openai' SDK is not installed "
                                    "(pip install -e '.[pydantic-ai,local-openai]')", **common)
    # Construct the pydantic_ai provider/model OBJECT (no network: the AsyncOpenAI client is
    # lazy and connects only on a request).
    try:
        build_local_model(kind, model_id, base_url)
        common["constructible"] = True
    except Exception as exc:  # pragma: no cover - defensive
        return LocalPreflightResult(LOCAL_NOT_CONSTRUCTIBLE,
                                    f"provider/model object not constructible: {type(exc).__name__}",
                                    **common)
    if probe:
        reachable = _server_reachable(base_url, connect_timeout_s)
        common.update(server_probed=True, server_reachable=reachable)
        if not reachable:
            return LocalPreflightResult(
                LOCAL_NOT_RUNNING,
                f"no server accepting TCP connections at {base_url} "
                "(start the local inference server, e.g. vLLM, then retry)", **common)
    return LocalPreflightResult(LOCAL_READY,
                                "local provider configured + constructible"
                                + (" + server reachable" if probe else
                                   "; server reachability deferred to run time"), **common)


def build_local_model(kind: str, model_id: str, base_url: str):
    """Construct a pydantic_ai OpenAIChatModel bound to a LOCAL OpenAI-compatible server.

    No network call: constructing the provider/AsyncOpenAI client is lazy. No real API key is
    used — pydantic_ai injects a non-secret placeholder for locally served models. Raises
    RuntimeError if the ``openai`` extra is missing.
    """
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.providers.ollama import OllamaProvider
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the 'openai' SDK is not installed; pip install -e '.[pydantic-ai,local-openai]' "
            "to use a local OpenAI-compatible backend") from exc
    # A local OpenAI-compatible server needs no real credential. Pass an explicit non-secret
    # placeholder so the underlying openai SDK never falls back to reading OPENAI_API_KEY from the
    # environment -- a real hosted secret must never be sent in the Authorization header of a request
    # to a local (self-hosted, possibly untrusted) endpoint.
    if kind == "ollama":
        provider = OllamaProvider(base_url=base_url, api_key=LOCAL_PLACEHOLDER_API_KEY)
    else:  # local-openai (vLLM and any other OpenAI-compatible server)
        provider = OpenAIProvider(base_url=base_url, api_key=LOCAL_PLACEHOLDER_API_KEY)
    return OpenAIChatModel(model_id, provider=provider)
