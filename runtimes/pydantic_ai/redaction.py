"""Secret redaction for provenance, logs, and error text (Phase 2/D4).

Any string that may reach a provenance record, a log line, or the model context is scrubbed
of credential-like content BEFORE it is stored. Redaction is deterministic and conservative:
it prefers to over-mask (e.g. a long opaque token) rather than leak a key.
"""
from __future__ import annotations

import re
from typing import Iterable

_PLACEHOLDER = "[REDACTED]"

# Provider/key shapes and generic key=value credential assignments.
_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),                       # OpenAI/Anthropic-style keys
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),                # Slack tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),                   # GitHub tokens
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),            # Bearer <token>
    re.compile(r"(?i)\b(api[_-]?key|apikey|token|secret|password|passwd|authorization|"
               r"access[_-]?key|client[_-]?secret)\b\s*[:=]\s*[\"']?[^\s\"']{6,}[\"']?"),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),                     # long base64-ish blobs
]


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with credential-like substrings replaced by ``[REDACTED]``.

    ``extra_secrets`` are exact values (e.g. the current ``ANTHROPIC_API_KEY`` read from the
    environment) that must be masked wherever they appear, even if they don't match a pattern.
    """
    if text is None:
        return text
    out = str(text)
    for secret in extra_secrets:
        if secret and len(secret) >= 4:
            out = out.replace(secret, _PLACEHOLDER)
    for pattern in _PATTERNS:
        out = pattern.sub(_PLACEHOLDER, out)
    return out


def secrets_from_env(env: dict) -> list[str]:
    """Collect credential-like values from an environment mapping so they can be masked."""
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
            "MISTRAL_API_KEY", "CO_API_KEY", "AWS_SECRET_ACCESS_KEY", "AZURE_API_KEY")
    values = []
    for k in keys:
        v = env.get(k)
        if v:
            values.append(v)
    # Also mask anything whose name looks secret-like.
    for name, value in env.items():
        if value and re.search(r"(?i)(key|token|secret|password)", name):
            values.append(value)
    return values
