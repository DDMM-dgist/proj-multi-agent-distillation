"""Framework V2 -- blind-test enforcement (Section 4).

R31's 1142-frame heldout has been inspected repeatedly and is
permanently ``HISTORICAL_BENCHMARK``, ``NOT_BLIND_FOR_R32``. Framework
V2 requires a genuinely fresh blind-test boundary for every new
campaign, and architecturally forbids the following stages from
touching a blind-test artifact:

  * acquisition
  * data_coverage
  * teacher_labeling (when producing new labels for training)
  * dataset_split (when *selecting* what goes into train/val -- the
    partitioner may READ blind identity to protect it, but must not use
    blind CONTENT to steer the split)
  * training
  * evaluation UNLESS explicitly authorised as the final-eval pass
  * uncertainty
  * deployment_md
  * recovery decisions

The enforcement primitive is deliberately simple: a blind-test
"boundary" is the set of artifact SHA-256 identities that are on the
blind side. Any read attempt is a call to ``guard_blind_access`` which
returns ``ALLOW`` or ``DENY`` (raising ``BlindTestAccessViolation``)
against a small allowlist keyed by ``(stage, purpose)``. Every attempt,
whether allowed or denied, is written to an append-only
``BlindTestAccessLog`` so the ex-post audit ("was the blind test
touched before final evaluation?") is deterministic.

Prompts alone are not enough. This module is what makes prompt-level
promises enforceable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import Field

from framework_v2.contracts import ContractBase, utc_now_iso


class BlindTestAccessViolation(RuntimeError):
    """Raised when a disallowed stage/purpose attempts to access a
    blind-test artifact. Fail-closed."""


ALLOW = "ALLOW"
DENY = "DENY"


class BlindTestBoundary(ContractBase):
    """The set of artifact identities that make up the blind test.

    ``allowlist`` is a mapping ``stage`` -> set of ``purpose`` strings.
    The default allowlist grants no stage any access; callers must
    explicitly grant final-evaluation and physical-validation access
    when they establish the boundary."""
    boundary_id: str
    blind_artifact_sha256s: list[str]
    allowlist: dict[str, list[str]] = Field(default_factory=dict)
    established_at: str = Field(default_factory=utc_now_iso)
    rationale: str = ""

    def is_blind(self, artifact_sha256: str) -> bool:
        return artifact_sha256 in set(self.blind_artifact_sha256s)

    def is_allowed(self, stage: str, purpose: str) -> bool:
        return purpose in set(self.allowlist.get(stage, []))


class BlindTestAccessAttempt(ContractBase):
    """One recorded access attempt (allowed or denied)."""
    attempt_id: str
    stage: str
    purpose: str
    artifact_sha256: str
    outcome: str  # "ALLOW" or "DENY"
    at: str = Field(default_factory=utc_now_iso)
    caller: str = ""
    rationale: str = ""


class BlindTestAccessLog:
    """Append-only JSONL log of every ``guard_blind_access`` call.

    Storage: ``<run_dir>/framework_v2/blind_test_access.jsonl``. Never
    rewritten. The final-report auditor reads this file to confirm that
    no blind artifact was touched by a disallowed stage before final
    evaluation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, attempt: BlindTestAccessAttempt) -> None:
        payload = json.dumps(attempt.model_dump(mode="json"),
                             sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")

    def iter_attempts(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield BlindTestAccessAttempt.model_validate(json.loads(line))

    def violations(self) -> list[BlindTestAccessAttempt]:
        return [a for a in self.iter_attempts() if a.outcome == DENY]


def guard_blind_access(
    *,
    boundary: BlindTestBoundary,
    stage: str,
    purpose: str,
    artifact_sha256: str,
    log: Optional[BlindTestAccessLog] = None,
    caller: str = "",
) -> str:
    """Fail-closed guard around every read of a possibly-blind artifact.

    * If the artifact is not on the blind boundary, returns ``ALLOW``
      and does not log (the artifact is public to the run).
    * If the artifact IS on the blind boundary and ``(stage, purpose)``
      is not in ``boundary.allowlist``, logs a DENY attempt and raises
      ``BlindTestAccessViolation``.
    * If both conditions permit access, logs ALLOW and returns ``ALLOW``.

    Returning ``ALLOW`` does not itself perform the read -- the caller
    is expected to use the return value as a precondition. Raising is
    the fail-closed path.
    """
    if not boundary.is_blind(artifact_sha256):
        return ALLOW

    outcome = ALLOW if boundary.is_allowed(stage, purpose) else DENY
    if log is not None:
        log.append(BlindTestAccessAttempt(
            attempt_id=_attempt_id(stage, purpose, artifact_sha256),
            stage=stage,
            purpose=purpose,
            artifact_sha256=artifact_sha256,
            outcome=outcome,
            caller=caller,
            rationale=(f"stage={stage!r} purpose={purpose!r} "
                       f"allowed={boundary.is_allowed(stage, purpose)}"),
        ))
    if outcome == DENY:
        raise BlindTestAccessViolation(
            f"blind-test access denied: stage={stage!r} purpose={purpose!r} "
            f"artifact_sha256={artifact_sha256[:12]}...; "
            f"boundary allowlist has no such entry"
        )
    return ALLOW


def _attempt_id(stage: str, purpose: str, artifact_sha256: str) -> str:
    import hashlib
    now = datetime.now(timezone.utc).isoformat()
    h = hashlib.sha256(
        f"{stage}|{purpose}|{artifact_sha256}|{now}".encode("utf-8")
    ).hexdigest()
    return "bta-" + h[:16]


__all__ = [
    "ALLOW", "DENY",
    "BlindTestBoundary",
    "BlindTestAccessAttempt",
    "BlindTestAccessLog",
    "BlindTestAccessViolation",
    "guard_blind_access",
]
