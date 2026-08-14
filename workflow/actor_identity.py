"""Canonical actor-identity model separating recovery proposal authority from approval authority.

This is the shared vocabulary between ``workflow.controller.RunController.propose_recovery``
(records who proposed a RecoveryPlan), ``approve_recovery`` (requires a human approval actor and
rejects self-approval), and ``authorize_recovery_capabilities`` (requires a human actor distinct
from the proposer before a RecoveryAuthorizationEnvelope may exist). Nothing here encodes any one
campaign's roles, chemistry, or model family -- ``actor_kind`` is a fixed, framework-level
three-way split (human / agent / system), and ``canonical_id`` is whatever free-form identifier
the caller supplies, normalized so equality is not a fragile raw-string comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ACTOR_KINDS = ("human", "agent", "system")


@dataclass(frozen=True)
class ActorIdentity:
    actor_kind: str
    canonical_id: str
    display_name: Optional[str] = None

    def as_dict(self) -> dict:
        return {"actor_kind": self.actor_kind, "canonical_id": self.canonical_id,
                "display_name": self.display_name}


def normalize_actor_identity(raw, *, field_name: str) -> ActorIdentity:
    """Normalize a bare string or a structured ``{actor_kind, canonical_id}`` mapping into an
    ``ActorIdentity`` with a whitespace/case-insensitive ``canonical_id``.

    A bare non-empty string is treated as a human display name -- this is the ONLY implicit
    default, so a human manually drafting a RecoveryPlan or approving one never needs to
    construct a structured identity. Every other actor_kind (``agent``, ``system``) must be
    declared explicitly by whatever produced the identity (e.g. an agent-facing typed bridge);
    it is never inferred from string shape or naming convention.
    """
    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError(f"{field_name} requires a non-empty actor identity")
        stripped = raw.strip()
        return ActorIdentity(actor_kind="human", canonical_id=stripped.casefold(),
                             display_name=stripped)
    if isinstance(raw, dict):
        kind = raw.get("actor_kind")
        if kind not in ACTOR_KINDS:
            raise ValueError(f"{field_name}.actor_kind must be one of {ACTOR_KINDS}, got {kind!r}")
        canonical_id = raw.get("canonical_id")
        if not isinstance(canonical_id, str) or not canonical_id.strip():
            raise ValueError(f"{field_name}.canonical_id must be a non-empty string")
        display_name = raw.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError(f"{field_name}.display_name must be a string if present")
        return ActorIdentity(actor_kind=kind, canonical_id=canonical_id.strip().casefold(),
                             display_name=display_name)
    raise ValueError(f"{field_name} requires a non-empty string or an actor-identity object")


def same_actor(a: ActorIdentity, b: ActorIdentity) -> bool:
    """Canonical-id equality used for the no-self-approval invariant.

    Deliberately ignores actor_kind and display_name: a proposer recorded under a given
    canonical_id (whatever kind it carried at proposal time) must still be recognized as the
    same actor if a later approval/authorization call is submitted under the identical
    canonical_id, even if that later call's own actor_kind label were to differ.
    """
    return a.canonical_id == b.canonical_id
