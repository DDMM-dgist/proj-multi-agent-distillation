"""Trusted executor / controller bridge (Phase 5).

Wires the action-dispatch enforcement (dispatch.authorize_and_execute) to the durable controller:
approvals and idempotency are read/written through the controller's manifest, so a costly action
needs a recorded human approval and a duplicate idempotency key never re-executes across process
restarts. The controller stays the SOLE durable-state owner — this bridge only calls its public,
additive v7 methods; it never edits manifest.json directly and never changes gate/recovery
semantics.
"""
from __future__ import annotations

from .dispatch import authorize_and_execute, default_registry


class ControllerApprovalStore:
    """Approval check backed by the controller's durable action_approvals record."""
    def __init__(self, controller):
        self._c = controller

    def has_approval(self, run_id, boundary, idempotency_key: str = "",
                     plan_sha256: str | None = None) -> bool:
        return self._c.has_action_approval(boundary, plan_sha256=plan_sha256)


class ControllerIdempotencyStore:
    """Idempotency backed by the controller manifest (survives restarts)."""
    def __init__(self, controller):
        self._c = controller

    def seen(self, key: str) -> bool:
        return self._c.action_seen(key)

    def get(self, key: str):
        entry = self._c.state.get("idempotency", {}).get(key)
        if entry is None:
            return None
        # Return a lightweight object with an `.executor` attr for the DUPLICATE outcome.
        class _Prior:
            executor = entry.get("action_type", "")
        return _Prior()

    def record(self, key: str, outcome) -> None:
        artifact_ref = ""
        if outcome.artifact:
            artifact_ref = outcome.artifact.get("sha256", "") or outcome.artifact.get("path", "")
        self._c.record_action(key, action_type=outcome.action_type, status=outcome.status,
                              artifact_ref=artifact_ref)


class ControllerRecoveryAuthorizationStore:
    """RecoveryAuthorizationEnvelope check backed by the controller's own
    verify_recovery_authorization -- an ADDITIONAL, narrower pre-check dispatch.py consults only
    when a normal per-action approval is absent; it never replaces or weakens
    APPROVAL_GATED_ACTIONS (see workflow.controller.verify_recovery_authorization's docstring)."""
    def __init__(self, controller):
        self._c = controller

    def verify(self, *, action_type, capability=None, artifact_roles=None, resource_usage=None):
        return self._c.verify_recovery_authorization(
            action_type=action_type, capability=capability,
            artifact_roles=artifact_roles, resource_usage=resource_usage)


def dispatch_via_controller(proposal, *, controller, registry=None, mode="dry_run",
                            progress_cb=None):
    """Authorize + (optionally) execute a proposed action with controller-backed approval and
    idempotency. Returns the ActionOutcome. Heavy compute runs only inside a registered trusted
    executor, only in mode='primary', and only after every enforcement check passes.

    A costly child action lacking a normal per-action approval is given one additional, narrower
    chance to be authorized via the current activated recovery iteration's
    RecoveryAuthorizationEnvelope (if any) -- never a substitute for approval on a run with no
    active recovery, and never a way to widen what an envelope itself was scoped to permit.

    ``progress_cb``, if given, is passed straight through to ``authorize_and_execute`` -- an
    optional, additive long-running-executor progress hook; no existing caller or executor is
    affected when it is omitted.
    """
    registry = registry if registry is not None else default_registry()
    outcome = authorize_and_execute(
        proposal, registry=registry,
        approvals=ControllerApprovalStore(controller),
        idempotency=ControllerIdempotencyStore(controller),
        recovery_authorization=ControllerRecoveryAuthorizationStore(controller), mode=mode,
        progress_cb=progress_cb)
    return outcome
