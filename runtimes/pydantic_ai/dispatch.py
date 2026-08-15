"""Action authorization + dispatch enforcement (Phase 4).

Every producer/analyst action a PydanticAI agent proposes passes through ``authorize_and_execute``
in this EXACT order (default-deny at each step):

    role -> role-manifest lookup -> tool/action allow check -> capability-status check
    -> approval-boundary check -> typed-parameter + artifact/hash check -> idempotency check
    -> trusted executor

The manifests (tool_manifests.py) and capability registry (actions.py) are ENFORCED here, not
merely documented: an action absent from the role's manifest is denied; a NOT_AVAILABLE /
NOT_IMPLEMENTED / OUT_OF_CURRENT_SCOPE capability is fail-closed; an approval-gated action without
an approval record cannot execute; a duplicate idempotency key does not re-execute. Heavy compute
is never run by the agent — only a trusted executor callable registered here runs, and only after
all checks pass.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .actions import APPROVAL_GATED_ACTIONS, ROLE_ALLOWED_ACTIONS, capability_status
from .tool_manifests import ROLE_TOOL_MANIFESTS, manifest_for

# Capability statuses that are terminal (never proposable/executable in the current scope).
_TERMINAL_CAPABILITY = {"NOT_AVAILABLE", "NOT_IMPLEMENTED", "OUT_OF_CURRENT_SCOPE"}


class ExternalActionPending(Exception):
    """A trusted executor raises this to report that the action was legitimately submitted but
    has not completed yet (e.g. queued external/async work) -- distinct from an ordinary failure
    (EXECUTOR_ERROR) and from an already-completed replay (DUPLICATE). Never consumes the
    idempotency key: a later dispatch with the same key re-checks/re-attempts instead of replaying
    a stale outcome, so polling for completion never looks like a duplicate-execution bug."""


class ActionOutcome(BaseModel):
    model_config = {"extra": "forbid"}
    status: str  # EXECUTED | DRY_RUN | DENIED | BLOCKED_CAPABILITY | APPROVAL_REQUIRED | DUPLICATE | INVALID | PENDING
    action_type: str
    role: str
    reason: str = ""
    idempotency_key: str = ""
    executor: str = ""
    artifact: Optional[dict] = None
    # Set only when an approval-gated action was authorized via a RecoveryAuthorizationEnvelope
    # (workflow.controller.verify_recovery_authorization) instead of a normal per-action approval
    # record -- present for audit/traceability, never itself a source of authorization.
    recovery_authorization_envelope_sha256: Optional[str] = None

    @property
    def executed(self) -> bool:
        return self.status == "EXECUTED"


@dataclass
class ActionDescriptor:
    """Binds an action_type to a trusted executor (or None = dry-run only) + policy."""
    action_type: str
    role: str
    cost_class: str = "light"
    approval_boundary: Optional[str] = None
    executor: Optional[Callable[[Any], dict]] = None      # trusted; runs only after all checks
    param_validator: Optional[Callable[[Any], tuple]] = None  # (ok: bool, message: str)


# --- Approval + idempotency stores (in-memory here; controller-backed in the bridge) ---------

class InMemoryApprovalStore:
    """Approval is per (run_id, boundary), optionally bound to an exact plan hash."""
    def __init__(self):
        self._granted: set = set()

    def grant(self, run_id: str, boundary: str, plan_sha256: str | None = None) -> None:
        self._granted.add((run_id, boundary, plan_sha256))

    def has_approval(self, run_id: str, boundary: str, idempotency_key: str = "",
                     plan_sha256: str | None = None) -> bool:
        return (run_id, boundary, plan_sha256) in self._granted


class InMemoryIdempotencyStore:
    def __init__(self):
        self._seen: dict = {}

    def seen(self, key: str) -> bool:
        return key in self._seen

    def get(self, key: str):
        return self._seen.get(key)

    def record(self, key: str, outcome: ActionOutcome) -> None:
        self._seen[key] = outcome


def _get(proposal, name: str, default=None):
    """Read a field from a proposal that may be a pydantic model or a plain dict."""
    if isinstance(proposal, BaseModel):
        return getattr(proposal, name, default)
    if isinstance(proposal, dict):
        return proposal.get(name, default)
    return getattr(proposal, name, default)


def _artifact_roles(proposal) -> list:
    """Roles of the proposal's declared input artifacts (for RecoveryAuthorizationEnvelope
    artifact-role matching). Never invents a role for an artifact that doesn't declare one."""
    roles = []
    for item in (_get(proposal, "input_artifacts", []) or []):
        role = item.role if isinstance(item, BaseModel) else (
            item.get("role") if isinstance(item, dict) else None)
        if role:
            roles.append(role)
    return roles


def _resource_usage(proposal) -> Optional[dict]:
    """An agent-declared ``parameters.resource_usage`` dict, or None. Never estimated on the
    agent's behalf -- a proposal that omits it simply cannot be checked against resource limits
    (see workflow.controller.verify_recovery_authorization, which then trivially passes that
    check rather than fail closed on an absence this module cannot honestly fill in)."""
    params = _get(proposal, "parameters", {}) or {}
    usage = params.get("resource_usage") if isinstance(params, dict) else None
    return usage if isinstance(usage, dict) else None


def default_registry() -> dict:
    """A registry entry for every in-scope role action, with approval boundaries wired but NO
    inline executor (dry-run by default). The trusted-executor wiring to controller/adapters is
    supplied by the controller bridge; tests inject executors to exercise the EXECUTED path."""
    reg: dict = {}
    for role, actions in ROLE_ALLOWED_ACTIONS.items():
        cost = ROLE_TOOL_MANIFESTS[role].cost_class
        for action in actions:
            reg[action] = ActionDescriptor(
                action_type=action, role=role, cost_class=cost,
                approval_boundary=APPROVAL_GATED_ACTIONS.get(action))
    return reg


def authorize_and_execute(proposal, *, registry: dict, approvals, idempotency,
                          mode: str = "dry_run", manifest_lookup=manifest_for,
                          recovery_authorization=None,
                          progress_cb: Optional[Callable[[dict], None]] = None) -> ActionOutcome:
    """Run the full enforcement pipeline for one proposed action. ``mode`` in
    {"dry_run","validate_only","primary"}; only "primary" runs a real executor.

    ``recovery_authorization``, if given, is a duck-typed object with a
    ``verify(*, action_type, capability=None, artifact_roles=None, resource_usage=None)`` method
    (see ``controller_bridge.ControllerRecoveryAuthorizationStore``, which delegates to
    ``workflow.controller.RunController.verify_recovery_authorization``). It is consulted ONLY
    when an approval-gated action lacks a normal per-action approval record, as an ADDITIONAL,
    narrower alternative path -- never a replacement for or weakening of
    ``APPROVAL_GATED_ACTIONS``: an action with no approval boundary is unaffected, and one with a
    boundary still requires either a normal approval or a matching envelope, exactly as before if
    ``recovery_authorization`` is omitted or returns no match.
    """
    role = _get(proposal, "requested_by_role", "")
    action = _get(proposal, "action_type", "")
    key = _get(proposal, "idempotency_key", "") or ""
    run_id = _get(proposal, "run_id", "") or ""

    def out(status: str, reason: str = "", **kw) -> ActionOutcome:
        return ActionOutcome(status=status, action_type=action, role=role, reason=reason,
                             idempotency_key=key, **kw)

    # (1) role-manifest lookup
    try:
        manifest = manifest_lookup(role)
    except KeyError:
        return out("DENIED", f"unknown role '{role}'")

    # (2)+(3) capability status (fail-closed) + action-allow check (default deny)
    cap = capability_status(action)
    if cap and cap.status in _TERMINAL_CAPABILITY:
        return out("BLOCKED_CAPABILITY", f"{action} is {cap.status}: {cap.reason}")
    if action not in manifest.proposable_actions:
        if cap and cap.status == "APPROVAL_REQUIRED":
            return out("APPROVAL_REQUIRED",
                       f"{action} requires approval and is not enabled in the current scope")
        return out("DENIED", f"'{action}' is not in the {role} manifest (default deny)")

    desc = registry.get(action)
    if desc is None:
        return out("DENIED", f"no executor registered for '{action}'")
    if desc.role != role:
        return out("DENIED", f"registry role mismatch for '{action}'")

    # (4) approval-boundary check. Acquisition execution is bound to the exact
    # validated plan hash, so a generic boundary approval cannot authorize a
    # different parent selection or augmentation parameter set.
    plan_sha256 = None
    if action == "acquire_structures":
        try:
            from .executors import acquisition_plan_sha256_from_proposal
            plan_sha256 = acquisition_plan_sha256_from_proposal(proposal)
        except Exception as exc:  # noqa: BLE001 - fail closed before approval/idempotency/executor
            return out("INVALID", f"PLAN_INPUT_REQUIRED: {exc}")
    envelope_sha256 = None
    if desc.approval_boundary and not approvals.has_approval(
            run_id, desc.approval_boundary, key, plan_sha256=plan_sha256):
        if recovery_authorization is not None:
            envelope_sha256 = recovery_authorization.verify(
                action_type=action, artifact_roles=_artifact_roles(proposal),
                resource_usage=_resource_usage(proposal))
        if envelope_sha256 is None:
            suffix = f" plan_sha256={plan_sha256}" if plan_sha256 else ""
            return out("APPROVAL_REQUIRED", f"action requires approval: {desc.approval_boundary}{suffix}")

    # (5) typed parameter + artifact/hash check
    if desc.param_validator is not None:
        ok, message = desc.param_validator(proposal)
        if not ok:
            return out("INVALID", message or "parameter/artifact validation failed")

    # (6) idempotency check
    if key and idempotency.seen(key):
        prior = idempotency.get(key)
        return out("DUPLICATE", "idempotency_key already processed",
                   executor=(prior.executor if prior else ""))

    # (7) trusted executor (heavy compute lives here, never in the agent)
    if mode != "primary" or desc.executor is None:
        # A dry-run/validate-only never consumes the idempotency key (no real side effect).
        return out("DRY_RUN", "dry-run: validated, no side effects",
                   executor=(desc.executor.__name__ if desc.executor else "none"),
                   recovery_authorization_envelope_sha256=envelope_sha256)
    try:
        if progress_cb is not None and "progress_cb" in inspect.signature(desc.executor).parameters:
            artifact = desc.executor(proposal, progress_cb=progress_cb)
        else:
            artifact = desc.executor(proposal)
    except ExternalActionPending as exc:
        # Legitimately still in flight -- not a failure, and not consumed as a completed
        # execution, so the SAME idempotency key can be dispatched again later to re-check.
        return out("PENDING", str(exc))
    except Exception as exc:  # noqa: BLE001 - executor/validator failure is fail-closed
        # A completion/preservation validator raises to signal failure; never emit a passing
        # artifact. INVALID for a validation failure, EXECUTOR_ERROR otherwise. Key not consumed.
        status = "INVALID" if type(exc).__name__ == "_ValidationFailure" else "EXECUTOR_ERROR"
        return out(status, f"{type(exc).__name__}: {exc}")
    outcome = out("EXECUTED", executor=desc.executor.__name__, artifact=artifact,
                 recovery_authorization_envelope_sha256=envelope_sha256)
    if key:
        idempotency.record(key, outcome)  # only real executions consume the key
    return outcome
