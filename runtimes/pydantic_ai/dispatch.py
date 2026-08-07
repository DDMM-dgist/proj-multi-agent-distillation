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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .actions import APPROVAL_GATED_ACTIONS, ROLE_ALLOWED_ACTIONS, capability_status
from .tool_manifests import ROLE_TOOL_MANIFESTS, manifest_for

# Capability statuses that are terminal (never proposable/executable in the current scope).
_TERMINAL_CAPABILITY = {"NOT_AVAILABLE", "NOT_IMPLEMENTED", "OUT_OF_CURRENT_SCOPE"}


class ActionOutcome(BaseModel):
    model_config = {"extra": "forbid"}
    status: str  # EXECUTED | DRY_RUN | DENIED | BLOCKED_CAPABILITY | APPROVAL_REQUIRED | DUPLICATE | INVALID
    action_type: str
    role: str
    reason: str = ""
    idempotency_key: str = ""
    executor: str = ""
    artifact: Optional[dict] = None

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
    """Approval is per (run_id, boundary). The controller-backed store checks approval records."""
    def __init__(self):
        self._granted: set = set()

    def grant(self, run_id: str, boundary: str) -> None:
        self._granted.add((run_id, boundary))

    def has_approval(self, run_id: str, boundary: str, idempotency_key: str = "") -> bool:
        return (run_id, boundary) in self._granted


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
                          mode: str = "dry_run", manifest_lookup=manifest_for) -> ActionOutcome:
    """Run the full enforcement pipeline for one proposed action. ``mode`` in
    {"dry_run","validate_only","primary"}; only "primary" runs a real executor."""
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

    # (4) approval-boundary check
    if desc.approval_boundary and not approvals.has_approval(run_id, desc.approval_boundary, key):
        return out("APPROVAL_REQUIRED", f"action requires approval: {desc.approval_boundary}")

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
                   executor=(desc.executor.__name__ if desc.executor else "none"))
    try:
        artifact = desc.executor(proposal)
    except Exception as exc:  # noqa: BLE001 - executor/validator failure is fail-closed
        # A completion/preservation validator raises to signal failure; never emit a passing
        # artifact. INVALID for a validation failure, EXECUTOR_ERROR otherwise. Key not consumed.
        status = "INVALID" if type(exc).__name__ == "_ValidationFailure" else "EXECUTOR_ERROR"
        return out(status, f"{type(exc).__name__}: {exc}")
    outcome = out("EXECUTED", executor=desc.executor.__name__, artifact=artifact)
    if key:
        idempotency.record(key, outcome)  # only real executions consume the key
    return outcome
