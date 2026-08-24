"""Orchestrator bridge: typed OrchestratorActionProposal -> controller call.

The Orchestrator's bridge surface (``tool_manifests.ORCHESTRATOR_BRIDGE_ACTIONS``) is
architecturally distinct from a producer/analyst's typed ActionProposal surface
(``runtimes.pydantic_ai.actions.ROLE_ALLOWED_ACTIONS`` + ``dispatch.authorize_and_execute``): a
producer proposes a scientific action that a trusted executor runs against files; the Orchestrator
instead calls the Controller's OWN already-authoritative methods to drive the existing gate/
recovery state machine. ``ROLE_ALLOWED_ACTIONS``/``CAPABILITY_REGISTRY``/
``executors.build_executor_registry()`` are untouched by this module and stay scoped to exactly
the 4 producer/analyst roles -- see
``tests/test_architecture_freeze.py::test_production_wiring_keeps_frozen_architecture_dimensions``,
which pins that boundary.

This module wires three bridge actions -- ``propose_recovery``, ``commit_teacher_validation_plan``
and ``propose_acquisition_plan`` -- to real controller calls. Every other name in
``ORCHESTRATOR_BRIDGE_ACTIONS`` is a real,
declared bridge action (present in the orchestrator's manifest, cross-checked by tests) but
intentionally NOT_IMPLEMENTED here: it fails closed with ``BLOCKED_CAPABILITY`` rather than
silently no-oping as if it succeeded. Wiring one of them up means adding it to
``_BRIDGE_EXECUTORS`` alongside its own executor -- never widening what an existing wired action
itself is allowed to do.

Calling ``propose_recovery`` through this bridge is still just a PROPOSAL:
``RunController.propose_recovery`` remains the sole authoritative validator and fails closed
exactly as it does when called directly (bad plan shape, stale diagnosis hash, unregistered
failure_category, protected-reference violation, loop-safety policy, ...). Nothing here approves
or executes a recovery -- ``approve_recovery``/``start_iteration`` remain distinct, separately
human/execution-gated controller calls this module never touches.
"""
from __future__ import annotations

from typing import Literal

from .actions import ActionProposalBase
from .dispatch import ActionOutcome
from .tool_manifests import ORCHESTRATOR_BRIDGE_ACTIONS, manifest_for


class OrchestratorActionProposal(ActionProposalBase):
    """Typed proposal for one Orchestrator bridge call. Distinct from a producer/analyst
    ActionProposal: this is intentionally NOT added to
    ``runtimes.pydantic_ai.actions.ROLE_ALLOWED_ACTIONS``/``ROLE_ACTION_MODELS``, which stay
    scoped to exactly the 4 producer/analyst roles (a pinned architecture invariant). For
    ``action_type="propose_recovery"``, ``parameters`` must carry ``run_dir`` (the run directory
    a live ``RunController`` binds to) and ``plan_path`` (an on-disk RecoveryPlan JSON file,
    typically produced by ``runtimes.pydantic_ai.recovery_bridge.build_recovery_plan_draft``).
    """
    model_config = {"extra": "forbid"}
    requested_by_role: Literal["orchestrator"] = "orchestrator"
    action_type: Literal[ORCHESTRATOR_BRIDGE_ACTIONS]  # type: ignore[valid-type]


def _exec_propose_recovery(proposal: "OrchestratorActionProposal", *, controller) -> dict:
    plan_path = proposal.parameters.get("plan_path")
    if not plan_path:
        raise ValueError("propose_recovery requires parameters.plan_path")
    # Trust boundary: the recorded proposer identity comes from this proposal's own
    # Pydantic Literal-typed `requested_by_role` (an LLM authoring the plan_path JSON cannot
    # forge this field's value), never from whatever `proposed_by` the plan payload itself may
    # contain. See RunController.propose_recovery's docstring for the fail-closed conflict
    # policy this enforces.
    trusted_proposer = {"actor_kind": "system", "canonical_id": proposal.requested_by_role}
    recovery = controller.propose_recovery(plan_path, proposer=trusted_proposer)
    return {"recovery_id": recovery["id"], "status": recovery["status"],
            "path": recovery["path"], "integrity": recovery["integrity"]}


def _exec_commit_teacher_validation_plan(proposal: "OrchestratorActionProposal", *, controller) -> dict:
    plan_path = proposal.parameters.get("plan_path")
    if not plan_path:
        raise ValueError("commit_teacher_validation_plan requires parameters.plan_path")
    # Same trust boundary as _exec_propose_recovery: the recorded proposer identity comes from
    # this proposal's own Pydantic Literal-typed `requested_by_role`, never from whatever
    # `proposed_by` the plan payload itself may contain.
    trusted_proposer = {"actor_kind": "system", "canonical_id": proposal.requested_by_role}
    record = controller.commit_teacher_validation_plan(plan_path, proposer=trusted_proposer)
    return {"status": record["status"], "path": record.get("path"),
            "evidence_profile_sha256": record.get("evidence_profile_sha256"),
            "selected_components": record.get("selected_components")}


def _exec_propose_acquisition_plan(proposal: "OrchestratorActionProposal", *, controller) -> dict:
    """Bind an autonomously-designed acquisition plan into the run as a new run-bound input.

    ``parameters.plan_path`` is the on-disk legacy 14-field ``AcquisitionPlan`` JSON that
    ``framework_v2.acquisition.plan_assembly.build_legacy_projection`` produced (already
    deterministically validated by ``validate_acquisition_plan_v2`` before we get here). Binding it
    through the frozen ``RunController.bind_new_input`` makes it a first-class run input with a
    content-addressed snapshot -- exactly the same mechanism ``initialize()`` uses for a
    human-supplied plan -- so the ACQUISITION stage's existing fail-closed
    ``_bind_acquisition_plan_for_stage`` / ``executors._validate_acquisition_plan`` gate accepts it
    unchanged. This bridge never generates candidates or runs the Teacher; it only binds the
    already-built plan artifact."""
    plan_path = proposal.parameters.get("plan_path")
    if not plan_path:
        raise ValueError("propose_acquisition_plan requires parameters.plan_path")
    record = controller.bind_new_input(plan_path, copy=True)
    return {"source": record["source"], "snapshot": record.get("snapshot"),
            "sha256": record["sha256"]}


# Bridge actions with a real, wired controller call. Every other declared bridge action fails
# closed as BLOCKED_CAPABILITY below -- adding support means adding an entry here, not weakening
# the allowlist check.
_BRIDGE_EXECUTORS = {"propose_recovery": _exec_propose_recovery,
                     "commit_teacher_validation_plan": _exec_commit_teacher_validation_plan,
                     "propose_acquisition_plan": _exec_propose_acquisition_plan}


def dispatch_orchestrator_action(proposal: OrchestratorActionProposal, *, controller,
                                 mode: str = "dry_run") -> ActionOutcome:
    """Enforce the bridge-action allowlist, then (``mode="primary"`` only) call the one wired
    controller method. Default-deny at every step, mirroring ``dispatch.authorize_and_execute``'s
    fail-closed spirit but scoped to the orchestrator's controller-calling bridge surface rather
    than the producer/analyst executor surface.

    ``mode`` in {"dry_run", "validate_only", "primary"}; only "primary" ever calls the controller.
    """
    role = proposal.requested_by_role
    action = proposal.action_type
    key = proposal.idempotency_key

    def out(status: str, reason: str = "", **kw) -> ActionOutcome:
        return ActionOutcome(status=status, action_type=action, role=role, reason=reason,
                             idempotency_key=key, **kw)

    try:
        manifest = manifest_for(role)
    except KeyError:
        return out("DENIED", f"unknown role '{role}'")
    if action not in manifest.bridge_actions:
        return out("DENIED", f"'{action}' is not a declared bridge action for {role}")

    executor = _BRIDGE_EXECUTORS.get(action)
    if executor is None:
        return out("BLOCKED_CAPABILITY",
                   f"'{action}' has no wired bridge executor yet (NOT_IMPLEMENTED)")

    if mode != "primary":
        return out("DRY_RUN", "dry-run: validated, no controller call made",
                   executor=executor.__name__)
    try:
        artifact = executor(proposal, controller=controller)
    except Exception as exc:  # noqa: BLE001 - controller call failure is fail-closed
        return out("EXECUTOR_ERROR", f"{type(exc).__name__}: {exc}")
    return out("EXECUTED", executor=executor.__name__, artifact=artifact)
