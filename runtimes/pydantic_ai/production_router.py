"""Single production entry point that routes an agent invocation to the correct acceptance
strategy by ROLE / typed-output — so a caller never hand-picks an internal function (Phase 6/0).

Strategies (selected from the role's typed output model, not by the caller):
- judge_gate      : JudgeVote -> canonical validation -> FileExchange accept (gate aggregation).
- producer_dispatch: role-scoped ActionProposal -> role/capability/approval/idempotency
                     -> trusted executor -> controller record (dispatch_via_controller).
- typed_result    : OrchestratorPlan / LiteratureEvidence -> Pydantic-validated typed output,
                     recorded as provenance; NO controller mutation.
- agent_result    : generic AgentResult (fallback / unknown role).

Mode semantics are uniform: only ``primary`` may mutate exchange/controller state; ``shadow`` and
``dry_run`` never do. Wrong role/output combinations are fail-closed. Provenance is written on
every path. This router changes NO scientific (gate/recovery) semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .driver import _write_provenance
from .role_outputs import select_output_model


@dataclass
class RouteResult:
    strategy: str
    accepted: bool
    controller_mutated: bool
    error: Optional[str]
    provenance_path: Path
    detail: object = None  # validated payload (judge/typed) or ActionOutcome (producer)


def acceptance_strategy(spec) -> str:
    model = select_output_model(spec).__name__
    if model == "JudgeVoteModel":
        return "judge_gate"
    if model.endswith("ActionProposal"):
        return "producer_dispatch"
    if model in ("OrchestratorPlan", "LiteratureEvidence"):
        return "typed_result"
    return "agent_result"


def _get(candidate, key, default=None):
    return candidate.get(key, default) if isinstance(candidate, dict) else default


def run_role(runtime, task, spec, context, *, controller=None, registry=None, mode="shadow") -> RouteResult:
    """Invoke one role and accept its result via the role-appropriate strategy. ``mode`` in
    {"primary","shadow","dry_run","validate_only"} — only primary mutates state."""
    invocation = runtime.run(task, spec, context)
    rec = invocation.provenance
    rec.mode = mode
    strategy = acceptance_strategy(spec)

    # A provider attempt that raised before output is a preserved failure; no acceptance.
    if getattr(rec, "failure_category", ""):
        path = _write_provenance(context, invocation)
        return RouteResult(strategy, False, False, rec.exception_message or rec.failure_category, path)

    if strategy in ("judge_gate", "agent_result"):
        return _accept_via_exchange(invocation, spec, task, context, mode, strategy)

    # producer_dispatch / typed_result: fail-closed Pydantic validation of the typed output
    # against the role's model (so a malformed/wrong-shape output is rejected regardless of which
    # runtime produced it — the real runtime already enforces output_type; this guards test/other
    # runtimes too).
    ok, err = _validate_typed(invocation.candidate, spec)
    if not ok:
        path = _write_provenance(context, invocation)
        return RouteResult(strategy, False, False, err, path)

    if strategy == "producer_dispatch":
        return _accept_via_dispatch(invocation, spec, context, controller, registry, mode)
    return _accept_typed_result(invocation, spec, context, mode)


def _validate_typed(candidate, spec):
    model = select_output_model(spec)
    try:
        model(**(candidate or {}))
        return True, None
    except Exception as exc:  # pydantic.ValidationError etc. -> fail-closed
        return False, f"typed-output validation failed: {type(exc).__name__}"


def _accept_via_exchange(invocation, spec, task, context, mode, strategy) -> RouteResult:
    from orchestration.exchange import FileExchangeRuntime, validate_agent_response
    from .models import ValidationErrorRecord
    from .redaction import redact
    rec = invocation.provenance
    error = None
    validated = None
    try:
        validated = validate_agent_response(invocation.candidate, spec, task)
    except (ValueError, KeyError, TypeError) as exc:
        error = redact(str(exc))
        # Preserve the contract-validation failure in the provenance record (mirrors the driver),
        # so a FAIL is auditable from the persisted artifact, not only from stdout.
        rec.validation_errors.append(
            ValidationErrorRecord(stage="contract_validation", message=error))
    accepted = False
    if validated is not None and mode == "primary":
        FileExchangeRuntime(context.exchange_dir).accept(spec, task["task_id"],
                                                          invocation.provenance.raw_response)
        accepted = True
        rec.accepted = True
        rec.controller_mutated = True   # exchange acceptance is the recorded state for this path
    path = _write_provenance(context, invocation)
    return RouteResult(strategy, accepted, rec.controller_mutated, error, path, validated)


def _accept_via_dispatch(invocation, spec, context, controller, registry, mode) -> RouteResult:
    rec = invocation.provenance
    # Fail-closed: the producer's typed output must claim the invoking role.
    claimed = _get(invocation.candidate, "requested_by_role")
    if claimed != getattr(spec, "name", None):
        path = _write_provenance(context, invocation)
        return RouteResult("producer_dispatch", False, False,
                           f"role mismatch: output claims '{claimed}', spec is "
                           f"'{getattr(spec, 'name', None)}'", path)
    if controller is None or registry is None:
        path = _write_provenance(context, invocation)
        return RouteResult("producer_dispatch", False, False,
                           "producer routing requires a controller + registry", path)
    from .controller_bridge import dispatch_via_controller
    dmode = "primary" if mode == "primary" else "dry_run"  # shadow/dry_run never mutate
    outcome = dispatch_via_controller(invocation.candidate, controller=controller,
                                      registry=registry, mode=dmode)
    accepted = outcome.status in ("EXECUTED", "DRY_RUN")
    mutated = outcome.status == "EXECUTED"
    rec.accepted = accepted
    rec.controller_mutated = mutated
    error = None if accepted else f"{outcome.status}: {outcome.reason}"
    path = _write_provenance(context, invocation)
    return RouteResult("producer_dispatch", accepted, mutated, error, path, outcome)


def _accept_typed_result(invocation, spec, context, mode) -> RouteResult:
    # The runtime already produced + Pydantic-validated the typed output (OrchestratorPlan /
    # LiteratureEvidence). It is advisory: recorded as provenance, no controller mutation.
    rec = invocation.provenance
    accepted = mode == "primary" and invocation.candidate is not None
    rec.accepted = accepted
    rec.controller_mutated = False
    path = _write_provenance(context, invocation)
    return RouteResult("typed_result", accepted, False, None, path, invocation.candidate)
