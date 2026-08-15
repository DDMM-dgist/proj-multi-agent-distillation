"""Single production entry point that routes an agent invocation to the correct acceptance
strategy by ROLE / typed-output — so a caller never hand-picks an internal function (Phase 6/0).

Strategies (selected from the role's typed output model, not by the caller):
- judge_gate      : JudgeVote -> canonical validation -> FileExchange accept (gate aggregation).
- producer_dispatch: role-scoped ActionProposal -> role/capability/approval/idempotency
                     -> trusted executor -> controller record (dispatch_via_controller).
- typed_result    : OrchestratorPlan / LiteratureEvidence -> Pydantic-validated typed output,
                     recorded as provenance; NO controller mutation.
- typed_reasoning_output: a task-declared, REGISTERED reasoning-output model (e.g. the Analyst's
                     RootCauseClassification) -> Pydantic-validated + an optional caller-supplied
                     contextual validator (fail-closed) -> hash-bound persisted artifact; NO
                     controller mutation and NO executor/action dispatch. Distinct from
                     typed_result: it is selected per-TASK (role_outputs.select_output_model's
                     ``task`` argument), carries its own contextual (not just shape) validation,
                     and its accepted artifact is bound to a sha256 for downstream provenance
                     (e.g. RecoveryPlanDraft.diagnosis_binding) rather than only written to the
                     provenance record. Generic: this router never names a specific reasoning
                     model; see role_outputs.register_reasoning_output_model for the registry.
- agent_result    : generic AgentResult (fallback / unknown role).

Mode semantics are uniform: only ``primary`` may mutate exchange/controller/reasoning-artifact
state; ``shadow`` and ``dry_run`` never do. Wrong role/output combinations are fail-closed.
Provenance is written on every path. This router changes NO scientific (gate/recovery) semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .driver import _write_provenance
from .role_outputs import is_reasoning_output_model, select_output_model


@dataclass
class RouteResult:
    strategy: str
    accepted: bool
    controller_mutated: bool
    error: Optional[str]
    provenance_path: Path
    detail: object = None  # validated payload (judge/typed) or ActionOutcome (producer)


@dataclass
class ReasoningOutputAcceptance:
    """``RouteResult.detail`` for the ``typed_reasoning_output`` strategy. ``instance`` is the
    validated Pydantic model (None if rejected); ``artifact_path``/``artifact_sha256`` are the
    hash-bound persisted copy a caller can bind downstream provenance to (e.g. a RecoveryPlan
    draft's diagnosis_binding), present only when accepted."""
    instance: object = None
    artifact_path: Optional[Path] = None
    artifact_sha256: Optional[str] = None


def acceptance_strategy(spec, task: Optional[dict] = None) -> str:
    model = select_output_model(spec, task)
    if model.__name__ == "JudgeVoteModel":
        return "judge_gate"
    if is_reasoning_output_model(model):
        return "typed_reasoning_output"
    if model.__name__.endswith("ActionProposal"):
        return "producer_dispatch"
    if model.__name__ in ("OrchestratorPlan", "LiteratureEvidence"):
        return "typed_result"
    return "agent_result"


def _get(candidate, key, default=None):
    return candidate.get(key, default) if isinstance(candidate, dict) else default


def run_role(runtime, task, spec, context, *, controller=None, registry=None, mode="shadow",
            reasoning_validator: Optional[Callable[[Any], Any]] = None) -> RouteResult:
    """Invoke one role and accept its result via the role-appropriate strategy. ``mode`` in
    {"primary","shadow","dry_run","validate_only"} — only primary mutates state.

    ``reasoning_validator``, used only by the ``typed_reasoning_output`` strategy, is an optional
    ``instance -> instance`` callable performing CONTEXTUAL fail-closed validation beyond Pydantic
    shape (e.g. root_cause.validate_root_cause_classification bound to this run's available
    artifacts/valid recovery targets). It raises to reject; this router stays ignorant of what it
    checks."""
    invocation = runtime.run(task, spec, context)
    rec = invocation.provenance
    rec.mode = mode
    strategy = acceptance_strategy(spec, task)

    # A provider attempt that raised before output is a preserved failure; no acceptance.
    if getattr(rec, "failure_category", ""):
        path = _write_provenance(context, invocation)
        return RouteResult(strategy, False, False, rec.exception_message or rec.failure_category, path)

    if strategy in ("judge_gate", "agent_result"):
        return _accept_via_exchange(invocation, spec, task, context, mode, strategy)

    # producer_dispatch / typed_result / typed_reasoning_output: fail-closed Pydantic validation
    # of the typed output against the role's model (so a malformed/wrong-shape output is rejected
    # regardless of which runtime produced it — the real runtime already enforces output_type;
    # this guards test/other runtimes too).
    instance, err = _validate_typed(invocation.candidate, spec, task)
    if err:
        path = _write_provenance(context, invocation)
        return RouteResult(strategy, False, False, err, path)

    if strategy == "producer_dispatch":
        return _accept_via_dispatch(invocation, spec, context, controller, registry, mode)
    if strategy == "typed_reasoning_output":
        return _accept_typed_reasoning_output(invocation, spec, context, mode, instance,
                                              reasoning_validator)
    return _accept_typed_result(invocation, spec, context, mode)


def _validate_typed(candidate, spec, task: Optional[dict] = None):
    model = select_output_model(spec, task)
    try:
        return model(**(candidate or {})), None
    except Exception as exc:  # pydantic.ValidationError etc. -> fail-closed
        return None, f"typed-output validation failed: {type(exc).__name__}: {exc}"


def _accept_via_exchange(invocation, spec, task, context, mode, strategy) -> RouteResult:
    from orchestration.exchange import (FileExchangeRuntime, bind_authoritative_judge_vote,
                                         validate_agent_response)
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
    # Deterministic-verdict ownership: record which verdict was ACCEPTED (deterministic for an
    # authoritative gate; the LLM's for advisory) plus the LLM's proposed verdict + any flagged
    # criterion contradictions, so the ownership boundary is auditable from the artifact.
    if validated is not None and getattr(spec, "result_contract", None) == "JudgeVote":
        _bound, brec = bind_authoritative_judge_vote(invocation.candidate, task)
        if brec.get("authoritative"):
            rec.accepted_verdict = brec["authoritative_verdict"]
            rec.llm_proposed_verdict = brec["llm_proposed_verdict"]
            rec.verdict_overridden = bool(brec["verdict_overridden"])
            rec.criterion_contradictions = list(brec["criterion_contradictions"])
        else:
            rec.accepted_verdict = validated.get("verdict")
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


def _persist_reasoning_artifact(context, rec, instance) -> tuple[Path, str]:
    """Hash-bind the accepted reasoning output as its own on-disk artifact (distinct from the
    provenance record, which is a per-attempt log): attempt-scoped filename so a retry never
    overwrites a prior accepted artifact, mirroring driver._write_record's convention."""
    from workflow.integrity import sha256_file
    out_dir = Path(context.exchange_dir).resolve() / "reasoning_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rec.task_id}.{rec.attempt_id}.{type(instance).__name__}.json"
    path.write_text(instance.model_dump_json(indent=2) + "\n")
    return path, sha256_file(path)


def _accept_typed_reasoning_output(invocation, spec, context, mode, instance,
                                   reasoning_validator) -> RouteResult:
    """A registered reasoning-output model (see role_outputs.register_reasoning_output_model):
    Pydantic-validated shape (already done by the caller) + an optional contextual validator,
    then — only in primary mode, only if accepted — persisted as a hash-bound artifact. Never
    mutates the controller and never dispatches an executor: accepting a diagnosis/reasoning
    result is not the same as acting on it."""
    rec = invocation.provenance
    error = None
    if reasoning_validator is not None:
        try:
            instance = reasoning_validator(instance)
        except Exception as exc:  # fail-closed: any contextual rejection is a rejection
            error = f"reasoning output validation failed: {type(exc).__name__}: {exc}"
    accepted = error is None and mode == "primary"
    rec.accepted = accepted
    rec.controller_mutated = False
    artifact_path = artifact_sha256 = None
    if accepted:
        artifact_path, artifact_sha256 = _persist_reasoning_artifact(context, rec, instance)
    path = _write_provenance(context, invocation)
    detail = ReasoningOutputAcceptance(instance=instance if error is None else None,
                                      artifact_path=artifact_path, artifact_sha256=artifact_sha256)
    return RouteResult("typed_reasoning_output", accepted, False, error, path, detail)
