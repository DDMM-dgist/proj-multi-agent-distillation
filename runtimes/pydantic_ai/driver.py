"""The acceptance pipeline. A runtime yields a candidate; the driver revalidates it with
the EXISTING contract validators and only then records it through the file exchange.
Pydantic parsing success is never sufficient (see the pipeline in the module docstring).

    runtime.run() -> candidate + provenance
      -> orchestration.exchange.validate_agent_response  (contract + JudgeVote lens)
      -> FileExchangeRuntime.accept  (raw preservation + record)   [primary mode]
    shadow mode: validate + persist provenance, but DO NOT accept into the exchange.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestration.exchange import FileExchangeRuntime, validate_agent_response

from .interface import AgentInvocation
from .models import RuntimeContext, ValidationErrorRecord


class DriverResult:
    def __init__(self, invocation, accepted, validated, error, provenance_path):
        self.invocation = invocation
        self.accepted = accepted
        self.validated = validated
        self.error = error
        self.provenance_path = provenance_path


def _write_provenance(context: RuntimeContext, invocation: AgentInvocation) -> Path:
    """Persist the full invocation record (raw + parsed + hashes + validation errors)."""
    prov_dir = Path(context.exchange_dir).resolve() / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    rec = invocation.provenance
    # Attempt-scoped filename so retries never overwrite a prior attempt's record.
    path = prov_dir / f"{rec.task_id}.{rec.attempt_id}.json"
    path.write_text(rec.model_dump_json(indent=2) + "\n")
    return path


def run_task(runtime, task, spec, context: RuntimeContext, *, shadow: bool = False):
    """Execute one task through a runtime and the existing validation pipeline.

    shadow=False (primary): a valid result is recorded via FileExchangeRuntime.accept
        (which itself preserves the raw response and enforces the contract again).
    shadow=True: validate and persist provenance only; NEVER accept into the exchange,
        so a shadow runtime cannot change controller-visible state.
    """
    invocation = runtime.run(task, spec, context)

    # Contract + physics-agnostic validation. This is the real gate — not Pydantic.
    validated = None
    error = None
    try:
        validated = validate_agent_response(invocation.candidate, spec, task)
    except (ValueError, KeyError, TypeError) as exc:
        error = str(exc)
        invocation.provenance.validation_errors.append(
            ValidationErrorRecord(stage="contract_validation", message=error))

    accepted = False
    if validated is not None and not shadow:
        exchange = FileExchangeRuntime(context.exchange_dir)
        # accept() re-preserves raw and re-validates; feed it the exact raw response.
        exchange.accept(spec, task["task_id"], invocation.provenance.raw_response)
        accepted = True
        invocation.provenance.accepted = True

    provenance_path = _write_provenance(context, invocation)
    return DriverResult(invocation, accepted, validated, error, provenance_path)
