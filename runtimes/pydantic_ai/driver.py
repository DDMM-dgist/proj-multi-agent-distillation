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
from .redaction import redact


class DriverResult:
    def __init__(self, invocation, accepted, validated, error, provenance_path):
        self.invocation = invocation
        self.accepted = accepted
        self.validated = validated
        self.error = error
        self.provenance_path = provenance_path


def _write_record(prov_dir: Path, rec) -> Path:
    # Attempt-scoped filename so retries never overwrite a prior attempt's record.
    path = prov_dir / f"{rec.task_id}.{rec.attempt_id}.json"
    path.write_text(rec.model_dump_json(indent=2) + "\n")
    return path


def _write_provenance(context: RuntimeContext, invocation: AgentInvocation) -> Path:
    """Persist the full invocation record AND every prior retry attempt (none overwritten)."""
    prov_dir = Path(context.exchange_dir).resolve() / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    for prior in getattr(invocation, "prior_attempts", []) or []:
        _write_record(prov_dir, prior)
    return _write_record(prov_dir, invocation.provenance)


def run_task(runtime, task, spec, context: RuntimeContext, *, shadow: bool = False,
             dry_run: bool = False, validate_only: bool = False):
    """Execute one task through a runtime and the existing validation pipeline.

    Modes (only ``primary`` may mutate exchange-visible state):
      - primary (default): a valid result is recorded via FileExchangeRuntime.accept
        (which re-preserves the raw response and enforces the contract again).
      - shadow: validate + persist provenance only; NEVER accept.
      - validate_only: validate + persist provenance; NEVER accept (explicit CI/preflight mode).
      - dry_run: validate + persist provenance; NEVER accept (producer side effects are also
        suppressed downstream by the executor bridge in Phase 4-5).
    """
    invocation = runtime.run(task, spec, context)
    rec = invocation.provenance
    mode = "primary"
    if shadow:
        mode = "shadow"
    elif validate_only:
        mode = "validate_only"
    elif dry_run:
        mode = "dry_run"
    rec.mode = mode

    error = None
    # A provider attempt that raised before producing output is already a preserved failure
    # record; there is no candidate to contract-validate. Do not accept.
    if getattr(rec, "failure_category", ""):
        error = rec.exception_message or rec.failure_category
        rec.controller_mutated = False
        provenance_path = _write_provenance(context, invocation)
        return DriverResult(invocation, False, None, error, provenance_path)

    # Contract + physics-agnostic validation. This is the real gate — not Pydantic.
    validated = None
    try:
        validated = validate_agent_response(invocation.candidate, spec, task)
    except (ValueError, KeyError, TypeError) as exc:
        error = redact(str(exc))
        invocation.provenance.validation_errors.append(
            ValidationErrorRecord(stage="contract_validation", message=error))

    accepted = False
    if validated is not None and mode == "primary":
        exchange = FileExchangeRuntime(context.exchange_dir)
        # accept() re-preserves raw and re-validates; feed it the exact raw response.
        exchange.accept(spec, task["task_id"], invocation.provenance.raw_response)
        accepted = True
        rec.accepted = True
        rec.controller_mutated = True

    provenance_path = _write_provenance(context, invocation)
    return DriverResult(invocation, accepted, validated, error, provenance_path)
