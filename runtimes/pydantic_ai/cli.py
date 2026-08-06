"""Production CLI for the PydanticAI runtime (Phase 2/D6).

Runs ONE task through a runtime and the existing validation pipeline, with explicit modes and
meaningful exit codes. A real provider call happens only in ``--runtime pydantic-ai`` mode AND
only when credentials preflight READY; with no credential the CLI exits PROVIDER_UNAVAILABLE
without contacting any provider (a missing key is never a silent success).

Usage:
    python -m runtimes.pydantic_ai.cli run-task \
        --runtime pydantic-ai --agent judge --agent-specs-dir agent_specs \
        --task task.json --exchange-dir runs/x/exchange --mode shadow

    python -m runtimes.pydantic_ai.cli run-task \
        --runtime mock --agent judge --agent-specs-dir agent_specs \
        --task task.json --exchange-dir /tmp/ex --mock-response resp.json --mode validate-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Meaningful, distinct exit codes.
EXIT_SUCCESS = 0
EXIT_VALIDATION_REJECTED = 2
EXIT_PROVIDER_UNAVAILABLE = 3
EXIT_PROVIDER_FAILURE = 4
EXIT_APPROVAL_REQUIRED = 5
EXIT_BLOCKED_POLICY = 6
EXIT_DUPLICATE = 7
EXIT_INTERNAL = 8

_PROVIDER_UNAVAILABLE_FAILURES = {"authentication_failure"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pydantic-ai-runtime", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run-task", help="run one task through the runtime")
    r.add_argument("--runtime", choices=("mock", "pydantic-ai"), required=True)
    r.add_argument("--agent", required=True, help="agent/role name (spec basename)")
    r.add_argument("--agent-specs-dir", default="agent_specs")
    r.add_argument("--task", required=True, help="path to the task JSON")
    r.add_argument("--exchange-dir", required=True)
    r.add_argument("--repo-root", default=".")
    r.add_argument("--read-allow", action="append", default=[],
                   help="read-only allow-list prefix (repeatable)")
    r.add_argument("--provider", default=None)
    r.add_argument("--model", default=None, help="provider model id, else $PYDANTIC_AI_MODEL")
    r.add_argument("--mode", choices=("primary", "shadow", "dry-run", "validate-only"),
                   default="shadow")
    r.add_argument("--mock-response", default=None,
                   help="[--runtime mock] file with the canned raw response JSON")
    r.add_argument("--correlation-id", default="")
    return p


def _print_kv(out, **kw):
    for k, v in kw.items():
        print(f"{k}: {v}", file=out)


def main(argv=None) -> int:
    import os
    args = _build_parser().parse_args(argv)
    if args.command != "run-task":  # pragma: no cover
        return EXIT_INTERNAL
    out = sys.stdout

    # Deferred imports so `--help` and non-pydantic environments don't require the extra.
    try:
        from orchestration.specs import load_agent_specs
        from .models import RuntimeContext
        from .driver import run_task
    except Exception as exc:  # pragma: no cover
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL

    try:
        specs = load_agent_specs(args.agent_specs_dir)
    except Exception as exc:
        print(f"could not load agent specs from {args.agent_specs_dir}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    if args.agent not in specs:
        print(f"unknown agent '{args.agent}'; known: {sorted(specs)}", file=sys.stderr)
        return EXIT_BLOCKED_POLICY
    spec = specs[args.agent]

    task_path = Path(args.task)
    try:
        task = json.loads(task_path.read_text())
    except Exception as exc:
        print(f"could not read task {task_path}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL

    model_id = args.model or os.environ.get("PYDANTIC_AI_MODEL")
    provider = args.provider or (model_id.split(":", 1)[0] if model_id and ":" in model_id else "mock")

    # Build the runtime.
    if args.runtime == "mock":
        from .mock_runtime import MockAgentRuntime
        if not args.mock_response:
            print("--runtime mock requires --mock-response", file=sys.stderr)
            return EXIT_INTERNAL
        raw = Path(args.mock_response).read_text()
        runtime = MockAgentRuntime(lambda t, s, ts: (raw, (0, 0)))
        provider = provider if provider != "mock" else "mock"
    else:  # pydantic-ai
        from .provider import preflight_credentials, build_provider_model
        pf = preflight_credentials()
        if pf.status != "READY":
            _print_kv(out, runtime="pydantic-ai", preflight=pf.status, reason=pf.reason,
                      provider=pf.provider, model=pf.model_id)
            return EXIT_PROVIDER_UNAVAILABLE
        from .pydantic_ai_runtime import PydanticAIRuntime
        model_id = pf.model_id
        provider = pf.provider
        runtime = PydanticAIRuntime(model=build_provider_model(model_id), usage_source="provider")

    ctx = RuntimeContext(
        exchange_dir=args.exchange_dir, repo_root=args.repo_root,
        provider=provider, model_id=model_id or "mock",
        read_allow_prefixes=args.read_allow, correlation_id=args.correlation_id)

    mode_flags = {
        "primary": {}, "shadow": {"shadow": True},
        "dry-run": {"dry_run": True}, "validate-only": {"validate_only": True},
    }[args.mode]

    try:
        res = run_task(runtime, task, spec, ctx, **mode_flags)
    except FileExistsError:
        print("duplicate task dispatch (task packet already exists)", file=sys.stderr)
        return EXIT_DUPLICATE
    except Exception as exc:  # pragma: no cover
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL

    rec = res.invocation.provenance
    _print_kv(
        out,
        task_path=str(task_path), task_id=task.get("task_id", ""), role=args.agent,
        runtime=args.runtime, provider=provider, model=model_id or "mock", mode=args.mode,
        result_path=str(Path(args.exchange_dir) / "results" / f"{task.get('task_id','')}.json"),
        raw_response_path=str(Path(args.exchange_dir) / "raw" / f"{task.get('task_id','')}.json"),
        provenance_path=str(res.provenance_path),
        accepted=res.accepted, validation_status=("ok" if res.validated is not None else "rejected"),
        attempt_id=rec.attempt_id, controller_mutation=rec.controller_mutated,
        failure_category=rec.failure_category or "")

    if rec.failure_category:
        return (EXIT_PROVIDER_UNAVAILABLE
                if rec.failure_category in _PROVIDER_UNAVAILABLE_FAILURES
                else EXIT_PROVIDER_FAILURE)
    if res.validated is None:
        return EXIT_VALIDATION_REJECTED
    return EXIT_SUCCESS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
