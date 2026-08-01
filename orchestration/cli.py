"""Inspect agent specs and create provider-neutral task packets."""
import argparse
import json
import sys
from pathlib import Path

from .exchange import FileExchangeRuntime, make_task, validate_agent_response
from .specs import load_agent_specs
from workflow.integrity import artifact_digest


SOURCE_SPEC_DIR = Path(__file__).resolve().parent.parent / "agent_specs"
INSTALLED_SPEC_DIR = Path(sys.prefix) / "share" / "distillation-agents" / "agent_specs"
DEFAULT_SPEC_DIR = SOURCE_SPEC_DIR if SOURCE_SPEC_DIR.is_dir() else INSTALLED_SPEC_DIR


def _input_reference(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must use ROLE=PATH")
    role, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not role.strip() or not path.exists():
        raise argparse.ArgumentTypeError(f"invalid input reference: {value}")
    return {"role": role, "path": str(path), "integrity": artifact_digest(path)}


def _context_value(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("context must use KEY=VALUE")
    key, raw = value.split("=", 1)
    if not key.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("context key and value must be non-empty")
    return key.strip(), raw.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", default=str(DEFAULT_SPEC_DIR))
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("validate-specs")
    sub.add_parser("list")
    task = sub.add_parser("make-task")
    task.add_argument("agent")
    task.add_argument("instruction")
    task.add_argument("exchange_dir")
    task.add_argument("--run-id")
    task.add_argument("--input", action="append", default=[], type=_input_reference)
    task.add_argument("--criterion", action="append", default=[])
    task.add_argument("--constraint", action="append", default=[])
    task.add_argument("--context", action="append", default=[], type=_context_value,
                      help="task context as KEY=VALUE; may be repeated")
    result = sub.add_parser("validate-result")
    result.add_argument("agent")
    result.add_argument("task")
    result.add_argument("result")
    args = parser.parse_args(argv)
    specs = load_agent_specs(args.spec_dir)
    if args.action == "validate-specs":
        print(f"validated {len(specs)} agent specifications")
    elif args.action == "list":
        for spec in specs.values():
            print(f"{spec.name}\t{spec.role_type}\t{spec.description}")
    elif args.action == "make-task":
        if args.agent not in specs:
            parser.error(f"unknown agent: {args.agent}")
        context = dict(args.context)
        if len(context) != len(args.context):
            parser.error("task context keys must be unique")
        packet = make_task(args.agent, args.instruction, run_id=args.run_id,
                           inputs=args.input, criteria=args.criterion,
                           constraints=args.constraint, context=context)
        path = FileExchangeRuntime(args.exchange_dir).dispatch(specs[args.agent], packet)
        print(path)
    else:
        if args.agent not in specs:
            parser.error(f"unknown agent: {args.agent}")
        task_payload = json.loads(Path(args.task).read_text())
        result_payload = json.loads(Path(args.result).read_text())
        validate_agent_response(result_payload, specs[args.agent], task_payload)
        print("valid agent result")


if __name__ == "__main__":
    main()
