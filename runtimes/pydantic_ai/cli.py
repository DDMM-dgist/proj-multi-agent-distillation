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
import os
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
MAX_PRODUCER_GENERATION_ATTEMPTS = 3
_AUTH_BINDING_MISMATCH = "authoritative action binding mismatch"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pydantic-ai-runtime", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="verify run inputs and build bounded evidence")
    preflight.add_argument("--run-dir", required=True)
    approve = sub.add_parser("approve", help="record explicit human approval for an action boundary")
    approve.add_argument("--run-dir", required=True)
    approve.add_argument("--boundary", required=True)
    approve.add_argument("--note", required=True)
    stage = sub.add_parser("run-stage", help="run one production-routed stage")
    stage.add_argument("--run-dir", required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--runtime", choices=("mock", "pydantic-ai"), required=True,
                       help="required; use pydantic-ai for production, mock only in tests")
    stage.add_argument("--mock-response", default=None)
    stage.add_argument("--agent-specs-dir", default="agent_specs")
    stage.add_argument("--exchange-dir", default=None)
    stage.add_argument("--repo-root", default=".")
    stage.add_argument("--auto-mock-judges", action="store_true",
                       help="test-only: create three PASS judge votes from frozen evidence")
    stage.add_argument("--mock-judge-response", action="append", default=[],
                       help="test-only: one canned JudgeVote JSON per judge context")
    r = sub.add_parser("run-task", help="run one task through the runtime")
    r.add_argument("--runtime", choices=("mock", "pydantic-ai"), required=True)
    r.add_argument("--agent", required=True, help="agent/role name (spec basename)")
    r.add_argument("--agent-specs-dir", default="agent_specs")
    r.add_argument("--task", required=True, help="path to the task JSON")
    r.add_argument("--exchange-dir", required=True)
    r.add_argument("--run-dir", default=None,
                   help="controller run dir (required for producer roles' dispatch)")
    r.add_argument("--repo-root", default=".")
    r.add_argument("--read-allow", action="append", default=[],
                   help="read-only allow-list prefix (repeatable)")
    r.add_argument("--provider", default=None,
                   help="provider kind: local-openai | ollama | anthropic; else $PYDANTIC_AI_PROVIDER")
    r.add_argument("--model", default=None, help="provider model id, else $PYDANTIC_AI_MODEL")
    r.add_argument("--base-url", default=None,
                   help="[local] OpenAI-compatible base URL, else $PYDANTIC_AI_BASE_URL")
    r.add_argument("--probe-server", action="store_true",
                   help="[local] TCP-probe the server during preflight (no inference)")
    r.add_argument("--mode", choices=("primary", "shadow", "dry-run", "validate-only"),
                   default="shadow")
    r.add_argument("--mock-response", default=None,
                   help="[--runtime mock] file with the canned raw response JSON")
    r.add_argument("--correlation-id", default="")
    return p


def _print_kv(out, **kw):
    for k, v in kw.items():
        print(f"{k}: {v}", file=out)



def _stage_config(controller, stage_name):
    import yaml
    cfg = yaml.safe_load(Path(controller.state["workflow_config"]).read_text()) or {}
    for stage in cfg.get("stages", []):
        if stage.get("name") == stage_name:
            return stage
    return {}


def _default_stage_route(stage_name):
    return {
        "teacher_baseline": ("simulation", "build_teacher_baseline", "costly_teacher_labeling"),
        "reference_validation": ("simulation", "validate_teacher_reference", "costly_teacher_labeling"),
        "acquisition": ("data-curator", "acquire_structures", "costly_teacher_labeling"),
        "teacher_labeling": ("data-curator", "label_with_teacher", "costly_teacher_labeling"),
        "training": ("ml-trainer", "train_committee", "costly_training"),
        "evaluation": ("ml-trainer", "evaluate_heldout_fidelity", None),
        "teacher_md": ("simulation", "run_teacher_md", "production_md"),
        "deployment_md": ("simulation", "run_student_md", "production_md"),
    }.get(stage_name)


def _input_source(controller, contains=None, suffix=None, exclude_contains=None):
    for record in controller.state.get("inputs", []):
        path = record.get("snapshot") or record.get("source")
        if not path:
            continue
        name = Path(path).name
        full = str(path)
        if contains and contains not in name and contains not in full:
            continue
        if exclude_contains and (exclude_contains in name or exclude_contains in full):
            continue
        if suffix and not name.endswith(suffix):
            continue
        return path
    return None


def _fill_default_parameters(controller, stage_name, params):
    if not params and stage_name == "reference_validation":
        stage = controller.stage(stage_name)
        outputs = stage.get("outputs") or []
        if len(outputs) != 2:
            raise ValueError("reference_validation requires exactly two declared outputs")
        teacher = _input_source(controller, "teacher", ".yaml") or _input_source(controller, "teacher.allegro", ".yaml")
        if not teacher:
            raise ValueError("reference_validation requires a bound Teacher configuration input")
        return {
            "teacher_config": teacher,
            "report_path": str((controller.run_dir / outputs[0]).resolve()),
            "predictions_path": str((controller.run_dir / outputs[1]).resolve()),
            "domain_fields": ["structural_domain"],
        }
    if stage_name == "teacher_baseline" and "structures_path" not in params:
        raise ValueError("teacher_baseline requires explicit pydantic_ai.parameters.structures_path")
    if params or stage_name != "teacher_baseline":
        return params
    stage = controller.stage(stage_name)
    outputs = stage.get("outputs") or []
    report = str((controller.run_dir / outputs[0]).resolve()) if outputs else str(
        controller.run_dir / "artifacts" / "teacher_baseline.json")
    teacher = _input_source(controller, "teacher", ".yaml") or _input_source(controller, "teacher.allegro", ".yaml")
    scope = _input_source(controller, "distillation_scope", ".yaml")
    profile = (_input_source(controller, "validation_profile", ".yaml") or
               _input_source(controller, "validation", ".yaml"))
    structures = (_input_source(controller, "bulk_cryst", ".xyz") or
                  _input_source(controller, suffix=".xyz", exclude_contains="protected_reference"))
    missing = [name for name, value in {
        "teacher_config": teacher,
        "distillation_scope": scope,
        "validation_profile": profile,
        "structures_path": structures,
    }.items() if not value]
    if missing:
        raise ValueError("teacher_baseline cannot infer required inputs: " + ", ".join(missing))
    return {
        "teacher_config": teacher,
        "distillation_scope": scope,
        "validation_profile": profile,
        "structures_path": structures,
        "report_path": report,
        "labeled_output": str(controller.run_dir / "artifacts" / "teacher_baseline_operational.extxyz"),
        "label_manifest_path": str(controller.run_dir / "artifacts" / "teacher_baseline_labels.manifest.json"),
        "applicability_status": "NOT_ESTABLISHED",
        "applicability_limitations": ["deployment-domain evidence gap remains NOT_ESTABLISHED"],
        "deployment_domain": {"stage": "teacher_baseline", "source": "declared operational structures"},
        "require_lineage": False,
    }


def _protected_reference_from_inputs(controller):
    import yaml
    from validation.protected_reference import validate_reference_config
    found = []
    for record in controller.state.get("inputs", []):
        raw = record.get("snapshot") or record.get("source")
        if not raw or Path(raw).suffix.lower() not in {".yaml", ".yml"}:
            continue
        try:
            payload = yaml.safe_load(Path(raw).read_text()) or {}
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("kind") == "protected-existing-dft":
            validate_reference_config(raw)
            found.append(str(Path(raw).resolve()))
    if len(found) > 1:
        raise ValueError("multiple protected reference configs are bound to this run")
    return found[0] if found else None


def _protection_consuming_action(action):
    return action in {"acquire_structures", "label_with_teacher", "train_committee"}


def _proposal_from_stage(controller, stage_name, stage_cfg):
    route = stage_cfg.get("pydantic_ai") or {}
    if not route:
        default = _default_stage_route(stage_name)
        if not default:
            raise ValueError("stage requires pydantic_ai role/action metadata")
        role, action, boundary = default
        params = {}
    else:
        role = route["role"]
        action = route["action"]
        boundary = route.get("approval_boundary")
        params = dict(route.get("parameters") or {})
    context = {"run_dir": str(controller.run_dir),
               "artifacts_dir": str(controller.run_dir / "artifacts"),
               "project_dir": controller.state["project_dir"]}
    def fmt(value):
        if isinstance(value, str):
            return value.format(**context)
        if isinstance(value, list):
            return [fmt(v) for v in value]
        if isinstance(value, dict):
            return {k: fmt(v) for k, v in value.items()}
        return value
    params = fmt(_fill_default_parameters(controller, stage_name, params))
    protected_reference = _protected_reference_from_inputs(controller)
    if action == "validate_teacher_reference":
        if not protected_reference:
            raise ValueError("reference_validation requires a controller-bound protected reference")
        existing = params.get("reference_yaml")
        if existing is not None and str(Path(existing).resolve()) != protected_reference:
            raise ValueError("stage proposal reference_yaml does not match the controller-bound protected reference")
        params["reference_yaml"] = protected_reference
    elif protected_reference and _protection_consuming_action(action):
        existing = params.get("reference_yaml")
        if existing is not None and str(Path(existing).resolve()) != protected_reference:
            raise ValueError("stage proposal reference_yaml does not match the controller-bound protected reference")
        params["reference_yaml"] = protected_reference
    return {
        "schema_version": 1,
        "run_id": controller.state["run_id"],
        "stage": stage_name,
        "requested_at": "controller-stage-runner",
        "rationale": f"execute run stage {stage_name} through trusted production dispatch",
        "parameters": params,
        "expected_outputs": list(controller.stage(stage_name).get("outputs", [])),
        "approval_boundary": boundary,
        "idempotency_key": route.get("idempotency_key", f"{controller.state['run_id']}:{stage_name}:001"),
        "dry_run": False,
        "requested_by_role": role,
        "action_type": action,
    }, role


_BOUND_PROPOSAL_FIELDS = {
    "run_id", "stage", "requested_by_role", "action_type", "approval_boundary",
    "idempotency_key", "expected_outputs", "parameters",
}


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def _resolve_run_output(run_dir, raw):
    path = Path(raw)
    if not path.is_absolute():
        if ".." in path.parts:
            raise ValueError("expected_outputs may not contain path traversal")
        path = run_dir / path
    resolved = path.resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError("expected_outputs must resolve inside the controller run_dir")
    return resolved


def _expected_outputs_match(run_dir, authoritative, candidate):
    if not isinstance(authoritative, list) or not isinstance(candidate, list):
        return False, "expected_outputs must be lists"
    if len(authoritative) != len(candidate):
        return False, "expected_outputs count differs"
    try:
        auth_paths = [_resolve_run_output(run_dir, value) for value in authoritative]
        cand_paths = [_resolve_run_output(run_dir, value) for value in candidate]
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    for index, (left, right) in enumerate(zip(auth_paths, cand_paths)):
        if left != right:
            return False, f"expected_outputs target differs at index {index}"
    return True, ""


def _proposal_binding_validator(authoritative, controller=None):
    run_dir = Path(controller.run_dir).resolve() if controller is not None else None

    def validate(candidate):
        for field in sorted(_BOUND_PROPOSAL_FIELDS):
            left = candidate.get(field) if isinstance(candidate, dict) else getattr(candidate, field, None)
            right = authoritative.get(field)
            if field == "expected_outputs" and run_dir is not None:
                ok, message = _expected_outputs_match(run_dir, right, left)
                if not ok:
                    return False, f"authoritative action binding mismatch for {field}: {message}"
                continue
            if _canonical(left) != _canonical(right):
                return False, f"authoritative action binding mismatch for {field}"
        return True, ""
    return validate


def _producer_task(stage_name, role, evidence_path, controller, authoritative_proposal,
                   retry_feedback=None):
    instruction = (f"Inspect and reason about the authoritative execution proposal for stage {stage_name}. "
                   "authoritative_action_proposal is immutable execution state. Return exactly the "
                   "same execution-critical binding: copy every execution-critical value exactly, "
                   "preserve relative path strings exactly, preserve ALL parameter keys, preserve "
                   "false values, null values, and empty lists/dicts if present, do not normalize "
                   "paths, do not resolve paths, do not drop keys because they appear optional or "
                   "default, do not stringify lists/dicts/booleans/nulls, do not move path parameters "
                   "into metadata or input_artifacts, and do not change action_type, approval boundary, "
                   "idempotency key, stage, expected outputs, or protected-reference binding. Only "
                   "rationale text may differ.")
    if retry_feedback:
        instruction += (" Your previous ActionProposal failed authoritative binding. "
                        "Use the corrective feedback in context.producer_retry_feedback and return "
                        "the COMPLETE authoritative ActionProposal exactly.")
    context = {"stage": stage_name,
               "authoritative_action_proposal": authoritative_proposal,
               "authoritative_parameter_keys": sorted(
                   (authoritative_proposal.get("parameters") or {}).keys()),
               "authoritative_expected_outputs": list(
                   authoritative_proposal.get("expected_outputs") or [])}
    if retry_feedback:
        context["producer_retry_feedback"] = retry_feedback
    return {
        "schema_version": 1,
        "task_id": f"{stage_name}-producer",
        "agent": role,
        "run_id": controller.state["run_id"],
        "created_at": "controller-stage-runner",
        "instruction": instruction,
        "inputs": [{"role": "bounded_evidence", "path": str(evidence_path)}],
        "criteria": list(controller.stage(stage_name).get("gate_criteria") or ["stage action is valid"]),
        "constraints": ["Return exactly one typed ActionProposal; do not run compute directly."],
        "context": context,
    }



def _load_candidate_from_provenance(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = payload.get("raw_response")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _parameter_diff_summary(authoritative, candidate):
    auth_params = authoritative.get("parameters") or {}
    cand_params = (candidate or {}).get("parameters") or {}
    missing = sorted(set(auth_params) - set(cand_params))
    extra = sorted(set(cand_params) - set(auth_params))
    changed = []

    def walk(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) & set(right)):
                walk(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if left != right:
                changed.append(path)
        elif left != right or type(left) is not type(right):
            changed.append(path)

    for key in sorted(set(auth_params) & set(cand_params)):
        walk(auth_params[key], cand_params[key], f"parameters.{key}")
    bound_changed = []
    for field in sorted(_BOUND_PROPOSAL_FIELDS - {"parameters"}):
        if _canonical((candidate or {}).get(field)) != _canonical(authoritative.get(field)):
            bound_changed.append(field)
    return {"missing_parameter_keys": missing,
            "extra_parameter_keys": extra,
            "changed_fields": sorted(bound_changed + changed)}


def _binding_retry_feedback(authoritative, candidate, message, attempt_index):
    summary = _parameter_diff_summary(authoritative, candidate)
    return {
        "previous_attempt": attempt_index,
        "binding_error": message,
        **summary,
        "instruction": (
            "Return the COMPLETE authoritative ActionProposal. Preserve every execution-critical "
            "field exactly, including all parameter keys, false/null/empty values, lists, dicts, "
            "and path strings. Do not move path parameters into metadata or input_artifacts."
        ),
    }


def _is_authoritative_binding_rejection(result):
    status = getattr(result.detail, "status", "")
    reason = getattr(result.detail, "reason", "") or result.error or ""
    return status == "INVALID" and reason.startswith(_AUTH_BINDING_MISMATCH), reason


def _run_producer_with_binding_retries(runtime, task, spec, context, *, controller, registry,
                                       authoritative_proposal, task_factory):
    from .production_router import run_role
    result = None
    feedback = None
    for attempt in range(1, MAX_PRODUCER_GENERATION_ATTEMPTS + 1):
        current_task = task if feedback is None else task_factory(feedback)
        result = run_role(runtime, current_task, spec, context, controller=controller,
                          registry=registry, mode="primary")
        retryable, reason = _is_authoritative_binding_rejection(result)
        if not retryable or attempt == MAX_PRODUCER_GENERATION_ATTEMPTS:
            return result
        candidate = _load_candidate_from_provenance(result.provenance_path)
        feedback = _binding_retry_feedback(authoritative_proposal, candidate, reason, attempt)
    return result

def _write_mock_response(path, proposal):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposal, indent=2) + "\n")
    return path


def _stage_outputs(controller, stage_name):
    paths = []
    for rel in controller.stage(stage_name).get("outputs", []):
        path = (controller.run_dir / rel).resolve()
        if path.exists():
            paths.append(path)
    return paths


def _stage_validation_outcomes(controller, stage_name):
    outcomes = []
    stage = controller.stage(stage_name)
    contract = stage.get("contract") or {}
    for event in controller.state.get("events", []):
        if event.get("type") != "external_stage_completed" or event.get("stage") != stage_name:
            continue
        outcomes.append({
            "stage": stage_name,
            "validator": contract.get("validator"),
            "contract_kind": contract.get("kind"),
            "contract_manifest": contract.get("manifest"),
            "contract_validated": bool(event.get("contract_validated")),
            "result": "PASS" if event.get("contract_validated") else "FAIL",
        })
    if not outcomes and contract:
        outcomes.append({
            "stage": stage_name,
            "validator": contract.get("validator"),
            "contract_kind": contract.get("kind"),
            "contract_manifest": contract.get("manifest"),
            "contract_validated": False,
            "result": "NOT_RUN",
        })
    return outcomes


def _direct_judge_artifact_paths(gate_context):
    from .bounded_evidence import DIRECT_JUDGE_ARTIFACT_BYTES
    paths = []
    for raw in gate_context.get("artifact_sha256", {}):
        path = Path(raw).resolve()
        if path.exists() and path.stat().st_size <= DIRECT_JUDGE_ARTIFACT_BYTES:
            paths.append(str(path))
    return paths


def _large_artifact_records(gate_context):
    from .bounded_evidence import DIRECT_JUDGE_ARTIFACT_BYTES
    records = []
    for raw, sha in (gate_context.get("artifact_sha256") or {}).items():
        path = Path(raw).resolve()
        size = path.stat().st_size if path.exists() else None
        if size is None or size > DIRECT_JUDGE_ARTIFACT_BYTES:
            records.append({"path": str(path), "sha256": sha, "size": size,
                            "direct_read_exposed": False,
                            "reason": "represented by bounded evidence summary"})
    return records


def _write_three_pass_votes(controller, stage_name):
    ctx = controller.gate_context(stage_name)
    votes = []
    for index, lens in enumerate(ctx["review_lenses"], 1):
        votes.append({
            "judge_id": f"judge-{index}",
            "review_lens": lens["id"],
            "verdict": "PASS",
            "criteria_checked": [
                {"criterion": c, "value_read": "bounded evidence and deterministic artifacts", "ok": True}
                for c in ctx["criteria"]
            ],
            "rationale": "synthetic integration-test judge context accepted the frozen evidence",
            "required_fix": "",
        })
    bundle = {**ctx, "decision": "PASS", "votes": votes}
    path = controller.run_dir / "gates" / f"{stage_name}.stage-runner.votes.json"
    path.write_text(json.dumps(bundle, indent=2) + "\n")
    controller.record_gate(stage_name, votes_path=path)
    return path



def _judge_task(stage_name, judge_index, lens, gate_context, evidence_path, controller):
    inputs = [{"role": "bounded_evidence", "path": str(evidence_path)}]
    for path in _direct_judge_artifact_paths(gate_context):
        inputs.append({"role": "stage_artifact", "path": path})
    large_artifacts = _large_artifact_records(gate_context)
    return {
        "schema_version": 1,
        "task_id": f"{stage_name}-judge-{judge_index}",
        "agent": "judge",
        "run_id": controller.state["run_id"],
        "created_at": "controller-stage-runner",
        "instruction": f"Judge stage {stage_name} against the frozen criteria and bounded evidence.",
        "inputs": inputs,
        "criteria": list(gate_context["criteria"]),
        "constraints": [
            "Use only the provided frozen evidence, compact summaries, hashes, and directly readable artifacts.",
            "Do not assume or inspect another Judge context or vote.",
            "Do not mark REVISE/FAIL solely because a large registered artifact is not exposed for full direct reading.",
            "Do not request compression or filtering merely for LLM readability when deterministic bounded evidence resolves the criterion.",
        ],
        "context": {"review_lens": lens["id"], "review_focus": lens["focus"],
                    "stage": stage_name, "artifact_sha256": gate_context["artifact_sha256"],
                    "large_artifact_policy": {
                        "direct_read_limit_bytes": 1_000_000,
                        "policy": ("Large Controller-registered scientific artifacts are authoritative "
                                   "provenance. Evaluate them through deterministic bounded summaries, "
                                   "hashes, manifests, and validation outcomes when supplied."),
                        "read_limit_is_scientific_failure": False,
                    },
                    "large_artifacts": large_artifacts},
    }


def judge_read_allowlist(gate_context, evidence_path):
    return [str(Path(evidence_path).resolve()), *_direct_judge_artifact_paths(gate_context)]


def run_three_judge_gate(controller, stage_name, specs, runtime_factory, runtime_context_factory,
                         evidence_path, *, mode="primary"):
    from orchestration.exchange import FileExchangeRuntime
    from .production_router import run_role
    gate_context = controller.gate_context(stage_name)
    exchange = FileExchangeRuntime(runtime_context_factory(1).exchange_dir)
    votes = []
    for index, lens in enumerate(gate_context["review_lenses"], 1):
        task = _judge_task(stage_name, index, lens, gate_context, evidence_path, controller)
        exchange.dispatch(specs["judge"], task)
        ctx = runtime_context_factory(index)
        res = run_role(runtime_factory(index), task, specs["judge"], ctx, mode=mode)
        if res.error or res.detail is None:
            raise RuntimeError(f"Judge {index} failed validation: {res.error}")
        vote = dict(res.detail)
        vote["judge_id"] = f"judge-{index}"
        votes.append(vote)
    decision = "FAIL" if any(v["verdict"] == "FAIL" for v in votes) else (
        "PASS" if all(v["verdict"] == "PASS" for v in votes) else "REVISE")
    bundle = {**gate_context, "decision": decision, "votes": votes}
    path = controller.run_dir / "gates" / f"{stage_name}.production.votes.json"
    path.write_text(json.dumps(bundle, indent=2) + "\n")
    controller.record_gate(stage_name, votes_path=path)
    return decision, path


def _cmd_preflight(args) -> int:
    from workflow.controller import RunController
    from .bounded_evidence import build_bounded_evidence
    c = RunController(args.run_dir)
    c.verify_inputs()
    evidence = c.run_dir / "exchange" / "bounded_evidence" / "preflight.json"
    artifacts = [r.get("snapshot") or r.get("source") for r in c.state.get("inputs", [])]
    build_bounded_evidence([a for a in artifacts if a], evidence,
                           protocol_refs=[c.state.get("workflow_config")])
    print(f"preflight: OK\nbounded_evidence: {evidence}")
    return EXIT_SUCCESS


def _cmd_approve(args) -> int:
    from workflow.controller import RunController
    c = RunController(args.run_dir)
    c.grant_action_approval(args.boundary, note=args.note)
    print(f"approval: {args.boundary}")
    return EXIT_SUCCESS


def _cmd_run_stage(args) -> int:
    from orchestration.specs import load_agent_specs
    from workflow.controller import RunController
    from .bounded_evidence import build_bounded_evidence
    from .executors import build_executor_registry
    from .mock_runtime import MockAgentRuntime
    from .models import RuntimeContext
    from .production_router import run_role
    from . import provider as _prov

    c = RunController(args.run_dir)
    try:
        c.verify_inputs()
        stage_cfg = _stage_config(c, args.stage)
        proposal, role = _proposal_from_stage(c, args.stage, stage_cfg)
        evidence_path = c.run_dir / "exchange" / "bounded_evidence" / f"{args.stage}.json"
        upstream = [a["path"] for a in c.state.get("artifacts", [])]
        build_bounded_evidence(upstream, evidence_path, protocol_refs=[c.state.get("workflow_config")])
        specs = load_agent_specs(args.agent_specs_dir)
        task = _producer_task(args.stage, role, evidence_path, c, proposal)
        exchange = Path(args.exchange_dir) if args.exchange_dir else c.run_dir / "exchange"

        def ctx_factory(_index, provider_name="mock", model_id="mock"):
            return RuntimeContext(exchange_dir=str(exchange), repo_root=args.repo_root,
                                  provider=provider_name, model_id=model_id,
                                  read_allow_prefixes=[str(evidence_path.parent), str(c.run_dir)])

        if args.runtime == "mock":
            response_path = Path(args.mock_response) if args.mock_response else _write_mock_response(
                exchange / "stage_runner" / f"{args.stage}.proposal.json", proposal)
            raw = response_path.read_text()
            producer_runtime = MockAgentRuntime(lambda t, s, ts: (raw, (0, 0)))
            if args.mock_judge_response:
                if len(args.mock_judge_response) != 3:
                    raise ValueError("--mock-judge-response must be supplied exactly three times")
                judge_raw = [Path(path).read_text() for path in args.mock_judge_response]

                def judge_runtime_factory(index):
                    return MockAgentRuntime(lambda t, s, ts, i=index: (judge_raw[i - 1], (0, 0)))
            else:
                judge_runtime_factory = None
            runtime_provider = "mock"
            runtime_model = "mock"
        else:
            kind = _prov.select_provider_kind()
            if kind in _prov.LOCAL_KINDS:
                pf = _prov.preflight_local(probe=False)
                if pf.status != _prov.LOCAL_READY:
                    print(f"provider unavailable: {pf.status}: {pf.reason}", file=sys.stderr)
                    return EXIT_PROVIDER_UNAVAILABLE
                if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                    print("APPROVAL_REQUIRED: set PYDANTIC_AI_SMOKE_CONFIRM=yes for one live local inference call")
                    return EXIT_APPROVAL_REQUIRED
                from .pydantic_ai_runtime import PydanticAIRuntime

                def live_runtime_factory(_index):
                    return PydanticAIRuntime(
                        model=_prov.build_local_model(kind, pf.model_id, pf.base_url),
                        usage_source="provider")
                producer_runtime = live_runtime_factory(0)
                judge_runtime_factory = live_runtime_factory
                runtime_provider = kind
                runtime_model = pf.model_id
            elif kind == "anthropic":
                pf = _prov.preflight_credentials()
                if pf.status != "READY":
                    print(f"provider unavailable: {pf.status}: {pf.reason}", file=sys.stderr)
                    return EXIT_PROVIDER_UNAVAILABLE
                if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                    print("APPROVAL_REQUIRED: set PYDANTIC_AI_SMOKE_CONFIRM=yes for one live provider call")
                    return EXIT_APPROVAL_REQUIRED
                from .pydantic_ai_runtime import PydanticAIRuntime

                def live_runtime_factory(_index):
                    return PydanticAIRuntime(model=_prov.build_provider_model(pf.model_id),
                                             usage_source="provider")
                producer_runtime = live_runtime_factory(0)
                judge_runtime_factory = live_runtime_factory
                runtime_provider = pf.provider
                runtime_model = pf.model_id
            else:
                print("provider unavailable: set PYDANTIC_AI_PROVIDER to local-openai|ollama|anthropic", file=sys.stderr)
                return EXIT_PROVIDER_UNAVAILABLE

        ctx = ctx_factory(0, runtime_provider, runtime_model)
        registry = build_executor_registry()
        binding_validator = _proposal_binding_validator(proposal, c)
        for descriptor in registry.values():
            descriptor.param_validator = binding_validator
        def producer_task_factory(feedback):
            return _producer_task(args.stage, role, evidence_path, c, proposal,
                                  retry_feedback=feedback)

        res = _run_producer_with_binding_retries(
            producer_runtime, task, specs[role], ctx, controller=c, registry=registry,
            authoritative_proposal=proposal, task_factory=producer_task_factory)
        status = getattr(res.detail, "status", "")
        if status == "APPROVAL_REQUIRED":
            print(f"APPROVAL_REQUIRED: {proposal.get('approval_boundary') or res.detail.reason}")
            return EXIT_APPROVAL_REQUIRED
        if status not in {"EXECUTED", "DUPLICATE"}:
            print(f"stage dispatch failed: {status}: {getattr(res.detail, 'reason', res.error)}", file=sys.stderr)
            return EXIT_VALIDATION_REJECTED

        declared = [(c.run_dir / rel).resolve() for rel in c.stage(args.stage).get("outputs", [])]
        missing = [str(path) for path in declared if not path.exists()]
        if missing:
            print("stage missing declared outputs: " + ", ".join(missing), file=sys.stderr)
            return EXIT_VALIDATION_REJECTED
        if c.stage(args.stage)["status"] != "completed":
            c.complete_external_stage(args.stage, declared)
        c = RunController(args.run_dir)
        if c.stage(args.stage)["status"] != "completed":
            print("controller stage did not complete", file=sys.stderr)
            return EXIT_VALIDATION_REJECTED
        build_bounded_evidence(declared, evidence_path, protocol_refs=[c.state.get("workflow_config")],
                               validation_outcomes=_stage_validation_outcomes(c, args.stage))

        decision = "NO_GATE"
        vote_path = None
        if c.stage(args.stage).get("gate_criteria"):
            if args.auto_mock_judges:
                vote_path = _write_three_pass_votes(c, args.stage)
                decision = "PASS"
                print(f"judge_votes: {vote_path}")
            else:
                if args.runtime == "mock" and judge_runtime_factory is None:
                    raise ValueError(
                        "mock run-stage with a gate requires either --auto-mock-judges or three --mock-judge-response files")
                gate_ctx = c.gate_context(args.stage)
                judge_allow = judge_read_allowlist(gate_ctx, evidence_path)

                def judge_ctx_factory(i):
                    return RuntimeContext(exchange_dir=str(exchange), repo_root=args.repo_root,
                                          provider=runtime_provider, model_id=runtime_model,
                                          read_allow_prefixes=list(judge_allow))
                decision, vote_path = run_three_judge_gate(
                    c, args.stage, specs, judge_runtime_factory, judge_ctx_factory, evidence_path)
            c = RunController(args.run_dir)
            if c.stage(args.stage)["gate"] != decision:
                print("controller gate did not record the aggregate decision", file=sys.stderr)
                return EXIT_VALIDATION_REJECTED
            if decision != "PASS":
                print(f"GATE_{decision}: recovery path is now controlled by the Controller")
                return EXIT_VALIDATION_REJECTED

        print(f"stage: {args.stage}\naction_status: {status}\ngate: {decision}\nbounded_evidence: {evidence_path}")
        return EXIT_SUCCESS
    except Exception as exc:
        print(f"run-stage failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_REJECTED

def main(argv=None) -> int:
    import os
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        return _cmd_preflight(args)
    if args.command == "approve":
        return _cmd_approve(args)
    if args.command == "run-stage":
        return _cmd_run_stage(args)
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
        # Route to the selected backend. A LOCAL (OpenAI-compatible/Ollama) backend needs NO
        # Anthropic credential; the hosted Anthropic path is kept but optional.
        from . import provider as _prov
        if args.provider:
            os.environ.setdefault("PYDANTIC_AI_PROVIDER", args.provider)
        if args.base_url:
            os.environ.setdefault("PYDANTIC_AI_BASE_URL", args.base_url)
        kind = _prov.select_provider_kind()

        def _confirm_missing(provider_name, model_name, note_extra=""):
            _print_kv(out, runtime="pydantic-ai", provider=provider_name, model=model_name,
                      confirmation="MISSING",
                      note=("set PYDANTIC_AI_SMOKE_CONFIRM=yes to authorize ONE real provider "
                            "call; no provider was called" + note_extra))
            return EXIT_APPROVAL_REQUIRED

        if kind in _prov.LOCAL_KINDS:
            pf = _prov.preflight_local(probe=args.probe_server)
            if pf.status != _prov.LOCAL_READY:
                # A not-running server is an OPERATIONAL status, not a runtime/scientific failure.
                _print_kv(out, runtime="pydantic-ai", provider=kind, preflight=pf.status,
                          reason=pf.reason, base_url=pf.base_url, model=pf.model_id,
                          anthropic_key_required=pf.anthropic_key_required)
                return EXIT_PROVIDER_UNAVAILABLE
            if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                return _confirm_missing(kind, pf.model_id, " (local inference)")
            from .pydantic_ai_runtime import PydanticAIRuntime
            model_id = pf.model_id
            provider = kind
            runtime = PydanticAIRuntime(
                model=_prov.build_local_model(kind, pf.model_id, pf.base_url),
                usage_source="provider")
        elif kind == "anthropic":
            pf = _prov.preflight_credentials()
            if pf.status != "READY":
                _print_kv(out, runtime="pydantic-ai", preflight=pf.status, reason=pf.reason,
                          provider=pf.provider, model=pf.model_id)
                return EXIT_PROVIDER_UNAVAILABLE
            # Explicit human confirmation gate for a REAL (billable) provider call. Preflight being
            # READY (credential present) is not sufficient; a live call also requires an intentional
            # PYDANTIC_AI_SMOKE_CONFIRM=yes, so merely having a key in the shell never bills.
            if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                return _confirm_missing(pf.provider, pf.model_id)
            from .pydantic_ai_runtime import PydanticAIRuntime
            model_id = pf.model_id
            provider = pf.provider
            runtime = PydanticAIRuntime(model=_prov.build_provider_model(model_id),
                                        usage_source="provider")
        else:
            _print_kv(out, runtime="pydantic-ai", preflight="NOT_CONFIGURED",
                      reason=("set PYDANTIC_AI_PROVIDER to local-openai|ollama|anthropic "
                              "(local needs PYDANTIC_AI_BASE_URL; no Anthropic key required)"))
            return EXIT_PROVIDER_UNAVAILABLE

    ctx = RuntimeContext(
        exchange_dir=args.exchange_dir, repo_root=args.repo_root,
        provider=provider, model_id=model_id or "mock",
        read_allow_prefixes=args.read_allow, correlation_id=args.correlation_id)

    cli_mode = {"primary": "primary", "shadow": "shadow", "dry-run": "dry_run",
                "validate-only": "validate_only"}[args.mode]

    # The production router selects the acceptance strategy from the role/typed output; producer
    # dispatch needs a controller + executor registry. No manual per-role function selection.
    from .production_router import run_role, acceptance_strategy
    from .executors import build_executor_registry
    controller = None
    strategy = acceptance_strategy(spec)
    if strategy == "producer_dispatch":
        if not args.run_dir:
            print("producer roles require --run-dir (controller manifest)", file=sys.stderr)
            return EXIT_INTERNAL
        from workflow.controller import RunController
        controller = RunController(args.run_dir)
    registry = build_executor_registry()

    try:
        res = run_role(runtime, task, spec, ctx, controller=controller, registry=registry,
                       mode=cli_mode)
    except FileExistsError:
        print("duplicate task dispatch (task packet already exists)", file=sys.stderr)
        return EXIT_DUPLICATE
    except Exception as exc:  # pragma: no cover
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL

    # canonical_validation is explicit so a shadow run (which never "accepts") still reports
    # whether validate_agent_response passed: for judge_gate/agent_result an empty error + a
    # non-None validated payload means the contract validator passed.
    canonical_validation = "n/a"
    if res.strategy in ("judge_gate", "agent_result"):
        canonical_validation = "passed" if (res.detail is not None and not res.error) else "failed"
    _print_kv(
        out,
        task_path=str(task_path), task_id=task.get("task_id", ""), role=args.agent,
        runtime=args.runtime, provider=provider, model=model_id or "mock", mode=args.mode,
        strategy=res.strategy, canonical_validation=canonical_validation,
        accepted=res.accepted, controller_mutation=res.controller_mutated,
        provenance_path=str(res.provenance_path), error=(res.error or ""))

    # Map the routed result to an exit code.
    outcome_status = getattr(res.detail, "status", None)  # ActionOutcome for producer_dispatch
    if outcome_status is not None:
        return {
            "EXECUTED": EXIT_SUCCESS, "DRY_RUN": EXIT_SUCCESS,
            "DENIED": EXIT_BLOCKED_POLICY, "BLOCKED_CAPABILITY": EXIT_BLOCKED_POLICY,
            "APPROVAL_REQUIRED": EXIT_APPROVAL_REQUIRED, "DUPLICATE": EXIT_DUPLICATE,
            "INVALID": EXIT_VALIDATION_REJECTED, "EXECUTOR_ERROR": EXIT_VALIDATION_REJECTED,
        }.get(outcome_status, EXIT_INTERNAL)
    if res.error and res.strategy in ("judge_gate", "agent_result"):
        return EXIT_VALIDATION_REJECTED
    if res.error:
        return EXIT_VALIDATION_REJECTED
    return EXIT_SUCCESS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
