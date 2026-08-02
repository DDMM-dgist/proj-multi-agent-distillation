"""Provider-neutral JSON task/result packets for agent runtimes."""
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

from .specs import AgentSpec, RESULT_STATUSES


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _objects(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{field} must be a list of objects")
    return [dict(item) for item in value]


def _references(value: Any, field: str) -> list[dict[str, Any]]:
    references = _objects(value, field)
    for item in references:
        if not isinstance(item.get("role"), str) or not item["role"].strip():
            raise ValueError(f"each {field} item requires a non-empty role")
        if not isinstance(item.get("path"), str) or not item["path"].strip():
            raise ValueError(f"each {field} item requires a non-empty path")
        if "integrity" in item and not isinstance(item["integrity"], Mapping):
            raise ValueError(f"{field} integrity must be an object")
    return references


def make_task(agent: str, instruction: str, *, run_id: str | None = None,
              inputs: list[dict[str, Any]] | None = None,
              criteria: list[str] | None = None,
              constraints: list[str] | None = None,
              context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not agent or not instruction.strip():
        raise ValueError("agent and instruction are required")
    return {
        "schema_version": 1,
        "task_id": str(uuid.uuid4()),
        "agent": agent,
        "run_id": run_id,
        "created_at": _now(),
        "instruction": instruction,
        "inputs": list(inputs or []),
        "criteria": list(criteria or []),
        "constraints": list(constraints or []),
        "context": dict(context or {}),
    }


def validate_task(payload: Mapping[str, Any], spec: AgentSpec) -> dict[str, Any]:
    required = {"schema_version", "task_id", "agent", "created_at", "instruction",
                "inputs", "criteria", "constraints", "context"}
    missing = required - set(payload)
    if missing:
        raise ValueError("agent task is missing: " + ", ".join(sorted(missing)))
    allowed = required | {"run_id"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("agent task has unknown fields: " + ", ".join(sorted(unknown)))
    if payload["schema_version"] != 1 or payload["agent"] != spec.name:
        raise ValueError("agent task schema or target does not match the agent specification")
    for field in ("task_id", "created_at", "instruction"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"agent task {field} must be a non-empty string")
    _references(payload["inputs"], "inputs")
    for field in ("criteria", "constraints"):
        value = payload[field]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip()
                                              for item in value):
            raise ValueError(f"agent task {field} must contain strings")
    if not isinstance(payload["context"], Mapping):
        raise ValueError("agent task context must be an object")
    if spec.result_contract == "JudgeVote":
        lens = payload["context"].get("review_lens")
        focus = payload["context"].get("review_focus")
        if not isinstance(lens, str) or not lens.strip():
            raise ValueError("Judge AgentTask context requires review_lens")
        if not isinstance(focus, str) or not focus.strip():
            raise ValueError("Judge AgentTask context requires review_focus")
    return dict(payload)


def validate_result(payload: Mapping[str, Any], spec: AgentSpec, *, task_id: str | None = None) -> dict[str, Any]:
    required = {"schema_version", "task_id", "agent", "status", "summary", "artifacts",
                "evidence", "requested_approval", "next_actions"}
    missing = required - set(payload)
    if missing:
        raise ValueError("agent result is missing: " + ", ".join(sorted(missing)))
    unknown = set(payload) - required
    if unknown:
        raise ValueError("agent result has unknown fields: " + ", ".join(sorted(unknown)))
    if payload["schema_version"] != 1 or payload["agent"] != spec.name:
        raise ValueError("agent result schema or producer does not match the agent specification")
    if task_id is not None and payload["task_id"] != task_id:
        raise ValueError("agent result task_id does not match the dispatched task")
    if payload["status"] not in RESULT_STATUSES:
        raise ValueError(f"unknown agent result status: {payload['status']!r}")
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("agent result summary must be a non-empty string")
    _references(payload["artifacts"], "artifacts")
    _references(payload["evidence"], "evidence")
    if payload["requested_approval"] is not None and not isinstance(
            payload["requested_approval"], Mapping):
        raise ValueError("requested_approval must be null or an object")
    approval = payload["requested_approval"]
    if approval is not None:
        if set(approval) != {"boundary", "reason"}:
            raise ValueError("requested_approval requires exactly boundary and reason")
        if approval["boundary"] not in spec.approval_boundaries:
            raise ValueError("requested approval is outside the agent specification")
        if not isinstance(approval["reason"], str) or not approval["reason"].strip():
            raise ValueError("requested approval reason must be a non-empty string")
    if not isinstance(payload["next_actions"], list) or any(
            not isinstance(item, str) or not item.strip() for item in payload["next_actions"]):
        raise ValueError("next_actions must contain strings")
    return dict(payload)


def validate_judge_vote(payload: Mapping[str, Any], criteria: list[str],
                        review_lens: str | None = None) -> dict[str, Any]:
    required = {"review_lens", "verdict", "criteria_checked", "rationale", "required_fix"}
    missing = required - set(payload)
    if missing:
        raise ValueError("judge vote is missing: " + ", ".join(sorted(missing)))
    unknown = set(payload) - required
    if unknown:
        raise ValueError("judge vote has unknown fields: " + ", ".join(sorted(unknown)))
    if payload["verdict"] not in {"PASS", "REVISE", "FAIL"}:
        raise ValueError("judge vote has an invalid verdict")
    if not isinstance(payload["review_lens"], str) or not payload["review_lens"].strip():
        raise ValueError("judge vote review_lens must be a non-empty string")
    if review_lens is not None and payload["review_lens"] != review_lens:
        raise ValueError("judge vote review_lens does not match the dispatched task")
    checked = _objects(payload["criteria_checked"], "criteria_checked")
    checked_names = []
    for item in checked:
        if set(item) != {"criterion", "value_read", "ok"}:
            raise ValueError("each criteria_checked item requires criterion, value_read, and ok")
        if not isinstance(item["criterion"], str) or not item["criterion"].strip():
            raise ValueError("checked criterion must be a non-empty string")
        if not isinstance(item["ok"], bool):
            raise ValueError("checked criterion ok must be boolean")
        checked_names.append(item["criterion"])
    if checked_names != criteria:
        raise ValueError("judge vote criteria do not match the ordered task criteria")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("judge vote rationale must be a non-empty string")
    if not isinstance(payload["required_fix"], str):
        raise ValueError("judge vote required_fix must be a string")
    if payload["verdict"] == "PASS" and (not criteria or not all(item["ok"] for item in checked)):
        raise ValueError("Judge PASS requires every non-empty criterion to pass")
    if payload["verdict"] != "PASS" and not payload["required_fix"].strip():
        raise ValueError("REVISE/FAIL judge votes require a concrete fix")
    return dict(payload)


def validate_agent_response(payload: Mapping[str, Any], spec: AgentSpec,
                            task: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a response according to the contract selected by the role spec."""
    validate_task(task, spec)
    if spec.result_contract == "JudgeVote":
        return validate_judge_vote(
            payload, list(task["criteria"]), task["context"]["review_lens"]
        )
    return validate_result(payload, spec, task_id=task["task_id"])


class FileExchangeRuntime:
    """Write task packets and collect results without choosing an LLM provider."""

    def __init__(self, exchange_dir: str | Path):
        self.exchange_dir = Path(exchange_dir).resolve()
        self.outbox = self.exchange_dir / "tasks"
        self.inbox = self.exchange_dir / "results"
        self.raw = self.exchange_dir / "raw"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)

    def dispatch(self, spec: AgentSpec, task: Mapping[str, Any]) -> Path:
        task = validate_task(task, spec)
        path = self.outbox / f"{task['task_id']}.json"
        if path.exists():
            raise FileExistsError(f"task packet already exists: {path}")
        path.write_text(json.dumps(task, indent=2) + "\n")
        return path

    def collect(self, spec: AgentSpec, task_id: str) -> dict[str, Any]:
        path = self.inbox / f"{task_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"agent result is not available: {path}")
        task_path = self.outbox / f"{task_id}.json"
        if not task_path.is_file():
            raise FileNotFoundError(f"dispatched task packet is missing: {task_path}")
        return validate_agent_response(json.loads(path.read_text()), spec,
                                       json.loads(task_path.read_text()))

    def _preserve_raw(self, task_id: str, raw_text: str) -> Path:
        """Write the unedited response bytes before any parse/validation.

        Re-submissions never overwrite a prior raw file: the second and later
        responses for a task land at ``{task_id}.1.json``, ``.2.json``, ... so
        the full audit trail of what each agent actually emitted is retained.
        """
        target = self.raw / f"{task_id}.json"
        suffix = 0
        while target.exists():
            suffix += 1
            target = self.raw / f"{task_id}.{suffix}.json"
        target.write_text(raw_text)
        return target

    def accept(self, spec: AgentSpec, task_id: str, raw_text: str) -> dict[str, Any]:
        """Bind an agent's raw response to its dispatched task with audit preservation.

        Order matters: the raw response is preserved on disk FIRST, so even a
        malformed or validation-failing response is auditable. Only after a
        successful contract validation (task_id binding, result/JudgeVote schema,
        and — for Judge tasks — the run-bound review_lens) is the validated
        payload recorded under ``results/``. A response with no dispatched task is
        never accepted.
        """
        task_path = self.outbox / f"{task_id}.json"
        if not task_path.is_file():
            raise FileNotFoundError(f"dispatched task packet is missing: {task_path}")

        raw_path = self._preserve_raw(task_id, raw_text)

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"agent response is not valid JSON (raw preserved at {raw_path}): {error}"
            ) from error
        try:
            validated = validate_agent_response(
                payload, spec, json.loads(task_path.read_text()))
        except ValueError as error:
            raise ValueError(f"{error} (raw preserved at {raw_path})") from error

        result_path = self.inbox / f"{task_id}.json"
        result_path.write_text(json.dumps(validated, indent=2) + "\n")
        return validated
