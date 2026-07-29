"""Load and validate declarative agent specifications.

The scientific workflow does not depend on a particular LLM provider.  A
runtime reads one of the YAML specifications in ``agent_specs/``, loads the
referenced Markdown prompt, and exchanges JSON-compatible task/result packets.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ROLE_TYPES = {"coordinator", "producer", "reviewer"}
RESULT_STATUSES = {"completed", "needs_input", "blocked", "failed"}
RESULT_CONTRACTS = {"AgentResult", "JudgeVote"}


def _nonempty_strings(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "non-empty " if not allow_empty else ""
        raise ValueError(f"{field} must be a {qualifier}list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


@dataclass(frozen=True)
class AgentSpec:
    schema_version: int
    name: str
    role_type: str
    description: str
    prompt_path: Path
    task_contract: str
    result_contract: str
    capabilities: tuple[str, ...]
    accepts: tuple[str, ...]
    returns: tuple[str, ...]
    approval_boundaries: tuple[str, ...]
    delegates: tuple[str, ...]

    @property
    def prompt(self) -> str:
        return self.prompt_path.read_text()


def load_agent_spec(path: str | Path, *, root: str | Path | None = None) -> AgentSpec:
    """Load one agent specification and resolve its canonical prompt."""
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"agent specification must be a mapping: {path}")
    required = {"schema_version", "name", "role_type", "description", "prompt",
                "task_contract", "result_contract",
                "capabilities", "accepts", "returns", "approval_boundaries", "delegates"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"agent specification is missing: {', '.join(sorted(missing))}")
    unknown = set(payload) - required
    if unknown:
        raise ValueError(f"agent specification has unknown fields: {', '.join(sorted(unknown))}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported agent specification schema_version")
    for field in ("name", "description", "prompt"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if payload["role_type"] not in ROLE_TYPES:
        raise ValueError(f"unknown role_type: {payload['role_type']!r}")
    if payload["task_contract"] != "AgentTask":
        raise ValueError("unsupported task_contract")
    if payload["result_contract"] not in RESULT_CONTRACTS:
        raise ValueError("unsupported result_contract")
    project_root = Path(root).resolve() if root else path.parent.parent
    prompt_path = (project_root / payload["prompt"]).resolve()
    try:
        prompt_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("agent prompt must stay inside the project root") from exc
    if not prompt_path.is_file():
        raise FileNotFoundError(f"agent prompt is missing: {prompt_path}")
    if prompt_path.read_text().lstrip().startswith("---"):
        raise ValueError("canonical prompts must not contain runtime-specific front matter")
    return AgentSpec(
        schema_version=1,
        name=payload["name"],
        role_type=payload["role_type"],
        description=payload["description"],
        prompt_path=prompt_path,
        task_contract=payload["task_contract"],
        result_contract=payload["result_contract"],
        capabilities=_nonempty_strings(payload["capabilities"], "capabilities", allow_empty=False),
        accepts=_nonempty_strings(payload["accepts"], "accepts", allow_empty=False),
        returns=_nonempty_strings(payload["returns"], "returns", allow_empty=False),
        approval_boundaries=_nonempty_strings(payload["approval_boundaries"],
                                               "approval_boundaries"),
        delegates=_nonempty_strings(payload["delegates"], "delegates"),
    )


def load_agent_specs(directory: str | Path, *, root: str | Path | None = None) -> dict[str, AgentSpec]:
    """Load all YAML specifications and reject duplicate or dangling delegates."""
    directory = Path(directory)
    specs = [load_agent_spec(path, root=root) for path in sorted(directory.glob("*.yaml"))]
    if not specs:
        raise ValueError(f"no agent specifications found in {directory}")
    result = {spec.name: spec for spec in specs}
    if len(result) != len(specs):
        raise ValueError("agent specification names must be unique")
    for spec in specs:
        unknown = set(spec.delegates) - set(result)
        if unknown:
            raise ValueError(f"agent {spec.name!r} has unknown delegates: {sorted(unknown)}")
        if spec.name in spec.delegates:
            raise ValueError(f"agent {spec.name!r} cannot delegate to itself")
    return result
