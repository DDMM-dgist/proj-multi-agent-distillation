"""Machine-readable per-role tool/capability manifests (Phase 3/E).

The read-only TOOL surface is uniform (all roles get the four bounded read tools in
``tool_registry.EXPOSED_READ_TOOLS`` — nothing can write, glob, list directories, or run a
shell). What differs per role is encoded here: which read roots it may touch, which typed
ACTIONS it may propose (never execute directly), which approvals gate those actions, its cost
and side-effect class, and the tools/capabilities it is explicitly DENIED.

These manifests are data (serializable to JSON/YAML) and are cross-checked against agent_specs
and the capability registry by tests so they cannot drift.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .actions import (
    ANALYST_ACTIONS,
    APPROVAL_GATED_ACTIONS,
    CAPABILITY_REGISTRY,
    DATA_CURATOR_ACTIONS,
    ML_TRAINER_ACTIONS,
    ROLE_ALLOWED_ACTIONS,
    SIMULATION_ACTIONS,
)
from .tool_registry import EXPOSED_READ_TOOLS

# Capabilities this runtime NEVER grants to any agent (the whole point vs. unrestricted Bash).
UNIVERSALLY_DENIED = (
    "Bash", "shell", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch",
    "arbitrary_python", "directory_listing", "glob", "scheduler_script",
    "controller_state_write", "read_other_judge_votes", "read_aggregate_verdict",
)

# The typed controller bridge the Orchestrator may call (implemented in Phase 4-5). Listed so
# the manifest is complete; each is validated + re-checked by the controller, never a raw write.
ORCHESTRATOR_BRIDGE_ACTIONS = (
    "get_controller_status", "get_stage_status", "get_gate_context", "dispatch_agent_task",
    "collect_agent_result", "propose_stage_action", "propose_gate_record",
    "request_human_approval", "read_human_decision", "propose_recovery",
    "verify_recovery_status", "get_active_configs", "get_artifact_inventory",
)

# The typed literature source backends the Literature role may use (Phase 3/E4).
LITERATURE_SOURCE_BACKENDS = (
    "doi_crossref", "arxiv", "official_document", "user_pdf_bundle", "repo_source_manifest",
    "stable_url",
)


class RoleToolManifest(BaseModel):
    model_config = {"extra": "forbid"}
    role: str
    role_type: Literal["coordinator", "producer", "reviewer"]
    allowed_read_tools: list[str] = Field(default_factory=lambda: list(EXPOSED_READ_TOOLS))
    denied_tools: list[str] = Field(default_factory=lambda: list(UNIVERSALLY_DENIED))
    proposable_actions: list[str] = Field(default_factory=list)
    bridge_actions: list[str] = Field(default_factory=list)
    source_backends: list[str] = Field(default_factory=list)
    approval_required: list[str] = Field(default_factory=list)
    read_roots: list[str] = Field(default_factory=lambda: ["run_dir", "agent_specs", "configs"])
    write_roots: list[str] = Field(default_factory=list)  # tools never write; executors do
    cost_class: Literal["read_only", "light", "hpc"] = "read_only"
    side_effect_class: Literal["none", "producer", "state_mutating", "scheduler"] = "none"
    dry_run_supported: bool = True
    idempotency_policy: str = "idempotency_key required for every proposed action"


def _approvals_for(actions) -> list[str]:
    return sorted({APPROVAL_GATED_ACTIONS[a] for a in actions if a in APPROVAL_GATED_ACTIONS})


ROLE_TOOL_MANIFESTS: dict[str, RoleToolManifest] = {
    "judge": RoleToolManifest(
        role="judge", role_type="reviewer", cost_class="read_only", side_effect_class="none",
        proposable_actions=[], write_roots=[]),
    "orchestrator": RoleToolManifest(
        role="orchestrator", role_type="coordinator", cost_class="read_only",
        side_effect_class="state_mutating", bridge_actions=list(ORCHESTRATOR_BRIDGE_ACTIONS),
        approval_required=["costly_training", "production_md", "reference_calculation",
                           "destructive_or_public_action"]),
    "literature": RoleToolManifest(
        role="literature", role_type="producer", cost_class="read_only", side_effect_class="none",
        source_backends=list(LITERATURE_SOURCE_BACKENDS)),
    "data-curator": RoleToolManifest(
        role="data-curator", role_type="producer", cost_class="hpc", side_effect_class="producer",
        proposable_actions=list(DATA_CURATOR_ACTIONS),
        approval_required=_approvals_for(DATA_CURATOR_ACTIONS)),
    "ml-trainer": RoleToolManifest(
        role="ml-trainer", role_type="producer", cost_class="hpc", side_effect_class="producer",
        proposable_actions=list(ML_TRAINER_ACTIONS),
        approval_required=sorted(set(_approvals_for(ML_TRAINER_ACTIONS)) | {"teacher_fine_tuning"})),
    "simulation": RoleToolManifest(
        role="simulation", role_type="producer", cost_class="hpc", side_effect_class="scheduler",
        proposable_actions=list(SIMULATION_ACTIONS),
        approval_required=sorted(set(_approvals_for(SIMULATION_ACTIONS)) |
                                 {"production_md", "reference_calculation", "scheduler_submission"})),
    "analyst": RoleToolManifest(
        role="analyst", role_type="producer", cost_class="light", side_effect_class="none",
        proposable_actions=list(ANALYST_ACTIONS)),
}


def manifest_for(role: str) -> RoleToolManifest:
    if role not in ROLE_TOOL_MANIFESTS:
        raise KeyError(f"no tool manifest for role '{role}'")
    return ROLE_TOOL_MANIFESTS[role]


def all_manifests_json() -> dict:
    """Serialize every role manifest (machine-readable export)."""
    return {role: m.model_dump() for role, m in ROLE_TOOL_MANIFESTS.items()}


# Sanity self-check at import: no manifest may reference an unavailable action.
_unavailable = set(CAPABILITY_REGISTRY)
for _m in ROLE_TOOL_MANIFESTS.values():
    _bad = set(_m.proposable_actions) & _unavailable
    assert not _bad, f"{_m.role} proposes unavailable actions: {_bad}"
