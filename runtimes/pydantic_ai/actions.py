"""Typed action catalog and role-scoped ActionProposal models (Phase 2/D2 + Phase 4 seed).

A producer/analyst agent never returns free-form ``dict`` output and never returns an
executable command string. It returns a role-scoped ``ActionProposal`` whose ``action_type`` is
constrained to that role's allowed, backed action set. Wrong-role action types fail validation
(Literal mismatch), and actions that have no validated backend are registered here with an
explicit non-available status so the agent cannot propose them as if usable.

The mapping ``action_type -> deterministic executor`` and the approval/dry-run/idempotency
pipeline are added in Phase 4-5; this module defines the typed contract they consume.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .models import EvidenceReference, NonEmptyStr

# --- Capability status registry --------------------------------------------------

CapabilityStatus = Literal[
    "AVAILABLE",             # backed by a validated deterministic executor in this repo
    "NOT_AVAILABLE",         # no validated backend; not in current SiO2 authoritative scope
    "NOT_IMPLEMENTED",       # planned interface, executor not written yet
    "APPROVAL_REQUIRED",     # backed + reachable, but gated on explicit human approval
    "OUT_OF_CURRENT_SCOPE",  # deliberately excluded from the current Teacher-baseline scope
]


class CapabilityEntry(BaseModel):
    """One capability's status, with the reason and what it would take to enable it."""
    model_config = {"extra": "forbid"}
    action_type: NonEmptyStr
    role: NonEmptyStr
    status: CapabilityStatus
    reason: NonEmptyStr
    required_backend: str = ""
    future_conditions: str = ""


# Actions the design names but that have NO validated backend / are out of current scope.
# The agent must not propose or execute these; they are surfaced only so the registry is honest.
CAPABILITY_REGISTRY: dict[str, CapabilityEntry] = {
    e.action_type: e
    for e in [
        CapabilityEntry(action_type="compute_eos", role="simulation", status="NOT_AVAILABLE",
                        reason="no EOS/bulk-modulus executor in validation/ or adapters/",
                        required_backend="EOS fit over compressed/expanded cells",
                        future_conditions="implement + unit-test an EOS executor, add to registry"),
        CapabilityEntry(action_type="compute_mechanics", role="simulation", status="NOT_AVAILABLE",
                        reason="no elastic-constant/mechanics executor present",
                        required_backend="elastic tensor via strain-stress or energy-strain"),
        CapabilityEntry(action_type="compute_ring_statistics", role="simulation", status="NOT_AVAILABLE",
                        reason="no ring-statistics executor present"),
        CapabilityEntry(action_type="compute_sq_fsdp", role="simulation", status="NOT_AVAILABLE",
                        reason="S(Q)/FSDP explicitly stubbed NOT_EVALUATED in structure_dynamics.py",
                        required_backend="structure factor from RDF or direct S(Q)"),
        CapabilityEntry(action_type="compute_adf", role="simulation", status="NOT_AVAILABLE",
                        reason="ADF stubbed NOT_EVALUATED in structure_dynamics.py; excluded from "
                               "the current required Teacher baseline",
                        required_backend="validated ADF implementation",
                        future_conditions="connect a verified ADF implementation, then set AVAILABLE"),
        CapabilityEntry(action_type="compute_channel_d", role="analyst", status="NOT_AVAILABLE",
                        reason="student-MD-vs-DFT (channel d) has no executor; requires carved DFT",
                        required_backend="DFT single points on student-MD frames"),
        CapabilityEntry(action_type="fine_tune_teacher", role="ml-trainer", status="OUT_OF_CURRENT_SCOPE",
                        reason="Teacher fine-tuning / ER is not part of the current authoritative "
                               "distillation scope",
                        required_backend="ER/replay-anchored fine-tune loop",
                        future_conditions="separate approved effort; keeps distillation/fine-tune "
                                           "boundary explicit"),
        CapabilityEntry(action_type="generate_dft_inputs", role="simulation", status="APPROVAL_REQUIRED",
                        reason="only an INCAR renderer exists; arbitrary DFT-input generation is not "
                               "done in this phase; a specific approved reference protocol may be "
                               "connected later via the typed reference-calculation proposal",
                        required_backend="approved reference_dft protocol + POTCAR/KPOINTS policy"),
        CapabilityEntry(action_type="run_dft", role="simulation", status="APPROVAL_REQUIRED",
                        reason="actual DFT/VASP execution is forbidden before explicit approval",
                        required_backend="scheduler + VASP runner + reference protocol"),
        CapabilityEntry(action_type="generate_scheduler_script", role="simulation", status="NOT_AVAILABLE",
                        reason="arbitrary scheduler-script generation is not permitted; use the typed "
                               "scheduler submit/query/collect bridge instead"),
    ]
}


def capability_status(action_type: str) -> Optional[CapabilityEntry]:
    """Return the non-available capability entry for an action_type, or None if not registered."""
    return CAPABILITY_REGISTRY.get(action_type)


# --- Allowed (in-scope) action types per role -----------------------------------
# These are the action types with a validated deterministic backend (or, for scheduler and
# costly producer actions, a typed+approval-gated bridge) required by the current SiO2 workflow.

DATA_CURATOR_ACTIONS = (
    "inspect_dataset", "summarize_source_categories", "sample_seed_pool",
    "reconstruct_lineage", "generate_group_split", "acquire_structures", "label_with_teacher",
    "validate_label_preservation", "build_dataset_manifest",
    "compare_deployment_coverage", "detect_duplicates", "detect_atomic_overlap",
    "build_data_coverage_report",
)
ML_TRAINER_ACTIONS = (
    "prepare_student_inputs", "train_committee", "collect_checkpoints",
    "evaluate_heldout_fidelity", "summarize_seed_variation",
    "compute_committee_disagreement", "validate_training_completion",
    "build_uncertainty_report",
)
SIMULATION_ACTIONS = (
    "build_teacher_baseline", "validate_teacher_reference", "run_teacher_md", "run_student_md", "compute_rdf", "compute_coordination",
    "compute_minimum_distance", "detect_force_spike", "compute_nve_drift",
    "validate_simulation_completion",
    "submit_scheduler_job", "query_scheduler_job", "collect_scheduler_artifact",
    "build_physical_validation_report",
)
ANALYST_ACTIONS = (
    "compare_force_errors", "compare_energy_errors", "summarize_committee_disagreement",
    "compare_rdf", "compare_coordination", "fit_nve_drift", "summarize_md_stability",
    "classify_root_cause", "generate_run_summary",
)

# Actions that always require explicit human approval before execution (costly/side-effecting).
APPROVAL_GATED_ACTIONS = {
    "build_teacher_baseline": "costly_teacher_labeling",
    "validate_teacher_reference": "costly_teacher_labeling",
    "acquire_structures": "costly_teacher_labeling",
    "label_with_teacher": "costly_teacher_labeling",
    "train_committee": "costly_training",
    "evaluate_heldout_fidelity": "costly_training",
    "run_teacher_md": "production_md",
    "run_student_md": "production_md",
    "submit_scheduler_job": "scheduler_submission",
}


# --- ActionProposal envelope (Phase F common fields) ----------------------------

class ActionProposalBase(BaseModel):
    """Common, strongly-typed proposal envelope. The LLM selects a typed action + parameters;
    it never returns an executable string. Per-action parameter schemas are enforced by the
    executor registry's per-action validator in Phase 4."""
    model_config = {"extra": "forbid"}
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    stage: NonEmptyStr
    requested_at: NonEmptyStr
    rationale: NonEmptyStr
    active_config_refs: list[NonEmptyStr] = Field(default_factory=list)
    advisory_claimed_config_hashes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional model-claimed config digests for prose audit only; these are never "
            "authoritative integrity assertions. Execution-critical hashes must be computed "
            "deterministically by the Controller/runtime and bound in parameters/manifests."
        ),
    )
    input_artifacts: list[EvidenceReference] = Field(default_factory=list)
    input_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outputs: list[NonEmptyStr] = Field(default_factory=list)
    estimated_cost: Optional[str] = None
    estimated_runtime: Optional[str] = None
    approval_boundary: Optional[str] = None
    idempotency_key: NonEmptyStr
    dry_run: bool = True
    required_validator: Optional[str] = None
    rollback_or_cleanup_policy: Optional[str] = None


class DataCuratorActionProposal(ActionProposalBase):
    requested_by_role: Literal["data-curator"] = "data-curator"
    action_type: Literal[DATA_CURATOR_ACTIONS]  # type: ignore[valid-type]


class MLTrainerActionProposal(ActionProposalBase):
    requested_by_role: Literal["ml-trainer"] = "ml-trainer"
    action_type: Literal[ML_TRAINER_ACTIONS]  # type: ignore[valid-type]


class SimulationActionProposal(ActionProposalBase):
    requested_by_role: Literal["simulation"] = "simulation"
    action_type: Literal[SIMULATION_ACTIONS]  # type: ignore[valid-type]


class AnalystActionProposal(ActionProposalBase):
    requested_by_role: Literal["analyst"] = "analyst"
    action_type: Literal[ANALYST_ACTIONS]  # type: ignore[valid-type]


ROLE_ACTION_MODELS = {
    "data-curator": DataCuratorActionProposal,
    "ml-trainer": MLTrainerActionProposal,
    "simulation": SimulationActionProposal,
    "analyst": AnalystActionProposal,
}

ROLE_ALLOWED_ACTIONS = {
    "data-curator": set(DATA_CURATOR_ACTIONS),
    "ml-trainer": set(ML_TRAINER_ACTIONS),
    "simulation": set(SIMULATION_ACTIONS),
    "analyst": set(ANALYST_ACTIONS),
}
