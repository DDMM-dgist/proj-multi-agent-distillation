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
    "reconstruct_lineage", "generate_group_split", "build_split_membership_population",
    "acquire_structures", "label_with_teacher",
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
    "resolve_deployment_checkpoint", "build_deployment_context",
    "submit_scheduler_job", "query_scheduler_job", "collect_scheduler_artifact",
    "build_physical_validation_report",
)
ANALYST_ACTIONS = (
    "compare_force_errors", "compare_energy_errors", "summarize_committee_disagreement",
    "compare_rdf", "compare_coordination", "fit_nve_drift", "summarize_md_stability",
    "classify_root_cause", "generate_run_summary",
)

# Actions that MAY require explicit human approval before execution (costly/side-effecting). This
# is a per-ACTION-TYPE DEFAULT boundary only. It is never, by itself, sufficient to decide whether a
# given proposal requires approval: dispatch.py always resolves the ACTUAL boundary through
# ``resolve_action_approval_boundary`` below, which relaxes the default when the action's own typed
# capabilities/effects prove the costly effect the boundary guards is not actually incurred.
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


# --- Typed costly-compute effect taxonomy ----------------------------------------
# The human-approval boundary an action requires is derived from the materially costly, non-trivially
# reversible EFFECTS the action actually performs for a given proposal -- never from its action name
# or the stage it happens to occupy. Each distinct costly effect maps to exactly one boundary. A
# geometry-only acquisition action, for example, performs NONE of these effects (it only generates
# candidate structures) and therefore must not inherit the Teacher-labeling boundary merely because
# it precedes Teacher labeling in the pipeline.
COSTLY_EFFECT_BOUNDARY = {
    "teacher_inference": "costly_teacher_labeling",   # fresh Teacher forward passes / new DFT labels
    "student_training": "costly_training",            # Student committee training / model search
    "production_md": "production_md",                  # production MD / long dynamics
    "scheduler_submission": "scheduler_submission",   # external HPC job submission
}


# Actions whose guarded costly effect is INHERENT -- performing it is the action's defining purpose,
# so the effect is ALWAYS incurred and the boundary can never be relaxed by any declared parameter.
_INHERENT_COSTLY_ACTIONS = frozenset({
    "label_with_teacher",         # always runs fresh Teacher inference to create new training labels
    "train_committee",            # always runs Student committee training
    "evaluate_heldout_fidelity",  # always runs the trained committee
    "run_teacher_md", "run_student_md",  # always run production dynamics
    "submit_scheduler_job",       # always submits an external scheduler job
})


# Teacher-evidence actions that run REAL Teacher forward passes on GPU to produce a report /
# comparison (a teacher-baseline operational-stability report; a Teacher-vs-DFT reference
# comparison). Running the Teacher over a population IS the materially costly, GPU-bound effect the
# ``costly_teacher_labeling`` boundary guards -- independently of whether the run also grows the
# training corpus. ``build_teacher_baseline`` has no reuse path, so it ALWAYS incurs it;
# ``validate_teacher_reference`` incurs it UNLESS the proposal binds a prior verified
# ``historical_report`` (the executor's verified-reuse path -- ``executors.
# _exec_validate_teacher_reference`` discriminates on exactly that key -- which recomputes metrics
# from already-materialized Teacher predictions and runs NO fresh Teacher inference).
#
# (Supersedes the R25 ``_CONDITIONALLY_GATED_VALIDATION_ACTIONS`` relaxation, which wrongly treated
# an affirmative "creates no new DFT/protected-reference labels" declaration -- a statement about
# CORPUS GROWTH -- as proof that no costly Teacher COMPUTE is incurred, and so let a fresh
# 9,295-frame Teacher baseline dispatch on GPU with ``action_approvals={}``. Corpus-growth
# provenance can never relax this compute boundary.)
_TEACHER_EVIDENCE_INFERENCE_ACTIONS = frozenset({
    "build_teacher_baseline", "validate_teacher_reference",
})


def _incurs_teacher_inference_effect(action_type: str, parameters: dict) -> bool:
    """Whether THIS proposal performs the costly effect that ``costly_teacher_labeling`` guards --
    materially costly, GPU-bound Teacher forward passes (fresh Teacher inference).

    Fail-closed: returns ``True`` (effect assumed incurred -> keep the gate) unless the proposal
    AFFIRMATIVELY proves, through a typed effect declaration appropriate to the action, that it
    performs NO fresh Teacher forward passes. Only an explicit signal counts; a missing field is
    never read as proof of absence.

    The relaxation signals are effect-based (does the action run the Teacher?), never name-based and
    never provenance-based (whether it grows the training corpus is a DIFFERENT concern that must
    not relax this compute boundary):

    * an action whose Teacher inference is its defining purpose (``_INHERENT_COSTLY_ACTIONS``, e.g.
      ``label_with_teacher``) can never prove otherwise;
    * ``build_teacher_baseline`` always runs the Teacher over the operational population to build
      its report -- it has no reuse path, so it always incurs the effect;
    * ``validate_teacher_reference`` runs fresh Teacher inference
      (``adapters.acquisition.label_with_teacher`` over the reference population) UNLESS the
      proposal binds a prior verified ``historical_report``, in which case the executor's
      verified-reuse path recomputes metrics from existing predictions and runs NO Teacher;
    * ``performs_teacher_inference`` is the acquisition-family signal: the framework
      (``cli._bind_acquisition_plan_for_stage``) deterministically classifies the ACTUAL bound
      acquisition recipe and injects this flag, OVERRIDING any self-asserted value -- a recipe the
      framework proves performs no Teacher inference yields ``False`` (cheap, reversible geometry-only
      structure generation), while a Teacher-driven recipe yields ``True``. Note the built-in
      ``augment-atoms`` and ``teacher-md`` recipes BOTH drive the Teacher ASE calculator during
      structure generation (augment-atoms via the Teacher-bound native config the executor writes),
      so both yield ``True``; an arbitrary adapter / unknown / unreadable recipe also fails closed to
      ``True``.
    """
    if action_type in _INHERENT_COSTLY_ACTIONS:
        return True
    if action_type == "build_teacher_baseline":
        return True
    if action_type == "validate_teacher_reference":
        # Verified-reuse (a bound historical_report) runs NO fresh Teacher; a fresh reference
        # validation runs label_with_teacher over the reference population = costly Teacher compute.
        return not bool((parameters or {}).get("historical_report"))
    performs = (parameters or {}).get("performs_teacher_inference")
    if performs is True:
        return True
    if performs is False:
        return False
    return True


def resolve_action_approval_boundary(action_type: str, default_boundary: Optional[str],
                                     parameters: Optional[dict] = None) -> Optional[str]:
    """The approval boundary an action ACTUALLY requires for THIS proposal, derived from its typed
    capabilities/effects -- not from its action name or its position in the pipeline.

    ``default_boundary`` is the per-action-type default the caller resolved (normally
    ``APPROVAL_GATED_ACTIONS.get(action_type)``). For the ``costly_teacher_labeling`` boundary, the
    default is RELAXED to ``None`` iff the proposal affirmatively proves it performs no fresh
    Teacher forward passes (see ``_incurs_teacher_inference_effect``); otherwise, and for every
    other boundary, the default is returned unchanged. Fail-closed throughout: any missing /
    non-boolean / affirmatively-costly declaration keeps the gate, and no boundary other than
    ``costly_teacher_labeling`` is ever relaxed here. Note that whether the action grows the
    training corpus (its DFT/protected-reference label provenance) is a SEPARATE concern that never
    relaxes this compute boundary -- running the Teacher on GPU is costly regardless.
    """
    if default_boundary is None:
        return None
    if default_boundary == "costly_teacher_labeling":
        if not _incurs_teacher_inference_effect(action_type, parameters or {}):
            return None
    return default_boundary


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
