"""Role-specific typed outputs and the output-model selector (Phase 2/D2).

Each role's PydanticAI ``output_type`` is a role-specific typed model, never a free-form dict:

- judge          -> JudgeVoteModel        (final result, already typed)
- data-curator   -> DataCuratorActionProposal
- ml-trainer     -> MLTrainerActionProposal
- simulation     -> SimulationActionProposal
- analyst        -> AnalystActionProposal
- orchestrator   -> OrchestratorPlan
- literature     -> LiteratureEvidence

``select_output_model`` maps a loaded AgentSpec to its model. Producer/analyst outputs are
ActionProposals (see actions.py); the executor bridge that turns an accepted proposal into a
validated artifact + AgentResult lands in Phase 4-5. Until then the driver keeps using the
canonical AgentResult path for producers; this module and its tests lock the typed contract.

A role is not always locked to exactly one typed output for every task it ever performs: a task
may declare ``context.expected_output_model`` naming a *registered reasoning-output model*
(``register_reasoning_output_model`` below) to ask that role to produce a different, genuinely
distinct typed result for that one invocation instead of its role's default proposal/plan model
-- e.g. the Analyst returning a typed ``RootCauseClassification`` (diagnosis) rather than an
``AnalystActionProposal`` (executable action) when asked to diagnose, not act. This is how
``production_router``'s ``typed_reasoning_output`` acceptance strategy stays generic: it never
special-cases a role or model name itself, it only asks this selector which model a given
(spec, task) pair resolves to. An unregistered name fails closed rather than silently falling
back to the role default, since that would mean silently accepting a different-shaped result
than the task actually asked for.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .actions import (
    AnalystActionProposal,
    DataCuratorActionProposal,
    MLTrainerActionProposal,
    SimulationActionProposal,
)
from .models import AgentResultModel, JudgeVoteModel, NonEmptyStr, RequestedApproval

# --- Literature typed output -----------------------------------------------------

class SourceRecord(BaseModel):
    """A normalized literature source. Fabricated sources are never allowed; when a source
    cannot be retrieved the Literature agent returns status BLOCKED / SOURCE_NOT_RETRIEVED."""
    model_config = {"extra": "forbid"}
    title: NonEmptyStr
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    identifier: str = ""            # DOI or other stable identifier
    source_type: NonEmptyStr        # journal | arxiv | official_doc | user_pdf | repo_manifest | url
    url_or_reference: str = ""
    retrieved_at: str = ""
    locator: str = ""               # page/section/table/figure
    supported_claim: str = ""
    value: Optional[float] = None
    unit: str = ""
    material: str = ""
    phase: str = ""
    method: str = ""
    temperature: str = ""
    pressure: str = ""
    evidence_limitation: str = ""
    access_status: Literal["retrieved", "blocked", "unknown", "source_not_retrieved"] = "retrieved"


class LiteratureEvidence(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["completed", "blocked", "unknown", "source_not_retrieved"]
    sources: list[SourceRecord] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    summary: NonEmptyStr


# --- Orchestrator typed output ---------------------------------------------------

class AgentTaskProposal(BaseModel):
    """A task the Orchestrator proposes to dispatch to a role (not the controller mutation
    itself — that goes through the typed controller bridge in Phase 4-5)."""
    model_config = {"extra": "forbid"}
    agent: NonEmptyStr
    instruction: NonEmptyStr
    rationale: NonEmptyStr
    criteria: list[NonEmptyStr] = Field(default_factory=list)


class OrchestratorPlan(BaseModel):
    model_config = {"extra": "forbid"}
    run_id: NonEmptyStr
    current_stage: NonEmptyStr
    rationale: NonEmptyStr
    proposed_tasks: list[AgentTaskProposal] = Field(default_factory=list)
    proposed_stage_action: Optional[str] = None   # controller action the bridge would validate
    approval_requests: list[RequestedApproval] = Field(default_factory=list)
    summary: NonEmptyStr


# --- Selector --------------------------------------------------------------------

_ROLE_OUTPUT_MODELS = {
    "judge": JudgeVoteModel,
    "data-curator": DataCuratorActionProposal,
    "ml-trainer": MLTrainerActionProposal,
    "simulation": SimulationActionProposal,
    "analyst": AnalystActionProposal,
    "orchestrator": OrchestratorPlan,
    "literature": LiteratureEvidence,
}


# --- Typed reasoning-output registry ----------------------------------------------
#
# Advisory, evidence-bound scientific reasoning results (RootCauseClassification,
# RecoveryPlanProposal, ...) that a role may be asked to produce for ONE task instead of its
# usual proposal/plan output. Registered here by name -- not by role -- so any current or future
# role can be asked, per-task, for any registered reasoning output; this module stays the single
# place that knows the mapping, and production_router never hardcodes a model name.
_REASONING_OUTPUT_MODELS: dict[str, type[BaseModel]] = {}


def register_reasoning_output_model(name: str, model: type[BaseModel]) -> None:
    _REASONING_OUTPUT_MODELS[name] = model


def is_reasoning_output_model(model: type) -> bool:
    return model in _REASONING_OUTPUT_MODELS.values()


def select_output_model(spec, task: Optional[dict] = None):
    """Return the typed output model for one (spec, task) invocation.

    If ``task.context.expected_output_model`` names a registered reasoning-output model, that
    model wins for this invocation regardless of role -- an unregistered name fails closed
    (raises) rather than silently falling back to the role's default output. Otherwise falls back
    to the generic AgentResultModel for any spec whose name is not one of the known roles (keeps
    backward compatibility).
    """
    if task is not None:
        hint = (task.get("context") or {}).get("expected_output_model")
        if hint is not None:
            try:
                return _REASONING_OUTPUT_MODELS[hint]
            except KeyError:
                raise ValueError(f"unregistered reasoning output model: {hint!r}") from None
    name = getattr(spec, "name", None)
    return _ROLE_OUTPUT_MODELS.get(name, AgentResultModel)


# Registered late (after the registry function exists) to avoid import-order issues; neither
# root_cause nor recovery_bridge depends on this module, so importing them here cannot create a
# cycle.
from .root_cause import RootCauseClassification  # noqa: E402
from .recovery_bridge import RecoveryPlanProposal  # noqa: E402
from .teacher_validation_plan import TeacherValidationPlanProposal  # noqa: E402

register_reasoning_output_model("RootCauseClassification", RootCauseClassification)
register_reasoning_output_model("RecoveryPlanProposal", RecoveryPlanProposal)
register_reasoning_output_model("TeacherValidationPlanProposal", TeacherValidationPlanProposal)
