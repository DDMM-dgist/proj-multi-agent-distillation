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


def select_output_model(spec):
    """Return the typed output model for a role. Falls back to the generic AgentResultModel
    for any spec whose name is not one of the seven known roles (keeps backward compatibility)."""
    name = getattr(spec, "name", None)
    return _ROLE_OUTPUT_MODELS.get(name, AgentResultModel)
