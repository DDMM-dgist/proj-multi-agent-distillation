"""Automatic pre-campaign autonomous acquisition planning (mirrors
``cli._commit_teacher_validation_plan_via_reasoning_roles``).

Given only the run's own frozen inputs, the deterministic ``framework_v2.acquisition`` pipeline
inventories sources/backends, models the target regimes, resolves regions under the tiered-trust
metadata policy, and analyzes coverage gaps -- then autonomously selects an acquisition strategy.
The ONE genuinely-scientific choice left (the low-level recipe: which parents, how many per
structure, and the generation params) is delegated to a real dispatched producer as a typed
``AcquisitionPlanProposal``, contextually validated, assembled into the full content-addressed
``AcquisitionPlanV2`` evidence chain, and gated by the deterministic
``validate_acquisition_plan_v2`` before being bound to the run as an input through the audited
``propose_acquisition_plan`` orchestrator bridge. A bounded semantic-correction retry feeds the
exact deterministic rejection reason back for up to two corrective attempts, exactly as the
Teacher-validation planner does.

Run-specific evidence gathering (probing the actual source pool / backends / Teacher, and realizing
the generation+selection chain from an accepted recipe) lives behind an injected
``AcquisitionPlanningProvider`` seam -- keeping THIS module material-agnostic and fully testable
with fakes. A run that registers no provider is a hard no-op (returns None immediately): the
autonomous path is strictly opt-in and never perturbs an existing run that supplies its own
acquisition plan as a human input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from framework_v2.acquisition.contracts import (
    AcquisitionPlanV2,
    AcquisitionStrategy,
    CampaignObjective,
    CandidateGenerationResult,
    CandidateSelectionResult,
    CanonicalLabelingRequest,
    CoverageGapAnalysis,
    RegionResolution,
    SourceAndCapabilityInventory,
    TargetRegimeModel,
)
from framework_v2.acquisition.validators import validate_acquisition_plan_v2

from .acquisition_plan import (
    AcquisitionPlanProposal,
    validate_acquisition_plan_proposal,
)


@dataclass(frozen=True)
class AcquisitionPlanningContext:
    """The deterministic evidence chain + admissible decision space a provider derives from a run's
    own frozen inputs. Everything here is material-derived FACT, never a scientific recipe choice."""
    objective: CampaignObjective
    inventory: SourceAndCapabilityInventory
    target_regime_model: TargetRegimeModel
    region_resolution: RegionResolution
    coverage: CoverageGapAnalysis
    strategy: AcquisitionStrategy
    admissible_parent_ids: tuple[str, ...]
    teacher_identity_sha256: str
    required_param_keys: tuple[str, ...] = ()
    param_bounds: Optional[dict[str, tuple[float, float]]] = None


@dataclass(frozen=True)
class RealizedAcquisition:
    """The full assembled + bound-ready plan the provider realizes from an accepted recipe."""
    generation_result: CandidateGenerationResult
    selection_result: CandidateSelectionResult
    labeling_request: CanonicalLabelingRequest
    plan: AcquisitionPlanV2
    legacy_projection: Optional[dict[str, Any]] = None
    existing_pool_projection: Optional[dict[str, Any]] = None


class AcquisitionPlanningProvider(Protocol):
    """The run-specific seam. ``applies`` gates whether autonomous acquisition planning runs at all
    for this controller; ``build_context`` derives the deterministic evidence chain + decision
    space; ``realize`` turns a validated recipe proposal into the fully-assembled
    ``AcquisitionPlanV2`` evidence chain (running generation/selection deterministically)."""

    def applies(self, controller) -> bool: ...

    def build_context(self, controller) -> AcquisitionPlanningContext: ...

    def realize(
        self, controller, context: AcquisitionPlanningContext,
        proposal: AcquisitionPlanProposal,
    ) -> RealizedAcquisition: ...


_PROVIDER: Optional[AcquisitionPlanningProvider] = None


def set_acquisition_planning_provider(provider: Optional[AcquisitionPlanningProvider]) -> None:
    """Register (or clear, with None) the run-specific planning provider. A future run wires its
    material-specific provider here at config/startup time; tests inject a fake. Absent a provider,
    ``plan_acquisition_via_reasoning_roles`` is a hard no-op."""
    global _PROVIDER
    _PROVIDER = provider


def get_acquisition_planning_provider() -> Optional[AcquisitionPlanningProvider]:
    return _PROVIDER


def _register_proposal_model_once() -> None:
    """Register ``AcquisitionPlanProposal`` as a reasoning-output model at runtime (never by editing
    the frozen ``role_outputs.py``), so a dispatched producer can be asked to emit it per-task."""
    from .role_outputs import register_reasoning_output_model
    register_reasoning_output_model("AcquisitionPlanProposal", AcquisitionPlanProposal)


@dataclass(frozen=True)
class ProducerRealizeResult:
    """Outcome of running the reused acquisition producer core: exactly one of ``realized`` (the
    validated, deterministically-gated plan) or ``failure`` (a terminal/pausing CampaignRunResult)
    is set. ``ctx`` is echoed back so a caller can bind provenance off the same evidence chain."""
    ctx: Optional[AcquisitionPlanningContext] = None
    realized: Optional[RealizedAcquisition] = None
    failure: Any = None


def run_acquisition_producer(
    controller, *, runtime, agent_specs_dir, exchange_dir, repo_root,
    mock_producer_response=None, emitter=None, provider=None,
    producer_role="data-curator", task_id_suffix="acquisition-plan",
    action_label="acquisition_plan_proposal",
) -> ProducerRealizeResult:
    """The material-agnostic producer CORE shared by Stage-3 acquisition planning and post-split
    TRAIN augmentation planning: build the provider's deterministic evidence chain + admissible
    decision space, dispatch a real producer to choose the low-level recipe, and gate its proposal
    through the SAME ``validate_acquisition_plan_proposal`` + ``validate_acquisition_plan_v2``
    (with a bounded semantic-correction retry). It performs NO binding -- the caller decides how to
    persist/bind whatever plan is realized -- so augmentation can reuse the identical reasoning
    producer, schemas, validators, provenance classes and stopping logic while binding its result
    as its own AugmentationPlan artifact rather than as the Stage-3 acquisition input."""
    from orchestration.specs import load_agent_specs

    from .mock_runtime import MockAgentRuntime
    from .models import RuntimeContext
    from .production_router import run_role

    from .events import CampaignEventEmitter  # noqa: PLC0415
    from .cli import (  # noqa: PLC0415
        CAMPAIGN_FAILED, CAMPAIGN_RESOURCE_BLOCKED, CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL,
        EXIT_APPROVAL_REQUIRED, EXIT_BLOCKED_POLICY, EXIT_PROVIDER_UNAVAILABLE,
        EXIT_VALIDATION_REJECTED, CampaignRunResult, _ProviderBlocked,
        _select_reasoning_provider_runtime,
    )
    from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap  # noqa: PLC0415

    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    _register_proposal_model_once()

    try:
        ctx = provider.build_context(c)
    except AcquisitionCapabilityGap as gap:
        emitter.emit("acquisition_capability_gap",
                     detail={"gap_kind": gap.gap_kind, "message": str(gap)})
        return ProducerRealizeResult(failure=CampaignRunResult(
            CAMPAIGN_RESOURCE_BLOCKED, EXIT_BLOCKED_POLICY,
            f"autonomous acquisition planning cannot proceed ({gap.gap_kind}): {gap}"))
    coverage_sha = ctx.coverage.content_sha256()
    strategy_kind = ctx.strategy.kind.value
    exchange = Path(exchange_dir) if exchange_dir else c.run_dir / "exchange"
    specs = load_agent_specs(agent_specs_dir)

    def ctx_factory(provider_name, model_id):
        return RuntimeContext(exchange_dir=str(exchange), repo_root=repo_root,
                              provider=provider_name, model_id=model_id,
                              read_allow_prefixes=[], tools_enabled=False)

    def build_task(prior_rejection=None):
        context = {
            "expected_output_model": "AcquisitionPlanProposal",
            "coverage_gap_sha256": coverage_sha,
            "strategy_kind": strategy_kind,
            "admissible_parent_ids": list(ctx.admissible_parent_ids),
            "required_param_keys": list(ctx.required_param_keys),
            "param_bounds": {k: list(v) for k, v in (ctx.param_bounds or {}).items()},
            "unsaturated_core_gaps": [g.regime_id for g in ctx.coverage.unsaturated_core_gaps()],
        }
        instruction = (
            "Decide the low-level acquisition recipe (which parents from "
            "context.admissible_parent_ids, how many candidates per structure, and the generation "
            "params) that best closes context.unsaturated_core_gaps under the already-selected "
            "context.strategy_kind. The coverage analysis establishes only what is admissible, "
            "never which recipe to use.")
        if prior_rejection is not None:
            context["prior_attempt_rejection"] = prior_rejection
            instruction += (" This is a correction of a prior proposal the authoritative validator "
                            "rejected for the exact reason in context.prior_attempt_rejection -- "
                            "do not repeat it.")
        return {
            "schema_version": 1, "task_id": f"{c.state['run_id']}-{task_id_suffix}",
            "agent": producer_role, "run_id": c.state["run_id"], "created_at": "run-campaign",
            "instruction": instruction, "inputs": [],
            "criteria": [
                "selected_parent_ids is a non-empty subset of context.admissible_parent_ids",
                "strategy_kind equals context.strategy_kind",
                "every key in context.required_param_keys is present in params and within "
                "context.param_bounds",
                "coverage_gap_sha256 equals context.coverage_gap_sha256",
                "rationale is evidence-bound to the unsaturated core gaps",
            ],
            "constraints": [
                "selected_parent_ids must be drawn only from context.admissible_parent_ids",
                "strategy_kind may not override context.strategy_kind",
                "params must satisfy every declared bound in context.param_bounds",
            ],
            "context": context,
        }

    task = build_task()
    if runtime == "mock":
        if not mock_producer_response:
            raise ValueError(
                "--mock-acquisition-response is required: an autonomous acquisition provider is "
                "registered and --runtime mock cannot self-generate an AcquisitionPlanProposal")
        mock_paths = [Path(p) for p in str(mock_producer_response).split(",")]
        counter = {"n": 0}

        def _next(_t, _s, _ts):
            idx = min(counter["n"], len(mock_paths) - 1)
            counter["n"] += 1
            return mock_paths[idx].read_text(), (0, 0)

        producer_runtime = MockAgentRuntime(_next)
        producer_provider, producer_model = "mock", "mock"
    else:
        try:
            (producer_runtime, producer_provider,
             producer_model) = _select_reasoning_provider_runtime()
        except _ProviderBlocked as exc:
            if exc.reason == "APPROVAL_REQUIRED":
                return ProducerRealizeResult(failure=CampaignRunResult(
                    CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, EXIT_APPROVAL_REQUIRED, exc.message))
            return ProducerRealizeResult(failure=CampaignRunResult(
                CAMPAIGN_RESOURCE_BLOCKED, EXIT_PROVIDER_UNAVAILABLE, exc.message))

    realized_holder: dict[str, RealizedAcquisition] = {}

    def proposal_validator(proposal):
        validated = validate_acquisition_plan_proposal(
            proposal, expected_run_id=c.state["run_id"],
            expected_coverage_gap_sha256=coverage_sha,
            admissible_strategy_kind=strategy_kind,
            admissible_parent_ids=ctx.admissible_parent_ids,
            required_param_keys=ctx.required_param_keys,
            param_bounds=ctx.param_bounds)
        realized = provider.realize(c, ctx, validated)
        issues = validate_acquisition_plan_v2(
            realized.plan, objective=ctx.objective, inventory=ctx.inventory,
            target_regime_model=ctx.target_regime_model,
            region_resolution=ctx.region_resolution, coverage=ctx.coverage,
            strategy=ctx.strategy, generation_result=realized.generation_result,
            selection_result=realized.selection_result,
            labeling_request=realized.labeling_request)
        if issues:
            raise ValueError("acquisition plan failed deterministic validation: " + "; ".join(issues))
        realized_holder["value"] = realized
        return validated

    max_attempts = 3
    res = None
    for attempt_number in range(1, max_attempts + 1):
        attempt_task = task if attempt_number == 1 else build_task(
            prior_rejection={"attempt": attempt_number - 1, "validation_error": res.error})
        emitter.emit("role_invocation_started", role=producer_role,
                     action=action_label, detail={"attempt": attempt_number})
        res = run_role(producer_runtime, attempt_task, specs[producer_role],
                       ctx_factory(producer_provider, producer_model), mode="primary",
                       reasoning_validator=proposal_validator)
        emitter.emit("role_invocation_completed", role=producer_role,
                     action=action_label,
                     detail={"accepted": res.accepted, "attempt": attempt_number})
        if res.accepted:
            break
    if not res.accepted:
        return ProducerRealizeResult(failure=CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            f"acquisition plan proposal rejected after {attempt_number} attempt(s) "
            f"(initial + up to {max_attempts - 1} semantic-correction retries): {res.error}"))

    return ProducerRealizeResult(ctx=ctx, realized=realized_holder["value"])


def plan_acquisition_via_reasoning_roles(
    controller, *, runtime, agent_specs_dir, exchange_dir, repo_root,
    mock_producer_response=None, emitter=None, provider=None,
    producer_role="data-curator",
):
    """Automatic pre-campaign autonomous acquisition planning. Returns None on success (a plan is
    now bound, campaign dispatch may proceed) or when the path does not apply; returns a
    terminal/pausing ``CampaignRunResult`` on failure. Structure deliberately mirrors
    ``cli._commit_teacher_validation_plan_via_reasoning_roles``."""
    import json as _json  # noqa: F401  (json already imported at module top)

    from .orchestrator_bridge import OrchestratorActionProposal, dispatch_orchestrator_action

    from .events import CampaignEventEmitter  # noqa: PLC0415
    from .cli import (  # noqa: PLC0415
        CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED, CampaignRunResult,
    )

    provider = provider if provider is not None else _PROVIDER
    if provider is None or not provider.applies(controller):
        return None

    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    produced = run_acquisition_producer(
        c, runtime=runtime, agent_specs_dir=agent_specs_dir, exchange_dir=exchange_dir,
        repo_root=repo_root, mock_producer_response=mock_producer_response, emitter=emitter,
        provider=provider, producer_role=producer_role)
    if produced.failure is not None:
        return produced.failure

    realized = produced.realized
    plan_dir = c.run_dir / "acquisition" / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{c.state['run_id']}.acquisition_plan.json"
    # Bind whichever executable projection the strategy realized. Both the legacy perturbation
    # projection and the existing-pool selection projection are consumable by the ACQUISITION stage
    # executor (the executor discriminates on the projection's own fields). A dynamics-protocol-only
    # plan has no consumable projection for the current executor and fails closed.
    executable_projection = realized.legacy_projection or realized.existing_pool_projection
    if executable_projection is None:
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            "autonomous acquisition produced a non-executable (dynamics-protocol) plan, which the "
            "current ACQUISITION stage executor cannot consume as a bound acquisition_plan input")
    plan_path.write_text(json.dumps(executable_projection, indent=2, sort_keys=True) + "\n")

    action_proposal = OrchestratorActionProposal(
        run_id=c.state["run_id"], stage="__pre_campaign__", requested_at="run-campaign",
        rationale="bind the autonomously-designed acquisition plan before the ACQUISITION stage",
        idempotency_key=f"{c.state['run_id']}:acquisition_planning:{realized.plan.content_sha256()}",
        action_type="propose_acquisition_plan",
        parameters={"run_dir": str(c.run_dir), "plan_path": str(plan_path)})
    outcome = dispatch_orchestrator_action(action_proposal, controller=c, mode="primary")
    if outcome.status != "EXECUTED":
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            f"propose_acquisition_plan dispatch failed: {outcome.status}: {outcome.reason}")
    emitter.emit("acquisition_plan_bound",
                 detail={"plan_sha256": realized.plan.content_sha256(),
                         "bound_input_sha256": outcome.artifact.get("sha256")})
    return None
