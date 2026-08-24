"""Autonomous acquisition planning: end-to-end + fail-closed tests for the framework-evolution
pipeline that replaces the human-supplied 14-field ``AcquisitionPlan`` with an
objective-conditioned autonomous planner (mirrors ``test_teacher_validation_plan_autonomy``).

The run-specific probing (which SiO2 pool / backends / Teacher inventory, and realizing the
generation+selection chain) lives behind an injected ``AcquisitionPlanningProvider`` seam, so
these tests inject a fully deterministic FAKE provider -- no real Teacher, GPU, or network. The
fake builds ONE valid ``framework_v2.acquisition`` evidence chain and reuses the exact same
objects on every call, so its content-SHAs are stable (every contract carries an ``established_at``
default that would otherwise drift on reconstruction).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPECS = str(ROOT / "agent_specs")

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


# --------------------------------------------------------------------------------------------
# Deterministic valid evidence-chain fixture (framework_v2.acquisition)
# --------------------------------------------------------------------------------------------
def _build_chain(*, projection="legacy"):
    """Build one internally-consistent ``framework_v2.acquisition`` evidence chain that passes
    ``validate_acquisition_plan_v2``. ``projection="legacy"`` produces a bound 14-field legacy
    projection (consumable by the existing ACQUISITION executor); ``projection="dynamics"``
    produces a dynamics-protocol plan (NOT consumable -- used to prove the planner fails closed
    at binding)."""
    from framework_v2.acquisition.contracts import (
        AcquisitionPhase,
        AcquisitionStrategy,
        AcquisitionStrategyKind,
        BackendCapabilityRecord,
        CampaignObjective,
        CandidateGenerationResult,
        CandidateSelectionResult,
        CanonicalLabelingRequest,
        CoverageGapAnalysis,
        GenerationProvenance,
        MetadataAuditVerdict,
        MetadataConsistencyAudit,
        ProtectedDisjointnessReport,
        RegionResolution,
        RegionResolutionMode,
        RegimeCoverage,
        RelevanceRole,
        SourceAndCapabilityInventory,
        SourceCategoryRecord,
        TargetRegime,
        TargetRegimeModel,
        TeacherCapabilityRecord,
    )
    from framework_v2.acquisition.plan_assembly import assemble_plan_v2

    objective = CampaignObjective(
        objective_id="obj-1", primary_target="generic target", claim_scope="generic scope",
        scope_contract_sha256="s" * 64, phase=AcquisitionPhase.INITIAL)
    inventory = SourceAndCapabilityInventory(
        inventory_id="inv-1", objective_sha256=objective.content_sha256(),
        sources=[SourceCategoryRecord(category="bulk", n_items=10, has_metadata=False,
                                      provenance_class="sanitized_pool")],
        backends=[BackendCapabilityRecord(
            backend_id="pert-backend", strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
            feasible=True, supported_capabilities=["perturb"])],
        teacher=TeacherCapabilityRecord(teacher_id="t", can_label=True, can_drive_dynamics=False,
                                        identity_sha256="tid"))
    target_regime_model = TargetRegimeModel(
        model_id="trm-1", objective_sha256=objective.content_sha256(), descriptor="d",
        regimes=[TargetRegime(regime_id="core1", label="core", relevance_role=RelevanceRole.CORE_TARGET,
                              membership_rule="always")])
    region_resolution = RegionResolution(
        resolution_id="rr-1", mode=RegionResolutionMode.DISCOVERED,
        domain_representation_sha256="dr" * 32,
        metadata_audit=MetadataConsistencyAudit(metadata_present=False, audited=False,
                                                verdict=MetadataAuditVerdict.PASS))
    coverage = CoverageGapAnalysis(
        analysis_id="cga-1", phase=AcquisitionPhase.INITIAL,
        target_regime_model_sha256=target_regime_model.content_sha256(),
        region_resolution_sha256=region_resolution.content_sha256(),
        per_regime=[RegimeCoverage(regime_id="core1", relevance_role=RelevanceRole.CORE_TARGET,
                                   current_count=0, target_count=10, saturation=0.0,
                                   novelty_headroom=1.0, gap_score=1.0, saturated=False)])
    strategy = AcquisitionStrategy(
        strategy_id="st-1", kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
        selected_backend_ids=["pert-backend"], coverage_gap_sha256=coverage.content_sha256(),
        inventory_sha256=inventory.content_sha256(),
        rationale="close the core1 coverage gap by local perturbation")
    generation_result = CandidateGenerationResult(
        result_id="gen-1", strategy_sha256=strategy.content_sha256(), backend_id="pert-backend",
        candidate_ids=["c0", "c1"],
        provenance=[GenerationProvenance(candidate_id="c0",
                                         strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
                                         backend_id="pert-backend", parent_id="p0"),
                    GenerationProvenance(candidate_id="c1",
                                         strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
                                         backend_id="pert-backend", parent_id="p0")],
        n_requested=2, n_generated=2, n_rejected=0)
    selection_result = CandidateSelectionResult(
        selection_id="sel-1", generation_result_sha256=generation_result.content_sha256(),
        selector="diversity", selected_candidate_ids=["c0", "c1"],
        disjointness_report=ProtectedDisjointnessReport(status="PASS", n_checked=2, n_overlaps=0,
                                                        dft_labels_used_as_selection_scores=False))
    labeling_request = CanonicalLabelingRequest(
        request_id="lab-1", selection_result_sha256=selection_result.content_sha256(),
        teacher_identity_sha256="tid", candidate_ids=["c0", "c1"], relabel_from_scratch=True)

    legacy_projection = None
    dynamics_protocol_sha256 = None
    if projection == "legacy":
        legacy_projection = {
            "schema_version": 1, "eligible_source_categories": ["bulk"],
            "selected_parent_structure_ids": ["p0"], "selected_source_global_indices": [0],
            "n_parents": 1, "n_per_structure": 2, "T_K": 300.0, "beta": 0.5,
            "sigma_range_A": [0.01, 0.1], "cell_sigma": None, "seed": 42,
            "expected_output_count": 2, "duplicate_handling": "reject",
            "protected_reference_exclusion_report": {
                "status": "PASS", "n_checked": 2, "n_overlaps": 0,
                "dft_labels_used_as_selection_scores": False}}
    else:
        dynamics_protocol_sha256 = "dyn" * 21 + "d"  # 64 hex-ish chars, content unimportant

    plan = assemble_plan_v2(
        plan_id="plan-1", objective=objective, inventory=inventory,
        target_regime_model=target_regime_model, region_resolution=region_resolution,
        coverage=coverage, strategy=strategy, generation_result=generation_result,
        selection_result=selection_result, labeling_request=labeling_request,
        legacy_projection=legacy_projection, dynamics_protocol_sha256=dynamics_protocol_sha256)
    return {
        "objective": objective, "inventory": inventory,
        "target_regime_model": target_regime_model, "region_resolution": region_resolution,
        "coverage": coverage, "strategy": strategy, "generation_result": generation_result,
        "selection_result": selection_result, "labeling_request": labeling_request,
        "plan": plan, "legacy_projection": legacy_projection,
    }


def _context_from_chain(chain, *, admissible_parent_ids=("p0",),
                        required_param_keys=("T_K", "beta", "seed"),
                        param_bounds=None):
    from runtimes.pydantic_ai.acquisition_planner import AcquisitionPlanningContext
    if param_bounds is None:
        param_bounds = {"T_K": (0.0, 1000.0), "beta": (0.0, 1.0)}
    return AcquisitionPlanningContext(
        objective=chain["objective"], inventory=chain["inventory"],
        target_regime_model=chain["target_regime_model"],
        region_resolution=chain["region_resolution"], coverage=chain["coverage"],
        strategy=chain["strategy"], admissible_parent_ids=tuple(admissible_parent_ids),
        teacher_identity_sha256="tid", required_param_keys=tuple(required_param_keys),
        param_bounds=param_bounds)


class _FakeProvider:
    """A deterministic, test-only ``AcquisitionPlanningProvider`` that returns the SAME cached
    context + realized chain on every call (stable content-SHAs)."""

    def __init__(self, context, chain, *, applies=True, legacy_override="unset"):
        self._context = context
        self._chain = chain
        self._applies = applies
        self._legacy_override = legacy_override
        self.realize_calls = 0

    def applies(self, controller):
        return self._applies

    def build_context(self, controller):
        return self._context

    def realize(self, controller, context, proposal):
        from runtimes.pydantic_ai.acquisition_planner import RealizedAcquisition
        self.realize_calls += 1
        legacy = (self._chain["legacy_projection"] if self._legacy_override == "unset"
                  else self._legacy_override)
        return RealizedAcquisition(
            generation_result=self._chain["generation_result"],
            selection_result=self._chain["selection_result"],
            labeling_request=self._chain["labeling_request"],
            plan=self._chain["plan"], legacy_projection=legacy)


# --------------------------------------------------------------------------------------------
# Minimal controller / proposal fixtures
# --------------------------------------------------------------------------------------------
def _init_controller(root, *, run_id="acq-autonomy"):
    from workflow.controller import RunController
    dummy = root / "dummy_input.txt"
    dummy.write_text("bound input placeholder\n")
    cfg = {"run_id": run_id, "inputs": [str(dummy)],
           "stages": [{"name": "acquisition", "command": None,
                       "gate": {"criteria": ["acquisition plan is complete"]}}]}
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump(cfg, sort_keys=False))
    run_dir = root / "run"
    RunController.initialize(workflow, run_dir)
    return RunController(run_dir)


def _proposal_payload(*, run_id, coverage_gap_sha256, strategy_kind="LOCAL_PERTURBATION",
                      selected_parent_ids=("p0",), params=None, rationale="close core1 gap"):
    if params is None:
        params = {"T_K": 300.0, "beta": 0.5, "seed": 42}
    return {"run_id": run_id, "coverage_gap_sha256": coverage_gap_sha256,
            "strategy_kind": strategy_kind, "selected_parent_ids": list(selected_parent_ids),
            "selected_source_global_indices": [0], "n_per_structure": 2, "params": params,
            "rationale": rationale}


def _write_proposal(root, name, payload):
    path = root / name
    path.write_text(json.dumps(payload))
    return path


def _plan(controller, provider, *, mock_response, run_id):
    from runtimes.pydantic_ai.acquisition_planner import plan_acquisition_via_reasoning_roles
    return plan_acquisition_via_reasoning_roles(
        controller, runtime="mock", agent_specs_dir=SPECS,
        exchange_dir=str(controller.run_dir / "exchange"), repo_root=str(ROOT),
        mock_producer_response=str(mock_response), provider=provider)


# ============================================================================================
# Case 1 -- objective-consistency Judge criterion is present in the ACQUISITION StageReviewSpec
# ============================================================================================
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class Case1_ObjectiveConsistencyCriterionPresent(unittest.TestCase):
    def test_acquisition_spec_has_objective_consistency_criterion(self):
        from framework_v2.review_spec import default_stage_review_specs
        from framework_v2.stages import CanonicalStage as S

        spec = default_stage_review_specs()[S.ACQUISITION.value]
        ids = [c.criterion_id for c in spec.criteria]
        self.assertIn("aq-objective-consistency", ids)
        crit = next(c for c in spec.criteria if c.criterion_id == "aq-objective-consistency")
        # It must resolve to a registered failure code (fail-closed ReviewCriterion validator).
        self.assertEqual(crit.failure_code, "dataset_coverage")


# ============================================================================================
# Case 2-5 -- validate_acquisition_plan_proposal fail-closed contextual checks (unit)
# ============================================================================================
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class Case2to5_ProposalValidatorFailsClosed(unittest.TestCase):
    def _validated(self, payload, **over):
        from runtimes.pydantic_ai.acquisition_plan import (
            AcquisitionPlanProposal, validate_acquisition_plan_proposal,
        )
        kwargs = dict(expected_run_id="r", expected_coverage_gap_sha256="cov-sha",
                      admissible_strategy_kind="LOCAL_PERTURBATION",
                      admissible_parent_ids=("p0", "p1"),
                      required_param_keys=("T_K",), param_bounds={"T_K": (0.0, 1000.0)})
        kwargs.update(over)
        proposal = AcquisitionPlanProposal(**payload)
        return validate_acquisition_plan_proposal(proposal, **kwargs)

    def _base(self, **over):
        p = dict(run_id="r", coverage_gap_sha256="cov-sha", strategy_kind="LOCAL_PERTURBATION",
                 selected_parent_ids=["p0"], selected_source_global_indices=[0],
                 n_per_structure=2, params={"T_K": 300.0}, rationale="x")
        p.update(over)
        return p

    def test_valid_proposal_passes(self):
        self._validated(self._base())  # no raise

    def test_stale_coverage_gap_sha256_rejected(self):
        from runtimes.pydantic_ai.acquisition_plan import AcquisitionPlanProposalValidationError
        with self.assertRaisesRegex(AcquisitionPlanProposalValidationError, "coverage_gap_sha256"):
            self._validated(self._base(coverage_gap_sha256="STALE"))

    def test_non_admissible_parent_rejected(self):
        from runtimes.pydantic_ai.acquisition_plan import AcquisitionPlanProposalValidationError
        with self.assertRaisesRegex(AcquisitionPlanProposalValidationError, "not admissible"):
            self._validated(self._base(selected_parent_ids=["p0", "p_not_admitted"]))

    def test_strategy_override_rejected(self):
        from runtimes.pydantic_ai.acquisition_plan import AcquisitionPlanProposalValidationError
        with self.assertRaisesRegex(AcquisitionPlanProposalValidationError, "strategy"):
            self._validated(self._base(strategy_kind="TEACHER_DRIVEN_MD"))

    def test_out_of_bounds_param_rejected(self):
        from runtimes.pydantic_ai.acquisition_plan import AcquisitionPlanProposalValidationError
        with self.assertRaisesRegex(AcquisitionPlanProposalValidationError, "admissible interval"):
            self._validated(self._base(params={"T_K": 5000.0}))

    def test_missing_required_param_rejected(self):
        from runtimes.pydantic_ai.acquisition_plan import AcquisitionPlanProposalValidationError
        with self.assertRaisesRegex(AcquisitionPlanProposalValidationError, "missing required"):
            self._validated(self._base(params={"beta": 0.5}))


# ============================================================================================
# Case 6 -- hard no-op when no provider is registered (path is strictly opt-in)
# ============================================================================================
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class Case6_HardNoOpWithoutProvider(unittest.TestCase):
    def test_returns_none_and_binds_nothing_when_no_provider(self):
        from runtimes.pydantic_ai.acquisition_planner import (
            plan_acquisition_via_reasoning_roles, set_acquisition_planning_provider,
        )
        set_acquisition_planning_provider(None)  # ensure no module-level provider
        with tempfile.TemporaryDirectory() as tmp:
            c = _init_controller(Path(tmp))
            n_inputs_before = len(c.state["inputs"])
            result = plan_acquisition_via_reasoning_roles(
                c, runtime="mock", agent_specs_dir=SPECS,
                exchange_dir=str(c.run_dir / "exchange"), repo_root=str(ROOT),
                mock_producer_response=None, provider=None)
            self.assertIsNone(result)
            from workflow.controller import RunController
            self.assertEqual(len(RunController(c.run_dir).state["inputs"]), n_inputs_before)

    def test_provider_that_does_not_apply_is_a_no_op(self):
        from runtimes.pydantic_ai.acquisition_planner import plan_acquisition_via_reasoning_roles
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _init_controller(root)
            chain = _build_chain()
            provider = _FakeProvider(_context_from_chain(chain), chain, applies=False)
            result = plan_acquisition_via_reasoning_roles(
                c, runtime="mock", agent_specs_dir=SPECS,
                exchange_dir=str(c.run_dir / "exchange"), repo_root=str(ROOT),
                mock_producer_response=None, provider=provider)
            self.assertIsNone(result)
            self.assertEqual(provider.realize_calls, 0)


# ============================================================================================
# Case 7 -- happy path: a valid recipe assembles, passes v2 validation, binds via bind_new_input
# ============================================================================================
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class Case7_SuccessfulAutonomousPlanBinds(unittest.TestCase):
    def test_valid_first_proposal_binds_and_incurs_one_llm_call(self):
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _init_controller(root)
            chain = _build_chain()
            ctx = _context_from_chain(chain)
            provider = _FakeProvider(ctx, chain)
            coverage_sha = chain["coverage"].content_sha256()
            proposal = _write_proposal(root, "ok.json", _proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha))

            n_before = len(c.state["inputs"])
            result = _plan(c, provider, mock_response=proposal, run_id=c.state["run_id"])
            self.assertIsNone(result, "success returns None so campaign dispatch may proceed")

            c2 = RunController(c.run_dir)
            self.assertEqual(len(c2.state["inputs"]), n_before + 1,
                             "the autonomous plan must be bound as a new run input")
            bound = c2.state["inputs"][-1]
            self.assertTrue(bound["source"].endswith(".acquisition_plan.json"))
            self.assertTrue(any(e["type"] == "input_bound" for e in c2.state["events"]))
            self.assertEqual(provider.realize_calls, 1)


# ============================================================================================
# Case 8 -- bounded semantic-correction retry: rejected-then-corrected -> binds; exactly 2 calls
# ============================================================================================
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class Case8_BoundedSemanticCorrectionRetry(unittest.TestCase):
    def _completed_events(self, run_dir):
        path = run_dir / "campaign_events.jsonl"
        if not path.is_file():
            return []
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [e for e in events if e.get("event") == "role_invocation_completed"
                and e.get("action") == "acquisition_plan_proposal"]

    def test_first_rejected_then_corrected_proposal_binds(self):
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _init_controller(root)
            chain = _build_chain()
            ctx = _context_from_chain(chain)
            provider = _FakeProvider(ctx, chain)
            coverage_sha = chain["coverage"].content_sha256()

            # Attempt 1: an out-of-bounds T_K -> rejected by validate_acquisition_plan_proposal.
            rejected = _write_proposal(root, "attempt1.json", _proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha,
                params={"T_K": 5000.0, "beta": 0.5, "seed": 42},
                rationale="out-of-bounds temperature"))
            # Attempt 2: corrected within bounds -> accepted, assembled, v2-validated, bound.
            corrected = _write_proposal(root, "attempt2.json", _proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha,
                rationale="corrected temperature within admissible bounds"))

            result = _plan(c, provider, mock_response=f"{rejected},{corrected}",
                           run_id=c.state["run_id"])
            self.assertIsNone(result)

            c2 = RunController(c.run_dir)
            self.assertTrue(c2.state["inputs"][-1]["source"].endswith(".acquisition_plan.json"))
            completed = self._completed_events(c.run_dir)
            self.assertEqual([e["detail"]["accepted"] for e in completed], [False, True])
            # realize is only reached once contextual validation passes -> only the accepted attempt.
            self.assertEqual(provider.realize_calls, 1)

    def test_never_corrected_proposal_fails_closed_within_bound(self):
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _init_controller(root)
            chain = _build_chain()
            ctx = _context_from_chain(chain)
            provider = _FakeProvider(ctx, chain)
            coverage_sha = chain["coverage"].content_sha256()
            n_before = len(c.state["inputs"])

            # The same non-admissible parent every attempt -- never corrected.
            always_bad = _write_proposal(root, "bad.json", _proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha,
                selected_parent_ids=["p_not_admitted"]))

            result = _plan(c, provider, mock_response=always_bad, run_id=c.state["run_id"])
            self.assertIsNotNone(result, "a never-corrected proposal must fail closed")
            from runtimes.pydantic_ai import cli
            self.assertEqual(result.exit_code, cli.EXIT_VALIDATION_REJECTED)

            c2 = RunController(c.run_dir)
            self.assertEqual(len(c2.state["inputs"]), n_before, "nothing may bind")
            completed = self._completed_events(c.run_dir)
            self.assertEqual(len(completed), 3, "initial attempt + at most 2 retries, then FAILED")
            self.assertEqual([e["detail"]["accepted"] for e in completed], [False, False, False])
            self.assertEqual(provider.realize_calls, 0)


# ============================================================================================
# Case 9 -- a dynamics-protocol (non-legacy) plan fails closed at binding
# ============================================================================================
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class Case9_DynamicsProtocolPlanFailsClosedAtBinding(unittest.TestCase):
    def test_non_legacy_realized_plan_is_not_bound(self):
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _init_controller(root)
            # The realized plan itself is a valid dynamics-protocol AcquisitionPlanV2 (passes v2
            # validation), but RealizedAcquisition.legacy_projection is None -> the current
            # ACQUISITION executor cannot consume it, so the planner must fail closed at binding.
            chain = _build_chain(projection="dynamics")
            ctx = _context_from_chain(chain)
            provider = _FakeProvider(ctx, chain, legacy_override=None)
            coverage_sha = chain["coverage"].content_sha256()
            n_before = len(c.state["inputs"])
            proposal = _write_proposal(root, "ok.json", _proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha))

            result = _plan(c, provider, mock_response=proposal, run_id=c.state["run_id"])
            self.assertIsNotNone(result)
            from runtimes.pydantic_ai import cli
            self.assertEqual(result.exit_code, cli.EXIT_VALIDATION_REJECTED)
            self.assertIn("non-executable (dynamics-protocol) plan", result.message)
            self.assertEqual(len(RunController(c.run_dir).state["inputs"]), n_before)


# ============================================================================================
# Case 10 -- the propose_acquisition_plan orchestrator bridge binds an on-disk plan directly
# ============================================================================================
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class Case10_OrchestratorBridgeBindsPlan(unittest.TestCase):
    def test_bridge_executor_binds_via_bind_new_input(self):
        from runtimes.pydantic_ai.orchestrator_bridge import (
            OrchestratorActionProposal, dispatch_orchestrator_action,
        )
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _init_controller(root)
            plan_path = root / "some.acquisition_plan.json"
            plan_path.write_text(json.dumps({"schema_version": 1, "n_parents": 1}))
            n_before = len(c.state["inputs"])

            proposal = OrchestratorActionProposal(
                run_id=c.state["run_id"], stage="__pre_campaign__", requested_at="test",
                rationale="bind autonomous acquisition plan", idempotency_key="k-1",
                action_type="propose_acquisition_plan",
                parameters={"run_dir": str(c.run_dir), "plan_path": str(plan_path)})

            # dry-run makes no controller call
            dry = dispatch_orchestrator_action(proposal, controller=c, mode="dry_run")
            self.assertEqual(dry.status, "DRY_RUN")
            self.assertEqual(len(RunController(c.run_dir).state["inputs"]), n_before)

            outcome = dispatch_orchestrator_action(proposal, controller=c, mode="primary")
            self.assertEqual(outcome.status, "EXECUTED")
            c2 = RunController(c.run_dir)
            self.assertEqual(len(c2.state["inputs"]), n_before + 1)
            self.assertEqual(c2.state["inputs"][-1]["sha256"], outcome.artifact["sha256"])

    def test_bridge_requires_plan_path(self):
        from runtimes.pydantic_ai.orchestrator_bridge import (
            OrchestratorActionProposal, dispatch_orchestrator_action,
        )
        with tempfile.TemporaryDirectory() as tmp:
            c = _init_controller(Path(tmp))
            proposal = OrchestratorActionProposal(
                run_id=c.state["run_id"], stage="__pre_campaign__", requested_at="test",
                rationale="missing plan path", idempotency_key="k-2",
                action_type="propose_acquisition_plan",
                parameters={"run_dir": str(c.run_dir)})
            outcome = dispatch_orchestrator_action(proposal, controller=c, mode="primary")
            self.assertEqual(outcome.status, "EXECUTOR_ERROR")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
