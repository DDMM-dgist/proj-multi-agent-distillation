"""Full Stage-3 (ACQUISITION) autonomous-planning INTEGRATION test.

Where ``test_autonomous_acquisition_planning`` injects a fully fake
``AcquisitionPlanningProvider`` (proving the planner *seam*), this module proves the
REAL framework-default path end-to-end through the user-mandated plugin
auto-materialization architecture:

    run-campaign (no human AcquisitionPlan bound)
      -> the framework-default ``FrameworkDefaultAcquisitionProvider`` applies
      -> a REGISTERED ``StructuralDescriptorProvider`` supplies descriptor-space FACTS
      -> ``materialize_acquisition_evidence`` freezes the generic pipeline
         (inventory / target-regime / region / coverage / strategy) into a typed artifact
      -> the dispatched producer emits a typed ``AcquisitionPlanProposal`` (the ONE
         genuinely-scientific choice)
      -> deterministic contextual validation + ``validate_acquisition_plan_v2``
      -> a bounded semantic-correction retry when the first proposal is inadmissible
      -> the plan is bound to the run as a new input via the audited orchestrator bridge
      -> the ACQUISITION stage runs and is gated by the EXISTING fixed 3-Judge committee
         (3 independent, mutually-blind lens contracts) under the ACQUISITION
         ``StageReviewSpec`` with the EXISTING consensus/gate semantics.

The whole chain is deterministic and hermetic: no real Teacher, GPU, or network. The
Teacher-capability probe is the REAL framework-default one (it pins identity to the
content of the run's own bound Teacher file, which the test writes); the backend probe
is the designed injected environment seam (this environment has no importable local
perturbation backend, so the test injects a feasible one -- exactly the seam a real
campaign uses to report its own environment's feasible backends).

The load-bearing proof: the *user* supplies NONE of n_parents / domain percentages /
parent ids / sigma / T / beta / seed. Every admissible value is a materialized FACT and
the recipe is the dispatched Agent's typed proposal.
"""
from __future__ import annotations

import json
import sys
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

# The low-level acquisition knobs a human must NEVER have to supply for the autonomous
# path -- proving their absence from user config is the point of the whole evolution.
_FORBIDDEN_USER_KNOBS = (
    "n_parents", "sigma_range_A", "cell_sigma", "seed", "T_K", "beta",
    "selected_parent_structure_ids", "n_per_structure",
)

# The immutable 3-slot judge committee (from workflow/review_lenses.py).
_CANONICAL_LENS_IDS = {"evidence_provenance", "scientific_validity", "reproducibility_deployment"}


# --------------------------------------------------------------------------------------------
# A deterministic, test-only material-agnostic descriptor plugin (the ONLY place material work
# would live). It advertises the DISCOVERED path (no trusted metadata) and yields a single
# unsaturated CORE_TARGET gap reachable by local perturbation.
# --------------------------------------------------------------------------------------------
class _SyntheticDescriptorProvider:
    """A hermetic ``StructuralDescriptorProvider``: no heavy deps, deterministic facts."""

    material_id = "synthetic-integration-material"

    def applies(self, *, controller, objective, scope_contract) -> bool:
        # Admissible for any campaign whose scope declares a PRIMARY_DEPLOYMENT region.
        from framework_v2.contracts import ScopeCategory
        return bool(scope_contract.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT))

    def build_descriptor_space_evidence(self, *, controller, objective, scope_contract):
        from framework_v2.acquisition.coverage_gap import RegimeCoverageInput
        from framework_v2.acquisition.contracts import RelevanceRole, SourceCategoryRecord
        from framework_v2.acquisition.descriptor_plugins import DescriptorSpaceEvidence
        from framework_v2.acquisition.strategy import StrategyEvidence
        from framework_v2.contracts import (
            DomainRegime, DomainRepresentation, ScopeCategory)

        primary = scope_contract.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT)[0]
        regime_id = primary.region_id
        scope_sha = objective.scope_contract_sha256

        def _discovered_representation() -> DomainRepresentation:
            return DomainRepresentation(
                representation_id=f"{regime_id}-synthetic-representation",
                kind="continuous",
                descriptor="synthetic 1-D descriptor space",
                regimes=[DomainRegime(
                    regime_id=regime_id, label="core", membership_rule="always",
                    within_scope_categories=[ScopeCategory.PRIMARY_DEPLOYMENT])],
                linked_scope_contract_sha256=scope_sha)

        return DescriptorSpaceEvidence(
            descriptor="synthetic 1-D descriptor space",
            source_records=(SourceCategoryRecord(
                category="bulk", n_items=10, has_metadata=False,
                provenance_class="sanitized_pool"),),
            discovered_representation_builder=_discovered_representation,
            regime_coverage_inputs=(RegimeCoverageInput(
                regime_id=regime_id, relevance_role=RelevanceRole.CORE_TARGET,
                current_count=0, saturation=0.0, novelty_headroom=1.0, target_count=10),),
            # parents_reach_gaps + not needing new configurations -> LOCAL_PERTURBATION.
            strategy_evidence=StrategyEvidence(
                pool_covers_gaps=False, parents_reach_gaps=True,
                gaps_require_new_configurations=False, seed_structures_exist=True),
            admissible_parent_ids=("p0", "p1"),
            # Scalar bounded knobs go in param_bounds; list/None knobs (sigma_range_A, cell_sigma)
            # are presence-checked only (they are not float-coercible), exactly as the local
            # perturbation legacy projection consumes them.
            required_param_keys=("T_K", "beta", "seed", "sigma_range_A", "cell_sigma"),
            param_bounds={"T_K": (0.0, 1000.0), "beta": (0.0, 1.0)},
            eligible_source_categories=("bulk",),
            selected_source_global_indices=(0,),
            metadata_present=False)


def _feasible_local_perturbation_backend_probe(controller, evidence):
    """Injected environment seam: report one feasible LOCAL_PERTURBATION backend.

    This is exactly what ``default_backend_probe`` does, except it does not depend on the
    optional ``augment_atoms`` package being importable in the test environment -- so the
    integration proof of the *happy* bind path is deterministic regardless of environment."""
    from framework_v2.acquisition.contracts import (
        AcquisitionStrategyKind, BackendCapabilityRecord)
    return [BackendCapabilityRecord(
        backend_id="local_perturbation.augment_atoms",
        strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
        feasible=True, supported_capabilities=["acquisition.local_perturbation"])]


def _scope_contract():
    from framework_v2.contracts import (
        DeploymentScopeContract, ScopeCategory, ScopeRegion)
    return DeploymentScopeContract(
        contract_id="acq-integration-scope",
        objective="autonomous acquisition integration deployment scope",
        regions=[ScopeRegion(
            region_id="primary", category=ScopeCategory.PRIMARY_DEPLOYMENT,
            membership_rule="descriptor in the primary deployment set")])


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class StageThreeAutonomousAcquisitionIntegration(unittest.TestCase):
    def setUp(self):
        from framework_v2.acquisition.descriptor_plugins import clear_descriptor_providers
        from runtimes.pydantic_ai.acquisition_planner import set_acquisition_planning_provider
        clear_descriptor_providers()
        set_acquisition_planning_provider(None)

    def tearDown(self):
        from framework_v2.acquisition.descriptor_plugins import clear_descriptor_providers
        from runtimes.pydantic_ai.acquisition_planner import set_acquisition_planning_provider
        clear_descriptor_providers()
        set_acquisition_planning_provider(None)

    # -- fixtures ----------------------------------------------------------------------------
    def _bound_controller(self, root, *, run_id="acq-integration"):
        from workflow.controller import RunController
        teacher = root / "teacher_model.pt"
        teacher.write_text("deterministic teacher model bytes for identity hashing\n")
        cfg = {
            "run_id": run_id,
            "teacher_evidence_sources": {"teacher_model_path": str(teacher)},
            "stages": [{
                "name": "acquisition",
                "command": [sys.executable, "-c",
                            "from pathlib import Path; "
                            "Path('artifacts').mkdir(exist_ok=True); "
                            "Path('artifacts/acquired.txt').write_text('acquired')"],
                "outputs": ["artifacts/acquired.txt"],
                "gate": {"criteria": ["the acquisition batch closes the target coverage gap"]},
            }],
        }
        workflow = root / "workflow.yaml"
        workflow.write_text(yaml.safe_dump(cfg, sort_keys=False))
        RunController.initialize(workflow, root / "run")
        c = RunController(root / "run")
        c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
        return RunController(c.run_dir), cfg

    def _install_default_provider(self, c):
        """Install the REAL framework-default provider with the synthetic plugin registered,
        then pin its single ``build_context`` result (production calls build_context exactly
        once per invocation; pinning lets the test read the materialized coverage SHA to author
        the mock proposal while guaranteeing the planner sees the identical frozen objects)."""
        from runtimes.pydantic_ai.default_acquisition_provider import (
            default_teacher_probe, install_default_acquisition_provider)

        provider = install_default_acquisition_provider(
            backend_probe=_feasible_local_perturbation_backend_probe,
            teacher_probe=default_teacher_probe,
            descriptor_providers=(_SyntheticDescriptorProvider(),))
        ctx = provider.build_context(c)  # the single real materialization
        provider.build_context = lambda controller, _ctx=ctx: _ctx
        return provider, ctx

    def _proposal_payload(self, *, run_id, coverage_gap_sha256, rationale,
                          T_K=300.0, selected_parent_ids=("p0",)):
        # The full recipe the dispatched Agent proposes. sigma_range_A / cell_sigma are the
        # local perturbation displacement knobs the legacy projection consumes.
        params = {"T_K": T_K, "beta": 0.5, "seed": 42,
                  "sigma_range_A": [0.01, 0.1], "cell_sigma": None}
        return {
            "run_id": run_id, "coverage_gap_sha256": coverage_gap_sha256,
            "strategy_kind": "LOCAL_PERTURBATION",
            "selected_parent_ids": list(selected_parent_ids),
            "selected_source_global_indices": [0], "n_per_structure": 2,
            "params": params, "rationale": rationale}

    def _write(self, root, name, payload):
        path = root / name
        path.write_text(json.dumps(payload))
        return path

    def _plan(self, c, provider, *, mock_response):
        from runtimes.pydantic_ai.acquisition_planner import plan_acquisition_via_reasoning_roles
        return plan_acquisition_via_reasoning_roles(
            c, runtime="mock", agent_specs_dir=SPECS,
            exchange_dir=str(c.run_dir / "exchange"), repo_root=str(ROOT),
            mock_producer_response=str(mock_response), provider=provider)

    # -- Part A: full autonomous planner E2E through the real materialization ----------------
    def test_autonomous_planner_materializes_and_binds_without_user_knobs(self):
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, cfg = self._bound_controller(root)

            # Precondition: no acquisition plan is supplied by the user.
            self.assertFalse(
                any(str(i.get("source", "")).endswith("acquisition_plan.json")
                    for i in c.state["inputs"]),
                "no human-authored acquisition plan may be present at start")

            provider, ctx = self._install_default_provider(c)
            # The provider self-gates ON for this v2 run with an acquisition stage + no plan.
            self.assertTrue(provider.applies(c))
            # Every admissible value is a materialized FACT, not user input.
            self.assertEqual(ctx.strategy.kind.value, "LOCAL_PERTURBATION")
            self.assertEqual(ctx.admissible_parent_ids, ("p0", "p1"))

            coverage_sha = ctx.coverage.content_sha256()
            n_before = len(c.state["inputs"])
            proposal = self._write(root, "ok.json", self._proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha,
                rationale="perturb p0 to populate the unsaturated core gap"))

            result = self._plan(c, provider, mock_response=proposal)
            self.assertIsNone(result, "a valid autonomous plan binds; None lets dispatch proceed")

            c2 = RunController(c.run_dir)
            self.assertEqual(len(c2.state["inputs"]), n_before + 1)
            bound = c2.state["inputs"][-1]
            self.assertTrue(bound["source"].endswith(".acquisition_plan.json"))
            self.assertTrue(any(e["type"] == "input_bound" for e in c2.state["events"]))

            # The bound plan SYNTHESIZES the low-level knobs from evidence + the Agent's recipe;
            # prove those knobs were NEVER present in any user-authored config.
            bound_plan = json.loads(Path(bound["source"]).read_text())
            for knob in ("n_parents", "sigma_range_A", "seed", "selected_parent_structure_ids"):
                self.assertIn(knob, bound_plan,
                              f"the framework must synthesize {knob}, not ask the user for it")
            user_cfg_text = json.dumps(cfg)
            for knob in _FORBIDDEN_USER_KNOBS:
                self.assertNotIn(knob, user_cfg_text,
                                 f"user config must not carry the low-level knob {knob!r}")

    # -- Part B: a bounded semantic-correction retry is exercised, then binds ----------------
    def test_inadmissible_first_proposal_is_corrected_then_binds(self):
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = self._bound_controller(root, run_id="acq-integration-retry")
            provider, ctx = self._install_default_provider(c)
            coverage_sha = ctx.coverage.content_sha256()

            rejected = self._write(root, "attempt1.json", self._proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha, T_K=5000.0,
                rationale="temperature above the admissible bound"))
            corrected = self._write(root, "attempt2.json", self._proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha, T_K=300.0,
                rationale="corrected temperature within the admissible bound"))

            result = self._plan(c, provider, mock_response=f"{rejected},{corrected}")
            self.assertIsNone(result)
            c2 = RunController(c.run_dir)
            self.assertTrue(c2.state["inputs"][-1]["source"].endswith(".acquisition_plan.json"))

    # -- Part C: the ACQUISITION stage is gated by the fixed 3-Judge committee ---------------
    def test_acquisition_stage_passes_the_three_judge_gate(self):
        from framework_v2.review_spec import default_stage_review_specs
        from framework_v2.stages import CanonicalStage as S
        from runtimes.pydantic_ai import closure_review as closure
        from workflow.controller import RunController

        STAGE = "acquisition"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = self._bound_controller(root, run_id="acq-integration-gate")
            provider, ctx = self._install_default_provider(c)
            coverage_sha = ctx.coverage.content_sha256()

            proposal = self._write(root, "ok.json", self._proposal_payload(
                run_id=c.state["run_id"], coverage_gap_sha256=coverage_sha,
                rationale="perturb p0 to populate the unsaturated core gap"))
            self.assertIsNone(self._plan(c, provider, mock_response=proposal))

            # Bind the EXISTING ACQUISITION StageReviewSpec (the canonical 3-lens committee).
            spec = default_stage_review_specs()[S.ACQUISITION.value]
            self.assertEqual(set(spec.lens_ids), _CANONICAL_LENS_IDS)
            self.assertIn("aq-objective-consistency", [cr.criterion_id for cr in spec.criteria])
            c = RunController(c.run_dir)
            c.bind_v2_stage_review_spec(STAGE, spec.model_dump(mode="json"))

            # Run the stage, then drive the real per-lens Judge review + consensus.
            c.run_stage(STAGE)
            c = RunController(c.run_dir)
            lenses = c.stage(STAGE)["gate_review_lenses"]
            criteria = c.stage(STAGE)["gate_criteria"]
            self.assertEqual(len(lenses), 3, "exactly three mutually-blind judge lenses")

            votes = []
            for i, lens in enumerate(lenses, 1):
                votes.append({
                    "judge_id": f"judge-{i}", "review_lens": lens["id"], "verdict": "PASS",
                    "criteria_checked": [
                        {"criterion": q, "value_read": "verified", "ok": True} for q in criteria],
                    "rationale": "committee reviewed the autonomous acquisition plan",
                    "required_fix": ""})

            artifacts = {a["path"]: a["sha256"] for a in c.stage_artifacts(STAGE)}
            facts = closure.deterministic_facts_for_stage(STAGE, artifacts, [])
            packet = closure.compile_review_packet(
                controller=c, stage_name=STAGE, spec=spec, facts=facts,
                decision_sha256="acq-integration-decision")
            reviews = [
                closure.judge_vote_to_review(
                    v, v["review_lens"], spec, packet,
                    run_id=c.state["run_id"], stage=STAGE, judge_index=i)
                for i, v in enumerate(votes, 1)]
            self.assertEqual(len(reviews), 3)
            bundle = {"stage": STAGE, "criteria": criteria, "review_lenses": lenses,
                      "artifact_sha256": artifacts, "decision": "PASS", "votes": votes,
                      "v2_review": closure.assemble_v2_review(packet, reviews)}
            gate_path = c.run_dir / "gates" / f"{STAGE}.votes.json"
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate_path.write_text(json.dumps(bundle))
            c.record_gate(STAGE, votes_path=gate_path)

            c = RunController(c.run_dir)
            self.assertEqual(c.stage(STAGE)["gate"], "PASS")
            gate = [e for e in c.state["events"] if e.get("type") == "gate"][-1]
            fw = gate.get("framework_v2") or {}
            self.assertEqual(fw.get("v2_packet_sha256"), packet.packet_sha256())
            self.assertEqual(fw.get("v2_review_spec_sha256"), spec.content_sha256())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
