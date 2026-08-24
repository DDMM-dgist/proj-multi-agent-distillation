"""Focused regression tests for the TWO evidence-triggered acquisition corrections.

GAP 1 -- autonomous AcquisitionPlan -> stage-dispatch wiring: an AcquisitionPlan that is
canonically BOUND to the run (autonomously by the run-campaign planner, or as a human input)
must be auto-resolved into the acquisition-stage proposal's ``acquisition_plan_path`` so the
acquisition executor consumes it WITHOUT a human/workflow.yaml hard-coding a per-run plan path.

GAP 2 -- objective-conditioned eligibility must constrain FPS: deterministic
farthest-point-sampling labeling-population sizing/selection for EXISTING_POOL_SELECTION must
optimize diversity ONLY WITHIN the scientifically admissible candidate population (the source
families the gated reasoning plane declared in-scope). It must never re-introduce a family the
canonical scope decision classified as out-of-scope, and must FAIL CLOSED with a typed
scope-eligibility gap rather than silently falling back to the full pool of all families.

The GAP-2 tests drive the REAL framework-default provider end to end (materialization ->
realize) with a hermetic, MATERIAL-AGNOSTIC multi-family pool: no material name is supplied to
the framework, no human N / per-family quota is supplied, the pool spans opaque family labels
"famA"/"famB"/"famC", and the reasoning-plane recipe names only a subset. The proof: the
realized selection contains ONLY frames from the named families, the size is a derived OUTPUT,
and an admissible set that maps to zero pool frames raises the typed gap.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

# The immutable 3-slot judge committee (from workflow/review_lenses.py) -- asserting it is
# unchanged proves the correction did not touch the closure/consensus machinery.
_CANONICAL_LENS_IDS = {"evidence_provenance", "scientific_validity", "reproducibility_deployment"}

# Material-family names that must NEVER be encoded in framework logic (the eligibility mask has to
# be produced generically from opaque category strings). These are the distinctive SiO2-x source
# families from the campaign pool; a generic corrector never mentions them.
_FORBIDDEN_MATERIAL_TOKENS = (
    "bulk_amo", "bulk_cryst", "siox", "sio2", "silicon_", "highpressure",
    "high_pressure", "quench", "vacancy",
)


def _scope_contract():
    from framework_v2.contracts import (
        DeploymentScopeContract, ScopeCategory, ScopeRegion)
    return DeploymentScopeContract(
        contract_id="gap-eligibility-scope",
        objective="objective-conditioned eligibility deployment scope",
        regions=[ScopeRegion(
            region_id="primary", category=ScopeCategory.PRIMARY_DEPLOYMENT,
            membership_rule="descriptor in the primary deployment set")])


def _multi_family_pool():
    """A hermetic 3-family pool (famA/famB/famC, 5 frames each) with dense generic features.

    Families are well-separated on ``n_atoms`` so the generic representation discovers >1 regime
    (adequate/discriminative), and frames vary within family so FPS has real diversity to work
    with. The family labels are OPAQUE -- no material name -- exactly what the mask must key on.
    """
    from framework_v2.acquisition.generic_representation import LoadedPool, PoolFrame

    families = {"famA": (6, 0.050, 2.00, 0.50),
                "famB": (40, 0.030, 2.50, 0.60),
                "famC": (90, 0.020, 3.00, 0.70)}
    frames = []
    for category, (n0, d0, nn0, mf0) in families.items():
        for i in range(5):
            frames.append(PoolFrame(
                item_id=f"{category}#{i}", category=category, frame_index=i,
                n_atoms=n0 + i,
                features={
                    "n_atoms": float(n0 + i),
                    "number_density_atoms_per_A3": d0 + i * 0.001,
                    "mean_min_neighbor_distance_A": nn0 + i * 0.02,
                    "max_species_fraction": mf0 + i * 0.01,
                }))
    return LoadedPool(
        manifest_path="synthetic-multifamily-pool-manifest.json",
        manifest_sha256="0" * 64, total_frames=len(frames),
        frames=tuple(frames), per_category_counts={"famA": 5, "famB": 5, "famC": 5})


class _MultiFamilyExistingPoolProvider:
    """A hermetic EXISTING_POOL_SELECTION descriptor provider over an opaque multi-family pool."""

    material_id = "synthetic-existing-pool-material"

    def __init__(self, pool):
        self._pool = pool

    def applies(self, *, controller, objective, scope_contract) -> bool:
        from framework_v2.contracts import ScopeCategory
        return bool(scope_contract.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT))

    def build_representation_result(self, *, controller, objective, scope_contract,
                                    region_classifier=None):
        from framework_v2.acquisition.generic_representation import (
            build_adequate_representation)
        return build_adequate_representation(
            self._pool, id_prefix=controller.state["run_id"], scope_contract=scope_contract,
            deployment_claim=objective.claim_scope, region_classifier=region_classifier)

    def build_descriptor_space_evidence(self, *, controller, objective, scope_contract):
        from framework_v2.acquisition.contracts import RelevanceRole, SourceCategoryRecord
        from framework_v2.acquisition.coverage_gap import RegimeCoverageInput
        from framework_v2.acquisition.descriptor_plugins import DescriptorSpaceEvidence
        from framework_v2.acquisition.strategy import StrategyEvidence
        from framework_v2.contracts import ScopeCategory

        result = self.build_representation_result(
            controller=controller, objective=objective, scope_contract=scope_contract)
        pool = result.pool
        representation = result.representation
        primary_regions = scope_contract.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT)
        # A SATURATED core (Case A) resolves deterministically to EXISTING_POOL_SELECTION: the
        # eligible existing pool already closes the gap, so a representative existing subset is
        # sized+selected for canonical labeling (never new-config generation).
        regime_coverage_inputs = tuple(
            RegimeCoverageInput(
                regime_id=region.region_id, relevance_role=RelevanceRole.CORE_TARGET,
                current_count=pool.total_frames, saturation=1.0, novelty_headroom=0.0,
                target_count=None)
            for region in primary_regions)
        categories = sorted(pool.per_category_counts.items())
        source_records = tuple(
            SourceCategoryRecord(category=c, n_items=int(n), has_metadata=False,
                                 provenance_class="sanitized_pool")
            for c, n in categories)

        def _discovered_representation():
            return representation

        return DescriptorSpaceEvidence(
            descriptor=result.spec.descriptor,
            source_records=source_records,
            discovered_representation_builder=_discovered_representation,
            regime_coverage_inputs=regime_coverage_inputs,
            strategy_evidence=StrategyEvidence(
                pool_covers_gaps=True, parents_reach_gaps=True,
                gaps_require_new_configurations=False, seed_structures_exist=True),
            admissible_parent_ids=tuple(f.item_id for f in pool.frames),
            required_param_keys=(),
            param_bounds={},
            eligible_source_categories=tuple(c for c, _ in categories),
            selected_source_global_indices=tuple(range(len(categories))),
            duplicate_handling="reject",
            saturation_threshold=0.8,
            metadata_present=False)


def _existing_pool_backend_probe(controller, evidence):
    """Injected environment seam: report one feasible EXISTING_POOL_SELECTION backend."""
    from framework_v2.acquisition.contracts import (
        AcquisitionStrategyKind, BackendCapabilityRecord)
    return [BackendCapabilityRecord(
        backend_id="existing_pool_selection.ase",
        strategy_kind=AcquisitionStrategyKind.EXISTING_POOL_SELECTION,
        feasible=True, supported_capabilities=["acquisition.existing_pool_selection"])]


class _Recipe:
    """The minimal shape ``realize`` reads for EXISTING_POOL_SELECTION: the reasoning plane's
    gated family-level decision. (n_per_structure/params are ignored by that branch.)"""

    n_per_structure = 1
    params: dict = {}

    def __init__(self, selected_parent_ids):
        self.selected_parent_ids = tuple(selected_parent_ids)


def _category_of(item_id: str) -> str:
    return item_id.rsplit("#", 1)[0]


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ObjectiveConditionedEligibilityTests(unittest.TestCase):
    """GAP 2 -- deterministic FPS is constrained to the admissible in-scope population."""

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

    def _bound_controller(self, root, *, run_id):
        from workflow.controller import RunController
        teacher = root / "teacher_model.pt"
        teacher.write_text("deterministic teacher model bytes for identity hashing\n")
        cfg = {
            "run_id": run_id,
            "teacher_evidence_sources": {"teacher_model_path": str(teacher)},
            "stages": [{
                "name": "acquisition",
                "pydantic_ai": {"role": "data-curator", "action": "acquire_structures"},
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

    def _install(self, c):
        from runtimes.pydantic_ai.default_acquisition_provider import (
            default_teacher_probe, install_default_acquisition_provider)
        provider = install_default_acquisition_provider(
            backend_probe=_existing_pool_backend_probe,
            teacher_probe=default_teacher_probe,
            descriptor_providers=(_MultiFamilyExistingPoolProvider(_multi_family_pool()),))
        ctx = provider.build_context(c)  # the single real materialization; populates the pool cache
        return provider, ctx

    def test_fps_selects_only_within_admissible_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = self._bound_controller(root, run_id="gap2-inscope")
            provider, ctx = self._install(c)
            self.assertEqual(ctx.strategy.kind.value, "EXISTING_POOL_SELECTION")
            # The materialized decision space is the FULL opaque pool (all 3 families, 15 frames).
            self.assertEqual(len(ctx.admissible_parent_ids), 15)

            # The gated reasoning plane declares only famA + famB in-scope (famC excluded).
            recipe = _Recipe(["famA#0", "famA#3", "famB#1", "famB#4"])
            realized = provider.realize(c, ctx, recipe)

            proj = realized.existing_pool_projection
            selected_ids = proj["selected_parent_structure_ids"]
            selected_cats = {_category_of(i) for i in selected_ids}
            # (2) deterministic FPS cannot select a frame outside the admissible scope.
            self.assertTrue(selected_cats.issubset({"famA", "famB"}))
            self.assertNotIn("famC", selected_cats)
            # The narrowed projection records ONLY the admissible families, not all 3.
            self.assertEqual(proj["eligible_source_categories"], ["famA", "famB"])
            # Global indices are FULL-pool positions (famA=0..4, famB=5..9, famC=10..14): every
            # selected index falls in the famA/famB band, proving the mask <-> executor agreement.
            self.assertTrue(all(gi < 10 for gi in proj["selected_source_global_indices"]))
            for gi, iid in zip(proj["selected_source_global_indices"], selected_ids):
                self.assertEqual(ctx.admissible_parent_ids[gi], iid)

            # (3) the size is a derived OUTPUT (no human N / per-family quota was supplied).
            self.assertEqual(proj["n_selected"], len(selected_ids))
            self.assertGreaterEqual(proj["n_selected"], 1)
            self.assertLessEqual(proj["n_selected"], 10)  # <= the 10 admissible frames
            self.assertEqual(
                proj["labeling_population_sizing"]["eligible_population_size"], 10)

            # (5) protected-reference isolation intact: disjointness PASS, DFT labels never used
            #     as selection scores.
            dj = realized.selection_result.disjointness_report
            self.assertEqual(dj.status, "PASS")
            self.assertFalse(dj.dft_labels_used_as_selection_scores)

    def test_single_admissible_family_excludes_all_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = self._bound_controller(root, run_id="gap2-single")
            provider, ctx = self._install(c)
            recipe = _Recipe(["famC#0", "famC#2", "famC#4"])
            realized = provider.realize(c, ctx, recipe)
            proj = realized.existing_pool_projection
            self.assertEqual(proj["eligible_source_categories"], ["famC"])
            self.assertTrue(
                all(_category_of(i) == "famC" for i in proj["selected_parent_structure_ids"]))
            # famC occupies the last 5 global positions (10..14).
            self.assertTrue(all(gi >= 10 for gi in proj["selected_source_global_indices"]))

    def test_admissible_family_with_zero_pool_frames_fails_closed(self):
        """An admissible set that maps to zero eligible frames raises the typed
        SCOPE_ELIGIBILITY_EMPTY gap -- it NEVER silently falls back to the full pool."""
        from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = self._bound_controller(root, run_id="gap2-empty")
            provider, ctx = self._install(c)
            recipe = _Recipe(["famZ#0"])  # a family not present in the pool
            with self.assertRaises(AcquisitionCapabilityGap) as cm:
                provider.realize(c, ctx, recipe)
            self.assertEqual(cm.exception.gap_kind, "SCOPE_ELIGIBILITY_EMPTY")

    def test_no_admissible_family_fails_closed(self):
        """An empty family decision is undecidable -- fail closed, never fall back to all families."""
        from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = self._bound_controller(root, run_id="gap2-undecidable")
            provider, ctx = self._install(c)
            with self.assertRaises(AcquisitionCapabilityGap) as cm:
                provider.realize(c, ctx, _Recipe([]))
            self.assertEqual(cm.exception.gap_kind, "SCOPE_ELIGIBILITY_UNDECIDABLE")


class GapOneAcquisitionPlanWiringTests(unittest.TestCase):
    """GAP 1 -- a controller-bound autonomous AcquisitionPlan is auto-consumed by the stage."""

    def _bound_controller(self, root, *, run_id):
        from workflow.controller import RunController
        teacher = root / "teacher_model.pt"
        teacher.write_text("deterministic teacher model bytes for identity hashing\n")
        cfg = {
            "run_id": run_id,
            "teacher_evidence_sources": {"teacher_model_path": str(teacher)},
            "stages": [{
                "name": "acquisition",
                "pydantic_ai": {"role": "data-curator", "action": "acquire_structures"},
                "command": [sys.executable, "-c", "print('noop')"],
                "outputs": ["artifacts/acquired.txt"],
                "gate": {"criteria": ["the acquisition batch closes the target coverage gap"]},
            }],
        }
        workflow = root / "workflow.yaml"
        workflow.write_text(yaml.safe_dump(cfg, sort_keys=False))
        RunController.initialize(workflow, root / "run")
        return RunController(root / "run"), cfg

    @unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
    def test_bound_plan_is_injected_into_stage_proposal(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, cfg = self._bound_controller(root, run_id="gap1-bound")

            plan = root / "gap1-bound.acquisition_plan.json"
            plan.write_text(json.dumps({
                "schema_version": 1, "pool_path": "pool.json",
                "selected_source_global_indices": [0, 1], "n_selected": 2}))
            c.bind_new_input(plan)

            proposal, _role = _proposal_from_stage(c, "acquisition", cfg["stages"][0])
            params = proposal["parameters"]
            self.assertIn("acquisition_plan_path", params,
                          "a bound AcquisitionPlan must be auto-resolved into the stage proposal")
            injected = Path(params["acquisition_plan_path"]).resolve()
            self.assertTrue(str(injected).endswith("acquisition_plan.json"))
            # The injected path is a genuine run-bound input (snapshot/source), never a fabricated
            # or workflow.yaml-hard-coded path.
            bound_paths = set()
            for rec in c.state["inputs"]:
                for key in ("snapshot", "source"):
                    if rec.get(key):
                        bound_paths.add(str(Path(rec[key]).resolve()))
            self.assertIn(str(injected), bound_paths)

    @unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
    def test_no_bound_plan_injects_nothing(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, cfg = self._bound_controller(root, run_id="gap1-none")
            proposal, _role = _proposal_from_stage(c, "acquisition", cfg["stages"][0])
            self.assertNotIn("acquisition_plan_path", proposal["parameters"])
            self.assertNotIn("acquisition_plan", proposal["parameters"])


class CorrectionScopeGuardrailTests(unittest.TestCase):
    """(4) no material-specific category name is encoded; (6) closure machinery unchanged."""

    def test_no_material_family_names_encoded_in_the_fix(self):
        # The objective-conditioned eligibility mask lives ENTIRELY in the default acquisition
        # provider; if any hard-coded material family leaked into framework logic it would be here.
        rel = "runtimes/pydantic_ai/default_acquisition_provider.py"
        text = (ROOT / rel).read_text().lower()
        for token in _FORBIDDEN_MATERIAL_TOKENS:
            self.assertNotIn(
                token, text,
                f"{rel} must not encode the material-specific token {token!r}; the "
                "eligibility mask must be produced generically from opaque category strings")

    def test_eligibility_mask_keys_on_opaque_category_split(self):
        """The mask derives families by peeling the frame index off an opaque ``category#index``
        id -- it never matches any material name."""
        from runtimes.pydantic_ai.default_acquisition_provider import (
            FrameworkDefaultAcquisitionProvider as P)
        self.assertEqual(P._source_category_of("anything_opaque#123"), "anything_opaque")
        self.assertEqual(P._source_category_of("famA#0"), "famA")

    @unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
    def test_three_judge_committee_unchanged(self):
        from framework_v2.review_spec import default_stage_review_specs
        from framework_v2.stages import CanonicalStage as S
        spec = default_stage_review_specs()[S.ACQUISITION.value]
        self.assertEqual(set(spec.lens_ids), _CANONICAL_LENS_IDS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
