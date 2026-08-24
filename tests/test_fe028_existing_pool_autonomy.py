"""FE-028 -- existing-pool acquisition autonomy: deterministic regression tests A-J.

These tests pin the FE-028 semantic correction: a SATURATED target pool with an eligible existing
population and a labeling population still to be chosen is a NORMAL, fully-DECIDABLE acquisition
state that resolves to EXISTING_POOL_SELECTION -- NOT StrategyUndecidable. The labeling-population
SIZE is a deterministic OUTPUT (farthest-point novelty knee), never a human/LLM input; an optional
target/ceiling acts only as an upper bound. Teacher dynamics capability is DERIVED from the run's
own bound Teacher-calculator config, never hardcoded. The classic LOCAL_PERTURBATION /
TEACHER_DRIVEN_MD routes, the plan-contract projection invariant, and genuine undecidability are all
preserved.

All tests are deterministic and free of GPU/Teacher/network. Runtime-dependent probes (which need
the optional pydantic-ai runtime deps) skip cleanly on a core-only install, matching every other
pydantic-ai test in this suite.
"""
import unittest

from framework_v2.acquisition.contracts import (
    AcquisitionPhase,
    AcquisitionPlanV2,
    AcquisitionStrategyKind,
    BackendCapabilityRecord,
    CandidateSelectionResult,
    CoverageGapAnalysis,
    ProtectedDisjointnessReport,
    RegimeCoverage,
    RelevanceRole,
    SourceAndCapabilityInventory,
    SourceCategoryRecord,
    TeacherCapabilityRecord,
)
from framework_v2.acquisition.generic_coverage import (
    FrameworkSizingParams,
    recommend_labeling_population_sizing,
)
from framework_v2.acquisition.strategy import (
    StrategyEvidence,
    StrategyUndecidable,
    select_strategy,
)

# Backend ids per kind (arbitrary, stable).
_BID = {
    AcquisitionStrategyKind.EXISTING_POOL_SELECTION: "existing_pool_selection.ase",
    AcquisitionStrategyKind.LOCAL_PERTURBATION: "augment_atoms.ase",
    AcquisitionStrategyKind.TEACHER_DRIVEN_MD: "teacher_dynamics.ase",
    AcquisitionStrategyKind.STRUCTURE_GENERATION: "structure_gen.ase",
}


def _inventory(*, kinds, can_drive_dynamics=False):
    """Inventory whose feasible backends are exactly ``kinds`` (a list of AcquisitionStrategyKind)."""
    backends = [
        BackendCapabilityRecord(backend_id=_BID[k], strategy_kind=k, feasible=True)
        for k in kinds
    ]
    return SourceAndCapabilityInventory(
        inventory_id="inv",
        objective_sha256="obj",
        sources=[SourceCategoryRecord(
            category="bulk", n_items=100, has_metadata=False, provenance_class="sanitized_pool")],
        backends=backends,
        teacher=TeacherCapabilityRecord(
            teacher_id="t", can_label=True, can_drive_dynamics=can_drive_dynamics),
    )


def _coverage(*, saturated, gap_score):
    """Single CORE_TARGET regime coverage with the given saturation state."""
    return CoverageGapAnalysis(
        analysis_id="cov",
        phase=AcquisitionPhase.INITIAL,
        target_regime_model_sha256="trm",
        region_resolution_sha256="rr",
        per_regime=[RegimeCoverage(
            regime_id="core", relevance_role=RelevanceRole.CORE_TARGET,
            current_count=40, saturation=(1.0 if saturated else 0.2),
            novelty_headroom=(0.0 if saturated else 0.9),
            gap_score=gap_score, saturated=saturated)],
    )


def _evidence(**kw):
    base = dict(
        pool_covers_gaps=False, parents_reach_gaps=False,
        gaps_require_new_configurations=False, seed_structures_exist=True,
        mixed_backends_required=False)
    base.update(kw)
    return StrategyEvidence(**base)


class StrategySemanticsTests(unittest.TestCase):
    """Tests A, G, H, I -- the strategy-selection semantic correction."""

    def test_A_saturated_pool_resolves_to_existing_pool_selection(self):
        """A: saturated core coverage + eligible existing pool -> EXISTING_POOL_SELECTION,
        NEVER StrategyUndecidable (the FE-028 conceptual-bug correction)."""
        inv = _inventory(kinds=[AcquisitionStrategyKind.EXISTING_POOL_SELECTION])
        cov = _coverage(saturated=True, gap_score=0.0)  # -> no unsaturated core gaps
        self.assertEqual(cov.unsaturated_core_gaps(), [])
        strat = select_strategy(
            strategy_id="s", inventory=inv, coverage=cov,
            evidence=_evidence(pool_covers_gaps=False, seed_structures_exist=True))
        self.assertEqual(strat.kind, AcquisitionStrategyKind.EXISTING_POOL_SELECTION)
        self.assertIn("no new-configuration generation required", strat.rationale)

    def test_A2_pool_covers_unsaturated_gaps_resolves_to_existing_pool(self):
        """A': existing pool coverage closes still-unsaturated core gaps -> EXISTING_POOL_SELECTION."""
        inv = _inventory(kinds=[AcquisitionStrategyKind.EXISTING_POOL_SELECTION])
        cov = _coverage(saturated=False, gap_score=1.0)
        strat = select_strategy(
            strategy_id="s", inventory=inv, coverage=cov,
            evidence=_evidence(pool_covers_gaps=True))
        self.assertEqual(strat.kind, AcquisitionStrategyKind.EXISTING_POOL_SELECTION)

    def test_G_unsaturated_gaps_requiring_new_configs_route_to_teacher_md(self):
        """G: unsaturated core gaps that require genuinely new configurations + a dynamics-capable
        Teacher -> TEACHER_DRIVEN_MD (case C derivation)."""
        inv = _inventory(kinds=[AcquisitionStrategyKind.TEACHER_DRIVEN_MD], can_drive_dynamics=True)
        cov = _coverage(saturated=False, gap_score=1.0)
        strat = select_strategy(
            strategy_id="s", inventory=inv, coverage=cov,
            evidence=_evidence(gaps_require_new_configurations=True))
        self.assertEqual(strat.kind, AcquisitionStrategyKind.TEACHER_DRIVEN_MD)

    def test_I_local_perturbation_route_preserved(self):
        """I: existing structures reach unsaturated gaps by local perturbation -> LOCAL_PERTURBATION
        (classic route unchanged by the FE-028 addition)."""
        inv = _inventory(kinds=[AcquisitionStrategyKind.LOCAL_PERTURBATION])
        cov = _coverage(saturated=False, gap_score=1.0)
        strat = select_strategy(
            strategy_id="s", inventory=inv, coverage=cov,
            evidence=_evidence(parents_reach_gaps=True, gaps_require_new_configurations=False))
        self.assertEqual(strat.kind, AcquisitionStrategyKind.LOCAL_PERTURBATION)

    def test_I2_teacher_md_requires_dynamics_capability(self):
        """I': TEACHER_DRIVEN_MD is admissible ONLY when the Teacher can drive dynamics; without it
        the same gap state is genuinely undecidable (fail-closed, not fabricated)."""
        inv = _inventory(kinds=[AcquisitionStrategyKind.TEACHER_DRIVEN_MD], can_drive_dynamics=False)
        cov = _coverage(saturated=False, gap_score=1.0)
        with self.assertRaises(StrategyUndecidable):
            select_strategy(
                strategy_id="s", inventory=inv, coverage=cov,
                evidence=_evidence(gaps_require_new_configurations=True))

    def test_H_genuine_undecidability_still_fails_closed(self):
        """H: when gaps genuinely need new configurations but only LOCAL_PERTURBATION exists (and no
        existing-pool coverage), the state is genuinely undecidable -> StrategyUndecidable. FE-028
        narrows StrategyUndecidable to the truly-irreducible case, it does not remove it."""
        inv = _inventory(kinds=[AcquisitionStrategyKind.LOCAL_PERTURBATION])
        cov = _coverage(saturated=False, gap_score=1.0)
        with self.assertRaises(StrategyUndecidable):
            select_strategy(
                strategy_id="s", inventory=inv, coverage=cov,
                evidence=_evidence(parents_reach_gaps=False,
                                   gaps_require_new_configurations=True))

    def test_H2_saturated_without_existing_pool_backend_still_undecidable(self):
        """H': saturated coverage but NO existing-pool backend feasible and no seeds-in-gap path is a
        genuine gap -- proves the EXISTING_POOL resolution is gated on the backend actually existing,
        not assumed."""
        inv = _inventory(kinds=[AcquisitionStrategyKind.LOCAL_PERTURBATION])
        cov = _coverage(saturated=True, gap_score=0.0)
        with self.assertRaises(StrategyUndecidable):
            select_strategy(
                strategy_id="s", inventory=inv, coverage=cov,
                evidence=_evidence(seed_structures_exist=True))


class LabelingSizingTests(unittest.TestCase):
    """Tests B, C, E -- deterministic labeling-population sizing (size as OUTPUT)."""

    # A pool with a clear novelty knee: one far point, then a tight cluster.
    KNEE_VECTORS = [[0.0], [10.0], [10.1], [10.2], [10.3]]
    # A pool whose marginal novelty never plateaus below the knee -> full-population fallback.
    NO_PLATEAU_VECTORS = [[0.0], [100.0], [50.0], [25.0], [75.0]]

    def test_B_size_is_output_without_any_human_N(self):
        """B: with target_labeled_population=None the recommended size is DERIVED from the novelty
        knee (a real subset, k<n), and no target constraint was applied."""
        ev = recommend_labeling_population_sizing(
            self.KNEE_VECTORS, params=FrameworkSizingParams(), sizing_id="sz",
            coverage_gap_sha256="c", protected_excluded_count=0,
            target_labeled_population=None, max_teacher_label_calls=None)
        self.assertFalse(ev.target_constraint_applied)
        self.assertFalse(ev.ceiling_clamped)
        self.assertFalse(ev.fallback_full_population)
        self.assertEqual(ev.eligible_population_size, 5)
        self.assertEqual(ev.recommended_population_size, 2)  # seed + one high-novelty pick
        self.assertLess(ev.recommended_population_size, ev.eligible_population_size)
        self.assertEqual(len(ev.selected_positions), ev.recommended_population_size)

    def test_C_compute_ceiling_caps_the_size(self):
        """C: an optional max_teacher_label_calls acts only as an upper bound and clamps the derived
        size; an optional target_labeled_population also only bounds from above."""
        ev = recommend_labeling_population_sizing(
            self.KNEE_VECTORS, params=FrameworkSizingParams(), sizing_id="sz",
            coverage_gap_sha256="c", target_labeled_population=None,
            max_teacher_label_calls=1)
        self.assertEqual(ev.recommended_population_size, 1)
        self.assertTrue(ev.ceiling_clamped)

        ev2 = recommend_labeling_population_sizing(
            self.NO_PLATEAU_VECTORS, params=FrameworkSizingParams(), sizing_id="sz",
            coverage_gap_sha256="c", target_labeled_population=3)
        # natural size would be full (5); the target upper-bound pulls it to 3.
        self.assertEqual(ev2.recommended_population_size, 3)
        self.assertTrue(ev2.target_constraint_applied)

    def test_E_full_population_conservative_fallback(self):
        """E: when no defensible smaller subset exists (novelty never plateaus, or too few members)
        the conservative fallback is the FULL eligible population -- never a human prompt."""
        ev = recommend_labeling_population_sizing(
            self.NO_PLATEAU_VECTORS, params=FrameworkSizingParams(), sizing_id="sz",
            coverage_gap_sha256="c")
        self.assertTrue(ev.fallback_full_population)
        self.assertEqual(ev.recommended_population_size, ev.eligible_population_size)

        ev_tiny = recommend_labeling_population_sizing(
            [[0.0], [1.0]], params=FrameworkSizingParams(), sizing_id="sz",
            coverage_gap_sha256="c")
        self.assertTrue(ev_tiny.fallback_full_population)
        self.assertEqual(ev_tiny.recommended_population_size, 2)


class ProtectedFilteringTests(unittest.TestCase):
    """Test D -- protected-reference filtering / disjointness is fail-closed."""

    def test_D_sizing_operates_on_protected_excluded_pool(self):
        """D: sizing operates on the protected-EXCLUDED eligible vectors and records how many
        protected structures were removed; eligible_population_size counts only eligibles."""
        ev = recommend_labeling_population_sizing(
            [[0.0], [10.0], [10.1]], params=FrameworkSizingParams(), sizing_id="sz",
            coverage_gap_sha256="c", protected_excluded_count=7)
        self.assertEqual(ev.eligible_population_size, 3)
        self.assertEqual(ev.protected_excluded_count, 7)

    def test_D2_disjointness_must_pass_to_bind(self):
        """D': a non-PASS protected-disjointness report cannot bind a selection (fail-closed), and a
        PASS report with a non-zero overlap count is itself rejected as inconsistent."""
        with self.assertRaises(Exception):
            ProtectedDisjointnessReport(status="PASS", n_checked=3, n_overlaps=1)

        bad = ProtectedDisjointnessReport(status="FAIL", n_checked=3, n_overlaps=2)
        with self.assertRaises(Exception):
            CandidateSelectionResult(
                selection_id="sel", generation_result_sha256="g", selector="fps",
                selected_candidate_ids=["a", "b"], disjointness_report=bad)


class PlanContractRegressionTests(unittest.TestCase):
    """Test J -- plan-contract projection invariant + strategy-kind recognition preserved."""

    def _plan(self, **projection):
        base = dict(
            plan_id="p", objective_sha256="o", inventory_sha256="i",
            target_regime_model_sha256="t", region_resolution_sha256="r",
            coverage_gap_sha256="c", strategy_sha256="s",
            generation_result_sha256="g", selection_result_sha256="sel",
            labeling_request_sha256="lr", phase=AcquisitionPhase.INITIAL)
        base.update(projection)
        return AcquisitionPlanV2(**base)

    def test_J_existing_pool_projection_is_a_valid_single_projection(self):
        """J: an AcquisitionPlanV2 carrying ONLY an existing_pool_projection satisfies the
        exactly-one-projection invariant and round-trips through the contract."""
        proj = {"schema_version": 1, "pool_path": "pool.json",
                "selected_source_global_indices": [0, 4], "n_selected": 2}
        plan = self._plan(existing_pool_projection=proj)
        self.assertEqual(plan.existing_pool_projection["n_selected"], 2)
        again = AcquisitionPlanV2.model_validate(plan.model_dump(mode="json"))
        self.assertEqual(again.existing_pool_projection, proj)

    def test_J2_two_projections_rejected(self):
        """J': the exactly-one-projection invariant still rejects a plan that sets both an
        existing-pool and a legacy projection (the FE-028 field is additive, not a loophole)."""
        with self.assertRaises(Exception):
            self._plan(existing_pool_projection={"pool_path": "x"},
                       legacy_projection={"n_parents": 1})

    def test_J3_no_projection_rejected(self):
        with self.assertRaises(Exception):
            self._plan()

    def test_J4_existing_pool_selection_is_a_recognized_strategy_kind(self):
        self.assertEqual(
            AcquisitionStrategyKind("EXISTING_POOL_SELECTION"),
            AcquisitionStrategyKind.EXISTING_POOL_SELECTION)


class TeacherCapabilityProbeTests(unittest.TestCase):
    """Test F -- production Teacher dynamics capability is DERIVED, not hardcoded."""

    def _probe(self):
        try:
            from runtimes.pydantic_ai.default_acquisition_provider import (
                _probe_teacher_dynamics_capability)
        except ModuleNotFoundError:
            self.skipTest("pydantic (optional runtime dep) not installed")
        return _probe_teacher_dynamics_capability

    def _controller(self, workflow_config):
        return type("C", (), {"state": {"workflow_config": workflow_config}})()

    def test_F_calculator_config_yields_dynamics_capable(self):
        """F: a bound Teacher config that declares a constructible ASE calculator -> can drive
        dynamics True (derived from the run's own evidence)."""
        import tempfile, os
        probe = self._probe()
        d = tempfile.mkdtemp(prefix="fe028_probe_")
        tc = os.path.join(d, "teacher.yaml")
        with open(tc, "w") as fh:
            fh.write("calculator:\n  module: ase.calculators.emt\n  class: EMT\n")
        wf = os.path.join(d, "workflow.yaml")
        with open(wf, "w") as fh:
            fh.write(f"stages:\n  - name: acquisition\n    params:\n      teacher_config: {tc}\n")
        can, why = probe(self._controller(wf))
        self.assertTrue(can, why)

    def test_F2_no_calculator_block_yields_labeling_only(self):
        """F': a bound Teacher config with NO calculator block -> can drive dynamics False (labeling
        only). The probe never returns a hardcoded True."""
        import tempfile, os
        probe = self._probe()
        d = tempfile.mkdtemp(prefix="fe028_probe_")
        tc = os.path.join(d, "teacher.yaml")
        with open(tc, "w") as fh:
            fh.write("model_path: /some/teacher.pth\n")
        wf = os.path.join(d, "workflow.yaml")
        with open(wf, "w") as fh:
            fh.write(f"stages:\n  - name: acquisition\n    params:\n      teacher_config: {tc}\n")
        can, why = probe(self._controller(wf))
        self.assertFalse(can, why)

    def test_F3_no_bound_teacher_config_yields_false(self):
        """F'': no bound teacher_config at all -> can drive dynamics False (not admissible)."""
        import tempfile, os
        probe = self._probe()
        d = tempfile.mkdtemp(prefix="fe028_probe_")
        wf = os.path.join(d, "workflow.yaml")
        with open(wf, "w") as fh:
            fh.write("stages:\n  - name: acquisition\n    params: {}\n")
        can, why = probe(self._controller(wf))
        self.assertFalse(can, why)


class EndToEndExistingPoolExecTests(unittest.TestCase):
    """Test K -- manifest-based existing-pool projection end-to-end through the SELECT executor:
    deterministic sizing -> FPS selection -> projection assembly -> deterministic validation ->
    executor with prior-label stripping + data_coverage-ready manifest. Skips on core-only installs
    (needs ase + the pydantic-ai runtime)."""

    def _deps(self):
        try:
            import numpy as np  # noqa: F401
            from ase.io import read, write  # noqa: F401
            from framework_v2.acquisition.selection import select_candidates  # noqa: F401
            from framework_v2.acquisition.contracts import (  # noqa: F401
                CandidateGenerationResult, GenerationProvenance, ProtectedDisjointnessReport)
            from framework_v2.acquisition.plan_assembly import build_existing_pool_projection  # noqa: F401
            from runtimes.pydantic_ai.executors import (  # noqa: F401
                _validate_existing_pool_plan, _is_existing_pool_plan,
                _exec_select_existing_pool, _load_pool_frames_global_order)
        except ModuleNotFoundError as e:
            self.skipTest(f"optional dep not installed: {e}")

    def test_K_existing_pool_exec_strips_labels_and_writes_coverage_manifest(self):
        self._deps()
        import json, os, tempfile
        import numpy as np
        from ase import Atoms
        from ase.io import read, write
        from framework_v2.acquisition.generic_coverage import (
            FrameworkSizingParams, recommend_labeling_population_sizing)
        from framework_v2.acquisition.selection import select_candidates
        from framework_v2.acquisition.contracts import (
            CandidateGenerationResult, GenerationProvenance, ProtectedDisjointnessReport)
        from framework_v2.acquisition.plan_assembly import build_existing_pool_projection
        from runtimes.pydantic_ai.executors import (
            _validate_existing_pool_plan, _is_existing_pool_plan,
            _exec_select_existing_pool, _load_pool_frames_global_order)

        def _mk(n, spread, seed):
            rng = np.random.default_rng(seed)
            at = Atoms("Si" * (n // 2) + "O" * (n - n // 2),
                       positions=rng.random((n, 3)) * spread, cell=[10, 10, 10], pbc=True)
            at.info["energy"] = -123.4          # a prior label that MUST be stripped
            at.arrays["forces"] = rng.random((n, 3))
            return at

        d = tempfile.mkdtemp(prefix="fe028_k_")
        inputs = os.path.join(d, "inputs")
        os.makedirs(inputs, exist_ok=True)
        cat_a = [_mk(6, 3.0, s) for s in range(4)]
        cat_b = [_mk(8, 6.0, 100 + s) for s in range(3)]
        write(os.path.join(inputs, "bulk_amo.sanitized.xyz"), cat_a, format="extxyz")
        write(os.path.join(inputs, "surface.sanitized.xyz"), cat_b, format="extxyz")
        manifest = {"total_frames": 7, "sanitized_pool_manifest_sha256": "deadbeef",
                    "categories": [
                        {"category": "bulk_amo", "sanitized_file": "bulk_amo.sanitized.xyz", "n_frames": 4},
                        {"category": "surface", "sanitized_file": "surface.sanitized.xyz", "n_frames": 3}]}
        manifest_path = os.path.join(inputs, "013-sanitized_pool_manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh)

        ordered, _ = _load_pool_frames_global_order(manifest_path)
        self.assertEqual(len(ordered), 7)
        item_ids, per_cat = [], {}
        for gi, cat, _ in ordered:
            j = per_cat.get(cat, 0)
            item_ids.append(f"{cat}#{j}")
            per_cat[cat] = j + 1
        vectors = [[float(len(at))] for _, _, at in ordered]

        sizing = recommend_labeling_population_sizing(
            vectors, params=FrameworkSizingParams(), sizing_id="sz",
            coverage_gap_sha256="abc", protected_excluded_count=0,
            target_labeled_population=None, max_teacher_label_calls=None)
        k = sizing.recommended_population_size

        prov = [GenerationProvenance(candidate_id=i, strategy_kind="EXISTING_POOL_SELECTION",
                                     backend_id="existing_pool_selection.ase", parent_id=i,
                                     exploration_only=True) for i in item_ids]
        gen = CandidateGenerationResult(result_id="g", strategy_sha256="s",
                                        backend_id="existing_pool_selection.ase",
                                        candidate_ids=list(item_ids), provenance=prov,
                                        n_requested=len(item_ids), n_generated=len(item_ids),
                                        n_rejected=0)

        def _dj(ids):
            return ProtectedDisjointnessReport(status="PASS", n_checked=len(ids), n_overlaps=0,
                                               dft_labels_used_as_selection_scores=False)

        sel = select_candidates(selection_id="sel", generation_result=gen, descriptors=vectors,
                                k=k, disjointness_checker=_dj,
                                selector="farthest_point_sampling", seed_index=0)
        posmap = {iid: i for i, iid in enumerate(item_ids)}
        gidx = [posmap[c] for c in sel.selected_candidate_ids]
        parents = [item_ids[i] for i in gidx]

        proj = build_existing_pool_projection(
            pool_path=manifest_path, eligible_source_categories=["bulk_amo", "surface"],
            selected_parent_structure_ids=parents, selected_source_global_indices=gidx,
            labeling_population_sizing=sizing.model_dump(mode="json"), selection_result=sel,
            duplicate_handling="reject")
        self.assertTrue(_is_existing_pool_plan(proj))
        validated = _validate_existing_pool_plan(proj)
        self.assertEqual(validated["n_selected"], k)

        plan_path = os.path.join(d, "run.acquisition_plan.json")
        with open(plan_path, "w") as fh:
            json.dump(proj, fh, indent=2, sort_keys=True)
        out_path = os.path.join(d, "candidates.extxyz")
        exec_manifest = os.path.join(d, "acquisition_manifest.json")

        class P:
            parameters = {"acquisition_plan_path": plan_path, "out_path": out_path,
                          "manifest_path": exec_manifest, "stage": "acquisition"}

        _exec_select_existing_pool(P())
        frames = read(out_path, index=":")
        self.assertEqual(len(frames), k)
        for at in frames:
            self.assertNotIn("energy", at.info, "prior label leaked")
            self.assertNotIn("forces", at.arrays, "prior forces leaked")
            self.assertTrue(at.info["parent_structure_id"].startswith("seed-pool:"))
            self.assertIn("source_global_index", at.info)
        with open(exec_manifest) as fh:
            mf = json.load(fh)
        self.assertEqual(mf["n_frames"], k)
        self.assertTrue(isinstance(mf["elements"], list) and mf["elements"])
        self.assertFalse(mf["performs_teacher_inference"])
        self.assertFalse(mf["dft_labels_used_as_selection_scores"])


if __name__ == "__main__":
    unittest.main()
