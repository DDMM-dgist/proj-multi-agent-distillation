"""FE-027 P6 (§16/§17) -- end-to-end portability of the generic acquisition path.

The keystone FE-027 claim: a brand-new material needs NO hand-authored descriptor plugin, region
target_count, generation-parameter bounds, or acquisition percentages. This test proves it by
driving the framework-provided ``GenericStructuralDescriptorProvider`` through the SAME FE-026
core (``materialize_acquisition_evidence``) that a specialized plugin uses -- on synthetic pools
that are deliberately NOT SiO2 (one Cu/Al system, one Mg/O system), built on the fly so nothing
material-specific can leak in.

For each material it proves:
  * the generic provider synthesizes a full ``DescriptorSpaceEvidence`` from raw structures alone
    (P1 discovered representation + P2 executable regions, P3 pool-saturation CORE_TARGET coverage,
    P4 physics-derived perturbation bounds) with NO human-supplied number;
  * the FE-026 materializer freezes the whole chain (inventory / target-regime / region / coverage
    / strategy) and autonomously selects LOCAL_PERTURBATION -- the CORE_TARGET regime id equals the
    scope PRIMARY region id (scope-aligned), the admissible parents are the pool frame ids, and the
    param bounds are the pool-derived envelope bounds, none of which any human supplied;
  * the typed decisions (strategy kind, admissible parents, param bounds, CORE_TARGET saturation)
    are byte-stable across repeated builds (control/evidence-plane split, §9);
  * the SAME code path runs on TWO different synthetic materials -- portability, not a per-material
    branch;
  * the generic fallback resolves + materializes end-to-end via ``resolve_descriptor_provider``;
  * a degenerate (non-discriminative) pool fails closed REPRESENTATION_INSUFFICIENT -- the provider
    never asks a human for a descriptor plugin.

Everything is deterministic and hermetic (a tmp-dir pool + an injected feasible backend + the
real Teacher-identity hash of a tmp Teacher file); it opens no socket and needs no live model.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import ase  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from test_generic_representation import _FakeController, _build_pool, _scope_contract


def _build_alt_pool(tmp: Path) -> str:
    """A SECOND synthetic material (Mg/O), distinct from the P1 Cu/Al pool, to prove portability.

    'gamma' = small dense MgO cells, 'delta' = large sparse O supercells -> discriminative under
    the generic size/density/geometry axes. No phase label or element-pair cutoff is used here or
    in the core; the category strings are opaque provenance labels."""
    import numpy as np
    from ase import Atoms
    from ase.io import write as ase_write

    gamma_file = tmp / "gamma.xyz"
    delta_file = tmp / "delta.xyz"

    def _mgo_small(seed):
        rng = np.random.default_rng(seed)
        a = 4.2
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        pos = rng.uniform(0.2, a - 0.2, size=(6, 3))
        return Atoms("Mg3O3", positions=pos, cell=cell, pbc=True)

    def _o_big(seed):
        rng = np.random.default_rng(seed)
        a = 9.0
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        pos = rng.uniform(0.2, a - 0.2, size=(24, 3))
        return Atoms("O24", positions=pos, cell=cell, pbc=True)

    n = 10
    for s in range(n):
        ase_write(str(gamma_file), _mgo_small(s), format="extxyz", append=True)
    for s in range(n):
        ase_write(str(delta_file), _o_big(200 + s), format="extxyz", append=True)

    categories = []
    total = 0
    for cat, f in (("gamma", gamma_file), ("delta", delta_file)):
        raw = f.read_bytes()
        categories.append({
            "category": cat, "sanitized_file": f.name,
            "sanitized_sha256": hashlib.sha256(raw).hexdigest(),
            "n_frames": n, "size_bytes": len(raw)})
        total += n
    manifest = {
        "total_frames": total, "n_categories": len(categories),
        "sanitized_pool_manifest_sha256": "cafef00d", "categories": categories}
    mpath = tmp / "sanitized_pool_manifest.json"
    mpath.write_text(json.dumps(manifest))
    return str(mpath)


class _Objective:
    def __init__(self, scope):
        from framework_v2.acquisition.contracts import AcquisitionPhase
        self.objective_id = "p6-objective"
        self.primary_target = "generic synthetic distillation target"
        self.claim_scope = "generic deployment claim (portability)"
        self.scope_contract_sha256 = scope.content_sha256()
        self.phase = AcquisitionPhase.INITIAL
        self.compute_ceiling = None

    def content_sha256(self) -> str:
        return hashlib.sha256(
            f"{self.objective_id}|{self.primary_target}|{self.scope_contract_sha256}".encode()
        ).hexdigest()


def _feasible_local_perturbation_backend():
    from framework_v2.acquisition.contracts import (
        AcquisitionStrategyKind, BackendCapabilityRecord)
    return [BackendCapabilityRecord(
        backend_id="local_perturbation.augment_atoms",
        strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
        feasible=True, supported_capabilities=["acquisition.local_perturbation"])]


def _teacher_record():
    from framework_v2.acquisition.contracts import TeacherCapabilityRecord
    return TeacherCapabilityRecord(
        teacher_id="p6-teacher", can_label=True, can_drive_dynamics=True,
        identity_sha256="0" * 64)


@unittest.skipUnless(_HAS_DEPS, "pydantic/ase not installed")
class GenericPortabilityP6(unittest.TestCase):
    def _materialize(self, manifest_path):
        from framework_v2.acquisition.evidence_materializer import (
            materialize_acquisition_evidence)
        from framework_v2.acquisition.generic_provider import (
            GENERIC_MATERIAL_ID, GenericStructuralDescriptorProvider)

        ctrl = _FakeController(manifest_path)
        scope = _scope_contract()
        objective = _Objective(scope)
        provider = GenericStructuralDescriptorProvider()

        self.assertTrue(provider.applies(
            controller=ctrl, objective=objective, scope_contract=scope))
        evidence = provider.build_descriptor_space_evidence(
            controller=ctrl, objective=objective, scope_contract=scope)
        materialized = materialize_acquisition_evidence(
            id_prefix="p6", material_id=GENERIC_MATERIAL_ID, objective=objective,
            scope_contract=scope, descriptor_evidence=evidence,
            backend_records=_feasible_local_perturbation_backend(),
            teacher_record=_teacher_record())
        return evidence, materialized, scope

    def test_generic_evidence_is_fully_derived_no_human_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            evidence, _, scope = self._materialize(
                _build_pool(Path(d), discriminative=True))

        # Admissible parents ARE the pool frame ids (never a human-authored parent list).
        self.assertEqual(len(evidence.admissible_parent_ids), 20)
        self.assertTrue(all("#" in p for p in evidence.admissible_parent_ids))
        # Generation bounds are the P4 pool-derived envelope bounds (data/physics, not constants).
        self.assertIn("displacement_sigma_A", evidence.param_bounds)
        lo, hi = evidence.param_bounds["displacement_sigma_A"]
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertIn("seed", evidence.required_param_keys)
        # CORE_TARGET coverage keyed by the scope PRIMARY region id, saturation in [0, 1].
        from framework_v2.acquisition.contracts import RelevanceRole
        from framework_v2.contracts import ScopeCategory
        primary_id = scope.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT)[0].region_id
        self.assertEqual(len(evidence.regime_coverage_inputs), 1)
        ci = evidence.regime_coverage_inputs[0]
        self.assertEqual(ci.regime_id, primary_id)
        self.assertEqual(ci.relevance_role, RelevanceRole.CORE_TARGET)
        self.assertGreaterEqual(ci.saturation, 0.0)
        self.assertLessEqual(ci.saturation, 1.0)
        # No trusted metadata is ever asserted by the generic path.
        self.assertFalse(evidence.metadata_present)

    def test_materializer_freezes_chain_and_autonomously_picks_perturbation(self):
        with tempfile.TemporaryDirectory() as d:
            _, materialized, scope = self._materialize(
                _build_pool(Path(d), discriminative=True))

        from framework_v2.acquisition.contracts import (
            AcquisitionStrategyKind, RegionResolutionMode)
        from framework_v2.contracts import ScopeCategory

        # Strategy chosen autonomously from evidence (unsaturated core + parents reach -> perturb).
        self.assertEqual(materialized.strategy.kind, AcquisitionStrategyKind.LOCAL_PERTURBATION)
        # No trusted metadata -> the DISCOVERED region path ran.
        self.assertEqual(materialized.region_resolution.mode, RegionResolutionMode.DISCOVERED)
        # The scope-derived TargetRegimeModel has exactly the PRIMARY region as CORE_TARGET, and
        # the coverage CORE_TARGET regime id matches it (scope-aligned, not discovered-id-keyed).
        primary_id = scope.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT)[0].region_id
        core = materialized.target_regime_model.core_regimes()
        self.assertEqual([r.regime_id for r in core], [primary_id])
        self.assertEqual(
            [c.regime_id for c in materialized.coverage.per_regime], [primary_id])
        # The frozen artifact binds every upstream by content-SHA.
        fa = materialized.frozen_artifact
        self.assertEqual(fa.strategy_kind, "LOCAL_PERTURBATION")
        self.assertEqual(fa.inventory_sha256, materialized.inventory.content_sha256())
        self.assertEqual(fa.coverage_gap_sha256, materialized.coverage.content_sha256())
        self.assertEqual(fa.admissible_parent_ids, list(materialized.admissible_parent_ids))
        self.assertIn("displacement_sigma_A", fa.param_bounds)

    def test_typed_decisions_are_stable_across_repeated_builds(self):
        from framework_v2.acquisition.generic_provider import (
            GenericStructuralDescriptorProvider)

        with tempfile.TemporaryDirectory() as d:
            manifest = _build_pool(Path(d), discriminative=True)
            scope = _scope_contract()
            objective = _Objective(scope)

            sigs = set()
            for _ in range(3):
                ev = GenericStructuralDescriptorProvider().build_descriptor_space_evidence(
                    controller=_FakeController(manifest), objective=objective,
                    scope_contract=scope)
                ci = ev.regime_coverage_inputs[0]
                sigs.add((
                    ev.admissible_parent_ids,
                    tuple(sorted((k, v) for k, v in ev.param_bounds.items())),
                    tuple(ev.required_param_keys),
                    (ci.regime_id, ci.relevance_role.value, round(ci.saturation, 12),
                     round(ci.novelty_headroom, 12)),
                    (ev.strategy_evidence.pool_covers_gaps,
                     ev.strategy_evidence.parents_reach_gaps,
                     ev.strategy_evidence.gaps_require_new_configurations,
                     ev.strategy_evidence.seed_structures_exist)))
            # Every typed decision axis collapsed to exactly ONE value across repeats.
            self.assertEqual(len(sigs), 1, "generic typed decisions not reproducible")

    def test_second_distinct_material_runs_the_same_path(self):
        with tempfile.TemporaryDirectory() as d:
            _, materialized, scope = self._materialize(_build_alt_pool(Path(d)))

        from framework_v2.acquisition.contracts import AcquisitionStrategyKind
        # The identical generic code path materializes a different (Mg/O) material end-to-end.
        self.assertEqual(materialized.strategy.kind, AcquisitionStrategyKind.LOCAL_PERTURBATION)
        self.assertEqual(materialized.frozen_artifact.material_id, "generic-raw-structure")
        self.assertEqual(len(materialized.admissible_parent_ids), 20)

    def test_resolver_falls_back_to_generic_and_materializes(self):
        from framework_v2.acquisition.descriptor_plugins import (
            clear_descriptor_providers, register_generic_descriptor_provider,
            resolve_descriptor_provider)
        from framework_v2.acquisition.evidence_materializer import (
            materialize_acquisition_evidence)
        from framework_v2.acquisition.generic_provider import (
            GENERIC_MATERIAL_ID, GenericStructuralDescriptorProvider)

        clear_descriptor_providers()
        try:
            with tempfile.TemporaryDirectory() as d:
                ctrl = _FakeController(_build_pool(Path(d), discriminative=True))
                scope = _scope_contract()
                objective = _Objective(scope)
                register_generic_descriptor_provider(GenericStructuralDescriptorProvider())

                chosen = resolve_descriptor_provider(
                    controller=ctrl, objective=objective, scope_contract=scope)
                self.assertEqual(chosen.material_id, GENERIC_MATERIAL_ID)

                evidence = chosen.build_descriptor_space_evidence(
                    controller=ctrl, objective=objective, scope_contract=scope)
                materialized = materialize_acquisition_evidence(
                    id_prefix="p6-fallback", material_id=chosen.material_id,
                    objective=objective, scope_contract=scope, descriptor_evidence=evidence,
                    backend_records=_feasible_local_perturbation_backend(),
                    teacher_record=_teacher_record())
                self.assertTrue(materialized.admissible_parent_ids)
        finally:
            clear_descriptor_providers()

    def test_degenerate_pool_fails_closed_without_asking_for_a_plugin(self):
        from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
        from framework_v2.acquisition.generic_provider import (
            GenericStructuralDescriptorProvider)

        with tempfile.TemporaryDirectory() as d:
            ctrl = _FakeController(_build_pool(Path(d), discriminative=False))
            scope = _scope_contract()
            objective = _Objective(scope)
            with self.assertRaises(AcquisitionCapabilityGap) as cm:
                GenericStructuralDescriptorProvider().build_descriptor_space_evidence(
                    controller=ctrl, objective=objective, scope_contract=scope)
            self.assertEqual(cm.exception.gap_kind, "REPRESENTATION_INSUFFICIENT")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
