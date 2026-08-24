"""FE-027 P1 -- the GENERIC, material-agnostic raw-structure representation path.

These tests prove the keystone portability claim of FE-027: the framework can synthesize a
usable, discriminative structural representation from RAW structural facts alone (species /
positions / cell / pbc) for a material it has never seen, with NO material name, phase label,
or element-pair cutoff anywhere in the core. The synthetic pool used here is a Cu/Al system --
deliberately NOT SiO2 -- built on the fly in a tmp dir, so nothing SiO2-specific can leak in.

Coverage:
  * schema-detection pool location + raw loading + generic per-frame feature computation;
  * comparative adequacy PASS on a discriminative pool;
  * fail-closed REPRESENTATION_INSUFFICIENT + recovery routing on a degenerate pool;
  * resolver priority: a specialized plugin always wins over the generic fallback;
  * a source-scan guard proving no material-specific tokens are hard-coded in the core logic.
"""
from __future__ import annotations

import io
import json
import tempfile
import tokenize
import unittest
from pathlib import Path

try:
    import pydantic  # noqa: F401
    import ase  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False


def _scope_contract():
    from framework_v2.contracts import (
        DeploymentScopeContract, ScopeCategory, ScopeRegion)
    return DeploymentScopeContract(
        contract_id="p1-generic-scope",
        objective="generic raw-structure representation deployment scope",
        regions=[ScopeRegion(
            region_id="primary", category=ScopeCategory.PRIMARY_DEPLOYMENT,
            membership_rule="descriptor in the primary deployment set")])


class _FakeController:
    """Minimal stand-in for the frozen workflow controller: only ``.state`` is read."""

    def __init__(self, manifest_path: str, run_id: str = "p1-run"):
        self.state = {"inputs": [{"snapshot": manifest_path}], "run_id": run_id}


def _write_frame(path_list, atoms):
    from ase.io import write as ase_write
    # extxyz preserves cell + pbc so the generic density/geometry axes are computable.
    ase_write(path_list, atoms, format="extxyz", append=True)


def _build_pool(tmp: Path, *, discriminative: bool) -> str:
    """Write a synthetic Cu/Al pool + a schema-valid manifest; return the manifest path.

    Discriminative pool: category 'alpha' = small dense Cu cells, 'beta' = large sparse Al
    supercells -> n_atoms/density clearly gap-split into >1 regime.
    Degenerate pool: every frame is the SAME structure -> all features identical -> a single
    regime under both the primary and the coarse alternative representation.
    """
    import numpy as np
    from ase import Atoms

    alpha_file = tmp / "alpha.xyz"
    beta_file = tmp / "beta.xyz"

    def _cu_small(seed):
        rng = np.random.default_rng(seed)
        a = 3.6
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        pos = rng.uniform(0.2, a - 0.2, size=(4, 3))
        return Atoms("Cu4", positions=pos, cell=cell, pbc=True)

    def _al_big(seed):
        rng = np.random.default_rng(seed)
        a = 8.0
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        pos = rng.uniform(0.2, a - 0.2, size=(20, 3))
        return Atoms("Al20", positions=pos, cell=cell, pbc=True)

    n = 10
    if discriminative:
        for s in range(n):
            _write_frame(str(alpha_file), _cu_small(s))
        for s in range(n):
            _write_frame(str(beta_file), _al_big(100 + s))
    else:
        fixed = _cu_small(0)
        for _ in range(n):
            _write_frame(str(alpha_file), Atoms(fixed))
        for _ in range(n):
            _write_frame(str(beta_file), Atoms(fixed))

    categories = []
    total = 0
    for cat, f in (("alpha", alpha_file), ("beta", beta_file)):
        raw = f.read_bytes()
        import hashlib
        categories.append({
            "category": cat,
            "sanitized_file": f.name,
            "sanitized_sha256": hashlib.sha256(raw).hexdigest(),
            "n_frames": n,
            "size_bytes": len(raw),
        })
        total += n

    manifest = {
        "total_frames": total,
        "n_categories": len(categories),
        "sanitized_pool_manifest_sha256": "deadbeef",
        "categories": categories,
    }
    mpath = tmp / "sanitized_pool_manifest.json"
    mpath.write_text(json.dumps(manifest))
    return str(mpath)


class _Objective:
    def __init__(self, scope):
        self.claim_scope = "generic deployment claim (portability smoke)"
        self.scope_contract_sha256 = scope.content_sha256()


@unittest.skipUnless(_HAS_DEPS, "pydantic/ase not installed")
class GenericRepresentationP1(unittest.TestCase):
    def setUp(self):
        from framework_v2.acquisition.descriptor_plugins import clear_descriptor_providers
        clear_descriptor_providers()

    def tearDown(self):
        from framework_v2.acquisition.descriptor_plugins import clear_descriptor_providers
        clear_descriptor_providers()

    def test_locate_load_and_features_from_raw_structures(self):
        from framework_v2.acquisition.generic_representation import (
            load_pool, locate_pool_manifest)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mpath = _build_pool(tmp, discriminative=True)
            ctrl = _FakeController(mpath)

            found_path, manifest = locate_pool_manifest(ctrl)
            self.assertEqual(str(found_path), mpath)
            self.assertEqual(manifest["total_frames"], 20)

            pool = load_pool(ctrl)
            self.assertEqual(pool.total_frames, 20)
            self.assertEqual(pool.per_category_counts, {"alpha": 10, "beta": 10})
            # Every frame carries the two always-computable axes; periodic frames also carry
            # density + geometry. No axis is imputed when absent.
            for f in pool.frames:
                self.assertIn("n_atoms", f.features)
                self.assertIn("max_species_fraction", f.features)
                self.assertIn("number_density_atoms_per_A3", f.features)
                self.assertIn("mean_min_neighbor_distance_A", f.features)
                self.assertGreater(f.features["max_species_fraction"], 0.0)
                self.assertLessEqual(f.features["max_species_fraction"], 1.0)

    def test_max_frames_per_category_head_slice(self):
        from framework_v2.acquisition.generic_representation import load_pool

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrl = _FakeController(_build_pool(tmp, discriminative=True))
            pool = load_pool(ctrl, max_frames_per_category=3)
            self.assertEqual(pool.per_category_counts, {"alpha": 3, "beta": 3})
            self.assertEqual(pool.total_frames, 6)

    def test_discriminative_pool_passes_adequacy(self):
        from framework_v2.acquisition.generic_representation import (
            build_adequate_representation, load_pool)
        from framework_v2.states import SemanticState

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrl = _FakeController(_build_pool(tmp, discriminative=True))
            scope = _scope_contract()
            pool = load_pool(ctrl)
            result = build_adequate_representation(
                pool, id_prefix="p1-run", scope_contract=scope,
                deployment_claim="generic deployment claim")

            self.assertFalse(result.recovered_from_primary)
            self.assertEqual(result.adequacy.verdict, SemanticState.PASS)
            self.assertGreater(len(result.representation.regimes), 1)
            # The chosen spec is the full primary axis set, generic provenance, no material name.
            self.assertEqual(result.spec.provenance, "generic_raw_structure_v1")
            self.assertIn("n_atoms", result.spec.continuous_variables)

    def test_degenerate_pool_fails_closed_representation_insufficient(self):
        from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
        from framework_v2.acquisition.generic_representation import (
            build_adequate_representation, load_pool)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrl = _FakeController(_build_pool(tmp, discriminative=False))
            scope = _scope_contract()
            pool = load_pool(ctrl)
            with self.assertRaises(AcquisitionCapabilityGap) as cm:
                build_adequate_representation(
                    pool, id_prefix="p1-run", scope_contract=scope,
                    deployment_claim="generic deployment claim")
            self.assertEqual(cm.exception.gap_kind, "REPRESENTATION_INSUFFICIENT")

    def test_provider_applies_and_builds_result(self):
        from framework_v2.acquisition.generic_provider import (
            GenericStructuralDescriptorProvider)
        from framework_v2.states import SemanticState

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrl = _FakeController(_build_pool(tmp, discriminative=True))
            scope = _scope_contract()
            obj = _Objective(scope)
            provider = GenericStructuralDescriptorProvider()

            self.assertTrue(provider.applies(
                controller=ctrl, objective=obj, scope_contract=scope))
            result = provider.build_representation_result(
                controller=ctrl, objective=obj, scope_contract=scope)
            self.assertEqual(result.adequacy.verdict, SemanticState.PASS)

    def test_provider_does_not_apply_without_pool(self):
        from framework_v2.acquisition.generic_provider import (
            GenericStructuralDescriptorProvider)

        # A controller whose only input is not a pool manifest -> applies() is False.
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            other = tmp / "not_a_manifest.json"
            other.write_text(json.dumps({"unrelated": True}))
            ctrl = _FakeController(str(other))
            scope = _scope_contract()
            obj = _Objective(scope)
            provider = GenericStructuralDescriptorProvider()
            self.assertFalse(provider.applies(
                controller=ctrl, objective=obj, scope_contract=scope))

    def test_resolver_prefers_specialized_over_generic_fallback(self):
        from framework_v2.acquisition.descriptor_plugins import (
            register_descriptor_provider, register_generic_descriptor_provider,
            resolve_descriptor_provider)
        from framework_v2.acquisition.generic_provider import (
            GenericStructuralDescriptorProvider)

        class _Specialized:
            material_id = "specialized-test-material"

            def applies(self, *, controller, objective, scope_contract):
                return True

            def build_descriptor_space_evidence(self, *, controller, objective, scope_contract):
                raise AssertionError("not exercised in this test")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrl = _FakeController(_build_pool(tmp, discriminative=True))
            scope = _scope_contract()
            obj = _Objective(scope)

            generic = GenericStructuralDescriptorProvider()
            register_generic_descriptor_provider(generic)
            specialized = _Specialized()
            register_descriptor_provider(specialized)

            # Both would apply, but the specialized plugin wins (generic never collides).
            chosen = resolve_descriptor_provider(
                controller=ctrl, objective=obj, scope_contract=scope)
            self.assertIs(chosen, specialized)

    def test_resolver_falls_back_to_generic_when_no_specialized(self):
        from framework_v2.acquisition.descriptor_plugins import (
            register_generic_descriptor_provider, resolve_descriptor_provider)
        from framework_v2.acquisition.generic_provider import (
            GenericStructuralDescriptorProvider)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrl = _FakeController(_build_pool(tmp, discriminative=True))
            scope = _scope_contract()
            obj = _Objective(scope)
            generic = GenericStructuralDescriptorProvider()
            register_generic_descriptor_provider(generic)
            chosen = resolve_descriptor_provider(
                controller=ctrl, objective=obj, scope_contract=scope)
            self.assertIs(chosen, generic)

    def test_core_has_no_material_specific_hardcodes(self):
        """Scan the generic core (comments + string literals stripped) for forbidden tokens.

        Stripping strings/comments lets the module DOCSTRING legitimately mention 'liquid' /
        'bulk_amo' as examples of opaque labels it does NOT interpret, while still catching any
        material name, phase, or element-pair cutoff baked into actual code logic.
        """
        from framework_v2.acquisition import (
            generic_provider, generic_representation)

        forbidden = (
            "sio2", "si_o", "si-o", "bulk_amo", "amorphous", "vacancy",
            "quartz", "cristobalite", "silica",
        )
        for mod in (generic_representation, generic_provider):
            src = Path(mod.__file__).read_text()
            code_only = []
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type in (tokenize.STRING, tokenize.COMMENT):
                    continue
                code_only.append(tok.string)
            blob = " ".join(code_only).lower()
            for bad in forbidden:
                self.assertNotIn(
                    bad, blob,
                    msg=f"forbidden material-specific token {bad!r} found in {mod.__file__}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
