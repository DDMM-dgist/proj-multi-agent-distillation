"""FE-029 -- physically-correct PBC nearest-neighbour + representation/sizing missing-axis
compatibility (no frame dropping, no imputation).

FE-029 is a MINIMAL generic correction on top of FE-027/FE-028. Two surfaces collided in the
ffv4c fresh proof run: the generic representation TOLERATES an uncomputable descriptor axis
(records it as an evidence gap, never imputed), while the labeling-population FPS sizing
required a DENSE vector on every candidate axis for every eligible frame -- so it failed closed
with ``REPRESENTATION_INCOMPLETE`` on the first frame missing an axis. The trip frame was a
single-atom PERIODIC primitive cell (``silicon_crystalline_main#100``): a one-atom crystalline
cell has a perfectly well-defined nearest-neighbour distance -- the nearest periodic image --
but the neighbour descriptor returned ``None`` because it only looked at OTHER atoms.

This suite proves both halves of the fix, with a deliberately non-SiO2 synthetic pool so
nothing material-specific can leak in:

  Part 1 -- ``_shortest_periodic_image_distance`` / ``_mean_min_neighbor_distance``:
    * one-atom periodic cubic cell           -> finite, = shortest lattice vector;
    * one-atom NON-orthogonal periodic cell   -> finite, = independent brute-force minimum;
    * multi-atom periodic cell                -> finite, = nearest OTHER atom (self-image larger);
    * isolated NON-periodic single atom       -> None (genuinely undefined; never fabricated).

  Part 2 -- ``derive_admissible_sizing_representation`` (the ffv4c reproduction):
    * a saturated eligible pool mixing periodic single-atom frames with multi-atom periodic
      frames + the ``mean_min_neighbor_distance_A`` candidate axis proves the self-image distance
      is computed -> the axis is universally available -> NO ``REPRESENTATION_INCOMPLETE`` ->
      the FULL candidate representation is admitted (not reduced) -> a dense sizing matrix over
      EVERY eligible frame -> deterministic FPS sizing yields a recommended population.

  Part 3 -- genuine missing-axis handling (non-periodic single atoms present):
    * axis incompleteness NEVER drops a frame and NEVER imputes a value; the sizing
      representation is deterministically reduced to the universally-available adequate axis set
      when that still discriminates the pool, OR fails closed with the typed
      ``AcquisitionCapabilityGap(REPRESENTATION_INSUFFICIENT)`` when no admissible axis subset is
      adequate.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import ase  # noqa: F401
    import numpy as np  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False


def _scope_contract():
    from framework_v2.contracts import (
        DeploymentScopeContract, ScopeCategory, ScopeRegion)
    return DeploymentScopeContract(
        contract_id="fe029-scope",
        objective="FE-029 representation/sizing compatibility deployment scope",
        regions=[ScopeRegion(
            region_id="primary", category=ScopeCategory.PRIMARY_DEPLOYMENT,
            membership_rule="descriptor in the primary deployment set")])


class _FakeController:
    """Minimal stand-in for the frozen workflow controller: only ``.state`` is read."""

    def __init__(self, manifest_path: str, run_id: str = "fe029-run"):
        self.state = {"inputs": [{"snapshot": manifest_path}], "run_id": run_id}


def _write_manifest(tmp: Path, cat_files: list[tuple[str, Path, int]]) -> str:
    """Write a schema-valid sanitized-pool manifest for the given (category, file, n) list."""
    categories = []
    total = 0
    for cat, f, n in cat_files:
        raw = f.read_bytes()
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
        "sanitized_pool_manifest_sha256": "fe029feed",
        "categories": categories,
    }
    mpath = tmp / "sanitized_pool_manifest.json"
    mpath.write_text(json.dumps(manifest))
    return str(mpath)


def _write_frames(path: Path, atoms_list) -> None:
    from ase.io import write as ase_write
    for a in atoms_list:
        ase_write(str(path), a, format="extxyz", append=True)


def _single_periodic(a: float, seed: int):
    """A one-atom PERIODIC cubic cell of edge ``a`` -- the ffv4c trip-frame shape."""
    from ase import Atoms
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
                 pbc=True)


def _multi_periodic(n_atoms: int, a: float, seed: int):
    from ase import Atoms
    import numpy as np
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.3, a - 0.3, size=(n_atoms, 3))
    return Atoms(f"Cu{n_atoms}", positions=pos, cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
                 pbc=True)


def _single_nonperiodic(seed: int):
    """An isolated NON-periodic single atom -- genuinely undefined neighbour/density axes."""
    from ase import Atoms
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], pbc=False)


def _candidate_primary_spec(pool, scope):
    """Build the full PRIMARY candidate RepresentationSpec exactly as the production path does."""
    from framework_v2.acquisition.generic_representation import (
        PRIMARY_CONTINUOUS_VARIABLES, _spec_for, build_representation)
    rep = build_representation(
        pool, representation_id="fe029-candidate-repr",
        descriptor="generic raw-structure descriptor space (size/density/geometry/composition)",
        continuous_variables=PRIMARY_CONTINUOUS_VARIABLES, scope_contract=scope)
    return _spec_for(rep, spec_id="fe029-candidate-spec",
                     continuous_variables=PRIMARY_CONTINUOUS_VARIABLES, scope_contract=scope)


@unittest.skipUnless(_HAS_DEPS, "ase/numpy/pydantic not installed")
class Fe029NeighbourPbc(unittest.TestCase):
    """Part 1 -- physically-correct nearest-neighbour under periodic boundary conditions."""

    def test_one_atom_periodic_cubic_is_shortest_lattice_vector(self):
        from framework_v2.acquisition.generic_representation import (
            _mean_min_neighbor_distance, _shortest_periodic_image_distance, compute_frame_features)
        atoms = _single_periodic(a=3.0, seed=0)
        self.assertAlmostEqual(_shortest_periodic_image_distance(atoms), 3.0, places=9)
        # One-atom periodic cell has a WELL-DEFINED nearest neighbour = nearest periodic image.
        self.assertAlmostEqual(_mean_min_neighbor_distance(atoms), 3.0, places=9)
        # The ffv4c symptom: the geometry axis is now PRESENT (was absent -> REPRESENTATION_INCOMPLETE).
        feats = compute_frame_features(atoms)
        self.assertIn("mean_min_neighbor_distance_A", feats)
        self.assertAlmostEqual(feats["mean_min_neighbor_distance_A"], 3.0, places=9)

    def test_one_atom_non_orthogonal_periodic_matches_brute_force(self):
        import itertools

        import numpy as np
        from ase import Atoms
        from framework_v2.acquisition.generic_representation import (
            _mean_min_neighbor_distance, _shortest_periodic_image_distance)
        cell = np.array([[3.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=cell.tolist(), pbc=True)
        # Independent brute-force shortest non-zero lattice translation over the same window.
        best = min(
            float(np.linalg.norm(np.array(c, dtype=float) @ cell))
            for c in itertools.product(range(-3, 4), repeat=3) if any(c))
        got = _shortest_periodic_image_distance(atoms)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, best, places=9)
        self.assertAlmostEqual(got, 3.0, places=9)  # the a-vector is shortest here
        self.assertAlmostEqual(_mean_min_neighbor_distance(atoms), best, places=9)

    def test_multi_atom_periodic_uses_nearest_other_atom(self):
        import numpy as np
        from ase import Atoms
        from framework_v2.acquisition.generic_representation import _mean_min_neighbor_distance
        # Two atoms 1.1 A apart in a large cell; the nearest OTHER atom (1.1) beats any self-image.
        atoms = Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
                      cell=[[20.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]], pbc=True)
        self.assertAlmostEqual(_mean_min_neighbor_distance(atoms), 1.1, places=6)

    def test_non_periodic_single_atom_is_genuinely_none(self):
        from framework_v2.acquisition.generic_representation import (
            _mean_min_neighbor_distance, _shortest_periodic_image_distance, compute_frame_features)
        atoms = _single_nonperiodic(seed=0)
        # No other atom AND no periodic image -> genuinely undefined. NEVER fabricated.
        self.assertIsNone(_shortest_periodic_image_distance(atoms))
        self.assertIsNone(_mean_min_neighbor_distance(atoms))
        feats = compute_frame_features(atoms)
        self.assertNotIn("mean_min_neighbor_distance_A", feats)
        self.assertNotIn("number_density_atoms_per_A3", feats)
        # The always-computable axes are still present and never imputed for the missing ones.
        self.assertIn("n_atoms", feats)
        self.assertIn("max_species_fraction", feats)


@unittest.skipUnless(_HAS_DEPS, "ase/numpy/pydantic not installed")
class Fe029Ffv4cReproduction(unittest.TestCase):
    """Part 2 -- the exact ffv4c failure: periodic single-atom frames no longer trip sizing."""

    def _saturated_pool_with_periodic_single_atoms(self, tmp: Path):
        from framework_v2.acquisition.generic_representation import load_pool
        prim = tmp / "crystalline_primitive.xyz"   # single-atom PERIODIC (ffv4c trip shape)
        bulk = tmp / "amorphous_bulk.xyz"           # multi-atom PERIODIC
        _write_frames(prim, [_single_periodic(a=2.7 + 0.1 * s, seed=s) for s in range(8)])
        _write_frames(bulk, [_multi_periodic(24, 9.0, seed=100 + s) for s in range(8)])
        mpath = _write_manifest(tmp, [("crystalline_primitive", prim, 8),
                                      ("amorphous_bulk", bulk, 8)])
        return load_pool(_FakeController(mpath))

    def test_periodic_single_atoms_keep_full_axes_no_incomplete(self):
        from framework_v2.acquisition.generic_representation import (
            PRIMARY_CONTINUOUS_VARIABLES, audit_axis_availability,
            derive_admissible_sizing_representation)
        from framework_v2.states import SemanticState

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pool = self._saturated_pool_with_periodic_single_atoms(tmp)
            scope = _scope_contract()
            candidate = _candidate_primary_spec(pool, scope)

            # The periodic self-image makes the geometry axis universally available -- so the
            # FULL candidate representation is admissible; nothing is REPRESENTATION_INCOMPLETE.
            counts = audit_axis_availability(pool, PRIMARY_CONTINUOUS_VARIABLES)
            for ax in PRIMARY_CONTINUOUS_VARIABLES:
                self.assertEqual(counts[ax], pool.total_frames,
                                 msg=f"axis {ax} not universal -> would have tripped ffv4c")

            sizing_rep = derive_admissible_sizing_representation(
                pool, candidate_spec=candidate, scope_contract=scope,
                deployment_claim="fe029 deployment claim", id_prefix="fe029-run")

            self.assertFalse(sizing_rep.reduced, "full candidate axes should be admitted, not reduced")
            self.assertEqual(tuple(sizing_rep.axes), tuple(PRIMARY_CONTINUOUS_VARIABLES))
            self.assertIn("mean_min_neighbor_distance_A", sizing_rep.axes)
            self.assertEqual(sizing_rep.adequacy.verdict, SemanticState.PASS)

    def test_dense_sizing_over_every_frame_yields_recommendation(self):
        from framework_v2.acquisition.generic_coverage import (
            FrameworkSizingParams, recommend_labeling_population_sizing)
        from framework_v2.acquisition.generic_representation import (
            derive_admissible_sizing_representation)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pool = self._saturated_pool_with_periodic_single_atoms(tmp)
            scope = _scope_contract()
            candidate = _candidate_primary_spec(pool, scope)
            sizing_rep = derive_admissible_sizing_representation(
                pool, candidate_spec=candidate, scope_contract=scope,
                deployment_claim="fe029 deployment claim", id_prefix="fe029-run")

            # A DENSE matrix over EVERY eligible frame -- no KeyError (no missing axis), no drop.
            axes = list(sizing_rep.axes)
            vectors = [[float(f.features[k]) for k in axes] for f in pool.frames]
            self.assertEqual(len(vectors), pool.total_frames)

            evidence = recommend_labeling_population_sizing(
                vectors, params=FrameworkSizingParams(), sizing_id="fe029-sizing",
                coverage_gap_sha256="fe029gap")
            self.assertEqual(evidence.eligible_population_size, pool.total_frames)
            self.assertGreaterEqual(evidence.recommended_population_size, 1)
            self.assertLessEqual(evidence.recommended_population_size, pool.total_frames)

    def test_derivation_is_deterministic(self):
        from framework_v2.acquisition.generic_representation import (
            derive_admissible_sizing_representation)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pool = self._saturated_pool_with_periodic_single_atoms(tmp)
            scope = _scope_contract()
            candidate = _candidate_primary_spec(pool, scope)
            a = derive_admissible_sizing_representation(
                pool, candidate_spec=candidate, scope_contract=scope,
                deployment_claim="fe029 deployment claim", id_prefix="fe029-run")
            b = derive_admissible_sizing_representation(
                pool, candidate_spec=candidate, scope_contract=scope,
                deployment_claim="fe029 deployment claim", id_prefix="fe029-run")
            self.assertEqual(tuple(a.axes), tuple(b.axes))
            self.assertEqual(a.spec.content_sha256(), b.spec.content_sha256())


@unittest.skipUnless(_HAS_DEPS, "ase/numpy/pydantic not installed")
class Fe029GenuineMissingAxis(unittest.TestCase):
    """Part 3 -- genuine missing-axis: reduce-when-adequate or fail-closed; never drop/impute."""

    def test_non_periodic_single_atoms_reduce_axes_keep_all_frames(self):
        from framework_v2.acquisition.generic_representation import (
            ALTERNATIVE_CONTINUOUS_VARIABLES, PRIMARY_CONTINUOUS_VARIABLES,
            audit_axis_availability, derive_admissible_sizing_representation, load_pool)
        from framework_v2.states import SemanticState

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            clusters = tmp / "clusters.xyz"   # NON-periodic single atoms: geometry/density undefined
            bulk = tmp / "bulk.xyz"           # multi-atom periodic
            _write_frames(clusters, [_single_nonperiodic(seed=s) for s in range(6)])
            _write_frames(bulk, [_multi_periodic(30, 10.0, seed=200 + s) for s in range(6)])
            mpath = _write_manifest(tmp, [("clusters", clusters, 6), ("bulk", bulk, 6)])
            pool = load_pool(_FakeController(mpath))
            scope = _scope_contract()
            candidate = _candidate_primary_spec(pool, scope)

            # Geometry/density axes are NOT universal (absent on the non-periodic single atoms).
            counts = audit_axis_availability(pool, PRIMARY_CONTINUOUS_VARIABLES)
            self.assertLess(counts["mean_min_neighbor_distance_A"], pool.total_frames)
            self.assertLess(counts["number_density_atoms_per_A3"], pool.total_frames)

            sizing_rep = derive_admissible_sizing_representation(
                pool, candidate_spec=candidate, scope_contract=scope,
                deployment_claim="fe029 deployment claim", id_prefix="fe029-run")

            # Reduced to the universally-available adequate axis set -- NOT the full candidate.
            self.assertTrue(sizing_rep.reduced)
            self.assertEqual(tuple(sizing_rep.axes), tuple(ALTERNATIVE_CONTINUOUS_VARIABLES))
            self.assertEqual(sizing_rep.adequacy.verdict, SemanticState.PASS)

            # NO frame dropped: a dense matrix over the reduced axes includes EVERY frame.
            axes = list(sizing_rep.axes)
            vectors = [[float(f.features[k]) for k in axes] for f in pool.frames]
            self.assertEqual(len(vectors), pool.total_frames)

            # NO imputation: the non-periodic single-atom frames still lack the dropped axes.
            cluster_frames = [f for f in pool.frames if f.category == "clusters"]
            self.assertTrue(cluster_frames)
            for f in cluster_frames:
                self.assertNotIn("mean_min_neighbor_distance_A", f.features)
                self.assertNotIn("number_density_atoms_per_A3", f.features)

    def test_no_admissible_axis_subset_fails_closed_typed_gap(self):
        from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
        from framework_v2.acquisition.generic_representation import (
            derive_admissible_sizing_representation, load_pool)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # Every frame is a single atom (n_atoms=1, max_species_fraction=1.0) -> the only
            # universally-available axes do NOT discriminate (one regime). Mixed PBC ensures the
            # geometry/density axes are not universal either -> no admissible adequate subset.
            per = tmp / "periodic_single.xyz"
            npr = tmp / "nonperiodic_single.xyz"
            _write_frames(per, [_single_periodic(a=3.0, seed=s) for s in range(5)])
            _write_frames(npr, [_single_nonperiodic(seed=s) for s in range(5)])
            mpath = _write_manifest(tmp, [("periodic_single", per, 5),
                                          ("nonperiodic_single", npr, 5)])
            pool = load_pool(_FakeController(mpath))
            scope = _scope_contract()
            candidate = _candidate_primary_spec(pool, scope)

            with self.assertRaises(AcquisitionCapabilityGap) as cm:
                derive_admissible_sizing_representation(
                    pool, candidate_spec=candidate, scope_contract=scope,
                    deployment_claim="fe029 deployment claim", id_prefix="fe029-run")
            self.assertEqual(cm.exception.gap_kind, "REPRESENTATION_INSUFFICIENT")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
