"""FE-057 -- bounded LOCAL_PERTURBATION generation.

Proves the backend can never spin in the upstream ``augment_atoms`` unbounded
"generate until n accepted" loop:

  * a degenerate (single-atom periodic) parent is declared inadmissible by preflight and returns
    immediately with zero attempts -- it does NOT loop forever;
  * a parent whose every candidate is force-rejected terminates at the attempt budget as
    EXHAUSTED_PARTIAL with the deficit recorded and nothing fabricated;
  * a parent whose every candidate is similarity-rejected likewise terminates at the budget;
  * a normal parent still reaches its requested child count (COMPLETE);
  * generation is deterministic given the seed (frozen-plan reproducibility);
  * checkpoint/resume reuses completed-parent work without regenerating or duplicating it;
  * the finalized-manifest summary records requested-vs-accepted, exhaustion and completeness.
"""
from __future__ import annotations

import unittest

try:
    import ase  # noqa: F401
    import numpy as np  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False


def _cubic_pool(n_side=2, spacing=2.0):
    import ase
    import numpy as np
    pts = [(x * spacing, y * spacing, z * spacing)
           for x in range(n_side) for y in range(n_side) for z in range(n_side)]
    L = n_side * spacing
    return ase.Atoms("Si" * len(pts), positions=np.array(pts, float),
                     cell=[L, L, L], pbc=True)


class _FakeCalc:
    """Minimal ASE-Calculator-shaped stand-in matching the augment_atoms ``label`` contract:
    ``calculate(atoms, props)`` populates ``.results['energy'|'forces']``. ``force_mag`` controls
    whether structures pass the force ceiling; ``energy`` is constant so the Metropolis probability
    is a fixed 0.25."""

    def __init__(self, *, force_mag=0.01, energy=0.0):
        import numpy as np
        self._force_mag = float(force_mag)
        self._energy = float(energy)
        self.results = {}
        self.calls = 0
        self._np = np

    def calculate(self, atoms, properties=None, system_changes=None):
        self.calls += 1
        f = self._np.full((len(atoms), 3), self._force_mag, dtype=float)
        self.results = {"energy": self._energy, "forces": f}


class _TeacherProvider:
    def __init__(self, calc):
        self._calc = calc

    @property
    def identity_sha256(self):
        return "fake-teacher"

    def make_ase_calculator(self):
        return self._calc


class _RaisingTeacherProvider:
    """A Teacher whose calculator construction fails loudly -- used to prove a full resume never
    reconstructs the (expensive) calculator or regenerates completed parents."""

    @property
    def identity_sha256(self):
        return "must-not-build"

    def make_ase_calculator(self):
        raise AssertionError("Teacher calculator must NOT be built when all parents are resumed")


def _config(**overrides):
    from framework_v2.acquisition.generators.local_perturbation import _PerturbConfig
    base = dict(n_per_structure=3, T=1000.0, beta=0.5, sigma_range=(0.15, 0.3), seed=7,
                units="eV", cell_sigma=None, max_force=30.0, min_separation=0.5,
                max_relax_steps=2, similarity_threshold=0.1)
    base.update(overrides)
    return _PerturbConfig(**base)


@unittest.skipUnless(_HAS_DEPS, "ase/numpy/pydantic not installed")
class BoundedPerturbationDriver(unittest.TestCase):
    def test_single_atom_periodic_is_inadmissible_and_does_not_loop(self):
        import ase
        from framework_v2.acquisition.generators.bounded_perturbation import (
            STATUS_INADMISSIBLE_DEGENERATE, StoppingPolicy, bounded_generate_for_parent,
            perturbation_admissibility)

        one_atom = ase.Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[3, 3, 3], pbc=True)
        admissible, reason = perturbation_admissibility(one_atom)
        self.assertFalse(admissible)
        self.assertIn("single_atom", reason)

        calc = _FakeCalc()
        children, rec = bounded_generate_for_parent(
            one_atom, calc, config=_config(), policy=StoppingPolicy(attempts_per_child_factor=5.0),
            parent_id="p", parent_index=0, seed=1)
        self.assertEqual(children, [])
        self.assertEqual(rec.terminal_status, STATUS_INADMISSIBLE_DEGENERATE)
        self.assertEqual(rec.attempts, 0)          # never entered the loop
        self.assertEqual(calc.calls, 0)            # never touched the PES

    def test_zero_acceptable_children_terminates_at_force_budget(self):
        from framework_v2.acquisition.generators.bounded_perturbation import (
            STATUS_EXHAUSTED_PARTIAL, StoppingPolicy, bounded_generate_for_parent)

        # Every candidate exceeds the force ceiling -> every attempt is force-rejected.
        calc = _FakeCalc(force_mag=100.0)
        policy = StoppingPolicy(attempts_per_child_factor=5.0)
        children, rec = bounded_generate_for_parent(
            _cubic_pool(), calc, config=_config(n_per_structure=3), policy=policy,
            parent_id="p", parent_index=0, seed=3)
        self.assertEqual(children, [])
        self.assertEqual(rec.terminal_status, STATUS_EXHAUSTED_PARTIAL)
        self.assertEqual(rec.accepted, 0)
        self.assertEqual(rec.deficit, 3)
        self.assertEqual(rec.attempts, rec.max_attempts)      # stopped exactly at the budget
        self.assertEqual(rec.max_attempts, 15)                 # 3 * 5
        self.assertEqual(rec.rejections["force"], rec.max_attempts)

    def test_high_similarity_rejection_terminates_at_budget(self):
        from framework_v2.acquisition.generators.bounded_perturbation import (
            STATUS_EXHAUSTED_PARTIAL, StoppingPolicy, bounded_generate_for_parent)

        # Vanishingly small displacements -> every child is a near-duplicate (similarity-rejected).
        calc = _FakeCalc(force_mag=0.01)
        policy = StoppingPolicy(attempts_per_child_factor=4.0)
        children, rec = bounded_generate_for_parent(
            _cubic_pool(), calc, config=_config(sigma_range=(1e-6, 2e-6)), policy=policy,
            parent_id="p", parent_index=0, seed=5)
        self.assertEqual(children, [])
        self.assertEqual(rec.terminal_status, STATUS_EXHAUSTED_PARTIAL)
        self.assertEqual(rec.attempts, rec.max_attempts)
        self.assertGreater(rec.rejections["similar"], 0)

    def test_normal_parent_reaches_requested_count(self):
        from framework_v2.acquisition.generators.bounded_perturbation import (
            STATUS_COMPLETE, StoppingPolicy, bounded_generate_for_parent)

        calc = _FakeCalc(force_mag=0.01)
        policy = StoppingPolicy(attempts_per_child_factor=40.0)
        children, rec = bounded_generate_for_parent(
            _cubic_pool(), calc, config=_config(n_per_structure=3), policy=policy,
            parent_id="p", parent_index=0, seed=11)
        self.assertEqual(rec.terminal_status, STATUS_COMPLETE)
        self.assertEqual(rec.accepted, 3)
        self.assertEqual(len(children), 3)
        self.assertEqual(rec.deficit, 0)

    def test_generation_is_deterministic(self):
        import numpy as np
        from framework_v2.acquisition.generators.bounded_perturbation import (
            StoppingPolicy, bounded_generate_for_parent)

        policy = StoppingPolicy(attempts_per_child_factor=40.0)
        kwargs = dict(config=_config(n_per_structure=3), policy=policy,
                      parent_id="p", parent_index=0, seed=17)
        c1, r1 = bounded_generate_for_parent(_cubic_pool(), _FakeCalc(), **kwargs)
        c2, r2 = bounded_generate_for_parent(_cubic_pool(), _FakeCalc(), **kwargs)
        self.assertEqual(r1.accepted, r2.accepted)
        self.assertEqual(len(c1), len(c2))
        for a, b in zip(c1, c2):
            self.assertTrue(np.allclose(a.positions, b.positions))


@unittest.skipUnless(_HAS_DEPS, "ase/numpy/pydantic not installed")
class BoundedGeneratorEndToEnd(unittest.TestCase):
    def _protocol(self, parents_path, *, n_per=2, seed=11):
        from framework_v2.acquisition.contracts import AcquisitionStrategyKind
        from framework_v2.acquisition.generators.base import GenerationProtocol
        return GenerationProtocol(
            protocol_id="proto", backend_id="local_perturbation.augment_atoms",
            strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION, strategy_sha256="s",
            n_requested=2 * n_per, target_regime_ids=[], parent_ids=["p0", "p1"],
            params={"parents_path": parents_path, "n_per_structure": n_per, "T_K": 1000.0,
                    "beta": 0.5, "sigma_range_A": [0.15, 0.3], "cell_sigma": None, "seed": seed,
                    "units": "eV", "max_relax_steps": 2})

    def _write_parents(self, path):
        from ase.io import write
        write(str(path), [_cubic_pool(), _cubic_pool()], format="extxyz")

    def test_checkpoint_resume_does_not_duplicate(self):
        import tempfile
        from pathlib import Path

        from framework_v2.acquisition.generators.bounded_perturbation import StoppingPolicy
        from framework_v2.acquisition.generators.local_perturbation import (
            LocalPerturbationGenerator)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            parents = td / "parents.extxyz"
            self._write_parents(parents)
            workdir = str(td / "gen")
            proto = self._protocol(str(parents), n_per=2)
            policy = StoppingPolicy(attempts_per_child_factor=40.0)

            calc = _FakeCalc(force_mag=0.01)
            res1 = LocalPerturbationGenerator(policy=policy).generate(
                proto, workdir=workdir, teacher=_TeacherProvider(calc))
            self.assertGreater(res1.n_generated, 0)
            progress = json_load(Path(workdir) / "generation_progress.json")
            self.assertTrue(all(r["terminal_status"] == "COMPLETE"
                                for r in progress["parents"].values()))

            # Resume with a Teacher that refuses to build a calculator: proves completed parents are
            # reused, not regenerated, and no children are duplicated.
            res2 = LocalPerturbationGenerator(policy=policy).generate(
                proto, workdir=workdir, teacher=_RaisingTeacherProvider())
            self.assertEqual(res1.candidate_ids, res2.candidate_ids)
            self.assertEqual(len(res2.candidate_ids), len(set(res2.candidate_ids)))

    def test_partial_status_recorded_in_summary(self):
        import tempfile
        from pathlib import Path

        from framework_v2.acquisition.generators.bounded_perturbation import StoppingPolicy
        from framework_v2.acquisition.generators.local_perturbation import (
            LocalPerturbationGenerator)
        from runtimes.pydantic_ai.train_augmentation import (
            _read_generation_progress, _summarize_generation)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            parents = td / "parents.extxyz"
            self._write_parents(parents)
            workdir = str(td / "gen")
            proto = self._protocol(str(parents), n_per=3)
            policy = StoppingPolicy(attempts_per_child_factor=4.0)

            # Force ceiling exceeded everywhere -> both parents exhaust with zero accepted.
            LocalPerturbationGenerator(policy=policy).generate(
                proto, workdir=workdir, teacher=_TeacherProvider(_FakeCalc(force_mag=100.0)))
            summary = _summarize_generation(_read_generation_progress(workdir))
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["accepted_children"], 0)
            self.assertEqual(summary["requested_children"], 6)
            self.assertEqual(summary["deficit_children"], 6)
            self.assertEqual(sorted(summary["exhausted_parents"]), ["p0", "p1"])

    def test_fail_closed_policy_raises_on_deficit(self):
        import tempfile
        from pathlib import Path

        from framework_v2.acquisition.generators.bounded_perturbation import StoppingPolicy
        from framework_v2.acquisition.generators.local_perturbation import (
            LocalPerturbationGenerator, PerturbationExhausted)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            parents = td / "parents.extxyz"
            self._write_parents(parents)
            proto = self._protocol(str(parents), n_per=3)
            policy = StoppingPolicy(attempts_per_child_factor=4.0, exhaustion_policy="fail_closed")
            with self.assertRaises(PerturbationExhausted):
                LocalPerturbationGenerator(policy=policy).generate(
                    proto, workdir=str(td / "gen"),
                    teacher=_TeacherProvider(_FakeCalc(force_mag=100.0)))


def json_load(path):
    import json
    return json.loads(path.read_text())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
