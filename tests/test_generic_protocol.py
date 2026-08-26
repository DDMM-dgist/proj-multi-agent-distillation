"""FE-027 P4 -- generic ProtocolEnvelopeBuilder for both backends (§10/§11).

On the same synthetic non-SiO2 pool used in P1-P3, proves that generation bounds are derived
from DATA + PHYSICS (the pool's own nearest-neighbor scale + versioned framework knobs), never
from a material name/phase/SiO2 constant:

  * LOCAL_PERTURBATION displacement is bounded by a fraction of the pool nearest-neighbor spacing;
  * TEACHER_DRIVEN_MD bounds n_md_steps from the compute ceiling and records that absolute
    temperature is NOT derivable from raw structure (presence-checked + unbounded_from_raw_structure),
    with physical admissibility enforced by output bounds instead;
  * output admissibility rejects overlapping / over-strained generated structures;
  * the builder fails closed (PROTOCOL_ENVELOPE_UNGROUNDED) when no pooled frame carries a
    computable nearest-neighbor distance -- bounds are never fabricated.
"""
from __future__ import annotations

import unittest

try:
    import ase  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from test_generic_regions import _representation


@unittest.skipUnless(_HAS_DEPS, "pydantic/ase not installed")
class GenericProtocolP4(unittest.TestCase):
    def _pool(self):
        _representation_result, pool, _scope = _representation(discriminative=True)
        return pool

    def test_perturbation_bounds_from_nn_scale(self):
        from framework_v2.acquisition.contracts import AcquisitionStrategyKind
        from framework_v2.acquisition.generic_protocol import (
            EnvelopeParams, build_perturbation_envelope)

        pool = self._pool()
        params = EnvelopeParams()
        env = build_perturbation_envelope(pool, params=params, envelope_id="pe", evidence_ref="ev")

        self.assertEqual(env.strategy_kind, AcquisitionStrategyKind.LOCAL_PERTURBATION)
        self.assertGreater(env.nn_scale_A, 0.0)
        # The augment_atoms recipe knobs are the envelope's decision space: cell-strain magnitude is
        # numerically bounded by a versioned fraction; the Metropolis temperature, acceptance
        # sharpness and displacement sampling range are presence-checked + recorded unbounded (the
        # MD-temperature pattern), with physical displacement safety enforced by the output floor.
        self.assertEqual(env.param_bounds["cell_sigma"], (0.0, params.max_cell_strain_frac))
        # augment_atoms constrains the acceptance sharpness to the closed unit interval.
        self.assertEqual(env.param_bounds["beta"], (0.0, 1.0))
        for key in ("T_K", "beta", "sigma_range_A", "seed"):
            self.assertIn(key, env.required_param_keys)
        for key in ("T_K", "sigma_range_A", "seed"):
            self.assertIn(key, env.presence_required_keys)
        for key in ("T_K", "sigma_range_A"):
            self.assertIn(key, env.unbounded_from_raw_structure)
            self.assertNotIn(key, env.param_bounds)
        # Output admissibility floor scales with the same nn scale.
        self.assertAlmostEqual(
            env.output_admissibility["min_interatomic_distance_A"],
            params.min_interatomic_frac_of_nn * env.nn_scale_A)
        self.assertEqual(env.evidence_refs, ["ev"])

    def test_md_envelope_bounds_steps_and_records_temperature_unbounded(self):
        from framework_v2.acquisition.contracts import (
            AcquisitionStrategyKind, ComputeCeiling)
        from framework_v2.acquisition.generic_protocol import (
            EnvelopeParams, build_md_envelope)

        pool = self._pool()
        params = EnvelopeParams()
        env = build_md_envelope(
            pool, params=params, envelope_id="me",
            compute_ceiling=ComputeCeiling(max_md_steps_total=500))

        self.assertEqual(env.strategy_kind, AcquisitionStrategyKind.TEACHER_DRIVEN_MD)
        self.assertEqual(env.param_bounds["n_md_steps"], (1.0, 500.0))
        # Absolute temperature is NOT derivable from raw structure: presence-checked + recorded.
        self.assertIn("temperature_K", env.presence_required_keys)
        self.assertIn("temperature_K", env.unbounded_from_raw_structure)
        self.assertNotIn("temperature_K", env.param_bounds)
        self.assertIn("temperature_K", env.required_param_keys)
        self.assertIn("n_md_steps", env.required_param_keys)

    def test_md_envelope_without_ceiling_omits_step_bound(self):
        from framework_v2.acquisition.generic_protocol import (
            EnvelopeParams, build_md_envelope)

        pool = self._pool()
        env = build_md_envelope(pool, params=EnvelopeParams(), envelope_id="me2")
        self.assertNotIn("n_md_steps", env.param_bounds)
        # Temperature still presence-required and recorded unbounded even with no ceiling.
        self.assertIn("temperature_K", env.presence_required_keys)
        self.assertIn("temperature_K", env.unbounded_from_raw_structure)

    def test_output_admissibility_rejects_overlap_and_overstrain(self):
        from framework_v2.acquisition.generic_protocol import (
            EnvelopeParams, build_perturbation_envelope, check_output_admissible)

        pool = self._pool()
        params = EnvelopeParams()
        env = build_perturbation_envelope(pool, params=params, envelope_id="pe")
        floor = env.output_admissibility["min_interatomic_distance_A"]

        # Admissible: well-separated, small volume change.
        self.assertEqual(
            check_output_admissible(
                env, min_interatomic_distance_A=floor + 0.5, volume_change_frac=0.05),
            [])
        # Atomic overlap.
        issues = check_output_admissible(
            env, min_interatomic_distance_A=floor - 0.1, volume_change_frac=0.0)
        self.assertTrue(any("atomic overlap" in i for i in issues))
        # Over-strained cell volume.
        issues2 = check_output_admissible(
            env, min_interatomic_distance_A=floor + 0.5,
            volume_change_frac=params.max_volume_change_frac + 0.1)
        self.assertTrue(any("volume change" in i for i in issues2))

    def test_fails_closed_when_no_nn_distance_available(self):
        from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
        from framework_v2.acquisition.generic_protocol import (
            EnvelopeParams, build_perturbation_envelope)

        pool = self._pool()
        # Strip the geometry axis from every frame -> the envelope cannot derive physical bounds.
        for f in pool.frames:
            f.features.pop("mean_min_neighbor_distance_A", None)
        with self.assertRaises(AcquisitionCapabilityGap) as cm:
            build_perturbation_envelope(pool, params=EnvelopeParams(), envelope_id="pe")
        self.assertEqual(cm.exception.gap_kind, "PROTOCOL_ENVELOPE_UNGROUNDED")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
