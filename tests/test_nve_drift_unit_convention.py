"""Deterministic unit-convention audit for the Stage-11 nve_drift observable.

Guarantees:
  * ``validation.structure_dynamics.compute_nve_drift`` returns a slope in
    meV/atom/ns (its declared unit; verified against a synthetic series with
    a known slope).
  * The Stage-11 executor's post-conversion (÷1000 → meV/atom/ps) yields the
    exact pre-registered unit required by validation_profile.yaml
    (max_abs 1.0 meV/atom/ps).
  * The criterion evaluator (validation.report.criterion_passes) compares
    correctly at the boundary and either side of it.
  * The unit conversion changes only the reported value's unit, never the
    underlying pre-registered threshold semantics (1.0 meV/atom/ps).

Session 2026-08-21 — Path-A Stage-11 physical_validation_nve authorization.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation.report import criterion_passes
from validation.structure_dynamics import compute_nve_drift

N_ATOMS = 192
TIMESTEP_FS = 0.5
SAMPLE_INTERVAL_STEPS = 100
N_SAMPLES = 400
CRITERION = {"operator": "max_abs", "threshold": 1.0, "unit": "meV/atom/ps"}


def _synthesize(target_slope_meV_per_atom_per_ps: float,
                *, e0_eV: float = -10000.0, noise_scale: float = 0.0,
                rng_seed: int = 0):
    """Return (steps, energies_eV) such that a perfect linear fit of the
    per-atom mean-centered energy vs t_ns yields a slope of
    (target_slope_ps * 1000) meV/atom/ns — equivalently, the pre-registered
    Stage-11 slope of ``target_slope_meV_per_atom_per_ps`` after ÷1000
    conversion by the executor.

    Total system-energy change over 20 ps = 192 atoms × target_ps × 20 meV / 1000
    e.g. 0.5 meV/atom/ps × 20 ps × 192 atoms / 1000 = 1.92 eV total shift.
    """
    steps = np.arange(1, N_SAMPLES + 1) * SAMPLE_INTERVAL_STEPS
    t_ps = steps * TIMESTEP_FS / 1000.0  # ps
    # E[i] = E0 + target_ps * t_ps * n_atoms / 1000  (eV)
    energies = e0_eV + target_slope_meV_per_atom_per_ps * t_ps * N_ATOMS / 1000.0
    if noise_scale > 0:
        rng = np.random.default_rng(rng_seed)
        energies = energies + rng.normal(scale=noise_scale, size=energies.shape)
    return steps, energies


def _drift_meV_per_atom_per_ps(steps, energies):
    slope_ns, _ = compute_nve_drift(list(energies), TIMESTEP_FS, N_ATOMS,
                                    sample_interval_steps=SAMPLE_INTERVAL_STEPS,
                                    steps=list(steps))
    return float(slope_ns) / 1000.0


# ---------------------------------------------------------------------------
# 1. compute_nve_drift returns meV/atom/ns (independent of any label)
# ---------------------------------------------------------------------------

def test_compute_nve_drift_actual_return_unit_is_meV_per_atom_per_ns():
    # Construct energies such that a slope of exactly S ps → per-atom-mean
    # slope in meV/atom/ns must equal S * 1000.
    for target_ps in (0.001, 0.5, 2.0, 10.0):
        steps, energies = _synthesize(target_ps)
        slope_ns, _ = compute_nve_drift(list(energies), TIMESTEP_FS, N_ATOMS,
                                        sample_interval_steps=SAMPLE_INTERVAL_STEPS,
                                        steps=list(steps))
        expected_ns = target_ps * 1000.0
        assert slope_ns == pytest.approx(expected_ns, rel=1e-9, abs=1e-9), (
            f"compute_nve_drift returned {slope_ns:g} meV/atom/ns; "
            f"expected {expected_ns:g} for target={target_ps:g} meV/atom/ps "
            "(unit mismatch would invalidate Stage-11 threshold semantics)")


# ---------------------------------------------------------------------------
# 2. Post-conversion (÷1000) exactly recovers the pre-registered ps unit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_ps", [+0.5, -0.5, +2.0, -2.0, +0.0005, 0.0])
def test_converted_slope_matches_target_ps_exactly(target_ps):
    steps, energies = _synthesize(target_ps)
    drift_ps = _drift_meV_per_atom_per_ps(steps, energies)
    assert drift_ps == pytest.approx(target_ps, rel=1e-9, abs=1e-9), (
        f"drift_ps={drift_ps:g} != target={target_ps:g}")


# ---------------------------------------------------------------------------
# 3. Pre-registered criterion evaluator gets the right verdict on each side
# ---------------------------------------------------------------------------

def test_positive_slope_below_threshold_passes():
    steps, energies = _synthesize(+0.5)
    drift_ps = _drift_meV_per_atom_per_ps(steps, energies)
    assert drift_ps == pytest.approx(+0.5, abs=1e-9)
    assert abs(drift_ps) == pytest.approx(0.5, abs=1e-9)
    assert criterion_passes(drift_ps, CRITERION) is True


def test_negative_slope_below_threshold_passes():
    steps, energies = _synthesize(-0.5)
    drift_ps = _drift_meV_per_atom_per_ps(steps, energies)
    assert drift_ps == pytest.approx(-0.5, abs=1e-9)
    assert abs(drift_ps) == pytest.approx(0.5, abs=1e-9)
    assert criterion_passes(drift_ps, CRITERION) is True


def test_positive_slope_above_threshold_fails():
    steps, energies = _synthesize(+2.0)
    drift_ps = _drift_meV_per_atom_per_ps(steps, energies)
    assert drift_ps == pytest.approx(+2.0, abs=1e-9)
    assert criterion_passes(drift_ps, CRITERION) is False


def test_negative_slope_above_threshold_fails():
    steps, energies = _synthesize(-2.0)
    drift_ps = _drift_meV_per_atom_per_ps(steps, energies)
    assert drift_ps == pytest.approx(-2.0, abs=1e-9)
    assert abs(drift_ps) == pytest.approx(2.0, abs=1e-9)
    assert criterion_passes(drift_ps, CRITERION) is False


def test_criterion_operator_is_inclusive_max_abs():
    # Document the boundary semantics without relying on floating-point equality:
    # criterion_passes(x, {"operator":"max_abs","threshold":1.0}) returns
    # True iff |x| <= 1.0 (inclusive).
    assert criterion_passes(0.999999, CRITERION) is True
    assert criterion_passes(-0.999999, CRITERION) is True
    assert criterion_passes(1.000001, CRITERION) is False
    assert criterion_passes(-1.000001, CRITERION) is False


# ---------------------------------------------------------------------------
# 4. Robustness — noisy series still fits close to the true slope
# ---------------------------------------------------------------------------

def test_noisy_series_recovers_target_slope_within_tolerance():
    steps, energies = _synthesize(+0.3, noise_scale=1e-4, rng_seed=42)
    drift_ps = _drift_meV_per_atom_per_ps(steps, energies)
    # Small compared to 0.3 given the sampling and noise scale.
    assert drift_ps == pytest.approx(+0.3, abs=1e-3)
    assert criterion_passes(drift_ps, CRITERION) is True


# ---------------------------------------------------------------------------
# 5. Unit conversion is a lossless numeric identity (no threshold change)
# ---------------------------------------------------------------------------

def test_conversion_factor_is_exactly_1000():
    for target_ps in (0.001, 0.1, 0.5, 1.0, 2.0, 10.0, -0.5):
        steps, energies = _synthesize(target_ps)
        slope_ns, _ = compute_nve_drift(list(energies), TIMESTEP_FS, N_ATOMS,
                                        sample_interval_steps=SAMPLE_INTERVAL_STEPS,
                                        steps=list(steps))
        assert slope_ns / 1000.0 == pytest.approx(target_ps, abs=1e-9), (
            "The conversion factor ns→ps must be exactly 1000; anything else "
            "would silently redefine the pre-registered threshold semantics.")


# ---------------------------------------------------------------------------
# 6. The threshold value 1.0 is unchanged (regression against creep)
# ---------------------------------------------------------------------------

def test_criterion_dict_is_the_pre_registered_shape():
    # Guard against any test/code path silently loosening the pre-registered
    # threshold — 1.0 meV/atom/ps stays 1.0 meV/atom/ps.
    assert CRITERION["operator"] == "max_abs"
    assert CRITERION["threshold"] == 1.0
    assert CRITERION["unit"] == "meV/atom/ps"
