#!/usr/bin/env python3
"""Structure/dynamics validation against configs/validation_profile.yaml.
Operates purely on an ASE-readable trajectory — never touches model
internals, so it needs no per-teacher/student-`kind` adapter.

Implements the checks common across the toolkit's worked examples:
  rdf, coordination, density, msd, nve_drift
(adf and sq_fsdp are left as extension points — same pattern, see the
`# TODO` markers below — because their exact form is more material-specific
than the others.)

Usage:
    python validation/structure_dynamics.py trajectory.traj configs/validation_profile.yaml \
        [--timestep-fs 1.0] [--temperature-log energies.csv]
"""
import argparse
import itertools

import numpy as np
import yaml
from ase.io import read
from ase.geometry.analysis import Analysis


def compute_rdf(frames, elements, r_max=6.0, nbins=200):
    """Partial RDFs for each element pair present, averaged over frames."""
    pairs = list(itertools.combinations_with_replacement(sorted(elements), 2))
    out = {}
    for e1, e2 in pairs:
        rdfs = []
        for atoms in frames:
            ana = Analysis(atoms)
            rdf = ana.get_rdf(r_max, nbins, elements=[e1, e2])[0]
            rdfs.append(rdf)
        out[f"{e1}-{e2}"] = np.mean(rdfs, axis=0)
    r = np.linspace(0, r_max, nbins)
    return r, out


def compute_coordination(frames, elements, cutoffs):
    """Mean coordination number per element, using per-pair cutoffs (dict
    {"Si-O": 2.0, ...} in Angstrom) — supply from validation_profile if you
    need non-default cutoffs; this uses a simple distance cutoff, not a
    bonding-order method."""
    counts = {el: [] for el in elements}
    for atoms in frames:
        d = atoms.get_all_distances(mic=True)
        syms = np.array(atoms.get_chemical_symbols())
        for el in elements:
            idx = np.where(syms == el)[0]
            if len(idx) == 0:
                continue
            cn = []
            for i in idx:
                n = 0
                for j in range(len(atoms)):
                    if j == i:
                        continue
                    pair = "-".join(sorted([el, syms[j]]))
                    cutoff = cutoffs.get(pair, cutoffs.get("default", 3.0))
                    if d[i, j] < cutoff:
                        n += 1
                cn.append(n)
            counts[el].append(np.mean(cn))
    return {el: float(np.mean(v)) for el, v in counts.items() if v}


def compute_density(frames):
    densities = []
    for atoms in frames:
        mass_g = atoms.get_masses().sum() / 6.02214076e23  # amu -> g
        vol_cm3 = atoms.get_volume() * 1e-24               # A^3 -> cm^3
        densities.append(mass_g / vol_cm3)
    return float(np.mean(densities)), float(np.std(densities))


def compute_msd(frames):
    """Per-species MSD relative to the first frame — a coarse, single-run
    estimate; for a real drift/diffusion analysis average over multiple
    committee seeds and independent trajectories."""
    ref = frames[0].get_positions()
    syms = np.array(frames[0].get_chemical_symbols())
    msd_t = []
    for atoms in frames:
        disp = atoms.get_positions() - ref
        msd_t.append((disp ** 2).sum(axis=1))
    msd_t = np.array(msd_t)  # (n_frames, n_atoms)
    return {el: msd_t[:, syms == el].mean(axis=1) for el in set(syms)}


def compute_nve_drift(energies, timestep_fs, n_atoms):
    """energies: array of total energy per frame (eV). Returns drift in
    meV/atom/ns via a linear fit."""
    t_ns = np.arange(len(energies)) * timestep_fs * 1e-6
    e_per_atom_meV = (energies - energies.mean()) / n_atoms * 1000
    slope, intercept = np.polyfit(t_ns, e_per_atom_meV, 1)
    resid = e_per_atom_meV - (slope * t_ns + intercept)
    return float(slope), float(resid.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory")
    ap.add_argument("validation_profile", help="configs/validation_profile.yaml")
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--temperature-log", help="optional CSV with a total-energy column, for NVE drift")
    args = ap.parse_args()

    with open(args.validation_profile) as f:
        profile = yaml.safe_load(f)
    checks = profile.get("checks", [])
    thresholds = profile.get("thresholds", {})

    frames = read(args.trajectory, index=":")
    elements = sorted(set(frames[0].get_chemical_symbols()))
    print(f"loaded {len(frames)} frames, elements={elements}, checks={checks}")

    if "density" in checks:
        mean_d, std_d = compute_density(frames)
        target = thresholds.get("density_g_cm3", {})
        print(f"density: {mean_d:.4f} +/- {std_d:.4f} g/cm3"
              + (f"  (target {target.get('target')} +/- {target.get('tolerance')})" if target else ""))

    if "rdf" in checks:
        r, rdfs = compute_rdf(frames, elements)
        for pair, g in rdfs.items():
            peak_r = r[np.argmax(g)]
            print(f"rdf[{pair}]: first-peak r ~= {peak_r:.3f} A (max g(r)={g.max():.2f})")

    if "coordination" in checks:
        cutoffs = thresholds.get("coordination_cutoffs_angstrom", {"default": 3.0})
        cn = compute_coordination(frames, elements, cutoffs)
        print(f"coordination: {cn}")

    if "msd" in checks:
        msd = compute_msd(frames)
        for el, series in msd.items():
            print(f"msd[{el}]: final={series[-1]:.4f} A^2 "
                  f"(non-diffusive plateau expected: {thresholds.get('msd_diffusive', 'unspecified')})")

    if "nve_drift" in checks:
        if args.temperature_log:
            import csv
            with open(args.temperature_log) as f:
                energies = np.array([float(row["total_energy"]) for row in csv.DictReader(f)])
        else:
            energies = np.array([a.get_potential_energy() for a in frames])
        drift, resid_std = compute_nve_drift(energies, args.timestep_fs, len(frames[0]))
        max_abs = thresholds.get("nve_drift_meV_per_atom_per_ns", {}).get("max_abs")
        flag = "" if max_abs is None else ("PASS" if abs(drift) < max_abs else "FAIL")
        print(f"nve_drift: {drift:+.4f} +/- {resid_std:.4f} meV/atom/ns {flag}")

    for c in ("adf", "sq_fsdp"):
        if c in checks:
            print(f"{c}: not implemented in this generic script — see the "
                  f"# TODO markers in validation/structure_dynamics.py to add "
                  f"a material-specific implementation, or port one from your "
                  f"own analysis pipeline.")


if __name__ == "__main__":
    main()
