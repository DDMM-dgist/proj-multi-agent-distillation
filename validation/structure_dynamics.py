#!/usr/bin/env python3
"""Structure/dynamics validation against configs/validation_profile.yaml.
Operates purely on an ASE-readable trajectory — never touches model
internals, so it needs no per-teacher/student-`kind` adapter.

Implements the checks common across the toolkit's worked examples:
  rdf, coordination, density, msd, nve_drift
(adf and sq_fsdp are left as extension points because their exact form is more material-specific
than the others.)

Usage:
    python validation/structure_dynamics.py trajectory.traj configs/validation_profile.yaml \
        [--timestep-fs 1.0] [--temperature-log energies.csv]
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from ase.io import read
from ase.data import atomic_numbers
from ase.geometry.rdf import get_rdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation.report import evidence_record, make_check


def compute_rdf(frames, elements, r_max=6.0, nbins=200):
    """Partial RDFs for each element pair present, averaged over frames."""
    pairs = list(itertools.combinations_with_replacement(sorted(elements), 2))
    out = {}
    r = None
    for e1, e2 in pairs:
        label = f"{e1}-{e2}"
        rdfs = []
        grid = None
        # Convert chemical symbols to atomic numbers for ASE get_rdf's `elements`
        # selector while preserving symbol-based report keys. ASE 3.29 accepts the
        # symbol form, whereas ASE 3.26 does not (it matches zero atoms -> divide by
        # zero -> non-finite RDF); atomic numbers work across the declared range.
        pair = [atomic_numbers[e1], atomic_numbers[e2]]
        for atoms in frames:
            rdf, distances = get_rdf(atoms, r_max, nbins, elements=pair)
            rdf = np.asarray(rdf, dtype=float)
            distances = np.asarray(distances, dtype=float)
            # Validate every frame result; never silently zero/truncate/interpolate.
            if rdf.ndim != 1 or distances.ndim != 1:
                raise ValueError(f"RDF for pair {label} is not one-dimensional")
            if rdf.shape[0] != nbins or distances.shape[0] != nbins:
                raise ValueError(
                    f"RDF for pair {label} has the wrong length (expected {nbins}, "
                    f"got rdf={rdf.shape[0]}, distances={distances.shape[0]})")
            if not np.isfinite(rdf).all() or not np.isfinite(distances).all():
                raise ValueError(
                    f"non-finite RDF for pair {label}; check r_max against the cell "
                    f"size and that both species are present in the frames")
            if grid is None:
                grid = distances
                if r is None:
                    r = distances
            elif not np.allclose(distances, grid, rtol=0, atol=1e-9):
                raise ValueError(
                    f"distance grid mismatch across frames for pair {label}")
            rdfs.append(rdf)
        mean_rdf = np.mean(rdfs, axis=0)
        if not np.isfinite(mean_rdf).all():
            raise ValueError(f"non-finite mean RDF for pair {label}")
        out[label] = mean_rdf
    return r, out


def _smooth_1d(y, window):
    """Deterministic moving-average smoother. window MUST be odd; window=1 is
    identity (no smoothing). Used for first-peak / first-minimum detection so
    the algorithm does not pick up single-bin noise."""
    if window <= 1:
        return np.asarray(y, dtype=float)
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


def rdf_first_peak_and_minimum(r, g, *, smoothing_window=5,
                                min_r_A=None, max_r_A=None):
    """Deterministic first-peak-position + first-peak-height + first-minimum-
    position extraction.

    Selection semantics (documented, not brittle):
      * apply a moving-average smoother of odd window ``smoothing_window`` to
        g(r); default = 5 bins (~0.15 A on the standard 0-6 A / 200-bin grid);
      * restrict the search window to [min_r_A, max_r_A] if given, else the
        full grid;
      * FIRST-PEAK = argmax of smoothed g(r) over the window;
      * FIRST-MINIMUM = argmin of smoothed g(r) over the sub-window starting
        immediately after the first-peak bin and ending at the window end.

    Returns a dict with `r_first_peak_A`, `g_first_peak`, `r_first_min_A`,
    `g_first_min`, `smoothing_window_bins`, and the search window used.
    Raises if any required quantity is not identifiable in the smoothed
    series (e.g., monotone decreasing).
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    if r.shape != g.shape or r.ndim != 1:
        raise ValueError("r and g must be 1-D arrays of matching length")
    g_smooth = _smooth_1d(g, smoothing_window)
    lo = 0 if min_r_A is None else int(np.searchsorted(r, min_r_A, side="left"))
    hi = len(r) if max_r_A is None else int(np.searchsorted(r, max_r_A, side="right"))
    if hi - lo < 3:
        raise ValueError("first-peak search window too narrow")
    peak_idx_local = int(np.argmax(g_smooth[lo:hi]))
    peak_idx = lo + peak_idx_local
    # First minimum: search strictly AFTER the peak, within the same window.
    min_lo = peak_idx + 1
    min_hi = hi
    if min_hi - min_lo < 2:
        raise ValueError("first-minimum search window (post-peak) too narrow")
    min_idx_local = int(np.argmin(g_smooth[min_lo:min_hi]))
    min_idx = min_lo + min_idx_local
    return {
        "r_first_peak_A": float(r[peak_idx]),
        "g_first_peak": float(g_smooth[peak_idx]),
        "g_first_peak_raw": float(g[peak_idx]),
        "r_first_min_A": float(r[min_idx]),
        "g_first_min": float(g_smooth[min_idx]),
        "smoothing_window_bins": int(smoothing_window),
        "search_window_A": [float(r[lo]), float(r[min(hi - 1, len(r) - 1)])],
        "peak_index_bin": peak_idx,
        "first_min_index_bin": min_idx,
    }


def compute_rdf_v2(frames, center_species, neighbor_species,
                   *, r_max=6.0, nbins=200):
    """Typed RDF for a single ordered species pair.

    Distinct from ``compute_rdf`` which computes all combinations-with-
    replacement pairs; ``compute_rdf_v2`` computes the ORDERED pair
    (center -> neighbor). The returned r-grid + g(r) can be piped through
    ``rdf_first_peak_and_minimum`` to extract chemistry-relevant scalars.

    Returns dict with `r_A`, `g_of_r`, `bin_width_A`, `r_max_A`, `nbins`,
    `center_species`, `neighbor_species`, `n_frames`.
    """
    if not frames:
        raise ValueError("compute_rdf_v2 requires at least one frame")
    pair = [atomic_numbers[center_species], atomic_numbers[neighbor_species]]
    rdfs = []
    grid = None
    for atoms in frames:
        g_frame, r_frame = get_rdf(atoms, r_max, nbins, elements=pair)
        r_frame = np.asarray(r_frame, dtype=float)
        g_frame = np.asarray(g_frame, dtype=float)
        if g_frame.shape[0] != nbins or r_frame.shape[0] != nbins:
            raise ValueError("RDF shape mismatch")
        if not np.isfinite(g_frame).all() or not np.isfinite(r_frame).all():
            raise ValueError("non-finite RDF sample")
        if grid is None:
            grid = r_frame
        elif not np.allclose(r_frame, grid, rtol=0, atol=1e-9):
            raise ValueError("distance grid mismatch across frames")
        rdfs.append(g_frame)
    mean_g = np.mean(rdfs, axis=0)
    bin_width = float(r_max) / float(nbins)
    return {
        "r_A": grid.tolist(),
        "g_of_r": [float(v) for v in mean_g],
        "bin_width_A": bin_width,
        "r_max_A": float(r_max),
        "nbins": int(nbins),
        "center_species": str(center_species),
        "neighbor_species": str(neighbor_species),
        "n_frames": int(len(frames)),
    }


def compute_species_coordination(frames, center_species, neighbor_species,
                                 cutoff_A, *,
                                 cutoff_source_ref=None,
                                 cutoff_frozen_before_student=None,
                                 max_topology=8):
    """Species-specific coordination.

    Counts, for each atom of ``center_species`` in every frame, the number of
    ``neighbor_species`` atoms within ``cutoff_A`` under minimum-image
    convention. Returns per-frame mean, aggregate mean, and a
    coordination-number histogram over 0..``max_topology``.

    Fail-closed: ``cutoff_A`` must be an explicit positive number; there is
    no default. ``cutoff_source_ref`` and ``cutoff_frozen_before_student``
    are accepted so the caller can pass through the policy's provenance
    fields, but they are ADVISORY here — the *policy layer* enforces the
    frozen-before-student invariant at contract construction; this function
    simply records the values in its returned metadata for downstream
    provenance surfacing.
    """
    if not frames:
        raise ValueError("compute_species_coordination requires at least one frame")
    if not (isinstance(cutoff_A, (int, float)) and float(cutoff_A) > 0):
        raise ValueError("cutoff_A must be a positive real number "
                          "(species-specific coordination has no default cutoff)")
    cutoff_A = float(cutoff_A)
    per_frame_means = []
    histogram = np.zeros(int(max_topology) + 1, dtype=int)
    n_center_total = 0
    for atoms in frames:
        d = atoms.get_all_distances(mic=True)
        syms = np.array(atoms.get_chemical_symbols())
        center_idx = np.where(syms == center_species)[0]
        neighbor_mask_by_j = (syms == neighbor_species)
        if len(center_idx) == 0:
            continue
        cns = []
        for i in center_idx:
            neighbors = np.where(neighbor_mask_by_j & (d[i] < cutoff_A) & (d[i] > 0))[0]
            n = int(len(neighbors))
            cns.append(n)
            if n <= max_topology:
                histogram[n] += 1
            else:
                histogram[-1] += 1  # collapse anything above the reported range
        per_frame_means.append(float(np.mean(cns)) if cns else 0.0)
        n_center_total += len(center_idx)
    if not per_frame_means:
        raise ValueError(f"no {center_species} atoms present in any frame")
    aggregate_mean = float(np.mean(per_frame_means))
    total_counted = int(histogram.sum())
    fractions = {int(k): float(histogram[k]) / total_counted
                 for k in range(len(histogram)) if histogram[k] > 0} if total_counted else {}
    return {
        "center_species": str(center_species),
        "neighbor_species": str(neighbor_species),
        "cutoff_A": cutoff_A,
        "cutoff_source_ref": cutoff_source_ref,
        "cutoff_frozen_before_student": cutoff_frozen_before_student,
        "per_frame_mean_coordination": per_frame_means,
        "aggregate_mean_coordination": aggregate_mean,
        "coordination_histogram": [int(x) for x in histogram],
        "coordination_fractions": fractions,
        "n_center_atoms_summed": int(n_center_total),
        "n_frames": int(len(frames)),
        "max_topology_bin": int(max_topology),
    }


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
    if not frames:
        raise ValueError("MSD requires at least one trajectory frame")
    symbols = frames[0].get_chemical_symbols()
    if any(len(atoms) != len(frames[0]) or atoms.get_chemical_symbols() != symbols
           for atoms in frames[1:]):
        raise ValueError("MSD requires a fixed atom count and atom ordering")
    ref = frames[0].get_positions()
    syms = np.array(frames[0].get_chemical_symbols())
    previous_scaled = frames[0].get_scaled_positions(wrap=True)
    unwrapped = ref.copy()
    msd_t = [np.zeros(len(frames[0]))]
    for atoms in frames[1:]:
        scaled = atoms.get_scaled_positions(wrap=True)
        delta = scaled - previous_scaled
        delta[:, atoms.get_pbc()] -= np.round(delta[:, atoms.get_pbc()])
        unwrapped += delta @ atoms.cell.array
        msd_t.append(((unwrapped - ref) ** 2).sum(axis=1))
        previous_scaled = scaled
    msd_t = np.array(msd_t)  # (n_frames, n_atoms)
    return {el: msd_t[:, syms == el].mean(axis=1) for el in set(syms)}


def compute_nve_drift(energies, timestep_fs, n_atoms, sample_interval_steps=1, steps=None):
    """energies: array of total energy per frame (eV). Returns drift in
    meV/atom/ns via a linear fit."""
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1 or len(energies) < 2 or not np.isfinite(energies).all():
        raise ValueError("NVE drift requires at least two finite energy samples")
    if float(timestep_fs) <= 0 or int(n_atoms) <= 0:
        raise ValueError("NVE drift requires positive timestep and atom count")
    if int(sample_interval_steps) <= 0:
        raise ValueError("NVE drift sample interval must be positive")
    if steps is None:
        steps = np.arange(len(energies)) * int(sample_interval_steps)
    steps = np.asarray(steps, dtype=float)
    if steps.shape != energies.shape or not np.isfinite(steps).all() or len(np.unique(steps)) < 2:
        raise ValueError("NVE drift steps must be finite, distinct, and match the energies")
    t_ns = steps * timestep_fs * 1e-6
    e_per_atom_meV = (energies - energies.mean()) / n_atoms * 1000
    slope, intercept = np.polyfit(t_ns, e_per_atom_meV, 1)
    resid = e_per_atom_meV - (slope * t_ns + intercept)
    return float(slope), float(resid.std())


def read_energy_log(path):
    """Read the CSV emitted by nve_drift.in.template.

    A whitespace-delimited ``step temp pe ke etotal`` file is also accepted for
    compatibility with runs made before the template was standardized.
    """
    with open(path) as handle:
        lines = [line.strip() for line in handle
                 if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"energy log is empty: {path}")
    if "," in lines[0]:
        rows = list(csv.DictReader(lines))
        required = {"step", "total_energy"}
        if not rows or not required <= set(rows[0]):
            raise ValueError("energy CSV requires step and total_energy columns")
        return (np.array([int(float(row["step"])) for row in rows]),
                np.array([float(row["total_energy"]) for row in rows]))
    parsed = [line.split() for line in lines]
    if parsed[0][0].lower() == "step":
        parsed = parsed[1:]
    if not parsed or any(len(row) < 5 for row in parsed):
        raise ValueError("whitespace energy log requires step temp pe ke etotal columns")
    return (np.array([int(float(row[0])) for row in parsed]),
            np.array([float(row[4]) for row in parsed]))


def compute_diffusivity(msd_by_species, timestep_fs, *, fit_start_frame, fit_end_frame,
                        sample_interval_steps=1, n_dims=3):
    """Einstein self-diffusivity from an EXPLICIT, provenance-bound MSD fit window.

    ``msd_by_species`` is the per-species MSD series dict returned by
    ``compute_msd`` (each value shape ``(n_frames,)``, units A^2). The MSD is
    fitted linearly, MSD(t) = 2*n_dims*D*t + c, over the half-open frame window
    ``[fit_start_frame, fit_end_frame)`` and D read off the slope. There is NO
    default window: the caller MUST declare it, and the window (in frames and in
    fs) is returned so it can be frozen into the validation target — a
    diffusivity number is meaningless without the window it was fit over.

    Time is reconstructed deterministically as
    ``t_fs = frame_index * sample_interval_steps * timestep_fs``. Fails closed on
    a malformed window, non-finite MSD, or non-positive timestep.
    """
    if not msd_by_species:
        raise ValueError("compute_diffusivity requires a non-empty per-species MSD mapping")
    if not (isinstance(timestep_fs, (int, float)) and float(timestep_fs) > 0):
        raise ValueError("compute_diffusivity requires positive timestep_fs")
    if int(sample_interval_steps) <= 0:
        raise ValueError("sample_interval_steps must be positive")
    if int(n_dims) not in (1, 2, 3):
        raise ValueError("n_dims must be 1, 2, or 3")
    s0, s1 = int(fit_start_frame), int(fit_end_frame)
    out = {}
    for species, series in msd_by_species.items():
        series = np.asarray(series, dtype=float)
        n = len(series)
        if not (0 <= s0 < s1 <= n):
            raise ValueError(
                f"invalid MSD fit window [{s0}, {s1}) for species {species!r} with {n} frames")
        if s1 - s0 < 2:
            raise ValueError("MSD fit window must contain at least two frames")
        window = series[s0:s1]
        if not np.isfinite(window).all():
            raise ValueError(f"non-finite MSD inside the fit window for species {species!r}")
        frame_idx = np.arange(s0, s1, dtype=float)
        t_fs = frame_idx * float(sample_interval_steps) * float(timestep_fs)
        slope, intercept = np.polyfit(t_fs, window, 1)  # A^2 / fs
        resid = window - (slope * t_fs + intercept)
        d_A2_per_fs = float(slope) / (2.0 * int(n_dims))
        out[str(species)] = {
            "diffusivity_A2_per_fs": float(d_A2_per_fs),
            "diffusivity_A2_per_ps": float(d_A2_per_fs * 1000.0),
            # 1 A^2/fs = (1e-8 cm)^2 / (1e-15 s) = 0.1 cm^2/s
            "diffusivity_cm2_per_s": float(d_A2_per_fs * 0.1),
            "msd_slope_A2_per_fs": float(slope),
            "fit_intercept_A2": float(intercept),
            "fit_residual_std_A2": float(resid.std()),
            "fit_window_frames": [s0, s1],
            "fit_window_t_fs": [float(t_fs[0]), float(t_fs[-1])],
            "n_fit_points": int(s1 - s0),
            "n_dims": int(n_dims),
            "sample_interval_steps": int(sample_interval_steps),
            "timestep_fs": float(timestep_fs),
        }
    return out


def compute_adf(frames, center_species, neighbor_species, *, r_cut_A,
                nbins=180, angle_min_deg=0.0, angle_max_deg=180.0):
    """Angular distribution function for neighbor-center-neighbor triplets.

    For each atom of ``center_species`` this considers every pair of neighbouring
    ``neighbor_species`` atoms within ``r_cut_A`` (minimum-image convention) and
    histograms the enclosed bond angle. It is fully generic: ``center_species``
    is a symbol, ``neighbor_species`` is a symbol or a sequence of symbols, and
    the angular window is configurable — no material-specific triplet (O-Si-O,
    etc.) is assumed anywhere. Fails closed on a malformed cutoff/bin/angle
    range or an absent center species; a geometry that yields zero triplets is
    reported honestly (``n_triplets == 0``, ``mean_angle_deg is None``) rather
    than fabricated.
    """
    import itertools as _itertools
    center_species = str(center_species)
    if isinstance(neighbor_species, (list, tuple, set)):
        neighbor_set = {str(s) for s in neighbor_species}
    else:
        neighbor_set = {str(neighbor_species)}
    if not neighbor_set:
        raise ValueError("compute_adf requires at least one neighbor species")
    if not (isinstance(r_cut_A, (int, float)) and float(r_cut_A) > 0):
        raise ValueError("compute_adf requires a positive r_cut_A")
    nbins = int(nbins)
    if nbins <= 0:
        raise ValueError("compute_adf requires positive nbins")
    lo, hi = float(angle_min_deg), float(angle_max_deg)
    if not (0.0 <= lo < hi <= 180.0):
        raise ValueError("compute_adf angle range must satisfy 0 <= min < max <= 180")
    if not frames:
        raise ValueError("compute_adf requires at least one frame")
    edges = np.linspace(lo, hi, nbins + 1)
    counts = np.zeros(nbins, dtype=float)
    n_triplets = 0
    n_center_seen = 0
    for atoms in frames:
        syms = np.array(atoms.get_chemical_symbols())
        d = atoms.get_all_distances(mic=True)
        center_idx = np.where(syms == center_species)[0]
        n_center_seen += int(len(center_idx))
        for i in center_idx:
            neigh = [j for j in range(len(atoms))
                     if j != i and syms[j] in neighbor_set and 0.0 < d[i, j] < float(r_cut_A)]
            for a, b in _itertools.combinations(neigh, 2):
                va = atoms.get_distance(i, a, mic=True, vector=True)
                vb = atoms.get_distance(i, b, mic=True, vector=True)
                na = float(np.linalg.norm(va))
                nb = float(np.linalg.norm(vb))
                if na == 0.0 or nb == 0.0:
                    continue
                cos = float(np.dot(va, vb) / (na * nb))
                cos = max(-1.0, min(1.0, cos))
                ang = float(np.degrees(np.arccos(cos)))
                if lo <= ang <= hi:
                    k = int(np.searchsorted(edges, ang, side="right") - 1)
                    k = min(max(k, 0), nbins - 1)
                    counts[k] += 1.0
                    n_triplets += 1
    if n_center_seen == 0:
        raise ValueError(f"no {center_species} atoms present in any frame")
    total = float(counts.sum())
    bin_centers = ((edges[:-1] + edges[1:]) / 2.0)
    distribution = (counts / total).tolist() if total > 0 else [0.0] * nbins
    return {
        "center_species": center_species,
        "neighbor_species": sorted(neighbor_set),
        "r_cut_A": float(r_cut_A),
        "nbins": nbins,
        "angle_range_deg": [lo, hi],
        "bin_centers_deg": [float(x) for x in bin_centers],
        "distribution": [float(x) for x in distribution],
        "counts": [float(x) for x in counts],
        "n_triplets": int(n_triplets),
        "n_frames": int(len(frames)),
        "mean_angle_deg": (float(np.average(bin_centers, weights=counts))
                           if total > 0 else None),
    }


def compute_md_stability_summary(frames, *, energies=None, forces_key="forces",
                                 min_separation_floor_A=0.5,
                                 max_force_ceiling_eV_A=None, temperatures=None):
    """Model-independent MD / trajectory stability summary.

    Every quantity is derived deterministically from the trajectory (and, where
    supplied, an explicit energy or temperature series); nothing is invented. A
    quantity that cannot be measured from the available data is reported as
    unavailable rather than defaulted:
      * ``energies_finite`` — only meaningful when ``energies`` is supplied;
      * ``forces_available`` / ``max_force_eV_per_A`` — read from
        ``frames[k].arrays[forces_key]`` when present, never recomputed by a model;
      * ``min_interatomic_separation_A`` — minimum over all frames (MIC);
      * ``temperature_available`` / ``temperature_mean_K`` /
        ``temperature_relative_std`` — from a supplied ``temperatures`` series, or
        from per-frame momenta via ASE when present, else unavailable.

    ``catastrophic_failure`` is True (with ``failure_reasons``) if any frame's
    minimum separation drops below ``min_separation_floor_A``, any supplied
    energy or present force is non-finite, or (when ``max_force_ceiling_eV_A`` is
    given) the maximum force magnitude exceeds it.
    """
    if not frames:
        raise ValueError("compute_md_stability_summary requires at least one frame")
    failure_reasons = []

    # --- energies (only when explicitly supplied) ---
    energies_finite = None
    if energies is not None:
        arr = np.asarray(energies, dtype=float)
        energies_finite = bool(np.isfinite(arr).all())
        if not energies_finite:
            failure_reasons.append("non_finite_energy")

    # --- forces (read from arrays only; never model-recomputed) ---
    forces_available = False
    max_force = None
    for atoms in frames:
        f = atoms.arrays.get(forces_key) if hasattr(atoms, "arrays") else None
        if f is None:
            continue
        forces_available = True
        f = np.asarray(f, dtype=float)
        if not np.isfinite(f).all():
            failure_reasons.append("non_finite_force")
            continue
        mag = float(np.sqrt((f ** 2).sum(axis=1)).max()) if f.size else 0.0
        max_force = mag if max_force is None else max(max_force, mag)
    if (max_force_ceiling_eV_A is not None and max_force is not None
            and max_force > float(max_force_ceiling_eV_A)):
        failure_reasons.append("max_force_exceeds_ceiling")

    # --- minimum interatomic separation (MIC) ---
    min_sep = None
    for atoms in frames:
        if len(atoms) < 2:
            continue
        d = atoms.get_all_distances(mic=True)
        iu = np.triu_indices(len(atoms), k=1)
        frame_min = float(d[iu].min())
        min_sep = frame_min if min_sep is None else min(min_sep, frame_min)
    if min_sep is not None and min_sep < float(min_separation_floor_A):
        failure_reasons.append("min_separation_below_floor")

    # --- temperature stability (explicit series, else momenta, else unavailable) ---
    temperature_available = False
    temp_mean = None
    temp_rel_std = None
    temp_series = None
    if temperatures is not None:
        temp_series = np.asarray(temperatures, dtype=float)
    else:
        derived = []
        for atoms in frames:
            try:
                if atoms.get_momenta() is not None and np.any(atoms.get_momenta()):
                    derived.append(float(atoms.get_temperature()))
            except Exception:
                derived = []
                break
        if derived and len(derived) == len(frames):
            temp_series = np.asarray(derived, dtype=float)
    if temp_series is not None and temp_series.size and np.isfinite(temp_series).all():
        temperature_available = True
        temp_mean = float(temp_series.mean())
        temp_rel_std = float(temp_series.std() / abs(temp_mean)) if temp_mean != 0 else None

    return {
        "n_frames": int(len(frames)),
        "n_atoms": int(len(frames[0])),
        "energies_finite": energies_finite,
        "forces_available": bool(forces_available),
        "max_force_eV_per_A": (float(max_force) if max_force is not None else None),
        "max_force_ceiling_eV_A": (float(max_force_ceiling_eV_A)
                                   if max_force_ceiling_eV_A is not None else None),
        "min_interatomic_separation_A": (float(min_sep) if min_sep is not None else None),
        "min_separation_floor_A": float(min_separation_floor_A),
        "temperature_available": bool(temperature_available),
        "temperature_mean_K": temp_mean,
        "temperature_relative_std": temp_rel_std,
        "catastrophic_failure": bool(failure_reasons),
        "failure_reasons": sorted(set(failure_reasons)),
    }


# --- material-specific observable plugin interface -------------------------------------------
# Built-in observable kinds are dispatched by validation.teacher_physical_validation; a campaign
# that needs a phase/defect/surface observable the generic engine does not implement registers it
# here (a callable ``fn(frames, params, context) -> dict``) WITHOUT editing the engine or
# hard-coding a material. Registration is fail-closed against clobbering a built-in or an existing
# plugin so an extension can never silently redefine a shared observable.
BUILTIN_OBSERVABLE_KINDS = frozenset({
    "rdf_peak_position", "rdf_peak_height", "species_coordination",
    "coordination_distribution", "density", "msd", "diffusivity", "adf", "nve_drift",
})
_OBSERVABLE_PLUGINS = {}


def register_observable(kind, fn):
    """Register a material-specific observable implementation under a unique ``kind``.

    ``fn`` must be callable as ``fn(frames, params, context) -> dict``. Raises on a
    duplicate/clobbering registration (a built-in kind or an already-registered
    plugin) so a plugin can never silently override a shared observable."""
    kind = str(kind)
    if kind in BUILTIN_OBSERVABLE_KINDS:
        raise ValueError(f"cannot override built-in observable kind: {kind}")
    if kind in _OBSERVABLE_PLUGINS:
        raise ValueError(f"observable kind already registered: {kind}")
    if not callable(fn):
        raise ValueError("observable plugin must be callable")
    _OBSERVABLE_PLUGINS[kind] = fn
    return kind


def unregister_observable(kind):
    """Remove a previously registered plugin (no-op if absent). Never touches built-ins."""
    _OBSERVABLE_PLUGINS.pop(str(kind), None)


def observable_plugin(kind):
    """Return the registered plugin callable for ``kind``, or None."""
    return _OBSERVABLE_PLUGINS.get(str(kind))


def registered_observable_kinds():
    """Return the sorted set of currently registered plugin kinds (excludes built-ins)."""
    return sorted(_OBSERVABLE_PLUGINS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory")
    ap.add_argument("validation_profile", help="configs/validation_profile.yaml")
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--sample-interval-steps", type=int, default=1,
                    help="MD steps between trajectory frames when no energy log supplies step values")
    ap.add_argument("--temperature-log", help="optional CSV with a total-energy column, for NVE drift")
    ap.add_argument("--output", help="optional common ValidationReport JSON path")
    args = ap.parse_args()

    with open(args.validation_profile) as f:
        profile = yaml.safe_load(f)
    checks = profile.get("checks", [])
    thresholds = profile.get("thresholds", {})

    frames = read(args.trajectory, index=":")
    if not frames:
        raise ValueError("validation trajectory contains no frames")
    elements = sorted(set(frames[0].get_chemical_symbols()))
    report_checks = []
    print(f"loaded {len(frames)} frames, elements={elements}, checks={checks}")

    if "density" in checks:
        mean_d, std_d = compute_density(frames)
        target = thresholds.get("density_g_cm3", {})
        criterion = None
        if target.get("target") is not None and target.get("tolerance") is not None:
            criterion = {"operator": "target_tolerance", "target": float(target["target"]),
                         "tolerance": float(target["tolerance"])}
        report_checks.append(make_check("structure", "density", mean_d, "g/cm3", criterion,
                                        details={"standard_deviation": std_d}))
        print(f"density: {mean_d:.4f} +/- {std_d:.4f} g/cm3"
              + (f"  (target {target.get('target')} +/- {target.get('tolerance')})" if target else ""))

    if "rdf" in checks:
        r, rdfs = compute_rdf(frames, elements)
        rdf_targets = thresholds.get("rdf_peak_angstrom", {})
        for pair, g in rdfs.items():
            peak_r = r[np.argmax(g)]
            target = rdf_targets.get(pair, {})
            criterion = None
            if target.get("target") is not None and target.get("tolerance") is not None:
                criterion = {"operator": "target_tolerance", "target": float(target["target"]),
                             "tolerance": float(target["tolerance"])}
            report_checks.append(make_check("structure", f"rdf_peak:{pair}", float(peak_r),
                                            "Angstrom", criterion, details={"max_g_r": float(g.max())}))
            print(f"rdf[{pair}]: first-peak r ~= {peak_r:.3f} A (max g(r)={g.max():.2f})"
                  + (f"  (target {target.get('target')} +/- {target.get('tolerance')})" if target else ""))

    if "coordination" in checks:
        cutoffs = thresholds.get("coordination_cutoffs_angstrom", {"default": 3.0})
        cn = compute_coordination(frames, elements, cutoffs)
        for element, value in cn.items():
            report_checks.append(make_check("structure", f"coordination:{element}", value,
                                            "neighbors"))
        print(f"coordination: {cn}")

    if "msd" in checks:
        msd = compute_msd(frames)
        for el, series in msd.items():
            report_checks.append(make_check("dynamics", f"msd_final:{el}",
                                            float(series[-1]), "Angstrom^2"))
            print(f"msd[{el}]: final={series[-1]:.4f} A^2 "
                  f"(non-diffusive plateau expected: {thresholds.get('msd_diffusive', 'unspecified')})")

    if "nve_drift" in checks:
        if args.temperature_log:
            steps, energies = read_energy_log(args.temperature_log)
        else:
            energies = np.array([a.get_total_energy() for a in frames])
            steps = None
        drift, resid_std = compute_nve_drift(
            energies, args.timestep_fs, len(frames[0]),
            sample_interval_steps=args.sample_interval_steps, steps=steps,
        )
        max_abs = thresholds.get("nve_drift_meV_per_atom_per_ns", {}).get("max_abs")
        criterion = None if max_abs is None else {"operator": "max_abs",
                                                  "threshold": float(max_abs)}
        report_checks.append(make_check("stability", "nve_drift", drift,
                                        "meV/atom/ns", criterion,
                                        details={"residual_std": resid_std}))
        flag = "" if max_abs is None else ("PASS" if abs(drift) <= max_abs else "FAIL")
        print(f"nve_drift: {drift:+.4f} +/- {resid_std:.4f} meV/atom/ns {flag}")

    for c in ("adf", "sq_fsdp"):
        if c in checks:
            report_checks.append(make_check("structure", c, reason="no generic implementation"))
            print(f"{c}: not implemented in this generic script — add "
                  f"a material-specific validator, or port one from your "
                  f"own analysis pipeline.")

    if args.output:
        evidence = [evidence_record("trajectory", args.trajectory),
                    evidence_record("validation_profile", args.validation_profile)]
        if args.temperature_log:
            evidence.append(evidence_record("energy_log", args.temperature_log))
        payload = {"schema_version": 1, "profile": str(Path(args.validation_profile).resolve()),
                   "checks": report_checks, "evidence": evidence}
        Path(args.output).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
