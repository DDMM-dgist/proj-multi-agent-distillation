# PC005 — Physical Validation Protocol (PREPARED; no new MD on this CPU server)

**State: `PC005 = PREPARED`.** The downstream physical-validation protocol is defined and its
scripts + historical baselines are inventoried. No new expensive MD is launched here; new MD runs
only when a NEW student potential exists (potential-dependent observables must be recomputed then).

## Observables, scripts, and historical baselines (reusable scripts in-repo)
| observable | script (reusable) | historical baseline (current/v5/v6 student) |
|---|---|---|
| RDF, CN, ADF, S(Q), FSDP, density | `production_12288/validation_out/validate_glass.py` | PASS 12/0/0: ρ 2.217 g/cm³; Si-O 1.610 Å; CN(Si) 4.001, CN(O) 2.001; ADF O-Si-O 109.4°, Si-O-Si 141.4°; FSDP Q≈1.58–1.60 Å⁻¹ |
| MSD / diffusion | `production_12288/melt_msd/compute_melt_msd.py` | 4000 K: D_Si 1.88e-9, D_O 2.67e-9 m²/s (R²≈0.996) |
| NVE energy drift | `teacher_diag/nve_drift/analyze_nve.py` | −0.005±0.007 meV/atom/ns (PASS ≪1) |
| structural stability | (NVE 500 ps) | max per-atom drift 1.09–1.25 Å, 0 anomalous atoms |
| defect-specific (SiO₂-x) | `sio2x_production/analysis_out/04_analyze.py` | per-config npz for pristine/random/sphere/plane × x006/x012 |
| persistent homology (voids) | `ph_analysis/run_ph.py` | pd0/1/2 birth-death per config |
| mechanics / softness | `sio2x_production/mechanics_out/run_aqs_softness.py` | nonaffine D²min per config |

## Protocol (MD settings, frozen from historical runs)
- Engine LAMMPS, `pair_style nn` (`O Si` order), `units metal`, `dt 0.001 ps`.
- Glass: NPT melt-quench, 1 K/ps; RDF over ~101 frames, ADF ~11 frames.
- NVE drift: NVT 300 K 10 ps → NVE 100 ps, dt 1 fs; drift threshold ≪ 1 meV/atom/ns.
- Defect analysis: `random`/`sphere`/`plane` modes at x = 0.06, 0.12 (structures frozen in PC002
  manifest as `prod:*`).

## Reuse vs recompute
- **Reusable as-is:** all scripts + the RDF/ADF/S(Q)/density/MSD/NVE/PH/mechanics **CSVs/npz** for
  the current/v5/v6 students — these are the comparison baselines.
- **Must recompute for a NEW student:** every observable that ran LAMMPS with a specific
  `potential_saved_bestmodel` is potential-dependent; PC005 re-runs the scripts above against the
  new committee **once a new student exists** (join point + PC003 complete). Not on this CPU server.

## Blocked-on
A new student potential (PC003). Protocol, scripts, settings, and baselines are ready now.
