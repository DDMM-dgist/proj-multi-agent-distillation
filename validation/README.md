# validation/

Scripts used to generate validation evidence from ASE-readable structures,
trajectories, and labels.

| Script | What it checks | Input |
|---|---|---|
| `four_channel_audit.py` | teacher-vs-DFT / student-vs-teacher / student-vs-DFT MAE, RMSE, R² (the core diagnostic — see `agents/analyst.md`) | one extxyz with `dft_*`/`teacher_*`/`student_*` labels |
| `committee_uncertainty.py` | does committee σ_F rank the true student-teacher error? Reports usable/skipped-frame coverage, committee size, configured top-fraction, and evidence hashes | one extxyz with per-seed `student_forces_seedNN` arrays + `teacher_forces` |
| `structure_dynamics.py` | RDF, coordination, density, MSD, NVE drift — driven by `configs/validation_profile.yaml`'s `checks`/`thresholds` | an MD trajectory (`.traj`, `.xyz`, ...) + the validation profile |
| `report.py` | common report status, threshold consistency, run-bound evidence hashes, and optional gate PASS policy | a ValidationReport JSON manifest |
| `surface_energy.py` | case validator for slab/bulk consistency, raw-evidence roles, and profile-bound surface deltas | a common ValidationReport containing `surface` entries |

Channel (d), student-MD-trajectory vs DFT single-points, is computed by running
`four_channel_audit.py` on DFT-labeled trajectory snapshots.

`adf` and `sq_fsdp` in `structure_dynamics.py` are left as extension points —
their natural form is more material-specific than RDF/coordination/density/MSD/NVE.
Use `structure_dynamics.py --output <report.json>` to emit the common report.
