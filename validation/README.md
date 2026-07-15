# validation/

Scripts that judges and the analyst point to as evidence. All three operate on
**ASE-readable structures/trajectories and labels**, never on model internals
— porting to a new teacher/student/material means pointing them at new data
and a new `configs/validation_profile.yaml`, not editing the scripts.

| Script | What it checks | Input |
|---|---|---|
| `four_channel_audit.py` | teacher-vs-DFT / student-vs-teacher / student-vs-DFT MAE, RMSE, R² (the core diagnostic — see `agents/analyst.md`) | one extxyz with `dft_*`/`teacher_*`/`student_*` labels |
| `committee_uncertainty.py` | does committee σ_F rank the true student-teacher error? (Pearson r, Spearman ρ, top-decile enrichment) | one extxyz with per-seed `student_forces_seedNN` arrays + `teacher_forces` |
| `structure_dynamics.py` | RDF, coordination, density, MSD, NVE drift — driven by `configs/validation_profile.yaml`'s `checks`/`thresholds` | an MD trajectory (`.traj`, `.xyz`, ...) + the validation profile |

Channel (d) — student-MD-trajectory vs DFT single-points, the "does the
student stay accurate where it's actually deployed" check — is computed by
running `four_channel_audit.py` on DFT-labeled snapshots carved from a
production trajectory (see `agents/simulation.md`'s small-cell recipe), not by
a separate script.

`adf` and `sq_fsdp` in `structure_dynamics.py` are left as extension points —
their natural form is more material-specific than RDF/coordination/density/MSD/NVE.
