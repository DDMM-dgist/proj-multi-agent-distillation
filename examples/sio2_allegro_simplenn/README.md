# Worked example: SiO2, Allegro → SIMPLE-NN

This is the instance this toolkit was extracted from — the one case run
end-to-end so far (see the manuscript, *"Auditable Multi-Agent Orchestration
for MLIP Distillation"*).

## What's here vs. what isn't

The concrete configs for this instance are `configs/examples/teacher.allegro.yaml`,
`configs/examples/student.simple-nn.yaml`, `configs/examples/uncertainty.yaml`,
`configs/examples/md_backend.yaml`, `configs/examples/reference_dft.yaml`, and
`configs/examples/validation_profile.yaml` — copy those into `configs/` (drop
the `examples/` level, keep the filenames or rename to `configs/teacher.yaml`
etc.) to reproduce this instance's *setup*.

**Not duplicated here** (too large / instance-specific / licensed):
- The teacher checkpoint, student committee checkpoints.
- The full dataset (10k+ augmented structures, the DFT-labeled seed pool, the
  uncertainty-selected DFT anchor cells).
- Production MD trajectories and the full result-CSV audit trail.
- `POTCAR` files (see `NOTICE.md`).

All of the above live in the original research repository for this project.
If you have access to it, the mapping is:

| This toolkit | Original repo location |
|---|---|
| `agents/*.md` | `.claude/agents/*.md` |
| `gates/gate_vote.workflow.js` | `gates/gate_vote.workflow.js` |
| `validation/four_channel_audit.py` | `teacher_diag/run_task_a.py`, `run_error_b_clean.py`, `run_error_c.py` (genericized/merged here) |
| `validation/committee_uncertainty.py` | `sio2x_production/committee_u_out/run_production_committee_u.py` (genericized here) |
| `validation/structure_dynamics.py` | `sio2x_production/04_analyze.py` (genericized here) |
| `templates/lammps/*.in.template` | `sio2x_production/{prod_md,quenching}.in`, `teacher_diag/nve_drift/nve_drift.in` (parameterized here) |
| `templates/dft/INCAR.scan.template` | `dft_labeling/cell_*/scan/INCAR` (parameterized here) |
| Teacher checkpoint, dataset, results | `gpu_finetune_handoff/`, `sio2x_production/`, `teacher_diag/`, `coordination_log.csv`, ... |

## Porting to your own system

1. Start from `configs/examples/*.yaml`, keep what applies (the gate mechanism,
   the four-channel methodology), replace `kind: allegro`/`kind: simple-nn`
   with your own teacher/student, and replace `configs/examples/validation_profile.yaml`'s
   silica-specific checks with your material's.
2. If your teacher/student `kind` isn't implemented yet in `adapters/`, add it
   there (see `configs/README.md` for the interface each adapter function needs
   to satisfy).
