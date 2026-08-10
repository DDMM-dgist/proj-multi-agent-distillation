# PC003 — SIMPLE-NN Student Workflow (PREPARED, not launched)

**State: `PC003 = PREPARED` · `NEW_PIPELINE_CURRENT_STUDENT = NONE`.** No student training launched.
Historical students are reference/baseline only. Final student training waits for the FINAL teacher
(join point). This recovers the workflow schema from local gpu2 assets sufficiently to run later.

Source of truth: the per-seed SIMPLE-NN bundles under `gpu_return_v5_committee/…/seed0{1..4}/` and
`v6_return_package/student/` (identical `input.yaml`/`params_*` across seeds).

## Input / preprocessing schema
- **SIMPLE-NN v2.0.0** (PyTorch). `data.type: symmetry_function`.
- Feature generation from a **`str_list`** of structures → per-structure **pickled symmetry-function
  feature files**; `train_list` / `valid_list` are text files pointing at those pickles.
- **Symmetry functions:** `params_Si`, `params_O`, **70 SF per element = 16 type-2 G2 (radial) + 54
  type-4 G4 (angular)**; **cutoff 5.0 Å** (matches the teacher). `use_scale: True` (→ `scale_factor`),
  `use_pca: True` (→ `pca`), `double_precision: True`.
- **Required teacher-label format:** each structure needs `energy` (eV, total) + `forces` (eV/Å)
  from the teacher; **no stress, no DFT anchor** (`read_stress: False`, `use_stress: False`). This
  is exactly what the distillation labeling (PC002 final) must emit.

## Student architecture / training config (frozen from `input.yaml`)
`nodes: '30-30'` · `acti_func: sigmoid` · optimizer Adam · `batch_size: 32` · `total_epoch: 2000`
· `learning_rate: 1e-4` · `l2_regularization: 1e-6` · `use_force: True` · `energy_coeff: 1.0` ·
`force_coeff: 0.1` · `E_loss_type: 1` · `F_loss_type: 1` · `use_gpu: True` · per-seed `random_seed`
(v6 committee: **234 / 345 / 555 / 777**).

## Training command / resources / outputs
- Preprocess: `generate_features` from `str_list` → feature pickles (+ `scale_factor`, `pca`).
- Train: SIMPLE-NN `run.py` reading `input.yaml` (train_list/valid_list). 4 seeds = a committee.
- Resources: 1 GPU/seed; historical wall **~37–89 h/seed** (2000 epochs); committee as an `afterok`
  SLURM chain. Output: `potential_saved_bestmodel` (per seed, ~330 KB, LAMMPS-deployable),
  `LOG_seed0N`, `scale_factor`, `pca`.
- **LAMMPS deployment:** `pair_style nn` + `pair_coeff * * potential_saved_bestmodel O Si` (element
  order **O Si**), `units metal`, `timestep 0.001`.

## Deterministic validation hooks
Committee u-disagreement: `eval_committee_umax_v{5,6}.py` (per-atom force disagreement,
u_α = √Σ_xyz Var(F over 4 models)); gate `u_max_mean < 0.30 eV/Å` on `sphere_x012` production MD.

## Blocked-on-final-teacher
Final labeling of the PC002 structural pool (energy+forces), feature generation on those labels,
and the 4-seed committee training. Do NOT run until FINAL_TEACHER_IDENTITY_RESOLVED.

## Not local (flag)
The exact GPU-side train command + `str_list`/feature pickles live in gpu2
`distillation_runs/.../05_TRAINING_v6/` — recover at execution time; `input.yaml` here is the config
SIMPLE-NN actually reads.
