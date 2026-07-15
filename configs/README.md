# configs/ — the adapter layer

This is the only place that should change when you port the workflow to a new
teacher, student, or material. Nothing in `agents/`, `gates/`, or `validation/`
should need editing.

Six config files define the workflow's model-independent interfaces. Each has
a `kind` field that `adapters/` dispatches on, plus interface-specific fields.
Support is reported at three levels: **verified** means an end-to-end reference
run exists outside this lightweight distribution; **implemented** means an
adapter code path exists but portability has not been demonstrated here;
**adapter-ready** means only the interface is documented. Do not collapse
these levels into a single "supported" claim.

| File | Required interface | Implemented `kind`s | Documented-only examples |
|---|---|---|---|
| `teacher.<name>.yaml` | given a structure → energy, forces, [stress] | `allegro` (verified reference); `mace` (implemented, unverified) | `nequip`, `gap`, `ace` |
| `student.<name>.yaml` | trainable; prediction via ASE factory; deployable to an MD engine | `simple-nn` reference; `grace-fs` runner/deployment (implemented, unverified) | `mtp`, `ace`, `compact-gnn` |
| `uncertainty.yaml` | per-configuration ranking score from a committee | `committee-force-std` (σ_F, architecture-agnostic — works for any committee of N same-`kind` students) | ensemble variance, dropout, extrapolation-grade scores |
| `md_backend.yaml` | run a trajectory from a deployed model | `lammps` | `ase-md`, model-native MD |
| `reference_dft.yaml` | first-principles energy/force/stress labels | `vasp` | other DFT codes (QE, CP2K, ...) via ASE calculators |
| `validation_profile.yaml` | deployment-relevant physical observables | generic RDF/coordination/density/MSD/NVE; silica ADF/FSDP are extension points | elasticity, diffusion, phonons, material-specific ADF/S(Q) |

## Interface contracts

### `teacher.<name>.yaml`
```yaml
kind: allegro                     # dispatch key -> adapters/teacher.py
checkpoint: /path/to/teacher.pth  # NOT committed to git (see NOTICE.md)
calculator:
  module: nequip.ase              # python import path
  class: NequIPCalculator
  env: distill-teacher-allegro    # which conda env's interpreter to invoke (see environment.yml)
emits_stress: true                # confirm empirically before relying on this (see adapters/teacher.py:check_stress_support)
```
Any teacher with an ASE `Calculator` satisfies this — that covers essentially
every modern MLIP (NequIP/Allegro, MACE, GAP via quippy, ACE via pyace, and
foundation models like MACE-MP-0/MatterSim/Orb).

### `student.<name>.yaml`
```yaml
kind: simple-nn
train:
  env: distill-student-simplenn
  config_template: templates/student/simple-nn.input.yaml.template
  descriptor_params: {Si: templates/student/params_Si, O: templates/student/params_O}
  total_epoch: 1500
  double_precision: false
  use_stress: true
committee:
  n_seeds: 4
deploy:
  lammps_pair_style: "nn"         # -> adapters/student.py:lammps_pair_style_block()
  elements: [Si, O]                # exact LAMMPS atom-type order; required
```
Adding a new student `kind` (e.g. an MTP via `mlip-2`) means implementing
`train_student(cfg)`, `load_student(cfg, seed)`, and, when ASE inference is
available, configuring a generic `predict.factory`; plus
`lammps_pair_style_block(cfg)` in `adapters/student.py` for that `kind` — the
agents (`ml-trainer.md`) already just call these functions.

The workflow exchanges architecture-neutral `ModelArtifact` and
`PredictionBatch` values from `adapters/contracts.py`. A prediction factory is
configured without modifying agent prompts:

```yaml
predict:
  factory: my_student_package.ase.make_calculator
  checkpoint_arg: checkpoint
  kwargs: {device: cpu}
```

Run `adapters.preflight.check_student_config(...)` before expensive training.

### `uncertainty.yaml`
```yaml
kind: committee-force-std     # sigma_F: per-atom force std across the student committee
aggregate: max                # or mean — how per-atom sigma_F rolls up to a per-frame score
```
This is already architecture-agnostic: it only needs N models of the same
student `kind` and doesn't inspect the student's internals.

### `md_backend.yaml`
```yaml
kind: lammps
binary: lmp_mpi
template_dir: templates/lammps/
```

### `reference_dft.yaml`
```yaml
kind: vasp
incar_template: templates/dft/INCAR.scan.template
# POTCAR is NOT specified here — supply your own path at runtime, per NOTICE.md
kspacing: 0.23
encut: 900
```

### `validation_profile.yaml`
```yaml
kind: silica
checks: [rdf, adf, coordination, density, msd, nve_drift, sq_fsdp]
# a different material replaces `checks` with its own deployment-relevant
# observables (e.g. elastic constants, phonon DOS, diffusion coefficients)
```

## Why this split (and what it does NOT claim)

Splitting the interface this way lets the SAME agent prompts and gate logic
target a different teacher/student pair. This is an implemented interface
property, not empirical proof of portability. The verified scope remains the
reference Allegro→SIMPLE-NN case until a second architecture is run end to end.
