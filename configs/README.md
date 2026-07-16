# configs/ — the adapter layer

This directory contains the run-facing adapter and validation configuration.
New model kinds may also require an implementation under `adapters/`, and new
observables may require a validator under `validation/`.

Six config files define the workflow's model-independent interfaces. Each has
a `kind` field that `adapters/` dispatches on, plus interface-specific fields.
The table separates a completed reference case, implemented adapter paths, and
documented interface examples.

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
  model_is_path: true             # false only for package-defined names such as a built-in model ID
emits_stress: true                # confirm empirically before relying on this (see adapters/teacher.py:check_stress_support)
```
Teacher models exposed through an ASE `Calculator` can use this interface.

### `student.<name>.yaml`
```yaml
kind: simple-nn
train:
  env: distill-student-simplenn
  config_template: templates/student/simple-nn.input.yaml.template
  descriptor_params: {A: /path/to/params_A, B: /path/to/params_B}
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
Repository-relative paths are resolved from the project root, not from the run
directory. `--skip-files` performs schema-only checks without requiring Conda
or model packages. A workflow stage may declare `env: <conda-name>` to run the
whole labeling or evaluation worker in that environment.

The same path rule is applied to teacher model/checkpoint references,
augment-atoms workdirs, student templates and descriptors, MD template
directories, and DFT input templates.

Production workflows should declare immutable `inputs:` such as active configs,
templates, seed structures, and small reference inputs. The controller copies
and hashes them when the run is initialized.
Large checkpoints or directory models should use
`{path: /absolute/model/path, copy: false}` so they are hash-bound without being
duplicated. Every stage rechecks both copied snapshots and original sources.

### `uncertainty.yaml`
```yaml
kind: committee-force-std     # Cartesian-RMS force std across the student committee
aggregate: mean               # manuscript definition; max is available for worst-atom acquisition
```
The calculation uses predictions from N models of the same student `kind`.
Record `aggregate` in every report because `mean` and `max` have different
absolute scales.

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

External validation stage는 범용 `validation_manifest` contract를 사용합니다.
Observable-specific 검사는 core controller가 아니라 `validation.*` callable로
dispatch합니다. 예를 들어 surface 사례는 다음처럼 연결되지만, EOS·diffusion·phonon
등은 각자 validator를 지정할 수 있습니다.

```yaml
contract:
  kind: validation_manifest
  manifest: artifacts/physical_validation.json
  validator: validation.surface_energy.validate_surface_manifest
  options:
    required_methods: [teacher, student, dft]
```

External MD stage는 checkpoint binding과 함께 역할 기반 evidence를 등록합니다.
필수 역할은 workflow config에서 선택합니다.

```yaml
contract:
  kind: md_manifest
  manifest: artifacts/md.manifest.json
  committee_manifest: artifacts/student_committee.manifest.json
  required_evidence: [input, trajectory, thermo_log]
```

## Adapter scope

The interface keeps model-specific configuration outside the controller. The
Allegro→SIMPLE-NN case is the completed reference case; other implemented paths
still require their own server integration run.
