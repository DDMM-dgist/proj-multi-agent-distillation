# configs/ — the adapter layer

This directory contains the run-facing adapter and validation configuration.
New model kinds provide configured callables or commands, and new observables
provide validators; neither requires a new branch in the controller.

Start new runs from `configs/templates/`. Files under `configs/examples/` are
built-in adapter recipes, while complete material/model combinations belong
under `examples/`; neither location supplies automatic defaults.

Seven config families define the workflow's model-independent interfaces. Each
has a `kind` identifier plus interface-specific fields. Generic callables and
commands are selected from config; only the packaged convenience recipes use
built-in dispatch.

| File | Required interface | Implemented `kind`s | Documented-only examples |
|---|---|---|---|
| `teacher.<name>.yaml` | given a structure → energy, forces, [stress] | generic ASE factory/class constructor; built-in example recipes for Allegro and MACE | any ASE calculator |
| `student.<name>.yaml` | trainable; checkpoint; prediction; optional deployment | callable or command adapter; built-in SIMPLE-NN and GRACE/FS recipes | any external trainer with the contract below |
| `acquisition.yaml` | structures with explicit parent lineage | callable adapter; built-in augment-atoms command and teacher-MD recipes | any generator satisfying the contract |
| `uncertainty.yaml` | per-configuration ranking score from a committee | `committee-force-std` (σ_F, architecture-agnostic — works for any committee of N same-`kind` students) | ensemble variance, dropout, extrapolation-grade scores |
| `md_backend.yaml` | run a trajectory from a deployed model | `lammps` | `ase-md`, model-native MD |
| `reference.yaml` | reference energy/force/stress calculation | callable renderer; built-in `vasp` input renderer | other electronic-structure backends |
| `validation_profile.yaml` | deployment-relevant physical observables | generic RDF/coordination/density/MSD/NVE report | any observable with a ValidationReport-compatible validator |

## Interface contracts

### `teacher.<name>.yaml`
```yaml
kind: my-teacher
model: /path/to/model             # omit when model_arg is null
calculator:
  factory: my_package.make_calculator
  model_arg: model
  model_is_path: true
  kwargs: {}
emits_stress: null                # confirm before relying on stress labels
```
Teacher models exposed through an ASE `Calculator` can use this interface.
Select the teacher environment on the workflow stage's `env` field; the
calculator adapter does not switch interpreters itself.
`calculator.factory` may name any callable. Alternatively, `module`/`class`,
optional `constructor`, `model_arg`, and `kwargs` describe construction without
adding a teacher-name branch to core code.

### `student.<name>.yaml`
```yaml
kind: my-student
adapter:
  train: my_package.distill.train       # (cfg, dataset, out_dir, seed) -> path/ModelArtifact
  load: my_package.distill.load         # optional: (cfg, checkpoint) -> path/ModelArtifact
  deploy: my_package.distill.deploy     # optional deployment block renderer
  preflight: my_package.distill.validate_config
committee: {n_seeds: 4}
predict:
  factory: my_package.ase.make_calculator
  checkpoint_arg: checkpoint
deploy:
  elements: [A, B]
```

An external CLI can instead use `train.command` with placeholders
`{dataset_path}`, `{out_dir}`, `{seed}`, and a required `train.artifact` path.

The workflow exchanges architecture-neutral `ModelArtifact` and
`PredictionBatch` values from `adapters/contracts.py`. A prediction factory is
configured without modifying agent prompts:

```yaml
predict:
  factory: my_student_package.ase.make_calculator
  checkpoint_arg: checkpoint
  kwargs: {device: cpu}
```

Run `python -m adapters.preflight` across the active teacher, student,
acquisition, uncertainty, MD, DFT, and validation configs before expensive work.
Repository-relative paths are resolved from the project root, not from the run
directory. `--skip-files` performs schema-only checks without requiring Conda
or model packages. A full check also verifies configured callables,
executables, templates, and model paths from the environment where it runs.
A workflow stage may declare `env: <conda-name>` to run the whole labeling or
evaluation worker in that environment.

The same path rule is applied to teacher model/checkpoint references,
augment-atoms workdirs, student templates and descriptors, MD template
directories, and DFT input templates.

Production workflows should declare immutable `inputs:` such as active configs,
templates, seed structures, and small reference inputs. The controller copies
and hashes them when the run is initialized.
Large checkpoints or directory models should use
`{path: /absolute/model/path, copy: false}` so they are hash-bound without being
duplicated. Every stage rechecks both copied snapshots and original sources.

Every stage that can receive a Judge PASS must also declare its ordered
criteria before run initialization:

```yaml
- name: evaluation
  command: null
  outputs: [artifacts/accuracy_report.json]
  gate:
    criteria:
      - held-out parent groups do not overlap with training
      - required error channels and coverage are complete
```

The controller snapshots this list in the run manifest. `gate-context` returns
the same criteria with the current artifact hashes, and a vote bundle using a
different list is rejected. Change scientific thresholds through a reviewed
config revision and input rebind; changing the criteria themselves requires a
new run rather than an edit made after seeing an artifact.

### `acquisition.yaml`

An acquisition adapter returns an ASE-readable structure file whose every
frame contains `parent_structure_id`. A generic backend supplies
`adapter.acquire`; the packaged `augment-atoms` command and teacher-MD paths are
optional recipes. Teacher-MD requires explicit timestep, friction units,
sampling interval, random seed, and center-of-mass policy. Acquisition generates
structures; teacher labeling remains a separate stage.

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

LAMMPS is a built-in backend. Another engine supplies `adapter.renderer` and
`adapter.runner` dotted callables in its config.

### `reference.yaml`
```yaml
kind: vasp
incar_template: templates/dft/INCAR.scan.template
reference_theory: SCAN
template_variables:
  XC_BLOCK: |
    METAGGA = SCAN
    LASPH = .TRUE.
# POTCAR is NOT specified here — supply your own path at runtime, per NOTICE.md
kspacing_inv_angstrom: 0.23
encut_ev: 900
```

VASP is a built-in reference-input renderer. Another electronic-structure code
supplies `adapter.renderer`; the core workflow does not require VASP.

### `validation_profile.yaml`
```yaml
kind: project-validation
checks: [rdf, adf, coordination, density, msd, nve_drift, sq_fsdp]
# choose implemented observables or attach validators for new ones
```

Implemented observables can be selected in config. A new observable requires a
validator that emits the common ValidationReport or another documented
`validation_manifest` contract; changing the name in YAML alone does not create
the scientific calculation.

External validation stage는 범용 `validation_manifest` contract를 사용합니다.
Observable-specific 검사는 core controller가 아니라 dotted validator callable로
dispatch합니다. 예를 들어 built-in surface 사례는 다음처럼 연결되지만, EOS·diffusion·phonon
등은 각자 validator를 지정할 수 있습니다.

```yaml
contract:
  kind: validation_manifest
  manifest: artifacts/physical_validation.json
  validator: validation.surface_energy.validate_surface_report
  options:
    profile_path: "{project_dir}/configs/validation_profile.yaml"
    required_methods: [teacher, student, dft]
```

The surface case report uses the common `ValidationReport` envelope. Each
teacher/student/reference slab and bulk energy names a raw-evidence role, and
the report's delta criteria are checked against the hash-bound active profile.

구조·동역학 결과는 공통 `ValidationReport` envelope로 기록할 수 있습니다.

```yaml
contract:
  kind: validation_manifest
  manifest: artifacts/validation_report.json
  validator: validation.report.validate_validation_report
  options:
    required_observables: [density, nve_drift]
    # These checks may be recorded as FAIL, but the controller will reject a
    # Judge PASS until each one is PASS.
    required_pass_observables: [density, nve_drift]
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

The controller and agent roles remain model-independent. Built-in recipes and
case studies demonstrate the contract; each new model/backend still requires
its own adapter and server integration test, not a modification of the core
controller or gate logic.
