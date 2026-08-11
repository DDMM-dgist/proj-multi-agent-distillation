# Generic adapter contracts (architecture-independent)

The controller (`workflow/controller.py`), the PC001–PC005 state logic, gate semantics, provenance,
and approval policy contain **no** architecture names (verified: 0 references to
Allegro/NequIP/SIMPLE-NN/MACE in the core). A new MLIP integrates by adding an adapter+config only.

## TEACHER adapter — `adapters/teacher.py` + `configs/teacher.<name>.yaml`
Required operations (all via a standard ASE calculator):
- **load** — `load_teacher(cfg)` builds the calculator from `calculator: {factory | module/class/constructor, model_arg, kwargs}` + `model`. No name dispatch.
- **label structures / predict energy / predict forces** — `adapters/acquisition.label_with_teacher` attaches `teacher_energy`/`teacher_forces` (+ optional `teacher_stress`).
- **identity / SHA / units** — the labeling manifest records `teacher_kind`, `teacher_model_sha256`, `teacher_config_sha256`, `units {energy: eV, forces: eV/Å}`.
Any ASE-calculator model qualifies (NequIP/Allegro, MACE, GAP/quippy, ACE/pyace, MACE-MP-0, MatterSim, Orb, …).

## STUDENT adapter — `adapters/student.py` + `configs/student.<name>.yaml`
Required operations:
- **prepare dataset** — reference labels attached from teacher output (`teacher_energy`/`teacher_forces`).
- **train** — dispatch order: `adapter.train` (dotted callable) → `train.command` (CLI) → built-in `kind` in {`simple-nn`, `grace-fs`, `mock`}. Seeds committee via `committee.seeds` (or `n_seeds`).
- **predict** — `predict_student` via `predict.command`/`adapter.predict`/built-in; emits `student_energy`/`student_forces`.
- **export/load deployment artifact** — `deploy` (e.g. LAMMPS `pair_style`, element order); `load_student` returns the checkpoint path.
- **identity / SHA / units** — committee manifest records per-seed path + `integrity.sha256`; units eV / eV/Å.
Per-structure `struct_weight` is supported (config `struct_weight_policy`); it must reach the backend (see the SIMPLE-NN reference: per-structure weighted str_list tags).

## SIMULATION adapter — `adapters/md_backend.py` (+ `workflow.steps run-md`)
Required operations: **load potential**, **execute requested simulation**, **emit trajectory/diagnostics**. Config-declared engine/backend; deployment artifact from the student adapter.

## VALIDATION — `validation/*` + `validation_profile.yaml`
Consumes only **model-independent scientific objects**: structures, energies, forces, trajectories, observables. No architecture coupling. Declares required observables + thresholds + reference sources; validators return a typed manifest the controller gate consumes.

## Integration rule
Adding a new architecture requires editing **only** `configs/*.yaml` (+ optionally a small adapter
module referenced by `adapter.*`/`factory`/`command`). It must NOT require changes to
`workflow/controller.py`, PC001–PC005 logic, gate semantics, provenance contracts, or approval policy.
