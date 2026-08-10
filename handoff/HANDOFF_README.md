# MLIP-distillation workflow — external handoff

This is a **runtime-neutral, architecture-independent** multi-agent MLIP-distillation workflow.
The controller and the PC001–PC005 scientific state logic contain **no** teacher/student
architecture assumptions — all architecture-specific behavior lives in **adapters + config**. The
SiO2 Allegro→SIMPLE-NN case is the **reference adapter example**, not a requirement.

## What you must provide (no framework internals needed)
1. **Teacher** — any model exposing an ASE calculator (NequIP/Allegro, MACE, GAP, ACE, MACE-MP-0,
   MatterSim, Orb, …). Point a `teacher.<name>.yaml` at it (see `ADAPTER_INTERFACE.md` + template).
2. **Student** — any trainable MLIP with a train + predict + deploy path. Provide a
   `student.<name>.yaml` (built-in `kind` OR `adapter.train`/`train.command`).
3. **Deployment domain + validation profile** — `distillation_scope.yaml` (system, composition,
   structure classes, T/P ranges, teacher-applicability evidence) + `validation_profile.yaml`
   (observables + thresholds + reference sources).
4. **Seed structures** and an acquisition choice (augment-atoms / teacher-MD / both), if generating.

## Connect a new Teacher architecture
Write `configs/teacher.<name>.yaml` with a `calculator:` block — either
`factory: module.callable` OR `module/class/constructor` + `model_arg` + `kwargs` — and `model:`.
The core `adapters/teacher.load_teacher` instantiates it generically; **no controller change**.

## Connect a new Student architecture
Write `configs/student.<name>.yaml`. Use a built-in `kind` if one fits, else declare
`adapter.train` (a dotted callable) or `train.command` (a CLI) + `predict` + `deploy`. Add per-run
`descriptor_params`/templates as needed. `adapters/student.train_student` dispatches generically.

## Specify material/application domain & validation
`distillation_scope.yaml` + `validation_profile.yaml` (templates provided). Validation consumes only
**model-independent objects**: structures, energies, forces, trajectories, observables.

## Human approval
Costly/irreversible actions (student committee training, production MD, DFT submission) are
approval-gated. Approval is a typed record; ordinary reversible stages don't re-prompt.

## PASS / REVISE / FAIL
Each stage: `run-stage` (execute + validate declared outputs/contract) → `gate` (PASS needs the
3-judge vote bundle; REVISE/FAIL sets `pending_recovery`, routed via `propose-recovery` →
`approve-recovery` → `start-iteration` to the responsible role). Deterministic numeric checks are
authoritative; the Judge reviews evidence/interpretation, never overrides deterministic facts.

## Launch & inspect a campaign
```bash
python -m workflow.controller init <workflow.yaml> runs/<run_name>
python -m workflow.controller run-stage runs/<run_name> <stage>
python -m workflow.controller gate runs/<run_name> <stage> --votes <bundle.json>   # or REVISE/FAIL
python -m workflow.controller status runs/<run_name>
```
Reproducibility guards: inputs are hash-bound at `init`; the project git revision is pinned. A
**code or bound-input change requires a NEW run attempt** (re-derivation) — in-place resume is only
for unchanged code+inputs. Run dirs + `configs/runs/` are instance-specific.

## Files in this kit
- `ADAPTER_INTERFACE.md` — the generic teacher/student/simulation/validation contracts.
- `templates/teacher.adapter.template.yaml`, `templates/student.adapter.template.yaml`
- `templates/workflow.example.yaml`, `templates/validation_profile.template.yaml`
- Minimal runnable example: `examples/mock/` (network-free, trains a mock student end-to-end).
- Reference real adapter example: the SiO2 Allegro→SIMPLE-NN configs under
  `configs/examples/` + `configs/runs/SIO2_DISTILLATION_DEV_V6_SMALL_R1/`.
