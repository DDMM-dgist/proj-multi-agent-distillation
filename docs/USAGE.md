# Using the framework

This is the detailed operator workflow. For the high-level picture, start with the
[README](../README.md); to integrate a new Teacher/Student architecture, read
[`handoff/HANDOFF_README.md`](../handoff/HANDOFF_README.md) and
[`handoff/ADAPTER_INTERFACE.md`](../handoff/ADAPTER_INTERFACE.md).

All commands below are shown as `python -m workflow.controller ...`. After `pip install -e .` the
console-script alias `distill-run ...` is equivalent. Always confirm exact flags with
`python -m workflow.controller <subcommand> --help`.

---

## 1. Prepare a research project

Gather the project inputs before initializing anything:

- material / system and composition
- Teacher MLIP (kind) and Teacher model path
- Student MLIP architecture and training config
- available structures (seed / source)
- Teacher training/reference provenance, if available
- an independent DFT holdout, if available (validation only — never training)
- the target application / deployment domain
- validation requirements (observables + thresholds + reference sources)
- the compute backend (LAMMPS, DFT code, etc.)
- the PydanticAI provider, if you use the agent runtime (optional)

## 2. Add / configure a Teacher

Write `configs/teacher.<name>.yaml` with a `calculator:` block — either
`factory: module.callable`, **or** `module` / `class` / `constructor` + `model_arg` + `kwargs` — and
a `model:` path. `adapters/teacher.load_teacher` instantiates it generically; the controller is never
edited. Any ASE-calculator model qualifies (NequIP/Allegro, MACE, GAP/quippy, ACE, MACE-MP-0,
MatterSim, Orb, …). Template: [`handoff/templates/teacher.adapter.template.yaml`](../handoff/templates/teacher.adapter.template.yaml).

## 3. Add / configure a Student

Write `configs/student.<name>.yaml`. Use a built-in `kind` (e.g. `simple-nn`, `grace-fs`, `mock`) if
one fits, otherwise declare `adapter.train` (a dotted callable) or `train.command` (a CLI), plus
`predict` and `deploy`. `adapters/student.train_student` dispatches generically. Per-structure
`struct_weight` is supported via `struct_weight_policy`. Template:
[`handoff/templates/student.adapter.template.yaml`](../handoff/templates/student.adapter.template.yaml).

## 4. Structure / data organization

Organize structures by purpose so provenance stays clean:

| Category | Use |
|---|---|
| `TEACHER_TRAINING_REFERENCE` | coverage baseline (what the Teacher already saw) |
| `DISTILLATION_CANDIDATE` | labeled by the Teacher, used to train the Student |
| `PRODUCTION_MD` | deployment-domain structures / trajectories |
| `INDEPENDENT_DFT_HOLDOUT` | reserved for validation only |

Acquisition (optional) can generate structures two ways, selected in config:

1. `augment-atoms` — distorted structures around existing seeds
2. `teacher-md` — ASE-MD snapshots from a foundation Teacher

Both preserve `parent_structure_id` lineage. Teacher labeling is a separate step from acquisition,
and every structure records the Teacher model/head, label units, source, and structure ID before it
passes the dataset gate. After labeling, train/validation/held-out splits are made on
`parent_structure_id` so augmented children of one seed never straddle splits; missing lineage keys
halt the split by default.

## 5. Validation profile

Write `validation_profile.yaml`: the observables, their thresholds, and their reference sources.
Validation consumes only **model-independent objects** (structures, energies, forces, trajectories,
observables). Observables that must numerically pass are declared as `required_pass_observables`;
sub-threshold results are still preserved as audit artifacts, but the Judge gate cannot record a PASS
until such a result actually passes. Template:
[`handoff/templates/validation_profile.template.yaml`](../handoff/templates/validation_profile.template.yaml).

## 6. Git-clean campaign initialization

The controller pins the project git revision at `init` and hash-binds declared inputs. For a clean,
reproducible pin:

- commit (or stash) your working tree **before** `init`,
- do not dirty tracked code while a campaign is in flight,
- during a run, write only to instance-specific paths (`runs/`, `configs/runs/`).

```bash
python -m workflow.controller init path/to/workflow.yaml runs/<CAMPAIGN_ID>
```

`init` copies + hashes the declared config/templates/seed structures into the run directory and
records the git commit and a dirty-diff hash. Large model checkpoints are not copied — they are
pinned in place by file/tree hash.

## 7. Run a campaign

Execute stages one at a time; each must record a PASS before the next can run.

```bash
python -m workflow.controller run-stage runs/<CAMPAIGN_ID> <STAGE_NAME>
```

A concrete `workflow.yaml` typically defines these stages (mapping to PC001–PC005):

| Stage name | PC | Purpose |
|---|---|---|
| `teacher_baseline` | PC001 | Teacher applicability + reference purpose |
| `acquisition` *(optional)* | PC002 | generate structures (augment-atoms / teacher-md) |
| `teacher_labeling` | PC002 | Teacher labels + provenance manifest |
| `dataset_split` | PC002 | lineage-disjoint train/valid/test split |
| `training` | PC003 | Student committee checkpoints + manifest |
| `evaluation` | PC004 | Student-vs-Teacher (+ DFT channels) — kept distinct |
| `physical_validation` | PC005 | declared observables per the profile |

See [`handoff/templates/workflow.example.yaml`](../handoff/templates/workflow.example.yaml) for the
canonical shape, and [`examples/mock/`](../examples/mock/) for a runnable network-free example.

## 8. Check status

```bash
python -m workflow.controller status runs/<CAMPAIGN_ID>
```

Shows each stage's status, its gate result, `pending_recovery`, and the campaign status.

### 8a. External-executor lifecycle (dispatched/long-running stages)

A stage whose `pydantic_ai` action is genuinely invoked (e.g. the `acquisition` stage's
`augment-atoms` subprocess, or any other dispatched executor that can run for minutes to hours)
moves through a bounded, fully representable lifecycle instead of only ever showing
`pending`/`completed`:

```
pending --(dispatch: trusted executor about to run)--> running
running --(executor returns + declared outputs present)--> completed
running --(executor raises an ordinary, fast, synchronous error)--> pending   (deferred, retryable)
running --(ExternalActionPending: legitimately still queued elsewhere)--> pending (deferred, resumable)
running --(Controller-owned wall-time budget elapses)--> timed_out            (terminal)
running --(executor operationally cancelled)--> cancelled                    (terminal)
```

`attempts` is incremented and a `stage_execution_started` event is recorded the moment the stage
enters `running` — **at dispatch time**, before the executor can possibly hang — so a killed or
still-hanging process is never lost as an unaccounted attempt. While `running`, a heartbeat
(`executor_heartbeat` events, carrying `pid`/`elapsed_s`/an executor-supplied `progress` payload)
is recorded periodically if the executor reports progress.

Only `completed` is eligible for a **PASS** gate. `timed_out`, `failed`, and `cancelled` are all
eligible for a **REVISE**/**FAIL** gate (`record_gate` accepts any of them) — an execution that
never completed is not stuck unrepresentable; it can still route straight into the same
propose-recovery / approve-recovery / start-iteration path described in §10.

A dispatched external executor with a configured `timeout_s` runs under
`workflow.subprocess_runner.run_bounded`: the subprocess is launched in its own process group, and
if the wall-time budget elapses, ONLY that process group is terminated (SIGTERM, then SIGKILL
after a grace period) — never any other process on the host. The acquisition adapter
(`adapters.acquisition.run_augment_atoms`) additionally runs a static feasibility pre-check
(`check_acquisition_feasibility`) before dispatching a subprocess at all: if the configured
`sigma_range`/`max_relax_steps`/`similarity_threshold` combination cannot plausibly ever satisfy
augment-atoms's own rejection-sampling acceptance criterion, the stage is rejected immediately
(fast, deterministic, pre-dispatch) instead of hanging.

## 9. PASS / REVISE / FAIL

After a stage is executed and its declared outputs validate, it is gated:

```bash
python -m workflow.controller gate runs/<CAMPAIGN_ID> <STAGE_NAME> --votes votes.json
```

- **PASS** requires a three-judge vote bundle bound to the stage's `gate.criteria` and the current
  artifact SHA-256. Three independent judges each apply a complementary review lens
  (evidence/provenance, scientific validity, reproducibility/deployment).
- Any single **FAIL** → the stage FAILs.
- A missing or malformed vote → **REVISE**.
- Deterministic numeric checks are authoritative; the Judge reviews evidence and interpretation and
  never overrides a deterministic fact.

A REVISE/FAIL sets `pending_recovery` and blocks advancement.

## 10. Recovery / human approval

Do not blindly re-run compute after a REVISE/FAIL. Instead:

```bash
python -m workflow.controller propose-recovery  runs/<CAMPAIGN_ID> plan.json
python -m workflow.controller approve-recovery  runs/<CAMPAIGN_ID> --approved-by <name>
python -m workflow.controller start-iteration   runs/<CAMPAIGN_ID>
python -m workflow.controller verify-recovery   runs/<CAMPAIGN_ID> report.json
```

The `RecoveryPlan` names the failure's root cause, responsible role, the stage to return to, the
data/config changes, any Teacher/DFT labeling or Student retraining, the revalidation items, and the
expected cost. Costly or irreversible actions (new DFT, Teacher retraining, large MD, expensive
HPC/data generation) require human approval; small reversible operations inside an approved campaign
may proceed automatically.

## 11. Starting a new iteration

`start-iteration` binds the approved plan to the failed-attempt artifacts and Judge evidence, then
invalidates from the return stage and forces re-execution. A `RecoveryExecutionReport` must name the
approved changes and the stages used for relabeling/retraining/revalidation before the previously
failing gate can PASS again. The controller compares the changed stage's registered artifacts against
the prior iteration and **blocks a PASS if the artifact you promised to change is byte-identical**.

> **Resume vs. re-derive.** In-place resume is only valid when code *and* bound inputs are unchanged.
> If tracked code or a hash-bound input changed, that is a new attempt/iteration (a re-derivation) —
> not a resume. Simple command/scheduler retries that change no configuration are execution retries,
> not scientific iterations.
