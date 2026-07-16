# Project instructions — distillation-agents

This directory hosts a **generic, auditable multi-agent workflow for
teacher→student MLIP distillation**. It is adapted to each run through configs
declared by that run's workflow, not by editing agent prompts or gate logic.

On a fresh clone, the project setting starts the `director` main agent and
registers all specialists from `.claude/agents/`. When a user asks to begin a
new distillation in natural language, invoke the `distill-start` skill. Do not
make the user manually copy agent definitions or active configs. Bootstrap
them conversationally, initialize a persistent run, present the first pilot
action, and wait at the documented approval boundaries.

## Read first
- `README.md` — what this is, quickstart.
- `configs/README.md` — the adapter interface every teacher/student/etc. config must satisfy.
- The active configs declared as workflow inputs, normally under
  `configs/runs/<run>/`. `configs/templates/` defines generic interfaces;
  `configs/examples/` and `examples/` contain optional built-in/case recipes.

## Roles
The Director (this session) coordinates producer agents — `literature`,
`data-curator`, `ml-trainer`, `simulation`, `analyst` — and convenes an
separate-context, mutually blind `judge` committee before any gated step is accepted. See
`agents/director.md` for the full loop.

## Rules

- **Do not hardcode a teacher/student name in core workflow or agent prompts.** If an agent
  needs to know which model it's training/evaluating, it reads the active
  `configs/*.yaml` and dispatches through `adapters/`. If you add support for a
  new teacher/student `kind`, configure its factory/callable/command contract
  and document the adapter — do not add a model-name branch to the controller
  or inline it in an agent's `.md`.
- **Mandatory gate before any artifact is accepted into the training set, model
  registry, or production record.** Convene `gates/gate_vote.workflow.js` with
  the stage's run-bound criteria from `gate-context` (see `gates/README.md`). A
  `FAIL` vote blocks unconditionally;
  do not proceed around it.
- **Cost check before expensive compute** (DFT batches, large training runs,
  large MD production runs): state the config, an estimated cost (GPU-hours /
  core-hours / wall-time), and wait for human confirmation before submitting.
- **Human approval required** for anything expensive or irreversible: new DFT
  labeling campaigns, committing/publishing results, deleting data. Producer
  agents report to the Director; the Director escalates these to the user.
- **Never redistribute** VASP `POTCAR` files or third-party model checkpoints
  whose license/provenance you haven't confirmed — see `NOTICE.md`.
- **Log everything.** The run manifest and hash-bound gate vote bundles are
  authoritative. Optional prose/CSV summaries must agree with them. Every
  training run's config, seed, and held-out metrics are reported by
  `ml-trainer` back to the Director, not just to stdout.
- **Bind code and evidence.** Run initialization records the Git commit and any
  dirty-tree content hash. Validation stages should emit the common
  `validation.report` envelope (or a documented observable-specific contract)
  and include integrity records for the files used as evidence.
- **Preflight the whole run.** Validate teacher, student, acquisition,
  uncertainty, MD, DFT, and validation configs before the first pilot. A
  schema-only check can run anywhere; executable/file checks belong in the
  corresponding server environment.
- Don't `git commit`/push without explicit instruction.
