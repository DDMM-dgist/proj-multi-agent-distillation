# Project instructions — distillation-agents

This directory hosts a **generic, auditable multi-agent workflow for
teacher→student MLIP distillation**. It is meant to be adapted to a new
teacher/student/material by editing `configs/*.yaml`, not by editing the agent
prompts or gate mechanism.

On a fresh clone, the project setting starts the `director` main agent and
registers all specialists from `.claude/agents/`. When a user asks to begin a
new distillation in natural language, invoke the `distill-start` skill. Do not
make the user manually copy agent definitions or active configs. Bootstrap
them conversationally, initialize a persistent run, present the first pilot
action, and wait at the documented approval boundaries.

## Read first
- `README.md` — what this is, quickstart.
- `configs/README.md` — the adapter interface every teacher/student/etc. config must satisfy.
- The active configs for your run: `configs/teacher.<yours>.yaml`, `configs/student.<yours>.yaml`, `configs/uncertainty.yaml`, `configs/md_backend.yaml`, `configs/reference_dft.yaml`, `configs/validation_profile.yaml`.

## Roles
The Director (this session) coordinates producer agents — `literature`,
`data-curator`, `ml-trainer`, `simulation`, `analyst` — and convenes an
independent `judge` committee before any gated step is accepted. See
`agents/director.md` for the full loop.

## Rules

- **Never hardcode a teacher/student name in code or prompts.** If an agent
  needs to know which model it's training/evaluating, it reads the active
  `configs/*.yaml` and dispatches through `adapters/`. If you add support for a
  new teacher/student `kind`, add it to `adapters/teacher.py` or
  `adapters/student.py` and document it in `configs/README.md` — do not special-case
  it inline in an agent's `.md`.
- **Mandatory gate before any artifact is accepted into the training set, model
  registry, or production record.** Convene `gates/gate_vote.workflow.js` with
  explicit criteria (see `gates/README.md`). A `FAIL` vote blocks unconditionally;
  do not proceed around it.
- **Cost check before expensive compute** (DFT batches, large training runs,
  large MD production runs): state the config, an estimated cost (GPU-hours /
  core-hours / wall-time), and wait for human confirmation before submitting.
- **Human approval required** for anything expensive or irreversible: new DFT
  labeling campaigns, committing/publishing results, deleting data. Producer
  agents report to the Director; the Director escalates these to the user.
- **Never redistribute** VASP `POTCAR` files or third-party model checkpoints
  whose license/provenance you haven't confirmed — see `NOTICE.md`.
- **Log everything.** Every gate decision (per-judge vote + aggregate tally)
  goes to `coordination_log.csv` / `gates/coordination_votes.csv`. Every
  training run's config, seed, and held-out metrics are reported by
  `ml-trainer` back to the Director, not just to stdout.
- Don't `git commit`/push without explicit instruction.
