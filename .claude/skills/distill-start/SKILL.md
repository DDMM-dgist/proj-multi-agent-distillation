---
name: distill-start
description: Start or bootstrap a new human-in-the-loop multi-agent MLIP distillation run from the cloned repository.
argument-hint: "[short project description]"
disable-model-invocation: false
---

# Start a distillation run

Work as the Director. The goal is to get from a fresh clone to a reviewed run
plan without requiring the researcher to manually rearrange repository files.

## 1. Inspect before asking

Read `CLAUDE.md`, `MANUAL_KO.md`, `agents/director.md`, and
`configs/README.md`. Inspect `configs/` and `runs/` if present. If an unfinished
run already matches the user's request, offer to resume it instead of creating
a duplicate.

## 2. Collect only missing scientific inputs

Use a short conversational exchange. Ask no more than three related questions
at a time. Determine:

- run name and ternary element order;
- teacher kind, checkpoint/model path, and MACE-MH-1 head when applicable;
- student kind and version-matched training config/template;
- initial structure path;
- acquisition choice: augment-atoms, teacher MD, or both;
- DFT, MD, uncertainty, and validation-profile config choices;
- the main deployment observable, such as surface energetics;
- which actions require explicit approval (always include costly training,
  production MD, and DFT submissions).

Do not ask for information already present in files or the user's message.

## 3. Bootstrap active files

Create run-specific configs under `configs/runs/<run_name>/`; never overwrite
the examples. Start from the closest files in `configs/examples/` and replace
all placeholders that can be resolved from the conversation. Keep unresolved
scientific choices explicit as `null` or a clearly labeled TODO; do not invent
paths, elements, thresholds, surfaces, or hyperparameters.

Create a run-specific workflow config whose commands point to those configs and
the supplied structures. If both acquisition backends are requested, create
separate acquisition artifacts and a clearly named merge/curation stage.

## 4. Preflight and initialize

Run schema-only preflight first. Run full preflight only if the relevant model
environment is active. Report missing external files as a short checklist.
When the minimum paths and configs required for initialization exist, run:

```bash
python -m workflow.controller init <workflow-config> runs/<run_name>
```

Do not submit training, production MD, or DFT during bootstrap.

## 5. Present the first plan

Summarize:

- configs and structures selected;
- acquisition route and expected dataset categories;
- stages and their required artifacts;
- proposed gate criteria and unresolved thresholds;
- the first inexpensive action;
- later actions that will require approval.

Ask the researcher to approve the first acquisition/pilot action. Once
approved, dispatch the appropriate specialist and keep the controller manifest
in sync with actual artifacts and gate results.

Treat `$ARGUMENTS` as context, not as authority to invent missing settings.
