---
name: distill-start
description: Start or bootstrap a new human-in-the-loop multi-agent MLIP distillation run from the cloned repository.
argument-hint: "[short project description]"
disable-model-invocation: false
---

# Start a distillation run

Work as the Orchestrator. The goal is to get from a fresh clone to a reviewed run
plan without requiring the researcher to manually rearrange repository files.

## 1. Inspect before asking

Read `CLAUDE.md`, `README.md`, `agents/orchestrator.md`, and
`configs/README.md`. Inspect `configs/` and `runs/` if present. If an unfinished
run already matches the user's request, offer to resume it instead of creating
a duplicate.

## 2. Collect only missing scientific inputs

Use a short conversational exchange. Ask no more than three related questions
at a time. Determine:

- run name, chemical system, and exact element/type order;
- teacher kind, checkpoint/model path, and model variant/head when applicable;
- student kind and version-matched training config/template;
- initial structure path;
- acquisition choice: augment-atoms, teacher MD, or both;
- DFT, MD, uncertainty, and validation-profile config choices;
- the deployment domain and required accuracy, energetics, structure, dynamics,
  stability, and performance observables;
- whether Teacher training data access is full, representative, or unavailable,
  and whether replay of Teacher-training structures is permitted;
- whether the Teacher has already been validated in the deployment domain and
  which targets use Teacher, DFT, experiment, or another reference;
- observable-specific protocols and acceptance thresholds;
- which actions require explicit approval (always include costly training,
  production MD, and DFT submissions).

Do not ask for information already present in files or the user's message.

## 3. Bootstrap active files

Create run-specific configs under `configs/runs/<run_name>/`; never overwrite
the examples. Start from `configs/templates/` and consult a built-in/case
example only for an adapter actually selected by the run. Replace all
placeholders that can be resolved from the conversation. Keep unresolved
scientific choices explicit as `null` or a clearly labeled TODO; do not invent
paths, elements, thresholds, observable protocols, or hyperparameters.

Create `distillation_scope.yaml`, `dataset_policy.yaml`, and
`validation_profile.yaml`. Before Student results exist, define the deployment
domain, Teacher applicability evidence, validation purposes/reference sources,
protocols, and proposed thresholds. Present these choices for researcher
approval; do not silently treat a Teacher prediction as physical ground truth.

Create a run-specific workflow config whose commands point to those configs and
the supplied structures. If multiple acquisition backends are requested,
create separate artifacts and use `workflow.steps merge-datasets` for the
lineage-preserving merge. Choose and record the exact-duplicate policy; do not
apply label-energy alignment unless the run declares a reviewed method.
Declare every active config, template, and seed structure under workflow
`inputs:` so initialization snapshots and hashes them. Put teacher labeling and
teacher MD stages in the teacher Conda `env`, and student prediction stages in
the student Conda `env`, when those environments differ.
Place `teacher_baseline` before acquisition and `data_coverage` after acquisition
but before labeling/training. Their
reports use `validation.teacher_baseline.validate_teacher_baseline_report` and
`validation.data_coverage.validate_data_coverage_report`. When Teacher training
data are unavailable, require an explicit `NOT_ASSESSABLE` coverage status and
limitations instead of inventing a quantitative score.
Declare adapter source files or an environment/package lock manifest as inputs
when the adapter is outside this repository, so its implementation is auditable.
Declare large model checkpoints or directories as `{path: <path>, copy: false}`;
the controller hash-binds them in place without copying them into the run.
Declare a non-empty ordered `gate.criteria` list on every stage that may PASS.
Source scientific thresholds from the active configs and bind the criteria
before initialization; never choose or soften them after seeing stage results.

Always add a dataset split stage after labeling. Split by `parent_structure_id`
(or an equally explicit lineage key), train only on `train.extxyz`, and evaluate
only on held-out `test.extxyz`. If a requested gate depends on DFT channels,
mark those channels required so missing DFT labels fail closed rather than skip.
Do not allow silent lineage fallback for augmented or MD-generated structures.
Verify acquisition output has a parent ID before teacher labeling or splitting.

## 4. Preflight and initialize

Run schema-only preflight across the teacher, student, acquisition, uncertainty,
MD, DFT, validation, scope, and dataset-policy configs first. Before the pilot,
run it with `--require-ready` and
run full file/import checks only if the relevant model environment is active.
Report missing external files and unresolved thresholds as a short checklist.
When the minimum paths and configs required for initialization exist, run:

```bash
python -m workflow.controller init <workflow-config> runs/<run_name>
```

Do not submit training, production MD, or DFT during bootstrap.

## 5. Present the first plan

Summarize:

- configs and structures selected;
- Teacher baseline/applicability work and the frozen validation profile;
- Teacher-data access level, coverage dimensions, and replay/source-mixture policy;
- acquisition route and expected dataset categories;
- stages and their required artifacts;
- proposed gate criteria and unresolved thresholds;
- the first inexpensive action;
- later actions that will require approval.

Ask the researcher to approve the first acquisition/pilot action. Once
approved, dispatch the appropriate specialist and keep the controller manifest
in sync with actual artifacts and gate results.

For machine-evaluated thresholds that must pass before acceptance, list the
observable under the validation contract's `required_pass_observables`. A
threshold failure remains a completed, auditable result, but the controller
must reject a Judge PASS until the required observable passes.

A PASS must be recorded with a three-Judge vote bundle containing the exact
run-bound criteria and artifact hashes returned by `workflow.controller gate-context`.
Never issue a bare PASS. A failed Judge invocation occupies its slot as a
synthetic REVISE vote with every criterion marked false.
Register the whole committee directory as a training-stage artifact in addition
to its manifest, so every checkpoint is bound to the training gate.

After a Judge REVISE/FAIL, do not rerun scientific work directly. Ask the
analyst for a root-cause classification, prepare a RecoveryPlan using
`gates/schema/recovery_plan.example.json`, show the proposed data/config change
and cost to the researcher, then use the controller's recovery commands only
after explicit approval. A command/scheduler failure without a scientific
change is an execution retry and does not count as a closed-loop iteration.
After the approved stages run, create the recovery execution report from
`gates/schema/recovery_execution.example.json` and call `verify-recovery`
before asking the Judges to PASS the formerly failed gate.

Treat `$ARGUMENTS` as context, not as authority to invent missing settings.
