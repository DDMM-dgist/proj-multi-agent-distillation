# Validation gates — separate-context, mutually blind judge committee

The operational, re-runnable gate for the distillation workflow. Each gate is
decided by three `judge` instances that run in **separate contexts**, read the
**same** artifact, and vote **blind to each other**. The
Director convenes them, tallies, and records **every individual vote** — so
split votes show up in the audit trail.

Gate criteria are supplied from each stage's `gate.criteria` in the active run
configuration and are snapshot-bound when the run is initialized.

## Pieces

- `../agents/judge.md` — the judge agent (Read/Grep/Glob/Bash, model=sonnet;
  evaluator only, never the producer, conservative default).
- `gate_vote.workflow.js` — the orchestration: spawns three judges in parallel with a
  forced structured verdict, applies the decision rule, returns the tally + votes.
- `schema/coordination_votes.example.csv` — one row **per judge per gate** (the real votes).
- `schema/coordination_log.example.csv` — one row per gate (aggregate tally + decision).

## How the Director convenes a gate

### Standard Claude Code

The main Director invokes three registered `judge` agents with identical
artifact paths and criteria. Each returns one JSON object without seeing the
other votes. Apply the same tally below: any FAIL blocks; otherwise unanimous
PASS is required; a missing/malformed vote is REVISE. Save the three JSON votes
and aggregate JSON under `runs/<run>/gates/`, then record the aggregate through
`python -m workflow.controller gate ... --votes <bundle.json>`.

The saved bundle must contain the stage name, non-empty criteria, exactly three
structured votes, the recomputed aggregate decision, and the SHA-256 mapping of
the registered stage artifacts. The controller recomputes the decision and
compares the hashes; a bare manual `PASS` is rejected. A researcher or Director
may still record `REVISE` or `FAIL` directly to stop work conservatively.

Before convening the judges, obtain the verified hash map and ordered criteria with
`python -m workflow.controller gate-context <run_dir> <stage>`. Pass its
`artifact_sha256` and `criteria` fields to the optional workflow. The controller
rejects a vote bundle whose criteria differ from this run-bound list. If a judge invocation fails,
the workflow emits a synthetic REVISE vote containing every criterion with
`ok: false`; it never fabricates a PASS or silently drops the failed slot.

```json
{
  "stage": "dataset_split",
  "criteria": ["train/test parent groups do not overlap"],
  "artifact_sha256": {"/absolute/path/train.extxyz": "..."},
  "decision": "PASS",
  "votes": [
    {"judge_id": "judge-1", "verdict": "PASS", "criteria_checked": [{"criterion": "...", "ok": true}],
     "rationale": "...", "required_fix": ""},
    {"judge_id": "judge-2", "verdict": "PASS", "criteria_checked": [{"criterion": "...", "ok": true}],
     "rationale": "...", "required_fix": ""},
    {"judge_id": "judge-3", "verdict": "PASS", "criteria_checked": [{"criterion": "...", "ok": true}],
     "rationale": "...", "required_fix": ""}
  ]
}
```

### Optional Workflow runtime

```
Workflow({ name: 'gate-vote', args: {
  gate:     'student-accuracy-gate',
  target:   'student_committee_v1',
  artifact: 'held-out E/F errors at <path>; teacher error at <path>',
  artifact_sha256: {'/absolute/path/to/report.json': '<hash from gate-context>'},
  criteria: [
    'student-vs-teacher force MAE <= <threshold from configs/student.yaml or validation_profile.yaml>',
    'teacher-vs-DFT force MAE reported alongside as a reference baseline',
    'committee force-std (sigma_F) reported, not just a point metric',
  ],
  n: 3,
  rule: 'unanimous',
}})
```

Store the returned vote bundle under the run's `gates/` directory and pass it
to `workflow.controller gate --votes`. The manifest and bound bundle are the
authoritative record; a CSV summary is optional.

## Adding a new gate type

You don't need new code — `gate_vote.workflow.js` takes arbitrary criteria.
Just decide what to check (usually sourced from `configs/validation_profile.yaml`
for physical checks, or `configs/student.yaml`/`configs/teacher.yaml` for
data/training checks) and call the workflow with those criteria.
