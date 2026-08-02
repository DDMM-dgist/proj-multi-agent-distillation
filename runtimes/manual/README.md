# Manual or external-agent frontend

This path is for an LLM API runner, another coding agent, or a human-operated
handoff system. It does not assume a provider.

Validate and inspect the role registry:

```bash
python -m orchestration.cli validate-specs
python -m orchestration.cli list
```

Create a task packet:

```bash
python -m orchestration.cli make-task data-curator \
  "Inspect the proposed acquisition dataset and return its lineage report." \
  runs/example/exchange --run-id example \
  --input dataset=/path/to/acquisition.extxyz \
  --criterion "Every frame has parent_structure_id"
```

The runtime reads the generated JSON task, `agent_specs/data-curator.yaml`, and
the referenced canonical prompt. The agent produces a response following
`orchestration/schema/agent_result.schema.json`.
Judge tasks instead return the stricter
`orchestration/schema/judge_vote.schema.json` contract and must receive their
ordered common criteria through repeated `--criterion` options. Create one task
per run-bound lens returned by `workflow.controller gate-context`:

```bash
python -m orchestration.cli make-task judge "Review this gate" \
  runs/example/exchange --run-id example \
  --criterion "Artifact is complete" \
  --context review_lens=evidence_provenance \
  --context review_focus="Audit hashes, lineage, and evidence completeness"
```

The other two Judge tasks use the same criteria but different lens IDs and
focus text. A Judge vote must echo its assigned `review_lens` unchanged.

The Orchestrator accepts a response with `accept-result`, which binds it to its
dispatched task with an audit guarantee:

```bash
python -m orchestration.cli accept-result data-curator \
  <task.json-or-task_id> runs/example/exchange \
  --response <raw_response.json>      # or pipe the raw response on stdin
```

`accept-result` preserves the **unedited** raw response under
`runs/example/exchange/raw/<task_id>.json` **before** any parsing or validation —
so even a malformed or contract-violating response is on disk for audit — then
validates it against the task and role contract (task_id binding; the
`agent_result` or `judge_vote` schema; and, for Judge tasks, that the vote echoes
the task's run-bound `review_lens`). Only a valid response is recorded under
`results/<task_id>.json`; an invalid one raises, naming the preserved raw path,
and writes no result. A re-submitted response never overwrites a prior raw file
(`<task_id>.1.json`, `.2.json`, ...), so the full trail is retained.

`validate-result <agent> <task.json> <result.json>` remains available as a
stateless check that persists nothing.

This exchange does not mark a controller stage complete. The Orchestrator verifies
the returned artifacts and performs the normal controller/gate transition.
