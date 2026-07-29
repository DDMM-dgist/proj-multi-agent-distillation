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
the referenced canonical prompt. It writes the agent response to
`runs/example/exchange/results/<task_id>.json` using
`orchestration/schema/agent_result.schema.json`.
Judge tasks instead return the stricter
`orchestration/schema/judge_vote.schema.json` contract and must receive their
ordered criteria through repeated `--criterion` options.

The Director validates a result with:

```bash
python -m orchestration.cli validate-result data-curator \
  <task.json> <result.json>
```

This exchange does not mark a controller stage complete. The Director verifies
the returned artifacts and performs the normal controller/gate transition.
