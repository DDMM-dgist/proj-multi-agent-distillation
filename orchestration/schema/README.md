# Agent exchange schemas

These JSON Schemas describe the portable boundary between an agent runtime and
the deterministic distillation workflow. They do not describe controller stage
artifacts or physical ValidationReports, which retain their existing contracts.

- `agent_spec.schema.json`: role registration and permissions
- `agent_task.schema.json`: one Director-to-agent assignment
- `agent_result.schema.json`: one agent-to-Director response
- `judge_vote.schema.json`: the Judge's stricter direct response

Judge votes continue to use the stricter gate schema under `gates/schema/`.
