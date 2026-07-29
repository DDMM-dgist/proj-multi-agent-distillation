# Claude Code frontend

Claude Code is the packaged reference frontend. Start `claude` at the repository
root and invoke `/distill-start`. Files under `.claude/agents/` register the
roles and point to the canonical prompts under `agents/`; `.claude/skills/`
provides start, status, and resume entry points.

Do not copy the canonical prompt into a wrapper. Runtime tool names and model
selection may live in `.claude/`; scientific decision rules may not.
