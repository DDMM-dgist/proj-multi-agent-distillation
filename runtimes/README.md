# Agent runtime frontends

The distillation controller, adapters, validators, and canonical roles are
runtime-neutral. A frontend is responsible only for loading `agent_specs/*.yaml`
and `agents/*.md`, isolating specialist contexts, and returning contract-shaped
results to the Director.

Supported reference paths:

| Frontend | Entry point | Scope |
|---|---|---|
| Claude Code | `.claude/agents/`, `.claude/skills/` | packaged reference frontend |
| Codex | root `AGENTS.md` | interactive coding-agent frontend |
| Manual/file exchange | `python -m orchestration.cli` | provider-neutral handoff and testing |

“Supported” does not mean that the repository launches every vendor's agent
process. The frontend owns model authentication and context creation; the
repository owns the scientific role, task/result contracts, controller state,
and evidence.

Provider-specific wrappers must remain thin. Scientific rules belong in the
canonical Markdown prompts and deterministic Python modules.
