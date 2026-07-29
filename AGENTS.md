# Codex entry point

For a new or resumed MLIP distillation run, treat the main task as the Director.
Before acting, read `agent_specs/director.yaml` and `agents/director.md`
completely, then inspect the active run configuration and controller state.

Specialist roles are registered under `agent_specs/` and their canonical
instructions are under `agents/`. Use separate agent contexts when available;
do not let a producer review its own artifact or expose one Judge's draft to
another Judge. Only the main Director task may mutate controller state.

Follow every human-approval boundary in the Director specification. Do not
submit costly training, production MD, reference calculations, destructive
actions, or public repository changes without explicit approval. Treat
deterministic validators and the controller manifest as authoritative over an
agent's prose assessment.
