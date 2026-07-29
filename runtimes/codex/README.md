# Codex frontend

Start Codex at the repository root. The root `AGENTS.md` tells the main task to
act as Director and to load the canonical Director specification and prompt.
When the Codex environment supports sub-agents, dispatch specialist work in
separate contexts according to `agent_specs/*.yaml`. Otherwise use the
file-exchange procedure and keep the Director as the single controller writer.

Example request:

```text
이 저장소의 runtime-neutral multi-agent workflow로 새 MLIP 증류 run을
시작해 주세요. 먼저 agent_specs/director.yaml과 agents/director.md를 읽고,
필요한 과학적 입력과 승인 경계를 저와 함께 정해 주세요.
```
