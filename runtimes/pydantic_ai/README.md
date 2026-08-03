# PydanticAI runtime (additive, optional)

A fourth agent-execution frontend alongside `claude/`, `codex/`, `manual/`. It gives the
repository **provider-neutral model invocation with typed output and a restricted,
read-only tool surface** — without touching the scientific workflow engine.

## What it does NOT change

- `workflow/controller.py` remains the sole owner of durable workflow/gate/retry/recovery
  state. This runtime never mutates controller state; it only produces a *candidate* result.
- The canonical contracts stay the JSON Schemas in `orchestration/schema/` and the
  validators in `orchestration/exchange.py` + `validation/`. The Pydantic models here are a
  typed *parsing* layer only.
- The Claude/Codex/manual runtimes are untouched.

## The acceptance pipeline (Pydantic parse is never enough)

```
runtime.run()                      # candidate result + full provenance record
  -> validate_agent_response(...)  # EXISTING contract + JudgeVote-lens validation
  -> FileExchangeRuntime.accept    # raw preservation + record   (primary mode)
```
A result that only parses as Pydantic but fails `validate_agent_response` is **not**
accepted (see `driver.py`). Type validation and physics/domain validation stay separate.

## Install & run

Core install has no Pydantic. Enable this runtime explicitly:

```bash
pip install -e .[pydantic-ai]      # adds pydantic + pydantic-ai only
```

Provider/model and API key come from the environment (never committed). Set them on
`RuntimeContext` and via the provider's standard env var:

```python
from runtimes.pydantic_ai.models import RuntimeContext
from runtimes.pydantic_ai.pydantic_ai_runtime import PydanticAIRuntime
from runtimes.pydantic_ai.driver import run_task

ctx = RuntimeContext(exchange_dir="runs/<run>/exchange", repo_root=".",
                     provider="anthropic", model_id="claude-...",
                     read_allow_prefixes=["runs/<run>"])   # read-only allow-list
result = run_task(PydanticAIRuntime(provider="anthropic", model_id="claude-..."),
                  task, spec, ctx)
```

Without a provider, use `MockAgentRuntime` (a test double) — the PoC and tests run with no
API key. This is the default path for CI and local development.

## Restricted tools (improves on unrestricted Bash)

In the Claude Code frontend every agent holds an unrestricted `Bash` tool and can read
anything on disk (`work/agent-framework-audit.md` §6). Here the real `pydantic_ai.Agent`
is given exactly two read-only tools — **`read_text` and `read_json`**. `EXPOSED_READ_TOOLS`
defines the expected and manifested tool surface; the Agent registers its tools explicitly,
and a network-free integration test verifies the Agent's actual registration matches it.
Both tools share the identical policy: every path is checked against an explicit allow-list
(resolved with `realpath`, so a symlink or `..` that escapes the allow-list is blocked and
prefix confusion like `/repo` vs `/repo-safe` is not possible), secret-like path components
are refused, only whitelisted text extensions are allowed (binaries blocked), files are
read as UTF-8, and a per-file and per-invocation byte budget is enforced. Each call records
exactly one invocation under its own tool name whose `ok` reflects the WHOLE operation: a
`read_json` invocation is `ok` only when file access, UTF-8 decoding, AND JSON parsing all
succeed. Access, decoding, and parsing failures are recorded as `ok=False` and returned to
the model as an explicit, distinguishable refusal (`ACCESS DENIED` / `READ ERROR` /
`INVALID ENCODING` / `INVALID JSON`), never a crash. **No directory-listing or glob tool is
exposed**, so a recursive dump of large files (WAVECAR, CHGCAR, trajectories) into the model
context is impossible by construction, not merely by a count limit. A review/planner agent
needs nothing more — heavy compute stays in the controller/adapters.

## Shadow mode (safe comparison before switching)

`run_task(..., shadow=True)` validates and records provenance but **never** accepts into
the exchange, so a shadow PydanticAI run cannot change controller-visible state. Use it to
compare this runtime against the current Claude runtime on the same tasks before making it
primary.

## Provenance

Every attempt writes a `RuntimeInvocationRecord` under `exchange/provenance/<task>.<attempt>.json`
holding the raw response, parsed result, prompt/input/tool hashes, tool-call log, token
usage, and any validation failure. Retries keep distinct attempt records; raw responses are
never overwritten.

## Retry / usage accounting

`RuntimeInvocationRecord.usage_source` labels token counts as `mock` / `test-model` /
`provider` / `estimated` / `unavailable`, so a mock or TestModel count is never mistaken
for billable provider usage. Each record carries `attempt_id`, optional `parent_attempt_id`,
and a `retry_category`.

Retry layers and their status in this PoC:

| retry layer | status |
|---|---|
| agent invocation retry (re-run the same task) | recorded as distinct attempts (no raw overwrite) |
| provider retry / HTTP 429 backoff | **not implemented** (arrives with a real provider, P4) |
| model/structured-output retry | **not implemented** (pydantic_ai handles some internally; not surfaced) |
| controller task retry | owned by `controller.py` (unchanged) |
| scientific recovery | owned by `controller.py` (unchanged) |

## What this PoC proves / does not prove

| Proven (mock + real TestModel, no network) | Not yet proven |
|---|---|
| Common `AgentRuntime` interface; mock and real share `build_invocation` | Real external provider tool calling |
| `task → runtime → existing validate_agent_response → accept` pipeline | Real structured-output retry, timeout, HTTP 429, provider failure |
| Pydantic parse success alone never accepts | Real provider-returned token usage |
| Raw + parsed output preserved; attempt-scoped provenance | Real model's evidence-adherence rate |
| Shadow mode leaves controller state unchanged | Scientific parity with the Claude runtime |
| Read-only allow-list; symlink/secret/binary/size/budget blocked + recorded | Producer-agent side-effect control (P7) |
| A real `pydantic_ai.Agent` runs via `TestModel`; valid→accept, wrong-lens→reject, blocked-tool→refusal recorded | Full compatibility across pydantic_ai versions beyond the pinned range |

## CI

- **core job** (`pip install -e .`): the optional deps are absent, so these runtime tests
  skip — this confirms core-CI compatibility.
- **`pydantic-ai-runtime` job** (`pip install -e .[pydantic-ai]`): installs the pinned
  optional deps and runs `tests/test_pydantic_ai_runtime.py`, **failing if every test
  skipped**. No provider API key; the real Agent is driven by `TestModel`.

## Version compatibility

Verified with `pydantic-ai-slim==0.8.1` and `opentelemetry-api` `>=1.26,<1.44` (1.44+
removed the private `opentelemetry._events` module that pydantic-ai-slim 0.8.x imports).
The pins in `pyproject.toml [project.optional-dependencies].pydantic-ai` reflect exactly
what was installed, imported, and tested on 2026-08-03.

## Rollback

Everything is additive: remove `runtimes/pydantic_ai/`, the optional dependency group, the
CI job, and `tests/test_pydantic_ai_runtime.py`, and the core (`controller.py`, validators,
adapters, existing runtimes) is untouched.

## Status

PydanticAI-compatible runtime **PoC**, experimental / optional. Exercised on the read-only
`judge` role with both `MockAgentRuntime` and a real `pydantic_ai.Agent` (TestModel, no
network). Next: an actual provider read-only smoke test (P4) and a Claude-vs-PydanticAI
golden shadow comparison (P5). See `work/agent-framework-transition-plan.md`.
