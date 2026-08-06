# PydanticAI Full-Runtime — Phase 1 Architecture Audit & Plan (2026-08-06)

Source-grounded audit of `origin/main` @ `914f404` (verified against actual code + tests, not
docs). Implementation branch: `feat/pydanticai-full-runtime` (worktree off main, clean).
Baseline env: clean Python 3.10.19 venv; pydantic 2.13.4, pydantic-ai-slim 0.8.1, ase 3.29.0,
numpy 2.2.6, scipy 1.15.3. Baseline tests: full suite **157 OK / 4 skipped**; runtime **29 OK**.

## 0. Baseline boundary (Phase 0)

- Claude baseline = the frozen Claude-Code Stage-6 (`teacher_labeling` PASS) run, owned by the
  fresh-run session, preserved read-only for runtime-migration comparison. This session does NOT
  modify, resume, or advance it. Recommended identifier: `claude-baseline-stage6`. A Git tag pins
  code, not run artifacts — a hash-bound baseline manifest + artifact inventory + vote-bundle
  hashes must be supplied separately (request from the baseline session if needed).

## 1. Current-state assessment (what is REAL today)

### 1a. PydanticAI runtime — import-only PoC, NOT wired to the controller
- Pydantic models (`runtimes/pydantic_ai/models.py`): `EvidenceReference`, `CriterionCheck`,
  `JudgeVoteModel`, `RequestedApproval`, `AgentResultModel`, `ToolInvocationRecord`,
  `ValidationErrorRecord`, `RuntimeContext`, `RuntimeInvocationRecord`. **`AgentTask` is a plain
  dict** (no mirror model). **Only Judge has a role-specific output (`JudgeVoteModel`); all other
  six roles share the generic `AgentResultModel`.** Richer return types in the specs
  (`RunPlan`, `DatasetManifest`, …) are prose labels with no schema.
- Acceptance pipeline (`driver.py:41-70`) is SOUND: `runtime.run()` → `validate_agent_response`
  → `FileExchangeRuntime.accept()` (re-validates). **Pydantic parse alone never accepts** (tested).
- Tool surface (`tool_registry.py`): `read_text`, `read_json` only. Strong security (realpath
  containment, symlink block, secret-component denylist, extension allow-list, 1 MB/file +
  4 MB/invocation budget, UTF-8/JSON validation, refusal-not-crash). **No dir-list/glob/write/
  shell.** Allow-list is per-invocation (caller-set prefixes), **not keyed to role**.
- Modes: `shadow` + primary only (`driver.run_task`). **No dry-run.** Shadow never mutates the
  exchange (and never touches the controller in any mode).

### 1b. Genuinely-absent capabilities (not merely untested)
- **Provider-exception-before-output loses the attempt record** — `agent.run_sync` (`pydantic_ai_runtime.py:78`)
  and `runtime.run()` (`driver.py:49`) are un-try/excepted; `_write_provenance` runs only after a
  successful return. A provider failure = no `RuntimeInvocationRecord`.
- **No retry/backoff/429/rate-limit.** `RuntimeContext.max_retries`/`timeout_s` are DEAD fields
  never read by `PydanticAIRuntime` (its ctor is `__init__(*, model=None, usage_source="provider")`).
- **No secret redaction** in any exception/validation-error string.
- **No production CLI** for the runtime — import-only; `[project.scripts]` has no runtime entry.
- **README constructor example is wrong** (`PydanticAIRuntime(provider=…, model_id=…)` → `TypeError`).
- **No `[anthropic]` extra on main** (the PR that added it was closed). `requirements.txt` still
  says `ase>=3.22` while pyproject is `>=3.23` (drift).

### 1c. Controller — solid sole state owner; strong gate
- `RunController` owns `manifest.json` (schema_version 6): stage ordering, attempts, artifact
  SHA-256 + git `code_revision` binding, gate recording, recovery + human-approval state machine,
  atomic writes, cascade invalidation. Three-judge gate (`_validate_vote_bundle`) is fail-closed:
  exactly 3 votes, ordered run-bound lenses, criteria == bound criteria, artifact-hash bound,
  **decision recomputed server-side** (any FAIL→FAIL; all-PASS→PASS; else REVISE); PASS requires
  the full validated bundle.
- **No runtime→controller hook.** The runtime writes to `exchange/provenance` + `exchange/results`;
  the controller reads `manifest.json` + artifacts. Two disconnected stores; the only bridge is a
  human running `complete_external_stage`/`record_gate`/`gate` CLIs. **No idempotency keys.**
  **No stale-running detection** (SIGKILL between `running`→`completed` leaves a stage `running`
  forever — the known bug, unmitigated).

### 1d. Executor / validator layer — rich, typed, but CLI-reached; no registry
- Deterministic compute exists and returns typed, hash-bound results: `adapters/*` (acquisition,
  student, md_backend, reference_dft, teacher, uncertainty, simple_nn_v2_*), `workflow/steps.py`
  (split/merge/train_committee/evaluate_committee/run_md), `validation/*` (four_channel_audit,
  committee_uncertainty, structure_dynamics, report, data_coverage, teacher_baseline, surface_energy).
- **No generic `action_type → callable` registry.** Dispatch is (a) config-`kind`/dotted-callable
  inside each adapter and (b) shell-command strings run by `RunController.run_stage`. The only
  importlib dotted-callable registry is validator-side (`workflow/contracts.validate_validation_manifest`).
- **Unbacked actions the design wants:** fine-tune/ER training; EOS, elastic/mechanics, ADF,
  S(Q)/FSDP, ring statistics, channel-(d) student-MD-vs-DFT, actual DFT execution (only INCAR
  render exists), and any scheduler/queue submission. Dedup is exact-sha256 only.

### 1e. How the 7 roles run TODAY
- Via the **Claude Code Agent tool driving the markdown prompts** (implicit) + `gates/gate_vote.workflow.js`.
  `runtimes/{claude,codex,manual}/` are READMEs/conventions, not provider-calling code. Tool grants
  live in `.claude/agents/*.md` front matter (Claude-Code-enforced), NOT in `orchestration/` code.
  **6 of 7 roles are granted unrestricted `Bash`** (all but Literature); 4 producers get `Write`/`Edit`.

## 2. Role · capability · tool matrix (current reality)

| Role | role_type | result_contract | typed output today | approval boundaries | tools granted (`.claude/agents`) | side-effect | controller path | production-ready |
|---|---|---|---|---|---|---|---|---|
| Orchestrator | coordinator | AgentResult | generic | costly_training, production_md, reference_calculation, destructive_or_public_action | Agent(6 roles),Read,Write,Edit,**Bash**,Glob,Grep,AskUserQuestion,Skill | state-mutating (sole controller writer) | direct | NO |
| Literature | producer | AgentResult | generic | — | Read,Grep,Glob,WebSearch,WebFetch | read-only | indirect | NO |
| Data Curator | producer | AgentResult | generic | costly_teacher_labeling | Read,Write,Edit,**Bash**,Glob,Grep | producer (writes) | indirect | NO |
| ML Trainer | producer | AgentResult | generic | costly_training, teacher_fine_tuning | Read,Write,Edit,**Bash**,Glob,Grep | producer (writes) | indirect | NO |
| Simulation | producer | AgentResult | generic | production_md, reference_calculation, scheduler_submission | Read,Write,Edit,**Bash**,Glob,Grep | producer + scheduler | indirect | NO |
| Analyst | producer | AgentResult | generic | — | Read,Grep,Glob,**Bash** | read-mostly (Bash, no Write) | indirect | NO |
| Judge | reviewer | **JudgeVote** | **typed** | — | Read,Grep,Glob,**Bash** | read-only | indirect (votes→gate) | PoC only (TestModel) |

Production-ready via actual PydanticAI provider = **none** (Judge is the only typed role and only
TestModel-tested). "Production readiness" per role requires: typed IO, role-scoped tool/action
registry, provider-tested, dry-run, sandbox-primary, controller integration, approval enforcement,
failure provenance, security tests.

## 3. Target architecture (honors Section A invariants)

Layered, additive; controller stays the single durable-state authority; PydanticAI = invocation
only; no LangGraph.

```
AgentTaskModel (typed mirror of canonical JSON Schema)
  → PydanticAIRuntime.run()  [provider invocation, role-scoped toolset, provenance-always]
  → typed role output (discriminated union; Judge=JudgeVote, producers=ActionProposal subtypes)
  → Pydantic parse → canonical JSON Schema → validate_agent_response
  → role/action-specific deterministic validator → policy/approval validator
  → dry-run render → input/hash validation → idempotency check
  → trusted executor (action_registry → controller/adapter)  [heavy compute here, NOT in agent]
  → artifact generation → artifact validator → AgentResult → three-Judge gate (controller)
```

Key components to build:
- **Typed input mirror** (`AgentTaskModel` + sub-models) with schema-drift tests.
- **Role-specific typed output** via discriminated union (no `dict[str,Any]`).
- **Provider config** consumed by the runtime (provider/model/timeout/retries/correlation-id).
- **Failure-always provenance** (try/except around invocation; redact secrets).
- **Bounded retry** (provider + structured-output, backoff+jitter, retryable classification, cost cap).
- **Production CLI** (`distill-agent run-task --runtime pydantic-ai …`, modes + exit codes).
- **Role-scoped tool/action registries** (machine-readable manifests; read-only tools + typed
  action registries per producer; Analyst versioned analysis registry; Orchestrator typed bridge;
  Literature typed source backend).
- **ActionProposal → trusted executor** bridge (`action_type → callable` registry mapping to the
  existing adapters/steps/validation functions; heavy compute stays in controller/adapters).
- **Controller integration** (provenance/attempt references in manifest; idempotency; background
  submit/poll; stale-running hardening).

## 4. Files to change / add (by phase; illustrative, will refine)

- **Phase 2 (common hardening):** `runtimes/pydantic_ai/models.py` (add AgentTaskModel + sub-models,
  provider fields, retry fields), `interface.py`, `pydantic_ai_runtime.py` (try/except+redaction+retry+
  timeout), `driver.py` (dry-run mode, provenance-always), NEW `runtimes/pydantic_ai/cli.py`,
  NEW `runtimes/pydantic_ai/retry.py`, NEW `runtimes/pydantic_ai/redaction.py`, `pyproject.toml`
  (add `[anthropic]` extra + runtime script), `README.md` fix. Tests: extend `tests/test_pydantic_ai_runtime.py`,
  NEW `tests/test_pydantic_ai_schema_drift.py`.
- **Phase 3 (tool registry):** NEW `runtimes/pydantic_ai/tool_manifests/*.yaml` (per role), extend
  `tool_registry.py` (role-scoped allow-list + new read-only summary tools), NEW read tools
  (`read_csv_summary`, `read_artifact_manifest`, `read_controller_status`, …). NEW
  `analysis/registry.py` (Analyst), `orchestration/bridge.py` (Orchestrator typed bridge),
  NEW `literature/sources.py` (typed source backend). Tests: NEW `tests/test_role_tools.py`.
- **Phase 4-5 (action proposal + controller integration):** NEW `runtimes/pydantic_ai/actions/*.py`
  (typed ActionProposal models + `action_type → executor` registry mapping to adapters/steps),
  `orchestration/exchange.py` (idempotency key, dedupe), `workflow/controller.py` (provenance/attempt
  reference fields, stale-running heartbeat/pid, `complete_external_stage` idempotency). Tests: NEW
  `tests/test_action_proposals.py`, `tests/test_controller_integration.py`.
- **Phase 6-8 (tests/CI/shadow/readiness):** `.github/workflows/ci.yml` (role/action + controller-
  integration jobs; keep min-ase), NEW `tests/test_dry_run_end_to_end.py`, NEW
  `examples/pydantic_ai_provider_smoke.py` (credential-gated, 7 roles), NEW shadow-comparison harness.
- Docs: the 20 deliverables in Section L; README + transition-plan updates.

Estimated scope: ~30-45 files touched/added; core semantics files (`controller.py`, validators,
adapters) changed only additively with regression tests.

## 5. Major risks

- **Backward-compatibility:** controller manifest schema is `schema_version 6`; adding provenance/
  attempt/heartbeat fields must bump to 7 with a migration + not break existing runs. Regression
  tests on `test_controller.py` mandatory.
- **Security:** widening the tool surface (producer action registries) must NOT re-introduce
  arbitrary shell/write. Every producer action must be a typed selection into a versioned registry;
  the LLM never returns an executable string. Redaction must cover provenance + logs.
- **Provider/version:** pydantic-ai-slim 0.8.x pin is fragile (opentelemetry `<1.44` constraint);
  a real provider adds the `anthropic` SDK dependency (version-sensitive). Structured-output +
  tool-calling semantics differ by provider — must be abstracted and documented.
- **Scope creep vs unbacked science actions:** EOS/mechanics/ADF/SQ/ring-stats/channel-d/DFT-exec/
  scheduler have no backing function; the runtime must expose only actions that have a validated
  executor, and mark the rest explicitly NOT-AVAILABLE (no fabricated capability).
- **Cost & non-determinism:** actual-provider calls cost money and are non-deterministic — gated
  behind explicit approval; golden-shadow parity uses small frozen task sets, reported as
  "golden-task parity check", not statistical equivalence.
- **Baseline immutability:** must not treat a Git tag as run-artifact immutability; need a hash-bound
  baseline manifest before any comparison claim.

## 6. Provider / API requirements

- **Anthropic** preferred (per Section D3). Requires `ANTHROPIC_API_KEY` + `PYDANTIC_AI_MODEL`
  (env only, never committed). Add a `[anthropic]` optional extra (pin the SDK to a tested version;
  the earlier smoke used `anthropic>=0.61,<0.65`). If no Anthropic credential is available, any
  other pydantic-ai-supported provider with a real credential may substitute, documenting the
  provider/SDK/structured-output/tool-calling/token-semantics differences.
- Actual-provider calls (Phase H8 smoke, Phase 7 shadow, Phase 9 run) are **cost events** →
  explicit researcher approval required before the first paid call (Section N). CI never runs them.

## 7. Estimated development time (engineering-days, rough)

| Phase | Scope | Est. |
|---|---|---|
| 2 | typed IO, provider config, failure provenance, retry, CLI | 4-6 d |
| 3 | role-scoped tool/action registries + read-only tools | 4-6 d |
| 4-5 | typed ActionProposal + controller integration + stale-running | 5-8 d |
| 6 | test matrix (schema/runtime/tool/role/controller/dry-run) | 4-6 d |
| 7 | golden shadow comparison harness + run | 2-3 d (+ provider cost) |
| 8 | readiness gate assembly | 1-2 d |
| docs | 20 deliverables + README/transition-plan | 2-3 d |
| **Total (impl+test, excl. Phase 9 science)** | | **~22-34 engineering-days** |

Phase 9 (fresh authoritative run) is a separate, approval-gated scientific effort with its own
compute budget (teacher labeling + 4-seed training + MD + optional DFT), not included above.

## 8. Recommended execution order for this session

Proceed network-free, local atomic commits only (no push/PR without approval; no provider calls;
no fresh run; no costly compute): Phase 2 → 3 → 4-5 → 6 → (Phase 7/8 blocked on provider approval).
Stop and report before: any public push/PR/merge; the first paid provider call; starting the fresh
scientific run; any costly training/MD/DFT/scheduler submission.
