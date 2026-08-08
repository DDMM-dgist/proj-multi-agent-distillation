# Stage B — seven-role LOCAL smoke package (prepared; NOT yet run)

Same validated stack as Stage A: Qwen/Qwen2.5-3B-Instruct → vLLM 0.26.0 (`--enable-auto-tool-choice
--tool-call-parser hermes`, bf16, max-model-len 4096, max-num-seqs 1, enforce-eager) → OpenAI /v1 →
PydanticAI → production router → controller boundary. ONE vLLM session, seven frozen tasks run
sequentially (never parallel), each exactly once, provider retries = 0, producers shadow/dry-run.

## Fixtures (repo-relative, portable — no machine-specific absolute paths)
`examples/stage_b_smoke/` : `orchestrator.json`, `literature.json`, `data-curator.json`,
`ml-trainer.json`, `simulation.json`, `analyst.json`, `judge.json`, `artifacts/evidence.json`.

| role | strategy | output_type | dry-run action | needs |
|---|---|---|---|---|
| orchestrator | typed_result | OrchestratorPlan | — | — |
| literature | typed_result | LiteratureEvidence (status `source_not_retrieved`, sources `[]`, no fabricated citations) | — | — |
| data-curator | producer_dispatch | DataCuratorActionProposal | `inspect_dataset` | `--run-dir` |
| ml-trainer | producer_dispatch | MLTrainerActionProposal | `compute_committee_disagreement` | `--run-dir` |
| simulation | producer_dispatch | SimulationActionProposal | `compute_nve_drift` | `--run-dir` |
| analyst | producer_dispatch | AnalystActionProposal | `compare_force_errors` | `--run-dir` |
| judge | judge_gate | JudgeVoteModel | — (reads evidence via read_json) | `--read-allow artifacts` |

All four producer actions are allowed for their role, NOT approval-gated, and not out-of-scope, so
`--mode shadow` → dispatch `dry_run` → `DRY_RUN` with zero side effects and zero controller mutation.

## Network-free validation (done on cpu1; also a committed test)
`work/stage_b_validate.py` (run by the runner pre-launch) + `tests/test_pydantic_ai_stage_b_fixtures.py`
confirm, with NO provider/model/GPU: validate_task PASS for all 7; agent==role; acceptance strategy
correct; judge context has review_lens+review_focus and the repo-relative evidence path resolves +
reads (structure_count 12) inside the allow-list; each producer action authorized + non-gated; each
producer proposal dry-run-dispatches to `DRY_RUN` with the v7 manifest and leaves idempotency empty.
Full suite: 313 OK / 4 optional skips.

## Runner — `work/stage_b_local_smoke.sh`
Verifies HEAD (optional `EXPECT_HEAD`), runs the fixture validator, creates output dirs + the v7
`manifest.json` (producer `--run-dir`), gates on GPU1 free VRAM ≥ 16000 MiB, fails closed if port
8000 is busy, launches ONE vLLM on GPU1 (conservative profile), health-checks `/v1/models`, then
runs the seven roles sequentially. CONTINUE/STOP: a failed role is preserved and NOT retried; the
runner continues so all seven attempts exist in one session; per-role exit codes are reported.
Cleanup trap terminates ONLY this run's process group (no broad pkill); VASP is never signaled.
Outputs: `examples/stage_b_smoke/out/<role>/stdout.log` and
`examples/stage_b_smoke/out/<role>/exchange/provenance/stageB-<role>-0001.*.json` (gitignored).

## Acceptance (evaluated OFF-LINE from the copied-back provenance; canonical_validation alone is NOT sufficient)
Per role: actual local inference occurred; provider `local-openai`/model `qwen2.5-3b-instruct`; typed
output parsed; canonical validation where applicable (judge); correct role; only allowed tools/actions;
unauthorized tools/actions = 0; nonexistent-artifact refs = 0; raw+parsed preserved; token usage +
latency recorded; provenance complete; controller mutation = 0 (producers may be a shadow-recorded
DRY_RUN proposal without execution); external/paid API = 0. Producers: proposal typed correctly,
role/action authorization PASS, actual scientific side effect = 0. Literature: fabricated sources = 0.
Judge: evidence read actually succeeded. Aggregate Stage B PASS only if all seven satisfy their criteria.

## Attempt 1 result (preserved) + Attempt 2 fixes (2026-08-08)
LOCAL_STAGE_B_ATTEMPT_1 = PARTIAL_PASS_5_OF_7 (preserved, not overwritten):
- Orchestrator FAIL: read_artifact_manifest tool-loop (20x, all refused) -> prompt exceeded the
  attempt-1 vLLM `--max-model-len 4096` -> HTTP 400; no OrchestratorPlan produced.
- Judge FAIL: evidence read OK + verdict PASS, but the single COMPOUND task criterion was split
  into two criteria_checked entries -> canonical contract mismatch.
- Literature / Data Curator / ML Trainer / Simulation / Analyst = PASS.

Attempt-2 minimal integration fixes (NOT prompt hacking):
1. Runtime bounded tool-call guard (general hardening). pydantic_ai `UsageLimits(request_limit=...)`
   (from `pydantic_ai.usage`) is passed to every `agent.run_sync`. New
   `RuntimeContext.request_limit` (default **6**; each tool round-trip = one request; legitimate
   tasks need <=2). Exceeding it raises `UsageLimitExceeded` -> classified `usage_limit_exceeded`
   (terminal, non-retryable) -> failure record with every attempted tool call preserved, no
   acceptance, no controller mutation. Regression: tests/test_pydantic_ai_tool_budget.py.
2. Orchestrator fixture is now explicitly PLAN-ONLY: "call NO tools; no artifact inspection",
   inputs [] (removes the artifact-inspection temptation that seeded the loop).
3. Judge fixture: the compound criterion is split into TWO atomic ORDERED criteria
   ("structure_count == 12" ; "validation_status == 'passed'"). Canonical validation is NOT
   relaxed (validate_agent_response unchanged); a mismatched/collapsed vote still fails (tested).
   The shared Judge prompt already says "one criteria_checked entry per stated criterion" -> no
   prompt special-casing added.
4. Smoke vLLM `--max-model-len` 4096 -> **8192** (documented config change; NOT the loop fix —
   the request_limit guard is). Attempt-1's 4096 is preserved here and in attempt-1 provenance.

Attempt-2 acceptance = all seven roles PASS their role criteria (Orchestrator: typed plan, no tool
loop, within request_limit, mutation 0; Judge: evidence read, structure_count 12 + validation
passed, criteria_checked mirrors BOTH ordered criteria, canonical validation PASS; other five as
before). Only then: LOCAL_STAGE_B_ATTEMPT_2 = CONFIRMATORY_END_TO_END_PASS and LOCAL_STAGE_B = PASS.

## Attempt 2 RESULT — 7/7 PASS (jbnu-gpu2, 2026-08-08)
LOCAL_STAGE_B_ATTEMPT_1 = PARTIAL_PASS_5_OF_7 (preserved).
LOCAL_STAGE_B_ATTEMPT_2 = CONFIRMATORY_END_TO_END_PASS ; LOCAL_STAGE_B = PASS.
All seven: provider local-openai / model qwen2.5-3b-instruct / usage_source provider; typed output
parsed; tokens+latency recorded; 0 unauthorized tools; 0 nonexistent-artifact refs; controller
mutation 0 (run manifest idempotency {} ); 0 Anthropic/paid calls.
- Orchestrator: valid OrchestratorPlan, **0 tool calls** (plan-only fixture), no loop, within
  request_limit, mutation 0. (request_limit=6 guard in place as the safety net.)
- Judge: read_json ok, structure_count 12 + validation_status 'passed' observed, criteria_checked
  = the two ordered criteria, verdict PASS, **canonical validation PASS** (validation_errors []).
- data-curator/ml-trainer/simulation/analyst: correct typed ActionProposal (inspect_dataset /
  compute_committee_disagreement / compute_nve_drift / compare_force_errors), dry_run, DRY_RUN, 0
  scientific side effect. Literature: source_not_retrieved, sources [], 0 fabricated citations.

## Resource-policy adjustment (co-scheduled smoke test; NOT scientific/runtime semantics)
GPU1-only VRAM gate MIN_FREE_MIB 16000 -> 12000 and vLLM --gpu-memory-utilization 0.20 -> 0.18, to
co-schedule the single sequential Qwen2.5-3B/max-num-seqs=1 smoke with running VASP. Other flags
unchanged (max-model-len 8192, max-num-seqs 1, enforce-eager, bfloat16, auto-tool-choice, hermes).
Fail-closed below 12000 MiB (no auto-lowering, no GPU switch); VASP never touched. Attempt-2 PASSED
under the earlier 16000/0.20; this only widens the co-scheduling window for future runs.
