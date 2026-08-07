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
