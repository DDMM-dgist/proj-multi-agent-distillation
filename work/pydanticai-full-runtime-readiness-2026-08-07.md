# PydanticAI full-runtime readiness (network-free) — 2026-08-07

Network-free assessment. Actual-provider validation is PENDING (no key; no call made).

## Role readiness

| role | typed input | typed output | tool manifest | runtime enforcement | controller integ. | approval | idempotency | failure provenance | dry-run | sandbox-primary | actual-provider | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orchestrator | AgentTaskModel | OrchestratorPlan | yes | yes | yes | yes | yes | yes | yes | yes | PENDING | NETWORK_FREE_READY |
| literature | AgentTaskModel | LiteratureEvidence | yes | yes | yes | yes | yes | yes | yes | yes | PENDING | NETWORK_FREE_READY |
| data-curator | AgentTaskModel | DataCuratorActionProposal | yes | yes | yes | yes | yes | yes | yes | yes | PENDING | NETWORK_FREE_READY |
| ml-trainer | AgentTaskModel | MLTrainerActionProposal | yes | yes | yes | yes | yes | yes | yes | yes | PENDING | NETWORK_FREE_READY |
| simulation | AgentTaskModel | SimulationActionProposal | yes | yes | yes | yes | yes | yes | yes | yes | PENDING | NETWORK_FREE_READY |
| analyst | AgentTaskModel | AnalystActionProposal | yes | yes | yes | yes | yes | yes | yes | yes | PENDING | NETWORK_FREE_READY |
| judge | AgentTaskModel | JudgeVote | yes | yes | yes | yes | yes | yes | yes | yes | PENDING | NETWORK_FREE_READY |

## Action readiness

| action_type | role | taxonomy | backing | validator | approval | real exec later | status |
|---|---|---|---|---|---|---|---|
| inspect_dataset | data-curator | READY_EXECUTOR | ase.io.read + metadata | - | - | no | READY_EXECUTOR |
| summarize_source_categories | data-curator | READY_EXECUTOR | frame metadata (cf. validation.data_coverage) | - | - | no | READY_EXECUTOR |
| sample_seed_pool | data-curator | READY_EXECUTOR | deterministic policy seed_pool_v1 | - | - | no | READY_EXECUTOR |
| reconstruct_lineage | data-curator | READY_EXECUTOR | adapters.acquisition.validate_lineage grouping | - | - | no | READY_EXECUTOR |
| generate_group_split | data-curator | READY_EXECUTOR | workflow.steps.split_dataset | workflow.steps split integrity | - | no | READY_EXECUTOR |
| label_with_teacher | data-curator | READY_HPC_APPROVAL_GATED | adapters.acquisition.label_with_teacher | labeling manifest integrity | costly_teacher_labeling | yes | READY_HPC_APPROVAL_GATED |
| validate_label_preservation | data-curator | READY_EXECUTOR | ase.io.read + acquisition.validate_lineage | artifact completeness | - | no | READY_EXECUTOR |
| build_dataset_manifest | data-curator | READY_EXECUTOR | workflow.integrity.artifact_digest | - | - | no | READY_EXECUTOR |
| compare_deployment_coverage | data-curator | READY_EXECUTOR | validation.data_coverage.validate_data_coverage_report | data_coverage validator | - | no | READY_EXECUTOR |
| detect_duplicates | data-curator | READY_EXECUTOR | workflow.steps._structure_fingerprint | - | - | no | READY_EXECUTOR |
| detect_atomic_overlap | data-curator | READY_EXECUTOR | ASE get_all_distances(mic=True) | - | - | no | READY_EXECUTOR |
| prepare_student_inputs | ml-trainer | READY_EXECUTOR | adapters.student._render_simple_nn_config | - | - | no | READY_EXECUTOR |
| train_committee | ml-trainer | READY_HPC_APPROVAL_GATED | workflow.steps.train_committee | - | costly_training | yes | READY_HPC_APPROVAL_GATED |
| collect_checkpoints | ml-trainer | READY_EXECUTOR | committee manifest convention | - | - | no | READY_EXECUTOR |
| evaluate_heldout_fidelity | ml-trainer | READY_HPC_APPROVAL_GATED | workflow.steps.evaluate_committee | four_channel_audit | - | yes | READY_HPC_APPROVAL_GATED |
| summarize_seed_variation | ml-trainer | READY_EXECUTOR | adapters.uncertainty.committee_force_std | - | - | no | READY_EXECUTOR |
| compute_committee_disagreement | ml-trainer | READY_EXECUTOR | adapters.uncertainty.committee_force_std | - | - | no | READY_EXECUTOR |
| validate_training_completion | ml-trainer | READY_EXECUTOR | workflow.integrity.artifact_digest | artifact completeness | - | no | READY_EXECUTOR |
| run_teacher_md | simulation | READY_HPC_APPROVAL_GATED | adapters.acquisition.run_teacher_md | - | production_md | yes | READY_HPC_APPROVAL_GATED |
| run_student_md | simulation | READY_HPC_APPROVAL_GATED | workflow.steps.run_md / adapters.md_backend.run | - | production_md | yes | READY_HPC_APPROVAL_GATED |
| compute_rdf | simulation | READY_EXECUTOR | validation.structure_dynamics.compute_rdf | - | - | no | READY_EXECUTOR |
| compute_coordination | simulation | READY_EXECUTOR | validation.structure_dynamics.compute_coordination | - | - | no | READY_EXECUTOR |
| compute_minimum_distance | simulation | READY_EXECUTOR | ASE get_all_distances(mic=True) | - | - | no | READY_EXECUTOR |
| detect_force_spike | simulation | READY_EXECUTOR | ASE forces + norm | - | - | no | READY_EXECUTOR |
| compute_nve_drift | simulation | READY_EXECUTOR | validation.structure_dynamics.compute_nve_drift | - | - | no | READY_EXECUTOR |
| validate_simulation_completion | simulation | READY_EXECUTOR | artifact existence + finiteness | artifact completeness | - | no | READY_EXECUTOR |
| submit_scheduler_job | simulation | READY_INTERFACE_BACKEND_NOT_CONFIGURED | scheduler interface (no HPC backend configured) | - | scheduler_submission | yes | READY_INTERFACE_BACKEND_NOT_CONFIGURED |
| query_scheduler_job | simulation | READY_INTERFACE_BACKEND_NOT_CONFIGURED | scheduler interface (no HPC backend configured) | - | - | yes | READY_INTERFACE_BACKEND_NOT_CONFIGURED |
| collect_scheduler_artifact | simulation | READY_INTERFACE_BACKEND_NOT_CONFIGURED | scheduler interface (no HPC backend configured) | - | - | yes | READY_INTERFACE_BACKEND_NOT_CONFIGURED |
| compare_force_errors | analyst | READY_EXECUTOR | validation.four_channel_audit.channel | - | - | no | READY_EXECUTOR |
| compare_energy_errors | analyst | READY_EXECUTOR | validation.four_channel_audit.channel | - | - | no | READY_EXECUTOR |
| summarize_committee_disagreement | analyst | READY_EXECUTOR | adapters.uncertainty.committee_force_std | - | - | no | READY_EXECUTOR |
| compare_rdf | analyst | READY_EXECUTOR | validation.structure_dynamics.compute_rdf | - | - | no | READY_EXECUTOR |
| compare_coordination | analyst | READY_EXECUTOR | validation.structure_dynamics.compute_coordination | - | - | no | READY_EXECUTOR |
| fit_nve_drift | analyst | READY_EXECUTOR | validation.structure_dynamics.compute_nve_drift | - | - | no | READY_EXECUTOR |
| summarize_md_stability | analyst | READY_EXECUTOR | compose NVE drift + min distance (validation.structure_dynamics) | - | - | no | READY_EXECUTOR |
| classify_root_cause | analyst | READY_REASONING_OUTPUT | Analyst typed reasoning output (not a deterministic executor) | - | - | no | READY_REASONING_OUTPUT |

## Summary counts
- READY_EXECUTOR: 28
- READY_HPC_APPROVAL_GATED: 5
- READY_INTERFACE_BACKEND_NOT_CONFIGURED: 3
- READY_REASONING_OUTPUT: 1

## CI jobs (evidence)
- test (3.10, 3.12): core, pydantic tests skip
- min-ase (3.10, ase==3.23.0): declared floor
- pydantic-ai-runtime (3.10): runtime module, fails if all skip
- pydantic-ai-full (3.10): FULL suite with extra (schema/tool/dispatch/controller/executor/scheduler/root-cause/golden-shadow/seven-role E2E)
- max-ase (3.12, ase==3.29.0): max tested dependency
- actual-provider smoke: NOT in CI (manual, credential+approval gated)

## Verdict

```
NETWORK_FREE_FULL_RUNTIME_READY
ACTUAL_PROVIDER_VALIDATION_PENDING
```

No in-scope SiO2 action is NOT_IMPLEMENTED. All seven roles have typed IO, enforced tool
manifests, controller integration, approval + idempotency, failure provenance, dry-run and
sandbox-primary paths. Actual-provider smoke + golden-shadow provider run remain, behind
explicit approval. Do NOT claim PRODUCTION_READY / end-to-end-validated before that.

## Production routing (item 0) — COMPLETE
Single production router (runtimes/pydantic_ai/production_router.run_role, used by the CLI)
auto-selects the acceptance strategy per role from the typed output (judge_gate / producer_dispatch
/ typed_result / agent_result). All seven roles route from `distill-agent-run`; wrong-role output
is fail-closed; shadow never mutates. 6 routing tests pass. No scientific semantics changed.

## Actual-provider credential preflight (item 1, 2026-08-07) — DOES NOT PASS (no call made)
- ANTHROPIC_API_KEY: absent; PYDANTIC_AI_MODEL: absent.
- preflight_credentials() => NOT_CONFIGURED (network-free; no provider contacted).
- anthropic SDK: NOT installed (the [anthropic] optional extra was not installed in this venv).
- pydantic-ai-slim: 0.8.1 installed.
- Effective caps if configured: timeout_s=120, provider_retries=2, structured_output_retries=1,
  max_total_calls=3.
- The model identifier cannot be validated without the SDK + credential; NOT guessing a name/price.

Stage A/B/C (paid provider calls) are BLOCKED until a credential + the anthropic SDK are provided.
To enable: `pip install -e ".[pydantic-ai,anthropic]"`; export ANTHROPIC_API_KEY and
PYDANTIC_AI_MODEL=anthropic:<a model id valid for the account>; re-run preflight -> READY.

## Backend pivot: local OpenAI-compatible LLM (2026-08-07, L1-L3)
Anthropic billing is unavailable, so the inference backend pivots to a LOCAL OpenAI-compatible
server (vLLM first). See work/local-provider-audit-2026-08-07.md for the source-grounded audit.
- L1 provider abstraction (test/local-openai/ollama/anthropic-optional) + L2 fail-closed local
  preflight DONE; L3 model shortlist produced (no download/run).
- Local path requires NO ANTHROPIC_API_KEY and NO real API key (non-secret placeholder). New
  extra `.[pydantic-ai,local-openai]` (openai SDK). Production router unchanged.
- Full suite 309 OK / 4 optional skips. No server launched, no inference, local commits only.

Status update:
- ANTHROPIC_LIVE_PROVIDER = BILLING_UNAVAILABLE / NOT_REQUIRED_FOR_LOCAL_RUNTIME
- NETWORK_FREE_FULL_RUNTIME_READY
- LOCAL_PROVIDER_IMPLEMENTATION_PENDING  (L1/L2 done, L3 shortlisted; L4 server run pending)
- ANTHROPIC_BILLING_BLOCKED_NON_FATAL

## L4A topology (2026-08-07): reachable cluster is CPU-only; GPU host required
- JBNU OpenPBS cluster: master + j001, 64c/527GB each, naccelerators=0, no CUDA/vLLM (CPU-only
  DFT/MD: VASP/LAMMPS modules). /home is shared 66T; /tmp is local to master (worktree/venv NOT
  visible on other nodes). Backing git repo lived under /tmp -> CLONED to shared /home:
  /home/hyunjin/mad-pydanticai-persist/proj-mad-pydanticai-full-runtime (HEAD b383ffa, 16 commits
  preserved). No push/PR/merge; /tmp worktree retained.
- Decision (user): L4 runs on a SEPARATE GPU host with vLLM (NOT CPU llama.cpp/Ollama), server +
  client co-located so PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1. Portable deployment runbook:
  work/gpu-host-deployment-runbook.md (git-bundle transfer, env, packages, CUDA verify, smoke
  model, vLLM launch template, Stage A command, provenance). Nothing launched/downloaded; awaiting
  GPU-host info (nvidia-smi, python, internet, filesystem, scheduler, ports).

Status:
- NETWORK_FREE_FULL_RUNTIME_READY
- LOCAL_PROVIDER_IMPLEMENTATION_COMPLETE
- LOCAL_LIVE_STAGE_A_PENDING  (blocked: no GPU on reachable cluster; GPU-host deployment prepared)
- ANTHROPIC_BILLING_BLOCKED_NON_FATAL
- ANTHROPIC_NOT_REQUIRED_FOR_LOCAL_RUNTIME

## L4C LOCAL Stage A — PASS (jbnu-gpu2, 2026-08-07)
Executed commit 7694fe3; vLLM 0.26.0 / Qwen2.5-3B-Instruct / tool-parser hermes on GPU1
(co-scheduled with VASP, conservative profile: gpu-mem-util 0.20, max-model-len 4096,
max-num-seqs 1, enforce-eager). Server + PydanticAI client co-located; base_url 127.0.0.1:8000.

- ATTEMPT_1 (task attempt-6a8dcd89): INTEGRATION_VALIDATED / EVIDENCE_GROUNDING_BLOCKED_BY_FIXTURE_PATH
  (read_json refused an out-of-allow-list cpu1 absolute path -> verdict REVISE; security guard
  behaved correctly). Preserved, not overwritten.
- ATTEMPT_2 (task attempt-f35300d8): CONFIRMATORY_END_TO_END_PASS. read_json ok=true (rel path,
  90 chars valid JSON), evidence.json read, structure_count 12 observed, verdict PASS, typed
  JudgeVote parse PASS, canonical validation PASS, validation_errors [], lens evidence_provenance,
  unauthorized tools 0, nonexistent-artifact citations 0, raw+parsed preserved, tokens 3340/129,
  latency 2.069s, stdout.log persisted (506B), controller_mutated false (shadow), Anthropic/paid
  calls 0. vLLM terminated after; VASP PID 559725 unaffected.

Status:
- NETWORK_FREE_FULL_RUNTIME_READY
- LOCAL_PROVIDER_IMPLEMENTATION_COMPLETE
- LOCAL_LIVE_STAGE_A = PASS
- ANTHROPIC_BILLING_BLOCKED_NON_FATAL
- ANTHROPIC_NOT_REQUIRED_FOR_LOCAL_RUNTIME
Next (separate approval): Stage B seven-role local smoke (producers dry-run) — NOT started.

## L4 Stage B — PASS (2026-08-08)
LOCAL_STAGE_B = PASS (attempt-2 7/7 confirmatory; attempt-1 preserved PARTIAL_PASS_5_OF_7).
Bounded tool-call guard (request_limit) added; smoke VRAM policy relaxed to 12000/0.18 (co-scheduled, non-semantic). Stage C not started.
