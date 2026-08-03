# Project Status Snapshot — 2026-08-03

Evidence-based. Verified against actual files, Git history, tests, and PRs — not memory.
Repo: `DDMM-dgist/proj-multi-agent-distillation`. No code/dependency changes were made to
produce this snapshot. Terms follow the glossary in §18.

## 1. Executive verdict

- **FRAMEWORK V1: COMPLETE** — the generic, auditable MLIP-distillation *software* workflow
  (controller, stage contracts, validators, three-lens gate, recovery, provenance, optional
  typed runtime) exists and passes its network-free tests in a clean, declared-floor–compliant
  environment (full suite 157 tests, 0 fail, 0 err, 33 skip; §5). The earlier dependency-range
  compatibility gap is **resolved** by merged **PR #3** (`45d2dab`): minimum supported ASE is
  now **3.23**, verified across **ASE 3.23.0–3.29.0**, with a `min-ase` CI job and RDF +
  teacher-MD regression tests (§5a). This is a **software-implementation** verdict, not a
  scientific one.
- **SCIENTIFIC DEMONSTRATION: PARTIAL** — the SiO2 case has already demonstrated real
  scientific results (Allegro→SIMPLE-NN distillation, a 4-seed committee, teacher–DFT /
  student–teacher / student–DFT force evaluation, committee-uncertainty analysis, amorphous
  structure + RDF/ADF + dynamics validation, ~51–59× throughput, and error/REVISE detection
  during audit). What is **not** yet demonstrated is a *completed* recovery iteration
  (REVISE/FAIL → data change → teacher relabel → student retrain → same-profile revalidation),
  a second material/model system end-to-end, and a controlled comparison to a
  single-agent/manual/fixed pipeline. Hence PARTIAL, not INCOMPLETE.
- **MANUSCRIPT & REPRODUCIBILITY: PARTIAL** — the manuscript predates this session's framework
  changes; its scientific claims can be strengthened only within the demonstrated scope above,
  and the public reproducibility package (DOI/manifests) is not finalized.

"Code exists" is NOT treated as "scientifically demonstrated" anywhere below. Implemented (§6)
and scientifically demonstrated (§1 / §13) are kept strictly separate; see the three-way split
in §7.

## 2. Sources reviewed

Available to this local Claude Code session's filesystem (`~/.claude/uploads/<session>/`):
- 통합 회의록 (Agent workflow 및 논문 수정) — `*Agent_workflow*.pdf` (×3)
- 회의록 요약 — `_____________.pdf` (×3)
- 현재 원고 (Korean, commented) — `main_ko_comment*.pdf` (×4)
- 소재과학 Agentic AI 연구 동향 + 패키지/RAG 아키텍처 — `materialsagenticai*ko.pdf` (×2)
- initial→revised 비교 — `initial_to_revised_comparison_ko.pdf`
- Repo: `README.md`, `workflow/*`, `orchestration/*`, `validation/*`, `adapters/*`,
  `gates/*`, `agent_specs/*`, `configs/*`, `examples/*`, `tests/*`, `runtimes/pydantic_ai/*`,
  `.github/workflows/ci.yml`, `pyproject.toml`, `work/agent-framework-audit.md`,
  `work/agent-framework-transition-plan.md`.

## 3. Source-availability limitations

The catalyst-literature PDFs (Yan et al., TheMeCat, Xu et al.) were **NOT AVAILABLE TO THE
LOCAL REPOSITORY AUDIT SESSION**: they were referenced in a prior ChatGPT conversation but were
not present in this local Claude Code session's repository filesystem or source directory, so
they were **not directly inspected here**.

This does **not** mean the sources do not exist. Their workflows — PDF/OCR extraction, catalyst
composition/unit and GHSV harmonization, literature-wide missingness analysis, DOI fixed
effects, batch-effect decomposition, DAG adjustment, DML/AIPW, MNAR sensitivity — are
classified as a **separate literature-mining / data-harmonization / causal-analysis scope**,
not as missing MLIP-distillation features (see §10 and §7-C). Their *design principles*
(source separation, no silent inference, human-in-the-loop, failure preservation) are already
reflected in this framework.

## 4. Git and PR state

- `origin/main` = `f5739f0` (Merge PR #1).
- provider-smoke branch `pydantic-ai-provider-smoke`, HEAD `121e314` (committed + pushed).
- On the provider-smoke branch, the only untracked files are the two status documents; there
  are no unrelated working-tree changes and no provider-smoke code was modified.
- **PR #1** "Add experimental PydanticAI-compatible runtime": **MERGED** 2026-08-03
  (branch `pydantic-ai-runtime`, commits `49da5df`, `4170581`, `6798ea7`).
- **PR #2** "Add optional Anthropic provider smoke harness": **DRAFT, OPEN, not merged**
  (branch `pydantic-ai-provider-smoke`, commit `121e314`; the harness IS committed — it is not
  an uncommitted local change).
- These status documents live on a SEPARATE `project-status-docs` worktree based on
  `origin/main`, and are intentionally NOT part of the provider-smoke branch or PR #2.
- Earlier session work already on main via prior merges: Director→Orchestrator rename,
  Si/O order fix, gate thresholds, teacher-first recovery, SIMPLE-NN eval adapter,
  review-lens differentiation, accept-result exchange.

## 5. Test state (reproducible clean core environment; network-free)

Verified in a **freshly created venv** (base interpreter CPython **3.10.19**; `python3.10` was
not on `PATH`, so a clean CPython 3.10 from a conda env was used as the venv base and the venv
is fully isolated — deps installed fresh from `pip install -e .`). Declared-floor–compliant
dependencies: `scipy 1.15.3` (≥1.10), `ase 3.29.0` (≥3.22), `numpy 2.2.6`, `pyyaml 6.0.3`;
`pip check` = "No broken requirements found"; **pydantic and pydantic_ai absent**.

Final merged-implementation test numbers (post PR #3, `45d2dab`; `python -m unittest`,
network-free; `passed` marked *calculated* = total − failures − errors − skipped):

```
floor  ASE 3.23.0  full suite : total 157  failures 0  errors 0  skipped 33  passed 124 (calculated)
latest ASE 3.29.0  full suite : total 157  failures 0  errors 0  skipped 33  passed 124 (calculated)
       ASE 3.26.0  test_adapters: total  54  failures 0  errors 0  skipped  4  passed  50 (calculated)
```

The 33 skips are 29 "pydantic not installed" + 4 SIMPLE-NN. Optional-dependency isolation is
correct (NOT a collection defect): `python -m unittest tests.test_pydantic_ai_runtime` in the
core env gives `Ran 29 / OK (skipped=29)` with **0 import errors and 0 `_FailedTest`**.

GitHub CI cross-check on compliant deps (re-verifiable): PR #1 (run 30788964831) and PR #3
(workflow run 30832918739 — `test (3.10)`, `test (3.12)`, `min-ase`, `pydantic-ai-runtime`) all
SUCCESS.

**An earlier local figure (70 tests, 7 errors, 1 failure) was an artifact of a stale interpreter
(`scipy 1.9.1` below the `scipy>=1.10` floor + a Python without the optional deps) and is
superseded.** Classified: 4 `_FailedTest` = stale-interpreter optional-import collection
artifacts; 2 errors (`spearman`, `uncertainty_cli`) = `scipy 1.9.1` floor artifacts; the
`test_rdf...` failure and `test_teacher_md...` error were the **real ASE-version issue now fixed**
(§5a).

External provider: NOT executed (no API call in any test path).

### 5a. DECLARED-DEPENDENCY ASE COMPATIBILITY — RESOLVED (merged PR #3, `45d2dab`)

The former gap (two adapter paths passed on `ase 3.29.0` but failed on `ase 3.26.0`, inside the
declared range) is **resolved**:

- **RDF symbol-selector issue resolved** — the RDF adapter converts chemical symbols to atomic
  numbers for `ase.geometry.rdf.get_rdf`'s `elements` selector (keeping symbol-based keys) and
  raises on non-finite/malformed results.
- **teacher-MD FixCom extxyz issue resolved** — MD snapshots omit every `FixCom` (which ASE 3.26
  cannot serialize), preserve all other seed constraints, and record scalar provenance
  (`source_had_fixcom`, `runtime_fixcom_applied`, `snapshot_fixcom_omitted`).
- **Minimum supported ASE = 3.23** (`pyproject` floor `ase>=3.23`; 3.22 lacks `ase.geometry.rdf`).
- **Verified ASE range = 3.23.0–3.29.0**; a **`min-ase` CI job** pins `ase==3.23.0`
  (with `pip check` + exact-version assert) and RDF + teacher-MD regression tests guard both paths.

## 6. What is implemented and verified (software; network-free tests)

- Teacher-first stage order + `TeacherBaselineReport` — `workflow/controller.py` init;
  `validation/teacher_baseline.py:13`.
- Parent-aware leakage-resistant split — `workflow/steps.py:22` (`grouping_key=parent_structure_id`, `:34-40`).
- Four error channels — `validation/four_channel_audit.py:56` (teacher_vs_dft / student_vs_teacher / student_vs_dft).
- Committee σ_F + rank correlation + top-decile enrichment — `adapters/uncertainty.py:17,44,49`;
  `validation/committee_uncertainty.py:68,89`.
- Data-coverage report — `validation/data_coverage.py:63`.
- Three run-bound review lenses, controller-enforced — `workflow/review_lenses.py`;
  `workflow/controller.py:148-150,189,552,619`.
- Gate vote bundle validation + unanimous rule — `workflow/controller.py:537` (`_validate_vote_bundle`).
- Artifact SHA-256 + Git-revision binding — `workflow/integrity.py`; `controller.py:43,107,211-214,445`.
- Recovery: categories/agents/plan/approve/iteration/verify — `controller.py:21-25,264,687,780,795,825`;
  `downstream_invalidated` event `:326`.
- External (SLURM) stage completion registration — `controller.py:501`.
- Runtime-neutral typed task/result/vote exchange + accept-result (raw preservation) — `orchestration/exchange.py`.
- Optional PydanticAI-compatible runtime (typed output, restricted read_text/read_json,
  provenance, shadow) — `runtimes/pydantic_ai/*` (merged in PR #1),
  tests `tests/test_pydantic_ai_runtime.py`.

## 7. Implementation vs. scientific demonstration (three-way split)

These three categories are kept strictly separate; none implies another.

**A. Implemented and software-verified** (code + network-free tests): controller, typed
contracts, stage ordering, artifact hashing, run-bound gate criteria, three review lenses,
REVISE/FAIL routing, RecoveryPlan, human approval, iteration history, RecoveryExecutionReport,
changed-artifact verification, PydanticAI-compatible optional runtime, restricted read tools.

**B. Scientifically demonstrated on the existing SiO2 case**: teacher→student distillation,
force fidelity (teacher–DFT / student–teacher / student–DFT), committee-uncertainty ranking,
amorphous structure + dynamics validation (RDF/ADF/MD), throughput gain, and error/REVISE
detection during the audit.

**C. In code but NOT yet demonstrated by a real materials cycle**: an executed recovery
iteration (relabel → student retrain → same-profile revalidation → before/after metrics), a
second material/model benchmark end-to-end, and a controlled single-agent/manual comparison.

Explicitly rejected (invalid) inferences:
- "Recovery code exists, therefore closed-loop was demonstrated." — FALSE (no executed cycle).
- "PydanticAI runtime exists, therefore an external Claude provider was validated." — FALSE.
- "A second-system config exists, therefore transferability was demonstrated." — FALSE
  (no second system is even present in `examples/`).

## 8. Prepared but NOT externally executed

- Real external-provider PydanticAI call — harness `examples/pydantic_ai_provider_smoke.py`
  (PR #2, Draft, committed). Classification: **PREPARED AND NETWORK-FREE TESTED, NOT EXTERNALLY
  EXECUTED**. Proven: manual smoke entry point, credential/model preflight, no-call exit when
  credentials are absent, network-free import/preflight tests, CI makes no API call. Not proven:
  a real Anthropic call, a real model's `read_json` selection, provider token usage, provider
  retry/429/timeout, external structured-output fidelity.
- Fresh SiO2 pilot teacher_baseline — dispatched to GPU SLURM earlier this session; the GPU
  resource was not available during the audit session, so the job remained queued and no
  completed manifest was returned. The precise scheduling cause is not asserted here pending a
  scheduler-record check.

## 9. Partial

- Application-specific validation extensibility — three distinct states:
  - **callable extension mechanism: IMPLEMENTED.** `workflow/contracts.py:79-100`
    (`validate_validation_manifest`) dispatches an external validation manifest to a
    config-selected, hash-bound **dotted callable** (`importlib.import_module` + `getattr`),
    with built-ins under `validation/` and external adapters permitted in their own package;
    the controller enforces that the validator is a dotted callable (`controller.py:175`).
    Built-in validators already reachable this way: `structure_dynamics`, `surface_energy`,
    `committee_uncertainty`, `four_channel_audit`, `teacher_baseline`, `data_coverage`.
  - **formal registry/discovery/packaging API: NOT IMPLEMENTED** (no entry-point registry, no
    auto-discovery, no packaged plugin distribution — a config must name the dotted path).
  - **second-domain scientific execution: NOT DEMONSTRATED** (no non-SiO2 validator has been
    run on a real second material).
- Manuscript sync — content exists but predates the framework changes (see §14).

## 10. Out of scope (separate project)

Catalyst literature mining / causal inference (Yan et al., TheMeCat, Xu et al.): **NOT AVAILABLE
TO THE LOCAL REPOSITORY AUDIT SESSION** and, by nature, a **SEPARATE PROJECT SCOPE**
(PDF/OCR extraction, composition/GHSV harmonization, DOI fixed effects, DML/AIPW, MNAR
sensitivity). The absence of these functions from the MLIP repository does **not** make
Framework V1 incomplete — they belong to a separate literature-mining and causal-analysis
workflow. See §7-C for the three-way relationship classification.

## 11. Source-to-implementation traceability matrix

Status codes: IV=implemented+verified (software), PD=partially demonstrated on SiO2,
PNE=prepared, not externally executed, PARTIAL, ND=not demonstrated, OOS=out of scope.
Sci-demo: Y=demonstrated on real SiO2, P=partial, N=not yet.

| Requirement | Source/rationale | Implementation | Evidence | SW status | Sci-demo | Remaining |
|---|---|---|---|---|---|---|
| Teacher baseline before Student eval | 회의록 2.1 | teacher_baseline stage first | teacher_baseline.py:13 | IV | P | run on fresh teacher |
| Validation target/profile | 회의록 2.1 | validation_profile | examples/sio2.../validation_profile.yaml | IV | P | re-anchor to measured teacher |
| Teacher training-data access level | 회의록 2.2 | DataCoverageReport access mode | data_coverage.py:63 | IV | N | real coverage run |
| Teacher distribution coverage | 회의록 2.2 | data_coverage | data_coverage.py | IV | N | real run |
| Replay-data policy | 회의록 2.3 | replay_policy field | data_coverage.py | IV | N | exercise a replay run |
| Dataset source & lineage | 회의록 2.2 | parent_structure_id | steps.py:22-40 | IV | P | fresh dataset lineage |
| Parent-aware split | design | split_dataset grouping | steps.py:22 | IV | P | fresh dataset |
| Duplicate/leakage prevention | design | group split + accept-result dedup | steps.py:34-40; exchange.py | IV | P | fresh dataset |
| Teacher labeling provenance | 회의록 2.2 | label_with_teacher manifest | adapters/acquisition.py | IV | P | fresh labeling |
| Four Student seeds/committee | manuscript | n_seeds:4 | student.simple-nn.yaml:23-24 | IV | Y | — (done on SiO2) |
| Teacher–DFT channel | manuscript | channel() | four_channel_audit.py:56 | IV | Y | — (done on SiO2) |
| Student–Teacher channel | manuscript | channel() | four_channel_audit.py:56 | IV | Y | — (done on SiO2) |
| Student–DFT channel | manuscript | channel() | four_channel_audit.py:56 | IV | Y | — (done on SiO2) |
| Student-MD–DFT deployment channel | manuscript | channel() (d) | four_channel_audit.py | PARTIAL | P | carved-frame DFT |
| Committee disagreement (σ_F) | manuscript | committee_force_std | adapters/uncertainty.py:17 | IV | Y | — (done on SiO2) |
| Callable validator extension mechanism | 회의록 2.5/2.8 | dotted-callable dispatch | contracts.py:79-100; controller.py:175 | IV | P | — (mechanism done) |
| Formal validator registry/discovery/packaging API | 회의록 2.5/2.8 | — (config names dotted path) | no entry-point registry | ND | N | build registry API |
| Second-domain validator scientific execution | 회의록 2.9 | built-ins present | validation/* | IV | N | run on real 2nd material |
| Required-pass observables | 회의록 2.5 | validation_profile checks | validation_profile.yaml | IV | P | fresh physical_validation |
| Three independent Judge contexts | manuscript | judge x3 blind | agents/judge.md; gate_vote.workflow.js | IV | P | fresh gate |
| Run-bound gate criteria | design | gate_criteria bound at init | controller.py:137 | IV | N | fresh run |
| Review-lens binding | advisor review | 3 lenses enforced | review_lenses.py; controller.py:552 | IV | N | fresh gate; error-detection study |
| Artifact hashing | design | artifact_digest | integrity.py; controller.py:445 | IV | N | fresh artifacts |
| Input & Git-revision binding | design | code_revision | controller.py:211-214 | IV | N | fresh run |
| PASS/REVISE/FAIL transitions | 회의록 2.4 | record_gate | controller.py:622 | IV | P | executed REVISE cycle |
| Failure-category routing | 회의록 2.4 | RECOVERY_CATEGORIES/AGENTS | controller.py:21-25 | IV | N | executed routing |
| RecoveryPlan | 회의록 2.4 | propose_recovery | controller.py:687 | IV | N | executed plan |
| Human approval | 회의록 2.4 | approve_recovery + boundaries | controller.py:780; orchestrator.md | IV | P | executed approval events |
| Return-stage invalidation | 회의록 2.4 | downstream_invalidated | controller.py:326 | IV | N | executed cycle |
| Teacher relabel declaration | 회의록 2.4 | RecoveryPlan fields | controller.py:687-757 | IV | N | executed relabel |
| New DFT declaration | 회의록 2.7 | RecoveryPlan/reference config | controller.py; reference_dft.yaml | IV | N | executed DFT |
| Student retraining declaration | 회의록 2.4 | RecoveryPlan | controller.py | IV | N | executed retrain |
| Same-profile revalidation | 회의록 2.4 | verify_recovery_execution | controller.py:825 | IV | N | executed revalidate |
| RecoveryExecutionReport | 회의록 2.4 | verify_recovery_execution | controller.py:825-846 | IV | N | executed report |
| Changed-artifact verification | 회의록 2.4 | hash compare in verify | controller.py:845-952 | IV | N | executed iteration |
| Iteration history | 회의록 2.4 | iterations in state | controller.py:215 | IV | N | executed iterations |
| ASE dependency-range compatibility | this session | RDF atomic-number selector + teacher-MD FixCom-omission policy | merged PR #3 (`45d2dab`); floor `ase>=3.23`; `min-ase` CI job; RDF + teacher-MD regression tests; verified ASE 3.23.0–3.29.0 | IV | N/A | — (resolved) |
| Actual SiO2 demonstration (distillation+validation) | manuscript | full pipeline on SiO2 | EXTERNAL SCIENTIFIC EVIDENCE — exact sibling-repository paths pending | IV | Y | recovery cycle (below) |
| Actual SiO2 recovery cycle | 회의록 2.4 | code present; not executed | NO EXECUTED-CYCLE EVIDENCE IN REPO | IV | N | GPU queue + trigger |
| Second material/model benchmark | 회의록 2.9 | — | examples = SiO2 variants only | ND | N | pick + run 2nd system |
| Manual/fixed/single-agent comparison | 연구동향 10.1; advisor | — | NO IMPLEMENTATION EVIDENCE | ND | N | design matched baseline |
| Shadow runtime | this session | run_task(shadow=True) | runtimes/pydantic_ai/driver.py | IV | N | provider shadow |
| PydanticAI typed task/result/vote | 연구동향 6.1 | models + exchange | runtimes/pydantic_ai/models.py; orchestration/exchange.py | IV | N | provider run |
| Existing validator as final authority | design | validate_agent_response gate | driver.py:55 | IV | N | provider run |
| Restricted read_text/read_json | 연구동향 5.2/9 | ReadOnlyToolset | tool_registry.py | IV | N | provider run |
| Runtime provenance | 연구동향 9 | RuntimeInvocationRecord | models.py | IV | N | provider run |
| Optional provider harness | 연구동향 6 | smoke script | examples/pydantic_ai_provider_smoke.py | PNE | N | real key + call |
| Actual external Anthropic provider execution | 연구동향 6 | harness only | — | PNE | N | API key + one call |
| Manuscript synchronization | 회의록 3.x | — | sibling paper repo, pre-dates changes | PARTIAL | N | sync after cycle |
| Figure 1 synchronization | 회의록 3.3 | — | sibling paper repo | PARTIAL | N | update recovery arrows |
| Public reproducibility package (DOI) | 회의록 Outlook | repo public; no DOI | README Data Availability stale | PARTIAL | N | DOI + manifests |

## 12. Framework completion verdict

**FRAMEWORK V1: COMPLETE.** The completion basis is the code merged to `origin/main` — the
existing controller, contracts, validators, three-lens gate, recovery, audit, and
runtime-neutral exchange, plus the optional typed runtime merged via **PR #1** — all passing
network-free tests in a clean, declared-floor–compliant venv (full suite 157 tests,
0 fail/0 err/33 skip; §5) and GitHub CI green. The dependency-range ASE compatibility is now
**verified across ASE 3.23.0–3.29.0** via merged **PR #3** (`45d2dab`), with the floor raised to
`ase>=3.23` and a `min-ase` CI job (§5a). **PR #2 (the optional Anthropic provider smoke harness)
is DRAFT and unmerged and is NOT part of the Framework V1 completion basis**; it is an optional,
external-only add-on (see §8). This is a software verdict only.

## 13. Scientific demonstration verdict

**SCIENTIFIC DEMONSTRATION: PARTIAL.** The SiO2 case already demonstrates distillation, a
4-seed committee, the three force channels, committee-uncertainty ranking, structure/dynamics
validation, throughput gain, and audit-time error detection (§7-B). Not yet demonstrated: an
executed recovery iteration, a second system, and a controlled single-agent/manual comparison
(§7-C). The fresh pilot's recovery cycle stalled in the SLURM queue.

## 14. Manuscript/reproducibility verdict

**MANUSCRIPT AND REPRODUCIBILITY: PARTIAL.** The manuscript exists in the sibling paper repo
but does not yet describe the typed agent-exchange contracts, the PydanticAI-compatible runtime,
or the as-merged review-lens differentiation; its four-channel/σ_F/validation-gated content is
from the earlier revision. Data Availability still says the package "will be deposited"; no DOI.
Claims can be strengthened only within the demonstrated scope (§13).

## 15. Claims currently supportable (with evidence)

- An auditable MLIP-distillation workflow was implemented (§6).
- On SiO2, teacher→student distillation with a 4-seed committee and three force channels,
  committee-uncertainty ranking, structure/dynamics validation, and throughput gain were
  demonstrated (§7-B).
- Artifacts and decisions are hash- and Git-revision-bound (integrity.py; controller.py:211).
- Three separate-context judge gates with run-bound criteria and enforced review lenses
  (controller.py:537,552).
- Failure-category recovery routing and human-approval boundaries are implemented
  (controller.py:21,687,780).
- An optional, provider-neutral PydanticAI-compatible runtime is implemented with a read-only
  tool boundary and provenance (runtimes/pydantic_ai/*), verified network-free.

## 16. Claims currently NOT supportable (hold)

- A completed autonomous closed-loop recovery cycle (no executed iteration).
- Superiority over manual/single-agent/fixed-pipeline workflows (no matched comparison).
- General transferability across materials/architectures (only SiO2 present).
- Actual Anthropic/external provider validation (no live call).
- Automatic selection of all validation modules (partial; no plugin API).
- That all source papers' workflows are implemented (catalyst mining/causal = separate scope).

## 17. Single recommended next scientific action

Complete **one** real SiO2 recovery iteration end-to-end (gate REVISE/FAIL → approved
RecoveryPlan → data change → teacher relabel if declared → 4-seed student retrain →
same-profile revalidation → verify-recovery with changed-artifact hashes → before/after metric
table). Preferred initial trigger candidate: `student_fidelity` or `dataset_coverage`; the
final trigger is chosen only after inspecting the authoritative SiO2 run and its gate evidence.
This is the single artifact that upgrades SCIENTIFIC DEMONSTRATION from PARTIAL toward COMPLETE.
Detailed ordering (no compute commands) is in `next-scientific-execution-plan-2026-08-03.md`.

## 18. Glossary (consistent terms used above)

- **Workflow** — the whole research flow (teacher baseline → data → distillation → validation →
  gate → recovery).
- **Validation profile** — the per-application set of observables, references, and thresholds.
- **Validation module** — an individual observable validator.
- **Gate** — the evidence-evaluation decision (PASS/REVISE/FAIL) governing whether to proceed.
- **Audit trail** — the recorded inputs, outputs, hashes, verdicts, and iterations.
- **Recovery** — the approved correction + re-execution after a REVISE/FAIL.
- **Scientific demonstration** — scope proven by real materials calculations and metrics.
- **Software implementation** — functionality confirmed by code and tests.

`closed-loop`, `active learning`, and `autonomous` are used only where real executed evidence
exists — which, for the recovery cycle, it does not yet.
