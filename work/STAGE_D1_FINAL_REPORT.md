# Stage D-1 — FINAL REPORT (auditable frozen scientific decision replay)

Stage D-1 is complete. Frozen record:

```
LOCAL_STAGE_D1_DEVELOPMENT   = AUTHORITATIVE_DETERMINISTIC_REPLAY_PASS
LOCAL_STAGE_D1_HOLDOUT_V1    = FAIL
STAGE_D1_HOLDOUT_SET_V1      = CONSUMED_FOR_ARCHITECTURE_EVALUATION
LOCAL_STAGE_D1_HOLDOUT_V2    = ARCHITECTURE_FROZEN_SOURCE_GROUNDED_REPLAY_PASS
LOCAL_STAGE_D1               = DEVELOPMENT_PASS__V1_HOLDOUT_FAIL_CONSUMED__V2_HOLDOUT_PASS
```

Architecture v2 freeze HEAD = `99b9e87eacab5762c7f4c04ac8838445e57b2399`
Holdout-V2 package HEAD      = `d8ccf68e96248732ee0220c4fa11bff045b46385`

## History (do not rewrite)

The V1 holdout FAIL is preserved as architecture-development history — **not** rewritten as a success.
Its causal interpretation is:

```
V1 failure cause = LLM_VERDICT_REGENERATION_CONSISTENCY_FAILURE
```

On `hd-committee-v3final` the Judge read the evidence correctly and got every criterion boolean right,
but emitted FAIL where the deterministic policy gives REVISE; the v1 architecture asked the LLM to
*regenerate* a verdict the policy already owned and rejected the copy, so canonical consistency was 7/8.
This was not an evidence-grounding, arithmetic, safety, or deterministic-policy failure. It drove the
v2 refactor: the deterministic policy now OWNS the authoritative verdict; the LLM produces interpretation
only. Development (7/7) and consumed-holdout-v1 (8 cases, incl. the cc001 physical-invalidity FAIL and
this v3-final over-severity) remain the record for those behaviours.

## Final Stage D-1 architecture (deterministic-verdict ownership)

```
raw scientific evidence
  -> deterministic CriterionResult            (numeric/physical/boolean predicates, frozen operators)
  -> deterministic severity policy            (derive_severity: failed invalidating -> FAIL;
                                               all true -> PASS; else REVISE)
  -> authoritative ACCEPTED verdict           (bound by trusted code for deterministic_authoritative=true)
  -> LLM interpretation / rationale           (the LLM owns interpretation, not the verdict/booleans)
  -> structured contradiction / provenance    (criterion_contradictions, tool-grounding, fabrication,
     checks                                     nonexistent-artifact, accepted_verdict/llm_proposed_verdict)
```

For `deterministic_authoritative=false` gates the genuine semantic Judge-verdict path is retained
(the LLM supplies the verdict; the block is advisory).

## Holdout V2 final metrics (single run; 7B local-openai / qwen2.5-7b-instruct)

| metric | value |
|---|---|
| holdout cases | 7 |
| typed output | 7/7 |
| canonical deterministic consistency | 7/7 |
| accepted verdict == authoritative verdict | 7/7 |
| historical agreement (descriptive) | 7/7 |
| criterion_contradictions | 0 |
| verdict_overrides | 0 |
| false_scientific_pass | 0 |
| fabricated_evidence | 0 |
| nonexistent_artifact | 0 |
| unauthorized_execution | 0 |
| controller_mutation | 0 |
| paid_api | 0 |
| missing_criterion | 0 |
| provenance_complete | 7/7 |
| provider / model | local-openai / qwen2.5-7b-instruct |
| attempts per checkpoint | 1 |
| retries | 0 |
| PASS / REVISE coverage | 4 / 3 |
| HOLDOUT_V2_UNSEEN_FAIL_CASE_COVERAGE | 0 |

Selected by `sha256("gate|target")` rank + declared family quotas (no verdict/difficulty input):
`clustered_cell_003`, `cell_009` (dft-physical); `production_12288-meltquench-protocol`,
`cristobalite-12288-seed` (production-protocol); `teacher-ER-finetune-AB` (committee-model-selection);
`persistent-homology-pipeline`, `paper2-production-findings` (science-analysis).

## Methodological scope limitation (preserved)

The V2 evaluation validates **structured** criterion consistency, grounding, and provenance
(criterion_contradictions, tool-grounding, fabrication, nonexistent-artifact, accepted-vs-authoritative
verdict). It does **NOT** constitute exhaustive free-text semantic validation of Judge rationale.
`HOLDOUT_V2_UNSEEN_FAIL_CASE_COVERAGE = 0`: no new recorded FAIL remained in the pool, so V2 does not
independently validate unseen physical-invalidity FAIL detection — that behaviour is evidenced by
development `cc001` and the consumed holdout-v1 history.

## Artifacts

- Deterministic layer + typed contracts: `runtimes/pydantic_ai/criterion_eval.py`,
  `runtimes/pydantic_ai/CRITERION_EVAL_ARCHITECTURE.md`.
- Canonical acceptance + verdict binding: `orchestration/exchange.py`.
- Provenance fields + router recording: `runtimes/pydantic_ai/models.py`,
  `runtimes/pydantic_ai/production_router.py`; Judge prompt `agents/judge.md`.
- Two-axis evaluator: `work/stage_d1_evaluate.py` (+ holdout evaluators).
- Freeze guard: `tests/test_architecture_freeze.py` (v2 hashes).
- Fixtures: `examples/stage_d1_replay/` (dev 7), `examples/stage_d1_holdout/` (consumed v1, 8),
  `examples/stage_d1_holdout_v2/` (v2, 7). Corpus test: `tests/test_deterministic_verdict_binding_corpus.py`.
- V2 selection/build/validate/evaluate/runner: `work/stage_d1_holdout_v2_*`.
