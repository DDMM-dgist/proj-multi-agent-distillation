# Stage D-1 holdout v2 — dataset status + deterministic selection procedure (NOT executed)

## Dataset status

```
STAGE_D1_HOLDOUT_SET_V1 = CONSUMED_FOR_ARCHITECTURE_EVALUATION
```

The 8-case v1 holdout (`examples/stage_d1_holdout/`, `hd-*`) has now been observed and was used to
evaluate the architecture (it produced the LLM_VERDICT_REGENERATION_CONSISTENCY_FAILURE that drove the
deterministic-verdict-ownership refactor). It must **never again** be described as an unseen holdout.
It may be reused only as a **POST_HOLDOUT_REGRESSION_SET**. Together with the original 7 development
cases (`examples/stage_d1_replay/`, `d1-*`) it forms a **15-case regression/development corpus** (see
`tests/test_deterministic_verdict_binding_corpus.py`).

## Remaining candidate pool for a NEW (v2) holdout

Real auditable decisions in `coordination_log.csv` / `gates/coordination_votes.csv` NOT used by the 7
development cases or the 8 consumed-holdout cases. Verdicts are listed here only as pool metadata for
stratification; the **selection is by a mechanical rule, not by picking outcomes**, and fixtures are
NOT built here.

DFT-label / physical gates (all PASS):
`cell_009`, `clustered_cell_000`, `clustered_cell_003`, `clustered_cell_006`, `clustered_cell_007`,
`clustered_cell_008`.

Committee / model-selection, production/protocol, provenance/reporting, science/analysis gates:
`production-cell-12288-cubic` (first vote, REVISE), `cristobalite-12288-seed` (PASS),
`er-finetune-gate` (REVISE), `ph-pipeline-gate` (PASS), `production-science-gate` / paper2 (REVISE),
`meltquench-protocol-gate` (REVISE).

Note: **no FAIL decision remains** in the pool — the only historical FAIL (cc001) is already in the
development set. "Include FAIL where available" is therefore unsatisfiable for v2 and will be recorded
as such rather than manufactured.

## Deterministic selection procedure (predefined; run only AFTER the v2 architecture is frozen and approved)

1. **Freeze the pool** from the two coordination CSVs at the current research-repo commit; record each
   candidate as `gate|target` with its recorded verdict and gate family.
2. **Stratify by gate family**: {dft-physical, committee/model-selection, production/protocol,
   provenance-reporting, science-analysis}.
3. **Coverage rule**: include both PASS and REVISE strata; include FAIL **iff** one exists in the pool
   (it does not for v2 — record `FAIL_UNAVAILABLE`). Do not select on ease of passing.
4. **Deterministic pick**: within each stratum, order candidates by `sha256("<gate>|<target>")` (hex
   ascending) and take the first `k` (k set by target size). This is reproducible and independent of
   the verdict — no cherry-picking.
5. **Target size**: 6–8 cases, balanced across strata subject to availability.
6. **Freeze the selection** (write the chosen `gate|target` list + their sha256 keys to a sealed file)
   **before** any fixture construction; keep the historical verdicts sealed (separate file, never in
   evidence), exactly as v1.
7. Only then build metrics-only evidence + generic criterion specs (frozen operators only), validate
   network-free, record deterministic predictions before inference, and run once.

## EXECUTED (2026-08-09) — selection frozen, package built, NOT run

Architecture v2 freeze HEAD = `99b9e87eacab5762c7f4c04ac8838445e57b2399` (freeze guard green).
`work/stage_d1_holdout_v2_select.py` was run; the 7-case selection is frozen in
`examples/stage_d1_holdout_v2/SELECTION_MANIFEST.json` (selection = `sha256("gate|target")` rank +
declared family quotas dft2/production2/committee1/science2; **no verdict/difficulty input**):

| target | gate family | rank | sha256[:16] | historical (post-freeze) | det. prediction |
|---|---|---|---|---|---|
| clustered_cell_003 | dft-physical | 0 | 1d1680674dd06f49 | PASS | PASS |
| cell_009 | dft-physical | 1 | 4ec6181a4ebb4983 | PASS | PASS |
| production_12288-meltquench-protocol | production-protocol | 0 | 6ff30384f3541664 | REVISE | REVISE |
| cristobalite-12288-seed | production-protocol | 1 | 7915e89b271f5d77 | PASS | PASS |
| teacher-ER-finetune-AB | committee-model-selection | 0 | 03ae2a5ae4e31057 | REVISE | REVISE |
| persistent-homology-pipeline | science-analysis | 0 | 5f8cf20914c74fef | PASS | PASS |
| paper2-production-findings | science-analysis | 1 | b9d9376e7704a61f | REVISE | REVISE |

- `HOLDOUT_V2_UNSEEN_FAIL_CASE_COVERAGE = 0` (no FAIL remains in the pool; not manufactured). Holdout V2
  contains PASS + REVISE families only. Physical-invalidity FAIL detection remains supported by dev cc001
  + consumed-holdout-v1 history; V2 does not independently validate unseen FAIL detection.
- `HOLDOUT_V2_REQUIRES_ARCHITECTURE_CHANGE = false` — every criterion uses only the frozen operators
  (`in_range`/`le`/`ge`/`eq`/`approx`/`exists`/`all`), the frozen severity policy, the unchanged Judge
  prompt, and no runtime change (freeze guard green). The evaluator gained an interpretation-layer
  (axis B) `criterion_contradictions` hard gate — an evaluator packaging rule for the new architecture,
  not a change to scientific decision semantics.
- Deterministic predictions (`DETERMINISTIC_PREDICTIONS.json`) recorded BEFORE inference; they match
  historical (descriptive only, not tuned).

## Guardrails

- Do **not** inspect/select cases by whether they are easy to pass — the sha256 rule decides.
- Do **not** reveal or build the v2 holdout yet (this document defines the procedure only).
- The v2 holdout must run against the **re-frozen** post-refactor architecture (freeze guard
  `tests/test_architecture_freeze.py`), and acceptance is unchanged: typed 8/8, canonical consistency
  8/8, all hard safety gates 0, no UNJUSTIFIED_DIFFERENCE, provenance complete.
