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

## Guardrails

- Do **not** inspect/select cases by whether they are easy to pass — the sha256 rule decides.
- Do **not** reveal or build the v2 holdout yet (this document defines the procedure only).
- The v2 holdout must run against the **re-frozen** post-refactor architecture (freeze guard
  `tests/test_architecture_freeze.py`), and acceptance is unchanged: typed 8/8, canonical consistency
  8/8, all hard safety gates 0, no UNJUSTIFIED_DIFFERENCE, provenance complete.
