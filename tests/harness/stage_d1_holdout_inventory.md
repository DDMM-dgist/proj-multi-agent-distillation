# Stage D-1 — HOLDOUT decision inventory (untouched; verdicts SEALED)

Purpose: the 7 checkpoints under `tests/fixtures/stage_d1_replay/` are now a **development set** — the
deterministic criterion layer, its specs, and the evaluator were built and observed against them, so
their pass on the frozen replay no longer certifies the architecture generalizes. This file inventories
a **separate holdout** of real auditable decisions to be replayed **only after the architecture is
frozen**, as the independent acceptance gate.

## Hard rules for this holdout

- **Verdicts are SEALED.** The historical decision (PASS/REVISE/FAIL) for every candidate below is
  **deliberately NOT reproduced in this file, in `examples/`, or in any `work/` script or test.** The
  verdicts live only in the research repo's `coordination_log.csv`
  (`../research-sio2-allegro-simplenn-distillation/coordination_log.csv`), which the runtime already
  treats as *reference, never read by an agent*. Do not copy a holdout verdict into the runtime tree
  until the holdout package is constructed, and even then the golden verdicts must be committed to a
  file the development logic has never seen.
- **Not built yet, not run yet.** This is an inventory only. Do **not** create holdout evidence JSONs,
  criterion specs, tasks, or golden files, and do **not** run inference on the holdout, until the user
  approves the frozen architecture. Building them now would re-observe them and destroy the holdout.
- **No spec authored against a holdout case.** Holdout criterion specs, when eventually built, must be
  produced only by the *already-frozen* generic generator (`tests/harness/stage_d1_gen_criteria.py` predicate
  families), never hand-tuned to a holdout decision.
- **Final acceptance = development AND holdout.** Acceptance requires the development replay AND the
  holdout replay to pass with all hard safety gates at zero.

## Development set already consumed (do NOT reuse as holdout)

`cell_001`, `clustered_cell_002`, `clustered_cell_001 (cc001)`, `v3-redistilled-committee-REJECT`,
`v5-committee-ADOPT`, `distillation-dataset-and-splits`, `student-aSiO2-structure-and-dynamics`.

## Holdout candidates (real auditable decisions; source pool = coordination_log.csv)

Each candidate is a real gate decision with BOTH machine-readable evidence AND an auditable recorded
verdict, disjoint from the development set. Chosen to stress the SAME predicate families the frozen
layer implements, plus one deliberate arithmetic-comparison stressor. Verdicts omitted by design.

| # | candidate decision (gate / target) | predicate family exercised | evidence source (research repo) |
|---|---|---|---|
| H1 | `dft-label-judge-gate / cell_016` | DFT physical validity: `E_per_atom in [-11,-8]`, `max_force le 50` (invalidating) | `teacher_diag/` DFT label OUTCAR/summary for cell_016 |
| H2 | `dft-label-judge-gate / cell_011` | same DFT physical validity (note: misleading INCAR comment, correct values — nuance) | `teacher_diag/` cell_011 |
| H3 | `judge-gate-clustered / clustered_cell_005` | same DFT physical validity (extreme O-deficient, still physical) | `teacher_diag/` clustered_cell_005 |
| H4 | `committee-reliability-gate / v3-final-committee-ADOPT` | committee regression: `u_max_mean_deploy le original`, `error_c le 0.368`, `F_RMSE le original` | `teacher_diag/` committee-error CSVs, STATUS.md |
| H5 | `committee-reliability-gate / v3-final-v2-committee-ADOPT` | same committee family (mixed improve/regress — nuance, max improves / mean regresses) | `teacher_diag/` committee CSVs |
| H6 | `production-sizing-gate / production-cell-12288-cubic-REVOTE` | **quantitative-comparison stressor**: a real human-judge *arithmetic* dispute (cooling-rate/step-count formula) — the exact error class the deterministic layer removes | `sio2x_production/` quenching.in + protocol notes |
| H7 | `error-decomposition-gate / 4-error-decomposition` | scope/labeling completeness (REVISE-class: values right, aggregation labels missing) | `teacher_diag/` error a/b/c CSVs |
| H8 | `committee-uncertainty-gate / vacancy-enrichment-and-AL-selection` | claim-vs-artifact traceability (REVISE-class: unsupported ratio, non-traceable chain) | `figs_al/` rationale + per-species ratio CSVs |

8 candidates: 3 DFT-physical (incl. one nuance), 2 committee-regression (incl. one mixed), 1 dedicated
arithmetic-comparison stressor, 2 completeness/traceability REVISE-class. Spans FAIL/REVISE/PASS-shaped
severities and both invalidating and non-invalidating predicate paths, without reusing any development
target.

## When the holdout is built (post-freeze, on approval)

1. For each candidate, extract metrics-only evidence JSON (no verdict, no prose conclusion) from the
   source above — same discipline as `tests/fixtures/stage_d1_replay/evidence/`.
2. Generate criterion specs **only** via the frozen generic predicate families in
   `tests/harness/stage_d1_gen_criteria.py`; author nothing per-case.
3. Commit the sealed golden verdicts to a file the development logic has never imported.
4. Run the frozen replay once; evaluate offline with `tests/harness/stage_d1_evaluate.py`
   (`--expected-provider local-openai --expected-model qwen2.5-7b-instruct`).
5. Acceptance = development replay pass AND holdout replay pass AND all hard safety gates zero AND no
   UNJUSTIFIED_DIFFERENCE.
