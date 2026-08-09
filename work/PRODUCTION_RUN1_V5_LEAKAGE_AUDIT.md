# Production Run 1 · R1.P1R — v5 Held-out Dataset-Leakage Audit (read-only; no compute)

Resolves *why* R1.P1 returned **LEAKAGE = REVISE**. Read-only data-lineage action: **no** teacher/student
inference, DFT, MD, training, scheduler, network, or semantic Judge. Parent R1.P1
(`prod-run1-v5-heldout-generalization`, HEAD `f7f5c65e…`) is **immutable and unchanged**. New append-only
run: `runs/production_run1/prod-run1-v5-leakage-resolution/`. Conclusions tagged **FACT / DERIVED /
INFERENCE / UNRESOLVED** with source paths. `RES = research-sio2-allegro-simplenn-distillation/`.

## Question

Are the six R1.P1-evaluated held-out cells (`cell_ho_02 … cell_ho_07`) demonstrably independent of every
structure used to **train or augment** the adopted v5 SIMPLE-NN committee — proven from data lineage, not
inferred from filenames or `held_out_confirmed=True`?

## Leakage levels (matching procedure declared before results)

| Level | Meaning | Check |
|---|---|---|
| **L0** | exact file duplicate | identical bytes (sha256) |
| **L1** | exact structure equivalent | same composition + box + wrapped-fractional-coord multiset (order-independent, PBC-wrapped, coords→1e-3 frac) |
| **L2** | same source frame / derivative | carved from a parent trajectory frame that itself entered v5 training |
| **L3** | near-duplicate | descriptive only; no invented hard cutoff |
| **L4** | same distribution only | not leakage |

## A. v5 training-data lineage recovered — **FACT** unless noted

- v5 committee = 4 seeds sharing **one data pool**; only `random_seed` differs (234/345/555/777),
  `shuffle_dataloader: True` (`gpu_return_v5_committee/v5_committee_bundle/seed0{1..4}/input.yaml`).
- v5 augmentation = augment-atoms **~10,000** frames (8k normal + 2k large), Allegro(T1)-labelled,
  **0 DFT anchors**; **seeds drawn from the KISTI DFT corpus** (`teacher/input_small.xyz` 50 +
  `input_large.xyz` 10) — **not** from `production_12288` MD dumps (`data_provenance/PROVENANCE.md §1,§3`).
- v5 is a **"T1 defect re-augment"** of that base (`sio2x_production/committee_u_out/v5_seed01_errorc/
  eval_errorc_v5_seed01.py`). **UNRESOLVED:** the v5-specific defect-re-augment frame set has **no local
  manifest/XYZ**.
- **INFERENCE:** the 4-seed committee shares a single pool, so leakage is a per-pool (not per-member)
  question.

## B. Complete v5 frame list exists locally? — **NO (FACT)**

`train_list`/`valid_list`/`test_list`/`ref_list` are referenced (relative, `absolute_path:False`) by the
v5 `input.yaml` but **no such files exist anywhere in RES** (repo-wide search: 0 hits). They are
KISTI-origin (canonical non-local GPU dir `…/05_DISTILLATION_DIRECT`; `gpu_finetune_handoff/distillation/
DISTILLATION_RECIPE.md`, `PROVENANCE.md §3,§6`). **This is the exact missing artifact.**

## C. Evaluated held-out lineage — **FACT** (`manifest_heldout.csv`, `input.data`)

| cell | dist | x | center CN | n_at | comp (O/Si) | box Å | parent dump # frame | input.data sha256₁₆ |
|---|---|---|---|---|---|---|---|---|
| ho_02 | random | x006 | 3 | 67 | 45/22 | 11.0 | random_sweep/x006 #20 | bd0edeb8b5fb3b4d |
| ho_03 | random | x015 | 1 | 73 | 48/25 | 11.0 | random_sweep/x015 #20 | eb85cbebf45f1f63 |
| ho_04 | clustered | sphere T1000 | 0 | 49 | 25/24 | 11.0 | anneal_calib_clustered/T1000 #20 | 2af12376ee66d31a |
| ho_05 | clustered | sphere T1000 | 1 | 49 | 22/27 | 11.0 | anneal_calib_clustered/T1000 #30 | 62fa7d527c0adff9 |
| ho_06 | clustered | plane T1000 | 0 | 63 | 35/28 | 11.0 | anneal_calib_clustered/plane_T1000 #20 | 946ab8395974e617 |
| ho_07 | clustered | plane T1000 | 1 | 52 | 28/24 | 11.0 | anneal_calib_clustered/plane_T1000 #30 | 9fbcafab0452660a |

Each is a carved ~11 Å cell + inner-sphere-frozen anneal (`out.data`); atom order changed vs parent.

## D. Exact duplicates (L0) — **NONE (FACT)**

No held-out `input.data` sha256 equals any local DFT-labelled AL cell (`dft_labeling/cell_*`,
`clustered_cell_*`). All six held-out SHAs are distinct.

## E. Periodic/order-equivalent duplicates (L1) — **NONE (FACT)**

Canonical composition+box+wrapped-frac-coord fingerprint of each held-out cell matches **no** local
candidate (11 AL11 frames + the DFT-labelled AL cells) — not even by composition+box alone
(`membership_matrix.csv`).

## F. Shared-parent / derived overlaps (L2) — **UNRESOLVED (FACT of absence)**

Held-out parents are `production_12288/random_sweep/{x006,x015}` and
`anneal_calib_clustered/{T1000,plane_T1000}` (frames 20/30) — **all present locally**. No local artifact
shows any of these frames entering v5 **training**; these dumps are referenced only for v5 **evaluation**
(`sio2x_production/committee_u_out/v5_random_sweep/…`). But the v5 defect-re-augment frame manifest is
non-local, so frame-level non-overlap **cannot be proven**. → **L2 UNRESOLVED.**

## G. Near-duplicate evidence (L3) — **NONE observed; no cutoff invented**

No local candidate shares even composition+box with a held-out cell, so no near-duplicate assessment was
needed; no similarity threshold was invented (no project tolerance applies here).

## H. AL-data overlap — **DERIVED**

- `AL11_with_umax_v5.xyz` is **v5 EVALUATION output** (the 11 AL cells scored by the v5 committee to append
  `uncertainty_u_alpha`), **not** v5 training data (`committee_umax_AL11_v5.log`) — corrects the naive
  "AL cells are in the bundle ⇒ training" reading.
- Whether the **11 AL cells** are in v5 *training* = **UNRESOLVED** (project script verbatim: *"may or may
  not be in the v5 re-augment training set"*). The **28 al_iter3** DFT cells postdate v5 and feed **v6**
  (`HANDOFF_v6.md`) → **not** in v5 training.
- **Crucially:** the 6 held-out cells are **structurally distinct** from the 11 AL cells (L0/L1 NONE), so
  the R1.P1 "11-AL-may-overlap" caveat pertains to **deployment error(c)**, **not** to the held-out set.

## I. Final LEAKAGE verdict — **REVISE**

- **L0 = NONE, L1 = NONE** (proven locally); **L2 = UNRESOLVED**; held-out ≠ 11 AL cells.
- **Strong circumstantial independence** (INFERENCE): v5 augmentation seeds are DFT-corpus, not production
  dumps; and v5 **fails** u_max on exactly the held-out parent regimes while **v6 adds those dumps as the
  OOD-remediation fix** — if they were already in v5 training they would not be the new remediation set.
- **But** the v5 defect-re-augment frame manifest is non-local ⇒ frame-level independence is **not proven**.
  Per the rule, **lack of evidence is not turned into PASS ⇒ REVISE.** ORIGINAL_VS_V5 stays **UNRESOLVED**.

## J. Original-student held-out predictions discovered? — **MISSING**

No `*orig*` prediction artifact exists under `heldout_dft_batch/analysis/` (`run_heldout_baseline_orig.py`
itself targets the v5 bundle). Not generated here.

## K. Exact blocker (verdict REVISE)

The **KISTI-origin v5 `train_list`/`valid_list` + the augmented XYZ — specifically the v5 T1
defect-re-augment frame manifest**. Recovering that single artifact set converts L2 to a provable
PASS/FAIL and lets the leakage verdict close.

## L. Recommended NEXT action (NOT executed)

**Retrieve the v5 training/augmentation frame manifest** (the KISTI `05_DISTILLATION_DIRECT` train/valid
lists + augmented XYZ) as a read-only provenance-recovery step, then re-run R1.P1R's L2 check. This is a
**data-transfer/provenance** action, not compute — it needs a human to fetch the KISTI artifact (network
to KISTI is out of scope here). Only after LEAKAGE resolves to PASS should ORIGINAL_VS_V5 be addressed
(and only then, if original held-out predictions remain MISSING, propose — not execute — a tightly bounded
original-student held-out inference for exactly cell_ho_02…07). **Do not** proceed to expensive teacher
DFT-anchored fine-tuning while this basic provenance question is open, unless justified independently of
the held-out claim.

## Status

- Parent R1.P1 unchanged (immutable); R1.P1R is a fresh append-only run.
- `LEAKAGE = REVISE`; `ORIGINAL_VS_V5 = UNRESOLVED` (unchanged). No historical record rewritten.
- No scientific model execution occurred (read-only CPU lineage audit).
