# Production Campaign 002 — Distillation Dataset Design (authoritative, one pass)

Independent, raw-data-first design of the distillation training distribution the **accepted teacher**
should supervise. **No teacher/student inference, DFT, MD, training, scheduler, network, or semantic
Judge; no student model inspected.** Reproducible: `work/production_campaign_002_dataset_design.py` →
`runs/production_campaign_002/pc002-distillation-dataset-design/`. Labels **FACT / DERIVED / INFERENCE /
UNRESOLVED**. `RES = research-sio2-allegro-simplenn-distillation/`.

## 1. PC001 handoff (FACT)

`TEACHER = base Allegro b56e20ff`, `TEACHER_ACCEPTED_FOR_DISTILLATION`; `DISTILLATION_DATASET_STAGE_
AUTHORIZED = true`; `NEW_PIPELINE_CURRENT_STUDENT = NONE`. PC001 is closed and not reopened.

## 2. Target deployment domain (frozen from PC001)

amorphous SiO₂; SiO₂₋ₓ dilute vacancy; **SiO₂₋ₓ clustered vacancy / void surface / under-coordinated Si**;
surfaces; high-T / liquid-distorted; ambient crystalline reference. (Not redefined to fit available data.)

## 3. Existing structure inventory (candidate pool — LOCAL, unlabeled production MD)  — **FACT**

| target domain | candidate source | size | labels |
|---|---|---|---|
| amorphous SiO₂ | `production_12288` 12288-atom melt-quench glass (rdf.dump) | 501 frames | UNLABELED |
| dilute vacancy | `production_12288/random_sweep` x0.03–0.24 (7 fractions) | **350 frames** | UNLABELED |
| clustered vacancy / void | `production_12288/anneal_calib_clustered` {T1000, plane_T1000} | **102 frames** + 8 carved cells | UNLABELED |
| 3000-atom configs | `sio2x_production` {pristine,random,sphere,plane}×{x006,x012} | 7 configs | UNLABELED |
| high-T / liquid | 12288 melt-quench 4000 K frames + `melt_msd` | trajectory | UNLABELED |

**Every target domain has candidate structures locally.** They are unlabeled production MD.

## 4. Existing Teacher-label inventory  — **FACT / DERIVED**

- **Local accepted-teacher labels:** minimal — `AL11_with_umax_v5.xyz` (11 AL cells) + augment seeds
  (input_small 50 / input_large 10). No bulk labeled training set locally.
- **Historical distillation training set (~10k):** teacher-labeled (augment-atoms, DFT-corpus seeds),
  **KISTI-only, not locally inspectable** (`PROVENANCE.md §3`). **DERIVED (R1.P1R2 provenance):** seeds
  came from the DFT corpus `SiOx_filtered`, **not** production clustered dumps ⇒ clustered production
  motifs **likely under-represented**; unverifiable locally.

## 5. Composition coverage

Candidates span SiO₂ (stoichiometric glass) → O-deficient SiO₂₋ₓ (random x0.03–0.24; clustered local_x up
to ~0.60 in carved cells). COVERED across composition. (`composition_distribution.csv`.)

## 6. Defect / topology coverage

Dilute random vacancy (350 frames, 7 x) and clustered sphere/plane vacancy + void (102 frames + 8 carved,
local_x 0.11–0.60) both have candidates. (`defect_distribution.csv`.)

## 7. Thermodynamic coverage

ρ ≈ 2.2 g/cm³ (production); T = 300 K + 4000 K melt available. COVERED for the deployment thermodynamic
range. (`thermodynamic_coverage.json`.)

## 8. Local-environment coverage

Under-coordinated / defect-Si environments present (the committee-uncertainty study measured defect-Si ≈
2.5–3× normal-Si — the existing proxy; per-frame coordination not recomputed here). Candidate diversity of
void-surface / clustered motifs confirmed by the carved-cell local_x spread.

## 9. Difficult / high-force structure coverage

The clustered / high-uncertainty configs (the committee-u-selected frames) are the difficult,
in-deployment-distribution structures — available as candidates.

## 10. Dataset imbalance — **DERIVED**

Candidate pool skews to bulk amorphous + dilute vacancy (abundant, easy) with clustered/void **candidates
present but not teacher-labeled** for the new pipeline. The scientifically important clustered/void region
is exactly where the **labels** (not the candidates) are missing/unverifiable.

## 11. Redundancy

MD dumps hold temporally-correlated frames (50–501 per config) → a dense frame sample is numerically large
but structurally repetitive; sub-sample by time spacing per config. Exact augment lineage is KISTI-only →
**NONBLOCKING** for the coverage decision. (`redundancy_summary.json`.)

## 12. Train / validation / test strategy (NEW pipeline; historical train_list NOT reused)

**GROUP split** by source trajectory / config / defect-family (not random frame split of correlated MD).
Train = broad target-domain teacher-labeled frames; Validation = held-out configs/time-segments; **Test =
the preserved DFT SCAN cells (Student-vs-DFT, PC004) + a group-disjoint held-out config.** Design held-out
independence up front (R1 lesson). (`split_design.json`.)

## 13. DFT assets preserved for PC004  — **FACT**

**42 SCAN cells** (dilute + clustered SiO₂₋ₓ) are **PRESERVED** as the PC004 Student-vs-DFT reference —
**not** consumed into distillation training. These are the only high-fidelity in-domain Student-vs-DFT
anchors. (`preserved_dft_test_assets.json`.)

## 14. Dataset gaps  — **DERIVED**

One target-domain **LABEL gap**: `SiO2x_clustered_vacancy_voidsurface` — candidates exist locally but lack
valid accepted-teacher labels for the new pipeline (historical labels KISTI/unverifiable + likely sparse
there). All other domains have candidates + reusable historical bulk labels.

## 15. Final PC002 decision — **`DISTILLATION_DATASET_REVISE`**

The dataset is broadly useful (amorphous + dilute covered; candidates for all domains) but has an
identifiable, **repairable** target-domain **label** gap (clustered/void). Not ACCEPTED (the flagged
region's labels are unverifiable/likely-sparse and can't be blindly reused), not INSUFFICIENT (candidates
exist locally), not UNRESOLVED (the gap + remedy are determined).

## 16. Reused existing assets

The historical teacher-labeled **bulk** (amorphous + dilute) may be **REUSED** where valid
(`REUSED_EXISTING_SCIENTIFIC_ASSET`); the 42 DFT cells are reused as PC004 reference. Only the
**clustered/void label region** needs new accepted-teacher labels.

## 17. Missing labels

Clustered-vacancy / void-surface structures need labeling with the accepted teacher `b56e20ff`. Suitable
**unlabeled** candidates already exist (`anneal_calib_clustered` 102 frames + 8 carved clustered cells +
committee-u-selected high-uncertainty clustered frames).

## 18. Required next action (prepared, NOT executed)

**One bounded `label_with_teacher` ActionProposal** (`examples/production_campaign_002/action_proposal.json`)
— label ~38 clustered/void (+ small dilute balancing) structures from existing production MD with the
accepted teacher `b56e20ff`, energy convention = the valid NequIPCalculator path (not the raw-torch C3
path). **Approval-gated (`costly_teacher_labeling`); `dry_run`; not executed.** No DFT/MD/training/
scheduler; no downstream chain.

## 19. PC003 authorization status

`STUDENT_TRAINING_STAGE_AUTHORIZED = false` — the dataset is REVISE, not yet ACCEPTED. PC003 is **not**
authorized and **no PC003 stub** is issued; the gate reopens only after the clustered gap is labeled and
the dataset re-evaluated.

## 20. NEW_PIPELINE_CURRENT_STUDENT = NONE

No student model was inspected, selected, compared, or set. Historical original/v5 committees appear only
as `EXISTING_HISTORICAL_STUDENT_ASSETS` (benchmarks). `NEW_PIPELINE_CURRENT_STUDENT = NONE`.

### Historical comparison — NON-AUTHORITATIVE

History shows the same clustered/defect gap drove repeated re-augmentation attempts (v3…v5 "defect
re-augment"). The independent PC002 decision (REVISE → label clustered/void) is consistent with that
history but is derived from the current coverage evidence, not adopted from it.
