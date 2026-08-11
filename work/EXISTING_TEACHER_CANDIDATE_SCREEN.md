# PC001 · Phase 1 — Cheap Existing-Teacher Candidate Screen

**Decision: `DO_NOT_ADVANCE` — no existing teacher improves oxygen-deficient force fidelity.**
Resource-aware step: screen every materially-distinct *already-produced* teacher on the central
held-out domains **before** spending any GPU on new training. If an existing model already solved
the gap, we would select it and never train. It does not, so we proceed to parallel mode.

Authoritative run dir: `runs/production_campaign_001/pc001-existing-teacher-screen/`
Machine-readable: `work/existing_teacher_candidate_screen.json`
Screen script: `work/pc001_existing_teacher_screen.py` (allegro env, NequIPCalculator valid path)

---

## 1. Candidates enumerated (with dedup)

| Candidate | SHA256 (prefix) | Identity | How screened |
|---|---|---|---|
| **base** `b56e20ff` | `b56e20ffc31da601…` | = `03_allegro_train` compiled output (byte-identical); the current pipeline teacher | **reused**, not rerun, from committed exact-heldout per-frame metrics |
| **teacher_finetuned_v2** `b3be4d2a` | `b3be4d2aa33ec5cd…` (4,907,282 B) | ER / energy-anchored fine-tune of the base | **fresh** NequIPCalculator inference on the same 373 frames (19.7 min, CPU) |
| **teacher_v6_finetuned** `277262dc` | `277262dc2d471247…` (4,907,282 B) | tier-weighted ER-FT, warm-start from a **T1** ckpt (not base), trained on 16 DFT-anchor defect cells to lower the defect-core force floor; from `v6_return_package/` (2026-08-10) | **fresh** NequIPCalculator inference on the same 373 frames (19.7 min, CPU) |

**Deduplicated (same base model, different container — not distinct candidates):** the `.pt2`
(`6c172722`) and `best.ckpt` (`51342b33`) are the base teacher in other formats; the three
`teacher_finetuned_v2` copies (`gpu_return/`, `gpu_return_v2/…`, `gpu_return_v3final/…`) are
byte-identical. → **3 materially-distinct teachers** exist (base, v2, v6); all three screened.

**v6 note.** `v6_return_package` shipped only `teacher/` weights + `validation/` summaries — its
ER-FT training data (`er_finetune_v6_*.xyz`, manifest) is **not local**, so an offline train/test
overlap (leakage) check is not possible. It is not needed here: v6 does **not** improve on the
held-out set, and any leakage would only have *helped* it, so the null/worse result is robust.
v6's Stage-1 `err(a)_core = 0.360 eV/Å` is certified **in-sample** (all 17 anchor cells are in its
training set); the summary itself flags held-out generalization was untested — this screen is that
test.

## 2. Screening set — 373 exact central-domain held-out frames

Restricted the exact seed-123 held-out TEST split to the three scientifically central domains,
taking `dataset_index` verbatim from the committed `fresh_test_per_frame_metrics.csv` so **every
candidate is scored on identical structures**. Base metrics are the committed values (no rerun);
only the candidate required fresh inference.

| Domain | N | Description |
|---|---|---|
| amorphous_SiO2 | 140 | bulk_amo / quench / quench_int_AL / liquid |
| SiO2x_dilute_vacancy | 149 | vacancy / vacancy_int_AL / SiOx_int_AL |
| SiO2x_clustered_vacancy_voidsurface | 84 | SiOx_max_AL / quench_max_AL / surfaces_max_AL |

## 3. Result — both fine-tuned candidates are worse on held-out defects

Force component MAE (eV/Å), candidate minus base (positive = worse):

| Domain | base MAE | **v2** MAE | Δv2 | **v6** MAE | Δv6 | improved by any? |
|---|---:|---:|---:|---:|---:|:--:|
| amorphous_SiO2 | 0.1797 | 0.1824 | +0.0027 | 0.1855 | +0.0058 | ✗ |
| SiO2x_dilute_vacancy | 0.2258 | 0.2267 | +0.0009 | 0.2300 | +0.0042 | ✗ |
| SiO2x_clustered_vacancy_voidsurface | 0.3528 | 0.3538 | +0.0010 | 0.3637 | **+0.0109** | ✗ |

v2's per-atom energy MAE also rose in every domain (5.29→5.62 / 10.85→11.48 / 21.13→22.99
meV/atom). **v6 is worse than v2 in all three domains** — most on clustered (+0.0109) — *despite*
its in-sample defect-core certification.

## 4. Advancement rule & decision

**Rule (no invented threshold):** a candidate advances **only if** it *clearly* improves dilute
and/or clustered oxygen-deficient fidelity **without** degrading amorphous — a monotone-improvement
comparison against the base, not a pass/fail accuracy cutoff.

**Neither `teacher_finetuned_v2` nor `teacher_v6_finetuned` improves anything** — every central
domain is worse on force MAE (and v2 also on normalized RMSE / energy). → **`DO_NOT_ADVANCE`** for
both (no full-1142 validation needed). The most notable result is v6: its anchor-cell ER-FT gain
(`err(a)_core` 0.398→0.360 **in-sample**) does **not** transfer to held-out defect structures — it
slightly *degrades* them, confirming the gap is a genuine held-out fit floor, not an anchoring
deficit.

## 5. Why this is the expected result (consistency check)

Consistent with the established PC001 root cause **`A_MODEL_TRAINING_UNDERFIT`**: the teacher's
error on the defect domains is a *fit floor* (train error ≈ held-out test error there:
clustered test/train ratio 1.05, dilute 1.02, amorphous 0.98). ER fine-tuning re-anchors targeted
quantities — v2 energy/elastic response, v6 the defect-core environments of 16 anchor cells — but
neither lowers that held-out force floor. v6 makes the point sharply: it *did* improve its anchor
cells in-sample, yet on held-out defect structures it is slightly worse, so the anchoring gain does
not generalize. An already-trained variant was not expected to — and does not — move held-out
oxygen-deficient force fidelity. **Only a new controlled training experiment can test whether the
floor is movable.**

## 6. Consequence → parallel mode

No existing teacher solves the gap. Per the resource-aware strategy we now run two tracks in
parallel — see **`work/PARALLEL_TEACHER_AND_PYDANTIC_WORKFLOW.md`**:
- **Track A** — ONE minimal controlled Allegro training experiment (approval-gated, cost-bounded).
- **Track B** — teacher-weight-independent PydanticAI downstream (PC002 dataset structural design
  now; PC003/PC004/PC005 infrastructure prep) that must not block on Track A.
Irreversible teacher-dependent compute (mass relabeling, final student training) is deferred to
the join point `FINAL_TEACHER_IDENTITY_RESOLVED ∧ DATASET_STRUCTURE_RESOLVED`.

---

### Guardrails honored
- No new training/DFT/MD/Student/Judge launched; architecture remains frozen.
- VASP `vasp_std` jobs untouched (only my own background inference ran, now complete).
- Base teacher **not** rerun; reused committed metrics. Candidate screened once (373 frames).
- No pass/fail accuracy threshold invented; decision is a monotone comparison to base.
