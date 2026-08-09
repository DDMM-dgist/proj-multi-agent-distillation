# PC001 — Teacher Validation on the EXACT Original Held-out Test Split (FINAL / AUTHORITATIVE)

Fresh Allegro inference (APPROVED) of the accepted teacher `b56e20ff` on the **exact seed-123 held-out
test split** of the original training corpus, reconstructed from `03_allegro_train/`. **No student data,
DFT, MD, training, network, or semantic Judge.** Unrelated VASP jobs untouched. Supersedes the
error_a/PCA_SOAP-based numbers. Run: `runs/production_campaign_001/pc001-teacher-exact-heldout-validation/`.
Labels **FACT / DERIVED / INFERENCE / UNRESOLVED**.

## A. Exact train / validation / test counts (FACT)

`03_allegro_train/dataset.xyz` = **11,424** frames. `tutorial_Allegro.yaml`:
`ASEDataModule split_dataset {train 0.8, val 0.1, test 0.1}, seed 123`. Reconstructed via the exact NequIP
implementation (`RandomSplitAndIndexDataset` → `torch.utils.data.random_split(dataset,[0.8,0.1,0.1],
Generator.manual_seed(123))`, subset order [train,val,test], `splits[2]`=test):
**train 9,140 / val 1,142 / test 1,142.**

## B. How the exact split was recovered (DERIVED)

No persisted split-index file exists (only wandb metadata). The split algorithm was read from the local
NequIP source (`nequip/data/dataset/utils.py::RandomSplitAndIndexDataset`) and replicated deterministically
(seed 123, torch `random_split`). **Validated by exact metric reproduction (§H)** — the reconstruction is
confirmed correct, not approximate.

## C. Exact held-out test frame count (FACT)

**1,142 frames, 125,233 atoms.** Domain composition frozen before inference (`test_frame_manifest.json`):
bulk_cryst 235, silicon_crystalline 124, vacancy 107, cluster 71, SiOx_int 42, SiOx_max 39, quench_max 41,
bulk_amo 30, quench 43, liquid 34, surfaces 64, high-pressure 60, elemental-Si families, etc.

## D–G. Fresh global held-out result vs original training log (FACT)

| metric | fresh (this run) | NequIP training-log test | match |
|---|---|---|---|
| force component MAE (eV/Å) | **0.15613** | 0.1561268 | **EXACT** (Δ 1.3e-6) |
| force component RMSE (eV/Å) | 0.47417 | 0.4741700 | **EXACT** |
| per-atom energy MAE (meV/atom, frame-mean) | **15.733** | 15.733 | **EXACT** (Δ 5e-5) |
| energy signed bias (meV/atom) | −5.51 | — | — |

**Generalization gap ≈ 0:** held-out test 0.156 vs training 0.149 (force); test 15.7 vs train 15.1
(energy) — **no overfitting.**

## H. Reproduction status — **REPRODUCED**

Both force and energy reproduce the original NequIP test-phase metrics **exactly**. (The auto-script first
flagged an energy "MISMATCH" because it compared an atom-weighted aggregation (8.89) to NequIP's
frame-mean metric (15.73); the correct frame-mean matches exactly — `CORRECTED_reproduction_and_verdict.json`.)
Exact reproduction confirms the reconstructed split and the model (`b56e20ff`) are the exact originals.

## I. Exact per-domain held-out metrics (FACT — `fresh_test_domain_metrics.csv`)

| domain | N | force MAE | force RMSE | median | p95 | energy MAE (meV) |
|---|---|---|---|---|---|---|
| ambient crystalline SiO₂ | 250 | **0.039** | 0.057 | 0.027 | 0.152 | 4.96 |
| amorphous SiO₂ | 140 | **0.184** | 0.195 | 0.194 | 0.278 | 5.29 |
| SiO₂ₓ dilute vacancy | 149 | **0.231** | 0.250 | 0.229 | 0.392 | 10.85 |
| **SiO₂ₓ clustered vacancy / void** | 84 | **0.348** | 0.359 | 0.345 | 0.480 | 21.13 |
| surfaces | 82 | 0.202 | 0.222 | 0.180 | 0.355 | 23.56 |
| OUT elemental Si | 273 | 0.068 | 0.087 | 0.053 | 0.169 | 13.82 |
| OUT high-pressure | 93 | 0.128 | 0.161 | 0.167 | 0.273 | 5.39 |
| OUT isolated cluster | 71 | 1.196 | 6.741 | 0.351 | 1.059 | 89.97 |

The `OUT isolated cluster` mean (1.196) + RMSE 6.74 / max 56.65 are dominated by a ~2-atom isolated cluster
(out-of-scope), as in error_a.

## J–K. Amorphous / dilute / clustered held-out comparison (FACT)

**amorphous 0.184 (N=140) → dilute 0.231 (N=149) → clustered 0.348 (N=84)** — monotonic, N adequate (not
sparse). Δ(dilute−amorphous)=+0.048, Δ(clustered−dilute)=+0.117. The historical error_a trend
(0.186/0.231/0.321) is **CONFIRMED REAL on the genuine held-out test** (clustered slightly worse:
0.348 vs 0.321). This is a **CONFIRMED_REAL_TEACHER_LIMITATION** in the oxygen-deficient / clustered target
sub-region — not a contamination artifact.

## L. Exact PCA_SOAP / error_a overlap (DERIVED; corrected)

`test_set.xyz` is **not** the authoritative held-out test — it is a *different* PCA/SOAP selection from the
**same 11,424 corpus**. Energy-key matching is unreliable (`test_set.xyz` uses `dft_energy`; `dataset.xyz`
uses `dft_free_energy` — different VASP quantities), so exact structural TRAIN/VAL/TEST fractions are
**UNRESOLVED-by-energy-key**. Of the energy-matchable frames the proportions are **train 396 / val 55 /
test 48 ≈ 80/10/10** → classify `test_set.xyz` as **TRAIN+VALIDATION+TEST MIXED**, **correcting** the
earlier "~90% training-contaminated" statement. error_a is therefore a **historical mixed-overlap
benchmark**, not held-out generalization — but note its per-domain trend happened to match the true
held-out trend closely.

## M. 1155 vs 1154 explanation (FACT)

`test_set.xyz` has **1,155** structures; a prior set-based dedup returned 1,154 unique `dft_energy` values
because exactly **one pair of frames shares an identical `dft_energy`** (1 duplicate value). Full
line-count parse = 1,155.

## N. Teacher identity (FACT)

`03_allegro_train/outputs/2025-11-13/12-39-46/compiled_model.nequip.pth` is **byte-identical** (SHA
`b56e20ff…`) to the accepted teacher. Architecture: Allegro cutoff 5.0, l_max 2, 4 layers, 256/128,
float32, ZBL; nequip 0.15.0 / allegro 0.7.1 / torch 2.6.0. Energy convention:
`per_type_energy_shifts = per_atom_energy_mean` (baked), evaluated via NequIPCalculator (valid path, not
the raw-torch C3 path).

## O. Three results kept separate

- **A. Original training-log test** (exact held-out, historical run): force 0.1561, energy 15.73.
- **B. Fresh exact-held-out test** (this run): force 0.15613, energy 15.733 → **REPRODUCED**.
- **C. PCA_SOAP / error_a** (mixed-overlap benchmark): force in-scope 0.152 / global 0.190 — **not** the
  held-out test; used only as historical context.
A and B are the authoritative generalization evidence.

## P. FINAL Teacher verdict — **`TEACHER_ACCEPTED_FOR_DISTILLATION`** (DERIVED; no invented threshold)

On the exact held-out test the teacher reproduces the pipeline metrics exactly and **generalizes with
near-zero overfitting**; it is excellent on crystalline (0.039), good on amorphous (0.184) / elemental Si
(0.068) / high-pressure (0.128). A **real, systematic** force degradation is **confirmed** in the
oxygen-deficient target domain (dilute 0.231, clustered **0.348** eV/Å). Per rule 13 this is **treated as
a real teacher limitation**, documented prominently (not dismissed): it **bounds the future student's
achievable accuracy** in the clustered oxygen-deficient / void region and is the **explicit priority
target for downstream coverage / active-learning teacher improvement (PC006/PC007)**. Absent a
source-grounded accuracy requirement, and given the teacher is the sole DFT-trained reference that
generalizes cleanly, the verdict is **ACCEPT with a confirmed-real caveat** — a transparent judgment call
that neither rejects the teacher nor silently defers the weakness.

## Q. Blocking / nonblocking

- **Blocking:** none (no threshold violated; teacher generalizes; best available reference).
- **Nonblocking (real, confirmed):** clustered oxygen-deficient held-out force ~0.35 eV/Å (~1.9× amorphous)
  → priority downstream teacher-improvement (AL) target; PC002 must ensure clustered coverage; energy
  convention validity is NequIPCalculator-path-specific (C3 caveat); PCA_SOAP overlap exact fractions
  unresolved-by-energy-key.

## R. PC002 authorization

PC001 closed: **`TEACHER_ACCEPTED_FOR_DISTILLATION`**; the earlier PC002 (dataset design → REVISE, label
the clustered/void gap) stands and is **reinforced** by this held-out result (clustered is the real weak
region). `STUDENT_STAGE_AUTHORIZED = false`; `NEW_PIPELINE_CURRENT_STUDENT = NONE`.

## S. Walltime

Fresh exact-held-out inference: **53.1 min** (1142 frames, CPU, NequIPCalculator).

---

## Force-scale-normalized held-out fidelity (closes PC001)

**Question:** is the amorphous 0.184 → dilute 0.231 → clustered 0.348 eV/Å absolute force-error rise a
genuine relative-fidelity deterioration or an absolute-scale effect (clustered = larger DFT forces)?
Computed from the **already-generated** exact-held-out per-frame error metrics + DFT force scale re-read
from `dataset.xyz` (DATA, not a model forward). Artifact:
`force_scale_normalized_domain_metrics.json`.

| domain | N | DFT comp RMS | abs err comp RMSE | **normalized RMSE** | normalized MAE | normalized vec err |
|---|---|---|---|---|---|---|
| ambient crystal | 250 | 1.42 | 0.085 | **0.060** | — | — |
| amorphous SiO₂ | 140 | 2.19 | 0.261 | **0.119** | 0.13 | ~0.13 |
| **dilute vacancy** | 149 | 1.32 | 0.347 | **0.262** | ~0.28 | ~0.28 |
| **clustered/void** | 84 | 3.49 | 0.541 | **0.155** | ~0.16 | ~0.16 |
| surfaces | 82 | 2.91 | 0.354 | **0.122** | — | — |

amorphous→dilute→clustered: absolute err RMSE ×2.07; DFT-scale ×1.59; **normalized RMSE 0.119→0.262→0.155**.

**Interpretation — C_MIXED (real relative component):**
- The **clustered** absolute-error peak is **largely an absolute-scale effect** — clustered DFT forces are
  1.6× larger (RMS 3.49 vs 2.19); its normalized RMSE (0.155) is only ~1.3× amorphous.
- The **dilute oxygen-deficient** domain shows the **worst *relative* fidelity** (normalized RMSE 0.262 =
  **2.2× amorphous**), despite the smallest DFT force scale (1.32) — partly amplified by the small-force
  error-floor, but 2.2× is substantial and genuine.
- ⇒ the **core oxygen-deficient target domain** has real relative-fidelity degradation (dilute 2.2×,
  clustered 1.3× vs amorphous), **not purely** an absolute-scale artifact.
- *Cosine similarity / component-R² were not computed* — per-atom teacher force arrays were not persisted
  by the exact-held-out run and re-forwarding is forbidden; the normalized-error + DFT-scale evidence
  suffices for the A/B/C determination.

## FINAL PC001 Teacher verdict — **`TEACHER_REVISE_BEFORE_DISTILLATION`**

(Supersedes the interim ACCEPT in §P / `CORRECTED_reproduction_and_verdict.json`.) The teacher reproduces
the pipeline held-out metrics exactly and generalizes with near-zero overfitting, and the clustered
absolute-error peak is mostly scale-driven — **but** force-scale-normalized fidelity is **genuinely
degraded in the oxygen-deficient target domain** (dilute normalized RMSE 2.2× amorphous; clustered 1.3×),
not merely absolute-scale. Absence of a project threshold does **not** imply acceptance; judged on relative
fidelity across the intended target domain (oxygen-deficient SiOx — the core distillation objective), the
conservative verdict is **REVISE**: improve the teacher in the oxygen-deficient (dilute + clustered)
domain via targeted DFT-anchored active learning **before** finalizing distillation, **or** establish a
source-grounded deployment accuracy requirement showing the current oxygen-deficient fidelity is adequate.
This is consistent with — and reinforces — PC002's own REVISE (the clustered/void label gap): the same
oxygen-deficient region needs both teacher improvement and dataset coverage.

**PC002 authorization: NO.** `STUDENT_STAGE_AUTHORIZED = false`; `NEW_PIPELINE_CURRENT_STUDENT = NONE`.
Recommended next (later task; not executed here): one minimal DFT-anchored active-learning action targeting
dilute + clustered oxygen-deficient vacancies to raise teacher relative fidelity in the core domain.
PC001 is now closed permanently on this authoritative evidence.

---

## Root cause of the defect-domain degradation (training-set coverage/diversity)

Diagnostic on the **exact seed-123 TRAIN split** of `dataset.xyz` (read-only; DFT forces from the data
file; **no inference**). Artifacts: `train_defect_coverage.json`, `root_cause_determination.json`; script
`work/pc001_train_defect_coverage.py`.

| domain | TRAIN frames | families | local_x range (distinct) | DFT force RMS mean | test⊂train force range |
|---|---|---|---|---|---|
| amorphous | 1067 | bulk_amo/quench/liquid | 0 (stoich) | 1.95 | 100% |
| **dilute vacancy** | **1192** | vacancy_int_AL 627, vacancy 228, SiOx_int_AL 337 | −0.05→0.75 (72) | 1.32 | **100%** |
| **clustered/void** | **707** | SiOx_max_AL 374, quench_max_AL 313, surfaces_max_AL 20 | 0→0.93 (16) | 3.62 | **100%** |

**Decision tree →**
- coverage insufficient? **NO** — dilute 1192, clustered 707 training frames (substantial).
- coverage OK but diversity insufficient? **NO** — wide compositional spread (72 / 16 distinct local_x),
  wide force-scale spread (clustered RMS 0.5–7.4), and **100% of held-out defect frames interpolate** the
  training defect force-scale range.
- **coverage + diversity sufficient → TEACHER/MODEL/TRAINING problem.** ✔ — reinforced by the **near-zero
  train→test gap** (0.149→0.156): the model fits training and held-out defects *equally*, so the defect
  error is an **irreducible fit floor**, not under-coverage or overfitting.
- DFT-labels problem? **POSSIBLE SECONDARY** — under-coordinated-Si / void-surface SCAN forces may be
  noisier/less-converged (flag for reference review; `surfaces_max_AL` is the only thin corner, 20 train
  frames). Unconfirmable without DFT re-checks.

**DETERMINATION: `TEACHER_MODEL_TRAINING_LIMITATION`** — the teacher has an intrinsic force-error floor
(~0.15–0.35 eV/Å) on defect structures, most visible in the low-force dilute domain where that floor
dominates the normalized error. It is **not** a training-data coverage or diversity problem.

**Refines the REVISE remedy (correcting the earlier suggestion):** because coverage + diversity are
already sufficient, **DFT-anchored active learning / dataset augmentation is unlikely to lower the floor**.
The indicated fix is **teacher model/training improvement** (increased capacity, low-force/defect-weighted
force loss, training-length/hyperparameter changes) **and/or a DFT-label quality review** of the defect
families — not more defect data. The FINAL PC001 verdict stays `TEACHER_REVISE_BEFORE_DISTILLATION`; the
cause of the required revision is now identified as **model/training-level**, and **PC002 remains
unauthorized**.
