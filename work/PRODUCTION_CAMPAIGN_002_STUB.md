# Production Campaign 002 — Distillation Dataset Design (STUB; not started)

Unlocked by **PC001 `TEACHER_ACCEPTED_FOR_DISTILLATION`**. This is a planning stub only — **no teacher
labeling, no student training, no DFT/MD, no compute.** It records the initial scientific question and
constraints; it is **not** answered here.

## Gate that authorizes PC002

- `TEACHER_STATUS = TEACHER_ACCEPTED_FOR_DISTILLATION` (PC001 final).
- `DISTILLATION_DATASET_STAGE_AUTHORIZED = true`; `STUDENT_STAGE_AUTHORIZED = false`.
- `NEW_PIPELINE_CURRENT_STUDENT = NONE`; historical original/v5 students are **benchmark assets only**.

## Initial scientific question

> **What training distribution should the accepted teacher (`b56e20ff`) supervise so that the future
> student covers the intended deployment domain — amorphous SiO₂ + dilute *and* clustered SiO₂₋ₓ vacancies
> / void surfaces — including the region where the teacher itself is hardest (clustered-defect force
> ~0.32–0.35 eV/Å)?**

## What PC002 must independently determine (later, not now)

1. **Structural domain to represent** — from the PC001 target domain: amorphous SiO₂, dilute + clustered
   SiO₂₋ₓ, void surfaces / under-coordinated Si, surfaces, ambient crystalline reference; at production
   density/temperature.
2. **Unlabeled candidate structures that already exist** — production-MD trajectories
   (`production_12288/`, `sio2x_production/`, `random_sweep`, `anneal_calib_clustered`) as the candidate
   reservoir; enumerate, do not assume.
3. **Teacher labels that already exist** — 39 SCAN cells + the historical ~10k augment (KISTI); classify
   which are reusable vs must be regenerated with the accepted teacher.
4. **Additional teacher labeling required** — especially in the PC001-flagged clustered-defect / void-
   surface region (the teacher's hardest in-domain area) so the student can at least reach the teacher's
   ceiling there.
5. **Train / validation / test split strategy** — with genuine held-out independence for any later
   generalization claim (the R1 leakage lesson: keep a provably-disjoint test set from the start).
6. **Coverage targets** — per target domain, a minimum representation so no in-scope region is
   under-represented (avoid the historical clustered-defect sparsity in *training* even though the DFT
   *reference* was COVERED).

## Explicitly out of scope for PC002

Student training (PC003), student validation (PC004), physical/MD validation (PC005), failure/coverage
diagnosis (PC006), active learning / new DFT (PC007). Do **not** auto-select historical v5 augmentation
data; evaluate reuse on evidence.

## Guardrails carried forward

Frozen architecture; append-only provenance; human approval mandatory for costly actions (teacher
labeling at scale, DFT, training, MD, scheduler); deterministic gates own verdicts; the C3 lesson (input
domain + energy convention explicit); the R1 lesson (design held-out independence up front).
