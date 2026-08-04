# SiO2 Current-Controller Baseline Onboarding Plan — 2026-08-04

Planning only. No controller run was created, no artifact registered, no gate executed. This
plan onboards the legacy v5 result as *imported evidence* into a new current-framework baseline
run — it does NOT disguise legacy work as a current-controller run.

## 1. Onboarding objective

Create a new current-framework baseline run that references the legacy v5 committee as imported,
hash-bound evidence, so a current three-lens gate can be run against it later. The legacy science
stays legacy; the current-framework verdict is established only by re-running current validators
and the current gate — never by copying the legacy PASS.

## 2. Legacy source identity

- Source repo: `…/research-sio2-allegro-simplenn-distillation`, git `71b96b3` (single init commit).
- Legacy gate: `v5-committee-ADOPT`, PASS 3/3, 2026-07-16 (`coordination_log.csv`,
  `gates/coordination_votes.csv`, `gates/vote_v5.workflow.js`).
- Legacy pipeline: JS gate workflow + `judge.md` committee — NOT `workflow/controller.py`.
- Recorded separately from the current onboarding-session revision (do not conflate).

## 3. Imported artifacts

From `si02-v5-legacy-baseline-manifest-2026-08-04.json` (hashes already computed read-only):
- 4 v5 student seed directories (`seed01–04`, directory digests).
- Teacher candidates (original + finetuned_v2) — exact v5 teacher UNRESOLVED (see §4).
- Validation-evidence CSVs (deployment random-sweep u_max, error(c), error a/b/c, NVE, etc.).
Each imported with its ORIGINAL hash preserved; marked `classification: legacy_scientific_baseline`,
`current_controller_authoritative: false`.

## 4. Missing artifacts

- Exact v5 teacher checkpoint identity ("T1") — not hash-pinned in legacy evidence.
- Base teacher training corpus (`training_set.xyz`) — remote (KISTI); required for any retrain/relabel.
- Current-framework `validation_profile.yaml` — does not exist yet (draft in
  `si02-current-validation-profile-draft-2026-08-04.yaml`).
- Student-MD–DFT (channel d) figure-ready evidence — partial/unresolved.
- Dataset lineage / `parent_structure_id` for the imported student's training set — remote.

## 5. Current code revision

- Onboarding-session revision = current `origin/main` (record the exact SHA at run creation).
- Stored DISTINCT from `source_git_revision` (`71b96b3`). No backfill of a past `code_revision`
  onto current artifacts; the imported artifacts keep their legacy provenance.

## 6. Proposed validation profile

Use `si02-current-validation-profile-draft-2026-08-04.yaml` (draft, unregistered). Human approval
required before it is frozen/registered. Committee-uncertainty 0.30 is entered as **diagnostic
(required: false)** per the threshold-provenance finding (§7).

## 7. Threshold provenance

RESOLVED (from `gates/vote_v5.workflow.js:23-28,41` + `coordination_votes.csv` 2026-07-16):
- Observable: `u_max_mean` (mean over frames of per-frame max committee force std), unit eV/Å.
- Evaluation subset: deployment/random-sweep configs (x=0.03…0.24) — the BINDING test set;
  AL11 / sphere_x012 are diagnostic.
- Role of 0.30: **paper reliability *aspiration*** — explicitly NOT an ADOPT blocker.
- v5 PASS basis: relative improvement over the ORIGINAL (u_max_mean ≤ original at every x) AND
  error(c) ≤ 0.368 AND F_RMSE ≤ ~0.31 — combined relative criteria, not the absolute 0.30.

## 8. Required-pass vs diagnostic observables

- Required-pass (candidate, pending approval): error(c) vs DFT; student–teacher force fidelity;
  physical observables with a defined reference+protocol (density, RDF Si-O) if their protocol is
  resolvable.
- Diagnostic (required: false): committee `u_max` vs the 0.30 aspiration; high-defect subset u_max;
  ADF/FSDP where protocol/threshold are not pinned.
- The legacy "beat-original" criterion is a **model-selection** rule, not an absolute required-pass;
  it is NOT auto-imported as a current required-pass without a fixed reference.

## 9. Current-gate execution plan

Run the current three-lens gate ONLY after §3–§8 are satisfied and validators have produced
current evidence. The gate verdict is whatever the current validators + judges return — the
legacy PASS is not copied.

## 10. No-compute checks (allowed now)

- Artifact hashing + manifest validation (done).
- Provenance binding of imported artifacts (legacy SHAs recorded).
- Criterion/profile drafting + binding review.
- Gate-input assembly from existing raw CSV evidence.

## 11. Compute-required checks (NOT now; approval-gated)

- Static re-evaluation of the imported committee on existing trajectories (light; needs model
  loading → explicit approval).
- Teacher relabeling, 4-seed retraining, deployment MD, new DFT (HPC; approval-gated).

## 12. Human approval boundary

Approval required before: freezing/registering the validation profile; any model loading; any
teacher relabeling / retraining / MD / DFT; any scheduler submission. None recorded here.

## 13. GO conditions

1. Legacy manifest reviewed and accepted.
2. Validation profile reviewed, thresholds' required/diagnostic status approved, then frozen.
3. **Exact v5 teacher ("T1") identity resolved (hash-pinned). This is GATE-BLOCKING: a
   teacher-dependent required check (student_teacher_force_error) cannot be bound while the
   teacher is unresolved, so the current three-lens gate MUST NOT run until it is resolved.**
4. Decision on which current evidence can be reused from raw CSVs vs must be recomputed.
5. Current code revision recorded at baseline-run creation.

## 14a. Decision sequence (ordered; nothing skipped)

1. Legacy artifact hashing and provenance binding (done — manifest JSON).
2. Validation-profile human approval (thresholds' required/diagnostic status), then freeze.
3. Current-controller baseline run creation (records current git SHA, distinct from legacy).
4. Register imported legacy artifacts (original hashes preserved; `authoritative:false`).
5. Rerun current validators where existing raw evidence is sufficient (no new compute).
6. Compute ONLY the missing validation evidence (approval-gated).
7. Run the current three-lens gate — **BLOCKED until the v5 teacher identity is resolved** (a
   teacher-dependent required check is otherwise unbindable; see GO #3 / STOP).
8. Act on the result:
   - **PASS** → do NOT force any recovery; baseline adoption complete; move to the next material
     system or define a stricter scientific question.
   - **REVISE/FAIL** → author a genuine RecoveryPlan, route by failure category; only THEN decide
     whether `dataset_coverage` recovery is warranted.

## 15. Compute-cost classification

- NO COMPUTE (allowed now): hashing, manifest validation, existing-CSV re-verification,
  criterion/profile binding, gate-input assembly.
- LIGHT LOCAL COMPUTE (approval before model loading): recompute metrics from existing
  trajectories; static evaluation of the existing committee checkpoints.
- HPC REQUIRED (approval-gated): teacher relabeling, 4-seed SIMPLE-NN retraining, deployment MD,
  new DFT anchors.
- This session performs NO COMPUTE only.

## 16. STOP conditions

- Threshold semantics contested or a required-pass lacks a fixed reference+protocol → keep that
  observable diagnostic; do not invent a threshold.
- **v5 teacher ("T1") identity unresolved → GATE-BLOCKING: do NOT run the current three-lens
  gate** (the teacher-dependent required check `student_teacher_force_error` cannot be bound).
  Onboarding is limited to evidence import + hashing only; no gate, no retrain/relabel, until the
  teacher is resolved (and the profile approved).
- Base training corpus unresolved (KISTI-remote) → do not attempt retrain/relabel.
- Any attempt to copy the legacy PASS into a current verdict → STOP; current verdict stays NONE
  until current validators + gate run.
