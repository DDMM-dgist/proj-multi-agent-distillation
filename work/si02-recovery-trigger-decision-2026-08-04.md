# SiO2 Recovery Trigger Decision — 2026-08-04

Planning only. No RecoveryPlan was registered, no approval recorded, no calculation started.
Companion to `si02-authoritative-run-audit-2026-08-04.md`.

## 1. Decision

**NO GENUINE RECOVERY TRIGGER SELECTED YET.**

**LEADING CURRENT-REVALIDATION CANDIDATE: `dataset_coverage`** (committee-uncertainty-driven).

Status (each line verified against inspected files):
- **LEGACY SCIENTIFIC BASELINE:** v5 committee; latest legacy gate = `v5-committee-ADOPT` **PASS 3/3**.
- **CURRENT-CONTROLLER AUTHORITATIVE BASELINE:** none.
- **OPEN LEGACY GATE FAILURE:** none (the last student gate PASSED).
- **RESIDUAL SCIENTIFIC CONCERN:** committee uncertainty remains elevated on high-defect AL cells
  (u_max above the *paper aspiration* of 0.30, which was **not** a required-pass criterion).
- **RECOVERY AUTHORIZATION:** not granted.

Recovery is NOT triggered. `dataset_coverage` is only the leading candidate for a *future*
current-framework revalidation, to be decided by a re-run current gate — not by the residual
u_max signal alone. It cannot become a recovery trigger until (a) a current-controller baseline
is onboarded and (b) a current three-lens gate actually returns REVISE/FAIL (see §14).

## 2. Evidence

- v5-committee is the adopted student; latest gate `v5-committee-ADOPT` = **PASS 3/3** (2026-07-16).
- **The v5 ADOPT criteria were relative-improvement-over-original**, NOT an absolute 0.30 cutoff
  (`gates/vote_v5.workflow.js:23-28,41`): (1) v5 `u_max_mean ≤ original` at every x of the
  deployment/random sweep; (2) error(c) ≤ 0.368; (3) F_RMSE ≤ ~0.31; (4) beats the failed v3
  re-distillations. All three judges confirmed these (`coordination_votes.csv`, 2026-07-16).
- **u_max < 0.30 eV/Å was the "paper reliability aspiration", explicitly NOT an ADOPT blocker**
  (`vote_v5.workflow.js:27,41`: "full <0.30 everywhere is the paper aspiration; ADOPT-over-original
  only requires strictly beating the original"). So exceeding 0.30 is NOT a gate-criterion violation.
- Residual concern (diagnostic): `committee_umax_AL11_v5.log` shows committee `u_max` over the 11
  AL cells = mean 0.334 / median 0.321 / p90 0.461 / max 0.464 eV/Å; deployment `u_max_mean` per x
  = 0.286/0.295/0.349/0.393/0.373 (v5) vs 0.303/0.358/0.377/0.421/0.419 (original) — improved at
  every x, still above the 0.30 *aspiration* at x≥~0.12 and clustered cells.
- error(c) v5 = 0.337 < 0.368; F_RMSE ≈ 0.285 < 0.309 — raw fidelity is NOT the gap.

## 3. Iteration 1 baseline artifacts

Local, reusable as a baseline (but NOT current-controller hash-bound yet):
- Student: `gpu_return_v5_committee/v5_committee_bundle/seed0{1,2,3,4}/potential_saved_bestmodel`
  (+ `input.yaml`, `params_O/Si`, `pca`, `scale_factor`).
- Committee-uncertainty baseline: `committee_umax_AL11_v5.log`, `AL11_with_umax_v5.xyz`,
  `sio2x_production/committee_u_out/summary_all_configs.csv` + `*_frame_stats.csv`.
- Teacher: `teacher/model.nequip.pth` (+ finetuned variants).
- Error baselines: `teacher_diag/error_{a,b,c}_*.csv`, `committee_error_summary.csv`.

## 4. Failure category (candidate only — no failure has been issued)

Candidate `dataset_coverage` — IF a current gate later flags it: the deployed committee is
under-covered on high-defect-fraction / clustered SiO2-x configurations, producing committee
`u_max` above the 0.30 aspiration there. This is a hypothesis for a future current-framework
gate to test, not an issued REVISE/FAIL.

## 5. Responsible agent

Data-curation / active-learning agent (add high-σ_F structures), then teacher-labeling and
4-seed student retraining agents. (Maps to `RECOVERY_AGENTS` once wired to the current controller.)

## 6. Return stage

Return to data-curation (append high-σ_F AL cells preserving lineage) → teacher relabel →
4-seed retrain → same-profile revalidation → gate.

## 7. Proposed data/config change

Add ~30–50 high-`u_max` AL cells concentrated at high defect fraction / clustered geometry
(precedent: legacy "AL iter-3 (30–50 cells)"), preserving `parent_structure_id` lineage; record
a `DataCoverageReport` delta. No replay unless declared.

## 8. Teacher relabel required

**Yes** (label the new AL cells with the same teacher checkpoint, hash-bound). Requires the
teacher env (`allegro`) and the base corpus for consistent labeling.

## 9. New DFT required

**No** for a coverage/uncertainty cycle (base project is DFT-anchor = 0). New DFT would be a
separate declaration; reference `dft_labeling/` ≈ 44 GB is remote.

## 10. Four-seed retraining required

**Yes** — retrain the 4-seed SIMPLE-NN committee on the augmented set (SIMPLE-NN env `simple-nn`
present; not run here). Approval-gated (costly_training).

## 11. Frozen revalidation profile

Must freeze a current-framework `validation_profile` encoding the legacy criteria:
committee `u_max < 0.30` (per-x on the deployment distribution + AL cells), error(c) `< 0.368`,
F_RMSE ≤ original, plus density / RDF(Si-O) / NVE-drift. **This profile does not exist locally
yet** and must be authored/frozen before revalidation.

## 12. Estimated computational cost

- Teacher relabeling of 30–50 cells: small.
- 4-seed SIMPLE-NN retrain: dominant cost (per-seed training × 4).
- Revalidation (committee-u + error + physical observables): moderate.
- No new DFT. Exact wall-time not estimable without scheduler/hardware access (unavailable here).

## 13. Human approval boundary

Approval required before every costly stage (teacher relabeling batch, 4-seed retraining, any
production MD). No approval is recorded by this document.

## 14. GO conditions (all required to promote to SELECTED)

1. Current-controller authoritative baseline registered from the v5 bundle (run_manifest +
   `code_revision` + artifact hashes).
2. Base teacher training corpus accessible (KISTI-remote) with lineage / `parent_structure_id`.
3. Frozen current-framework `validation_profile` (u_max<0.30, error(c)<0.368, F_RMSE, physical).
4. SIMPLE-NN + teacher environments confirmed runnable (envs present; not yet validated).
5. Scheduler / compute access (absent this session).
6. Chosen AL increment (high-defect-fraction / clustered cells) and cost estimate.

## 15. STOP conditions

- Any GO condition unmet → do not start compute; remain PROVISIONAL.
- No issued gate REVISE/FAIL and no current-controller baseline → do not fabricate a failure or
  synthesize a gate; keep PROVISIONAL.
- If the residual signal cannot be reproduced against a hash-bound baseline → downgrade to
  `NO GENUINE TRIGGER (current framework)` and report.
