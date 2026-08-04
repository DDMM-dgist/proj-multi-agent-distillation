# SiO2 Authoritative Run Audit — 2026-08-04

Read-only audit. No run was initialized, no state changed, no calculation started. Every claim
below is from an inspected file with an absolute path; unverified items are marked explicitly.

## 1. Executive verdict

- **No authoritative *current-controller* run exists.** There is no `runs/` directory, no
  `workflow/controller.py` run_manifest, and no `schema_version` state JSON anywhere in the
  toolkit or the evidence repo (`grep -r schema_version --include=*.json` → none).
- The existing SiO2 science was produced by a **legacy JS gate pipeline** (`gates/*.workflow.js`
  + `.claude/agents/judge.md`), logged in `coordination_log.csv` / `gates/coordination_votes.csv`.
- That legacy pipeline **already ran and closed a recovery loop**: v3 (FAIL/REVISE) → **v5
  committee ADOPTED, unanimous PASS 3/3, 2026-07-16** ("AL loop CLOSED for this iteration").
- Therefore the latest gate state is **PASS, not an open REVISE/FAIL** — there is no *issued*
  failure to trigger recovery. **No genuine recovery trigger is selected yet.**
- The residual u_max > 0.30 is measured against the **paper *aspiration*** (not a required-pass
  gate criterion — `vote_v5.workflow.js:27,41`); it is a **leading current-revalidation
  candidate (`dataset_coverage`)**, not an approved recovery.

Status: LEGACY SCIENTIFIC BASELINE = v5 committee (last legacy gate ADOPT PASS 3/3);
CURRENT-CONTROLLER AUTHORITATIVE BASELINE = none; OPEN LEGACY GATE FAILURE = none;
RESIDUAL SCIENTIFIC CONCERN = committee uncertainty elevated on high-defect AL cells;
RECOVERY AUTHORIZATION = not granted.

## 2. Search scope

Roots searched read-only (not an unbounded home scan):
- Toolkit (audit worktree, `origin/main` @ `35f90aa`).
- Evidence repo **RES** = `…/materials-ml-kit/research-sio2-allegro-simplenn-distillation` (106 GB local mirror).
- `…/materials-ml-kit/teacher`, `…/materials-ml-kit/distillation-agents`.
- Paper repo `…/paper-sio2-agentic-distillation/work/` (evidence_inventory, gpu_handoff, speed_benchmark).

## 3. Candidate runs

| Candidate | Path | Purpose | Framework | Run manifest | Gate evidence | Authority |
|---|---|---|---|---|---|---|
| Toolkit `runs/` | (toolkit) | current-controller runs | current `controller.py` | **NONE** | — | NONE (does not exist) |
| Legacy gate pipeline | `RES/gates/`, `RES/coordination_log.csv`, `RES/coordination_votes.csv` | 13-gate judge-committee decisions | legacy `gates/*.workflow.js` + `judge.md` | none (CSV logs, not controller manifest) | YES (22 PASS/20 REVISE/3 FAIL; 40 judge votes) | LEGACY (adopted science) |
| v5-committee student | `RES/gpu_return_v5_committee/v5_committee_bundle/seed01–04/` | adopted 4-seed student | legacy re-distillation | none | v5-ADOPT PASS 3/3 (2026-07-16) | LEGACY, adopted — NOT current-controller |
| v3 / v3-final / v3-final-v2 | `RES/gpu_return_v3final/…`, gates `vote_v3final*.js` | rejected re-distillations | legacy | none | FAIL / REVISE | LEGACY, superseded by v5 |
| production_12288 melt-quench | `RES/production_12288/` | production MD (Paper-2 science) | legacy | none | meltquench-protocol REVISE (accepted) | LEGACY (Paper-2 scope) |

## 4. Authoritative-run decision

- **Current-controller authoritative run: NONE FOUND.**
- **v5-committee: AUTHORITATIVE SCIENTIFIC RESULT (legacy pipeline) — classified `LEGACY`,
  NOT AUTHORITATIVE FOR FRAMEWORK RECOVERY.** It was produced by the JS gate workflow, not
  `workflow/controller.py`; it has no run_manifest, no `code_revision` binding, and no
  `workflow/integrity` artifact-hash binding, so it cannot anchor a current-controller
  `propose_recovery`/`verify_recovery` cycle (which compares hash-bound baseline vs post-fix
  artifacts) without first being registered as a controller baseline.
- Other candidates: v3 variants = `LEGACY/SUPERSEDED`; production_12288 = `LEGACY/Paper-2 scope`.

## 5. Code and manifest binding

- RES git: single init commit `71b96b3` ("init: SiO2 Allegro→SIMPLE-NN distillation project");
  working tree dirty (STATUS.md/PROJECT.md/CLAUDE.md modified). The repo is a snapshot, not a
  per-run provenance store.
- No `code_revision`, config-hash, teacher-model-hash, dataset-hash, checkpoint-hash, or
  validation-profile-hash bound by the current framework was found in any run artifact.
- Gate criteria are encoded in `gates/run_decision_gates.workflow.js` (13 gates) and per-vote
  rationale in `coordination_votes.csv` — human/agent notes, not controller-bound criteria.

## 6. Existing scientific evidence map

Resolved (absolute paths under RES, verified present):
- **Teacher checkpoint**: `teacher/model.nequip.pth` (4.7 MB, original); finetuned variants
  `gpu_return*/…/teacher_finetuned_v2.nequip.pth` (4.7 MB) + `…_best.ckpt` (17 MB).
- **Student v5 4-seed committee**: `gpu_return_v5_committee/v5_committee_bundle/seed0{1,2,3,4}/`
  (each: `input.yaml`, `params_O`, `params_Si`, `pca`, `potential_saved_bestmodel`, `scale_factor`);
  plus `AL11_with_umax_v5.xyz`, `committee_umax_AL11_v5.log`.
- **Four-error CSVs**: `teacher_diag/error_a_allegro_vs_dft.csv` (129 KB),
  `error_b_clean_simplenn_vs_allegro.csv` (164 KB), `error_c_simplenn_vs_dft.csv` (157 KB),
  `committee_error_summary.csv` (20 KB), `eos_all_{allegro,student,dft}.csv`, `figs/F11_nve_drift.csv`.
- **Committee-uncertainty**: `sio2x_production/committee_u_out/summary_all_configs.csv`
  (sphere_x012 u_max_mean 0.3747, u_max_max 0.6233), 8 `*_frame_stats.csv`, `spatial_enrichment.csv`.
- **Gate trail**: `coordination_log.csv` (17 rows), `gates/coordination_votes.csv` (40 votes).
- **Throughput**: paper repo `work/speed_benchmark/out_np*.log`, `out_size_*.log`, `log.lammps`.

Unresolved / not local:
- **Base teacher training corpus** (`training_set.xyz`) — **EVIDENCE PATH NOT RESOLVED locally**;
  evidence_inventory states it is "on KISTI only" (remote). Held-out `test_set.xyz` basis noted.
- **Student-MD–DFT (channel d)** — "partially available / needs final selection" per inventory.
- **FSDP/S(Q) figure-ready CSV** — not located locally.
- Full remote project ≈ 45 GB (dominated by `dft_labeling/` ≈ 44 GB) — remote.

## 7. Gate evidence

Latest state and history (from `coordination_log.csv`, verified):
- **Latest gate = `v5-committee-ADOPT` PASS, unanimous 3/3, 2026-07-16T15:07** — "v5 … FIRST
  re-distillation to BEAT original on BOTH deployment distribution AND error(c) 0.337<0.368;
  F_RMSE~0.285<0.309; v5 ADOPTED → replaces original student; AL loop CLOSED for this iteration.
  Caveats: 11 AL cells may overlap v5 train; u_max still >0.30 at x≥0.12; clustered_cell hardest."
- Genuine historical REVISE/FAIL (all since remediated → v5): `clustered_cell_001` **FAIL**
  (atom-overlap DFT-carve defect); `v3-redistilled` **FAIL**; `v3-final`/`v3-final-v2` **REVISE**
  (u_max>0.30); `data-provenance` REVISE (split not committed); `error-decomposition` REVISE;
  `committee-uncertainty` REVISE (3.5× unsupported); `er-finetune` REVISE (ckpt mismatch);
  `production-science` REVISE (Spearman sign); `meltquench-protocol` REVISE (cooling-rate bug).
- **No OPEN REVISE/FAIL currently exists.** The last decision on the student is PASS.

## 8. Recovery-trigger candidates

Evaluated against real evidence (score 0–3 each: evidence, gate basis, action clarity,
revalidation feasibility, compute affordability, scientific value):

- **dataset_coverage (committee-uncertainty-driven)** — evidence 3, gate-basis **1 (aspiration,
  not a required-pass criterion)**, action 3, revalidation 2, affordability 2, value 3. Residual:
  v5 committee `u_max` on the 11 AL cells is mean 0.334 / median 0.321 / p90 0.461 (verified in
  `committee_umax_AL11_v5.log`) — above the **0.30 eV/Å paper aspiration** (which the v5 gate
  explicitly treated as non-binding), worst at high defect fraction / clustered cells. The legacy
  log points a remedy path ("AL iter-3, 30–50 cells"). **Strongest candidate for a future
  current-gate test — but not an issued failure.**
- **student_fidelity (student–teacher F-MAE)** — evidence 2, gate-basis 2, action 2. error(c)
  0.337 already beats 0.368; the open issue is *uncertainty/coverage*, not raw fidelity → weaker.
- **physical_validation** — evidence 1. Physical-validation gate PASSED (density 2.21, RDF 1.61,
  CN 4/2, NVE PASS); no open physical failure → weak.
- **simulation_protocol** — evidence 1 (Paper-2 scope). meltquench cooling-rate bug was found and
  accepted/documented; not a student recovery → out of scope here.

## 9. Selected / provisional / no-trigger decision

**NO GENUINE RECOVERY TRIGGER SELECTED YET.**
**LEADING CURRENT-REVALIDATION CANDIDATE: dataset_coverage (committee-uncertainty-driven).**

Rationale: the latest student gate is a PASS (v5 ADOPT 3/3, loop closed); there is no open
REVISE/FAIL. The residual u_max > 0.30 is measured against a *paper aspiration*, which the v5
gate explicitly treated as non-binding (`vote_v5.workflow.js:27,41`), so it is not a
gate-criterion violation and does not by itself authorize recovery. Recovery can only be
triggered after a current-controller baseline is onboarded (§11 / onboarding plan) and a re-run
current three-lens gate actually returns REVISE/FAIL. Until then: candidate, not trigger.

## 10. Recovery compute preflight (read-only; nothing executed)

- Teacher checkpoint readable: **YES** (`teacher/model.nequip.pth`; not loaded).
- Teacher calculator env: **conda env `allegro` present** (not activated/loaded).
- 4-seed student configs/potentials: **YES** (v5 bundle seed01–04).
- SIMPLE-NN env: **conda env `simple-nn` present** (not run).
- Base training dataset readable: **NO — remote (KISTI)**; only `al_iter3/seeds/train_seeds.xyz`
  local.
- `parent_structure_id` completeness: **not verifiable locally** (base dataset remote).
- Current-framework `validation_profile`: **NONE local** (only the toolkit example).
- DFT anchors: base project is DFT-anchor = 0 by design; reference DFT `dft_labeling/` ≈ 44 GB remote.
- Scheduler: **NOT available in this session** (no `squeue`/`sinfo`).
- Output location / storage: not provisioned (no run initialized).

## 11. Missing inputs (to promote PROVISIONAL → SELECTED)

1. A **current-controller authoritative baseline run** registering the v5 committee (run_manifest
   + `code_revision` + artifact hashes) so recovery can hash-bind Iteration 1.
2. **Access to the base teacher training corpus** (KISTI-remote `training_set.xyz`) for
   retrain/relabel, with lineage / `parent_structure_id`.
3. A **frozen current-framework `validation_profile`** encoding the criteria the legacy gate used
   (committee `u_max < 0.30`, error(c) `< 0.368`, F_RMSE, density/RDF/NVE).
4. **Scheduler / compute access** (SLURM or GPU) — absent in this session.

## 12. Cost-relevant facts

- Student is 4-seed SIMPLE-NN (BPNN); retrain cost is the dominant expense; teacher relabeling of
  a 30–50 cell AL increment is comparatively small; no new DFT required for a coverage cycle.
- Legacy AL increment size precedent: "AL iter-3 (30–50 cells)" (from the v3-final-v2 note).
- Full remote data ≈ 45 GB; only summary CSVs + checkpoints are local.

## 13. Recommended next action

Do not start compute. First **register/reproduce a current-controller authoritative baseline**
from the local v5 committee bundle + committee-uncertainty summary, and **obtain remote training
data + a frozen validation_profile**. Then the dataset_coverage trigger can be promoted and one
recovery iteration proposed under the current controller.

## 14. Claims allowed after this audit

- A legacy multi-gate judge pipeline produced a full PASS/REVISE/FAIL trail and closed one
  re-distillation recovery loop (v3 → v5 adopted), verifiable in `coordination_log.csv` /
  `coordination_votes.csv`.
- The adopted v5 4-seed student and the original teacher checkpoint are present locally.
- A genuine, quantified residual coverage gap remains (v5 u_max > 0.30 at high defect fraction).

## 15. Claims still unsupported

- That an authoritative *current-controller* SiO2 run exists (it does not).
- That a current-framework recovery iteration has been executed (none has).
- That the residual coverage gap has been remediated (it is an accepted caveat, not fixed).
- Any numeric baseline reconstructed from manuscript prose rather than the inspected CSVs.
