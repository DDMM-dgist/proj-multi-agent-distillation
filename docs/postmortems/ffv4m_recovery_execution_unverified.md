# Postmortem — ffv4m: FINAL_CAMPAIGN_SOFTWARE_INVARIANT_FAILURE

- **Run:** `sio2-sox-allegro-simplenn-ffv4m`
- **Classification:** `FINAL_CAMPAIGN_SOFTWARE_INVARIANT_FAILURE`
- **Failed stage:** `data_coverage` (Stage 4)
- **Failure mode:** `RECOVERY_EXECUTION_UNVERIFIED` (campaign exit_code 10)
- **Nature:** software / recovery-contract invariant failure — **NOT** a scientific REVISE/FAIL, **NOT** a transient/infra error.
- **Disposition:** permanently stopped, preserved byte-immutable as diagnostic evidence. Never to be resumed (final-run immutability). This postmortem is written **outside** the run directory.

## Timeline (KST, 2026-08-25)

| Time | Event |
|---|---|
| — | Stage 1 `teacher_baseline` → 3/3 PASS (build_teacher_baseline, GPU1, 9295 frames) |
| — | Stage 2 `reference_validation` → PASS (validate_teacher_reference, exact-action approved) |
| 10:09:34 | Stage 3 `acquisition` → executed; 3/3 PASS |
| 10:10:57 | Stage 4 `data_coverage` gate → **3/3 REVISE** (`gate_recorded decision=REVISE`) |
| 10:10:57–11:26 | `recovery_started` → analyst `recovery_diagnosis` → orchestrator `recovery_plan_proposal` → `recovery_proposed` (id=1) |
| ~10:12 | `approve-recovery` (id=1) granted → `approval_granted` |
| 10:13:04 | recovery corrective action executed (`EXECUTED`), re-registered `data_coverage.json` |
| 10:13:07 | data_coverage stage re-run producer started |
| 10:13:15 | producer returned **`status=DUPLICATE, accepted=false`** → `campaign_paused outcome=RECOVERY_EXECUTION_UNVERIFIED exit_code=10` |

## Exact failure capture

### Original data_coverage gate result
- Verdict: **3/3 REVISE** (all three lenses).
- Votes file integrity: size `25060`, sha256 `0cf3ddd29d918391a2275a1c668d94312ba2dcb007bc2b7e6b2140b1da5871d4`.
- Gate-bound artifact: `artifacts/data_coverage.json` sha256 `6350d02d1a1369a65000afdc89056e87a962c39d4b77625d15af32431334f1d2`.

### RootCause diagnosis (analyst)
- File sha256: `d3d298b2b8db42f17bd1cddabbf686a04121f984d89905b947db676597933dd2`.
- `failure_category = evidence_gap`; `failure_domain = insufficient_evidence`; `confidence = 0.99`.
- Explicitly **excluded** `dataset_coverage` (scientific coverage inadequacy), `reference_disagreement`, `student_fidelity`, `data_quality`. The artifact passed general deterministic manifest validation; the REVISE was a **provenance/criterion-level evidence-exposure gap**, not a numerical/accuracy failure.

### RecoveryPlan identity
- File sha256: `4b702b553dbfc5aff0d253d0b68b052287f39e939f1bb444420d98c51a9fe768`.
- `diagnosis_artifact_sha256 = d3d298b2…`; recovery `id=1`; `capability = data_repair`; `responsible_agent = data-curator`; `return_stage = data_coverage`.
- `labeling.teacher_relabel=false`, `labeling.new_dft=false`, `student_training.retrain=false` — declared cheap/reversible.
- **Corrective `action_type = build_data_coverage_report`.**

### required_evidence requested by the recovery
1. geometry-only Teacher-training access and limitations;
2. all declared coverage dimensions by config_type, with exact `NOT_ASSESSABLE` where unsupported;
3. acquisition-manifest identity/hash and deterministic lineage match;
4. Teacher-test exclusion before candidate-pool statistics, with identifiers/hashes + pre/post counts + zero-overlap result;
5. authoritative criterion-specific deterministic check results.

### Artifact hashes across the recovery
- Baseline `data_coverage.json` sha256: `6350d02d1a1369a65000afdc89056e87a962c39d4b77625d15af32431334f1d2`.
- Regenerated `data_coverage.json` sha256: `6350d02d1a1369a65000afdc89056e87a962c39d4b77625d15af32431334f1d2` — **byte-identical**.
- Executor result: **`DUPLICATE`** (`accepted=false`) at `2026-08-25T01:13:15Z`.

### Recovery verification failure
`RECOVERY_EXECUTION_UNVERIFIED` (exit 10): *"no artifact at or downstream of return_stage 'data_coverage' has changed since the recovery baseline."* The hash-change verification invariant fired correctly and refused to let the failed gate re-pass.

### Confirmations
- **No scientific reacquisition occurred.** Recovery capability was `data_repair` (report regeneration), not `simulation_rerun`/acquisition. Acquisition artifacts (`acquisition.manifest.json`, `acquisition_candidates.extxyz`) are unchanged from 10:09:33–34; no `acquire_structures` re-executed; the bound AcquisitionPlan was **not** superseded and N/composition/strategy were **not** altered.
- **No costly downstream action executed.** Executors ran only for `teacher_baseline`, `reference_validation`, `acquisition`, `data_coverage`. `teacher_labeling` / `train_committee` / `deployment_md` / `physical_validation` / DFT never ran. Only two costly approvals were ever granted (both exact-action, non-transitive): `build_teacher_baseline`, `validate_teacher_reference`.

## Root cause (software / recovery-contract)

The `data_repair` RecoveryPlan dispatched corrective `action_type = build_data_coverage_report` — **the same deterministic executor** that produced the original report. That executor has **no typed channel to consume the recovery's declared `required_evidence`**, so re-running it regenerates a byte-identical artifact (`DUPLICATE`). The recovery-execution-verification invariant requires a changed artifact hash at/downstream of `return_stage` to prove the corrective action did real work; a deterministic no-op can never satisfy it. Result: an unbreakable `DUPLICATE → RECOVERY_EXECUTION_UNVERIFIED` condition.

Contributing structural gap: the `recovery_capability_roster` maps capability→agent only (e.g. `data_repair → data-curator`) and does **not** declare which state/evidence transitions each capability's executor can actually materialize. There is no pre-acceptance compatibility check that a plan's `required_evidence` / requested state change is producible by the chosen corrective capability+executor before approval/dispatch.

## Generic invariant to enforce (fix target)

> A RecoveryPlan must never dispatch a corrective action whose executor cannot consume and materialize the recovery's declared `required_evidence` (or otherwise produce the state transition that recovery-execution verification requires). If unsupported, the framework must fail closed or route to a genuinely capable recovery capability — never dispatch a deterministic no-op repair.

This defect was discovered by a real fresh campaign. The fix is generic and lives entirely **outside** ffv4m; ffv4m is not resumed. Corrected framework is validated by a new regression that reproduces this exact failure class, then a fresh run `ffv4n` is created.
