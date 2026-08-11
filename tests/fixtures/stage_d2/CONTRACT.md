# Stage D-2 C1 — post-hoc MSD execution contract (PREPARATION; not executed)

First real state-advancing Stage D-2 action. Prepared against architecture-v2 freeze HEAD
`99b9e87eacab5762c7f4c04ac8838445e57b2399`. **Nothing is executed; no scientific result files exist.**

## Selected input

- Trajectory: `…/sio2x_production/random_x006/traj.dump`
- SHA256: `53ddba6a02747efb9d545415ec6468ff41c76c5a845f7afdf4cc7e71c3067591` · 25,856,871 bytes
- Ensemble: NVT 300 K, 150 ps; dump every 1000 steps (dt 1 fs) → **1 ps/frame, 151 frames**, 2940 atoms
- Fixed cubic box **L = 35.4975 Å** (L/2 = 17.7487 Å); columns `id type x y z fx fy fz`
- Species: type 1 = O (1940), type 2 = Si (1000); no existing MSD artifact for this trajectory

## PBC sufficiency (critical)

`D2_C1_INPUT_INSUFFICIENT_FOR_VALID_MSD = false` — **with a documented limitation.** The dump is
**wrapped-only** (no `xu/yu/zu`, no image flags). Reconstruction uses **minimum-image continuity
unwrapping** (accumulate per-frame min-image steps). Wrapped-only data **cannot prove** the absence of
a true ≥ L/2 inter-frame jump (a min-image step is ≤ L/2 by construction), so this rests on the
physical assumption that per-frame displacement ≪ L/2 — assured for a 300 K amorphous solid at 1 ps
cadence (thermal motion ~0.1 Å). A deterministic **proxy gate** STOPS the run if the max observed
min-image step ≥ 0.25·L. This is min-image continuity with a declared safety bound + documented
assumption — **not** a silent wrapped-coordinate pseudo-MSD. For a *guarantee*, re-dump with unwrapped
coords / image flags (recommended if a hard guarantee is required).

## Frame selection

All frames if count ≤ 200, else even stride to ≤ 200. Here 151 ≤ 200 → **all 151 frames**. Minimum 10.

## MSD algorithm

Per-atom displacement = cumulative min-image steps from frame 0; MSD(t) = mean over atoms of squared
displacement; also per-type MSD. Derived diagnostics over a **pre-declared** late window (last 30% of
frames): late mean/std/slope, linear-fit R², 3D Einstein diffusion estimate (slope/6). **No plateau or
diffusion threshold is invented** — thresholds would only be authoritative if source-grounded, and
none exists, so the physical interpretation is advisory (Judge).

## A. Authoritative artifact/computation-validity gate (`deterministic_authoritative=true`)

`tests/fixtures/stage_d2/criteria/posthoc_msd_validity.json` — frozen operators over executor-recorded fields;
evaluated by the frozen `criterion_eval` → bound verdict (LLM owns neither the booleans nor the
verdict). Invalidating (→ FAIL on failure): input exists, sha256 matches, parsed, **PBC precondition**,
atom count constant, timesteps increasing, MSD finite, MSD non-negative, initial MSD ≈ 0, source
byte-identical after, writes only under run dir. Completeness (→ REVISE): frame count ≤ 200 and ≥ 10,
output rows match, summary fields present, output sha256 recorded, runtime within ceiling.

## B. Derived diagnostics (data, not pass/fail)

MSD(t), late mean/std/slope, R², D estimate, window/fit definition, frame count, timestep spacing,
species/atom counts.

## C. Advisory semantic Judge gate (`deterministic_authoritative=false`)

`tests/fixtures/stage_d2/judge_interpretation_task.json` — the Judge answers, grounded ONLY in the generated
artifact + declared diagnostics (no fabricated threshold): bounded amorphous-solid behavior over the
window? sustained diffusion? window sufficient or longer analysis required? Genuine semantic verdict
retained (advisory path).

## Artifacts — PLANNED vs GENERATED

| artifact | status |
|---|---|
| `tests/fixtures/stage_d2/action_proposal.json` | **PLANNED** (validated vs frozen AnalystActionProposal) |
| `tests/fixtures/stage_d2/input_manifest.json` | **PLANNED** |
| `tests/fixtures/stage_d2/criteria/posthoc_msd_validity.json` | **PLANNED** |
| `tests/fixtures/stage_d2/judge_interpretation_task.json` | **PLANNED** |
| `tests/fixtures/stage_d2/run_manifest.template.json` | **PLANNED** template |
| `tests/fixtures/stage_d2/<run_id>/approval.json` | GENERATED at approval (human) |
| `tests/fixtures/stage_d2/<run_id>/msd.csv`, `msd_summary.json` | GENERATED at execution |
| `tests/fixtures/stage_d2/<run_id>/criterion_results.json`, `judge_interpretation.json` | GENERATED |
| `tests/fixtures/stage_d2/<run_id>/provenance.json`, `run_manifest.json` | GENERATED |

`msd.csv` columns: `frame_index, timestep, time_ps, msd_all, msd_type1, msd_type2`.
`msd_summary.json`: `n_frames, n_atoms, box_L, dt_ps, species_counts, late_window_frames,
late_window_frac, late_mean_msd, late_std_msd, late_slope, late_fit_r2, diffusion_estimate`.

## Authorization + trusted executor paths

- Proposal role: **analyst**; `action_type = summarize_md_stability` (frozen), `parameters.subtype =
  posthoc_msd`. Approval boundary `stage_d2_first_state_advancing_action` — **explicit human approval
  required**; preparing the proposal is not approval.
- Trusted executor: `runtimes/pydantic_ai/stage_d2_executor.py::run_posthoc_msd` — enforces approval,
  fresh run dir (no overwrite), source allow-list + sha256, PBC proxy gate, CPU-only single-process,
  5-min ceiling, writes only under the run dir. **Not invoked during preparation.**

## Idempotency / mutation / rollback

Source trajectory + entire Stage D-1 tree **read-only**; executor refuses an existing run dir; no
writes outside the run dir; on any failure the run dir is removed and no partial output is accepted as
final.

## Stop conditions

- **ADVANCE** only if: authoritative validity = PASS **and** Judge (advisory) consistent **and**
  provenance complete **and** within ceiling.
- **REVISE** if the validity result is REVISE-class (completeness gap) or the Judge flags the window
  insufficient.
- **STOP** if: PBC proxy gate fails (`STOP_PBC_INSUFFICIENT`); any invalidating validity criterion
  fails → FAIL; run dir exists; sha mismatch; approval absent; ceiling exceeded; write attempted
  outside the run dir.

## Exact action that would occur after explicit approval

Write `tests/fixtures/stage_d2/d2c1-posthoc-msd-random_x006/approval.json`, then run
`stage_d2_executor.run_posthoc_msd` once on the pinned trajectory (CPU, ≤ 5 min): parse → select all
151 frames → verify PBC proxy gate → min-image-continuity MSD(t) + per-type + late-window diagnostics →
write `msd.csv` + `msd_summary.json` → record the axis-A validity fields → evaluate the authoritative
validity gate (frozen `criterion_eval`, bound verdict) → run the advisory Judge interpretation →
write `criterion_results.json`, `judge_interpretation.json`, `provenance.json`, `run_manifest.json`.
No scheduler, no MD/DFT/training/teacher inference, no GPU, no network.
