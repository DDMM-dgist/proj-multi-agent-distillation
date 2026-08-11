# Stage D-2 C1 — FINAL REPORT (post-hoc MSD; first real state-advancing action)

```
STAGE_D2_C1_AXIS_A                    = PASS
STAGE_D2_C1_SEMANTIC_JUDGE_ATTEMPT_1  = INCOMPLETE_REQUEST_LIMIT   (root cause READ_ALLOW_PATH_RESOLUTION_LOOP)
STAGE_D2_C1_SEMANTIC_JUDGE_ATTEMPT_2  = REVISE                     (valid typed + canonical advisory result)
STAGE_D2_C1_SEMANTIC_JUDGE            = REVISE
STAGE_D2_C1_TRANSITION               = REVISE
STAGE_D2_C1                          = AXIS_A_PASS__SEMANTIC_REVISE
```

## Chronology

**C1 scientific action** — approved once; CPU-only post-hoc MSD on `random_x006/traj.dump` (NVT 300 K,
151 frames @ 1 ps, 2940 atoms, fixed cubic L = 35.4975 Å). Deterministic **Axis-A artifact validity =
PASS** (17/17 criteria via the frozen `criterion_eval`; bound verdict). No GPU, no network, no scheduler.

**Execution-provenance caveat** — the orchestration wrapper (`work/stage_d2_execute.py`) was NOT
committed at the approved execution HEAD `b5762a1d`; the exact wrapper snapshot + SHA256 are preserved
(`execution_wrapper_snapshot.py`; `wrapper_committed_at_execution=false`), and the scientific artifacts
were validated and retained. Recorded as `AXIS_A_PASS_WITH_EXECUTION_WRAPPER_PROVENANCE_CAVEAT`.

**Semantic Judge — Attempt 1** — `INCOMPLETE_REQUEST_LIMIT`, root cause `READ_ALLOW_PATH_RESOLUTION_LOOP`:
bare evidence filenames resolved against repo-root CWD, outside the run-dir read allow-list → 6 refused
reads → `UsageLimitExceeded`. No scientific verdict produced. Preserved exactly, never relabeled.

**Semantic Judge — Attempt 2** — path fix (full repo-relative evidence paths + no-manifest instruction)
succeeded: 2 tool reads, both OK on the repo-relative paths, no bare paths, no manifest, no refused
loop, no `UsageLimitExceeded`. Valid typed JudgeVote; **canonical validation passed**; **verdict
REVISE**; one attempt, no retry; `controller_mutation=false`; no scientific-artifact mutation.
provider=local-openai, model=qwen2.5-7b-instruct, usage_source=provider (local loopback only).

## Final

```
STAGE_D2_C1_AXIS_A       = PASS
STAGE_D2_C1_SEMANTIC_JUDGE = REVISE
STAGE_D2_C1_TRANSITION   = REVISE
STAGE_D2_C1             = AXIS_A_PASS__SEMANTIC_REVISE
```

Stage D-2 C1 does **not** ADVANCE: the artifact is deterministically valid, but the genuine advisory
semantic Judge returns REVISE. The advisory verdict is NOT rebound to the Axis-A PASS.

## Scientific interpretation scope (from the Attempt-2 rationale, grounded in the recorded diagnostics)

- The observed 150-ps MSD is compatible with **bounded / solid-like** behavior over the observed window
  (MSD_all(final) 0.0861 Å²; late mean 0.0922 Å², std 0.00368 Å²).
- **No sustained diffusion** is evident over that window (late slope 5.74e-6 Å²/ps, R² 4.3e-4).
- A stronger **long-time transport / rigorous self-diffusion** interpretation is **not** sufficiently
  supported by this artifact alone (short window + variation → REVISE).
- `pbc_hard_guarantee = false`; wrapped x/y/z only; no xu/yu/zu, no image flags; minimum-image
  continuity assumption applies (proxy max min-image step 1.655 Å, max_step/L 0.0466 — plausibility,
  not proof).
- `apparent_D_estimate_under_continuity_assumption = 9.57e-7 Å²/ps` is **apparent only, not a rigorous
  self-diffusion coefficient** (the Judge stated this explicitly).

## Future remediation (recorded; NOT authorized)

- A longer trajectory / analysis window.
- Preferably a trajectory dumped with **unwrapped coordinates (`xu/yu/zu`) or image flags** so PBC
  displacement reconstruction has a hard guarantee rather than the continuity assumption.

## Append-only provenance integrity

Additive Attempt-2 closure artifacts (generated DETERMINISTICALLY from the preserved Attempt-2 exchange
provenance `02d531f6`, no LLM): `judge_interpretation_attempt2.json`, `judge_provenance_attempt2.json`,
`semantic_transition_attempt2.json`, `run_manifest.after_judge_attempt2.json`. Preserved byte-identical:
`msd.csv`, `msd_summary.json`, `criterion_results.json`, `approval.json`, `execution_wrapper_snapshot.py`,
the historical DEFERRED `judge_interpretation.json`, both exchange provenances (`2fe2fc26` + `02d531f6`),
and the original `run_manifest.json`. No Stage D-1 artifact modified. No inference occurred in closure.
