# PC004 — Student Validation Protocol (PREPARED, teacher-weight-independent infra)

**State: `PC004 = PREPARED`.** The validation *infrastructure* is defined now and does not require
final teacher weights; only the numeric execution waits for a trained student + final teacher.

## Two validation axes
1. **Student vs Teacher** (distillation-fidelity): does the student reproduce the teacher it was
   distilled from? Metric = force/energy error of student vs teacher on held-out structures
   (matches the historical proxy: SIMPLE-NN valid-set F RMSE, v6 = 0.284–0.289 eV/Å).
2. **Student vs DFT** (absolute accuracy): student vs the **11 preserved SCAN DFT cells**
   (`scan_labeled_structures/sio2x_AL_labels_11cells.xyz`, `dft_exclusion=YES_never_train`).
   This is the only committed absolute ground truth; it is the reason those 11 cells are frozen out
   of PC002/PC003 training.

## Metrics (per axis)
- **Energy:** per-atom energy MAE (meV/atom); report convention explicitly.
- **Force:** component MAE + RMSE (eV/Å); per-atom vector error; normalized (err / DFT-force-scale)
  so absolute-scale differences don't masquerade as fidelity loss.
- **Domain-resolved:** amorphous / dilute / clustered separation (the exact three central domains),
  never a single aggregate — the whole point is the oxygen-deficient regime.

## Leakage prevention (hard rule)
- The 11 SCAN DFT cells are in **no** student training or validation-tuning set.
- Student-vs-Teacher held-out frames come from PC002's val stratum (structure-level split, no
  trajectory shared with train).
- The reconstruction uses the SAME exact-split / normalized-fidelity machinery already proven in
  PC001 (`runs/production_campaign_001/…`), so definitions are identical across teacher and student
  evaluation.

## Deterministic PASS / REVISE gates
- Enacted by the frozen judge-committee pattern (3 independent judges vote on the CSVs, not the
  prose), re-vote after fixes — same gate the runtime already owns.
- Gate inputs are the domain-resolved metric CSVs; thresholds are recorded per-run (not invented
  here — set when a concrete student exists, against the teacher's own held-out numbers as the
  reference ceiling).

## Reusable machinery (already in-repo)
- `runs/production_campaign_001/pc001-teacher-exact-heldout-validation/` (exact-split + per-frame
  metric definitions) → reuse verbatim for the student.
- `work/pc001_force_scale_normalized.py` (normalized fidelity) → reuse for student-vs-DFT.

## Blocked-on
A trained student committee (PC003) + final teacher identity. Infra, metric definitions, split
logic, and gate wiring are ready now.
