# EXTERNAL GPU — Target-focused teacher fine-tune (PC-A1)

**One controlled experiment.** Warm-start the SiO₂ Allegro teacher from its base
checkpoint and fine-tune it on a target-enriched corpus to lower the **held-out
oxygen-deficient (dilute + clustered) force floor without degrading amorphous**,
while replaying broad-domain structures to prevent catastrophic forgetting.
Architecture is UNCHANGED; only the training emphasis changes. **No new DFT.**
This is *not* a hyperparameter sweep and *not* a higher-capacity model (that is
only a documented fallback — see the top-level `EXTERNAL_GPU_TARGET_TEACHER_TRAINING_PACKAGE.md`).

This package was prepared on a CPU-only server; **all training happens here on the
GPU server.**

---

## 0. Contents
```
README.md                     ← this runbook
ENVIRONMENT.md                ← exact package versions (nequip 0.15.0 / allegro 0.7.1 / torch 2.6.0)
base_teacher/best.ckpt        ← warm-start source (sha256[:16] 51342b33; 17.8 MB)
dataset/
  fine_tune_corpus.xyz        ← 3,766 frames = 2,966 target core + 800 replay (TRAINING INPUT)
  fine_tune_train.xyz         ← 3,389 frames (reference deterministic split)
  fine_tune_val.xyz           ←   377 frames (reference deterministic split; used by verify_and_eval.py)
configs/
  finetune_allegro_sio2x_targetfocus.yaml
manifests/
  frame_manifest.csv          ← every frame: dataset_index, config_type, domain, role, ft_split, nSi/nO, content_sha
  selection_report.json       ← core/replay counts + per-family replay allocation
  source_provenance.json      ← dataset SHA, seed-123 split, base-teacher SHA
scripts/
  run_training.sh             ← launcher (env check + nequip-train)
  compile_model.sh            ← compile best.ckpt → deployable .nequip.pth
  verify_and_eval.py          ← GPU-side sanity fidelity on the internal val set
```

## 1. Resolve-first (confirm before launching)
1. **Env** matches `ENVIRONMENT.md` (nequip 0.15.0, allegro 0.7.1, torch 2.6.0). A 0.16.x
   nequip will NOT load `base_teacher/best.ckpt`.
2. **Warm-start mechanism.** The config uses `nequip.model.ModelFromCheckpoint`
   pointing at `base_teacher/best.ckpt`. If your nequip build lacks it, launch with
   the CLI checkpoint path instead:
   `nequip-train configs/finetune_allegro_sio2x_targetfocus.yaml fit.ckpt_path=base_teacher/best.ckpt`
   (that path also restores optimizer state — keep `optimizer.lr=0.001` explicit).
3. **Stats.** With `ModelFromCheckpoint` the per-type energy shifts/scales come from
   the checkpoint. Do NOT recompute energy shifts on the fine-tune corpus.
4. **Operative split.** Training uses nequip's own `seed=123` 0.9/0.1 split of
   `fine_tune_corpus.xyz` (deterministic given the fixed file + seed). The separate
   `fine_tune_train/val.xyz` + the manifest `ft_split` column are the CPU-side
   reference split; you do not need to wire them in unless you prefer explicit files.

## 2. Run
```bash
cd <package_root>
bash scripts/run_training.sh              # env check → nequip-train
# checkpoints land in outputs/<date>/<time>/best.ckpt , last.ckpt
```
Deliberate settings (vs base): optimizer `lr 0.001` (10× lower), loss `forces:total_energy = 4:1`
(base 1:1), `EarlyStopping patience 40 / min_delta 1e-4`, `max_epochs 400`, `max_time 1 day`.

## 3. GPU requirements & walltime estimate
- **1 GPU** (A100/H100/RTX-class ≥16 GB). Single-device (see the CUDA-index gotcha in `ENVIRONMENT.md`).
- **Estimate ~6–16 h**, expected to early-stop before the 1-day `max_time`. Basis: the
  v6 ER-FT ran 158 epochs on 157 frames in **26m54s** (~0.17 min/epoch); scaling
  linearly to this corpus (≈3,389 train frames) gives ~3–4 min/epoch, and the base run
  converged by ~epoch 158 → order ~10 h. Treat as an estimate, not a guarantee.

## 4. Compile the result
```bash
bash scripts/compile_model.sh outputs/<date>/<time>/best.ckpt teacher_targetfocus_ft.nequip.pth
# prints sha256 and does a NequIPCalculator load check
```

## 5. Verify successful completion (GPU-side sanity)
```bash
python scripts/verify_and_eval.py teacher_targetfocus_ft.nequip.pth
```
Confirms the model loads and reports force MAE per domain on the internal val set.
**This is a sanity check only.** Acceptance is decided on the CPU workflow by the
authoritative **373-frame target-domain held-out screen** (§7).

Success = (a) `best.ckpt` written, (b) test/val `forces_mae` finite and ≤ the base
teacher's on the internal val, (c) no NaN/crash in the log, (d) compiled model loads.

## 6. Copy back to the CPU workflow (minimum set)
Return these — nothing else (no giant caches):
- `teacher_targetfocus_ft.nequip.pth`  (compiled teacher)  + its **sha256**
- `outputs/<date>/<time>/best.ckpt`  (Lightning checkpoint)
- the resolved training config actually used
- the full training log (stdout + `csv_logs/.../metrics.csv` or wandb summary)
- final train/val/test metrics (energy_mae, forces_mae)
- a one-line run manifest: git/pkg hash, env versions, GPU, walltime, epochs, early-stop reason

## 7. What happens next on the CPU side (do NOT do this on the GPU)
The returned teacher is registered (model + SHA), then **screened on the exact 373
target-domain held-out frames** (amorphous 140 / dilute 149 / clustered 84), reusing
the base metrics. Advancement rule (no invented threshold): advance only if it
**clearly improves dilute and/or clustered without degrading amorphous**; if so, run
the full 1,142-frame held-out validation and select it as the final teacher. If not,
`b56e20ff` is retained as the resource-constrained baseline with its limitation recorded.
