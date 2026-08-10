# TRACK A — External-GPU Target Teacher Training Package (PREPARED, NOT EXECUTED)

**State: `TRACK_A = EXTERNAL_GPU_TRAINING_PREPARED`.** No Allegro training, fine-tuning,
or GPU job was launched on this CPU server. This document + the portable package under
`work/external_gpu_teacher_package/` are the complete deliverable; the actual training
runs later on a separate GPU-capable server.

Prerequisite established: the existing-teacher screen (`work/EXISTING_TEACHER_CANDIDATE_SCREEN.md`)
closed with **`NO_EXISTING_TEACHER_IMPROVES_TARGET_HELDOUT_FIDELITY`** (base / v2 / v6 all
`DO_NOT_ADVANCE`), and the root cause is `A_MODEL_TRAINING_UNDERFIT` (held-out defect force
floor, train≈test). A new controlled training experiment is therefore the only way to test
whether that floor is movable.

---

## A. Portable package contents
`work/external_gpu_teacher_package/` (104 MB total; integrity in `manifests/SHA256SUMS.txt`):

| path | what |
|---|---|
| `README.md` | GPU-server runbook (env → run → compile → verify → copy-back) |
| `ENVIRONMENT.md` | exact stack: nequip 0.15.0 / allegro 0.7.1 / torch 2.6.0+cu124 |
| `base_teacher/best.ckpt` | warm-start source, sha256[:16] `51342b33` (17.8 MB) |
| `dataset/fine_tune_corpus.xyz` | **training input**, 3,766 frames, sha256[:32] `46433913…` |
| `dataset/fine_tune_val.xyz` | 377-frame internal-val reference (used by `verify_and_eval.py`) |
| `configs/finetune_allegro_sio2x_targetfocus.yaml` | the fine-tune config |
| `manifests/frame_manifest.csv` | per-frame: dataset_index, config_type, domain, role, ft_split, nSi/nO, content_sha |
| `manifests/selection_report.json` | core/replay counts + per-family replay allocation |
| `manifests/source_provenance.json` | dataset SHA, seed-123 split, base-teacher SHA |
| `scripts/{run_training.sh,compile_model.sh,verify_and_eval.py}` | launcher / compile / GPU-side sanity |

## B. Exact training-package data provenance
- **Source**: `03_allegro_train/dataset.xyz`, sha256[:32] `382d0b2b…`. Only the **first 11,424
  frames** (the original training pool; exact NequIP test-metric reproduction confirmed) are used;
  later-appended frames are excluded.
- **Split**: `torch.utils.data.random_split(range(11424),[0.8,0.1,0.1], Generator.manual_seed(123))`
  — identical to the base `tutorial_Allegro.yaml` `data.seed=123`.
- **Labels**: existing DFT (`energy=`/`forces`, = `dft_free_energy`) round-tripped through ASE with
  the standard keys — byte-consistent with how nequip read the original. **No new DFT.**
- **Leakage guard** (asserted at build time): the corpus is disjoint from the original VAL (1,142)
  and TEST (1,142) splits. Nothing the held-out screen scores is in training.

## C. Target / replay frame counts (corpus = 3,766)
**Core = 2,966** — every central-domain frame of the seed-123 TRAIN split (no subsampling):
| domain | frames | raw families (difficulty per A3) |
|---|---:|---|
| amorphous_SiO₂ | 1,067 | bulk_amo, quench, quench_int_AL, liquid |
| SiO2x_dilute_vacancy | 1,192 | **SiOx_int_AL** (hard), **vacancy_int_AL**, vacancy (easy) |
| SiO2x_clustered_vacancy/void | 707 | **SiOx_max_AL**, **quench_max_AL**, surfaces_max_AL |

**Replay = 800** — deterministic sqrt-stratified even-spacing over sorted dataset_index per
non-central TRAIN family (min 2, cap = family size); target 800, hit exactly. Covers the broad
domains present in the original pool (anti-forgetting): crystalline SiO₂ `bulk_cryst` 142 +
`bulk_cryst_hp` 56, elemental-Si `silicon_crystalline_main` 97 / `silicon_others` 62 /
`silicon_defects` 57 / `silicon_bulk_amo` 34 / `silicon_liquid` 23 / `silicon_surfaces` 40,
surfaces `surfaces` 68 / `surfaces_int_AL` 35, high-pressure `highpressure_int_AL` 57 /
`highpressure_max_AL` 35, `cluster` 66, `SiOx_crystal_amorphous_interfaces` 28. (Elemental-Si
allotrope families `dia/bt/sh/...` are NOT in the original 11,424 pool — they were in later-appended
frames — so they are correctly absent.)

**Enrichment is by composition** (79 % of the corpus is target domain, and the hard raw families
`SiOx_int_AL`/`vacancy_int_AL`/`SiOx_max_AL`/`quench_max_AL` are included in full) — **not** by a
domain-name weighting scheme (per A3). Internal val = 377 (deterministic, stratified reference;
GPU run uses nequip's own seed-123 0.9/0.1 split of the corpus).

## D. Proposed Allegro fine-tuning config (one controlled experiment)
Warm-start from `base_teacher/best.ckpt` via `nequip.model.ModelFromCheckpoint`; **architecture
copied EXACTLY from the base** (`tutorial_Allegro.yaml`): r_max 5.0, l_max 2, 4 layers, 256 scalar
/ 128 tensor features, 32 Bessels, radial_chemical_embed_dim 256, allegro MLP 2×128, readout 1×32,
ZBL, EMA 0.999, float32. **Only the optimization emphasis changes:**

| knob | base | fine-tune | why |
|---|---|---|---|
| optimizer LR | 0.01 | **0.001** | 10× lower — standard warm-start fine-tune |
| loss `forces:total_energy` | 1:1 | **4:1** | stronger force emphasis (force fidelity is the target) |
| data composition | full corpus | **target-enriched (79 % defect) + replay** | concentrate capacity on the underfit domains |
| EarlyStopping | patience 20 | **patience 40, min_delta 1e-4** | protect the base; stop when val plateaus |
| max_epochs / max_time | 300000 / 3 d | **400 / 1 d** | fine-tune is short |
| ReduceLROnPlateau | 0.5 / p5 / 1e-6 | 0.5 / p10 / 1e-6 | gentler decay for the short horizon |

Evidence basis (A3): original `tutorial_Allegro.yaml` (arch + base optimizer), the older
`gpu_finetune_handoff/config/finetune_allegro_sio2x.yaml` (LR 0.001 + tighter early-stop pattern),
and the v2/v6 ER-FT summaries (both re-anchored energy/defect-core but did **not** move held-out
force fidelity → this experiment instead re-weights forces + enriches the hard raw families rather
than repeating an energy/anchor-centric fine-tune).

## E. External GPU requirements / estimated cost
- **1 GPU** (A100/H100/RTX ≥16 GB), single-device (CUDA-index gotcha — see ENVIRONMENT.md).
- **~6–16 h wall**, expected early-stop before the 1-day cap. Basis: v6 ER-FT = 158 epochs / 157
  frames in **26m54s** (~0.17 min/epoch) → scaled to ~3,389 train frames ≈ 3–4 min/epoch; base
  converged ~epoch 158 ⇒ order ~10 h. Estimate, not a guarantee.
- Disk: negligible (corpus 79 MB, ckpts ~18 MB each).

## F. Exact command that will be run externally
```bash
cd external_gpu_teacher_package
bash scripts/run_training.sh                                   # → nequip-train configs/finetune_allegro_sio2x_targetfocus.yaml
bash scripts/compile_model.sh outputs/<date>/<time>/best.ckpt teacher_targetfocus_ft.nequip.pth
python scripts/verify_and_eval.py teacher_targetfocus_ft.nequip.pth
# copy back: teacher_targetfocus_ft.nequip.pth (+sha256), best.ckpt, resolved config, full log, metrics, run manifest
```
(If `ModelFromCheckpoint` is unavailable: `nequip-train configs/finetune_allegro_sio2x_targetfocus.yaml fit.ckpt_path=base_teacher/best.ckpt`.)

## A6. Fallback — DO NOT prepare/launch yet
IF this target-focused **same-architecture** fine-tune fails to clearly improve held-out
dilute/clustered force fidelity without degrading amorphous, THEN the next possible external
experiment is a **target-focused higher-capacity Allegro** (e.g. larger l_max / more layers /
wider features, trained from scratch or warm-started as feasible). It is **documented only** here;
no config, package, or job for it has been prepared, and no sweep is planned. Decide it only after
the first experiment's result returns and is screened.
