# Teacher training split — provenance record

This directory is a **durable, version-controlled, run-independent** record. It
is not bound to any `configs/runs/*` run and must not be edited to reflect a
specific run's state. If a claim here needs correction, correct it here
directly (new evidence, new note) — do not fork a run-specific copy.

It answers one question with reproducible evidence: which of the 11,424 frames
in `seed_pool_11424` were in the original Allegro Teacher's train / validation
/ test partitions on KISTI, and what does that imply for evidence that reuses
those frames (e.g. `teacher_baseline_slice_manifest.json`'s 2,134-frame
operational baseline, which is byte-identical across every run r2–r11 that has
been checked, so nothing here binds to r11 specifically — the manifest content
is a repo-level, run-independent artifact, only its *location* happens to live
under a run directory).

## Chain of custody (hashes computed directly in this repo/session)

| Artifact | Path | sha256 |
|---|---|---|
| Recovered source dataset | `gpu_finetune_handoff/kisti_assets/kisti_pack/dataset.xyz` | `382d0b2b35ed9c571314ff59df71e9c989397ad7360735e3dc4c12b8b6bcabd4` |
| Recovered bundle archive | `gpu_finetune_handoff/kisti_assets/kisti_pack.tar.gz` (per bundle's own `kisti_pack.sha256`) | `c23a3d8fe485e1f21044c5b2265abb1b2f954893aa0c281f96a5fa683a422105` |
| Recovered resolved Hydra config | `gpu_finetune_handoff/kisti_assets/kisti_pack/teacher_run/.hydra/config.yaml` | `984a025c8beece593748100abc9f1ecce63bda989b252c11eabbb943ab4cb499` |
| Recovered Lightning checkpoint | `gpu_finetune_handoff/kisti_assets/kisti_pack/teacher_run/best.ckpt` | `51342b332ba04287df54e349e6fc5ac9f1409180fd619be462fb73ebe2771703` |
| Deployed compiled Teacher | `local_inputs/sio2_fresh/teacher/teacher_current_compiled.nequip.pth` | `b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57` |
| This split manifest | `configs/provenance/teacher_training_split_manifest.json` | recompute at read time — content is regenerated, not append-only |

`dataset.xyz` was independently, exhaustively (all 11,424 frames, not sampled)
content-cross-checked against this repo's own `local_inputs/sio2_fresh/seed_pool_11424/`
per-category files (natoms + energy match, 0 mismatches) — this is what
licenses treating `seed_pool_11424` and the recovered `dataset.xyz` as the same
underlying frame pool.

`best.ckpt`'s EMA parameter group (25 tensors, 1,081,946 values) is bit-exact
(`torch.equal()` true, max abs diff 0.0, all 25/25 tensors) against the deployed
compiled Teacher's `named_parameters()`. This is a direct tensor-value proof of
linkage, not a metadata-only inference.

## Split reconstruction

- **Algorithm**: `nequip.data.dataset.utils.RandomSplitAndIndexDataset` (nequip
  0.15.0/0.16.1, read verbatim from the installed package source), which calls
  the real, unmodified `torch.utils.data.random_split(dataset, lengths, generator)`.
  This reconstruction calls that same real torch function — it does not
  reimplement its rounding/assignment behavior.
- **Seed / order / fractions**: seed `123`, subset order `[train, val, test]`,
  fractions `[0.8, 0.1, 0.1]` — all read directly from
  `.hydra/config.yaml` (`data.seed`, `data.split_dataset`).
- **Result**: 9,140 / 1,142 / 1,142 frames (train/val/test), summing to 11,424.
- **Reconstruction scripts** (copied here verbatim from the working session,
  not rewritten): `scripts/parse_and_split.py` (frame-order + category
  cross-verification against `seed_pool_11424`), `scripts/do_split.py` (the
  actual split call), `scripts/overlap.py` (overlap of the 2,134-frame
  operational baseline against the reconstructed partitions). These scripts
  use `/tmp/kisti_verify/*.json` as intermediate scratch files between steps;
  those intermediates are not themselves preserved, only their final outputs
  (folded into `teacher_training_split_manifest.json`).

### PyTorch-version caveat — not resolved, only disclosed

Local reconstruction ran under torch `2.12.1+cu130`. KISTI's original training
ran under torch `2.6.0+cu124` (`kisti_pack/env/versions.txt`). Whether
`torch.utils.data.random_split`'s CPU-generator RNG stream is bit-identical
across that version range has **not been empirically tested** — it is
plausible (PyTorch does not advertise breaking CPU-generator-stream changes
between these versions) but unconfirmed. The 9,140/1,142/1,142 membership
recorded here should be read as *independently re-derived from the algorithm,
seed, fractions, and order recovered from source* — not as *empirically proven
identical* to KISTI's actual run-time index assignment.

A concrete, low-cost way to close this gap later: re-run only
`scripts/do_split.py` (a pure index-permutation call over `range(11424)`, no
model, no Teacher inference) inside an actual `torch==2.6.0+cu124`
environment, and compare the resulting index-set hash against this manifest's.
`torch==2.6.0` is confirmed available from PyPI's version index
(`pip index versions torch` lists it) and cu124 wheels are available from
`download.pytorch.org`'s index per `KISTI_REQUIREMENTS.md`'s Item 3 —
so building that environment is feasible, but has not been done; no such
environment has been created in this repo, and this file records only that a
gap exists, not that it has been closed.

## Checkpoint / test-partition role — verified, not assumed

Read directly from `.hydra/config.yaml`:

- `monitored_metric: val0_epoch/weighted_sum` (line 9) — a **validation**-partition
  metric.
- `EarlyStopping` (lines 52-55), `ModelCheckpoint` (lines 56-60, the callback
  that produces `best.ckpt`), and the `ReduceLROnPlateau` LR scheduler
  (line 89) all monitor `${monitored_metric}`, i.e. the same validation metric.
- No callback active during training references any test-partition metric.

Read directly from the SLURM stdout log (`teacher_run/allegro_581561.out`):

- Exactly one `TEST RUN START` / `TEST RUN END` block appears, at
  2025-11-13 23:04:19 to 23:04:37 — after training had already finished. This
  is Lightning's one-shot `trainer.test()` call; the reported `test0_epoch/*`
  metrics were produced once, for reporting, and never fed back into training.

**Conclusion**: the **validation** partition (1,142 frames, including 202 of
the 2,134 operational-baseline frames) directly influenced checkpoint
selection, early stopping, and the LR schedule — it is not independent
generalization evidence. The **test** partition (1,142 frames, including 208
of the operational-baseline frames) was touched exactly once, after training
concluded, solely for aggregate metric reporting, and was never used in any
training-time decision — these 208 frames constitute genuine held-out
generalization evidence (in the sense of being absent from every training-time
signal; this does not certify that the historical aggregate `test0_epoch/*`
metrics are re-derivable per-frame from this manifest). The 1,724
operational-baseline frames in the **train** partition are training data and
carry no generalization-evidence value at all.

| Partition | Frames | Operational-baseline overlap | Role in checkpoint/LR/early-stop selection | Generalization evidence? |
|---|---|---|---|---|
| train | 9,140 | 1,724 | direct (loss) | No |
| validation | 1,142 | 202 | direct (`monitored_metric`) | No |
| test | 1,142 | 208 | none (evaluated once, post-hoc) | Yes |

## What this file does NOT do

- Does not mutate, re-run, or bind to R11 or any other specific run.
- Does not initialize R12.
- Does not run Teacher inference.
- Does not generate acquisition structures.
- Does not select any coverage threshold or acquisition count.
