#!/usr/bin/env bash
# TRACK A / PC-A1 — target-focused warm-start fine-tune launcher.
# Run ON THE GPU SERVER, from the package root:  bash scripts/run_training.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# --- env sanity (must be the base TRAINING stack, NOT 0.16.x) ---
python - <<'PY'
import nequip, torch
print("nequip", nequip.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert nequip.__version__.startswith("0.15"), "Expected nequip 0.15.x (base training stack). Abort and fix env."
assert torch.cuda.is_available(), "No CUDA device visible."
PY

# offline W&B by default (no network dependency); override with WANDB_MODE=online
export WANDB_MODE="${WANDB_MODE:-offline}"

# Single-GPU mask recommended (this allegro build silently returns garbage on a
# non-zero CUDA index while other GPUs are visible — see package README).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[run] nequip-train configs/finetune_allegro_sio2x_targetfocus.yaml"
nequip-train configs/finetune_allegro_sio2x_targetfocus.yaml
echo "[run] done. Checkpoints under the hydra output dir (outputs/<date>/<time>/)."
