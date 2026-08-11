#!/usr/bin/env bash
# TRACK A / PC-A1 — compile the fine-tuned Lightning checkpoint to a deployable
# ASE .nequip.pth (same command family used to deploy the base + v6 teachers).
# Usage:  bash scripts/compile_model.sh <path/to/best.ckpt> [out.nequip.pth]
set -euo pipefail
CKPT="${1:?usage: compile_model.sh <best.ckpt> [out.nequip.pth]}"
OUT="${2:-teacher_targetfocus_ft.nequip.pth}"

# NOTE: confirm flag order against your installed nequip 0.15.x (this is the
# form used to deploy teacher_v6_finetuned):
nequip-compile --mode torchscript --device cpu --target ase "$CKPT" "$OUT"

echo "compiled: $OUT"
sha256sum "$OUT"
# quick load check
python - "$OUT" <<'PY'
import sys
from nequip.ase import NequIPCalculator
c = NequIPCalculator.from_compiled_model(sys.argv[1], device="cpu",
        chemical_species_to_atom_type_map={"O":"O","Si":"Si"})
print("NequIPCalculator load OK")
PY
