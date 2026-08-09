#!/usr/bin/env python3
"""Stage D-2 C3 MODEL-LOAD-ONLY preflight (reproducible). Loads the compiled teacher via the TRUSTED
adapter and reports metadata + device compatibility. NO structure is fed through the model; NO forward
pass; NO E/F artifact is produced. Run in the real nequip/allegro env; on a GPU host pass --device
cuda:<id> to also confirm GPU load compatibility (still NO forward pass).

  conda run -n allegro python work/stage_d2_c3_model_load_preflight.py [--device cpu|cuda:1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai.stage_d2_c3_teacher_adapter import TrustedAllegroAdapter  # noqa: E402

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
MODEL = f"{RES}/gpu_finetune_handoff/models/teacher_current_compiled.nequip.pth"
MODEL_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
ALLOW = [f"{RES}/gpu_finetune_handoff/models/"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    adapter = TrustedAllegroAdapter(MODEL, expected_sha256=MODEL_SHA, allow_prefixes=ALLOW)  # sha+allow guards
    info = adapter.load(device=a.device)   # LOAD ONLY — no forward
    # confirm the species mapping the C3 conversion will use (LAMMPS 1->O, 2->Si), NO structure fed
    mapping_ok = (adapter.type_names == ["O", "Si"]
                  and adapter.species_index("O") == 0 and adapter.species_index("Si") == 1)
    print(json.dumps({"model_load": "SUCCESS_no_forward", "model_path": MODEL, "info": info,
                      "lammps_type_1->O->index": adapter.species_index("O"),
                      "lammps_type_2->Si->index": adapter.species_index("Si"),
                      "species_mapping_ok": mapping_ok}, indent=2))
    return 0 if mapping_ok else 1


if __name__ == "__main__":
    sys.exit(main())
