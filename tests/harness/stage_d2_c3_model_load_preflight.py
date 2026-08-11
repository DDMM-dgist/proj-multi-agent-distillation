#!/usr/bin/env python3
"""Stage D-2 C3 MODEL-LOAD-ONLY preflight (reproducible). Loads the compiled teacher via the TRUSTED
adapter and reports metadata + device compatibility. NO structure is fed through the model; NO forward
pass; NO E/F artifact is produced. Run in the real nequip/allegro env; on a GPU host pass --device
cuda:<id> to also confirm GPU load compatibility (still NO forward pass).

  conda run -n allegro python tests/harness/stage_d2_c3_model_load_preflight.py [--device cpu|cuda:1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai.stage_d2_c3_teacher_adapter import TrustedAllegroAdapter  # noqa: E402

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
MODEL = f"{RES}/gpu_finetune_handoff/models/teacher_current_compiled.nequip.pth"
MODEL_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
ALLOW = [f"{RES}/gpu_finetune_handoff/models/"]


def _gpu_free_mib(device):
    try:
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = int(device.split(":")[1]) if ":" in device else 0
            free, total = torch.cuda.mem_get_info(idx)
            return {"free_MiB": free // (1 << 20), "total_MiB": total // (1 << 20)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:120]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    gpu_before = _gpu_free_mib(a.device)
    adapter = TrustedAllegroAdapter(MODEL, expected_sha256=MODEL_SHA, allow_prefixes=ALLOW)  # sha+allow guards
    info = adapter.load(device=a.device)   # LOAD ONLY — no forward, NO structure
    gpu_after = _gpu_free_mib(a.device)
    load_mib = None
    if gpu_before and gpu_after and "free_MiB" in gpu_before and "free_MiB" in gpu_after:
        load_mib = gpu_before["free_MiB"] - gpu_after["free_MiB"]
    # confirm the species mapping the C3 conversion will use (LAMMPS 1->O, 2->Si), NO structure fed
    mapping_ok = (adapter.type_names == ["O", "Si"]
                  and adapter.species_index("O") == 0 and adapter.species_index("Si") == 1)
    # clean unload
    unloaded = False
    try:
        import torch
        adapter._model = None
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
        unloaded = True
    except Exception:  # noqa: BLE001
        pass
    gpu_final = _gpu_free_mib(a.device)
    print(json.dumps({"model_load": "SUCCESS_no_forward", "device": a.device, "model_path": MODEL,
                      "info": info, "gpu_free_before": gpu_before, "gpu_free_after_load": gpu_after,
                      "model_load_MiB": load_mib, "gpu_free_after_unload": gpu_final,
                      "clean_unload": unloaded,
                      "lammps_type_1->O->index": adapter.species_index("O"),
                      "lammps_type_2->Si->index": adapter.species_index("Si"),
                      "species_mapping_ok": mapping_ok,
                      "note": "MODEL LOAD ONLY — no structure, no forward pass, no E/F"}, indent=2))
    return 0 if mapping_ok else 1


if __name__ == "__main__":
    sys.exit(main())
