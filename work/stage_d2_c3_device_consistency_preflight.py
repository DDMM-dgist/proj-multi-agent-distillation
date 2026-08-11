#!/usr/bin/env python3
"""Stage D-2 C3 DEVICE-CONSISTENCY preflight — the last no-forward step before scientific execution.

Source-grounded on the attempt-2 failure: the deployed model was invoked and died INSIDE edge
normalization with ``Expected all tensors to be on the same device, but found at least two devices,
cuda:1 and cpu`` — the model's buffers (e.g. ``_rmax_recip``) were on cuda:1 while the built input was
on CPU. This preflight loads the exact teacher, builds the exact mini216 input, and inspects
the device of EVERY model parameter/buffer AND EVERY input tensor, then fails closed if they are not all
consistently placed on the target device. It performs ZERO model calls — it would have caught attempt-2
before scientific execution. Run in the allegro env.

  conda run -n allegro python work/stage_d2_c3_device_consistency_preflight.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai.stage_d2_c3_teacher_adapter import (  # noqa: E402
    TrustedAllegroAdapter, device_consistency_report)
from runtimes.pydantic_ai.stage_d2_c3_teacher_executor import parse_lammps_data  # noqa: E402

C3 = ROOT / "examples" / "stage_d2_c3"


def device_consistency_preflight(device: str = "cpu"):
    params = json.loads((C3 / "action_proposal.json").read_text())["parameters"]
    src = params["source_structure"]; tmap = params["type_symbol_map"]
    adapter = TrustedAllegroAdapter(params["teacher_model"], expected_sha256=params["model_sha256"],
                                    allow_prefixes=params["read_allow_prefixes"])
    load_prov = adapter.load(device=device)                 # load-only (no forward)
    positions, types, n_atoms, box_L = parse_lammps_data(src)
    data, _conv = adapter.build_model_input(positions, types, box_L, tmap)   # NO model call (device-placed)

    model_dev = adapter.model_device_report()               # every param/buffer device (NO forward)
    input_dev = adapter.input_device_report(data)           # every input tensor device (NO forward)
    report = device_consistency_report(model_dev["devices"], input_dev["devices"], model_dev["target_device"])
    checks = {
        "model_forward_called": False, "teacher_EF_generated": False,
        "target_device": model_dev["target_device"],
        "model_devices": model_dev["devices"], "input_devices": input_dev["devices"],
        "model_offenders": report["model_offenders"], "input_offenders": report["input_offenders"],
        "mixed_cpu_cuda": report["mixed"], "consistent": report["ok"],
        "model_param_buffer_devices": model_dev, "input_field_devices": input_dev,
        "r_max": adapter.r_max, "type_names": adapter.type_names, "load_provenance": load_prov,
    }
    return report["ok"], checks


def main():
    ap = argparse.ArgumentParser(description="Stage D-2 C3 device-consistency preflight (NO forward).")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ok, checks = device_consistency_preflight(device=a.device)
    print(json.dumps({"device_consistency_preflight_ok": ok, "device": a.device,
                      "model_forward_called": False, "teacher_EF_generated": False,
                      "checks": checks}, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
