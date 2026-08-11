#!/usr/bin/env python3
"""Stage D-2 C3 MODEL-INPUT-BUILD preflight — one step beyond model-load, still NO model forward.

Parses the exact mini216 structure, builds the exact nequip 0.16.1 model input via the trusted adapter
(sha+allow guards, load-only, compute_neighborlist_), and verifies keys/shapes/types/cell/PBC/edge_index
— WITHOUT invoking the TorchScript model. This would have caught the attempt-1
`AtomicDataDict.with_edge_vectors` API mismatch before scientific execution. Records
model_forward_called=false / teacher_EF_generated=false. Run in the allegro env.

  conda run -n allegro python tests/harness/stage_d2_c3_input_build_preflight.py --device cpu|cuda:1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai.stage_d2_c3_teacher_adapter import TrustedAllegroAdapter  # noqa: E402
from runtimes.pydantic_ai.stage_d2_c3_teacher_executor import parse_lammps_data  # noqa: E402

C3 = ROOT / "tests" / "fixtures" / "stage_d2_c3"


def build_input_preflight(device: str = "cpu"):
    params = json.loads((C3 / "action_proposal.json").read_text())["parameters"]
    src = params["source_structure"]; tmap = params["type_symbol_map"]
    adapter = TrustedAllegroAdapter(params["teacher_model"], expected_sha256=params["model_sha256"],
                                    allow_prefixes=params["read_allow_prefixes"])
    load_prov = adapter.load(device=device)                 # load-only (no forward)
    positions, types, n_atoms, box_L = parse_lammps_data(src)
    data, conv = adapter.build_model_input(positions, types, box_L, tmap)   # NO model call

    import torch  # lazy
    from nequip.data import AtomicDataDict as A
    pos = data[A.POSITIONS_KEY]; at = data[A.ATOM_TYPE_KEY]; ei = data.get(A.EDGE_INDEX_KEY)
    cell = data[A.CELL_KEY]; pbc = data[A.PBC_KEY]
    type_idx = set(int(x) for x in at.tolist())
    checks = {
        "model_forward_called": False, "teacher_EF_generated": False,
        "n_atoms": n_atoms, "n_atoms_216": n_atoms == 216,
        "pos_shape": list(pos.shape), "pos_shape_216x3": list(pos.shape) == [216, 3],
        "atom_types_len": int(at.numel()), "atom_types_len_216": int(at.numel()) == 216,
        "type_indices": sorted(type_idx), "type_indices_subset_01": type_idx <= {0, 1},
        "composition": conv["composition"], "composition_O144_Si72": conv["composition"] == {"O": 144, "Si": 72},
        "cell": cell.reshape(3, 3).tolist(), "cell_diag_L": [round(cell.reshape(3, 3)[i][i].item(), 4) for i in range(3)],
        "cell_matches_box": all(abs(cell.reshape(3, 3)[i][i].item() - box_L) < 1e-6 for i in range(3)),
        "pbc": [bool(x) for x in pbc.reshape(-1).tolist()], "pbc_all_true": bool(pbc.all().item()),
        "edge_index_present": ei is not None,
        "edge_index_shape": list(ei.shape) if ei is not None else None,
        "edge_index_2xE_Epos": (ei is not None and ei.shape[0] == 2 and ei.shape[1] > 0),
        "required_fields_present": all(k in data for k in
                                       (A.POSITIONS_KEY, A.CELL_KEY, A.PBC_KEY, A.ATOM_TYPE_KEY, A.EDGE_INDEX_KEY)),
        "r_max": adapter.r_max, "type_names": adapter.type_names, "load_provenance": load_prov,
    }
    ok = all(checks[k] for k in ("n_atoms_216", "pos_shape_216x3", "atom_types_len_216",
                                 "type_indices_subset_01", "composition_O144_Si72", "cell_matches_box",
                                 "pbc_all_true", "edge_index_present", "edge_index_2xE_Epos",
                                 "required_fields_present"))
    return ok, checks


def main():
    ap = argparse.ArgumentParser(description="Stage D-2 C3 model-input-build preflight (NO forward).")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ok, checks = build_input_preflight(device=a.device)
    print(json.dumps({"input_build_preflight_ok": ok, "device": a.device,
                      "model_forward_called": False, "teacher_EF_generated": False,
                      "checks": checks}, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
