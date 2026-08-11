#!/usr/bin/env python3
"""Stage D-2 C3 EXECUTION-ENVIRONMENT preflight — import/load-contract validation ONLY.

Checks, BEFORE any scientific execution or run-dir creation, that the committed wrapper's Python
dependencies + inputs are present: pydantic (repo requires >=2.0) + pydantic_core, the repo modules
(criterion_eval, TrustedAllegroAdapter, the C3 executor), torch, nequip, the exact teacher + structure
files with matching SHA256, and CUDA availability when a cuda device is requested. Performs ZERO model
forward. Fails CLOSED. This module itself imports WITHOUT pydantic so it can REPORT a missing pydantic
(the exact failure that aborted execution-launch attempt 1).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:      # so `runtimes.*` resolves when run as `python tests/harness/...`
    sys.path.insert(0, str(ROOT))
C3 = ROOT / "tests" / "fixtures" / "stage_d2_c3"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_env(*, device: str = "cpu", require_cuda: bool = None, repo_root: Path = ROOT):
    """Return (ok, checks). ok=False (fail closed) if any required import/input/device check fails.
    No forward pass; no run-dir creation."""
    require_cuda = device.startswith("cuda") if require_cuda is None else require_cuda
    checks, ok = {}, True

    def imp(name):
        nonlocal ok
        try:
            m = importlib.import_module(name)
            checks[name] = "OK " + str(getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            checks[name] = f"FAIL:{type(e).__name__}"; ok = False

    for mod in ("pydantic", "pydantic_core",
                "runtimes.pydantic_ai.criterion_eval",
                "runtimes.pydantic_ai.stage_d2_c3_teacher_executor",
                "runtimes.pydantic_ai.stage_d2_c3_teacher_adapter",
                "torch", "nequip"):
        imp(mod)

    params = json.loads((C3 / "action_proposal.json").read_text())["parameters"]
    for path_key, sha_key in (("source_structure", "source_sha256"), ("teacher_model", "model_sha256")):
        p = params[path_key]
        exists = Path(p).is_file()
        checks[path_key + "_exists"] = exists
        if not exists:
            ok = False; continue
        match = _sha(p) == params[sha_key]
        checks[path_key + "_sha_match"] = match
        ok = ok and match

    if require_cuda:
        try:
            import torch
            cu = bool(torch.cuda.is_available())
            checks["cuda_available"] = cu
            ok = ok and cu
        except Exception as e:  # noqa: BLE001
            checks["cuda_available"] = f"FAIL:{type(e).__name__}"; ok = False
    return ok, checks


def main():
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Stage D-2 C3 import/load-contract preflight (no forward).")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ok, checks = check_env(device=a.device)
    print(json.dumps({"env_preflight_ok": ok, "device": a.device, "checks": checks,
                      "note": "IMPORT/LOAD CONTRACT ONLY — no model forward"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
