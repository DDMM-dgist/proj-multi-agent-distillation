#!/usr/bin/env python3
"""Stage D-2 C3 one-shot TRUSTED execution wrapper — the EXACT committed code that performs the single
approved teacher forward pass. Committed BEFORE execution so it is in the approved HEAD (no execution-
wrapper provenance caveat, unlike C1).

Contract: verify EXPECT_HEAD -> require explicit approval -> verify structure+teacher SHA -> construct
TrustedAllegroAdapter INTERNALLY (no arbitrary/agent/CLI forward_fn) -> load the exact teacher -> convert
exactly one mini216 -> ONE model forward -> run_teacher_single_point -> write append-only under the fresh
C3 run dir -> evaluate deterministic Axis-A/B -> preserve source/model SHA before+after -> record
adapter/model/device/dtype/env provenance -> clean unload -> refuse existing run dir -> NO follow-up
(NO semantic Judge here; that is a separate later approval). One structure, one forward, one GPU.

The CLI exposes only --device / --expect-head / --approval; it NEVER accepts a forward function or a
Python expression. ``adapter_factory`` is a CODE-LEVEL test seam (default = the real adapter); tests
inject a synthetic-output adapter. The real path uses only the committed TrustedAllegroAdapter.

  conda run -n allegro python work/stage_d2_c3_execute.py \
      --device cuda:1 --expect-head <sha> --approval runs/stage_d2_c3/d2c3-teacher-sp-mini216/approval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))
from runtimes.pydantic_ai import stage_d2_c3_teacher_executor as EX  # noqa: E402
from runtimes.pydantic_ai.criterion_eval import (  # noqa: E402
    derive_severity, evaluate_criteria, render_authoritative_block)

RUN_ID = "d2c3-teacher-sp-mini216"
C3 = ROOT / "examples" / "stage_d2_c3"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class ExecutionRefused(RuntimeError):
    pass


def _real_adapter_factory(model_path, expected_sha256, allow_prefixes):
    from runtimes.pydantic_ai.stage_d2_c3_teacher_adapter import TrustedAllegroAdapter
    return TrustedAllegroAdapter(model_path, expected_sha256=expected_sha256, allow_prefixes=allow_prefixes)


def execute(*, approval: dict, device: str = "cuda:0", expect_head: str = None,
            run_dir: str = None, repo_root: Path = ROOT, adapter_factory=None,
            clock=None) -> dict:
    """Perform the single approved teacher single-point. Deterministic orchestration; the forward pass
    comes ONLY from the trusted adapter constructed here. Returns a report dict."""
    clock = clock or time.monotonic
    repo_root = Path(repo_root)
    # 1. EXPECT_HEAD
    head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).decode().strip()
    if expect_head and head != expect_head:
        raise ExecutionRefused(f"HEAD {head} != EXPECT_HEAD {expect_head}")
    # 2. approval
    if not approval or approval.get("approved") is not True or not str(approval.get("approver", "")).strip():
        raise ExecutionRefused("explicit approval artifact required (approved=true)")
    if approval.get("authorizes_subsequent_actions") is True:
        raise ExecutionRefused("approval must not authorize subsequent actions")
    proposal = json.loads((C3 / "action_proposal.json").read_text())
    params = proposal["parameters"]
    src, model = params["source_structure"], params["teacher_model"]
    rd = Path(run_dir) if run_dir else (repo_root / "runs" / "stage_d2_c3" / RUN_ID)
    # 16. refuse existing run dir (executor also refuses; check early)
    if rd.exists():
        raise ExecutionRefused(f"run dir exists (no overwrite): {rd}")
    # 3/4. structure + teacher SHA
    src_before, model_before = _sha(src), _sha(model)
    if src_before != params["source_sha256"]:
        raise ExecutionRefused("structure sha256 mismatch")
    if model_before != params["model_sha256"]:
        raise ExecutionRefused("teacher model sha256 mismatch")
    # 5/6/7. construct trusted adapter INTERNALLY (no CLI/agent forward_fn); load exact teacher
    factory = adapter_factory or _real_adapter_factory
    adapter = factory(model, params["model_sha256"], params["read_allow_prefixes"])
    load_prov = adapter.load(device=device)
    forward_fn = adapter.build_forward_fn()          # trusted callable (8/9/10 happen inside the executor)
    # 8/9/10/11. one structure -> ONE forward -> executor writes teacher_ef.json + forces.csv under rd
    result = EX.run_teacher_single_point(proposal=proposal, run_dir=str(rd), approval=approval,
                                         forward_fn=forward_fn, clock=clock)
    if result.status != "OK":
        raise ExecutionRefused(f"executor STOP: {result.status} {result.reason}")
    # provenance augmentation of teacher_ef.json (14): model dtype/device/type_names/cutoff/versions/shape
    ef = json.loads((rd / "teacher_ef.json").read_text())
    ef["force_array_shape"] = [result.artifact["n_atoms"], 3]
    ef["model_device"] = device
    ef["model_dtype"] = getattr(adapter, "model_dtype", load_prov.get("model_dtype"))
    ef["model_type_names"] = getattr(adapter, "type_names", load_prov.get("type_names"))
    ef["cutoff_A"] = getattr(adapter, "r_max", load_prov.get("r_max"))
    ef["inference_wall_time_s"] = result.validity.get("runtime_s")
    ef["software_versions"] = {k: load_prov.get(k) for k in ("python", "torch", "nequip", "allegro")}
    (rd / "teacher_ef.json").write_text(json.dumps(ef, indent=2) + "\n")
    # copy the PLANNED manifests into the run dir (append-only)
    (rd / "input_manifest.json").write_text((C3 / "input_manifest.json").read_text())
    (rd / "model_manifest.json").write_text((C3 / "model_manifest.json").read_text())
    (rd / "approval.json").write_text(json.dumps(approval, indent=2) + "\n")
    # 12. deterministic Axis-A/B validity gate (frozen criterion_eval; bound verdict)
    spec = json.loads((C3 / "criteria" / "teacher_ef_validity.json").read_text())
    results = evaluate_criteria(result.validity, spec)
    verdict = derive_severity(results)
    (rd / "criterion_results.json").write_text(json.dumps(
        {"deterministic_authoritative": True, "authoritative_verdict": verdict,
         "criterion_results": [r.model_dump() for r in results],
         "provenance_block": render_authoritative_block(results)}, indent=2) + "\n")
    # 13. source/model unchanged after
    src_after, model_after = _sha(src), _sha(model)
    source_unchanged = (src_after == src_before and model_after == model_before)
    # 15. clean unload
    unloaded = False
    try:
        adapter._model = None
        import torch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        unloaded = True
    except Exception:  # noqa: BLE001
        pass
    provenance = {"run_id": RUN_ID, "stage": "stage_d2_c3", "action": "label_with_teacher",
                  "subtype": "teacher_single_point", "package_head": head,
                  "adapter": "TrustedAllegroAdapter (committed; no arbitrary forward_fn)",
                  "model_identity": "base/pre-fine-tune KISTI Allegro, compiled TorchScript deploy-only",
                  "model_sha256": model_before, "structure_sha256": src_before,
                  "load_provenance": load_prov, "device": device,
                  "authoritative_verdict": verdict, "source_model_unchanged": source_unchanged,
                  "clean_unload": unloaded, "semantic_judge": "NOT run (separate later approval)",
                  "artifacts_sha256": {f: _sha(rd / f) for f in
                                       ("approval.json", "input_manifest.json", "model_manifest.json",
                                        "teacher_ef.json", "forces.csv", "criterion_results.json")}}
    (rd / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (rd / "run_manifest.json").write_text(json.dumps(
        {"status": "OK", "authoritative_verdict": verdict, "accepted": verdict == "PASS",
         "source_model_unchanged": source_unchanged, "clean_unload": unloaded,
         "semantic_judge": "deferred (separate approval)"}, indent=2) + "\n")
    return {"authoritative_verdict": verdict, "E_per_atom_eV": result.validity.get("E_per_atom_eV"),
            "max_force_eV_A": result.validity.get("max_force_eV_A"),
            "n_atoms": result.artifact["n_atoms"], "composition": result.artifact["composition"],
            "source_model_unchanged": source_unchanged, "clean_unload": unloaded,
            "artifacts": provenance["artifacts_sha256"]}


def main():
    ap = argparse.ArgumentParser(description="Stage D-2 C3 one-shot trusted teacher single-point.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expect-head", default=None)
    ap.add_argument("--approval", required=True, help="path to the approved approval.json")
    a = ap.parse_args()                       # NOTE: no --forward / no python expression is accepted
    approval = json.loads(Path(a.approval).read_text())
    print(json.dumps(execute(approval=approval, device=a.device, expect_head=a.expect_head), indent=2))


if __name__ == "__main__":
    main()
