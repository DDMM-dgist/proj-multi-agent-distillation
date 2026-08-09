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


_REQUIRED_LIMITS = {"structures": 1, "forward_passes": 1, "gpus": 1}
_TRUE_LIMITS = ("no_scheduler", "no_md", "no_dft", "no_training", "no_additional_teacher_labeling",
                "no_paid_api", "no_external_network", "no_overwrite")


def _validate_approval(approval: dict, params: dict) -> None:
    """Fail-closed validation of the EXTERNAL approval content (steps 3-9). approved=false is never
    treated as active."""
    if approval.get("approved") is not True or not str(approval.get("approver", "")).strip():
        raise ExecutionRefused("approval not active (approved must be true with an approver)")
    if approval.get("action") != "label_with_teacher" or approval.get("subtype") != "teacher_single_point":
        raise ExecutionRefused("approval action/subtype does not match label_with_teacher/teacher_single_point")
    if approval.get("structure_sha256") != params["source_sha256"]:
        raise ExecutionRefused("approval structure_sha256 does not match the proposal")
    if approval.get("teacher_sha256") != params["model_sha256"]:
        raise ExecutionRefused("approval teacher_sha256 does not match the proposal")
    if approval.get("authorizes_subsequent_actions") is not False:
        raise ExecutionRefused("approval must set authorizes_subsequent_actions=false")
    lim = approval.get("limits") or {}
    for k, v in _REQUIRED_LIMITS.items():
        if lim.get(k) != v:
            raise ExecutionRefused(f"approval limit {k} must be {v}")
    if not (isinstance(lim.get("scientific_inference_wall_time_s_max"), (int, float))
            and lim["scientific_inference_wall_time_s_max"] <= 60):
        raise ExecutionRefused("approval wall-time limit must be <= 60 s")
    for k in _TRUE_LIMITS:
        if lim.get(k) is not True:
            raise ExecutionRefused(f"approval limit {k} must be true")


def _write_failure(rd: Path, head: str, approval_ref: dict, state: str, reason: str) -> None:
    """Failure atomicity: preserve the actual execution attempt (append-only) without pretending a
    scientific prediction occurred, and with no automatic retry."""
    forward_happened = (rd / "teacher_ef.json").exists()
    rec = {"status": state, "reason": reason, "package_head": head, "source_approval": approval_ref,
           "forward_pass_completed": forward_happened, "scientific_verdict": None,
           "automatic_retry": False,
           "note": ("failure AFTER the model forward — attempt preserved append-only; do not rerun under "
                    "this run identity" if forward_happened else
                    "failure BEFORE the model forward — no scientific prediction occurred")}
    (rd / "run_manifest.json").write_text(json.dumps(rec, indent=2) + "\n")
    (rd / "provenance.json").write_text(json.dumps({**rec, "run_id": RUN_ID, "stage": "stage_d2_c3"}, indent=2) + "\n")


def execute(*, approval_path, device: str = "cuda:0", expect_head: str = None,
            run_dir: str = None, repo_root: Path = ROOT, adapter_factory=None,
            clock=None) -> dict:
    """Perform the single approved teacher single-point via the EXTERNAL-approval flow:
    verify HEAD -> read external approval read-only -> validate -> verify SHAs -> fresh run dir ->
    snapshot approval into it -> ONE forward -> outputs -> Axis-A/B -> immutability + clean unload.
    The forward comes ONLY from the trusted adapter constructed here. Returns a report dict."""
    clock = clock or time.monotonic
    repo_root = Path(repo_root)
    proposal = json.loads((C3 / "action_proposal.json").read_text())
    params = proposal["parameters"]
    src, model = params["source_structure"], params["teacher_model"]
    # 1. EXPECT_HEAD
    head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).decode().strip()
    if expect_head and head != expect_head:
        raise ExecutionRefused(f"HEAD {head} != EXPECT_HEAD {expect_head}")
    # 2. read EXTERNAL approval read-only (+ its sha, for provenance) — never modified
    approval_path = Path(approval_path)
    approval_sha = _sha(approval_path)
    approval = json.loads(approval_path.read_text())
    # 3-9. validate approval schema/content + limits + authorizes_subsequent_actions=false
    _validate_approval(approval, params)
    # 6/7. verify exact structure + teacher path + SHA (files on disk)
    src_before, model_before = _sha(src), _sha(model)
    if src_before != params["source_sha256"]:
        raise ExecutionRefused("structure sha256 mismatch")
    if model_before != params["model_sha256"]:
        raise ExecutionRefused("teacher model sha256 mismatch")
    rd = Path(run_dir) if run_dir else (repo_root / "runs" / "stage_d2_c3" / RUN_ID)
    # 10. target run dir must NOT exist (fresh-run guard owned by the wrapper; not weakened)
    if rd.exists():
        raise ExecutionRefused(f"run dir exists (no overwrite): {rd}")
    # 11. atomically create the fresh run dir
    rd.mkdir(parents=True, exist_ok=False)
    # 12. snapshot the validated approval into the run dir (immutable execution-time snapshot)
    (rd / "approval.json").write_text(json.dumps(approval, indent=2) + "\n")
    approval_ref = {"external_approval_path": str(approval_path), "external_approval_sha256": approval_sha}
    # from here a run dir EXISTS -> use failure-atomic recording; NO automatic retry
    try:
        # 14. construct trusted adapter INTERNALLY (no CLI/agent forward_fn); load exact teacher
        factory = adapter_factory or _real_adapter_factory
        adapter = factory(model, params["model_sha256"], params["read_allow_prefixes"])
        load_prov = adapter.load(device=device)
        forward_fn = adapter.build_forward_fn()      # trusted callable
        # 15/16. exactly ONE forward -> executor writes teacher_ef.json + forces.csv into the fresh dir
        result = EX.run_teacher_single_point(proposal=proposal, run_dir=str(rd), approval=approval,
                                             forward_fn=forward_fn, clock=clock, run_dir_precreated=True)
        if result.status != "OK":
            _write_failure(rd, head, approval_ref, f"EXECUTION_FAILED_{result.status}", result.reason)
            raise ExecutionRefused(f"executor STOP: {result.status} {result.reason}")
    except ExecutionRefused:
        raise
    except Exception as exc:  # noqa: BLE001
        after = (rd / "teacher_ef.json").exists()
        _write_failure(rd, head, approval_ref,
                       "EXECUTION_FAILED_AFTER_FORWARD" if after else "EXECUTION_FAILED_BEFORE_FORWARD",
                       f"{type(exc).__name__}: {exc}")
        raise ExecutionRefused(f"execution failed ({'after' if after else 'before'} forward): {exc}") from exc
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
    # copy the PLANNED manifests into the run dir (append-only); approval.json already snapshotted (step 12)
    (rd / "input_manifest.json").write_text((C3 / "input_manifest.json").read_text())
    (rd / "model_manifest.json").write_text((C3 / "model_manifest.json").read_text())
    # 17. deterministic Axis-A/B validity gate (frozen criterion_eval; bound verdict)
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
                  "source_approval": approval_ref,   # 13: external approval path + sha (never modified)
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
    ap.add_argument("--approval", required=True, help="path to the EXTERNAL approval json (read-only)")
    a = ap.parse_args()                       # NOTE: no --forward / no python expression is accepted
    print(json.dumps(execute(approval_path=a.approval, device=a.device, expect_head=a.expect_head), indent=2))


if __name__ == "__main__":
    main()
