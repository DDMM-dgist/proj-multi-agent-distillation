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

  conda run -n allegro python tests/harness/stage_d2_c3_execute.py \
      --device cuda:1 --expect-head <sha> --attempt 3 \
      --approval tests/fixtures/stage_d2_c3/approvals/d2c3-teacher-sp-mini216-attempt3.approval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "harness"))
# NOTE: criterion_eval / the executor are imported LAZILY inside execute() AFTER the env preflight, so a
# missing dependency (e.g. pydantic — the launch-attempt-1 failure) is reported by the preflight and
# fails closed BEFORE any run-dir creation, instead of crashing at wrapper import time.

BASE_RUN_ID = "d2c3-teacher-sp-mini216"     # attempt-1 (this exact id) is the IMMUTABLE failed run
C3 = ROOT / "tests" / "fixtures" / "stage_d2_c3"


def run_id_for(attempt: int) -> str:
    """Deterministic run identity. Attempt 1 == the immutable failed run (base id); scientific
    execution uses attempt>=2 with an explicit -attemptN suffix."""
    return BASE_RUN_ID if attempt == 1 else f"{BASE_RUN_ID}-attempt{attempt}"


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


# Execution state machine (failure atomicity): the lifecycle a scientific execution passes through, and
# the failure class for an exception raised in each. Attempt 2 (self._model(data) invoked, RuntimeError
# device mismatch INSIDE the model, no E/F returned) maps to DURING_FORWARD — NOT the coarse
# BEFORE_FORWARD the old "teacher_ef.json exists?" heuristic produced.
_STATES = ("PRE_FORWARD", "FORWARD_STARTED", "FORWARD_COMPLETED", "ARTIFACT_COMMITTED")
_FAILURE_NOTE = {
    "EXECUTION_FAILED_BEFORE_FORWARD":
        "failure BEFORE the model was invoked (PRE_FORWARD) — no forward, no scientific prediction",
    "EXECUTION_FAILED_DURING_FORWARD":
        "the deployed model forward was INVOKED (FORWARD_STARTED) but did not complete — no valid E/F "
        "was returned; this is a model-invocation failure, NOT a completed scientific prediction; no "
        "automatic retry",
    "EXECUTION_FAILED_AFTER_FORWARD":
        "the model forward returned E/F (FORWARD_COMPLETED) but a later step failed — attempt preserved "
        "append-only; do not rerun under this run identity; no automatic retry",
}


def classify_failure(*, forward_invoked: bool, forward_completed: bool, artifact_exists: bool) -> str:
    """Map the observed execution phase to a failure class. A model invocation that did not successfully
    return (forward_invoked and not forward_completed) is DURING_FORWARD (attempt-2's class)."""
    if forward_completed or artifact_exists:
        return "EXECUTION_FAILED_AFTER_FORWARD"
    if forward_invoked:
        return "EXECUTION_FAILED_DURING_FORWARD"
    return "EXECUTION_FAILED_BEFORE_FORWARD"


def _write_failure(rd: Path, head: str, approval_ref: dict, state: str, reason: str, *,
                   model_forward_invoked: bool = False, model_forward_completed: bool = False,
                   valid_prediction_generated: bool = False) -> None:
    """Failure atomicity: preserve the actual execution attempt (append-only) without pretending a
    scientific prediction occurred, and with no automatic retry. Records durable, separate counters —
    model_forward_invoked vs model_forward_completed vs valid_prediction_generated — so a DURING-forward
    model failure is never conflated with a completed prediction."""
    rec = {"status": state, "reason": reason, "package_head": head, "source_approval": approval_ref,
           "model_forward_invoked": model_forward_invoked,
           "model_forward_completed": model_forward_completed,
           "valid_prediction_generated": valid_prediction_generated,
           "forward_pass_completed": model_forward_completed,   # retained key (== model_forward_completed)
           "scientific_verdict": None, "automatic_retry": False,
           "note": _FAILURE_NOTE.get(state, "")}
    (rd / "run_manifest.json").write_text(json.dumps(rec, indent=2) + "\n")
    (rd / "provenance.json").write_text(json.dumps({**rec, "run_id": Path(rd).name, "stage": "stage_d2_c3"}, indent=2) + "\n")


def execute(*, approval_path, device: str = "cuda:0", expect_head: str = None,
            run_dir: str = None, repo_root: Path = ROOT, adapter_factory=None,
            clock=None, env_check=None, attempt: int = 3) -> dict:
    """Perform the single approved teacher single-point via the EXTERNAL-approval flow:
    ENV PREFLIGHT (import/load contract) -> verify HEAD -> read external approval read-only -> validate
    -> verify SHAs -> fresh run dir -> snapshot approval into it -> ONE forward -> outputs -> Axis-A/B ->
    immutability + clean unload. The forward comes ONLY from the trusted adapter constructed here.
    Returns a report dict."""
    clock = clock or time.monotonic
    repo_root = Path(repo_root)
    proposal = json.loads((C3 / "action_proposal.json").read_text())
    params = proposal["parameters"]
    src, model = params["source_structure"], params["teacher_model"]
    # 0. EXECUTION-ENVIRONMENT preflight (import/load contract only; no forward). Fails CLOSED here,
    #    BEFORE HEAD/approval/run-dir — this is the launch-attempt-1 (missing pydantic) failure class.
    if env_check is None:
        from stage_d2_c3_env_preflight import check_env
        env_check = check_env
    env_ok, env_report = env_check(device=device)
    if not env_ok:
        raise ExecutionRefused(f"execution-environment preflight FAILED (PRE_EXECUTION_IMPORT; no run "
                               f"dir created; no forward): {env_report}")
    # deps now confirmed present -> import the pydantic-dependent modules lazily
    from runtimes.pydantic_ai import stage_d2_c3_teacher_executor as EX
    from runtimes.pydantic_ai.criterion_eval import (
        derive_severity, evaluate_criteria, render_authoritative_block)
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
    # attempt identity: refuse reusing the immutable failed attempt-1 (nequip API mismatch) or attempt-2
    # (device mismatch) ids; scientific execution is attempt>=3
    if attempt < 3:
        raise ExecutionRefused("attempts 1 and 2 are immutable failed runs; scientific execution uses attempt>=3")
    run_id = run_id_for(attempt)
    rd = Path(run_dir) if run_dir else (repo_root / "tests" / "fixtures" / "stage_d2_c3" / run_id)
    _immutable_dirs = {(repo_root / "tests" / "fixtures" / "stage_d2_c3" / BASE_RUN_ID).resolve(),
                       (repo_root / "tests" / "fixtures" / "stage_d2_c3" / f"{BASE_RUN_ID}-attempt2").resolve()}
    if rd.name in (BASE_RUN_ID, f"{BASE_RUN_ID}-attempt2") or rd.resolve() in _immutable_dirs:
        raise ExecutionRefused("refusing to target an immutable failed attempt (attempt-1 / attempt-2) run directory")
    # 10. target run dir must NOT exist (fresh-run guard owned by the wrapper; not weakened)
    if rd.exists():
        raise ExecutionRefused(f"run dir exists (no overwrite): {rd}")
    # 11. atomically create the fresh run dir
    rd.mkdir(parents=True, exist_ok=False)
    # 12. snapshot the validated approval into the run dir (immutable execution-time snapshot)
    (rd / "approval.json").write_text(json.dumps(approval, indent=2) + "\n")
    approval_ref = {"external_approval_path": str(approval_path), "external_approval_sha256": approval_sha}
    # from here a run dir EXISTS -> use failure-atomic recording; NO automatic retry
    adapter = None                                   # bound before the try so the except can read phase flags
    try:
        # 14. construct trusted adapter INTERNALLY (no CLI/agent forward_fn); load exact teacher
        factory = adapter_factory or _real_adapter_factory
        adapter = factory(model, params["model_sha256"], params["read_allow_prefixes"])
        load_prov = adapter.load(device=device)
        forward_fn = adapter.build_forward_fn()      # trusted callable (resets forward phase flags)
        # 15/16. exactly ONE forward -> executor writes teacher_ef.json + forces.csv into the fresh dir
        result = EX.run_teacher_single_point(proposal=proposal, run_dir=str(rd), approval=approval,
                                             forward_fn=forward_fn, clock=clock, run_dir_precreated=True)
        if result.status != "OK":
            _write_failure(rd, head, approval_ref, f"EXECUTION_FAILED_{result.status}", result.reason,
                           model_forward_invoked=bool(getattr(adapter, "forward_invoked", False)),
                           model_forward_completed=bool(getattr(adapter, "forward_completed", False)))
            raise ExecutionRefused(f"executor STOP: {result.status} {result.reason}")
    except ExecutionRefused:
        raise
    except Exception as exc:  # noqa: BLE001
        # classify BEFORE / DURING / AFTER from the adapter's forward-phase flags (not from the mere
        # existence of teacher_ef.json): attempt-2's in-model device mismatch is DURING_FORWARD.
        invoked = bool(getattr(adapter, "forward_invoked", False))
        completed = bool(getattr(adapter, "forward_completed", False))
        artifact_exists = (rd / "teacher_ef.json").exists()
        state = classify_failure(forward_invoked=invoked, forward_completed=completed,
                                 artifact_exists=artifact_exists)
        _write_failure(rd, head, approval_ref, state, f"{type(exc).__name__}: {exc}",
                       model_forward_invoked=invoked, model_forward_completed=completed,
                       valid_prediction_generated=False)
        phase = {"EXECUTION_FAILED_BEFORE_FORWARD": "before", "EXECUTION_FAILED_DURING_FORWARD": "during",
                 "EXECUTION_FAILED_AFTER_FORWARD": "after"}[state]
        raise ExecutionRefused(f"execution failed ({phase} forward): {exc}") from exc
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
    provenance = {"run_id": run_id, "attempt": attempt,
                  "supersedes": {"attempt1_run": BASE_RUN_ID,
                                 "attempt1_status": "EXECUTION_FAILED_BEFORE_FORWARD (immutable; NEQUIP_0_16_1_ATOMICDATADICT_API_MISMATCH)",
                                 "attempt2_run": f"{BASE_RUN_ID}-attempt2",
                                 "attempt2_status": "EXECUTION_FAILED_DURING_FORWARD (immutable; MODEL_INPUT_OR_BUFFER_DEVICE_MISMATCH_CPU_VS_CUDA1)"},
                  "model_forward_invoked": True, "model_forward_completed": True,
                  "valid_prediction_generated": True,
                  "stage": "stage_d2_c3", "action": "label_with_teacher",
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
    return {"run_id": run_id, "attempt": attempt,
            "authoritative_verdict": verdict, "E_per_atom_eV": result.validity.get("E_per_atom_eV"),
            "max_force_eV_A": result.validity.get("max_force_eV_A"),
            "n_atoms": result.artifact["n_atoms"], "composition": result.artifact["composition"],
            "source_model_unchanged": source_unchanged, "clean_unload": unloaded,
            "artifacts": provenance["artifacts_sha256"]}


def main():
    ap = argparse.ArgumentParser(description="Stage D-2 C3 one-shot trusted teacher single-point.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expect-head", default=None)
    ap.add_argument("--approval", required=True, help="path to the EXTERNAL approval json (read-only)")
    ap.add_argument("--attempt", type=int, default=3, help="scientific run attempt (>=3; attempts 1 and 2 are immutable failed)")
    a = ap.parse_args()                       # NOTE: no --forward / no python expression is accepted
    print(json.dumps(execute(approval_path=a.approval, device=a.device, expect_head=a.expect_head,
                             attempt=a.attempt), indent=2))


if __name__ == "__main__":
    main()
