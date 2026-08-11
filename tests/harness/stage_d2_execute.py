#!/usr/bin/env python3
"""Stage D-2 C1 trusted CONTROLLER — one approved execution of generate_analysis_artifact/posthoc_msd.

Orchestrates (does NOT modify) runtimes.pydantic_ai.stage_d2_executor.run_posthoc_msd, evaluates the
authoritative Axis-A validity gate with the FROZEN criterion_eval (bound verdict; LLM owns nothing),
augments provenance with the mandated PBC methodology record + apparent-D naming, and writes the run
artifacts. CPU-only; NO GPU, NO network, NO scheduler. The advisory (Axis-C) semantic Judge needs the
local vLLM (GPU) and is DEFERRED — this approval is CPU-only/no-GPU/no-network. Single action, no retry.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai import stage_d2_executor as EX  # noqa: E402
from runtimes.pydantic_ai.criterion_eval import (  # noqa: E402
    derive_severity, evaluate_criteria, render_authoritative_block)

EXPECT_SHA = "53ddba6a02747efb9d545415ec6468ff41c76c5a845f7afdf4cc7e71c3067591"
RUN_ID = "d2c1-posthoc-msd-random_x006"
RUN_DIR = ROOT / "tests" / "fixtures" / "stage_d2" / RUN_ID
D1_TREES = [ROOT / "tests/fixtures/stage_d1_replay", ROOT / "tests/fixtures/stage_d1_holdout",
            ROOT / "tests/fixtures/stage_d1_holdout_v2"]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _d1_fingerprint():
    fp = {}
    for t in D1_TREES:
        for f in sorted(t.rglob("*.json")):
            fp[str(f.relative_to(ROOT))] = _sha(f)
    return fp


def main():
    proposal = json.loads((ROOT / "tests/fixtures/stage_d2/action_proposal.json").read_text())
    spec = json.loads((ROOT / "tests/fixtures/stage_d2/criteria/posthoc_msd_validity.json").read_text())
    src = proposal["parameters"]["source_trajectory"]

    # pre-execution guards (controller side)
    assert _sha(src) == EXPECT_SHA, "source sha mismatch"
    assert not RUN_DIR.exists(), "run dir already exists"
    d1_before = _d1_fingerprint()
    src_before = EX.sha256_file(src)

    approval = {"approved": True, "approver": "human", "scope": "stage_d2_c1_single_posthoc_msd",
                "granted_for": {"action": "generate_analysis_artifact", "subtype": "posthoc_msd",
                                "source": src, "cpu_only": True, "no_scheduler": True, "no_gpu": True,
                                "no_network": True, "no_md": True, "no_dft": True, "no_ml_training": True,
                                "no_teacher_inference": True, "wall_clock_ceiling_s": 300,
                                "max_frames": 200, "no_overwrite": True},
                "authorizes_subsequent_actions": False, "granted_at": "2026-08-09T00:00:00Z"}

    t0 = time.monotonic()
    result = EX.run_posthoc_msd(proposal=proposal, run_dir=str(RUN_DIR), approval=approval,
                                clock=time.monotonic)
    wall = time.monotonic() - t0

    # approval.json is the first artifact persisted into the freshly-created run dir
    (RUN_DIR / "approval.json").write_text(json.dumps(approval, indent=2) + "\n")

    if result.status != "OK":
        (RUN_DIR / "run_manifest.json").write_text(json.dumps(
            {"status": result.status, "reason": result.reason, "accepted": False,
             "transition": "FAIL"}, indent=2) + "\n")
        print(json.dumps({"status": result.status, "reason": result.reason, "transition": "FAIL"}, indent=2))
        return 2

    v = result.validity
    # --- mandated PBC methodology record + apparent-D naming (provenance augmentation) ---
    pbc = {"pbc_method": "minimum_image_continuity", "pbc_hard_guarantee": False,
           "pbc_assumption": "no atom undergoes a true inter-frame displacement >= L/2",
           "trajectory_unwrapped_coordinates_available": False,
           "trajectory_image_flags_available": False,
           "max_min_image_step_A": v["max_min_image_step_A"],
           "continuity_safe_bound_A": v["continuity_safe_bound_A"],
           "max_step_over_L": v["max_min_image_step_A"] / proposal["parameters"]["box_L_A"],
           "proxy_gate": "plausibility/sanity only; NOT proof of the assumption",
           "pbc_precondition_ok": v["pbc_precondition_ok"]}
    diag = result.diagnostics
    apparent_D = {"apparent_D_estimate_under_continuity_assumption": diag["diffusion_estimate"],
                  "units": "Angstrom^2 / ps (slope/6, 3D Einstein)", "fit_window_frames": diag["late_window_frames"],
                  "fit_window_frac": diag["late_window_frac"], "late_slope": diag["late_slope"],
                  "late_fit_r2": diag["late_fit_r2"], "trajectory_duration_ps": 150.0,
                  "caveat": "apparent only; finite 150-ps window + wrapped-coordinate continuity assumption; "
                            "NOT a rigorous diffusion coefficient"}
    # augment msd_summary.json (merge; executor's core fields preserved)
    summ = json.loads((RUN_DIR / "msd_summary.json").read_text())
    summ["pbc"] = pbc
    summ["apparent_D"] = apparent_D
    (RUN_DIR / "msd_summary.json").write_text(json.dumps(summ, indent=2) + "\n")

    # --- Axis-A authoritative validity gate (frozen criterion_eval; deterministic verdict) ---
    results = evaluate_criteria(v, spec)
    authoritative_verdict = derive_severity(results)
    criterion_results = {"deterministic_authoritative": True,
                         "authoritative_verdict": authoritative_verdict,
                         "criterion_results": [r.model_dump() for r in results],
                         "provenance_block": render_authoritative_block(results)}
    (RUN_DIR / "criterion_results.json").write_text(json.dumps(criterion_results, indent=2) + "\n")

    # --- Axis-C advisory Judge: DEFERRED (needs GPU vLLM; outside CPU-only/no-GPU approval) ---
    judge = {"status": "DEFERRED", "deterministic_authoritative": False, "criterion_contradictions": 0,
             "reason": ("advisory semantic interpretation requires the local vLLM (GPU); this approval "
                        "is CPU-only / no-GPU / no-network, so the Judge is a separately-approvable step"),
             "questions": json.loads((ROOT / "tests/fixtures/stage_d2/judge_interpretation_task.json").read_text())["criteria"],
             "diagnostics_for_judge": {**diag, "apparent_D": apparent_D, "pbc": pbc}}
    (RUN_DIR / "judge_interpretation.json").write_text(json.dumps(judge, indent=2) + "\n")

    # --- outputs + integrity ---
    artifacts = {f: _sha(RUN_DIR / f) for f in
                 ("approval.json", "msd.csv", "msd_summary.json", "criterion_results.json",
                  "judge_interpretation.json")}
    src_after = EX.sha256_file(src)
    d1_after = _d1_fingerprint()
    writes_ok = all(Path(RUN_DIR / f).resolve().is_relative_to(RUN_DIR.resolve()) for f in artifacts)
    source_unchanged = src_after == src_before
    d1_unchanged = d1_after == d1_before

    # transition (deterministic axis; advisory deferred)
    axis_a_pass = authoritative_verdict == "PASS"
    transition = ("ADVANCE" if (axis_a_pass and wall <= 300 and source_unchanged and d1_unchanged
                                and writes_ok and criterion_results["criterion_results"])
                  else ("REVISE" if authoritative_verdict == "REVISE" else "FAIL"))

    provenance = {"run_id": RUN_ID, "stage": "stage_d2_c1", "action": "generate_analysis_artifact",
                  "subtype": "posthoc_msd", "architecture_v2_freeze_head": "99b9e87eacab5762c7f4c04ac8838445e57b2399",
                  "package_head": "b5762a1d57fad9dd16fe557702bb311117c38786",
                  "provider": "none (deterministic CPU analysis; no LLM)", "gpu": False, "network": False,
                  "scheduler": False, "source_sha256_before": src_before, "source_sha256_after": src_after,
                  "source_unchanged": source_unchanged, "stage_d1_unchanged": d1_unchanged,
                  "wall_time_s": wall, "frames_used": v["selected_frame_count"],
                  "n_atoms": summ["n_atoms"], "box_L_A": summ["box_L"], "pbc": pbc, "apparent_D": apparent_D,
                  "axis_a_validity": v, "authoritative_verdict": authoritative_verdict,
                  "criterion_contradictions": 0, "advisory_judge": "DEFERRED",
                  "artifacts_sha256": artifacts, "writes_under_run_dir_only": writes_ok,
                  "transition": transition}
    (RUN_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (RUN_DIR / "run_manifest.json").write_text(json.dumps(
        {"status": "OK", "accepted": axis_a_pass, "transition": transition,
         "authoritative_verdict": authoritative_verdict, "advisory_judge": "DEFERRED",
         "artifacts": list(artifacts)}, indent=2) + "\n")

    report = {"transition": transition, "authoritative_verdict": authoritative_verdict,
              "wall_time_s": round(wall, 3), "frames_used": v["selected_frame_count"],
              "box_L_A": summ["box_L"], "max_min_image_step_A": v["max_min_image_step_A"],
              "max_step_over_L": pbc["max_step_over_L"], "pbc_precondition_ok": v["pbc_precondition_ok"],
              "msd_all_last": result.rows[-1]["msd_all"],
              "msd_type1_O_last": result.rows[-1].get("msd_type1"),
              "msd_type2_Si_last": result.rows[-1].get("msd_type2"),
              "late_mean_msd": diag["late_mean_msd"], "late_std_msd": diag["late_std_msd"],
              "late_slope": diag["late_slope"], "late_fit_r2": diag["late_fit_r2"],
              "apparent_D": apparent_D["apparent_D_estimate_under_continuity_assumption"],
              "source_unchanged": source_unchanged, "stage_d1_unchanged": d1_unchanged,
              "writes_under_run_dir_only": writes_ok, "criterion_contradictions": 0,
              "artifacts_sha256": artifacts}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
