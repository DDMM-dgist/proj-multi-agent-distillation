#!/usr/bin/env python3
"""Stage D-2 C1 PREPARATION planner (post-hoc MSD). Writes PLANNED manifests/templates only — NO
scientific result files, NO execution. The typed proposal is validated against the FROZEN
AnalystActionProposal schema (action_type=summarize_md_stability; posthoc_msd carried in parameters,
since actions.py is frozen and adds no new action_type). Deterministic; NO network/model/GPU.

PLANNED artifacts written here: action_proposal.json, input_manifest.json, run_manifest.template.json,
criteria/posthoc_msd_validity.json, judge_interpretation_task.json. GENERATED-at-execution artifacts
(msd.csv, msd_summary.json, criterion_results.json, judge_interpretation.json, provenance.json,
run_manifest.json under runs/stage_d2/<run_id>/) are NOT created here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "examples" / "stage_d2"
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai.actions import AnalystActionProposal  # noqa: E402  (frozen schema)
from runtimes.pydantic_ai.stage_d2_executor import (  # noqa: E402
    CONTINUITY_SAFE_FRAC, INITIAL_MSD_TOL, LATE_WINDOW_FRAC, MAX_FRAMES, MIN_FRAMES, WALLTIME_CEILING_S)

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
SRC = f"{RES}/sio2x_production/random_x006/traj.dump"
SRC_SHA = "53ddba6a02747efb9d545415ec6468ff41c76c5a845f7afdf4cc7e71c3067591"
BOX_L = 35.49749412495787
N_ATOMS = 2940
RUN_ID = "d2c1-posthoc-msd-random_x006"
RUN_DIR = f"runs/stage_d2/{RUN_ID}"

PROPOSAL = {
    "schema_version": 1, "run_id": RUN_ID, "stage": "stage_d2_c1",
    "requested_at": "2026-08-09T00:00:00Z", "requested_by_role": "analyst",
    "action_type": "summarize_md_stability",
    "rationale": "Stage D-2 C1: first real state-advancing action. Compute a post-hoc MSD (a new "
                 "dynamical artifact never produced for this trajectory) from an existing 300 K NVT "
                 "production trajectory to characterize amorphous-solid displacement, exercising the "
                 "full propose->approve->execute->artifact->deterministic-validation->Judge loop.",
    "input_artifacts": [{"role": "production_trajectory", "path": SRC, "integrity": {"sha256": SRC_SHA}}],
    "input_artifact_hashes": {SRC: SRC_SHA},
    "parameters": {
        "subtype": "posthoc_msd",
        "source_trajectory": SRC, "source_sha256": SRC_SHA,
        "source_allow_prefixes": [f"{RES}/sio2x_production/"],
        "selected_frame_rule": f"all frames if count<= {MAX_FRAMES} else even stride to <= {MAX_FRAMES}",
        "max_frames": MAX_FRAMES, "min_frames": MIN_FRAMES,
        "analysis_method": "mean-square displacement from frame 0, mean over atoms; per-type MSD too",
        "pbc_method": ("minimum-image CONTINUITY unwrapping: per-frame min-image component steps "
                       "accumulated (fixed cubic box). Dump is WRAPPED-only (no xu/yu/zu, no image "
                       "flags), so absence of a true >=L/2 jump is NOT provable; proxy gate STOPS if "
                       f"max min-image step >= {CONTINUITY_SAFE_FRAC}*L. Valid under the physical "
                       "assumption per-frame displacement << L/2 (300 K solid, 1 ps cadence)."),
        "timestep_ps": 0.001, "dump_interval_steps": 1000, "box_L_A": BOX_L, "n_atoms": N_ATOMS,
        "columns_available": ["id", "type", "x", "y", "z", "fx", "fy", "fz"],
        "type_species_map": {"1": "O", "2": "Si"},
        "late_window_frac": LATE_WINDOW_FRAC, "initial_msd_tol": INITIAL_MSD_TOL,
        "wall_time_ceiling_s": WALLTIME_CEILING_S,
        "cpu_only": True, "single_process": True, "no_scheduler": True, "no_gpu": True,
        "no_network": True, "no_overwrite": True, "run_dir": RUN_DIR,
    },
    "expected_outputs": [f"{RUN_DIR}/msd.csv", f"{RUN_DIR}/msd_summary.json",
                         f"{RUN_DIR}/criterion_results.json", f"{RUN_DIR}/judge_interpretation.json",
                         f"{RUN_DIR}/provenance.json", f"{RUN_DIR}/run_manifest.json"],
    "estimated_cost": "CPU-minutes", "estimated_runtime": "<= 5 min",
    "approval_boundary": "stage_d2_first_state_advancing_action",
    "idempotency_key": f"{RUN_ID}-v1", "dry_run": True,
    "required_validator": "stage_d2_posthoc_msd_validity",
    "rollback_or_cleanup_policy": ("source trajectory + Stage D-1 tree read-only; executor refuses an "
                                   "existing run dir; on any failure the run dir is removed and no "
                                   "partial output is accepted as final."),
}

# Authoritative artifact/computation-validity criteria (deterministic_authoritative=true), frozen ops.
INV = True
VALIDITY_SPEC = [
    {"criterion": "input trajectory exists", "operator": "eq", "lhs": {"field": "input_exists"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "input SHA256 matches the proposal", "operator": "eq", "lhs": {"field": "input_sha256_matches"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "trajectory parsed successfully", "operator": "eq", "lhs": {"field": "parsed_ok"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "PBC continuity precondition holds (max min-image step < 0.25*L)", "operator": "eq", "lhs": {"field": "pbc_precondition_ok"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "atom count constant across selected frames", "operator": "eq", "lhs": {"field": "atom_count_constant"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "timesteps strictly increasing", "operator": "eq", "lhs": {"field": "timesteps_increasing"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "MSD values finite", "operator": "eq", "lhs": {"field": "msd_all_finite"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "MSD values non-negative", "operator": "eq", "lhs": {"field": "msd_all_nonneg"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "initial MSD approximately zero", "operator": "le", "lhs": {"field": "msd_initial_abs"}, "rhs": {"const": INITIAL_MSD_TOL}, "invalidating": INV},
    {"criterion": "source trajectory byte-identical after run", "operator": "eq", "lhs": {"field": "source_byte_identical_after"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "writes occur only under the fresh run directory", "operator": "eq", "lhs": {"field": "writes_under_run_dir_only"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "selected frame count <= 200", "operator": "le", "lhs": {"field": "selected_frame_count"}, "rhs": {"const": MAX_FRAMES}},
    {"criterion": "selected frame count >= predeclared minimum", "operator": "ge", "lhs": {"field": "selected_frame_count"}, "rhs": {"const": MIN_FRAMES}},
    {"criterion": "output row count matches selected-frame count", "operator": "eq", "lhs": {"field": "output_row_count_matches"}, "rhs": {"const": True}},
    {"criterion": "all required summary fields exist", "operator": "eq", "lhs": {"field": "summary_fields_present"}, "rhs": {"const": True}},
    {"criterion": "output SHA256 recorded", "operator": "eq", "lhs": {"field": "output_sha256_present"}, "rhs": {"const": True}},
    {"criterion": "runtime within the 5-minute ceiling", "operator": "eq", "lhs": {"field": "runtime_ok"}, "rhs": {"const": True}},
]

JUDGE_TASK = {
    "schema_version": 1, "task_id": f"{RUN_ID}-interpretation", "agent": "judge",
    "created_at": "2026-08-09T00:00:00Z",
    "instruction": ("Read msd.csv + msd_summary.json for this run. Interpret ONLY from the generated "
                    "artifact + declared diagnostics; do NOT fabricate a plateau/diffusion threshold."),
    "inputs": [], "criteria": [
        "Is the computed MSD behavior consistent with a bounded amorphous-solid trajectory over the observed window?",
        "Is there evidence of sustained diffusion over the observed window?",
        "Is the trajectory/window sufficient to support the interpretation, or is longer analysis required?"],
    "constraints": ["read-only; ground the interpretation only in the generated artifact + diagnostics"],
    "context": {"review_lens": "scientific_validity",
                "review_focus": "Semantic (advisory) interpretation of post-hoc MSD dynamics.",
                "deterministic_authoritative": False,
                "note": ("ADVISORY gate: no source-grounded numeric plateau/diffusion threshold exists, "
                         "so the physical interpretation is a genuine semantic Judge verdict. The "
                         "artifact-validity gate is separate and authoritative.")},
}

RUN_MANIFEST_TEMPLATE = {
    "_status": "PLANNED_TEMPLATE — populated only at execution under runs/stage_d2/<run_id>/",
    "run_id": RUN_ID, "stage": "stage_d2_c1", "action_type": "summarize_md_stability",
    "subtype": "posthoc_msd", "architecture_v2_freeze_head": "99b9e87eacab5762c7f4c04ac8838445e57b2399",
    "planned_artifacts": {
        "action_proposal.json": "PLANNED (this preparation)", "input_manifest.json": "PLANNED",
        "approval.json": "GENERATED at approval (human)", "msd.csv": "GENERATED at execution",
        "msd_summary.json": "GENERATED", "criterion_results.json": "GENERATED",
        "judge_interpretation.json": "GENERATED", "provenance.json": "GENERATED",
        "run_manifest.json": "GENERATED"},
    "guards": ["explicit human approval required", "fresh run dir (no overwrite)",
               "source read-only + sha256 verified", "writes only under run dir",
               "no scheduler / MD / DFT / training / GPU / network", "wall-time <= 5 min, CPU only"],
}

INPUT_MANIFEST = {
    "source_trajectory": SRC, "source_sha256": SRC_SHA, "size_bytes": 25856871,
    "n_frames": 151, "n_atoms": N_ATOMS, "box_L_A": BOX_L, "ensemble": "NVT 300 K, 150 ps",
    "timestep_ps": 0.001, "dump_interval_steps": 1000, "frame_spacing_ps": 1.0,
    "columns": ["id", "type", "x", "y", "z", "fx", "fy", "fz"],
    "coordinates": "WRAPPED (no unwrapped xu/yu/zu, no image flags)",
    "type_species_map": {"1": "O", "2": "Si"}, "species_counts": {"1": 1940, "2": 1000},
    "pbc_reconstruction": "minimum-image continuity (proxy-gated; wrapped-only assumption documented)",
    "d2_c1_input_insufficient_for_valid_msd": False,
}


def main():
    (BASE / "criteria").mkdir(parents=True, exist_ok=True)
    AnalystActionProposal(**PROPOSAL)   # validate against the FROZEN schema (raises on drift)
    (BASE / "action_proposal.json").write_text(json.dumps(PROPOSAL, indent=2) + "\n")
    (BASE / "input_manifest.json").write_text(json.dumps(INPUT_MANIFEST, indent=2) + "\n")
    (BASE / "criteria" / "posthoc_msd_validity.json").write_text(json.dumps(VALIDITY_SPEC, indent=2) + "\n")
    (BASE / "judge_interpretation_task.json").write_text(json.dumps(JUDGE_TASK, indent=2) + "\n")
    (BASE / "run_manifest.template.json").write_text(json.dumps(RUN_MANIFEST_TEMPLATE, indent=2) + "\n")
    print("wrote Stage D-2 C1 PLANNED artifacts (proposal validated against frozen AnalystActionProposal):")
    for p in ("action_proposal.json", "input_manifest.json", "criteria/posthoc_msd_validity.json",
              "judge_interpretation_task.json", "run_manifest.template.json"):
        print(f"  examples/stage_d2/{p}")
    print("NOTE: NO scientific result files created; run dir NOT created; execution NOT performed.")


if __name__ == "__main__":
    main()
