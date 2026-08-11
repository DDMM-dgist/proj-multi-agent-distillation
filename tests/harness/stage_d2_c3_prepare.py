#!/usr/bin/env python3
"""Stage D-2 C3 PREPARATION planner — one teacher (Allegro) single-point on the mini216 a-SiO2 cell.

Writes PLANNED manifests/templates only — NO inference, NO model load, NO result files. The typed
proposal is validated against the FROZEN DataCuratorActionProposal (action_type=label_with_teacher —
the existing allow-listed, approval-gated 'costly_teacher_labeling' action; no new action_type). The
deterministic criteria separate (A) artifact/computation validity from (B) reused frozen SiO2 DFT-scale
physical-validity, and an OPTIONAL advisory Judge covers (C) semantic interpretation. Deterministic; no
network/model/GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests" / "fixtures" / "stage_d2_c3"
sys.path.insert(0, str(ROOT))
from runtimes.pydantic_ai.actions import DataCuratorActionProposal  # noqa: E402  (frozen schema)
from runtimes.pydantic_ai.stage_d2_c3_teacher_executor import (  # noqa: E402
    E_PER_ATOM_RANGE, MAX_FORCE_BOUND)

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
SRC = f"{RES}/teacher_diag/nve_drift/mini216_nvt_fixed.data"
SRC_SHA = "3d2dd2464d83ca144e2c6d51382b83546b1152da06a386bfe3672550f4348364"
MODEL = f"{RES}/gpu_finetune_handoff/models/teacher_current_compiled.nequip.pth"
MODEL_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
N_ATOMS = 216
RUN_ID = "d2c3-teacher-sp-mini216"
RUN_DIR = f"tests/fixtures/stage_d2_c3/{RUN_ID}"

PROPOSAL = {
    "schema_version": 1, "run_id": RUN_ID, "stage": "stage_d2_c3",
    "requested_at": "2026-08-09T00:00:00Z", "requested_by_role": "data-curator",
    "action_type": "label_with_teacher",
    "rationale": "Stage D-2 C3: one real teacher (Allegro) single-point on the gate-confirmed "
                 "representative 216-atom a-SiO2 mini-cell (mini216_nvt_fixed.data), which carries NO "
                 "existing teacher E/F artifact. Produces a genuinely NEW E/F artifact through the full "
                 "proposal->approval->trusted-execution->deterministic-validation->provenance loop. One "
                 "forward pass; no MD/DFT/training.",
    "input_artifacts": [{"role": "structure", "path": SRC, "integrity": {"sha256": SRC_SHA}},
                        {"role": "teacher_model", "path": MODEL, "integrity": {"sha256": MODEL_SHA}}],
    "input_artifact_hashes": {SRC: SRC_SHA, MODEL: MODEL_SHA},
    "parameters": {
        "subtype": "teacher_single_point",
        "source_structure": SRC, "source_sha256": SRC_SHA,
        "teacher_model": MODEL, "model_sha256": MODEL_SHA,
        "read_allow_prefixes": [f"{RES}/teacher_diag/nve_drift/", f"{RES}/gpu_finetune_handoff/models/"],
        "expected_n_atoms": N_ATOMS, "composition": {"O": 144, "Si": 72},
        "type_symbol_map": {"1": "O", "2": "Si"}, "cutoff_A": 5.0,
        "software_env": "nequip 0.15 / allegro 0.7.1 + torch (compiled .nequip.pth inference env)",
        "one_forward_pass": True, "device": "one explicitly selected RTX 6000 Ada GPU",
        "no_scheduler": True, "no_training": True, "no_md": True, "no_dft": True,
        "no_paid_api": True, "no_overwrite": True, "run_dir": RUN_DIR,
        "estimated_gpu_mem_mib": "< 1000 (216 atoms, one forward pass)", "estimated_runtime": "seconds",
    },
    "expected_outputs": [f"{RUN_DIR}/teacher_ef.json", f"{RUN_DIR}/forces.csv",
                         f"{RUN_DIR}/criterion_results.json", f"{RUN_DIR}/provenance.json",
                         f"{RUN_DIR}/run_manifest.json"],
    "estimated_cost": "one teacher forward pass (GPU-seconds)", "estimated_runtime": "<= 1 min",
    "approval_boundary": "costly_teacher_labeling", "idempotency_key": f"{RUN_ID}-v1", "dry_run": True,
    "required_validator": "stage_d2_c3_teacher_ef_validity",
    "rollback_or_cleanup_policy": ("source structure + teacher model + Stage D-1 tree read-only; executor "
                                   "refuses an existing run dir (idempotency); on failure the run dir is "
                                   "removed and no partial artifact is accepted as final."),
}

INV = True
# A. ARTIFACT / COMPUTATION VALIDITY (deterministic_authoritative=true)
VALIDITY_SPEC = [
    {"criterion": "input structure SHA256 matches the proposal", "operator": "eq", "lhs": {"field": "input_sha256_matches"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "teacher model SHA256 matches the proposal", "operator": "eq", "lhs": {"field": "model_sha256_matches"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "structure parsed successfully", "operator": "eq", "lhs": {"field": "structure_parsed"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "expected atom count preserved (216)", "operator": "eq", "lhs": {"field": "atom_count_preserved"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "total energy finite", "operator": "eq", "lhs": {"field": "energy_finite"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "energy per atom finite", "operator": "eq", "lhs": {"field": "energy_per_atom_finite"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "forces finite", "operator": "eq", "lhs": {"field": "forces_finite"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "force array shape is N x 3", "operator": "eq", "lhs": {"field": "force_shape_is_Nx3"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "max |F| finite", "operator": "eq", "lhs": {"field": "max_force_finite"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "source structure + teacher model unchanged after run", "operator": "eq", "lhs": {"field": "source_model_unchanged"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "writes occur only under the fresh C3 run directory", "operator": "eq", "lhs": {"field": "writes_under_run_dir_only"}, "rhs": {"const": True}, "invalidating": INV},
    {"criterion": "artifact hashes recorded", "operator": "eq", "lhs": {"field": "artifact_hashes_recorded"}, "rhs": {"const": True}},
    # B. PHYSICAL VALIDITY (REUSED frozen SiO2 DFT-scale ranges; provenance in CONTRACT.md)
    {"criterion": "energy per atom is physical (between -11 and -8 eV/atom; reused frozen SiO2 DFT-scale range)", "operator": "in_range", "lhs": {"field": "E_per_atom_eV"}, "rhs": {"low": E_PER_ATOM_RANGE[0], "high": E_PER_ATOM_RANGE[1]}, "invalidating": INV},
    {"criterion": "max |F| is physical (<= 50 eV/Angstrom; reused frozen SiO2 sanity bound)", "operator": "le", "lhs": {"field": "max_force_eV_A"}, "rhs": {"const": MAX_FORCE_BOUND}, "invalidating": INV},
]

# C. OPTIONAL advisory semantic Judge (deterministic_authoritative=false)
JUDGE_TASK = {
    "schema_version": 1, "task_id": f"{RUN_ID}-interpretation", "agent": "judge",
    "created_at": "2026-08-09T00:00:00Z",
    "instruction": (f"Read teacher_ef.json for this run via read_json on '{RUN_DIR}/teacher_ef.json' "
                    "(full repo-relative path; do NOT use a bare filename; do NOT read any manifest). "
                    "Interpret ONLY from that artifact. Do NOT recompute the model. Do NOT invent a "
                    "threshold. The authoritative artifact + physical validity (Axis A/B) is decided "
                    "deterministically and is not yours to change."),
    "inputs": [], "criteria": [
        "The predicted energy per atom is consistent with a representative amorphous a-SiO2 single-point "
        "(comparable in scale to the DFT-labeled AL cells, ~ -9.4 to -9.9 eV/atom)",
        "The predicted force magnitudes are physically reasonable for an equilibrated a-SiO2 mini-cell"],
    "constraints": [f"read-only: read_json on {RUN_DIR}/teacher_ef.json only; no bare filename; no manifest",
                    "advisory semantic interpretation only; do not recompute; do not invent thresholds",
                    "do not alter any artifact; the deterministic Axis-A/B validity is authoritative"],
    "context": {"review_lens": "scientific_validity",
                "review_focus": "Advisory interpretation of the teacher single-point E/F on mini216 a-SiO2.",
                "deterministic_authoritative": False,
                "note": "ADVISORY only; separate from and does not override the deterministic validity gate."},
}

INPUT_MANIFEST = {"source_structure": SRC, "source_sha256": SRC_SHA, "size_bytes": 18098,
                  "format": "LAMMPS data (atomic)", "n_atoms": N_ATOMS, "atom_types": 2,
                  "type_symbol_map": {"1": "O", "2": "Si"}, "composition": {"O": 144, "Si": 72},
                  "box_L_A": 14.835545077426339, "provenance": "teacher_diag/nve_drift; gate-confirmed "
                  "representative amorphous a-SiO2 (error(d) mini-cell)",
                  "already_teacher_labeled": False}
MODEL_MANIFEST = {"teacher_model": MODEL, "model_sha256": MODEL_SHA, "size_bytes": 4905990,
                  "framework": "nequip/allegro (compiled .nequip.pth)", "cutoff_A": 5.0,
                  "chemical_symbols": ["O", "Si"],
                  "identity_note": "teacher_current_compiled.nequip.pth (gpu_finetune_handoff/models). "
                  "Confirm base-vs-finetuned teacher version before execution if it matters for the claim."}
RUN_MANIFEST_TEMPLATE = {"_status": "PLANNED_TEMPLATE — populated only at execution under tests/fixtures/stage_d2_c3/<run_id>/",
                         "run_id": RUN_ID, "stage": "stage_d2_c3", "action_type": "label_with_teacher",
                         "subtype": "teacher_single_point",
                         "planned_artifacts": {"action_proposal.json": "PLANNED", "input_manifest.json": "PLANNED",
                         "model_manifest.json": "PLANNED", "approval.json": "GENERATED at approval",
                         "teacher_ef.json": "GENERATED at execution", "forces.csv": "GENERATED",
                         "criterion_results.json": "GENERATED", "provenance.json": "GENERATED",
                         "run_manifest.json": "GENERATED"},
                         "guards": ["explicit human approval (costly_teacher_labeling)", "fresh run dir (no overwrite)",
                         "source+model read-only + sha256 verified", "one forward pass, one GPU",
                         "no scheduler / MD / DFT / training / paid API / network", "writes only under run dir"]}


def main():
    (BASE / "criteria").mkdir(parents=True, exist_ok=True)
    DataCuratorActionProposal(**PROPOSAL)   # validate against the FROZEN schema (raises on drift)
    (BASE / "action_proposal.json").write_text(json.dumps(PROPOSAL, indent=2) + "\n")
    (BASE / "input_manifest.json").write_text(json.dumps(INPUT_MANIFEST, indent=2) + "\n")
    (BASE / "model_manifest.json").write_text(json.dumps(MODEL_MANIFEST, indent=2) + "\n")
    (BASE / "criteria" / "teacher_ef_validity.json").write_text(json.dumps(VALIDITY_SPEC, indent=2) + "\n")
    (BASE / "judge_interpretation_task.json").write_text(json.dumps(JUDGE_TASK, indent=2) + "\n")
    (BASE / "run_manifest.template.json").write_text(json.dumps(RUN_MANIFEST_TEMPLATE, indent=2) + "\n")
    print("wrote Stage D-2 C3 PLANNED artifacts (proposal validated vs frozen DataCuratorActionProposal):")
    for p in ("action_proposal.json", "input_manifest.json", "model_manifest.json",
              "criteria/teacher_ef_validity.json", "judge_interpretation_task.json", "run_manifest.template.json"):
        print(f"  tests/fixtures/stage_d2_c3/{p}")
    print("NO inference; NO model load; NO result files; run dir NOT created.")


if __name__ == "__main__":
    main()
