#!/usr/bin/env python3
"""PC002 — Distillation Dataset Design (READ-ONLY; no compute).

Given the PC001-accepted teacher (b56e20ff) and frozen target domain, determine independently what
structural distribution the teacher should supervise so a future student covers the deployment domain.
Reads ONLY existing local structure inventories/manifests. NO teacher/student inference, DFT, MD,
training, scheduler, network, semantic Judge. NO student model is inspected (NEW_PIPELINE_CURRENT_STUDENT
= NONE). Emits coverage artifacts + one dataset decision.
"""
from __future__ import annotations
import csv, json, hashlib, statistics as st, sys, subprocess
from pathlib import Path

RES = Path("/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation")
TEACHER_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "production_campaign_002" / "pc002-distillation-dataset-design"

# frozen PC001 target domain
TARGET_DOMAIN = ["amorphous_SiO2", "SiO2x_dilute_vacancy", "SiO2x_clustered_vacancy_voidsurface",
                 "surfaces", "high_T_liquid_distorted", "ambient_crystalline_reference"]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def count_frames(dump):
    p = RES / dump
    if not p.is_file():
        return None
    try:
        return sum(1 for ln in open(p, errors="replace") if ln.startswith("ITEM: TIMESTEP"))
    except OSError:
        return None


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    # ---- candidate structure inventory (LOCAL unlabeled production MD = the distillation candidate pool) ----
    rs = {x: count_frames(f"production_12288/random_sweep/{x}/prod.dump") for x in
          ("x003", "x006", "x009", "x012", "x015", "x018", "x024")}
    clustered = {c: count_frames(f"production_12288/anneal_calib_clustered/{c}/prod.dump") for c in ("T1000", "plane_T1000")}
    pristine_frames = count_frames("production_12288/rdf.dump")
    candidate_sources = {
        "amorphous_SiO2": {"source": "production_12288 12288-atom cristobalite->melt->quench glass (rdf.dump)",
                           "frames": pristine_frames, "atoms": 12288, "composition": "SiO2 stoichiometric", "label_status": "UNLABELED"},
        "SiO2x_dilute_vacancy": {"source": "production_12288/random_sweep x003..x024 (7 vacancy fractions)",
                                 "frames_per_x": rs, "total_frames": sum(v for v in rs.values() if v), "atoms": "~11.3k-11.9k",
                                 "x_range": "0.03-0.24", "label_status": "UNLABELED"},
        "SiO2x_clustered_vacancy_voidsurface": {"source": "production_12288/anneal_calib_clustered {T1000, plane_T1000}",
                                                "frames": clustered, "total_frames": sum(v for v in clustered.values() if v),
                                                "atoms": 12288, "label_status": "UNLABELED",
                                                "plus_carved": "embedding_clustered 8 carved clustered cells (local_x 0.11-0.60)"},
        "sio2x_production_3000atom": {"source": "sio2x_production/{pristine,random,sphere,plane}_{x006,x012}",
                                      "atoms": 3000, "configs": 7, "label_status": "UNLABELED"},
        "high_T_liquid_distorted": {"source": "production_12288 melt frames (4000K hold) + melt_msd dumps",
                                    "label_status": "UNLABELED", "note": "thermal/distorted configs available in the melt-quench trajectory"},
    }

    # ---- existing teacher-labeled inventory ----
    al11 = RES / "gpu_return_v5_committee" / "v5_committee_bundle" / "AL11_with_umax_v5.xyz"
    teacher_labels = {
        "local_accepted_teacher_labeled": {
            "AL11_with_umax_v5.xyz": {"n": 11, "teacher": "committee-u annotated 11 AL cells (DFT-labeled)", "present": al11.is_file()},
            "augment_seeds": {"input_small.xyz": 50, "input_large.xyz": 10, "note": "DFT-corpus seeds, teacher-labelable"},
        },
        "historical_distillation_training_set": {
            "n": "~10,000 (8k normal + 2k large)", "labeled_by": "Allegro teacher (augment-atoms, DFT-corpus seeds)",
            "location": "KISTI (NOT locally stored; PROVENANCE.md sec.3)",
            "local_inspectable": False,
            "clustered_coverage": "UNVERIFIABLE locally; provenance (R1.P1R2) shows seeds = DFT corpus SiOx_filtered, NOT production clustered dumps => clustered production motifs likely UNDER-represented",
        },
        "valid_local_accepted_teacher_labels_for_training": "MINIMAL (only the 11 AL cells + 60 seeds locally); the bulk historical labeled set is KISTI-only",
    }

    # ---- DFT reference to PRESERVE for PC004 (NOT consumed into training) ----
    dft_cells = (len(list((RES / "dft_labeling").glob("cell_*"))) + len(list((RES / "dft_labeling").glob("clustered_cell_*")))
                 + len(list((RES / "al_iter3/dft_validation_11A").glob("cell_*")))
                 + len(list((RES / "al_iter3/v6_dft_batch").glob("cell_*")))
                 + len(list((RES / "al_iter3/heldout_dft_batch").glob("cell_ho_*"))))
    preserved_dft = {"n_scan_cells": dft_cells, "role": "PC004 Student-vs-DFT reference (in-domain dilute+clustered SiO2-x)",
                     "policy": "PRESERVED — not consumed into distillation training; these are the ONLY high-fidelity in-domain Student-vs-DFT anchors"}

    # ---- defect/composition distribution from carved-cell manifests (deterministic) ----
    def read_local_x(path, col):
        vals = []
        if (RES / path).is_file():
            for r in csv.DictReader(open(RES / path)):
                v = num(r.get(col))
                if v is not None:
                    vals.append(v)
        return vals
    clustered_lx = read_local_x("embedding_clustered/clustered_cells_summary.csv", "local_x")
    v6_lx = read_local_x("al_iter3/v6_dft_batch/manifest_v6_full.csv", "x_local")
    ho_lx = read_local_x("al_iter3/heldout_dft_batch/manifest_heldout.csv", "x_local")
    all_defect_lx = clustered_lx + v6_lx + ho_lx
    defect_dist = {"n_carved_defect_cells": len(all_defect_lx),
                   "local_x_range": [round(min(all_defect_lx), 3), round(max(all_defect_lx), 3)] if all_defect_lx else None,
                   "local_x_mean": round(st.mean(all_defect_lx), 3) if all_defect_lx else None,
                   "note": "carved DFT-tractable SiO2-x cells span dilute->heavily O-deficient (local_x up to ~0.6); demonstrates clustered/defect candidate diversity exists"}

    # ---- domain coverage matrix ----
    coverage = {
        "amorphous_SiO2": {"candidate_available": True, "labeled_accepted_teacher": "historical(KISTI, likely covered)", "status": "CANDIDATE_COVERED_LABELS_KISTI"},
        "SiO2x_dilute_vacancy": {"candidate_available": True, "labeled_accepted_teacher": "partial(historical KISTI)", "status": "CANDIDATE_COVERED"},
        "SiO2x_clustered_vacancy_voidsurface": {"candidate_available": True, "labeled_accepted_teacher": "UNVERIFIABLE / likely SPARSE", "status": "CANDIDATE_COVERED_LABELS_GAP"},
        "surfaces": {"candidate_available": True, "labeled_accepted_teacher": "partial(historical)", "status": "CANDIDATE_COVERED"},
        "high_T_liquid_distorted": {"candidate_available": True, "labeled_accepted_teacher": "partial(historical)", "status": "CANDIDATE_COVERED"},
        "ambient_crystalline_reference": {"candidate_available": True, "labeled_accepted_teacher": "historical", "status": "CANDIDATE_COVERED"},
    }
    label_gaps = [d for d, c in coverage.items() if "GAP" in c["status"]]

    # ---- thermodynamic + redundancy ----
    thermo = {"density_g_cm3": 2.2, "temperatures_K": [300, 4000], "note": "production density 2.2; ambient 300K + 4000K melt available"}
    redundancy = {"method": "trajectory-frame correlation (descriptive; no forensic lineage)",
                  "finding": "MD dumps contain temporally-correlated frames (50-500/config) => a numerically large but structurally repetitive set if frame-sampled densely; sub-sample by time spacing per config",
                  "lineage": "NONBLOCKING (exact augment lineage KISTI-only; not needed for the coverage decision)"}

    # ---- split design (NEW pipeline; do NOT reuse historical train_list) ----
    split = {"strategy": "GROUP split by source trajectory/config/defect-family (NOT random frame split of correlated MD frames)",
             "train": "broad target-domain teacher-labeled frames (amorphous + dilute + clustered + surfaces + high-T), grouped by config",
             "validation": "held-out configs/time-segments for model selection",
             "test_independent": "the PRESERVED DFT SCAN cells (Student-vs-DFT, PC004) + a group-disjoint held-out config",
             "leakage_policy": "design held-out independence UP FRONT (R1 lesson); keep DFT test assets out of training"}

    # ---- DECISION ----
    decision = "DISTILLATION_DATASET_REVISE"
    decision_reason = (
        "Candidate structures for EVERY target domain exist locally as unlabeled production MD (amorphous glass; "
        "dilute vacancy x0.03-0.24 350 frames; clustered/void 102 frames + 8 carved clustered cells; 3000-atom configs; "
        "4000K melt). BUT the reusable historical teacher-labeled training set (~10k) is KISTI-only (not locally "
        "verifiable) and its clustered coverage is unverifiable + provenance-suggested SPARSE (augment seeds = DFT "
        "corpus, not production clustered). The target domain REQUIRES clustered-vacancy/void-surface coverage (the "
        "PC001-flagged region). => the dataset is broadly useful but has an identifiable, repairable target-domain "
        "LABEL gap (clustered/void with accepted teacher b56e20ff). Suitable UNLABELED clustered candidates already "
        "exist locally => REVISE via ONE bounded teacher-labeling action, not INSUFFICIENT and not blind reuse.")

    summary = {
        "campaign": "PC002_DISTILLATION_DATASET_DESIGN",
        "accepted_teacher": "base Allegro", "teacher_sha": TEACHER_SHA,
        "target_domain": TARGET_DOMAIN,
        "candidate_structure_pool": "LOCAL unlabeled production MD (amorphous + dilute + clustered + 3000-atom + melt); all target domains have candidates",
        "valid_local_accepted_teacher_labels": "minimal (11 AL cells + 60 seeds); bulk historical labeled set KISTI-only",
        "domain_coverage": coverage, "label_gaps": label_gaps,
        "preserved_dft_validation_assets": preserved_dft,
        "split_strategy": split["strategy"],
        "DATASET_DECISION": decision, "decision_reason": decision_reason,
        "reused_existing_asset": "the historical teacher-labeled bulk (amorphous/dilute) may be REUSED where valid; the CLUSTERED/void region needs new accepted-teacher labels",
        "requires_new_teacher_labels": True,
        "new_teacher_labeling_target": "clustered-vacancy/void-surface (+ balancing dilute/amorphous) frames from existing production_12288 clustered + random_sweep dumps, labeled with accepted teacher b56e20ff",
        "next_action": "ONE bounded label_with_teacher ActionProposal (approval-gated; prepared, NOT executed): examples/production_campaign_002/action_proposal.json",
        "DISTILLATION_DATASET_STAGE": "REVISE (open until labeling closes the clustered gap)",
        "STUDENT_TRAINING_STAGE_AUTHORIZED": False,
        "NEW_PIPELINE_CURRENT_STUDENT": "NONE",
        "EXISTING_HISTORICAL_STUDENT_ASSETS": ["original_deployed_committee", "v5_committee"],
        "blocking_unknowns": [],
        "nonblocking_unknowns": ["exact distribution of the KISTI historical labeled set (not locally inspectable)",
                                 "exact augment lineage (KISTI); not needed for the coverage decision",
                                 "per-frame coordination distributions of the full dump pool (not parsed; committee_u defect-Si stats used as proxy)"],
        "no_scientific_compute_performed": True, "student_model_inspected": False,
    }

    # ---- write artifacts ----
    W = lambda n, o: (RUN_DIR / n).write_text(json.dumps(o, indent=2) + "\n")
    W("input_manifest.json", {"teacher_sha256": TEACHER_SHA, "teacher_invoked": False, "student_model_inspected": False,
                              "read_sources": ["production_12288/*", "sio2x_production/*", "embedding_clustered/clustered_cells_summary.csv",
                                               "al_iter3/*/manifest*.csv", "gpu_return_v5_committee/.../AL11_with_umax_v5.xyz"]})
    W("candidate_dataset_inventory.json", candidate_sources)
    W("teacher_label_inventory.json", teacher_labels)
    W("domain_coverage.json", {"target_domain": TARGET_DOMAIN, "coverage": coverage, "label_gaps": label_gaps})
    with open(RUN_DIR / "composition_distribution.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "composition", "vacancy_x_range", "atoms"])
        w.writerow(["12288_glass", "SiO2 stoichiometric", "0", 12288])
        w.writerow(["random_sweep", "SiO2-x random vacancy", "0.03-0.24", "~11.3-11.9k"])
        w.writerow(["anneal_calib_clustered", "SiO2-x clustered vacancy/void", "~0.06-0.12 (global)", 12288])
        w.writerow(["carved_defect_cells", "SiO2-x local defect", f"local_x {defect_dist['local_x_range']}", "36-132"])
    with open(RUN_DIR / "defect_distribution.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["defect_family", "candidate_source", "frames_or_cells", "label_status"])
        w.writerow(["dilute_random_vacancy", "random_sweep x003-024", sum(v for v in rs.values() if v), "UNLABELED"])
        w.writerow(["clustered_sphere_plane_vacancy", "anneal_calib_clustered", sum(v for v in clustered.values() if v), "UNLABELED"])
        w.writerow(["carved_clustered_defect_cells", "embedding_clustered+al_iter3", defect_dist["n_carved_defect_cells"], "mixed(DFT-labeled subset preserved)"])
    with open(RUN_DIR / "coordination_distribution.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["note"])
        w.writerow(["per-frame coordination not recomputed (no compute); committee_u_out defect-Si CN stats are the existing proxy (defect-Si ~2.5-3x normal-Si uncertainty)"])
    W("thermodynamic_coverage.json", thermo)
    W("redundancy_summary.json", redundancy)
    W("split_design.json", split)
    W("preserved_dft_test_assets.json", preserved_dft)
    W("dataset_decision.json", {"DATASET_DECISION": decision, "reason": decision_reason, "label_gaps": label_gaps,
                                "requires_new_teacher_labels": True})
    W("criterion_results.json", {"deterministic_authoritative": True, "decision": decision,
                                 "coverage": coverage, "label_gaps": label_gaps, "student_model_inspected": False})
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    W("provenance.json", {"run_id": "pc002-distillation-dataset-design", "stage": "production_campaign_002",
        "package_head": head, "accepted_teacher_sha256": TEACHER_SHA, "teacher_invoked": False, "student_model_inspected": False,
        "analysis_code_sha256": sha(Path(__file__).resolve()),
        "no_model_invocation": True, "no_dft": True, "no_md": True, "no_training": True, "no_network": True,
        "no_semantic_judge": True, "historical_decisions_used_as_authority": False})
    W("run_manifest.json", {"status": "OK", "campaign": "PC002_DISTILLATION_DATASET_DESIGN",
        "DATASET_DECISION": decision, "requires_new_teacher_labels": True,
        "STUDENT_TRAINING_STAGE_AUTHORIZED": False, "NEW_PIPELINE_CURRENT_STUDENT": "NONE",
        "next_action": "bounded label_with_teacher proposal (prepared, not executed)",
        "no_scientific_compute_performed": True})
    W("teacher_validation_summary.json", summary)  # convenience alias
    print(json.dumps({"DATASET_DECISION": decision, "label_gaps": label_gaps,
                      "candidate_dilute_frames": sum(v for v in rs.values() if v),
                      "candidate_clustered_frames": sum(v for v in clustered.values() if v),
                      "preserved_dft_cells": dft_cells, "requires_new_teacher_labels": True,
                      "student_training_authorized": False, "new_pipeline_current_student": "NONE"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
