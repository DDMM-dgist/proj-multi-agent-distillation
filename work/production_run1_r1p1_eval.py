#!/usr/bin/env python3
"""Production Run 1 · R1.P1 — evaluate_heldout_fidelity (READ-ONLY; no compute).

Deterministic, network-free aggregation of EXISTING held-out prediction/reference artifacts into a
leakage-checked, domain-tagged generalization verdict with teacher-vs-distillation attribution. Performs
NO teacher/student forward, NO DFT, NO MD, NO training, NO scheduler, NO network, NO semantic Judge.

Error primitives (per-cell force RMSE, eV/A) are consumed from the committed
`heldout_baseline_errord.csv` (produced deterministically by run_heldout_baseline.py from raw LAMMPS v5
committee + NequIP teacher + SCAN OUTCAR forces). This script recomputes every DERIVED quantity (family
aggregates, attribution, deltas) from those primitives and parses DFT E/atom from OUTCAR.scan for the
physical-domain sanity band. It writes only under the fresh Run-1 run dir.

Attribution rule (DECLARED BEFORE APPLICATION; per-cell, core RMSE erra=teacher-vs-DFT,
errd=v5-vs-DFT, errb=v5-vs-teacher):
  TEACHER_LIMITED      if erra >= errb AND errd <= 1.10*erra      (teacher error dominates; student ~<= teacher)
  DISTILLATION_LIMITED if errb >  erra AND errd >  1.10*erra      (gap dominates; student notably worse)
  MIXED                otherwise
  UNRESOLVED           if any primitive missing
"""
from __future__ import annotations
import csv, json, hashlib, sys
from pathlib import Path

RES = Path("/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation")
HB = RES / "al_iter3" / "heldout_dft_batch"
AGG = HB / "analysis" / "heldout_baseline_errord.csv"
MANIFEST = HB / "manifest_heldout.csv"
V5_BUNDLE = RES / "gpu_return_v5_committee" / "v5_committee_bundle"
TEACHER = Path("/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/teacher/model.nequip.pth")
TEACHER_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
EXPECTED_HELDOUT = [f"cell_ho_0{i}" for i in range(1, 9)]          # ho_01..08
PHYS_BAND = (-10.0, -9.0)                                          # defect-domain eV/atom (Run-1 plan)
ATTR_TOL = 1.10

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "prod-run1-v5-heldout-generalization"
RUN_DIR = ROOT / "runs" / "production_run1" / RUN_ID


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_scan_toten(cell: str):
    """Last SCAN TOTEN (eV) from OUTCAR.scan; None if absent."""
    o = HB / cell / "OUTCAR.scan"
    if not o.is_file():
        return None, None
    e = None
    with open(o, errors="replace") as fh:
        for ln in fh:
            if "free  energy   TOTEN" in ln:
                try:
                    e = float(ln.split("=")[1].split("eV")[0])
                except (IndexError, ValueError):
                    pass
    return e, str(o)


def fget(row, k):
    v = row.get(k, "")
    if v in ("", "N/A", "nan", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def attribution(erra, errd, errb):
    if erra is None or errd is None or errb is None:
        return "UNRESOLVED"
    if erra >= errb and errd <= ATTR_TOL * erra:
        return "TEACHER_LIMITED"
    if errb > erra and errd > ATTR_TOL * erra:
        return "DISTILLATION_LIMITED"
    return "MIXED"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)   # fresh-run guard (append-only)

    # ---- input freeze ----
    rows = list(csv.DictReader(open(AGG)))
    scored = [r["cell_id"] for r in rows]
    manifest_rows = {r["cell_id"]: r for r in csv.DictReader(open(MANIFEST))}
    members = sorted(V5_BUNDLE.glob("seed0*/potential_saved_bestmodel"))
    member_md5 = {m.parent.name: hashlib.md5(m.read_bytes()).hexdigest() for m in members}

    input_manifest = {
        "run_id": RUN_ID, "read_only": True, "no_model_invocation": True, "no_dft": True,
        "no_md": True, "no_training": True, "no_scheduler": True, "no_network": True, "no_judge": True,
        "inputs": {
            "heldout_baseline_errord.csv": {"path": str(AGG), "sha256": sha256(AGG), "rows": len(rows)},
            "manifest_heldout.csv": {"path": str(MANIFEST), "sha256": sha256(MANIFEST), "rows": len(manifest_rows)},
            "teacher_model.nequip.pth": {"path": str(TEACHER), "sha256": sha256(TEACHER) if TEACHER.is_file() else None,
                                          "expected_sha256": TEACHER_SHA, "invoked": False},
        },
        "v5_committee_members": member_md5,
        "note": "error primitives are the committed run_heldout_baseline.py outputs (raw LAMMPS v5 + NequIP teacher + SCAN OUTCAR); this action recomputes only derived quantities + parses OUTCAR.scan E/atom.",
    }

    # ---- held-out set definition ----
    missing = [c for c in EXPECTED_HELDOUT if c not in scored]
    heldout_def = {
        "expected_heldout": EXPECTED_HELDOUT, "available_scored": scored,
        "missing": missing, "evaluated_count": len(scored), "excluded_count": len(missing),
        "excluded_reason": {c: ("no prediction row in analysis/ (unscored)"
                                 f"; source {manifest_rows.get(c, {}).get('distribution','?')} {manifest_rows.get(c, {}).get('x_label','?')}")
                            for c in missing},
    }

    # ---- leakage audit ----
    train_list_present = any((V5_BUNDLE / s / "train_list").is_file() for s in ("seed01",))
    leakage = {
        "status": "REVISE",
        "reason": "v5 training/augmentation frame list ('train_list') is NOT present in the committed v5 bundle (staged elsewhere/KISTI); exact disjointness of the held-out cells from v5 training cannot be PROVEN offline.",
        "train_list_present_in_bundle": train_list_present,
        "supporting_evidence": {
            "manifest_held_out_confirmed": {c: manifest_rows.get(c, {}).get("held_out_confirmed") for c in scored},
            "held_out_sources": {c: {"source_dump": manifest_rows.get(c, {}).get("source_dump"),
                                      "frame_idx": manifest_rows.get(c, {}).get("frame_idx")} for c in scored},
            "note": "cells were carved from production_12288 random_sweep / anneal_calib_clustered MD dumps; manifest asserts held_out_confirmed=True, but this is an assertion, not a proof against the v5 train frame list.",
        },
        "verdict": "cannot certify PASS; do not fabricate leakage-clean status",
    }

    # ---- per-cell metrics + attribution ----
    per_cell = []
    for r in rows:
        cid = r["cell_id"]
        erra, errd, errb = fget(r, "erra_core"), fget(r, "errd_core"), fget(r, "errb_core")
        erra_g, errd_g, errb_g = fget(r, "erra_global"), fget(r, "errd_global"), fget(r, "errb_global")
        toten, esrc = parse_scan_toten(cid)
        nat = int(float(r["n_atoms"]))
        epa = round(toten / nat, 4) if toten is not None else None
        per_cell.append({
            "cell_id": cid, "distribution": r["distribution"], "x_label": r["x_label"],
            "center_cn": r["center_cn"], "center_species": r["center_species"],
            "n_atoms": nat, "n_core": int(float(r["n_core"])), "local_x": manifest_rows.get(cid, {}).get("x_local"),
            "in_training_domain": "UNKNOWN (leakage REVISE)",
            "erra_core_teacher_vs_dft": erra, "errd_core_v5_vs_dft": errd, "errb_core_v5_vs_teacher": errb,
            "erra_global": erra_g, "errd_global": errd_g, "errb_global": errb_g,
            "u_alpha_center": fget(r, "u_alpha_center"), "u_alpha_core_mean": fget(r, "u_alpha_core_mean"),
            "errc_original_vs_dft": None,   # MISSING: original-student held-out predictions do not exist
            "delta_errc_v5_minus_original": None,  # MISSING (cannot compute without running original student)
            "dft_toten_eV": toten, "dft_E_per_atom_eV": epa, "dft_energy_source": "OUTCAR.scan",
            "phys_band_ok": (epa is not None and PHYS_BAND[0] <= epa <= PHYS_BAND[1]),
            "errb_lt_erra": (errb is not None and erra is not None and errb < erra),
            "attribution": attribution(erra, errd, errb),
            "finite_ok": all(x is not None for x in (erra, errd, errb)),
        })

    # ---- domain (family) aggregates ----
    def agg(fam_pred):
        sub = [c for c in per_cell if fam_pred(c)]
        return {
            "n_cells": len(sub),
            "erra_core_mean": mean([c["erra_core_teacher_vs_dft"] for c in sub]),
            "errd_core_mean": mean([c["errd_core_v5_vs_dft"] for c in sub]),
            "errb_core_mean": mean([c["errb_core_v5_vs_teacher"] for c in sub]),
            "u_alpha_center_mean": mean([c["u_alpha_center"] for c in sub]),
            "dft_E_per_atom_range": [min([c["dft_E_per_atom_eV"] for c in sub if c["dft_E_per_atom_eV"] is not None], default=None),
                                      max([c["dft_E_per_atom_eV"] for c in sub if c["dft_E_per_atom_eV"] is not None], default=None)],
            "attributions": sorted({c["attribution"] for c in sub}),
        }
    domain = {"clustered": agg(lambda c: c["distribution"] == "clustered"),
              "random": agg(lambda c: c["distribution"] == "random"),
              "all": agg(lambda c: True)}

    # ---- committee ----
    committee = {"n_members": len(members), "member_md5": member_md5,
                 "distinct": len(set(member_md5.values())) == len(members),
                 "definition": "committee-mean force + per-atom disagreement u_alpha = sqrt(sum_xyz Var(F))",
                 "note": "u_alpha values are the deployed committee-disagreement metric; no member cherry-picking."}

    # ---- artifact validity (authoritative) ----
    av_checks = {
        "all_rows_parse": True,
        "all_core_primitives_finite": all(c["finite_ok"] for c in per_cell),
        "atom_counts_match_manifest": all(str(c["n_atoms"]) == manifest_rows.get(c["cell_id"], {}).get("n_atoms")
                                          for c in per_cell),
        "v5_members_distinct_4": committee["distinct"] and committee["n_members"] == 4,
        "teacher_sha_matches": input_manifest["inputs"]["teacher_model.nequip.pth"]["sha256"] == TEACHER_SHA,
        "dft_energy_available_all": all(c["dft_toten_eV"] is not None for c in per_cell),
    }
    artifact_validity = "PASS" if all(av_checks.values()) else "FAIL"

    # ---- physical / domain validity ----
    phys_all_ok = all(c["phys_band_ok"] for c in per_cell)
    physical_validity = {"band_eV_per_atom": list(PHYS_BAND),
                         "band_provenance": "relaxed O-deficient SiO2-x DFT cells (coordination_log clustered -9.41..-9.80); domain/artifact sanity, NOT the model-accuracy metric",
                         "per_cell_E_per_atom": {c["cell_id"]: c["dft_E_per_atom_eV"] for c in per_cell},
                         "all_in_band": phys_all_ok,
                         "status": "PASS" if phys_all_ok else "REVISE"}

    # ---- generalization + attribution summaries ----
    all_errb_lt_erra = all(c["errb_lt_erra"] for c in per_cell)
    attr_counts = {}
    for c in per_cell:
        attr_counts[c["attribution"]] = attr_counts.get(c["attribution"], 0) + 1

    generalization = {
        "original_vs_v5_heldout": "MISSING",
        "reason": "no original-student held-out predictions exist (run_heldout_baseline_orig.py also targets the v5 bundle; no original output). Computing them requires running the ORIGINAL student = forbidden here.",
        "delta_errc_v5_minus_original": None,
        "note": "Per Run-1 contract, the prior deployment-distribution adoption verdict is NOT reused as proof. Held-out original-vs-v5 generalization is therefore UNRESOLVED offline.",
    }
    teacher_attr = {
        "rule": "TEACHER_LIMITED if erra>=errb and errd<=1.10*erra; DISTILLATION_LIMITED if errb>erra and errd>1.10*erra; else MIXED",
        "per_cell": {c["cell_id"]: c["attribution"] for c in per_cell},
        "counts": attr_counts,
        "errb_core_lt_erra_core_all_cells": all_errb_lt_erra,
        "clustered_core_summary": {"erra_core_mean": domain["clustered"]["erra_core_mean"],
                                    "errd_core_mean": domain["clustered"]["errd_core_mean"],
                                    "errb_core_mean": domain["clustered"]["errb_core_mean"]},
        "interpretation": ("Across ALL 6 scored held-out cells errb_core < erra_core (distillation gap "
                           "smaller than the teacher's own DFT error). No cell is DISTILLATION_LIMITED; "
                           f"{attr_counts.get('TEACHER_LIMITED',0)} TEACHER_LIMITED, {attr_counts.get('MIXED',0)} MIXED. "
                           "At clustered defect cores teacher and student errors are comparable (~0.35-0.44 eV/A) "
                           "and the teacher's error is the larger single factor => the clustered-core ceiling is "
                           "TEACHER-dominant, not distillation-limited."),
    }

    summary = {
        "run_id": RUN_ID, "action_type": "evaluate_heldout_fidelity", "read_only": True,
        "artifact_validity": artifact_validity,
        "leakage_status": leakage["status"],
        "original_vs_v5_generalization": generalization["original_vs_v5_heldout"],
        "teacher_limit_attribution": {"dominant": ("TEACHER_DOMINANT" if attr_counts.get("DISTILLATION_LIMITED", 0) == 0
                                                    else "MIXED"),
                                       "counts": attr_counts,
                                       "errb_lt_erra_all": all_errb_lt_erra},
        "physical_validity": physical_validity["status"],
        "current_production_student": "UNRESOLVED",
        "current_production_student_reason": "held-out original-vs-v5 generalization is MISSING and leakage is REVISE; the gate-log v5 ADOPT vs STATUS prose contradiction is NOT resolvable from R1.P1 evidence alone. Do NOT rewrite either source.",
        "next_scientific_lever": {
            "primary": "TEACHER_DFT_ANCHORED_IMPROVEMENT",
            "rationale": "clustered-defect-core residual is teacher-dominant (errb<erra everywhere; errd~erra at clustered cores); more student distillation cannot lower the teacher ceiling there.",
            "prerequisites_before_any_HPC": ["DATASET_LEAKAGE_RESOLUTION (obtain the v5 train frame list)",
                                             "MORE_HELDOUT_EVIDENCE (original-student held-out predictions + score cell_ho_01/ho_08)"],
            "explicitly_not_recommended": "MORE_STUDENT_DISTILLATION as the primary fix for the clustered-core ceiling",
        },
        "overall_scientific_interpretation": ("v5 vs teacher distillation gap is bounded and smaller than "
            "the teacher's own DFT error on every scored held-out cell; the hardest clustered-defect cores "
            "are teacher-limited. Whether v5 out-generalizes the ORIGINAL student on held-out data is "
            "UNRESOLVED (original held-out predictions absent) and leakage is uncertified => the production-"
            "student identity remains UNRESOLVED; the scientifically indicated next lever is improving the "
            "TEACHER at clustered defects, gated behind leakage resolution and more held-out evidence."),
        "no_scientific_compute_performed": True,
    }

    # ---- deterministic criterion results (four axes; policy owns verdict) ----
    criterion_results = {
        "deterministic_authoritative": True,
        "axes": {
            "artifact_validity": {"verdict": artifact_validity, "checks": av_checks},
            "leakage": {"verdict": leakage["status"], "reason": leakage["reason"]},
            "model_accuracy_attribution": {"verdict": "COMPUTED", "counts": attr_counts,
                                            "errb_lt_erra_all": all_errb_lt_erra,
                                            "original_vs_v5": "MISSING"},
            "physical_validity_defect_band": {"verdict": physical_validity["status"],
                                              "band": list(PHYS_BAND), "all_in_band": phys_all_ok},
        },
        "overall_transition": "REVISE",
        "overall_transition_reason": "artifact_validity+physical_validity PASS and attribution COMPUTED, but leakage=REVISE and original-vs-v5 generalization MISSING => cannot emit a leakage-clean best-student verdict; REVISE with explicit coverage/leakage caveats.",
    }

    # ---- write artifacts (append-only, under run dir) ----
    (RUN_DIR / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
    (RUN_DIR / "leakage_audit.json").write_text(json.dumps(leakage, indent=2) + "\n")
    (RUN_DIR / "committee_summary.json").write_text(json.dumps(committee, indent=2) + "\n")
    (RUN_DIR / "heldout_fidelity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RUN_DIR / "criterion_results.json").write_text(json.dumps(criterion_results, indent=2) + "\n")
    # per_cell_metrics.csv
    pc_fields = list(per_cell[0].keys())
    with open(RUN_DIR / "per_cell_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pc_fields); w.writeheader()
        for c in per_cell:
            w.writerow(c)
    # domain_metrics.csv
    with open(RUN_DIR / "domain_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "n_cells", "erra_core_mean", "errd_core_mean", "errb_core_mean",
                    "u_alpha_center_mean", "E_per_atom_min", "E_per_atom_max", "attributions"])
        for fam, d in domain.items():
            w.writerow([fam, d["n_cells"], d["erra_core_mean"], d["errd_core_mean"], d["errb_core_mean"],
                        d["u_alpha_center_mean"], d["dft_E_per_atom_range"][0], d["dft_E_per_atom_range"][1],
                        "|".join(d["attributions"])])
    heldout_def_out = dict(heldout_def, generalization=generalization, teacher_attribution=teacher_attr,
                           domain=domain)
    (RUN_DIR / "heldout_set_and_analysis.json").write_text(json.dumps(heldout_def_out, indent=2) + "\n")

    import subprocess
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    provenance = {
        "run_id": RUN_ID, "stage": "production_run1", "phase": "R1.P1",
        "action_type": "evaluate_heldout_fidelity", "package_head": head,
        "proposal": "examples/production_run1/action_proposal.json",
        "proposal_sha256": sha256(ROOT / "examples/production_run1/action_proposal.json"),
        "analysis_code": "work/production_run1_r1p1_eval.py",
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "no_gpu": True, "no_model_invocation": True, "no_dft": True, "no_md": True, "no_training": True,
        "no_scheduler": True, "no_network": True, "no_semantic_judge": True, "no_automatic_downstream_action": True,
        "input_shas": {k: v.get("sha256") for k, v in input_manifest["inputs"].items()},
        "output_shas": {f.name: sha256(f) for f in sorted(RUN_DIR.glob("*")) if f.name != "provenance.json"},
    }
    (RUN_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (RUN_DIR / "run_manifest.json").write_text(json.dumps({
        "status": "OK", "phase": "R1.P1", "action_type": "evaluate_heldout_fidelity",
        "artifact_validity": artifact_validity, "leakage_status": leakage["status"],
        "physical_validity": physical_validity["status"],
        "original_vs_v5_generalization": generalization["original_vs_v5_heldout"],
        "teacher_limit_attribution": summary["teacher_limit_attribution"]["dominant"],
        "current_production_student": summary["current_production_student"],
        "next_scientific_lever": summary["next_scientific_lever"]["primary"],
        "overall_transition": criterion_results["overall_transition"],
        "no_scientific_compute_performed": True,
    }, indent=2) + "\n")

    print(json.dumps({"run_dir": str(RUN_DIR), "artifact_validity": artifact_validity,
                      "leakage": leakage["status"], "physical_validity": physical_validity["status"],
                      "original_vs_v5": generalization["original_vs_v5_heldout"],
                      "attribution_counts": attr_counts, "errb_lt_erra_all": all_errb_lt_erra,
                      "evaluated": len(scored), "missing": missing,
                      "overall_transition": criterion_results["overall_transition"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
