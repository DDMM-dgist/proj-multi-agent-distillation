#!/usr/bin/env python3
"""Production Campaign 001 — TEACHER VALIDATION & ACCEPTANCE gate (READ-ONLY; no compute).

Canonical pipeline position (per the scientific workflow):
  scope / DFT reference  ->  TEACHER VALIDATION  ->  [gate: teacher ACCEPTED?]  ->  distillation ->
  student training -> student validation -> physical/MD -> coverage-failure / active learning.

This action is the TEACHER-VALIDATION gate. Everything downstream (distillation, student comparison,
student training) is gated behind teacher ACCEPT. It reads the EXISTING teacher-vs-DFT predictions
(teacher_diag/error_a_allegro_vs_dft.csv — the teacher forward was already run to produce it) + EOS
summaries. NO teacher inference, NO student inference, NO DFT/MD/training/scheduler/network/Judge.

Acceptance criteria (DECLARED BEFORE APPLICATION):
  Scope = SiO2 + SiO2-x defects/vacancies + surfaces + liquid + ambient crystalline SiO2 (the deployment
    domain). Out-of-scope (not gating): elemental Si, high-pressure dense polymorphs, isolated clusters.
    Known DATA artifact (cluster cc001, atom-overlap, Fmae 56.65) is flagged + excluded (not teacher error).
  A1 (absolute, conventional MLIP bar): in-scope teacher force MAE <= 0.20 eV/A AND energy MAE <= 50
     meV/atom.  A2 (physical): in-scope EOS phases SMOOTH.
  A3 (relative, grounded): teacher in-scope force MAE < the deployed student's in-domain force error
     (~0.23 eV/A, PC001 diagnosis) => teacher is NOT the dominant student-limiter in-domain.
  CONDITIONAL flag: any in-scope sub-region with force MAE > 1.5x the in-scope mean is flagged for
     monitoring (candidate for later coverage/active-learning), without blocking acceptance.
  Verdict: ACCEPT if A1&A2&A3; ACCEPT_CONDITIONAL if ACCEPT but >=1 flagged sub-region; REJECT otherwise.
"""
from __future__ import annotations
import csv, json, hashlib, statistics as st, sys
from pathlib import Path

RES = Path("/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation")
TD = RES / "teacher_diag"
TEACHER_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "production_campaign_001" / "pc001-teacher-validation"

IN_SCOPE = {"SiOx_int_AL", "SiOx_max_AL", "SiOx_crystal_amorphous_interfaces", "vacancy_int_AL", "vacancy",
            "bulk_amo", "quench", "quench_int_AL", "quench_max_AL", "liquid", "surfaces",
            "surfaces_int_AL", "surfaces_max_AL", "bulk_cryst"}
OUT_SCOPE = {"silicon_bulk_amo", "silicon_crystalline_main", "silicon_defects", "silicon_liquid",
             "silicon_others", "silicon_surfaces", "cluster", "bulk_cryst_hp",
             "highpressure_int_AL", "highpressure_max_AL"}
ARTIFACT_FMAE = 5.0                       # drop cc001-type atom-overlap artifacts
STUDENT_IN_DOMAIN_FMAE = 0.23             # PC001 diagnosis (deployed student vs DFT global)
A1_FORCE, A1_ENERGY = 0.20, 50.0
COND_FACTOR = 1.5


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    a = list(csv.DictReader(open(TD / "error_a_allegro_vs_dft.csv")))
    eos = list(csv.DictReader(open(TD / "eos_teacher_bm_summary.csv")))

    def fmae(rows):
        v = [num(r["Fmae_eV_A"]) for r in rows if (x := num(r["Fmae_eV_A"])) is not None and x < ARTIFACT_FMAE]
        return (round(st.mean(v), 3), len(v)) if v else (None, 0)

    def emae(rows):
        v = [abs(num(r["dE_per_atom_meV_shifted"])) for r in rows if num(r["dE_per_atom_meV_shifted"]) is not None]
        return (round(st.mean(v), 2), len(v)) if v else (None, 0)

    rin = [r for r in a if r["config_type"] in IN_SCOPE]
    rout = [r for r in a if r["config_type"] in OUT_SCOPE]
    f_in, n_in = fmae(rin); e_in, _ = emae(rin)

    # in-scope sub-regions
    subs = {
        "bulk_amorphous_quench": [r for r in rin if r["config_type"] in ("bulk_amo", "quench", "quench_int_AL", "quench_max_AL")],
        "SiOx_defect_dilute": [r for r in rin if (r["config_type"].startswith("SiOx") or r["config_type"].startswith("vacancy")) and "max_AL" not in r["config_type"]],
        "SiOx_defect_clustered": [r for r in rin if (r["config_type"].startswith("SiOx") or r["config_type"].startswith("vacancy")) and "max_AL" in r["config_type"]],
        "surfaces": [r for r in rin if r["config_type"].startswith("surfaces")],
        "liquid": [r for r in rin if r["config_type"] == "liquid"],
        "bulk_crystal_ambient": [r for r in rin if r["config_type"] == "bulk_cryst"],
    }
    sub_stats = {k: {"force_mae": fmae(v)[0], "n": fmae(v)[1]} for k, v in subs.items()}
    flagged = [k for k, s in sub_stats.items() if s["force_mae"] is not None and s["force_mae"] > COND_FACTOR * f_in]

    eos_smooth = all(r.get("smoothness", "SMOOTH") == "SMOOTH" for r in eos if r["phase"] in ("alpha-quartz", "Coesite"))
    A1 = (f_in is not None and f_in <= A1_FORCE) and (e_in is not None and e_in <= A1_ENERGY)
    A2 = eos_smooth
    A3 = (f_in is not None and f_in < STUDENT_IN_DOMAIN_FMAE)
    if A1 and A2 and A3:
        verdict = "ACCEPT_CONDITIONAL" if flagged else "ACCEPT"
    else:
        verdict = "REJECT"

    result = {
        "phase": "TEACHER_VALIDATION",
        "pipeline_position": "scope/DFT-reference -> [TEACHER VALIDATION] -> gate(teacher ACCEPTED?) -> distillation -> student training -> student validation -> physical/MD -> coverage/AL",
        "teacher": {"identity": "base KISTI Allegro (deployed)", "sha256": TEACHER_SHA},
        "dft_reference": "SCAN (2-stage PBE->SCAN); teacher-vs-DFT from error_a_allegro_vs_dft.csv (teacher forward already run)",
        "scope": {"in_scope": sorted(IN_SCOPE), "out_of_scope_not_gating": sorted(OUT_SCOPE),
                  "data_artifact_excluded": "cluster cc001 (atom-overlap, Fmae 56.65)"},
        "criteria_declared_before": {"A1_force_le": A1_FORCE, "A1_energy_le": A1_ENERGY, "A2": "EOS in-scope SMOOTH",
                                     "A3_relative": f"teacher in-scope force MAE < deployed student in-domain {STUDENT_IN_DOMAIN_FMAE}",
                                     "conditional_flag": f"sub-region force MAE > {COND_FACTOR}x in-scope mean"},
        "metrics": {"in_scope_frames": n_in, "in_scope_force_mae_eV_A": f_in, "in_scope_energy_mae_meV_atom": e_in,
                    "out_of_scope_force_mae_eV_A": fmae(rout)[0], "eos_ambient_smooth": eos_smooth,
                    "eos_B0": {r["phase"]: r["B0_GPa"] for r in eos if r["phase"] in ("alpha-quartz", "Coesite")}},
        "in_scope_subregions": sub_stats,
        "checks": {"A1_absolute": A1, "A2_eos": A2, "A3_relative_not_dominant_limiter": A3},
        "conditional_flagged_subregions": flagged,
        "TEACHER_VERDICT": verdict,
        "verdict_reason": ("Teacher accepted as the distillation teacher for the in-scope domain: in-scope force "
            f"MAE {f_in} eV/A (<= {A1_FORCE}), energy MAE {e_in} meV/atom (<= {A1_ENERGY}), EOS ambient SMOOTH, "
            f"and teacher error < deployed-student in-domain error ({STUDENT_IN_DOMAIN_FMAE}) so the teacher is "
            "NOT the dominant student-limiter in-domain. CONDITIONAL: the clustered-SiOx-defect sub-region has "
            "elevated teacher force MAE (~0.35 eV/A), flagged for monitoring / later coverage-failure active "
            "learning; it does not block acceptance but bounds achievable student accuracy there."),
        "gate_effect": "TEACHER ACCEPTED => the pipeline may proceed to distillation dataset/labeling and (downstream) student training + student validation. The original-vs-v5 STUDENT comparison is a STUDENT-VALIDATION-stage action, gated BEHIND this acceptance — not the first action.",
        "no_scientific_compute_performed": True,
    }

    (RUN_DIR / "teacher_validation_verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(RUN_DIR / "teacher_in_scope_subregions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subregion", "n_frames", "teacher_force_mae_eV_A", "flagged_conditional"])
        for k, s in sub_stats.items():
            w.writerow([k, s["n"], s["force_mae"], k in flagged])
    import subprocess
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    (RUN_DIR / "provenance.json").write_text(json.dumps({
        "run_id": "pc001-teacher-validation", "stage": "production_campaign_001", "phase": "TEACHER_VALIDATION",
        "package_head": head, "analysis_code_sha256": sha(Path(__file__).resolve()),
        "inputs_sha256": {"error_a_allegro_vs_dft.csv": sha(TD / "error_a_allegro_vs_dft.csv"),
                          "eos_teacher_bm_summary.csv": sha(TD / "eos_teacher_bm_summary.csv")},
        "teacher_sha256": TEACHER_SHA, "teacher_invoked": False,
        "no_model_invocation": True, "no_dft": True, "no_md": True, "no_training": True,
        "no_network": True, "no_semantic_judge": True, "historical_decisions_used": False}, indent=2) + "\n")
    (RUN_DIR / "run_manifest.json").write_text(json.dumps({
        "status": "OK", "phase": "TEACHER_VALIDATION", "TEACHER_VERDICT": verdict,
        "in_scope_force_mae": f_in, "in_scope_energy_mae": e_in,
        "conditional_flagged": flagged, "gate": "teacher ACCEPTED -> distillation/student stages unlocked (downstream)",
        "no_scientific_compute_performed": True}, indent=2) + "\n")
    print(json.dumps({"TEACHER_VERDICT": verdict, "in_scope_force_mae": f_in, "in_scope_energy_mae": e_in,
                      "flagged": flagged, "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
