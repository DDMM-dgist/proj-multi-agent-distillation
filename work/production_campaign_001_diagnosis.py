#!/usr/bin/env python3
"""Production Campaign 001 — independent, raw-data-first scientific diagnosis (READ-ONLY; no compute).

Recomputes teacher/student force+energy errors by domain family DIRECTLY from the raw error CSVs
(the numerical source of truth), independently of any historical decision. NO model/DFT/MD/training/
network/Judge. Writes a domain-comparison table + machine-readable summary under a fresh run dir.
"""
from __future__ import annotations
import csv, json, hashlib, statistics as st, sys
from pathlib import Path

RES = Path("/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation")
TD = RES / "teacher_diag"
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "production_campaign_001" / "pc001-independent-diagnosis"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def fam(ct):
    ct = ct or ""
    if ct.startswith("SiOx"): return "SiOx_defect"
    if ct.startswith("bulk_cryst"): return "bulk_crystal"
    if ct.startswith("bulk_amo") or ct.startswith("quench"): return "bulk_amorphous_quench"
    if ct.startswith("silicon"): return "elemental_Si"
    if ct.startswith("surfaces"): return "surfaces"
    if ct == "liquid": return "liquid"
    if ct == "cluster": return "cluster"
    if ct.startswith("vacancy"): return "vacancy_SiOx"
    if ct.startswith("highpressure"): return "high_pressure"
    return ct or "other"


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    a = list(csv.DictReader(open(TD / "error_a_allegro_vs_dft.csv")))
    b = list(csv.DictReader(open(TD / "error_b_clean_simplenn_vs_allegro.csv")))
    c = list(csv.DictReader(open(TD / "error_c_simplenn_vs_dft.csv")))

    def col(rows, key, elig=False):
        d = {}
        for r in rows:
            if elig and r.get("snn_eligible") != "True":
                continue
            v = num(r[key])
            if v is None:
                continue
            d.setdefault(fam(r["config_type"]), []).append(v)
        return d

    Fa = col(a, "Fmae_eV_A")
    Fc = col(c, "Fmae_snn_vs_dft_eV_A", True)
    Fb = col(b, "Fmae_snn_vs_alleg_eV_A", True)
    Ustd = col(b, "Fstd_committee_eV_A", True)

    def attribution(ma, mc, mb):
        if None in (ma, mc, mb):
            return "STUDENT_NOT_EVALUATED" if mc is None else "n/a"
        if mb > ma:
            return "DISTILLATION_LIMITED"
        if mc <= 1.10 * ma:
            return "TEACHER_LIMITED"
        return "MIXED"

    rows_out = []
    for f in sorted(set(list(Fa) + list(Fc))):
        va, vc, vb, vu = Fa.get(f, []), Fc.get(f, []), Fb.get(f, []), Ustd.get(f, [])
        ma = round(st.mean(va), 3) if va else None
        mc = round(st.mean(vc), 3) if vc else None
        mb = round(st.mean(vb), 3) if vb else None
        mu = round(st.mean(vu), 3) if vu else None
        rows_out.append({"domain_family": f, "n_frames": len(va),
                         "errA_teacher_vs_dft_Fmae": ma, "errC_student_vs_dft_Fmae": mc,
                         "errB_student_vs_teacher_Fmae": mb, "committee_Fstd": mu,
                         "attribution": attribution(ma, mc, mb)})

    allA = sorted(v for r in a if (v := num(r["Fmae_eV_A"])) is not None)
    allC = sorted(v for r in c if r.get("snn_eligible") == "True" and (v := num(r["Fmae_snn_vs_dft_eV_A"])) is not None)
    allB = [v for r in b if r.get("snn_eligible") == "True" and (v := num(r["Fmae_snn_vs_alleg_eV_A"])) is not None]
    eA = [abs(v) for r in a if (v := num(r["dE_per_atom_meV_shifted"])) is not None]
    eC = [abs(v) for r in c if r.get("snn_eligible") == "True" and (v := num(r["dE_snn_vs_dft_per_atom_meV_shifted"])) is not None]
    outliers = [{"config_type": r["config_type"], "Fmae": round(v, 2)}
                for r in a if (v := num(r["Fmae_eV_A"])) is not None and v > 5]

    def q(v, p): return round(v[min(len(v) - 1, int(p * len(v)))], 3)
    summary = {
        "common_test_set": {"error_a_rows": len(a), "error_bc_eligible": sum(1 for r in c if r.get("snn_eligible") == "True"),
                            "source": "teacher_diag/error_{a,b,c}.csv (shared idx+config_type over the 1155-frame test_set)"},
        "global_force_mae_eV_A": {"teacher_vs_dft_errA": round(st.mean(allA), 3),
                                   "student_vs_dft_errC": round(st.mean(allC), 3),
                                   "student_vs_teacher_errB": round(st.mean(allB), 3)},
        "global_force_tails": {"errA": {"p50": q(allA, .5), "p90": q(allA, .9), "p99": q(allA, .99), "max": round(allA[-1], 2)},
                               "errC": {"p50": q(allC, .5), "p90": q(allC, .9), "p99": q(allC, .99), "max": round(allC[-1], 2)}},
        "energy_mae_meV_atom_shift": {"teacher_vs_dft": round(st.mean(eA), 2), "student_vs_dft": round(st.mean(eC), 2)},
        "teacher_force_outliers_gt5": outliers,
        "per_domain": rows_out,
        "model_comparison_fairness": {
            "original_student_dft_eval": "error_c on the 1155-frame test_set (881 SiO2-eligible)",
            "v5_student_dft_eval": "R1 errd on 6 held-out cells ONLY",
            "common_original_vs_v5_dft_set": "NONE => original-vs-v5 ranking NOT determinable offline",
            "candidate_common_clean_set": "28 al_iter3 DFT cells postdate both students (v5 + original) => usable as a fair common set once both are evaluated on it",
        },
    }

    with open(RUN_DIR / "per_domain_model_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader()
        for r in rows_out: w.writerow(r)
    (RUN_DIR / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import subprocess
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    prov = {"run_id": "pc001-independent-diagnosis", "stage": "production_campaign_001", "phase": "diagnosis",
            "package_head": head, "analysis_code_sha256": sha(Path(__file__).resolve()),
            "inputs_sha256": {f: sha(TD / f) for f in ("error_a_allegro_vs_dft.csv",
                              "error_b_clean_simplenn_vs_allegro.csv", "error_c_simplenn_vs_dft.csv")},
            "no_model_invocation": True, "no_dft": True, "no_md": True, "no_training": True,
            "no_network": True, "no_semantic_judge": True, "historical_decisions_used": False}
    (RUN_DIR / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
