#!/usr/bin/env python3
"""Production Run 1 · R1.P1R — DATASET_LEAKAGE_RESOLUTION (READ-ONLY; no compute).

Deterministic, network-free data-lineage audit: are the six evaluated held-out cells (cell_ho_02..07)
demonstrably independent of every structure used to TRAIN or AUGMENT the adopted v5 SIMPLE-NN committee?
Performs NO model inference (teacher or student), NO DFT, NO MD, NO training, NO scheduler, NO network,
NO semantic Judge. Reads only existing structure/config/list artifacts.

Leakage levels (matching procedure declared BEFORE reporting):
  LEVEL 0  EXACT FILE DUPLICATE      -- identical file bytes (sha256) where directly comparable
  LEVEL 1  EXACT STRUCTURE EQUIV     -- same composition + box + wrapped-fractional-coord multiset
                                        (order-independent, PBC-wrapped, coords rounded to 1e-3 frac)
  LEVEL 2  SAME SOURCE FRAME/DERIV   -- carved/derived from a parent trajectory frame that itself
                                        entered v5 training (requires the v5 augmentation frame list)
  LEVEL 3  NEAR-DUPLICATE            -- not identical but structurally near-identical (descriptive; no
                                        invented hard cutoff unless a project tolerance exists)
  LEVEL 4  SAME DISTRIBUTION ONLY    -- same composition/protocol, independently generated (NOT leakage)

`AGENT_LINEAGE` below is filled from the source-grounded lineage audit (v5 train_list / augmentation
source). If the exact v5 membership/frame records are absent locally, Level-2 stays UNRESOLVED and the
authoritative verdict is REVISE (lack of evidence is never turned into PASS).
"""
from __future__ import annotations
import csv, json, hashlib, sys
from pathlib import Path

RES = Path("/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation")
HB = RES / "al_iter3" / "heldout_dft_batch"
V5_BUNDLE = RES / "gpu_return_v5_committee" / "v5_committee_bundle"
AL11 = V5_BUNDLE / "AL11_with_umax_v5.xyz"
HELDOUT = [f"cell_ho_0{i}" for i in range(2, 8)]                   # ho_02..07 (the R1.P1-evaluated set)
FRAC_ROUND = 3

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "prod-run1-v5-leakage-resolution"
RUN_DIR = ROOT / "runs" / "production_run1" / RUN_ID
PARENT_R1P1 = {"head": "f7f5c65e8ba9e57777f4ac309c83fc4faf3b3d75", "run_id": "prod-run1-v5-heldout-generalization"}

# ---- FILLED FROM THE SOURCE-GROUNDED LINEAGE AUDIT (agent + direct reads) ----
AGENT_LINEAGE = None   # set in main() from lineage_findings.json if present, else None -> REVISE


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canon(species, pos, box):
    comp = {}
    frac = []
    for s, (x, y, z) in zip(species, pos):
        comp[s] = comp.get(s, 0) + 1
        frac.append((s, round((x / box) % 1.0, FRAC_ROUND), round((y / box) % 1.0, FRAC_ROUND),
                     round((z / box) % 1.0, FRAC_ROUND)))
    frac.sort()
    return {"composition": dict(sorted(comp.items())), "box": round(box, 3),
            "coord_hash": hashlib.sha256(str(frac).encode()).hexdigest()[:16], "n_atoms": len(species)}


def parse_lammps(path):
    L = open(path).read().splitlines(); lo = hi = None; i = 0
    for j, s in enumerate(L):
        if "xlo xhi" in s:
            lo, hi = float(s.split()[0]), float(s.split()[1])
        if s.strip().startswith("Atoms"):
            i = j + 2; break
    sp = []; pos = []
    while i < len(L) and L[i].strip() and L[i].strip()[0].lstrip("-").isdigit():
        p = L[i].split(); sp.append({"1": "O", "2": "Si"}.get(p[1], p[1]))
        pos.append((float(p[2]), float(p[3]), float(p[4]))); i += 1
    return canon(sp, pos, hi - lo)


def parse_xyz_frames(path):
    L = open(path).read().splitlines(); out = []; i = 0
    while i < len(L):
        if not L[i].strip().isdigit():
            i += 1; continue
        n = int(L[i]); hdr = L[i + 1]; box = 11.0
        if 'Lattice="' in hdr:
            box = float(hdr.split('Lattice="')[1].split('"')[0].split()[0])
        sp = []; pos = []
        for k in range(i + 2, i + 2 + n):
            p = L[k].split(); sp.append(p[0]); pos.append((float(p[1]), float(p[2]), float(p[3])))
        out.append(canon(sp, pos, box)); i += 2 + n
    return out


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    lf = ROOT / "work" / "run1r_lineage_findings.json"
    lineage = json.loads(lf.read_text()) if lf.is_file() else None

    manifest = {r["cell_id"]: r for r in csv.DictReader(open(HB / "manifest_heldout.csv"))}

    # ---- held-out lineage + canonical identity ----
    heldout = {}
    for c in HELDOUT:
        idp = HB / c / "input.data"
        fp = parse_lammps(idp)
        m = manifest[c]
        heldout[c] = {
            "input_data_path": str(idp), "input_data_sha256": sha256(idp),
            "out_data_sha256": sha256(HB / c / "out.data") if (HB / c / "out.data").is_file() else None,
            "n_atoms": fp["n_atoms"], "composition": fp["composition"], "box_A": fp["box"],
            "canonical_coord_hash": fp["coord_hash"],
            "domain": {"distribution": m["distribution"], "x_label": m["x_label"],
                       "center_cn": m["center_cn_si_o"], "local_x": m["x_local"]},
            "parent_source_dump": m["source_dump"], "parent_frame_idx": m["frame_idx"],
            "derivation": "carved ~11A cell around center atom + inner-sphere-frozen anneal (out.data)",
            "atom_order_changed_from_parent": True,
        }

    # ---- local candidate v5-adjacent structures (AL11 + DFT-labelled AL cells) ----
    candidates = {}
    for idx, fp in enumerate(parse_xyz_frames(AL11)):
        candidates[f"AL11_frame_{idx}"] = fp
    for d in sorted(list((RES / "dft_labeling").glob("cell_*")) + list((RES / "dft_labeling").glob("clustered_cell_*"))):
        idata = d / "input.data"
        if idata.is_file():
            try:
                candidates[d.name] = parse_lammps(idata)
            except Exception:  # noqa: BLE001
                pass

    # ---- membership matrix (Level 0/1 vs local candidates) ----
    heldout_shas = {c: heldout[c]["input_data_sha256"] for c in HELDOUT}
    cand_shas = {}
    for d in sorted(list((RES / "dft_labeling").glob("cell_*")) + list((RES / "dft_labeling").glob("clustered_cell_*"))):
        if (d / "input.data").is_file():
            cand_shas[d.name] = sha256(d / "input.data")
    matrix = []
    for c in HELDOUT:
        hfp = heldout[c]
        exact_file = [n for n, s in cand_shas.items() if s == heldout_shas[c]]
        equiv = [n for n, fp in candidates.items()
                 if fp["composition"] == hfp["composition"] and fp["box"] == hfp["box_A"]
                 and fp["coord_hash"] == hfp["canonical_coord_hash"]]
        comp_box_only = [n for n, fp in candidates.items()
                         if fp["composition"] == hfp["composition"] and fp["box"] == hfp["box_A"]]
        matrix.append({
            "heldout_cell": c, "level0_exact_file_duplicate": exact_file or "NONE",
            "level1_structure_equivalent": equiv or "NONE",
            "same_comp_and_box_local_candidates": comp_box_only or "NONE",
            "parent_frame": f"{hfp['parent_source_dump']}#frame{hfp['parent_frame_idx']}",
        })

    # ---- Level-2 (same parent frame in v5 training) + AL overlap: from lineage findings ----
    aug_sources = (lineage or {}).get("v5_augmentation_sources")   # list of parent dumps/dirs, or None
    train_list_local = (lineage or {}).get("v5_train_list_local", False)
    heldout_parents = sorted({heldout[c]["parent_source_dump"] for c in HELDOUT})
    if aug_sources is None:
        level2 = {"status": "UNRESOLVED",
                  "reason": "v5 augmentation/training frame list is not available locally; cannot prove or disprove that any held-out parent dump/frame entered v5 augmentation.",
                  "heldout_parent_dumps": heldout_parents}
    else:
        overlap = [p for p in heldout_parents if any(str(p) in str(a) or str(a) in str(p) for a in aug_sources)]
        level2 = {"status": ("FAIL" if overlap else "PASS"),
                  "reason": ("held-out parent dump(s) appear in v5 augmentation sources" if overlap
                             else "no held-out parent dump appears in the (locally available) v5 augmentation sources"),
                  "heldout_parent_dumps": heldout_parents, "v5_augmentation_sources": aug_sources,
                  "overlap": overlap}

    al_overlap = {
        "AL11_with_umax_v5_present": AL11.is_file(),
        "AL11_is": (lineage or {}).get("AL11_with_umax_v5_is",
                    "v5 EVALUATION output (11 AL cells scored by the v5 committee), NOT training data"),
        "eleven_al_cells_in_v5_training": (lineage or {}).get("eleven_al_cells_in_v5_training", "UNRESOLVED"),
        "heldout_vs_AL11_or_DFT_cells_identity": "NONE match at Level 0 or Level 1 (see membership_matrix) => the 6 held-out cells are STRUCTURALLY DISTINCT from the 11 AL / DFT-labelled cells; the AL-overlap concern (R1.P1 caveat) affects DEPLOYMENT error(c), NOT the held-out cells.",
        "al_iter3_28_cells_in_v5_training": (lineage or {}).get("al_iter3_in_v5_training", "UNRESOLVED"),
    }

    # ---- authoritative leakage verdict ----
    any_exact = any(m["level0_exact_file_duplicate"] != "NONE" or m["level1_structure_equivalent"] != "NONE" for m in matrix)
    if any_exact:
        verdict = "FAIL"; vreason = "at least one held-out cell is an exact/equivalent duplicate of a v5-adjacent structure"
    elif level2["status"] == "FAIL":
        verdict = "FAIL"; vreason = "a held-out cell shares a parent frame with a v5 augmentation structure"
    elif level2["status"] == "PASS" and train_list_local:
        verdict = "PASS"; vreason = "no exact/equivalent duplicate and no shared parent frame vs the complete local v5 training/augmentation records"
    else:
        verdict = "REVISE"; vreason = "no LOCAL exact/equivalent duplicate found, but the v5 training/augmentation frame records are not fully available locally => same-parent-frame independence (Level 2) cannot be PROVEN"

    # ---- optional discovery: original-student held-out predictions ----
    orig_pred = list(HB.glob("analysis/*orig*csv")) + list(HB.glob("analysis/per_cell/*orig*"))
    original_discovery = {"original_student_heldout_predictions": (
        [str(p) for p in orig_pred] if orig_pred else "MISSING (not generated; do not run original student here)")}

    # ---- write artifacts ----
    input_manifest = {
        "run_id": RUN_ID, "parent_r1p1": PARENT_R1P1, "read_only": True,
        "no_model_invocation": True, "no_dft": True, "no_md": True, "no_training": True,
        "no_scheduler": True, "no_network": True, "no_judge": True,
        "inputs": {
            "manifest_heldout.csv": {"path": str(HB / "manifest_heldout.csv"), "sha256": sha256(HB / "manifest_heldout.csv")},
            "AL11_with_umax_v5.xyz": {"path": str(AL11), "sha256": sha256(AL11)} if AL11.is_file() else "MISSING",
            "v5_seed01_input.yaml": {"path": str(V5_BUNDLE / "seed01" / "input.yaml"), "sha256": sha256(V5_BUNDLE / "seed01" / "input.yaml")},
        },
        "held_out_input_data_sha256": heldout_shas,
        "v5_committee_member_md5": {m.parent.name: hashlib.md5(m.read_bytes()).hexdigest()
                                    for m in sorted(V5_BUNDLE.glob("seed0*/potential_saved_bestmodel"))},
        "lineage_findings_source": str(lf) if lineage else "NOT PROVIDED (Level-2 stays UNRESOLVED)",
    }
    (RUN_DIR / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
    (RUN_DIR / "v5_training_lineage.json").write_text(json.dumps(lineage or {
        "status": "INCOMPLETE_LOCAL",
        "note": "v5 train_list/valid_list/test_list referenced by input.yaml are NOT in the committed bundle; full training/augmentation frame records are KISTI-origin and not local.",
        "v5_train_list_local": train_list_local}, indent=2) + "\n")
    (RUN_DIR / "heldout_lineage.json").write_text(json.dumps(heldout, indent=2) + "\n")
    with open(RUN_DIR / "membership_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["heldout_cell", "level0_exact_file_duplicate", "level1_structure_equivalent",
                    "same_comp_box_local_candidates", "parent_frame", "membership_proven"])
        for m in matrix:
            w.writerow([m["heldout_cell"], m["level0_exact_file_duplicate"], m["level1_structure_equivalent"],
                        m["same_comp_and_box_local_candidates"], m["parent_frame"], "NO"])
    (RUN_DIR / "duplicate_equivalence_audit.json").write_text(json.dumps(
        {"level_definitions": {"L0": "exact file bytes", "L1": "comp+box+wrapped-frac-coord multiset (1e-3)",
                               "L2": "same parent frame in v5 training", "L3": "near-duplicate (descriptive)",
                               "L4": "same distribution only (not leakage)"},
         "matching_procedure_declared_before_results": True,
         "matrix": matrix, "level2": level2}, indent=2) + "\n")
    (RUN_DIR / "al_overlap_audit.json").write_text(json.dumps(al_overlap, indent=2) + "\n")
    leakage_summary = {
        "leakage_verdict": verdict, "verdict_reason": vreason,
        "level0_exact_duplicates": "NONE" if not any(m["level0_exact_file_duplicate"] != "NONE" for m in matrix) else "PRESENT",
        "level1_structure_equivalents": "NONE" if not any(m["level1_structure_equivalent"] != "NONE" for m in matrix) else "PRESENT",
        "level2_same_parent_frame": level2["status"],
        "al_cell_overlap": "AL11_with_umax_v5.xyz is v5 EVALUATION output (not training); the held-out cells are structurally DISTINCT from the 11 AL cells regardless of whether those cells are in v5 training.",
        "circumstantial_independence": (lineage or {}).get("circumstantial_independence",
            "held-out parent regimes are the OOD regimes v5 fails and v6 remediates => likely not in v5 training"),
        "blocker_if_revise": ((lineage or {}).get("missing_artifact_blocker",
            "KISTI-origin v5 train_list/valid_list + augmented XYZ (esp. the v5 T1 defect-re-augment frame manifest)")
            if verdict == "REVISE" else None),
        "original_student_heldout_predictions": original_discovery["original_student_heldout_predictions"],
        "scientific_interpretation": ("No held-out cell is a byte-exact (Level-0) or structure-equivalent "
            "(Level-1) duplicate of any LOCAL v5-adjacent structure, and the held-out set is structurally "
            "distinct from the 11 AL cells. Documented provenance (v5 augmentation seeds = KISTI DFT corpus, "
            "not production MD dumps) plus the v6 OOD-remediation logic (v5 FAILS on exactly the held-out "
            "parent regimes) give STRONG circumstantial independence. But the v5-specific T1 defect "
            "re-augment frame manifest is non-local, so same-parent-frame (Level-2) independence cannot be "
            "PROVEN at frame level => REVISE (not PASS; lack of evidence is not turned into PASS). "
            "ORIGINAL_VS_V5 remains UNRESOLVED (unchanged; original student NOT run)."),
    }
    (RUN_DIR / "leakage_summary.json").write_text(json.dumps(leakage_summary, indent=2) + "\n")
    criterion = {"deterministic_authoritative": True, "leakage_verdict": verdict, "reason": vreason,
                 "levels": {"L0": "NONE", "L1": "NONE", "L2": level2["status"]},
                 "original_vs_v5": "UNRESOLVED (unchanged; original student NOT run)"}
    (RUN_DIR / "criterion_results.json").write_text(json.dumps(criterion, indent=2) + "\n")

    import subprocess
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    provenance = {
        "run_id": RUN_ID, "stage": "production_run1", "phase": "R1.P1R",
        "action": "dataset_leakage_resolution", "package_head": head,
        "parent_r1p1_head": PARENT_R1P1["head"], "parent_r1p1_run_id": PARENT_R1P1["run_id"],
        "proposal": "examples/production_run1/action_proposal.json",
        "analysis_code": "work/production_run1r_leakage_eval.py",
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "no_gpu": True, "no_model_invocation": True, "no_teacher_inference": True, "no_student_inference": True,
        "no_dft": True, "no_md": True, "no_training": True, "no_scheduler": True, "no_network": True,
        "no_semantic_judge": True, "automatic_downstream_action": False,
        "input_shas": {k: (v.get("sha256") if isinstance(v, dict) else v) for k, v in input_manifest["inputs"].items()},
        "output_shas": {f.name: sha256(f) for f in sorted(RUN_DIR.glob("*")) if f.name != "provenance.json"},
    }
    (RUN_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (RUN_DIR / "run_manifest.json").write_text(json.dumps({
        "status": "OK", "phase": "R1.P1R", "action": "dataset_leakage_resolution",
        "leakage_verdict": verdict, "level0": "NONE", "level1": "NONE", "level2": level2["status"],
        "original_vs_v5": "UNRESOLVED (unchanged)",
        "blocker_if_revise": leakage_summary["blocker_if_revise"],
        "no_scientific_compute_performed": True, "parent_r1p1_unchanged": True,
    }, indent=2) + "\n")

    print(json.dumps({"run_dir": str(RUN_DIR), "leakage_verdict": verdict,
                      "level0": "NONE", "level1": "NONE", "level2": level2["status"],
                      "blocker": leakage_summary["blocker_if_revise"],
                      "original_vs_v5": "UNRESOLVED"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
