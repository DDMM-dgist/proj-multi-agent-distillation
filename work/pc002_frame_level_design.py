#!/usr/bin/env python3
"""PC002 — REAL frame-level distillation structural design (teacher-label-INDEPENDENT).

Enumerates frame-level candidate pools, maps domains explicitly, selects an evidence-based
frame-level Student ensemble with redundancy control + a source/trajectory-aware deterministic
train/val split, and freezes all manifests WITHOUT teacher labels. The 11 SCAN DFT cells are held
out (PC004). Elemental-Si (nO==0) frames are excluded as OOD for a SiOx student.

Candidate pools (LOCAL, frame-level):
  A) dataset.xyz  frames [0,11423]  = ORIGINAL teacher corpus structures (all domains, config_type)
     (frames [11424,13897] are POST-ORIGINAL appended data — NOT enumerated as corpus)
  B) sio2x_production/<mode>/traj.dump  = deployment defect MD (pristine/random/sphere/plane x006/x012)
"""
import sys, json, csv, hashlib
from pathlib import Path
import numpy as np
from ase.io import iread

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
DATASET = f"{RES}/03_allegro_train/dataset.xyz"
SCAN = f"{RES}/scan_labeled_structures/sio2x_AL_labels_11cells.xyz"
N_CORPUS = 11424
PROD_STRIDE = 10   # keep every 10th production MD frame (redundancy control)
CORPUS_FAMILY_CAP = 400   # per-config cap for broad replay redundancy control (central kept uncapped)
OUT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime/work")

PROD = {"pristine": ("STOICHIOMETRIC_AMORPHOUS", {1:8,2:14}),
        "random_x006": ("DILUTE_OXYGEN_DEFICIENT", {1:8,2:14}),
        "random_x012": ("DILUTE_OXYGEN_DEFICIENT", {1:8,2:14}),
        "sphere_x006": ("CLUSTERED_VOID", {1:8,2:14}),
        "sphere_x012": ("CLUSTERED_VOID", {1:8,2:14}),
        "plane_x006":  ("CLUSTERED_VOID", {1:8,2:14}),
        "plane_x012":  ("CLUSTERED_VOID", {1:8,2:14})}

# explicit config_family -> scientific_domain map (oxide frames only; nO==0 overrides to OOD)
FAMILY_DOMAIN = {
  "bulk_amo": "STOICHIOMETRIC_AMORPHOUS", "quench": "STOICHIOMETRIC_AMORPHOUS", "amorph": "STOICHIOMETRIC_AMORPHOUS",
  "vacancy": "DILUTE_OXYGEN_DEFICIENT", "vacancy_int_AL": "DILUTE_OXYGEN_DEFICIENT",
  "SiOx_int_AL": "DILUTE_OXYGEN_DEFICIENT", "quench_int_AL": "DILUTE_OXYGEN_DEFICIENT",
  "interstitial": "DILUTE_OXYGEN_DEFICIENT", "divacancy": "DILUTE_OXYGEN_DEFICIENT",
  "SiOx_max_AL": "CLUSTERED_VOID", "quench_max_AL": "CLUSTERED_VOID", "surfaces_max_AL": "CLUSTERED_VOID",
  "cluster": "CLUSTERED_VOID",
  "surfaces": "SURFACE", "surfaces_int_AL": "SURFACE",
  "liquid": "HIGH_T_LIQUID", "liq": "HIGH_T_LIQUID",
  "bulk_cryst": "AMBIENT_CRYSTAL", "bulk_cryst_hp": "AMBIENT_CRYSTAL",
  "highpressure_int_AL": "AMBIENT_CRYSTAL", "highpressure_max_AL": "AMBIENT_CRYSTAL",
  "SiOx_crystal_amorphous_interfaces": "OTHER_RELEVANT_SIO",
}
CENTRAL = {"STOICHIOMETRIC_AMORPHOUS", "DILUTE_OXYGEN_DEFICIENT", "CLUSTERED_VOID"}
BROAD   = {"SURFACE", "HIGH_T_LIQUID", "AMBIENT_CRYSTAL", "OTHER_RELEVANT_SIO"}

def h_struct(a):
    m = hashlib.sha256()
    m.update(np.round(a.get_positions(), 4).tobytes())
    m.update("".join(a.get_chemical_symbols()).encode())
    try: m.update(np.round(a.get_cell().array, 4).tobytes())
    except Exception: pass
    return m.hexdigest()[:16]

def comp(a):
    s = a.get_chemical_symbols(); nSi = s.count("Si"); nO = s.count("O")
    x = round(2 - nO/nSi, 4) if (nSi and nO) else None
    return nSi, nO, (round(nO/nSi, 4) if nSi else None), x

def domain_of(family, nO):
    if nO == 0: return "OOD_OR_EXCLUDED"           # elemental Si / non-oxide (composition override)
    d = FAMILY_DOMAIN.get(family)
    return d if d else "UNRESOLVED"

def main():
    rows = []
    # ---- pool A: original corpus frames [0,11423] ----
    for i, a in enumerate(iread(DATASET, index=f"0:{N_CORPUS}")):
        fam = a.info.get("config_type", "UNKNOWN")
        nSi, nO, osi, x = comp(a)
        dom = domain_of(fam, nO)
        has_dft = bool(a.calc and a.calc.results.get("energy") is not None)
        rows.append({"structure_id": f"corpus:{i:05d}", "structure_hash": h_struct(a),
                     "source_file": "03_allegro_train/dataset.xyz", "source_trajectory": fam,
                     "frame_index": i, "raw_source_family": fam, "scientific_domain": dom,
                     "natoms": len(a), "N_Si": nSi, "N_O": nO, "O_over_Si": osi, "oxygen_deficiency_x": x,
                     "DFT_reference_status": ("teacher_corpus_DFT" if has_dft else "none"),
                     "student_training_eligible": dom not in ("OOD_OR_EXCLUDED","UNRESOLVED"),
                     "preserve_for_PC004": False, "notes": ""})
    n_corpus = len(rows)
    # ---- pool B: production deployment trajectories (streamed, strided-tagged) ----
    for mode, (dom, zmap) in PROD.items():
        f = f"{RES}/sio2x_production/{mode}/traj.dump"
        if not Path(f).exists(): continue
        for j, a in enumerate(iread(f, format="lammps-dump-text", index=":")):
            # map LAMMPS types -> elements (pair_coeff order O Si => 1->O,2->Si)
            a.numbers = np.array([zmap[t] for t in a.get_atomic_numbers()]) if set(a.get_atomic_numbers())<= {1,2} else a.numbers
            nSi, nO, osi, x = comp(a)
            sel = (j % PROD_STRIDE == 0)
            rows.append({"structure_id": f"prod:{mode}:{j:04d}", "structure_hash": h_struct(a),
                         "source_file": f"sio2x_production/{mode}/traj.dump", "source_trajectory": f"prod:{mode}",
                         "frame_index": j, "raw_source_family": f"prod_{mode}", "scientific_domain": dom,
                         "natoms": len(a), "N_Si": nSi, "N_O": nO, "O_over_Si": osi, "oxygen_deficiency_x": x,
                         "DFT_reference_status": "none",
                         "student_training_eligible": True, "preserve_for_PC004": False,
                         "notes": ("stride_keep" if sel else "stride_drop_redundant")})
    n_prod = len(rows) - n_corpus

    # ---- PC004 DFT exclusions: 11 SCAN cells ----
    excl = []
    for k, a in enumerate(iread(SCAN, index=":")):
        nSi, nO, osi, x = comp(a)
        excl.append({"scan_cell_index": k, "cell_id": a.info.get("cell_id", f"cell_{k}"),
                     "structure_hash": h_struct(a), "natoms": len(a), "N_Si": nSi, "N_O": nO,
                     "oxygen_deficiency_x": x, "role": "HELDOUT_DFT_reference_PC004", "student_training_eligible": False})
    excl_hashes = {e["structure_hash"] for e in excl}

    # ---- SELECTION (evidence-based; redundancy-controlled) ----
    # central corpus: keep all eligible; broad corpus: cap per family (even-spacing); production: strided keep
    by_fam = {}
    for r in rows:
        if r["source_file"].endswith("dataset.xyz"):
            by_fam.setdefault(r["raw_source_family"], []).append(r)
    selected = []
    redundancy = {"production_stride": PROD_STRIDE, "corpus_family_cap_broad": CORPUS_FAMILY_CAP, "capped_families": {}, "dropped": {}}
    for r in rows:
        # never select excluded/unresolved/ood, or a frame matching a PC004 DFT cell
        if not r["student_training_eligible"]:
            continue
        if r["structure_hash"] in excl_hashes:
            r["preserve_for_PC004"] = True; r["student_training_eligible"] = False; continue
        if r["source_file"].endswith("traj.dump"):
            if r["notes"] == "stride_keep": selected.append(r)
            continue
        # corpus frame
        dom = r["scientific_domain"]
        if dom in CENTRAL:
            selected.append(r)                       # keep all central (target domains)
        elif dom in BROAD:
            fam = r["raw_source_family"]; grp = sorted(by_fam[fam], key=lambda z: z["frame_index"])
            n = len(grp)
            if n <= CORPUS_FAMILY_CAP:
                selected.append(r)
            else:
                keep_idx = {grp[round(t*(n-1)/(CORPUS_FAMILY_CAP-1))]["frame_index"] for t in range(CORPUS_FAMILY_CAP)}
                if r["frame_index"] in keep_idx: selected.append(r)
                else: redundancy["dropped"][fam] = redundancy["dropped"].get(fam,0)+1
                redundancy["capped_families"][fam] = CORPUS_FAMILY_CAP
    sel_ids = {r["structure_id"] for r in selected}

    # ---- source/trajectory-aware deterministic train/val split (contiguous tail=val per group) ----
    groups = {}
    for r in selected:
        groups.setdefault(r["source_trajectory"], []).append(r)
    train, val = [], []
    for g, items in groups.items():
        items = sorted(items, key=lambda z: z["frame_index"])
        ncut = max(1, int(round(len(items)*0.1)))
        vset = set(id(x) for x in items[-ncut:])       # contiguous tail -> val (no adjacent straddle)
        for it in items:
            (val if id(it) in vset else train).append(it)

    # ---- write all manifests ----
    keys = ["structure_id","structure_hash","source_file","source_trajectory","frame_index","raw_source_family",
            "scientific_domain","natoms","N_Si","N_O","O_over_Si","oxygen_deficiency_x","DFT_reference_status",
            "student_training_eligible","preserve_for_PC004","notes"]
    def wcsv(name, data):
        with open(OUT/name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(data)
    wcsv("pc002_candidate_frame_inventory.csv", rows)
    wcsv("pc002_selected_structure_manifest.csv", selected)
    wcsv("pc002_student_train_structure_manifest.csv", train)
    wcsv("pc002_student_valid_structure_manifest.csv", val)
    with open(OUT/"pc004_dft_exclusion_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(excl[0].keys())); w.writeheader(); w.writerows(excl)

    def dom_counts(data):
        d = {}
        for r in data: d[r["scientific_domain"]] = d.get(r["scientific_domain"],0)+1
        return dict(sorted(d.items(), key=lambda kv:-kv[1]))
    counts = {"candidate_total": len(rows), "candidate_corpus": n_corpus, "candidate_production": n_prod,
              "candidate_by_domain": dom_counts(rows),
              "selected_total": len(selected), "selected_by_domain": dom_counts(selected),
              "train_total": len(train), "val_total": len(val),
              "train_by_domain": dom_counts(train), "val_by_domain": dom_counts(val),
              "selected_by_source": {"corpus": sum(1 for r in selected if r['source_file'].endswith('dataset.xyz')),
                                     "production": sum(1 for r in selected if r['source_file'].endswith('traj.dump'))}}
    (OUT/"pc002_domain_counts.json").write_text(json.dumps(counts, indent=2)+"\n")
    (OUT/"pc002_domain_mapping.json").write_text(json.dumps(
        {"family_to_domain": FAMILY_DOMAIN, "composition_override": "nO==0 -> OOD_OR_EXCLUDED (elemental Si)",
         "production_source_to_domain": {m: d for m,(d,_) in PROD.items()},
         "central_domains": sorted(CENTRAL), "broad_domains": sorted(BROAD),
         "unresolved_policy": "oxide frame with unmapped family -> UNRESOLVED (not selected)"}, indent=2)+"\n")
    redundancy["dropped_total"] = sum(redundancy["dropped"].values())
    (OUT/"pc002_redundancy_report.json").write_text(json.dumps(redundancy, indent=2)+"\n")
    print(json.dumps(counts, indent=2))
    print("PC004 exclusions:", len(excl), "SCAN DFT cells")

if __name__ == "__main__":
    sys.exit(main())
