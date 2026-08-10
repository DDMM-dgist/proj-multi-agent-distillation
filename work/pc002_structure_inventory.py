#!/usr/bin/env python3
"""PC002 — teacher-INDEPENDENT structural inventory of the locally-available distillation
building blocks. Freezes structure IDENTITY (path, SHA, domain, composition, source, intended
role, DFT-exclusion status) — NO teacher labels are read or generated. Grounds PC002_STRUCTURE_SELECTION."""
import sys, json, csv, hashlib
from pathlib import Path
import numpy as np
from ase.io import read

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
TEACH = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/teacher"
OUT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime/work")
Z2S = {8: "O", 14: "Si"}

def sha_file(p, n=16):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()[:n]

def comp(atoms):
    s = atoms.get_chemical_symbols(); nSi = s.count("Si"); nO = s.count("O")
    # SiO2-x : x = 2 - O/Si  (for SiO2 x=0). Only meaningful when nSi>0 and material is oxide.
    x = round(2 - nO/nSi, 4) if nSi and nO else None
    return nSi, nO, x

def data_frame(path):
    # LAMMPS data, atomic style; pair_coeff order O Si => type1=O(8), type2=Si(14)
    a = read(path, format="lammps-data", Z_of_type={1: 8, 2: 14}, style="atomic")
    return a

rows = []
# 1) production defect cells (teacher-independent structures)
prod = {"pristine": "amorphous_SiO2", "random_x006": "SiO2x_dilute_vacancy", "random_x012": "SiO2x_dilute_vacancy",
        "sphere_x006": "SiO2x_clustered_vacancy_voidsurface", "sphere_x012": "SiO2x_clustered_vacancy_voidsurface",
        "plane_x006": "SiO2x_clustered_vacancy_voidsurface", "plane_x012": "SiO2x_clustered_vacancy_voidsurface"}
for name, dom in prod.items():
    p = f"{RES}/sio2x_production/structs/{name}.data"
    if not Path(p).exists(): continue
    try:
        a = data_frame(p); nSi, nO, x = comp(a)
        rows.append({"structure_id": f"prod:{name}", "path": p, "sha16": sha_file(p), "domain": dom,
                     "n_frames": 1, "natoms": len(a), "nSi": nSi, "nO": nO, "x_SiO2minus": x,
                     "source": "sio2x_production MD (single relaxed cell)", "intended_role": "student_train_candidate",
                     "dft_exclusion": "no"})
    except Exception as e:
        rows.append({"structure_id": f"prod:{name}", "path": p, "sha16": sha_file(p), "domain": dom,
                     "n_frames": None, "natoms": None, "error": str(e)[:120], "intended_role": "student_train_candidate", "dft_exclusion": "no"})

# 2) preserved SCAN DFT cells (PC004 held-out; NEVER student training)
c = f"{RES}/scan_labeled_structures/sio2x_AL_labels_11cells.xyz"
if Path(c).exists():
    cells = read(c, index=":")
    for i, a in enumerate(cells):
        nSi, nO, x = comp(a)
        rows.append({"structure_id": f"scan_dft:{a.info.get('cell_id', i)}", "path": c, "sha16": sha_file(c),
                     "domain": "SiO2x_dilute_or_clustered_SCAN", "n_frames": 1, "natoms": len(a),
                     "nSi": nSi, "nO": nO, "x_SiO2minus": x, "source": "SCAN DFT AL cells (11)",
                     "intended_role": "HELDOUT_DFT_reference_PC004", "dft_exclusion": "YES_never_train"})

# 3) augment seeds (DFT-labeled pre-relabel; small)
for name, tag in [("input_small.xyz", "normal_cell_seed"), ("input_large.xyz", "large_cell_seed")]:
    p = f"{TEACH}/{name}"
    if not Path(p).exists(): continue
    frames = read(p, index=":")
    doms = {}
    for a in frames:
        ct = a.info.get("config_type", "?"); doms[ct] = doms.get(ct, 0)+1
    a0 = frames[0]; nSi, nO, x = comp(a0)
    rows.append({"structure_id": f"seed:{name}", "path": p, "sha16": sha_file(p), "domain": "augment_seed_pool",
                 "n_frames": len(frames), "natoms": len(a0), "nSi": nSi, "nO": nO, "x_SiO2minus": x,
                 "source": f"augment-atoms {tag}; config_types={doms}", "intended_role": "augment_seed",
                 "dft_exclusion": "no"})

# write manifest
keys = ["structure_id","path","sha16","domain","n_frames","natoms","nSi","nO","x_SiO2minus","source","intended_role","dft_exclusion"]
with open(OUT/"pc002_structure_manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

summ = {"n_structures_inventoried": len(rows),
        "by_domain": {}, "by_role": {}, "dft_excluded": sum(1 for r in rows if r.get("dft_exclusion","").startswith("YES"))}
for r in rows:
    summ["by_domain"][r["domain"]] = summ["by_domain"].get(r["domain"],0)+1
    summ["by_role"][r["intended_role"]] = summ["by_role"].get(r["intended_role"],0)+1
(OUT/"pc002_structure_inventory.json").write_text(json.dumps({"summary": summ, "structures": rows}, indent=2, default=str)+"\n")
print(json.dumps(summ, indent=2, default=str))
print("wrote pc002_structure_manifest.csv + pc002_structure_inventory.json")
