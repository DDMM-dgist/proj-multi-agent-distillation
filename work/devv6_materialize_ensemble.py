#!/usr/bin/env python3
"""DEV-V6 campaign: materialize the FROZEN PC002 5,552-structure ensemble into a single extxyz
for the controller's teacher_labeling stage. Structures ONLY (no teacher labels; any stored
production forces are dropped). Each frame carries info: structure_id, structure_hash,
scientific_domain, parent_structure_id (=source_group lineage key), campaign_id, ft_split.
Deterministic; reads the frozen manifests + the frozen split. NO redesign of PC002."""
import sys, csv
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase import Atoms

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
DATASET = f"{RES}/03_allegro_train/dataset.xyz"
ROOT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime")
W = ROOT/"work"
OUT = W/"devv6_ensemble.xyz"
CAMP = "SIO2_DISTILLATION_DEV_V6_001"

sel = list(csv.DictReader(open(W/"pc002_selected_structure_manifest.csv")))
train_ids = {r["structure_id"] for r in csv.DictReader(open(W/"pc002_student_train_structure_manifest.csv"))}
val_ids   = {r["structure_id"] for r in csv.DictReader(open(W/"pc002_student_valid_structure_manifest.csv"))}

# group corpus indices; production by (mode,frame)
corpus_idx = {int(r["frame_index"]): r for r in sel if r["source_file"].endswith("dataset.xyz")}
prod_rows = [r for r in sel if r["source_file"].endswith("traj.dump")]

out_by_id = {}
# corpus: one read, pick needed frames
print(f"reading corpus frames for {len(corpus_idx)} structures ...", flush=True)
for i, a in enumerate(read(DATASET, index=":")):
    if i in corpus_idx:
        r = corpus_idx[i]
        b = Atoms(symbols=a.get_chemical_symbols(), positions=a.get_positions(),
                  cell=a.get_cell(), pbc=a.get_pbc())   # structure only, no calc/labels
        out_by_id[r["structure_id"]] = (r, b)
# production: read each dump once, map types 1->O,2->Si
from collections import defaultdict
prod_by_mode = defaultdict(dict)
for r in prod_rows: prod_by_mode[r["source_group"]][int(r["frame_index"])] = r
for sg, fmap in prod_by_mode.items():
    mode = sg.split(":",1)[1]
    f = f"{RES}/sio2x_production/{mode}/traj.dump"
    for j, a in enumerate(read(f, format="lammps-dump-text", index=":")):
        if j in fmap:
            r = fmap[j]
            nums = [ {1:8,2:14}[t] for t in a.get_atomic_numbers() ] if set(a.get_atomic_numbers())<={1,2} else a.get_atomic_numbers()
            b = Atoms(numbers=nums, positions=a.get_positions(), cell=a.get_cell(), pbc=True)
            out_by_id[r["structure_id"]] = (r, b)

# write in manifest order with provenance info
frames = []
missing = []
for r in sel:
    sid = r["structure_id"]
    if sid not in out_by_id: missing.append(sid); continue
    _, b = out_by_id[sid]
    b.info["structure_id"] = sid
    b.info["structure_hash"] = r["structure_hash"]
    b.info["scientific_domain"] = r["scientific_domain"]
    b.info["parent_structure_id"] = r["source_group"]     # lineage key for split integrity
    b.info["campaign_id"] = CAMP
    b.info["ft_split"] = "train" if sid in train_ids else ("valid" if sid in val_ids else "unassigned")
    frames.append(b)
write(OUT, frames, format="extxyz")
print(f"wrote {len(frames)} structures -> {OUT}")
print(f"missing: {len(missing)} (should be 0)")
if missing: print("MISSING sample:", missing[:5]); sys.exit(1)
# quick split tally
tr = sum(1 for f in frames if f.info['ft_split']=='train'); va = sum(1 for f in frames if f.info['ft_split']=='valid')
print(f"split: train={tr} valid={va}")
