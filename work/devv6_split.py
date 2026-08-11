#!/usr/bin/env python3
"""DEV-V6 dataset split = the FROZEN PC002 split (NOT a fresh random split).
Partition teacher_labeled.extxyz by the per-frame info['ft_split'] embedded from PC002.
Writes train.extxyz (4987) + validation.extxyz (558) + test.extxyz (558; == validation, the
within-distribution held-out) + split_manifest.json. Usage: devv6_split.py <labeled.extxyz> <outdir>"""
import sys, json
from pathlib import Path
from ase.io import read, write

labeled, outdir = sys.argv[1], sys.argv[2]
out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
frames = read(labeled, index=":")
train = [a for a in frames if a.info.get("ft_split") == "train"]
val   = [a for a in frames if a.info.get("ft_split") == "valid"]
write(out/"train.extxyz", train)
write(out/"validation.extxyz", val)
write(out/"test.extxyz", val)   # within-distribution held-out doubles as PC004 axis-A + early-stop monitor
tr_par = {a.info.get("parent_structure_id") for a in train}
va_par = {a.info.get("parent_structure_id") for a in val}
man = {"policy": "frozen_pc002_ft_split (source-aware blocked+buffer)",
       "n_train": len(train), "n_validation": len(val), "n_test": len(val),
       "train_parent_groups": len(tr_par), "validation_parent_groups": len(va_par),
       "parent_group_overlap": len(tr_par & va_par),
       "lineage_note": "WITHIN-DISTRIBUTION: train and validation SHARE parent lineages by design (blocked+buffer, no exact/adjacent leakage). NOT cross-lineage independent. Independent benchmark = 11 SCAN DFT cells (PC004 axis B).",
       "no_exact_frame_leakage": "guaranteed by frozen PC002 (train∩val structure_id = 0, shared-hash = 0)"}
(out/"split_manifest.json").write_text(json.dumps(man, indent=2)+"\n")
print(json.dumps({k: man[k] for k in ("n_train","n_validation","n_test","parent_group_overlap")}))
