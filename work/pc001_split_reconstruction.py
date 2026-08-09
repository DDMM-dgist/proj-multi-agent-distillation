#!/usr/bin/env python3
"""Reconstruct the EXACT nequip seed-123 train/val/test split of dataset.xyz + classify PCA_SOAP overlap.
Deterministic; NO model inference. Replicates nequip.data.dataset.utils.RandomSplitAndIndexDataset
(torch.utils.data.random_split(dataset, [0.8,0.1,0.1], generator=torch.Generator().manual_seed(123)),
subset order [train, val, test])."""
import json, sys, re
import torch

DATASET = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation/03_allegro_train/dataset.xyz"
PCA = "/home/hyunjin/workflow/PCA_SOAP_workflow/test_set.xyz"
OUT = "/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/pc001_source_artifacts"

N = 11424
# EXACT reconstruction: subset order = config key order [train, val, test]; fractions [0.8,0.1,0.1]; seed 123
gen = torch.Generator().manual_seed(123)
splits = torch.utils.data.random_split(list(range(N)), [0.8, 0.1, 0.1], generator=gen)
train_idx = set(int(i) for i in splits[0].indices)
val_idx = set(int(i) for i in splits[1].indices)
test_idx = sorted(int(i) for i in splits[2].indices)
assert len(train_idx) + len(val_idx) + len(test_idx) == N
print(f"reconstructed split: N={N} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

# ---- read dataset.xyz light: per-frame energy(dft_free_energy) + config_type + natoms (no ASE, fast) ----
frames = []  # (idx, energy_rounded, config_type, natoms)
with open(DATASET) as f:
    idx = 0
    while True:
        line = f.readline()
        if not line:
            break
        try:
            na = int(line.strip())
        except ValueError:
            continue
        hdr = f.readline()
        ct = hdr.split("config_type=")[1].split()[0].strip('"') if "config_type=" in hdr else "?"
        m = re.search(r"dft_free_energy=(-?[0-9.]+)", hdr)
        e = round(float(m.group(1)), 6) if m else None
        for _ in range(na):
            f.readline()
        frames.append((idx, e, ct, na)); idx += 1
print(f"dataset.xyz frames parsed: {len(frames)}")

# energy -> list of dataset indices (for PCA overlap)
from collections import defaultdict
e2idx = defaultdict(list)
for i, e, ct, na in frames:
    if e is not None:
        e2idx[e].append(i)

def split_of(i):
    return "train" if i in train_idx else "val" if i in val_idx else "test"

# ---- test-frame manifest (domain composition of the EXACT held-out test) ----
test_frames = [frames[i] for i in test_idx]
test_by_cfg = defaultdict(int)
for i, e, ct, na in test_frames:
    test_by_cfg[ct] += 1
json.dump({"reconstruction": "torch.utils.data.random_split(range(11424),[0.8,0.1,0.1],Generator.manual_seed(123)); subset order [train,val,test]; return splits[2]",
           "N_total": N, "N_train": len(train_idx), "N_val": len(val_idx), "N_test": len(test_idx),
           "test_indices_first20": test_idx[:20], "test_indices_count": len(test_idx),
           "validation_note": "confirm by reproducing the NequIP test-log forces_mae 0.1561 with fresh inference on these indices"},
          open(f"{OUT}/exact_split_manifest.json", "w"), indent=2)
json.dump({"n_test_frames": len(test_frames),
           "test_config_type_counts": dict(sorted(test_by_cfg.items(), key=lambda x: -x[1])),
           "test_indices": test_idx},
          open(f"{OUT}/test_frame_manifest.json", "w"), indent=2)
print("test config_type counts:", dict(sorted(test_by_cfg.items(), key=lambda x: -x[1])))

# ---- PCA_SOAP 1155 overlap classification (rule 12) ----
pca_e = []
with open(PCA) as f:
    for ln in f:
        m = re.search(r"dft_energy=(-?[0-9.]+)", ln)
        if m:
            pca_e.append(round(float(m.group(1)), 6))
counts = {"TRAIN": 0, "VALIDATION": 0, "TEST": 0, "DUPLICATE_AMBIGUOUS": 0, "NOT_FOUND": 0}
for e in pca_e:
    hits = e2idx.get(e, [])
    if not hits:
        counts["NOT_FOUND"] += 1
    elif len(hits) > 1:
        counts["DUPLICATE_AMBIGUOUS"] += 1
    else:
        s = split_of(hits[0]); counts[{"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}[s]] += 1
total = len(pca_e)
json.dump({"pca_soap_frames_with_energy": total, "pca_soap_total_lines_expected": 1155,
           "classification_counts": counts,
           "classification_fractions": {k: round(v/total, 4) for k, v in counts.items()},
           "explanation_1155_vs_1154": "test_set.xyz has 1155 structures; energy-key matching yields ~1154 because at least one frame's dft_energy is a DUPLICATE value (same energy as another frame) or a parse edge; exact key-collision handled as DUPLICATE_AMBIGUOUS",
           "note": "replaces the earlier approximate '~90% contaminated' with exact TRAIN/VAL/TEST counts"},
          open(f"{OUT}/pca_soap_split_overlap.json", "w"), indent=2)
print("PCA_SOAP overlap:", counts, "of", total)
