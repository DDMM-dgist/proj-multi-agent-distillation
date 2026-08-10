#!/usr/bin/env python3
"""Track A groundwork: exact seed-123 split x config_type distribution over the FIRST 11424 frames
(the original training pool; extra appended frames excluded). No frame loading — uses the ordered
config_type list + torch random_split(seed=123). Validates against known central train counts."""
import json, sys
from pathlib import Path
import torch

SC = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad")
ROOT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime")
N = 11424
DOMAIN = {"amorphous_SiO2": {"bulk_amo","quench","quench_int_AL","liquid"},
          "SiO2x_dilute_vacancy": {"vacancy_int_AL","vacancy","SiOx_int_AL"},
          "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL","quench_max_AL","surfaces_max_AL"}}
CENTRAL = {c for s in DOMAIN.values() for c in s}
def dfam(c):
    for d,s in DOMAIN.items():
        if c in s: return d
    return None

cfgs = [l.strip().split("=",1)[1] for l in open(SC/"config_types_ordered.txt")][:N]
assert len(cfgs) == N, len(cfgs)
gen = torch.Generator().manual_seed(123)
sp = torch.utils.data.random_split(list(range(N)), [0.8,0.1,0.1], generator=gen)
train_idx = sorted(int(x) for x in sp[0].indices)
val_idx   = sorted(int(x) for x in sp[1].indices)
test_idx  = sorted(int(x) for x in sp[2].indices)

def dist(idxs):
    d = {}
    for i in idxs: d[cfgs[i]] = d.get(cfgs[i],0)+1
    return dict(sorted(d.items(), key=lambda kv:-kv[1]))

train_dist = dist(train_idx)
# central train validation
central_train = {d:0 for d in DOMAIN}
for i in train_idx:
    fam = dfam(cfgs[i])
    if fam: central_train[fam]+=1
# non-central (replay pool) train families
noncentral = {c:n for c,n in train_dist.items() if c not in CENTRAL}

out = {
  "N_original_pool": N, "file_total_frames_with_config_type": sum(1 for _ in open(SC/"config_types_ordered.txt")),
  "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
  "central_train_by_domain": central_train,
  "expected_central": {"amorphous_SiO2":1067,"SiO2x_dilute_vacancy":1192,"SiO2x_clustered_vacancy_voidsurface":707},
  "central_train_matches_expected": central_train == {"amorphous_SiO2":1067,"SiO2x_dilute_vacancy":1192,"SiO2x_clustered_vacancy_voidsurface":707},
  "central_train_total": sum(central_train.values()),
  "noncentral_train_pool_total": sum(noncentral.values()),
  "noncentral_train_by_config": noncentral,
  "central_train_by_config": {c:n for c,n in train_dist.items() if c in CENTRAL},
}
(ROOT/"work"/"trackA_split_distribution.json").write_text(json.dumps(out, indent=2)+"\n")
# also persist the exact index lists for the dataset builder
(ROOT/"work"/"trackA_split_indices.json").write_text(json.dumps(
    {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, )+"\n")
print(json.dumps({k:out[k] for k in ("split_sizes","central_train_by_domain","central_train_matches_expected",
                                     "central_train_total","noncentral_train_pool_total")}, indent=2))
print("NONCENTRAL TRAIN FAMILIES (replay pool):")
print(json.dumps(noncentral, indent=2))
