#!/usr/bin/env python3
"""DEV-SMALL: deterministic stratified 400-structure subset of the FROZEN PC002 5,552 ensemble.
Full PC002 remains frozen/untouched. Production lineages represented; large ~3000-atom production
frames counted inside their scientific domains (not a separate domain). 11 SCAN DFT cells asserted
absent. Writes small manifests + filters devv6_ensemble.xyz -> devv6small_ensemble.xyz with a DEV
train/valid split. Deterministic (no RNG)."""
import csv, json, math, hashlib
from pathlib import Path
from collections import defaultdict
from ase.io import read, write

W = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime/work")
TARGET = {"SiO2x_dilute" : None}  # placeholder; real map below by scientific_domain
DOM_TARGET = {"DILUTE_OXYGEN_DEFICIENT":140, "CLUSTERED_VOID":108, "STOICHIOMETRIC_AMORPHOUS":46,
              "AMBIENT_CRYSTAL":43, "SURFACE":32, "HIGH_T_LIQUID":23, "OTHER_RELEVANT_SIO":8}
PROD_SUBQUOTA = {"STOICHIOMETRIC_AMORPHOUS":2, "DILUTE_OXYGEN_DEFICIENT":4, "CLUSTERED_VOID":4}  # 10 large prod frames
VAL_FRAC = 0.10

def pick_even(items, k):
    n=len(items)
    if k<=0 or n==0: return []
    if k>=n: return list(items)
    if k==1: return [items[n//2]]
    return [items[round(j*(n-1)/(k-1))] for j in range(k)]

sel = list(csv.DictReader(open(W/"pc002_selected_structure_manifest.csv")))
for r in sel: r["natoms"]=int(r["natoms"])
dft_hashes = {r["structure_hash"] for r in csv.DictReader(open(W/"pc004_dft_exclusion_manifest.csv"))}

chosen = []
for dom, tgt in DOM_TARGET.items():
    rows = [r for r in sel if r["scientific_domain"]==dom]
    prod = [r for r in rows if r["provenance"]=="PRODUCTION_MD"]
    corp = [r for r in rows if r["provenance"]!="PRODUCTION_MD"]
    # production: spread across trajectories (source_group), even-spaced within each
    picks_prod=[]
    q = PROD_SUBQUOTA.get(dom,0)
    if q and prod:
        by_traj = defaultdict(list)
        for r in prod: by_traj[r["source_group"]].append(r)
        trajs = sorted(by_traj)
        per = [q//len(trajs)]*len(trajs)
        for i in range(q % len(trajs)): per[i]+=1
        for t,kk in zip(trajs, per):
            picks_prod += pick_even(sorted(by_traj[t], key=lambda z:int(z["frame_index"])), kk)
    # corpus: round-robin across raw_source_family, even-spaced within family (avoid neighbors)
    need = tgt - len(picks_prod)
    by_fam = defaultdict(list)
    for r in corp: by_fam[r["raw_source_family"]].append(r)
    fams = sorted(by_fam)
    for f in fams: by_fam[f]=sorted(by_fam[f], key=lambda z:int(z["frame_index"]))
    # allocate need across families proportionally to sqrt(size), min 1 where possible
    sizes={f:len(by_fam[f]) for f in fams}; tot=sum(math.sqrt(s) for s in sizes.values()) or 1
    alloc={f:max(1, round(need*math.sqrt(sizes[f])/tot)) for f in fams}
    # trim/expand to exactly `need`
    picks_corp=[]
    for f in fams:
        picks_corp += pick_even(by_fam[f], min(alloc[f], sizes[f]))
    # deterministic order, then cut/extend to need
    picks_corp = sorted({r["structure_id"]:r for r in picks_corp}.values(), key=lambda z:z["structure_id"])
    if len(picks_corp) > need: picks_corp = pick_even(picks_corp, need)
    elif len(picks_corp) < need:  # top up from remaining corpus deterministically
        have={r["structure_id"] for r in picks_corp}
        extra=[r for r in sorted(corp, key=lambda z:z["structure_id"]) if r["structure_id"] not in have]
        picks_corp += extra[:need-len(picks_corp)]
    chosen += picks_prod + picks_corp

# integrity
ids={r["structure_id"] for r in chosen}
assert len(ids)==len(chosen), "duplicate structure in DEV subset"
assert dft_hashes.isdisjoint({r["structure_hash"] for r in chosen}), "PC004 DFT cell leaked into DEV subset"
n_prod=sum(1 for r in chosen if r["provenance"]=="PRODUCTION_MD")

# DEV split: per-domain hash-ordered 90/10 (deterministic, decorrelated, source-aware enough for DEV)
train=[]; val=[]
bydom=defaultdict(list)
for r in chosen: bydom[r["scientific_domain"]].append(r)
for dom, rows in bydom.items():
    rows=sorted(rows, key=lambda z:z["structure_hash"])
    nv=max(1, round(len(rows)*VAL_FRAC))
    val+=rows[-nv:]; train+=rows[:-nv]
train_ids={r["structure_id"] for r in train}; val_ids={r["structure_id"] for r in val}

# write manifests
keys=list(sel[0].keys())
def wcsv(name,data):
    with open(W/name,"w",newline="") as fh:
        wr=csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); wr.writeheader(); wr.writerows(data)
wcsv("devv6small_selected_manifest.csv", chosen)
wcsv("devv6small_train_manifest.csv", train)
wcsv("devv6small_valid_manifest.csv", val)

# filter the already-materialized 5552 ensemble down to the 400 + set DEV ft_split
allframes = read(W/"devv6_ensemble.xyz", index=":")
keep=[]
for a in allframes:
    sid=a.info.get("structure_id")
    if sid in ids:
        a.info["ft_split"]="train" if sid in train_ids else "valid"
        a.info["campaign_id"]="SIO2_DISTILLATION_DEV_V6_SMALL_001"
        a.info["dev_subset_of"]="PC002_FULL_DATASET"; a.info["development_only"]=True
        keep.append(a)
# order by chosen manifest
order={r["structure_id"]:i for i,r in enumerate(chosen)}
keep.sort(key=lambda a: order[a.info["structure_id"]])
write(W/"devv6small_ensemble.xyz", keep, format="extxyz")

def dc(data):
    d=defaultdict(int)
    for r in data: d[r["scientific_domain"]]+=1
    return dict(sorted(d.items(), key=lambda kv:-kv[1]))
summary={"DEV_SUBSET_OF":"PC002_FULL_DATASET","DEVELOPMENT_ONLY":True,"FINAL_SCIENTIFIC_DATASET":False,
         "n_selected":len(chosen),"n_production_large_frames":n_prod,
         "by_domain":dc(chosen),"target_by_domain":DOM_TARGET,
         "train":len(train),"valid":len(val),"train_by_domain":dc(train),"valid_by_domain":dc(val),
         "dft_cells_excluded":len(dft_hashes),"dft_overlap":0,
         "materialized":str(W/"devv6small_ensemble.xyz")}
(W/"devv6small_domain_counts.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
