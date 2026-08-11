#!/usr/bin/env python3
"""TRACK A — build the portable target-focused fine-tune dataset (PREPARE ONLY; no training here).

Core = ALL central-domain frames of the exact seed-123 TRAIN split (amorphous/dilute/clustered).
Replay = deterministic sqrt-stratified, diversity-aware selection from the remaining (non-central)
TRAIN frames to prevent catastrophic forgetting. NO original val/test frames ever enter training
(leakage guard). Writes extxyz (standard energy= / forces keys, byte-consistent with how nequip
read the original) + full per-frame manifest + provenance. Deterministic (no RNG beyond the fixed
seed-123 split)."""
import sys, json, csv, hashlib, math
from pathlib import Path
import numpy as np
import torch
from ase.io import read, write

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
DATASET = f"{RES}/03_allegro_train/dataset.xyz"
DATASET_SHA = "382d0b2b35ed9c571314ff59df71e9c9"  # sha256 prefix (32) of dataset.xyz
BASE_CKPT_SHA = "51342b332ba04287"                 # base teacher Lightning best.ckpt
BASE_COMPILED_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
N = 11424
REPLAY_TARGET = 800
VAL_EVERY = 10   # deterministic internal val: every 10th within each stratum (rank%10==5)

PKG = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime/work/external_gpu_teacher_package")
DATADIR = PKG / "dataset"; MANDIR = PKG / "manifests"

DOMAIN = {"amorphous_SiO2": {"bulk_amo","quench","quench_int_AL","liquid"},
          "SiO2x_dilute_vacancy": {"vacancy_int_AL","vacancy","SiOx_int_AL"},
          "SiO2x_clustered_vacancy_voidsurface": {"SiOx_max_AL","quench_max_AL","surfaces_max_AL"}}
CENTRAL = {c for s in DOMAIN.values() for c in s}
def dfam(c):
    for d,s in DOMAIN.items():
        if c in s: return d
    return None

def frame_sha(a):
    h = hashlib.sha256()
    h.update(f"{len(a)}".encode())
    e = a.calc.results.get("energy") if (a.calc and a.calc.results) else None
    h.update(f"{None if e is None else round(float(e),6)}".encode())
    h.update(np.round(a.get_positions(),6).tobytes())
    F = a.calc.results.get("forces") if (a.calc and a.calc.results) else None
    if F is not None: h.update(np.round(np.asarray(F),6).tobytes())
    h.update("".join(a.get_chemical_symbols()).encode())
    return h.hexdigest()[:12]

def deterministic_pick(sorted_idx, k):
    n = len(sorted_idx)
    if k >= n: return list(sorted_idx)
    if k == 1: return [sorted_idx[n//2]]
    return [sorted_idx[round(j*(n-1)/(k-1))] for j in range(k)]

def main():
    DATADIR.mkdir(parents=True, exist_ok=True); MANDIR.mkdir(parents=True, exist_ok=True)
    print("reading frames 0:%d ..." % N, flush=True)
    frames = read(DATASET, index="0:%d" % N)
    assert len(frames) == N, len(frames)
    gen = torch.Generator().manual_seed(123)
    sp = torch.utils.data.random_split(list(range(N)), [0.8,0.1,0.1], generator=gen)
    train_idx = set(int(x) for x in sp[0].indices)
    val_idx   = set(int(x) for x in sp[1].indices)
    test_idx  = set(int(x) for x in sp[2].indices)

    # classify TRAIN frames
    core = []            # (idx, cfg, domain)
    replay_pool = {}     # family -> [idx,...]
    for i in sorted(train_idx):
        cfg = frames[i].info.get("config_type", "UNKNOWN")
        fam = dfam(cfg)
        if fam: core.append((i, cfg, fam))
        else: replay_pool.setdefault(cfg, []).append(i)

    # sqrt-stratified deterministic replay selection
    fams = {f: sorted(v) for f, v in replay_pool.items()}
    sqrt_w = {f: math.sqrt(len(v)) for f, v in fams.items()}
    W = sum(sqrt_w.values())
    alloc = {}
    for f, v in fams.items():
        a = max(2, round(REPLAY_TARGET * sqrt_w[f] / W))
        alloc[f] = min(a, len(v))
    replay = []
    for f, v in fams.items():
        for i in deterministic_pick(v, alloc[f]):
            replay.append((i, f, "replay:"+f))

    corpus = [(i, cfg, dom, "core") for (i, cfg, dom) in core] + \
             [(i, cfg, dom, "replay") for (i, cfg, dom) in replay]
    corpus.sort(key=lambda r: r[0])

    # LEAKAGE GUARD
    cidx = {r[0] for r in corpus}
    assert cidx.isdisjoint(val_idx), "corpus overlaps original VAL"
    assert cidx.isdisjoint(test_idx), "corpus overlaps original TEST"
    assert len(cidx) == len(corpus), "duplicate frame in corpus"

    # deterministic internal val split, stratified by (role, domain/family)
    strata = {}
    for (i, cfg, dom, role) in corpus:
        key = (role, dom if role=="core" else cfg)
        strata.setdefault(key, []).append(i)
    internal_val = set()
    for key, idxs in strata.items():
        for rank, i in enumerate(sorted(idxs)):
            if rank % VAL_EVERY == 5:
                internal_val.add(i)

    # write manifest + xyz
    rows = []
    ft_train, ft_val = [], []
    for (i, cfg, dom, role) in corpus:
        a = frames[i]
        syms = a.get_chemical_symbols()
        nSi = syms.count("Si"); nO = syms.count("O")
        split = "val" if i in internal_val else "train"
        rows.append({"dataset_index": i, "config_type": cfg, "domain": dom, "role": role,
                     "ft_split": split, "natoms": len(a), "nSi": nSi, "nO": nO,
                     "O_over_Si": round(nO/nSi, 4) if nSi else None, "content_sha12": frame_sha(a)})
        (ft_val if split=="val" else ft_train).append(a)

    write(DATADIR / "fine_tune_train.xyz", ft_train, format="extxyz")
    write(DATADIR / "fine_tune_val.xyz",   ft_val,   format="extxyz")

    with open(MANDIR / "frame_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # domain/role summaries
    def tally(pred):
        d = {}
        for r in rows:
            if pred(r): d[r["domain"]] = d.get(r["domain"],0)+1
        return dict(sorted(d.items()))
    core_by_dom = {}
    for r in rows:
        if r["role"]=="core": core_by_dom[r["domain"]] = core_by_dom.get(r["domain"],0)+1
    replay_by_fam = {}
    for r in rows:
        if r["role"]=="replay": replay_by_fam[r["config_type"]] = replay_by_fam.get(r["config_type"],0)+1

    sel = {
      "method": "core = all central-domain seed-123 TRAIN frames; replay = sqrt-stratified deterministic "
                "even-spacing over sorted dataset_index per non-central family (min 2, cap n_family), "
                "target %d. Internal val = deterministic rank%%%d==5 per (role,domain/family) stratum. No RNG "
                "beyond the fixed seed-123 split." % (REPLAY_TARGET, VAL_EVERY),
      "core_total": sum(core_by_dom.values()), "core_by_domain": core_by_dom,
      "replay_target": REPLAY_TARGET, "replay_total": len(replay), "replay_by_family": dict(sorted(replay_by_fam.items(), key=lambda kv:-kv[1])),
      "replay_allocation_requested": dict(sorted(alloc.items(), key=lambda kv:-kv[1])),
      "corpus_total": len(corpus),
      "ft_train_frames": len(ft_train), "ft_val_frames": len(ft_val),
      "ft_val_by_domain_role": {},
    }
    vb = {}
    for r in rows:
        if r["ft_split"]=="val":
            k = f"{r['role']}:{r['domain'] if r['role']=='core' else r['config_type']}"
            vb[k] = vb.get(k,0)+1
    sel["ft_val_by_domain_role"] = dict(sorted(vb.items()))
    (MANDIR / "selection_report.json").write_text(json.dumps(sel, indent=2)+"\n")

    prov = {
      "source_dataset": {"path": DATASET, "sha256_prefix32": DATASET_SHA, "original_pool_frames": N,
                         "note": "first %d frames = original training pool; exact reproduction of NequIP test metrics confirmed. Extra appended frames excluded." % N},
      "split": {"mechanism": "torch.utils.data.random_split(range(%d),[0.8,0.1,0.1], Generator.manual_seed(123))" % N,
                "sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
                "note": "identical to the original tutorial_Allegro.yaml data.seed=123 split"},
      "base_teacher_warm_start": {"lightning_best_ckpt_sha16": BASE_CKPT_SHA, "compiled_model_sha256": BASE_COMPILED_SHA,
                                  "source": "03_allegro_train/outputs/2025-11-13/12-39-46/best.ckpt"},
      "leakage_guard": "corpus disjoint from original VAL and TEST (asserted at build time)",
    }
    (MANDIR / "source_provenance.json").write_text(json.dumps(prov, indent=2)+"\n")

    print(json.dumps({"core_by_domain": core_by_dom, "core_total": sum(core_by_dom.values()),
                      "replay_total": len(replay), "corpus_total": len(corpus),
                      "ft_train": len(ft_train), "ft_val": len(ft_val),
                      "replay_by_family": sel["replay_by_family"]}, indent=2))

if __name__ == "__main__":
    sys.exit(main())
