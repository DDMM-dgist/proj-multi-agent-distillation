#!/usr/bin/env python3
"""Data-Curator Student-representability preflight (§3-4) + deterministic refill (§4).
Empirically tests which natoms the frozen SIMPLE-NN descriptor can featurize (real feature-gen via the
proven adapter), classifies the 400 DEV frames, and refills UNREPRESENTABLE ones with same-domain,
representable, non-duplicate, non-DFT frames from the frozen 5,552. No arbitrary cutoff; evidence-based.
Does NOT modify the 5,552 source record. Writes R1 ensemble + manifests + report."""
import sys, json, csv, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase.io import read, write
from adapters import load_config
from adapters.student import train_student

PROJ = Path(__file__).resolve().parents[1]; W = PROJ/"work"
LAB = PROJ/"runs/SIO2_DISTILLATION_DEV_V6_SMALL_002/artifacts/teacher_labeled.extxyz"
SMOKE_CFG = W/"devv6_smoke"/"student.smoke.yaml"
labeled = read(str(LAB), index=":")   # 400, with teacher_energy/forces + info
by_id = {a.info["structure_id"]: a for a in labeled}
sizes = sorted(set(len(a) for a in labeled))
# representative sizes to test: all small distinct (<20) + one mid + one large
test_sizes = [s for s in sizes if s < 20] + [s for s in sizes if s >= 100][:1] + [max(sizes)]
test_sizes = sorted(set(test_sizes))
cfg = load_config(str(SMOKE_CFG))
tmp = Path(tempfile.mkdtemp(prefix="repr_", dir=W))
size_result = {}
for s in test_sizes:
    frame = next(a for a in labeled if len(a) == s)
    ds = tmp/f"one_{s}.extxyz"; write(str(ds), [frame], format="extxyz")
    out = tmp/f"out_{s}"
    try:
        train_student(cfg, str(ds), out, 234)
        size_result[s] = "PASS"
    except Exception as e:
        size_result[s] = f"FAIL: {type(e).__name__}: {str(e)[:80]}"
    print(f"size {s}: {size_result[s]}", flush=True)
# threshold = smallest tested size that PASSED
passed = sorted(s for s, r in size_result.items() if r == "PASS")
min_repr = passed[0] if passed else 10**9
large_ok = (size_result.get(max(sizes), "").startswith("PASS"))
# classify all 400
rows = []
for a in labeled:
    sid = a.info["structure_id"]; n = len(a); dom = a.info.get("scientific_domain", "?")
    # representable if its size (or interpolated) passed; small below min_repr -> unrepresentable
    if n in size_result:
        repr_ok = size_result[n] == "PASS"
    else:
        repr_ok = n >= min_repr and (large_ok or n < max(sizes))
    rows.append({"structure_id": sid, "natoms": n, "scientific_domain": dom,
                 "structure_hash": a.info.get("structure_hash",""),
                 "representable": repr_ok,
                 "failure": ("" if repr_ok else size_result.get(n, f"natoms<{min_repr}_min_representable"))})
unrepr = [r for r in rows if not r["representable"]]
repr_ids = {r["structure_id"] for r in rows if r["representable"]}

# refill from frozen 5552 (pc002_selected_structure_manifest.csv): same domain, representable size,
# not already in the 400, non-DFT (they're already excluded from the 5552), no dup hash
sel5552 = list(csv.DictReader(open(W/"pc002_selected_structure_manifest.csv")))
have_hashes = {a.info.get("structure_hash","") for a in labeled}
have_ids = set(by_id)
bydom_pool = {}
for r in sel5552:
    if r["structure_id"] in have_ids: continue
    if int(r["natoms"]) < min_repr: continue
    if r["structure_hash"] in have_hashes: continue
    bydom_pool.setdefault(r["scientific_domain"], []).append(r)
for dom in bydom_pool: bydom_pool[dom].sort(key=lambda z: z["structure_id"])  # deterministic
replacements = []
used = set()
for u in unrepr:
    dom = u["scientific_domain"]; pool = bydom_pool.get(dom, [])
    pick = next((r for r in pool if r["structure_id"] not in used), None)
    if pick:
        used.add(pick["structure_id"]); replacements.append({"removed": u["structure_id"], "removed_natoms": u["natoms"],
            "added": pick["structure_id"], "added_natoms": int(pick["natoms"]), "domain": dom})

# build R1 selection = (representable 400) + replacements
r1_ids = set(repr_ids) | {r["added"] for r in replacements}
# materialize R1 ensemble (structures only, no labels) from devv6_ensemble.xyz (5552) + keep domain/split later
full = read(str(W/"devv6_ensemble.xyz"), index=":")
full_by_id = {a.info["structure_id"]: a for a in full}
missing = [i for i in r1_ids if i not in full_by_id]
r1_frames = []
for i in sorted(r1_ids):
    a = full_by_id[i].copy()
    a.info["campaign_id"] = "SIO2_DISTILLATION_DEV_V6_SMALL_R1"
    r1_frames.append(a)
# domain counts
def dc(ids):
    d = {}
    for i in ids:
        dom = full_by_id[i].info.get("scientific_domain","?") if i in full_by_id else "?"
        d[dom] = d.get(dom,0)+1
    return dict(sorted(d.items(), key=lambda kv:-kv[1]))
report = {
  "test_sizes": test_sizes, "size_result": size_result, "min_representable_natoms": min_repr,
  "large_frame_ok": large_ok, "n_dev": len(rows), "n_unrepresentable": len(unrepr),
  "unrepresentable": unrepr[:40], "n_replacements": len(replacements), "replacements": replacements[:40],
  "n_R1": len(r1_ids), "R1_by_domain": dc(r1_ids), "missing_in_ensemble": missing}
(W/"devv6_R1_representability_report.json").write_text(json.dumps(report, indent=2)+"\n")
write(str(W/"devv6_R1_ensemble.xyz"), r1_frames, format="extxyz")
print(json.dumps({k: report[k] for k in ("size_result","min_representable_natoms","large_frame_ok",
     "n_unrepresentable","n_replacements","n_R1","R1_by_domain")}, indent=2))
