#!/usr/bin/env python3
"""Deterministic teacher-label validation (§9). NOT an LLM judge — pure numeric/structural checks.
Exit 0 + PASS json if valid; exit 1 + FAIL json otherwise (controller stage fails deterministically).
Usage: devv6_validate_labels.py <teacher_labeled.extxyz> <selected_manifest.csv> <dft_exclusion.csv> <out.json>"""
import sys, csv, json, hashlib
import numpy as np
from ase.io import read

labeled, manifest_csv, dft_csv, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def h(a):
    m=hashlib.sha256(); m.update(np.round(a.get_positions(),4).tobytes())
    m.update("".join(a.get_chemical_symbols()).encode())
    try: m.update(np.round(a.get_cell().array,4).tobytes())
    except Exception: pass
    return m.hexdigest()[:16]

man = {r["structure_id"]: r for r in csv.DictReader(open(manifest_csv))}
dft_hashes = {r["structure_hash"] for r in csv.DictReader(open(dft_csv))}
frames = read(labeled, index=":")
errs=[]; ok=0; seen=set()
for i,a in enumerate(frames):
    sid=a.info.get("structure_id")
    if sid is None: errs.append(f"frame {i}: missing structure_id"); continue
    if sid in seen: errs.append(f"{sid}: duplicate"); continue
    seen.add(sid)
    if sid not in man: errs.append(f"{sid}: not in DEV manifest"); continue
    r=man[sid]
    if h(a) != r["structure_hash"]: errs.append(f"{sid}: structure_hash mismatch")
    if a.info.get("structure_hash") in dft_hashes or h(a) in dft_hashes: errs.append(f"{sid}: PC004 DFT cell present in training pool")
    if int(r["natoms"]) != len(a): errs.append(f"{sid}: natoms mismatch")
    syms=set(a.get_chemical_symbols())
    if not syms <= {"O","Si"}: errs.append(f"{sid}: unexpected species {syms}")
    e=a.info.get("teacher_energy")
    if e is None or not np.isfinite(float(e)): errs.append(f"{sid}: energy missing/non-finite")
    F=a.arrays.get("teacher_forces")
    if F is None: errs.append(f"{sid}: forces missing")
    else:
        F=np.asarray(F)
        if F.shape != (len(a),3): errs.append(f"{sid}: forces shape {F.shape} != ({len(a)},3)")
        elif not np.all(np.isfinite(F)): errs.append(f"{sid}: non-finite forces")
    if not [x for x in errs if x.startswith(sid)]: ok+=1
missing=[sid for sid in man if sid not in seen]
if missing: errs.append(f"missing {len(missing)} expected structures e.g. {missing[:3]}")

verdict = "PASS" if (not errs and ok==len(man)) else "FAIL"
res={"verdict":verdict,"n_expected":len(man),"n_labeled":len(frames),"n_ok":ok,
     "dft_cells_excluded_checked":len(dft_hashes),"n_errors":len(errs),"errors":errs[:25],
     "teacher_energy_key":"info.teacher_energy","teacher_forces_key":"arrays.teacher_forces"}
open(out,"w").write(json.dumps(res,indent=2)+"\n")
print(json.dumps({"verdict":verdict,"n_expected":len(man),"n_labeled":len(frames),"n_ok":ok,"n_errors":len(errs)}))
sys.exit(0 if verdict=="PASS" else 1)
