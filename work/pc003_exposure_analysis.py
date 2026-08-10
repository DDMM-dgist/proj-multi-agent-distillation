#!/usr/bin/env python3
"""PC003 — effective training exposure under 3 struct_weight policies (NO training, NO teacher labels).

Derived from SIMPLE-NN v2.0.0 source (MSELoss reduction='none'; struct_weight per-structure, applied
in TRAINING only):
  * force-loss contribution of structure s  ∝  weight_s * N_s   (weight multiplies each of its N_s*3
    component losses; batch normalizes by total components) -> weight_s ∝ 1/N_s equalizes structures.
  * energy-loss contribution of structure s ∝  weight_s        (per-atom energy, per-structure mean).
Policies:
  A HISTORICAL/UNWEIGHTED  weight_s = 1
  B SIZE-NORMALIZED        weight_s ∝ 1/N_s
  C SIZE+DOMAIN            B, then modest amorphous x2 (matrix under-represented), renormalized
Exposure fractions (sum to 1) reported per domain + per source, for: frames, atoms,
effective FORCE-loss, effective ENERGY-loss.
"""
import csv, json
from pathlib import Path
W = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime/work")
rows=[r for r in csv.DictReader(open(W/"pc002_selected_structure_manifest.csv"))]
for r in rows: r["natoms"]=int(r["natoms"])
DOMS=["STOICHIOMETRIC_AMORPHOUS","DILUTE_OXYGEN_DEFICIENT","CLUSTERED_VOID","AMBIENT_CRYSTAL","SURFACE","HIGH_T_LIQUID","OTHER_RELEVANT_SIO"]
def frac(vals):
    t=sum(vals.values()) or 1.0
    return {k:round(v/t,4) for k,v in vals.items()}
def by(keyfn, wfn):
    d={}
    for r in rows: d[keyfn(r)]=d.get(keyfn(r),0.0)+wfn(r)
    return d

import math
# geometric mean of raw 1/N weights (to center the clip)
_geo=math.exp(sum(math.log(1.0/r["natoms"]) for r in rows)/len(rows))
CAP=math.sqrt(8.0)  # bounded dynamic range = 8x (clip normalized weight to [1/2.83, 2.83])
def wA(r): return 1.0
def wB(r): return 1.0/r["natoms"]                                  # pure size-normalized (over-corrects)
def wC(r):                                                         # BOUNDED size-normalized (ratio<=8)
    rw=(1.0/r["natoms"])/_geo
    return min(max(rw,1.0/CAP),CAP)

def exposure(wfn):
    # force contribution ∝ weight*N ; energy contribution ∝ weight
    dom_force=by(lambda r:r["scientific_domain"], lambda r: wfn(r)*r["natoms"])
    dom_energy=by(lambda r:r["scientific_domain"], lambda r: wfn(r))
    src_force=by(lambda r:("production" if r["provenance"]=="PRODUCTION_MD" else "corpus"), lambda r: wfn(r)*r["natoms"])
    return frac(dom_force), frac(dom_energy), frac(src_force)

frames=frac(by(lambda r:r["scientific_domain"], lambda r:1.0))
atoms =frac(by(lambda r:r["scientific_domain"], lambda r:float(r["natoms"])))
src_frames=frac(by(lambda r:("production" if r["provenance"]=="PRODUCTION_MD" else "corpus"), lambda r:1.0))
src_atoms =frac(by(lambda r:("production" if r["provenance"]=="PRODUCTION_MD" else "corpus"), lambda r:float(r["natoms"])))

out={"n_structures":len(rows),"total_atoms":sum(r["natoms"] for r in rows),
     "frame_exposure_by_domain":frames,"atom_exposure_by_domain":atoms,
     "source_frame_fraction":src_frames,"source_atom_fraction":src_atoms,"policies":{}}
# weight range (normalized so mean weight = 1) for B
import statistics
invN=[1.0/r["natoms"] for r in rows]; m=statistics.fmean(invN)
wnorm=[w/m for w in invN]
for name,wfn in [("A_HISTORICAL_UNWEIGHTED",wA),("B_SIZE_NORMALIZED_PURE",wB),("C_SIZE_NORMALIZED_BOUNDED",wC)]:
    df,de,sf=exposure(wfn)
    out["policies"][name]={"force_exposure_by_domain":df,"energy_exposure_by_domain":de,
                           "force_exposure_by_source":sf}
out["policyB_weight_norm_mean1"]={"min_weight_(largest_cell)":round(min(wnorm),4),
    "max_weight_(smallest_cell)":round(max(wnorm),4),"ratio":round(max(wnorm)/min(wnorm),1)}
(W/"pc003_effective_domain_exposure.json").write_text(json.dumps(out,indent=2)+"\n")
# compact print
print("SOURCE force exposure  A vs B:")
for name in ("A_HISTORICAL_UNWEIGHTED","B_SIZE_NORMALIZED_PURE","C_SIZE_NORMALIZED_BOUNDED"):
    print(f"  {name:26s} production={out['policies'][name]['force_exposure_by_source'].get('production')}")
print("DOMAIN force exposure:")
print(f"  {'domain':26s} {'frames':>7} {'atoms(A)':>9} {'B':>7} {'C':>7}")
for d in DOMS:
    print(f"  {d:26s} {frames.get(d,0):>7} {atoms.get(d,0):>9} {out['policies']['B_SIZE_NORMALIZED_PURE']['force_exposure_by_domain'].get(d,0):>7} {out['policies']['C_SIZE_NORMALIZED_BOUNDED']['force_exposure_by_domain'].get(d,0):>7}")
print("B weight range (mean=1):", out["policyB_weight_norm_mean1"])
