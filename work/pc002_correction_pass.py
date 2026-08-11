#!/usr/bin/env python3
"""PC002 focused CORRECTION pass (teacher-label-independent).

Fixes vs the prior 6,436 draft:
 1. Rebuild candidate space over FULL dataset.xyz (13,898) + production (1,057), provenance-tagged
    (ORIGINAL_TEACHER_CORPUS / POST_ORIGINAL_TRAINING_DATA / PRODUCTION_MD).
 2. Inventory + disposition the appended 2,474 frames (STUDENT / PC004_DFT_REFERENCE / REDUNDANT /
    OOD / UNRESOLVED). DFT-labeled independent SiOx appended frames are RESERVED as PC004 references
    (protect independent DFT before maximizing training size), NOT put in Student training.
 3. Trim redundant, off-deployment high-pressure crystalline (evidence-based diversity, not size).
 4. Corrected leakage-safe split: production = blocked temporal + BUFFER gap per trajectory;
    corpus = per-family hash-ordered (decorrelated, near-dup-guarded) split.
 5. Frame-weighted AND atom-weighted exposure + force-component counts + labeling cost.
No teacher labels. No training/DFT/MD.
"""
import sys, json, csv, hashlib
from pathlib import Path
import numpy as np
from ase.io import iread

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
DATASET = f"{RES}/03_allegro_train/dataset.xyz"
SCAN = f"{RES}/scan_labeled_structures/sio2x_AL_labels_11cells.xyz"
N_ORIG = 11424; N_TOTAL = 13898
PROD_STRIDE = 10
OUT = Path("/tmp/claude-1002/-home-hyunjin-CLADE-SiO2-x-distillatio-materials-ml-kit-paper-sio2-agentic-distillation/192ca6bd-72d4-47f0-b066-84bc7a70e6fa/scratchpad/proj-mad-pydanticai-full-runtime/work")

# per-family broad-replay caps (central kept uncapped). High-pressure crystalline trimmed hard.
BROAD_CAP = {"bulk_cryst": 300, "bulk_cryst_hp": 100, "highpressure_int_AL": 100, "highpressure_max_AL": 100,
             "surfaces": 300, "surfaces_int_AL": 150, "SiOx_crystal_amorphous_interfaces": 108,
             "liquid": 9999, "liq": 9999}
PROD = {"pristine": "STOICHIOMETRIC_AMORPHOUS", "random_x006": "DILUTE_OXYGEN_DEFICIENT",
        "random_x012": "DILUTE_OXYGEN_DEFICIENT", "sphere_x006": "CLUSTERED_VOID",
        "sphere_x012": "CLUSTERED_VOID", "plane_x006": "CLUSTERED_VOID", "plane_x012": "CLUSTERED_VOID"}
PROD_GEN = "historical Student/MLIP MD (LAMMPS pair_style nn) — positions only; stored fx/fy/fz NOT authoritative labels"
FAMILY_DOMAIN = {
  "bulk_amo":"STOICHIOMETRIC_AMORPHOUS","quench":"STOICHIOMETRIC_AMORPHOUS","amorph":"STOICHIOMETRIC_AMORPHOUS",
  "vacancy":"DILUTE_OXYGEN_DEFICIENT","vacancy_int_AL":"DILUTE_OXYGEN_DEFICIENT","SiOx_int_AL":"DILUTE_OXYGEN_DEFICIENT",
  "quench_int_AL":"DILUTE_OXYGEN_DEFICIENT","interstitial":"DILUTE_OXYGEN_DEFICIENT","divacancy":"DILUTE_OXYGEN_DEFICIENT",
  "SiOx_max_AL":"CLUSTERED_VOID","quench_max_AL":"CLUSTERED_VOID","surfaces_max_AL":"CLUSTERED_VOID","cluster":"CLUSTERED_VOID",
  "surfaces":"SURFACE","surfaces_int_AL":"SURFACE","liquid":"HIGH_T_LIQUID","liq":"HIGH_T_LIQUID",
  "bulk_cryst":"AMBIENT_CRYSTAL","bulk_cryst_hp":"AMBIENT_CRYSTAL","highpressure_int_AL":"AMBIENT_CRYSTAL",
  "highpressure_max_AL":"AMBIENT_CRYSTAL","SiOx_crystal_amorphous_interfaces":"OTHER_RELEVANT_SIO"}
CENTRAL = {"STOICHIOMETRIC_AMORPHOUS","DILUTE_OXYGEN_DEFICIENT","CLUSTERED_VOID"}

def h(a):
    m=hashlib.sha256(); m.update(np.round(a.get_positions(),4).tobytes())
    m.update("".join(a.get_chemical_symbols()).encode())
    try: m.update(np.round(a.get_cell().array,4).tobytes())
    except: pass
    return m.hexdigest()[:16]
def comp(a):
    s=a.get_chemical_symbols(); nSi=s.count("Si"); nO=s.count("O")
    return nSi,nO,(round(nO/nSi,4) if nSi else None),(round(2-nO/nSi,4) if (nSi and nO) else None)
def dom_of(fam,nO):
    if nO==0: return "OOD_OR_EXCLUDED"
    return FAMILY_DOMAIN.get(fam,"UNRESOLVED")

def main():
    scan_hashes=set();
    for a in iread(SCAN,index=":"): scan_hashes.add(h(a))
    rows=[]; seen={}
    # corpus (full 13898) with provenance tag
    for i,a in enumerate(iread(DATASET,index=f"0:{N_TOTAL}")):
        fam=a.info.get("config_type","UNKNOWN"); nSi,nO,osi,x=comp(a); H=h(a)
        prov="ORIGINAL_TEACHER_CORPUS" if i<N_ORIG else "POST_ORIGINAL_TRAINING_DATA"
        has_dft=bool(a.calc and a.calc.results.get("energy") is not None)
        dup = H in seen
        rows.append({"structure_id":f"corpus:{i:05d}","structure_hash":H,"provenance":prov,
            "source_file":"03_allegro_train/dataset.xyz","source_group":fam,"frame_index":i,
            "raw_source_family":fam,"scientific_domain":dom_of(fam,nO),"natoms":len(a),"N_Si":nSi,"N_O":nO,
            "O_over_Si":osi,"oxygen_deficiency_x":x,"DFT_label":has_dft,"dup_of":seen.get(H,""),
            "generator_model":"VASP_DFT" if has_dft else "unknown","state_meta":""})
        if not dup: seen[H]=f"corpus:{i:05d}"
    n_orig=sum(1 for r in rows if r["provenance"]=="ORIGINAL_TEACHER_CORPUS")
    n_app=sum(1 for r in rows if r["provenance"]=="POST_ORIGINAL_TRAINING_DATA")
    # production (streamed, strided-tagged)
    for mode,dom in PROD.items():
        f=f"{RES}/sio2x_production/{mode}/traj.dump"
        if not Path(f).exists(): continue
        for j,a in enumerate(iread(f,format="lammps-dump-text",index=":")):
            if set(a.get_atomic_numbers())<={1,2}:
                a.numbers=np.array([{1:8,2:14}[t] for t in a.get_atomic_numbers()])
            nSi,nO,osi,x=comp(a); H=h(a)
            rows.append({"structure_id":f"prod:{mode}:{j:04d}","structure_hash":H,"provenance":"PRODUCTION_MD",
                "source_file":f"sio2x_production/{mode}/traj.dump","source_group":f"prod:{mode}","frame_index":j,
                "raw_source_family":f"prod_{mode}","scientific_domain":dom,"natoms":len(a),"N_Si":nSi,"N_O":nO,
                "O_over_Si":osi,"oxygen_deficiency_x":x,"DFT_label":False,"dup_of":"",
                "generator_model":PROD_GEN,"state_meta":f"production MD {mode}","keep_stride":(j%PROD_STRIDE==0)})
    n_prod=sum(1 for r in rows if r["provenance"]=="PRODUCTION_MD")

    # ---- disposition every candidate ----
    for r in rows:
        H=r["structure_hash"]; dom=r["scientific_domain"]; prov=r["provenance"]
        r["overlap_PC004"]= H in scan_hashes
        if r["overlap_PC004"]:
            r["disposition"]="PC004_DFT_REFERENCE"; r["student_eligible"]=False; continue
        if r.get("dup_of"):
            r["disposition"]="REDUNDANT"; r["student_eligible"]=False; continue
        if dom in ("OOD_OR_EXCLUDED","UNRESOLVED"):
            r["disposition"]=dom.replace("OOD_OR_EXCLUDED","OOD_EXCLUDED"); r["student_eligible"]=False; continue
        if prov=="POST_ORIGINAL_TRAINING_DATA":
            # protect independent DFT: appended DFT-labeled oxide frames -> reserve as PC004 reference
            if r["DFT_label"]:
                r["disposition"]="PC004_DFT_REFERENCE_CANDIDATE"; r["student_eligible"]=False; continue
            r["disposition"]="STUDENT_DISTILLATION_CANDIDATE"; r["student_eligible"]=True; continue
        r["disposition"]="STUDENT_DISTILLATION_CANDIDATE"; r["student_eligible"]=True

    # ---- SELECT (central uncapped; broad capped per family; production strided) ----
    corp_by_fam={}
    for r in rows:
        if r["provenance"] in ("ORIGINAL_TEACHER_CORPUS","POST_ORIGINAL_TRAINING_DATA") and r["student_eligible"]:
            corp_by_fam.setdefault(r["raw_source_family"],[]).append(r)
    selected=[]; cap_report={}
    for r in rows:
        if not r.get("student_eligible"): continue
        if r["provenance"]=="PRODUCTION_MD":
            if r.get("keep_stride"): selected.append(r)
            continue
        dom=r["scientific_domain"]; fam=r["raw_source_family"]
        if dom in CENTRAL:
            selected.append(r)
        else:
            grp=sorted(corp_by_fam[fam],key=lambda z:z["frame_index"]); n=len(grp); cap=BROAD_CAP.get(fam,200)
            if n<=cap: selected.append(r); cap_report[fam]=n
            else:
                keep={grp[round(t*(n-1)/(cap-1))]["frame_index"] for t in range(cap)}
                if r["frame_index"] in keep: selected.append(r)
                cap_report[fam]=cap
    sel_ids={r["structure_id"] for r in selected}

    # ---- corrected leakage-safe split ----
    train=[]; val=[]; buffer_dropped=0
    groups={}
    for r in selected: groups.setdefault(r["source_group"],[]).append(r)
    for g,items in groups.items():
        if g.startswith("prod:"):
            items=sorted(items,key=lambda z:z["frame_index"])
            nv=max(1,int(round(len(items)*0.15)))
            vg=items[-nv:]; buf=items[-(nv+1):-nv]; tr=items[:-(nv+1)] if len(items)>nv+1 else items[:-nv]
            buffer_dropped+=len(buf); val+=vg; train+=tr
        else:
            items=sorted(items,key=lambda z:z["structure_hash"])   # decorrelated per-family split
            nv=max(1,int(round(len(items)*0.1))); val+=items[-nv:]; train+=items[:-nv]

    # ---- exposure: frame + atom weighted, per domain and per source ----
    def expo(data):
        d={}
        for r in data:
            k=r["scientific_domain"]; d.setdefault(k,[0,0]); d[k][0]+=1; d[k][1]+=r["natoms"]
        tf=sum(v[0] for v in d.values()); ta=sum(v[1] for v in d.values())
        return {k:{"N_frames":v[0],"N_atoms":v[1],"force_components":3*v[1],
                   "frac_frames":round(v[0]/tf,4),"frac_atoms":round(v[1]/ta,4)} for k,v in sorted(d.items())}, tf, ta
    sel_expo,tf,ta=expo(selected)
    src_atoms={"corpus":sum(r["natoms"] for r in selected if r["provenance"]!="PRODUCTION_MD"),
               "production":sum(r["natoms"] for r in selected if r["provenance"]=="PRODUCTION_MD")}
    src_frames={"corpus":sum(1 for r in selected if r["provenance"]!="PRODUCTION_MD"),
                "production":sum(1 for r in selected if r["provenance"]=="PRODUCTION_MD")}

    # source-group counts per split/domain
    def groups_per(data):
        d={}
        for r in data: d.setdefault(r["scientific_domain"],set()).add(r["source_group"])
        return {k:len(v) for k,v in sorted(d.items())}
    def dc(data):
        d={}
        for r in data: d[r["scientific_domain"]]=d.get(r["scientific_domain"],0)+1
        return dict(sorted(d.items()))

    # appended disposition summary
    app=[r for r in rows if r["provenance"]=="POST_ORIGINAL_TRAINING_DATA"]
    app_dom={}; app_disp={}
    for r in app:
        app_dom[r["scientific_domain"]]=app_dom.get(r["scientific_domain"],0)+1
        app_disp[r["disposition"]]=app_disp.get(r["disposition"],0)+1

    # PC004 references (11 SCAN + any appended DFT-ref candidates)
    dft_ref_candidates=[r for r in rows if r["disposition"] in ("PC004_DFT_REFERENCE","PC004_DFT_REFERENCE_CANDIDATE")]

    # ---- write manifests ----
    keys=["structure_id","structure_hash","provenance","source_file","source_group","frame_index",
          "raw_source_family","scientific_domain","natoms","N_Si","N_O","O_over_Si","oxygen_deficiency_x",
          "DFT_label","overlap_PC004","disposition","student_eligible","generator_model"]
    def w(name,data):
        with open(OUT/name,"w",newline="") as fh:
            wr=csv.DictWriter(fh,fieldnames=keys,extrasaction="ignore"); wr.writeheader(); wr.writerows(data)
    w("pc002_candidate_frame_inventory.csv",rows)
    w("pc002_selected_structure_manifest.csv",selected)
    w("pc002_student_train_structure_manifest.csv",train)
    w("pc002_student_valid_structure_manifest.csv",val)
    w("pc002_appended_2474_inventory.csv",app)
    w("pc004_dft_reference_candidates.csv",dft_ref_candidates)

    summary={
      "candidate_total":len(rows),"by_provenance":{"ORIGINAL_TEACHER_CORPUS":n_orig,
        "POST_ORIGINAL_TRAINING_DATA":n_app,"PRODUCTION_MD":n_prod},
      "appended_2474":{"n":len(app),"by_domain":dict(sorted(app_dom.items(),key=lambda kv:-kv[1])),
        "by_disposition":app_disp,"all_DFT_labeled":all(r["DFT_label"] for r in app)},
      "selected_total":len(selected),"selected_by_domain":dc(selected),
      "selected_by_source_frames":src_frames,"selected_by_source_atoms":src_atoms,
      "exposure_by_domain":sel_expo,"total_frames":tf,"total_atoms":ta,"total_force_components":3*ta,
      "atom_weight_note":("SIMPLE-NN force loss (F_loss_type=1) is atom-weighted (mean over all force "
        "components); energy loss (E_loss_type=1) is per-atom then per-structure mean. So atom fractions "
        "≈ force-training exposure; frame fractions ≈ energy-training exposure."),
      "production_atom_share":round(src_atoms["production"]/ta,4),
      "train_val":{"train":len(train),"val":len(val),"buffer_dropped":buffer_dropped,
        "train_by_domain":dc(train),"val_by_domain":dc(val),
        "train_source_groups_per_domain":groups_per(train),"val_source_groups_per_domain":groups_per(val),
        "split_rule":"production: blocked temporal split with 1-frame BUFFER gap per trajectory (no adjacent straddle); corpus: per-family hash-ordered 90/10 (independent DFT structures, decorrelated)"},
      "pc004_dft_references":{"scan_cells_frozen":sum(1 for r in dft_ref_candidates if r["disposition"]=="PC004_DFT_REFERENCE"),
        "appended_reserved":sum(1 for r in dft_ref_candidates if r["disposition"]=="PC004_DFT_REFERENCE_CANDIDATE"),
        "total":len(dft_ref_candidates)},
      "labeling_cost":{"total_frames":len(selected),"total_atoms":ta,
        "corpus_frames":src_frames["corpus"],"corpus_atoms":src_atoms["corpus"],
        "production_frames":src_frames["production"],"production_atoms":src_atoms["production"],
        "note":"teacher (Allegro) inference cost ~ N_atoms; no labeling performed"},
      "crystalline_trim":{"before_AMBIENT_CRYSTAL":1366,"caps_applied":BROAD_CAP,
        "rationale":"high-pressure crystalline (bulk_cryst_hp/highpressure_*) is off-deployment (ambient amorphous/SiOx) and internally redundant; trimmed hard, ambient bulk_cryst kept at 300"},
    }
    (OUT/"pc002_domain_counts.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("candidate_total","by_provenance","appended_2474","selected_total",
        "selected_by_domain","selected_by_source_frames","selected_by_source_atoms","production_atom_share",
        "train_val","pc004_dft_references","labeling_cost")},indent=2,default=str))

if __name__=="__main__": sys.exit(main())
