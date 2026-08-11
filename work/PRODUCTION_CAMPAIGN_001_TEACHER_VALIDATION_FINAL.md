# PC001 — Teacher Validation (FINAL / AUTHORITATIVE)

Corrected, complete teacher-acceptance gate. **No student metrics** enter this gate; **no acceptance
threshold is invented**. Read-only over teacher-vs-DFT evidence — **no teacher/student inference, DFT, MD,
training, scheduler, network, or semantic Judge.** Supersedes the preliminary
`pc001-teacher-validation` (`ACCEPT_CONDITIONAL`, provisional), which is preserved immutable. Evidence:
`work/production_campaign_001_teacher_validation_final.py` →
`runs/production_campaign_001/pc001-teacher-validation-final/`. Labels **FACT / DERIVED / INFERENCE /
UNRESOLVED**. `RES = research-sio2-allegro-simplenn-distillation/`.

## 1. Target scientific domain (declared BEFORE examining performance)

**Target material:** amorphous SiO₂ + oxygen-deficient SiO₂₋ₓ. **In-scope domains:** amorphous SiO₂ (melt/
quench/liquid); SiO₂₋ₓ **dilute** vacancy; SiO₂₋ₓ **clustered** vacancy / void surface / under-coordinated
Si; surfaces; ambient crystalline SiO₂ (reference). **Deployment:** production MD + defect physical
validation at ρ≈2.2, ambient–moderate T. **Out-of-scope / diagnostic:** elemental Si, high-pressure dense
polymorphs (stishovite/pyrite/PbO₂/bulk_cryst_hp), isolated small clusters. (DERIVED — `PROJECT.md`,
`DISTILLATION_RECIPE.md`.)

## 2. DFT reference inventory (FACT)

Teacher-vs-DFT test set (`error_a`): **1155 frames**, DFT = **SCAN** (2-stage PBE→SCAN), units eV / eV/Å,
24 config families. Plus 39 SCAN AL cells (11 original + 28 al_iter3), gate-passed convergence. In-scope
frames = **727**; out-of-scope = 428.

## 3. Reference-quality audit (Axis A → PASS)

All frames finite E/F. Only two Fmae>2 eV/Å: **idx 277** (natoms=2 isolated cluster, Fmae 56.65, dE
−3424 meV/atom) and **idx 244** (natoms=5 cluster, 2.48). Both **out-of-scope tiny isolated clusters** →
classified `DATA_ARTIFACT_or_DEGENERATE` / OOD; excluded from in-scope stats **by domain**, not
cherry-picking. (Corrects the earlier loose "cc001 atom-overlap" label — the 56.65 outlier is a 2-atom
cluster, idx 277, not the 46-atom clustered_cell_001.)

## 4. Teacher candidate inventory (Axis B → PASS)

Started `CURRENT_TEACHER = UNRESOLVED`. Candidates: **base Allegro `b56e20ff`** (deployed; broad DFT
evaluation via error_a) and **fine-tuned v2 `b3be4d2a…`** (not deployed; `INSUFFICIENT_COMMON_EVIDENCE` —
no common DFT eval vs base). **Evaluated teacher = base Allegro `b56e20ff`** — the only one with broad DFT
evaluation; candidates not ranked on incompatible sets.

## 5. Teacher identity (FACT)

`teacher/model.nequip.pth` == deployed, SHA **`b56e20ff…1b1c57`**; Allegro/NequIP compiled TorchScript,
cutoff 5.0 Å, symbols [O,Si]. Training data = KISTI DFT(SCAN) corpus (frame-level hashes KISTI-origin —
identity solid, provenance nonblocking).

## 6. Prediction provenance (FACT)

`error_a` teacher forces via `nequip.ase.NequIPCalculator(compiled).get_forces()`; energy via
`.get_potential_energy()` — the **valid** energy path (not the raw-torch `total_energy` path that caused
the C3 mismatch). `Fmae_eV_A` = per-frame **force COMPONENT MAE** = `|F_teacher−F_dft|.mean()` over
atoms×3. `dE_per_atom_meV` = (E_teacher−E_dft)/N×1000; `shifted` removes the global mean bias.

## 7. Force fidelity (Axis C → PASS)  — force COMPONENT MAE, eV/Å

| scope | n | mean | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| global (all) | 1155 | 0.190 | 0.109 | 0.312 | 0.365 | 0.462 | 56.65 |
| **in-scope** | 727 | **0.152** | 0.142 | 0.319 | 0.372 | 0.434 | 0.890 |
| out-of-scope | 428 | 0.254 | 0.066 | 0.292 | 0.351 | 0.645 | 56.65 |

Covered target domain is force-accurate; the global max (56.65) is the out-of-scope 2-atom cluster.
Species-resolved (Si/O) = **UNRESOLVED** from the CSV (needs per-atom re-parse; no compute here).

## 8. Energy fidelity (Axis D → PASS)

In-scope shift-corrected E-MAE **18.8 meV/atom** (median 18.7, p95 34.4, max 119). Global mean bias
(the per-type-shift convention offset) ≈ −16 meV/atom. **Convention closed** for the NequIPCalculator/
error_a path (matches DFT to ~16 meV/atom). **Caveat:** this validity is path-specific; the raw
deployed-torch `total_energy` convention (C3) is separate and must not be assumed equal for future
direct-torch labeling.

## 9. Domain-resolved fidelity (force component MAE / energy shift, per target domain)

| target domain | n | force mean | force p95 | force max | energy MAE | coverage |
|---|---|---|---|---|---|---|
| amorphous_SiO₂ | 152 | 0.186 | 0.278 | 0.364 | 15.8 | COVERED |
| SiO₂ₓ dilute vacancy | 132 | 0.231 | 0.389 | 0.524 | 19.4 | COVERED |
| **SiO₂ₓ clustered vac / void surface** | 79 | **0.321** | 0.411 | 0.457 | 16.4 | COVERED |
| surfaces | 68 | 0.216 | 0.426 | 0.890 | 18.6 | COVERED |
| ambient crystalline SiO₂ | 296 | 0.040 | 0.099 | 0.247 | 20.7 | COVERED |

## 10. SiOx / vacancy-specific analysis

The elevated force error concentrates on **`*_max_AL`** (active-learning-selected **maximally-uncertain**)
clustered frames (~0.33–0.35 eV/Å component MAE) vs dilute/`int_AL` (~0.19–0.34). These are the hardest,
deliberately-selected configs — the teacher's genuine ceiling on the hardest in-domain motifs, **not** a
uniform collapse. It **is** target-domain (clustered vacancy / void surface is in scope), but the domain
is **COVERED** (79 DFT frames), so this is a **hard-physics + AL-selection** limitation, not sparse
coverage. **Blocking now? NO** — no source-grounded threshold is violated, the teacher stays physically
consistent, and it is the sole DFT-trained reference; remediation = ensure distillation coverage (PC002) +
monitor in student/physical validation and later AL. **Not silently deferred** — recorded as an explicit
PC002 requirement + coverage/AL watch item.

## 11. Outlier forensic analysis (Axis F → PASS)

| structure | domain | Teacher Fmae | DFT dE/atom | characteristic | classification | in in-scope stat? |
|---|---|---|---|---|---|---|
| idx 277 | cluster (2 atoms) | 56.65 | −3424 meV | isolated dimer, degenerate | DATA_ARTIFACT_or_DEGENERATE | no (out-of-scope) |
| idx 244 | cluster (5 atoms) | 2.48 | +426 meV | tiny isolated cluster | VALID_OOD_STRESS_TEST | no (out-of-scope) |

No in-domain structure is mislabeled an artifact; both with/without-outlier in-scope stats are reported
(out-of-scope excluded by domain).

## 12. Target-domain coverage matrix (Axis E → PASS)

All five in-scope target domains have **≥68 DFT reference frames** (79–296) → **COVERED**; none SPARSE or
MISSING. (Coverage is descriptive counting; the clustered domain is covered but hard — see §10.)

## 13. Energy-reference audit (Axis D detail)

`per_type_energy_shifts = per_atom_energy_mean` (baked, non-trainable). error_a used NequIPCalculator ⇒
absolute energy on the DFT reference (bias ≈ −16 meV/atom = the offset). **Axis D = PASS** for this path;
the raw deployed-torch path (C3) remains an unclosed, separate convention.

## 14. Teacher-specific physical evidence (Axis G → PASS)

Teacher EOS (`eos_teacher_bm_summary.csv`, teacher `b56e20ff`): ambient SiO₂ phases **SMOOTH**, physical
B0 (α-quartz ≈202, coesite ≈228 GPa). **No student physical validation used as teacher evidence.**

## 15. Acceptance-criterion provenance

**No source-grounded teacher-vs-DFT threshold exists.** The only documented threshold is a **student**
distillation-gap target (Student F-MAE vs teacher ≤0.175, `DISTILLATION_RECIPE.md:153`) — excluded from
the teacher gate. Therefore **no 0.20 eV/Å / 50 meV/atom (or any) hard cutoff is invented**; the verdict
is derived from the axes + quantitative evidence, with the ACCEPT decision flagged as scientific
interpretation.

## 16. Axes A–G (each independent; not averaged)

| axis | verdict | basis |
|---|---|---|
| A DFT_REFERENCE_VALIDITY | **PASS** | finite E/F; only out-of-scope tiny-cluster outliers, classified |
| B TEACHER_IDENTITY_AND_PROVENANCE | **PASS** | SHA `b56e20ff`, cutoff 5.0, [O,Si]; training provenance KISTI (nonblocking) |
| C TEACHER_FORCE_FIDELITY | **PASS** | in-scope component MAE 0.152; covered domain accurate; clustered sub-region flagged |
| D TEACHER_ENERGY_FIDELITY | **PASS** | shift-corrected 18.8 meV/atom; NequIPCalculator convention closed (path-specific caveat) |
| E TARGET_DOMAIN_COVERAGE | **PASS** | all 5 target domains ≥68 DFT frames (COVERED) |
| F OUTLIER_AND_FAILURE_MODE | **PASS** | extreme errors are out-of-scope tiny clusters; no in-domain mislabel |
| G TEACHER_PHYSICAL_CONSISTENCY | **PASS** | teacher EOS SMOOTH, physical ambient B0 |

## 17. Final Teacher verdict

**`TEACHER_ACCEPTED_FOR_DISTILLATION`** (DERIVED — scientific interpretation over deterministic axis
evidence; no invented threshold). The base Allegro teacher `b56e20ff` is a valid, physically-consistent,
DFT-trained supervisor: in-scope force component MAE ~0.15 eV/Å, energy ~19 meV/atom (valid convention),
EOS smooth; the only extreme errors are out-of-scope tiny clusters; no teacher-vs-DFT threshold to fail.
The elevated clustered-defect region is a hard-physics caveat handed to PC002 + coverage/AL, not a
teacher-model defect.

## 18. Blocking issues

**NONE.** No axis FAIL; no source-grounded threshold violated.

## 19. Nonblocking issues

1. Clustered-vacancy / void-surface teacher force elevated (~0.32–0.35 eV/Å component MAE, AL-hardest
   frames) → **PC002 must ensure distillation-set coverage there**; monitor in student + physical
   validation + coverage/AL.
2. Absolute-energy validity is **NequIPCalculator-path-specific**; raw deployed-torch `total_energy` (C3)
   is a separate, unclosed convention.
3. Full teacher training-set frame hashes are KISTI-origin (identity solid; frame provenance not local).
4. Species-resolved (Si/O) force breakdown UNRESOLVED from the CSV.

## 20. Next scientific campaign (dictated by the verdict)

Teacher ACCEPTED ⇒ **PRODUCTION CAMPAIGN 002 = DISTILLATION DATASET DESIGN** (stub:
`work/PRODUCTION_CAMPAIGN_002_STUB.md`). `DISTILLATION_DATASET_STAGE_AUTHORIZED = true`;
`STUDENT_STAGE_AUTHORIZED = false`; **`NEW_PIPELINE_CURRENT_STUDENT = NONE`**;
`EXISTING_HISTORICAL_STUDENT_ASSETS = [original, v5]` (benchmarks only). **The original-vs-v5 comparison is
NOT the next authoritative action** (it is a later student-validation benchmark); its files remain as
exploratory history.
