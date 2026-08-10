# PC002 — Distillation Dataset STRUCTURAL Design (frame-level) — CORRECTION PASS

**State: `PC002_SOURCE_STRUCTURE_INVENTORY = COMPLETE` · `PC002_DATASET_STRUCTURE_RESOLVED = TRUE`
(corrected ensemble) · `PC002_TEACHER_LABELING = WAIT_FINAL_TEACHER`.**

Artifacts: `pc002_candidate_frame_inventory.csv` · `pc002_domain_counts.json` ·
`pc002_selected_structure_manifest.csv` · `pc002_student_train/valid_structure_manifest.csv` ·
`pc002_appended_2474_inventory.csv` (empty — see §0) · `pc004_dft_exclusion_manifest.csv` ·
`pc004_dft_reference_candidates.csv` · `pc002_structure_design_decision.json` ·
`dataset_provenance_split_fact.json` · builder `pc002_correction_pass.py`.

## 0. HEADLINE CORRECTION — dataset.xyz has 11,424 frames, NOT 13,898
The prior "13,898 frames = 11,424 original + 2,474 appended" is **retracted as false**. Authoritative
counts: ASE frame count = **11,424**; `grep -c 'Lattice='` (one per frame) = **11,424**. The "13,898"
was a `grep -o config_type=` **token** artifact — **2,474 frame-header lines carry `config_type=`
twice** (11,424 + 2,474 = 13,898); **0** occurrences on non-header lines. **There are no appended
frames** — dataset.xyz is the original teacher corpus in full. So the appended-frame
inventory/disposition (§2 of the request) resolves to **0 frames exist**; nothing to add. The
11,424-frame Track-A constraint is unaffected (it is the whole corpus).

## 1. Candidate space (frame-level, teacher-independent) — 12,481
| pool | provenance | frames |
|---|---|---|
| `dataset.xyz` [0,11423] | ORIGINAL_TEACHER_CORPUS | 11,424 |
| `sio2x_production/*/traj.dump` (7 × 151) | PRODUCTION_MD | 1,057 |
| appended | POST_ORIGINAL_TRAINING_DATA | **0** (does not exist) |

Production frames are **structures only**; their stored `fx/fy/fz` come from historical Student/MLIP
MD (`pair_style nn`) and are **NOT authoritative labels** — `generator_model` + source trajectory
recorded as provenance. Final energy/force labels come from the FINAL teacher at the join point.

## 2. Selected ensemble — **5,552 frames** (was 6,436; corrected)
| domain | selected | | domain | selected |
|---|---:|---|---|---:|
| DILUTE_OXYGEN_DEFICIENT | 1,937 | | AMBIENT_CRYSTAL | **600** (was 1,366) |
| CLUSTERED_VOID | 1,504 | | SURFACE | **450** (was 567) |
| STOICHIOMETRIC_AMORPHOUS | 640 | | HIGH_T_LIQUID | 313 |
| | | | OTHER_RELEVANT_SIO | 108 |

**Correction — high-pressure crystalline trimmed** (evidence-based diversity, not size): AMBIENT_CRYSTAL
was 966/1366 high-pressure (`bulk_cryst_hp` + `highpressure_int_AL` + `highpressure_max_AL`), which is
**off-deployment** (deployment is ambient amorphous/SiOx) and internally redundant. Caps: ambient
`bulk_cryst` 300; each high-pressure family 100. Result: more compact, deployment-focused ensemble.
Oxygen-deficient (dilute+clustered) = **62%** of frames.

## 3. Frame-weighted vs atom-weighted exposure (decisive — SIMPLE-NN loss verified)
From `simple_nn/models/loss.py` (v2.0.0): **energy loss (`E_loss_type=1`)** = per-atom energy,
**per-structure averaged**; **force loss (`F_loss_type=1`)** = `mean` over **all force components** →
**atom-weighted**. So **frame fractions ≈ energy exposure; atom fractions ≈ force exposure.**

Total atoms **1,020,697** (force components 3,062,091). **Production = 32.1% of atoms but only 2.0% of
frames** → the 112 production frames (7 lineages, ~2,900 atoms) dominate force training.
**PC003 recommendation:** use SIMPLE-NN `struct_weight` to offset this if the 7 production lineages
over-weight force training. (Full per-domain frame/atom/force-component table in `pc002_domain_counts.json`.)

## 4. Leakage-safe split — 4,987 train / 558 val (corrected)
- **Production (true MD):** blocked temporal split with a **1-frame buffer gap** per trajectory (7
  buffer frames dropped) — no adjacent frame straddles train/val.
- **Corpus (independent DFT structures, not MD):** per-family **hash-ordered** 90/10 (decorrelated;
  near-duplicate-guarded via exact-hash dedup).
- Verified: **train∩val = 0**, **shared hashes = 0**.
- **Source-group independence caveat (§7, reported honestly):** blocked splits mean train and val
  **share lineages** — val is held-out frames from the *same* source groups, not independent lineages.
  Single-lineage domains (HIGH_T_LIQUID, OTHER_RELEVANT_SIO = 1 group; amorphous production = 1
  trajectory) measure interpolation-within-lineage, not cross-lineage generalization. Whole-trajectory
  holdout isn't possible (too few independent lineages/domain); blocked+buffer is the accepted fallback.

## 5. Independent DFT protection (PC004)
The **11 SCAN DFT cells** remain the frozen PC004 reference set (`pc004_dft_exclusion_manifest.csv`),
never in Student train/val (hash-verified). Appended DFT-labeled SiOx frames would have been *reserved*
as additional independent references — but **0 exist** (no appended data). `pc004_dft_reference_candidates.csv`
records the disposition logic.

## 6. Labeling cost (no labeling performed)
Total **5,552 frames / 1,020,697 atoms** — corpus 5,440 frames / 693,337 atoms; production 112 frames /
327,360 atoms. Allegro teacher inference cost ~ N_atoms. Ensemble is compact (not structurally elegant
but expensive).

## 7. Domain-balance justification
Dilute+clustered 62% directly targets the deployment oxygen-deficient regime (the teacher's weakest
domain); amorphous 640 is the matrix; broad replay (crystal 600 + surface 450 + liquid 313 + other 108)
retained for MD stability; high-pressure crystalline trimmed (off-deployment). Counts are driven by pool
availability + deployment relevance, not target percentages.

## 8. Guardrails
No teacher labeling; production treated as structures-only; DFT refs isolated & verified; redundancy
controlled; no new MD/DFT; VASP untouched; architecture frozen; PC001 not reopened; GPU branch untouched.
