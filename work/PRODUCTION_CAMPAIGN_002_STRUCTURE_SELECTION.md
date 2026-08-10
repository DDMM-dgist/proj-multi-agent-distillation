# PC002 — Distillation Dataset STRUCTURAL Design (frame-level, teacher-independent)

**State: `PC002_SOURCE_STRUCTURE_INVENTORY = COMPLETE` · `PC002_DATASET_STRUCTURE_RESOLVED = TRUE`
· `PC002_TEACHER_LABELING = WAIT_FINAL_TEACHER`.**

> Correction of a prior premature closure: the earlier 20-structure seed inventory
> (`pc002_structure_manifest.csv`) is the **source inventory**, NOT the resolved dataset. PC002 is
> resolved only now, with an actual **frame-level** Student ensemble frozen below. Teacher labels
> are NOT generated (the final teacher may change after the external GPU remediation).

Artifacts: `pc002_candidate_frame_inventory.csv` · `pc002_domain_mapping.json` ·
`pc002_selected_structure_manifest.csv` · `pc002_student_train_structure_manifest.csv` ·
`pc002_student_valid_structure_manifest.csv` · `pc002_domain_counts.json` ·
`pc002_redundancy_report.json` · `pc004_dft_exclusion_manifest.csv` ·
`pc002_structure_design_decision.json` · builder `pc002_frame_level_design.py`.

---

## 1. Frame-level candidate pools (LOCAL, teacher-independent) — 12,481 frames
| pool | source | frames | notes |
|---|---|---|---|
| A. original teacher corpus | `dataset.xyz` frames **[0,11423]** | 11,424 | all domains, `config_type` metadata; post-original [11424,13897] (2,474) **excluded** |
| B. production deployment MD | `sio2x_production/{pristine,random,sphere,plane}_x00{6,12}/traj.dump` | 1,057 | 7 trajectories × 151 frames, 2,880–3,000 atoms; the exact deployment defect domains |

Every candidate frame carries: `structure_id, structure_hash, source_file, source_trajectory,
frame_index, raw_source_family, scientific_domain, natoms, N_Si, N_O, O_over_Si,
oxygen_deficiency_x, DFT_reference_status, student_training_eligible, preserve_for_PC004, notes`
(ASE-read; no grep line-number frame identity).

## 2. Explicit domain mapping (`pc002_domain_mapping.json`)
Taxonomy: STOICHIOMETRIC_AMORPHOUS · DILUTE_OXYGEN_DEFICIENT · CLUSTERED_VOID · SURFACE ·
HIGH_T_LIQUID · AMBIENT_CRYSTAL · OTHER_RELEVANT_SIO · OOD_OR_EXCLUDED. Assigned by explicit
`config_family → domain` map, with a **composition override**: any frame with `N_O == 0` → OOD
(elemental Si), regardless of name. Oxide frames whose family is unmapped → UNRESOLVED (not
selected). Production frames mapped by source (pristine→amorphous, random→dilute, sphere/plane→clustered).

Candidate distribution: AMBIENT_CRYSTAL 3593 · OOD 2670 · DILUTE 2207 · CLUSTERED 2045 ·
AMORPHOUS 775 · SURFACE 770 · HIGH_T_LIQUID 313 · OTHER_SIO 108.

## 3. Selected Student ensemble — **6,436 frames** (`pc002_selected_structure_manifest.csv`)
| domain | selected | | domain | selected |
|---|---:|---|---|---:|
| DILUTE_OXYGEN_DEFICIENT | 1,937 | | AMBIENT_CRYSTAL | 1,366 |
| CLUSTERED_VOID | 1,505 | | SURFACE | 567 |
| STOICHIOMETRIC_AMORPHOUS | 640 | | HIGH_T_LIQUID | 313 |
| | | | OTHER_RELEVANT_SIO | 108 |

- **Oxygen-deficient (dilute + clustered) = 3,442 = 53 %**; central (incl. amorphous) = 4,082 = 63 %;
  broad replay = 2,354 = 37 %. Source: 6,324 corpus + 112 strided production.
- **This deliberately corrects the historical imbalance** (old student pool ≈ 7,330 base + 17 defect
  = 0.2 % defect). Counts are evidence-based (available pool sizes, oxygen-deficiency coverage,
  redundancy, historical under-representation, SiOx deployment target, PC004 leakage) — not arbitrary.
- Elemental-Si (2,670 `N_O==0` frames) excluded as OOD for a SiOx student. Amorphous (640) is the
  smallest central domain purely by pool availability; augmentable later from pristine production +
  the 12,288-atom glass if the trained student shows amorphous weakness.

## 4. Redundancy control (`pc002_redundancy_report.json`)
- Production MD (consecutive frames, high redundancy): **temporal stride 10** → 112 kept of 1,057.
- Broad corpus families: **per-family cap 400** by deterministic even-spacing over sorted
  frame_index (central defect/amorphous families kept uncapped — they are the target). No SOAP/PCA
  study. Deterministic; drops logged.

## 5. Train / validation split — 5,789 / 647 (`..._train/_valid_structure_manifest.csv`)
Source/trajectory-aware: group by `source_trajectory`, assign the **contiguous last 10 %** of each
group to val so adjacent near-identical frames never straddle the split. Deterministic; **train∩val
= 0** (verified). Per-domain val coverage present for all 7 domains.

## 6. PC004 DFT exclusions (`pc004_dft_exclusion_manifest.csv`) — hard, verified
The **11 SCAN DFT cells** (`sio2x_AL_labels_11cells.xyz`; 4 dilute x=0 + 7 clustered x 0.114–0.60)
are frozen out of all Student training/validation and reserved for Student-vs-DFT (PC004). Checked
deterministically by structure hash: **selected ∩ exclusion = 0**.

## 7. Teacher-independence & what waits for the join
Only structure identity/hash/domain/role/provenance/exclusion are frozen. **No teacher labeling**
(no `b56e20ff` mass-labeling). At the join point the FINAL teacher labels this exact frozen
ensemble (energy+forces), then PC003 trains. The full historical ~10k augmented pool is not local;
this locally-grounded frame-level ensemble is the resolved design and does not depend on it.

## 8. Guardrails honored
No teacher labels generated; DFT refs isolated & verified; production redundancy controlled; no new
MD/DFT; elemental-Si OOD documented not silently dropped; VASP untouched; architecture frozen.
