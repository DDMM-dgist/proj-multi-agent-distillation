# PC002 — Distillation Dataset STRUCTURAL Design (teacher-independent)

**State: `PC002_STRUCTURE_SELECTION = RESOLVED` · `PC002_FINAL_TEACHER_LABELING = BLOCKED_ON_FINAL_TEACHER`.**

PC002 resolves the *structural* dataset — which structures, which domains, what coverage, what
train/val roles, what stays out — **independent of the final teacher's weights**. No teacher
labels are read or generated here (that is `PC002_FINAL_TEACHER_LABELING`, deferred to the join
point so we never mass-relabel with a teacher that may be replaced).

Artifacts: `work/pc002_structure_manifest.csv`, `work/pc002_structure_inventory.json`,
`work/pc002_structure_inventory.py`, `work/production_campaign_002_structure_state.json`.

---

## 1. Governing design lesson (grounded, not assumed)
`gpu_finetune_handoff/distillation/DISTILLATION_DIAGNOSIS_2026-07-03.md` established that the
historical **student** distillation pool **under-sampled the SiO₂-x / oxygen-deficient regime**;
the v6 re-distillation pool (`v6_data_manifest.csv`) was **7,347 frames = 7,330 `base_pool`
(182-atom cells) + only 17 defect anchors** — defects were ≈0.2 % of the pool. The obsolete
"KISTI-only distillation dataset" framing is dropped: the historical distillation ran on gpu2 and
the relevant building blocks exist locally (below).

**PC002 design decision:** the student structural pool MUST carry explicit, substantial coverage
of the three central domains (amorphous, dilute vacancy, clustered/void) plus the broad-support
domains, so the student is not structurally blind to the exact regime the teacher is weakest in.
Coverage is a **structural** requirement, decided now; label *quality* depends on the final
teacher and is handled at the join.

## 2. Coverage design (structural targets, teacher-independent)
| domain | structural requirement | local building blocks (frozen) |
|---|---|---|
| amorphous SiO₂ | bulk glass, melt-quench snapshots across T | `prod:pristine`; `sio2x_production/*/traj.dump`; 12,288-atom glass (`production_12288/`) |
| dilute O-vacancy | distributed vacancies, x≈0.06–0.24 | `prod:random_x006`, `prod:random_x012`; production random sweep |
| clustered / void | sphere + plane void modes, x≈0.06–0.12 | `prod:sphere_x006/x012`, `prod:plane_x006/x012` |
| surfaces | SiO₂ surfaces | teacher-corpus `surfaces*` families (structural reference) |
| liquid / high-T | melt configurations | `melt_msd` 4000 K trajectory; `liquid`/`liq` families |
| crystalline | ambient + high-pressure SiO₂ | `bulk_cryst`, `bulk_cryst_hp` families |
| composition range | x = 0 → ~0.6 | SCAN cells `local_x` 0–0.60; production sweep 0.03–0.24 |

## 3. Frozen structure manifest (local, teacher-independent) — 20 entries
Schema (per B2 — identity only, NOT labels): `structure_id, path, sha16, domain, n_frames,
natoms, nSi, nO, x_SiO2minus, source, intended_role, dft_exclusion`.

- **7 production train-candidates** (`intended_role=student_train_candidate`): pristine (amorphous)
  + random/sphere/plane × x006/x012 (dilute + clustered/void). Single relaxed cells now; the
  full trajectory frames are the sampling reservoir at labeling time.
- **11 SCAN DFT cells** (`intended_role=HELDOUT_DFT_reference_PC004`, `dft_exclusion=YES_never_train`):
  the only committed project DFT ground truth — 4 dilute (x=0) + 7 clustered (x 0.114–0.60). **These
  are frozen OUT of all student training** and reserved for PC004 Student-vs-DFT.
- **2 augment seed pools** (`input_small.xyz` 50, `input_large.xyz` 10): the augment-atoms seeds.

## 4. Train / validation structural design
- **Student train/val split is by STRUCTURE, stratified by domain** (amorphous/dilute/clustered/
  broad), deterministic seed, so no trajectory leaks between train and val (frames from one MD
  trajectory stay on one side). Ratio 0.9/0.1 (matching historical SIMPLE-NN practice).
- **The 11 SCAN DFT cells are excluded from BOTH** train and val (they are PC004 test only).
- Domain balance target enforced at selection time (not left to trajectory frame counts), to
  correct the historical amorphous-dominated pool.

## 5. What is RESOLVED now vs deferred
**Resolved (teacher-independent):** the coverage design (§2), the frozen local structure manifest
+ schema (§3), the train/val structural strategy (§4), and the DFT-exclusion set (§3). This is
`PC002_STRUCTURE_SELECTION = RESOLVED`.

**Deferred to the join point** (`PC002_FINAL_TEACHER_LABELING`):
- The full ~10,000-frame augmented pool is NOT stored locally (generated + consumed on gpu2/KISTI;
  `data_provenance/PROVENANCE.md` §6 lists this as an open gap). Its regeneration (augment-atoms
  rattle→relax on the seeds) and **all teacher labeling** wait for the FINAL teacher — we do **not**
  generate provisional base-teacher labels that would be thrown away if the external GPU teacher
  improves.
- Final per-frame label manifest (energy/forces keys, units) is produced only after teacher identity
  is fixed.

## 6. Guardrails honored
No teacher labeling performed; no mass label generation; DFT-only references frozen out of training;
obsolete "KISTI-only" framing corrected; no VASP touched; architecture frozen.
