# PARALLEL TEACHER + PYDANTIC WORKFLOW

The workflow is split **physically and logically** into two tracks that proceed in parallel. This
CPU server does Track B and *prepares* (never executes) Track A.

```
TRACK_A = EXTERNAL_GPU_TRAINING_PREPARED       (train later on a GPU-capable server)
TRACK_B = CPU_PYDANTIC_WORKFLOW_ACTIVE         (running now, this server)

join_condition = FINAL_TEACHER_IDENTITY_RESOLVED  AND  PC002_DATASET_STRUCTURE_RESOLVED
```

## Teacher status (closed input)
Existing-teacher screen complete → **`NO_EXISTING_TEACHER_IMPROVES_TARGET_HELDOUT_FIDELITY`**
(base / v2 / v6 all `DO_NOT_ADVANCE`; see `EXISTING_TEACHER_CANDIDATE_SCREEN.md`). No full-1142
validation for v2/v6. Root cause `A_MODEL_TRAINING_UNDERFIT` (held-out defect force floor) —
**not reopened**.

## Dataset provenance (CORRECTED — `work/dataset_provenance_split_fact.json`)
`dataset.xyz` (sha `382d0b2b…`) has **11,424** frames total — the full ORIGINAL_TEACHER_CORPUS
[0,11423]. The earlier **"13,898 / 2,474 appended" claim is RETRACTED as false**: it was a
`grep -o config_type=` token artifact (2,474 headers carry the token twice); ASE + `Lattice=` counts
both = 11,424; **no appended frames exist**. Track-A's 11,424 constraint is unaffected.

## TRACK A — EXTERNAL GPU (DATA PREPARED, NOT EXECUTED; recipe owned by GPU branch)
Deliverable: `work/EXTERNAL_GPU_TARGET_TEACHER_TRAINING_PACKAGE.md` + portable package
`work/external_gpu_teacher_package/` (104 MB; corpus 3,766 frames = 2,966 target core + 800 replay;
warm-start ckpt `51342b33`; scripts, manifests, READMEs, integrity SHA256SUMS).
- **The training YAML is `CPU_PREP_DRAFT_ONLY`** — the external GPU server owns the authoritative
  recipe (LR, loss weights, architecture, epoch cap, sampling, checkpoint policy). The authoritative
  CPU-side Track-A outputs are the **DATA artifacts only**: source dataset, split identities,
  target/replay frame identities, base checkpoint, provenance, SHA256s, portable manifests.
- A6 fallback (documented only): higher-capacity Allegro, decided only if the GPU experiment fails.

## TRACK B — CPU PydanticAI (ACTIVE, running now)
| campaign | state | deliverable |
|---|---|---|
| PC002 structural design | **`PC002_DATASET_STRUCTURE_RESOLVED = TRUE`** (frame-level, correction pass); `SOURCE_INVENTORY=COMPLETE`; labeling `WAIT_FINAL_TEACHER` | `PRODUCTION_CAMPAIGN_002_STRUCTURE_SELECTION.md`; **5,552-frame** ensemble (high-pressure crystalline trimmed), train 4,987 / val 558, candidate 12,481 (11,424 corpus + 1,057 prod, 0 appended), atom-weighted exposure (prod 32%), leakage-safe blocked+buffer split, `pc002_structure_design_decision.json` |
| PC003 student workflow | **`CONFIG_AND_POLICY_RESOLVED`**; `PC003_TRAINING = WAIT_FINAL_TEACHER_LABELS`; `NEW_PIPELINE_CURRENT_STUDENT = NONE` | weighting **C_SIZE_NORMALIZED_BOUNDED** (production force 32%→18.6%, defects 66%; SIMPLE-NN force loss verified atom-weighted, struct_weight per-tag training-only); `pc003_current_student_config.json` + loss/weighting/exposure/decision/batching/resource/label-schema/committee/gate/acceptance/interface JSONs; arch = historical reuse; committee 4 seeds (234/345/555/**777 VERIFIED**); u_max = INITIAL_REFERENCE (0.15 coded, 0.30 not a gate) |
| PC004 student validation | PREPARED (teacher-independent infra); DFT exclusions verified | `pc004_validation_protocol.json`, `pc004_validation_manifest.json`, `pc004_dft_exclusion_manifest.csv` (11), `PC004_VALIDATION_PROTOCOL_PREP.md` |
| PC005 physical validation | PREPARED (scripts + baselines; no new MD) | `pc005_physical_validation_protocol.json`, `pc005_existing_asset_inventory.json`, `PC005_PHYSICAL_VALIDATION_PROTOCOL_PREP.md` |

Track B does **not** wait for Track A. It stops only at the first genuinely teacher-dependent
expensive operation (mass labeling / final student training).

## JOIN POINT — when the externally trained teacher returns
1. Register candidate model + SHA256 (append-only provenance).
2. Screen on the exact **373-frame** target-domain held-out set (reuse base metrics; no rerun of base).
3. **If improved** (clear dilute/clustered gain, no amorphous degradation): run full **1,142-frame**
   held-out validation → select as FINAL teacher if valid.
4. **If not improved:** retain `b56e20ff` as the resource-constrained baseline and record
   `TEACHER_LIMITATION_ACCEPTED_FOR_RESOURCE_CONSTRAINED_INITIAL_PIPELINE`.
5. FINAL teacher → **PC002 final labeling** (label the resolved structural pool; regenerate the
   full augmented pool as needed).
6. → **PC003** student committee training (SIMPLE-NN 4-seed).
7. → **PC004** validation (Student vs Teacher, Student vs DFT; domain-resolved; gated).
8. → **PC005** physical validation (re-run scripts against the new student).

## CPU-SERVER HARD RULES (enforced)
No Allegro training / teacher fine-tuning / DFT / new long MD / final student training before final
teacher / mass teacher-labeling of PC002 yet; do not touch unrelated VASP jobs; do not modify the
frozen architecture; do not restart provenance/root-cause audits; no push/PR/merge.
