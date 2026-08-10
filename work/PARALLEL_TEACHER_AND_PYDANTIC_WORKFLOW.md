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

## TRACK A — EXTERNAL GPU (PREPARED, NOT EXECUTED HERE)
Deliverable: `work/EXTERNAL_GPU_TARGET_TEACHER_TRAINING_PACKAGE.md` + portable package
`work/external_gpu_teacher_package/` (104 MB; corpus 3,766 frames = 2,966 target core + 800 replay;
warm-start ckpt `51342b33`; config, scripts, manifests, READMEs, integrity SHA256SUMS).
- Experiment PC-A1: target-focused **same-architecture** warm-start fine-tune (LR 0.001, forces:energy
  4:1, early stop). One controlled experiment; **no** sweep. Existing DFT labels only; **no new DFT**.
- A6 fallback (documented only, not prepared): target-focused higher-capacity Allegro, decided only
  if PC-A1 fails.

## TRACK B — CPU PydanticAI (ACTIVE, running now)
| campaign | state | deliverable |
|---|---|---|
| PC002 structural design | **STRUCTURE_SELECTION = RESOLVED**; labeling blocked | `PRODUCTION_CAMPAIGN_002_STRUCTURE_SELECTION.md`, `pc002_structure_manifest.csv` (20 structs), state json |
| PC003 student workflow | PREPARED (`NEW_PIPELINE_CURRENT_STUDENT = NONE`) | `PC003_STUDENT_WORKFLOW_PREP.md` |
| PC004 student validation | PREPARED (infra teacher-independent) | `PC004_VALIDATION_PROTOCOL_PREP.md` |
| PC005 physical validation | PREPARED (scripts + baselines; no new MD) | `PC005_PHYSICAL_VALIDATION_PROTOCOL_PREP.md` |

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
