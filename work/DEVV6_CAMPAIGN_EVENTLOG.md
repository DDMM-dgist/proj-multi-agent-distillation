# SIO2_DISTILLATION_DEV_V6_001 — campaign event log

**Controller-driven.** Authoritative state = `runs/SIO2_DISTILLATION_DEV_V6_001/manifest.json`
+ `python -m workflow.controller status runs/SIO2_DISTILLATION_DEV_V6_001`. This file mirrors that
state for humans; it is NOT a hand-simulated substitute.

- `campaign_type = DEVELOPMENT` · `final_scientific_result = false`
- Provisional teacher = **v6** (`teacher_v6_finetuned`), sha256 `277262dc2d47124748d885a9f26b1f7a1697066183c624188826ff787a501dd5`, role `PROVISIONAL_DEVELOPMENT_TEACHER`
- **PC001**: `scientific = REVISE` (preserved, unchanged) · `development_override = PROCEED_WITH_KNOWN_LIMITATION` (`pc001_development_override.json`)
- **Approval**: `USE_V6_FOR_DEVELOPMENT_DISTILLATION_CAMPAIGN` (`approval.json`)

## Transitions (from controller)
| ts | prev → new | stage | role/action | backend | artifacts | gate |
|----|-----------|-------|-------------|---------|-----------|------|
| init | (none) → pending×5 | all 5 | controller init; 9 inputs hash-bound (7 configs + `devv6_ensemble.xyz` + v6 model `copy:false`) | agox | `manifest.json`, `inputs/000-…006`, `workflow.yaml` | — |
| run-stage | pending → **RUNNING** (attempt 1) | teacher_labeling | DataCurator · `adapters.acquisition label` | **allegro env, CPU** | `logs/teacher_labeling.attempt-1.log`; on completion → `artifacts/teacher_labeled.extxyz` + `teacher_labels.manifest.json` | pending (post-completion) |

## Current stage table
`teacher_labeling RUNNING` · dataset_split pending · training pending · evaluation pending · physical_validation pending

## Next automatic transitions
1. teacher_labeling completes → **deterministic label validation** (§9: all 5,552 IDs present, hash match, energy present, forces shape (N,3), finite, `O Si` order, no PC004 DFT-cell inclusion, no dup/missing) → controller `gate`.
2. PASS → `dataset_split` (frozen PC002 ft_split via `work/devv6_split.py`) → `training` (SIMPLE-NN 4-seed, simple-nn env — costly, within approved scope) → `evaluation` (student-vs-teacher on 558) → `physical_validation`.
3. Any deterministic REVISE/FAIL → controller `propose-recovery` (routed to the responsible role); actions within the approved reversible scope proceed automatically; new DFT / new Allegro training / major generation / new long MD / different architecture STOP at the human-approval boundary.

## Scope guards
Teacher v6 is PROVISIONAL (never called final/accepted/improved). GPU Allegro T1 branch untouched.
No new DFT / no new Allegro training / no VASP interaction. PC002 not redesigned; PC003 policy not reopened.
When GPU T1 returns: register separately, run a distinct FINAL scientific campaign; do not merge v6 dev metrics.
