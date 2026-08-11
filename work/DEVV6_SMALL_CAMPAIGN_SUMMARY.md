# SIO2_DISTILLATION_DEV_V6_SMALL_002 — end-to-end automation-test outcome

Purpose: prove the PydanticAI/controller runtime advances a distillation campaign end-to-end,
cheaply (400-structure DEV subset), with v6 as a PROVISIONAL development teacher. NOT final science.

## Controller-driven advancement (proven)
Driven by `work/devv6_campaign_runner.py` (a minimal RESUMABLE runner composing existing controller
commands — `run_stage`/`record_gate`/`status`; no new state machine, no architecture change):

| stage | status | gate | notes |
|---|---|---|---|
| teacher_labeling | completed | PASS | v6 labeled 400 (sha 277262dc); energy+forces; provenance manifest |
| label_validation | completed | PASS | deterministic §9 validator (400/400, hashes, forces (N,3) finite, O Si, no DFT leak, no dup) |
| dataset_split | completed | PASS | frozen ft_split (train 360 / valid 40); no exact/adjacent leakage |
| training | **FAILED** | pending | real runtime failure — see routing below |
| evaluation | pending | — | blocked upstream |
| physical_validation | pending | — | blocked upstream |

- **CLAUDE_CODE_IS_NOT_THE_CAMPAIGN_ORCHESTRATOR = TRUE** (one runner call advanced 3 gates; I did not hand-call transitions).
- **CONTROLLER_OWNS_STAGE_PROGRESSION = TRUE** (`manifest.json` is the authority; gates enforced).
- **CAMPAIGN_IS_RESUMABLE = TRUE** (proven by the `_SMALL_001 -> _002` clean re-init after the git-integrity guard fired).
- **Gates**: recorded via `DEV_DETERMINISTIC_ATTESTATION` (3 lenses, each backed by a real deterministic check;
  no LLM provider on this CPU node — no API key, no local vLLM). Real LLM judge committee is reserved for the
  final scientific campaign.

## Training failure — classified + routed (§16), not hidden
`adapters/student.py train_student -> NotImplementedError: student kind='simple-nn-v2' requires adapter.train
or train.command`. Category `other` (TRAINING_CONFIG_INCOMPLETE); responsible role **ml-trainer**;
return_stage `training`. Full remediation (student config -> adapter schema + `params_Si/params_O` descriptor
assets + version-correct SIMPLE-NN env [CPU-only torch + sklearn/scipy pin violations] + adequate compute for
360 structs incl 10x ~2900-atom cells x 4 seeds) exceeds a trivial reversible tweak -> **WAIT_ML_TRAINER**.
Record: `runs/SIO2_DISTILLATION_DEV_V6_SMALL_002/recovery/training_failure_routing.json`.

## Boundaries / honesty
- `AUGMENTATION_STATUS = BYPASSED_FOR_SMALL_DEVELOPMENT_CAMPAIGN` (see `PC002_AUGMENTATION_POLICY.md`).
- Full PC002 (5,552) frozen + unmodified; 11 SCAN DFT cells never in training (overlap 0).
- GPU Allegro T1 branch untouched; 64 VASP jobs untouched; no new DFT/Allegro; PC001/PC002/PC003 not reopened.
- Superseded: full-v6 `SIO2_DISTILLATION_DEV_V6_001` (SUPERSEDED_BY_SMALL) and `_SMALL_001` (guard re-init).

## To resume training (next ML-Trainer step)
Fix the student config + supply descriptor assets + version-correct env, then `rebind-inputs` (or fresh init)
and re-run `training`. Real SIMPLE-NN execution belongs on a GPU/version-correct backend, dispatched via the
trusted executor — not this CPU node.
