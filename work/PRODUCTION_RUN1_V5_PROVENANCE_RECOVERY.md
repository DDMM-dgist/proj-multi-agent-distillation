# Production Run 1 · R1.P1R2 — v5 Training-Provenance Recovery (human checklist)

Local read-only forensics + recovery **preparation**. **No KISTI connection, no network, no scp/rsync, no
model/DFT/MD/training/Judge.** Produces the minimal set of KISTI artifacts a human must copy so that
R1.P1R3 can resolve Level-2 leakage. Machine-readable form:
`work/production_run1_v5_provenance_recovery_manifest.json`. Labels: **FACT / DERIVED / INFERENCE /
UNRESOLVED**.

## 1. Why recovery is needed

R1.P1R proved **L0 (exact) = NONE** and **L1 (structure-equivalent) = NONE** for the held-out cells vs all
local structures, but **L2 (same parent frame in v5 training) = UNRESOLVED**, because the v5 training
structure lists and the augmented structure database are **not present locally** (KISTI-origin). Without
them, frame-level independence of the held-out cells from v5 training cannot be proven. (FACT)

## 2. Current status

`R1.P1R LEAKAGE = REVISE` · `ORIGINAL_VS_V5 = UNRESOLVED`. These remain unchanged until the artifacts below
are recovered and checked by R1.P1R3. **No historical record is rewritten by this action.** (FACT)

## 3. Exact historical v5 run identity (FACT / DERIVED)

- Committee = 4 SIMPLE-NN seeds sharing **one data pool**; differ only by `random_seed`
  (234/345/555/777) (`…/v5_committee_bundle/seed0{1..4}/input.yaml`).
- Deployed member md5: `42a85cd0…`, `52507ee6…`, `bab2e896…`, `92f669b4…` (local).
- v5 = augment-atoms base (~10k, 0 DFT anchors) + a **T1 defect re-augment**.

## 4. Historical KISTI workdir (DERIVED)

- **`/home/hyunjin/2026/03/01_DISTILLATION/01_SIO2/05_DISTILLATION_DIRECT`** — confidence
  **STRONGLY_SUPPORTED** (named "GPU: original distillation run dir",
  `gpu_finetune_handoff/distillation/DISTILLATION_RECIPE.md:112,163`).
- KISTI data root: `/home/hyunjin/workflow/PCA_SOAP_workflow/` (`data_provenance/PROVENANCE.md`).
- CPU augment-notebook dir: `/home/hyunjin/workflow/SIO2_DISTILLATION_DATA/`.
- **HOST = KISTI HPC GPU cluster; specific node name UNRESOLVED.**
- **UNRESOLVED:** the exact v5 *defect-re-augment* subdir (may be a variant under the workdir above).

## 5. Minimum required files (recover these)

| Logical name | Expected KISTI path | Needed for |
|---|---|---|
| `v5_train_list` | `…/05_DISTILLATION_DIRECT/**/train_list` | train membership |
| `v5_valid_list` | `…/05_DISTILLATION_DIRECT/**/valid_list` | train/val membership |
| `v5_augmented_structures` | `…/05_DISTILLATION_DIRECT/**/*augmented*.xyz` (the labeled training DB the lists index) | train + augmentation + parent-frame lineage |
| `v5_defect_reaugment_seed_manifest` | `…/05_DISTILLATION_DIRECT/**/(v5 defect re-augment seed list or generator + input xyz)` | **decisive** augmentation/parent-frame lineage |

`v5_test_list` is **optional** (v5 used `test:False`).

## 6. Optional files (do NOT transfer for leakage)

Training logs, optimizer states, full model checkpoints, stdout. The deployed `potential_saved_bestmodel`
is already local. **Prefer the smallest provenance transfer.**

## 7. Expected remote paths

All under `HISTORICAL_HOST:/home/hyunjin/2026/03/01_DISTILLATION/01_SIO2/05_DISTILLATION_DIRECT/` (search
its SIMPLE-NN subdirs for `train_list`/`valid_list` and the augmented `*.xyz`). The v5 defect-re-augment
seed source may sit alongside or under `/home/hyunjin/workflow/SIO2_DISTILLATION_DATA/`.

## 8. Expected relationships between files (INFERENCE — why lists alone are NOT sufficient)

`generate_features:False` + `preprocess:False` in the v5 `input.yaml` ⇒ `train_list`/`valid_list` index
**pre-generated feature files** (structure labels), **not** parent structures. Mapping a list entry back to
a parent frame requires the **augmented structure DB** (+ for the defect re-augment, its **seed manifest**).
So: `train_list → feature/label → augmented_structures frame → seed/parent`. **All three tiers are
required** to test held-out-parent membership.

## 9. Held-out parent keys to test (query set — FACT, `manifest_heldout.csv`)

| cell | parent dump # frame | comp O/Si | local_x |
|---|---|---|---|
| ho_02 | random_sweep/x006 #20 | 45/22 | −0.023 |
| ho_03 | random_sweep/x015 #20 | 48/25 | 0.040 |
| ho_04 | anneal_calib_clustered/T1000 #20 | 25/24 | 0.479 |
| ho_05 | anneal_calib_clustered/T1000 #30 | 22/27 | 0.593 |
| ho_06 | anneal_calib_clustered/plane_T1000 #20 | 35/28 | 0.375 |
| ho_07 | anneal_calib_clustered/plane_T1000 #30 | 28/24 | 0.417 |

Supporting independence signal (INFERENCE): the base augment seeds come from the **DFT corpus SiOx_filtered**
(M3GNet/PCA-selected, `SiO2-x_Augment_SEED.ipynb` — local in `gpu_finetune_handoff.tar.gz`), **not** from
`production_12288` dumps. The only residual L2 risk is the v5 **defect re-augment** seed source (item above).

## 10. What result will permit LEAKAGE PASS / FAIL

- **PASS** — no recovered v5 training/augmentation frame derives from any held-out parent dump/frame (or
  matches a held-out carved structure at L0/L1).
- **FAIL** — at least one recovered v5 training/augmentation frame derives from a held-out parent
  dump/frame or matches a held-out carved structure.
- **REVISE (persists)** — the recovered lists reference a database still not transferable, or the v5
  defect-re-augment seed source remains unrecoverable.

## 11. What remains unresolved if a file cannot be recovered

- Missing `v5_augmented_structures` **or** `v5_defect_reaugment_seed_manifest` ⇒ L2 stays **UNRESOLVED**,
  leakage stays **REVISE**; the held-out generalization claim (and hence the v5-vs-original decision) cannot
  be made leakage-clean. In that case, an alternative is to **construct a fresh, provably-independent
  held-out set** for any future v5-adoption claim — but that is a later action, not part of R1.P1R2/R3.

## 12. Next prospective action (NOT executed)

`R1.P1R3 = INGEST_RECOVERED_V5_PROVENANCE_AND_RESOLVE_L2` — contract at
`examples/production_run1/R1P1R3_CONTRACT.md`. Do **not** prepare original-student inference, teacher
fine-tuning, or DFT until LEAKAGE resolves to PASS.
