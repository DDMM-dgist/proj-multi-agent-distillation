# R1.P1R3 — INGEST_RECOVERED_V5_PROVENANCE_AND_RESOLVE_L2 (contract; NOT executed)

Prospective contract for the action **after** a human manually transfers the KISTI v5 provenance artifacts
listed in `work/production_run1_v5_provenance_recovery_manifest.json`. **This contract is prepared only; it
is not executed here, and it authorizes no transfer.** R1.P1R3 is a read-only, deterministic,
network-free, no-compute action.

## Preconditions (checked at R1.P1R3 start; else it refuses)

- The minimum-required artifacts (`v5_train_list`, `v5_valid_list`, `v5_augmented_structures`,
  `v5_defect_reaugment_seed_manifest`) are present at explicitly-passed local paths (transferred by a
  human out of band). R1.P1R3 accepts **only explicitly transferred files**; it never fetches anything.
- Parent runs immutable: `prod-run1-v5-heldout-generalization` (R1.P1) and `prod-run1-v5-leakage-resolution`
  (R1.P1R) are unchanged. R1.P1R3 writes a **fresh** append-only run
  `runs/production_run1/prod-run1-v5-l2-resolve/`.

## R1.P1R3 will (deterministic, read-only)

1. **SHA-256 each transferred input** and record it; preserve originals **read-only** (never modify).
2. **Parse train/valid/test membership** from the recovered lists → the set of v5 training/validation
   structure labels.
3. **Resolve augmentation frame identities**: map each list entry → augmented-structure-DB frame → seed/
   parent identity, using the augmented DB + defect-re-augment seed manifest.
4. **Compare against the held-out query keys** (`held_out_parent_query_keys` in the manifest: cell_ho_02…07
   parent dump#frame + carved `input.data` SHA + composition) at:
   - **L0** exact file/SHA, **L1** canonical composition+box+wrapped-frac-coord multiset,
   - **L2** same parent `source_dump`#`frame_idx` (or derived-from) present in v5 training/augmentation.
5. **Issue the authoritative LEAKAGE verdict** via the frozen deterministic `criterion_eval` (policy owns
   the verdict; LLM owns nothing):
   - **PASS** — no held-out cell overlaps v5 training/augmentation at L0/L1/L2.
   - **FAIL** — at least one held-out cell overlaps (report which, and its level + evidence).
   - **REVISE** — a required mapping tier is still missing/unresolvable.
   Distinguish **TRAIN vs VALIDATION vs TEST vs AUGMENTATION** membership (a held-out cell found only in
   validation/test is reported separately, not as training leakage).
6. **Perform ZERO model calls, ZERO DFT/MD/training/scheduler/network, ZERO semantic Judge.**
7. Emit: `input_manifest.json` (transferred-file SHAs), `membership_resolution.csv`,
   `l2_membership_matrix.csv`, `leakage_verdict.json`, `criterion_results.json`, `provenance.json`,
   `run_manifest.json`. Historical R1.P1/R1.P1R results are **not** rewritten.

## Transition after R1.P1R3

- **LEAKAGE = PASS** → the held-out set is independent; only then address `ORIGINAL_VS_V5` (if original
  held-out predictions remain MISSING, *propose* — not execute — a tightly bounded original-student
  held-out inference for exactly cell_ho_02…07).
- **LEAKAGE = FAIL** → do not use the contaminated cell(s) for any clean v5-vs-original claim; recommend
  constructing/recovering a genuinely independent held-out set.
- **LEAKAGE = REVISE** → name the exact still-missing artifact; do not start teacher DFT-anchored
  fine-tuning while unresolved (unless justified independently of the held-out claim).

## Not authorized by this contract

Any KISTI/network connection or scp/rsync; any model inference (teacher/student), DFT, MD, training,
scheduler; the carving phase R1.P2; architecture changes; rewriting Stage-D or R1.P1/R1.P1R history;
push/PR/merge.
