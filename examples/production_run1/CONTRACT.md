# Production Run 1 · Phase 1 — execution contract (PREPARATION; not executed)

First production action on the frozen architecture: a **read-only, no-compute** deterministic held-out
generalization verdict on the adopted **v5** SIMPLE-NN committee. **Nothing is executed; no model is
loaded/invoked; no DFT/MD/training/scheduler; no result files exist; no run dir is created.** No human
approval is required because the action has **no scientific side effect** (read-only over existing
artifacts). This contract is the analogue of the C3 CONTRACT, adapted for a read-only analysis action.

## Action

- Role **ml-trainer**; `action_type = evaluate_heldout_fidelity` (frozen `ML_TRAINER_ACTIONS`; **not** in
  `APPROVAL_GATED_ACTIONS`). Proposal `action_proposal.json` validates against the frozen
  `MLTrainerActionProposal` (verified). `dry_run=true`, `approval_boundary=null`.
- **Purpose:** aggregate the EXISTING held-out v5+teacher-vs-DFT per-cell errors into a leakage-checked,
  domain-tagged generalization verdict with teacher-vs-distillation attribution.

## Inputs (read-only; SHA-recorded at execution)

`heldout_dft_batch/analysis/heldout_baseline_errord.csv` + `per_cell/cell_ho_{02..07}_baseline.csv`
(v5=errd, teacher=erra, gap=errb, u_α); `manifest_heldout.csv` (provenance/domain); `HANDOFF_v6.md`
(train/val seeds → leakage); `coordination_log.csv` (DFT judge-gate PASS + clustered DFT E/atom
−9.41…−9.80); `teacher_diag/error_a_allegro_vs_dft.csv` (teacher reference 0.190); teacher SHA
`b56e20ff…` (identity only — **model NOT invoked**). **Coverage caveat:** ho_01/ho_08 unscored (6 of 8).

## Four deterministic acceptance axes (authoritative; NOT reused from Stage D)

1. **ARTIFACT VALIDITY** — CSVs parse; atom counts == `manifest_heldout.csv`; all E/F finite; DFT OUTCARs
   already judge-gate-PASSED; v5 = 4 md5-distinct models; teacher SHA == `b56e20ff…`.
2. **LEAKAGE** — each held-out cell's source config/frame **disjoint** from v5 training/augmentation
   sources → PASS; if the v5 frame list is unavailable offline → **REVISE** (cannot certify), not PASS.
3. **MODEL ACCURACY** — v5 `error(c)_core` vs (a) original-committee held-out `error(c)` [improvement] and
   (b) teacher `error(a)_core` same family [distillation-gap ceiling]. **Attribution:** teacher-limited iff
   `error(c)_core ≈ error(a)_core` (small `error(b)`). Reference scale (eV/Å): teacher global 0.190,
   near-ambient student 0.145, 11-AL-cell 0.368.
4. **PHYSICAL VALIDITY (defect-domain band)** — held-out DFT E/atom ∈ **[−10.0, −9.0] eV/atom**, grounded
   in the relaxed O-deficient SiO₂₋ₓ DFT cells (`coordination_log.csv` −9.41…−9.80). **NOT** the
   equilibrium [−11,−8] band that C3's Axis-B mis-applied to a strained input.

## Output artifacts (generated only at execution, under the run dir)

`leakage_certificate.json`, `per_family_error_table.csv`, `heldout_generalization_verdict.json`,
`criterion_results.json` (bound deterministic verdict; LLM owns nothing), `provenance.json` (package HEAD,
tool identity, per-input SHA-256). Run dir `runs/production_run1/prod-run1-v5-heldout-generalization/`.

## C3 lesson enforced (input domain explicit)

Every held-out cell in the verdict records `distribution`, `x_label`, `center_cn`, `local_x`, `n_atoms`,
and **`in_training_domain`**. No cell is called "representative"/"in-domain" from a filename. The physical
band is the **defect-domain** band; artifact validity, model accuracy, physical validity, and campaign
success are reported as **four separate axes**, never collapsed into one threshold.

## Authorization / side effects / stop conditions

- **No approval required** (read-only; no teacher inference / DFT / MD / training / scheduler / network).
- Writes only under the run dir; refuses a pre-existing run dir; mutates **no** research artifact; **no**
  automatic downstream action (P2–P5 are separate, later, and several are approval-gated).
- **ADVANCE** → emit the verdict + attribution + iteration-2 recommendation. **REVISE** → leakage
  uncertifiable offline or ho_01/ho_08 needed first (verdict emitted with coverage caveat). **STOP/FAIL**
  → any artifact-validity failure or a DFT E/atom outside the defect band.

## Missing adapter (documented, not built)

No trusted executor/adapter yet exists for `evaluate_heldout_fidelity` (a read-only aggregator) — classified
`MISSING_ADAPTER_FOR_NEW_SCIENTIFIC_ACTION`. The **action_type already exists** in the frozen taxonomy, so
this adapter is to be built later **under the frozen contracts** (like the C3 teacher adapter was), and is
**not** an architecture change. **Not built in this planning task.**

## Exact first action after (implicit) go — NOT executed now

Prepare-only. When built and run, the read-only aggregator consumes the inputs above, evaluates the four
axes via the frozen `criterion_eval` (bound verdict), and writes the run-dir artifacts. **No approval, no
model, no DFT/MD/training, no scheduler, no network, no downstream chain.**
