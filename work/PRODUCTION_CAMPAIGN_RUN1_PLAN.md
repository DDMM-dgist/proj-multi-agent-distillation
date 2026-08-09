# Production Campaign — Run 1 Plan (PLANNING ONLY; nothing executed)

Planning deliverable for the first REAL production scientific campaign on the **frozen** architecture
baseline. **No production computation is executed.** No teacher inference, DFT, MD, training, scheduler
job, or semantic Judge is run. Every major conclusion is tagged **FACT / DERIVED / INFERENCE /
UNRESOLVED** with a source path. Evidence root `RES = research-sio2-allegro-simplenn-distillation/`.

---

## 1. Architecture baseline

- **FACT** — baseline HEAD `0c5892e5e91910ed2c7e53f0e10b6b8d10ef7738`; suite 466 passed / 8 skipped;
  `ARCHITECTURE_DESIGN/VALIDATION/REPRESENTATIVE_REAL_ACTION_VALIDATION = COMPLETE`. Frozen
  responsibilities (unchanged): PydanticAI = typed agent interaction / role execution / semantic
  interpretation; `workflow/controller.py` = sole durable workflow-state authority; trusted
  executors/adapters = scientific side effects; deterministic validators = authoritative gates; semantic
  Judge = advisory only; human approval = mandatory boundary for costly/scientific side effects;
  append-only provenance. This plan uses the architecture **as-is** (§13).

## 2. Current scientific state (reconstructed from source, not prose)

Teacher→student distillation of an Allegro (NequIP) teacher into a SIMPLE-NN committee for amorphous
SiO₂ and oxygen-deficient SiO₂₋ₓ, with committee-uncertainty-guided active learning.

- **FACT** Teacher = **base** KISTI Allegro, deployed compiled TorchScript `teacher/model.nequip.pth`
  == `RES/gpu_finetune_handoff/models/teacher_current_compiled.nequip.pth`, SHA
  `b56e20ff…1b1c57`, cutoff 5.0 Å, symbols [O,Si]. A fine-tuned `teacher_finetuned_v2` (SHA `b3be4d2a…`)
  exists but is **not deployed**; its ER-FT gain did not survive distillation
  (`gpu_finetune_handoff/DISTILLATION_DIAGNOSIS_2026-07-03.md`).
- **FACT** Student = SIMPLE-NN Behler-Parrinello, 4-seed committee, 70 sym-funcs/element, nodes 30-30,
  cutoff 5.0 Å (`gpu_return_v5_committee/v5_committee_bundle/seed01/input.yaml`). Uncertainty metric
  `u_α = sqrt(Σ_xyz Var(F))` per atom; carve threshold 0.15 eV/Å.
- **FACT** Physical validation of the deployed student PASSES (12288-atom glass: ρ 2.217, Si–O 1.610 Å,
  CN(Si) 4.00, O–Si–O 109.41°, FSDP, NVE drift −0.0049 meV/atom/ns, MSD plateau, mechanics, PH) —
  `production_12288/validation_out/summary.txt`, `coordination_log.csv` physical-validation-gate PASS.

## 3. Existing data / model inventory

| Asset | Path | Count / key numbers | Source |
|---|---|---|---|
| DFT(SCAN) labelled cells | `dft_labeling/`, `al_iter3/{dft_validation_11A,v6_dft_batch,heldout_dft_batch}` | **39 converged SCAN cells** (11 original + 4 + 16 + 8) | DATA audit; OUTCARs present |
| Held-out DFT test cells | `al_iter3/heldout_dft_batch/cell_ho_01..08` | 8 (4 random x003–x015 + 4 clustered sphere/plane T1000 x012) | `manifest_heldout.csv` |
| Held-out v5+teacher vs DFT errors | `heldout_dft_batch/analysis/heldout_baseline_errord.csv` + `per_cell/cell_ho_{02..07}_baseline.csv` | **6 cells scored** (ho_01, ho_08 absent) | FACT (files present) |
| Teacher error(a) vs DFT | `teacher_diag/error_a_allegro_vs_dft.csv` | 1155 frames; E-MAE raw 26.8 meV/atom; F-MAE 0.190 eV/Å | recomputed |
| Student error(c) vs DFT | `teacher_diag/error_c_simplenn_vs_dft.csv` | F-MAE 0.233 eV/Å; 11-AL-cell "0.368 reference" | recomputed |
| v5 committee bundle | `gpu_return_v5_committee/v5_committee_bundle/seed0{1..4}/potential_saved_bestmodel` | 4 md5-distinct SIMPLE-NN | FACT |
| Deployed ("original") committee | `/home/hyunjin/workflow/SIMPLE_NN_DISTILLATION_CE/0{1..4}_potential_saved_bestmodel` | 4 seeds | FACT (external path) |
| Production MD | `production_12288/`, `sio2x_production/`, `random_sweep/x003..x024` | 12288-atom glass + vacancy sweeps | FACT |
| Gate decision trail | `coordination_log.csv` | 27 rows: 16 PASS / 10 REVISE / 1 FAIL | FACT |

## 4. What has already been completed

- **FACT** Teacher trained; student committee distilled; a-SiO₂ + SiO₂₋ₓ production MD; full physical
  validation (RDF/ADF/CN/S(Q)/FSDP/MSD/NVE/mechanics/PH) PASS.
- **FACT** 4-error decomposition: error(a) teacher-vs-DFT 0.190, error(b) student-vs-teacher 0.175,
  error(c) student-vs-DFT 0.233 eV/Å (near-ambient much lower).
- **FACT** Committee-uncertainty AL: production-MD uncertainty study (2026-06-16) → carve 8 clustered
  DFT-tractable cells → DFT(SCAN) label (7 PASS, 1 carve-FAIL) → v6/heldout DFT batches (39 cells total).
- **FACT** Re-distillation attempts v3/v3-final/v3-final-v2 REJECTED (never beat original). **v5 ADOPTED
  2026-07-16** (coordination_log final row): "first re-distillation to beat original on BOTH deployment
  distribution AND error(c) 0.337<0.368 … AL loop CLOSED for this iteration."
- **UNRESOLVED** The v5 adoption carries an explicit caveat: **"11 AL cells may overlap v5 train (error(c)
  partly in-dist); u_max still >0.30 at x≥0.12; clustered_cell hardest."** STATUS.md prose elsewhere still
  says "keep original student" — a **gate-log-vs-prose contradiction** on the current best-student identity.

## 5. Current scientific bottleneck (exactly one)

**BOTTLENECK (DERIVED, source-grounded):** *The adopted v5 improvement is not yet verified leakage-clean
on held-out OOD data, and the held-out evidence indicates the residual error at clustered defect cores is
**teacher-limited, not distillation-limited** — so the correct next lever (teacher DFT-anchored fine-tune
vs more student distillation) is undecided.* Evidence, from `heldout_baseline_errord.csv` (v5=errd,
teacher=erra, gap=errb; per-atom core = defect region):

| Family (held-out) | v5 error(c)_core | teacher error(a)_core | gap error(b)_core | v5 u_α center |
|---|---|---|---|---|
| random (ho_02 x006, ho_03 x015) | **0.345** | 0.297 | 0.182 | 0.30 |
| clustered (ho_04–07 sphere/plane x012, CN0/CN1) | **0.389** | **0.381** | 0.233 | 0.18 |

- **DERIVED** At clustered defect cores the teacher itself is only ~0.38 eV/Å accurate vs DFT (vs 0.19
  global), and v5 ≈ teacher there (0.389 ≈ 0.381) → the student has essentially reached the teacher's
  ceiling; **more distillation cannot close this gap; the teacher must improve** (teacher-in-the-loop).
- This is the most recent open scientific question in the gate trail (the v5-adoption caveat), and it
  gates whether to spend HPC on iteration-2 and on *what*. It was **not** chosen for architecture
  convenience — it is the literal unresolved item at the frontier of the recorded workflow.

## 6. Production Campaign Run 1 — objective

**PRODUCTION_CAMPAIGN_RUN_1 = Leakage-clean held-out generalization verdict on the adopted v5 committee,
with residual-error attribution (teacher vs distillation), to decide the iteration-2 lever — without any
new compute.**

- `scientific_question` — Does v5's adopted improvement hold on **provably held-out** OOD cells, and is the
  residual OOD error teacher-limited or distillation-limited?
- `current_state` — v5 ADOPTED but caveated (possible train/test leakage; clustered still hardest);
  best-student identity contradicted between gate log and prose.
- `target_state` — A deterministic verdict: (i) leakage certificate for the held-out cells; (ii)
  per-family v5/teacher/gap errors with explicit domain tags; (iii) attribution (teacher-limited vs
  distillation-limited); (iv) a decision recommendation for iteration-2 (teacher fine-tune vs re-distill)
  — all from existing artifacts.
- `input_artifacts` — §3 (held-out per-cell CSVs, DFT OUTCARs already gate-PASSED, v5 bundle identity,
  teacher SHA, error(a)/(c) CSVs).
- `expected_new_artifacts` — `heldout_generalization_verdict.json`, `leakage_certificate.json`,
  `per_family_error_table.csv`, `criterion_results.json`, `provenance.json`.
- `success_criteria` / `failure_criteria` — §10.
- `decision_gates` — one deterministic gate (§10); no semantic Judge (§11).
- `human-approval boundaries` — **none required** (read-only, no side effect; §12).
- `compute` — CPU, seconds, no scheduler, no network (§9).
- `provenance` — append-only run dir; input SHAs recorded; leakage certificate immutable (§15).
- `stop conditions` — §16.

## 7. Exact workflow phases

Run 1 is deliberately a **single-phase, read-only** campaign (smallest meaningful step; interpretable
before any HPC). Later phases are listed but **not part of Run 1** and **not authorized**.

| Phase | Purpose | Role | action_type | Deterministic validation | Semantic? | Approval | Executor/tool | Artifact | Platform | Cost | PASS→ | REVISE→ | FAIL→ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1.P1 (Run 1)** | leakage-clean held-out generalization verdict + attribution | ml-trainer | `evaluate_heldout_fidelity` | artifact validity + leakage + model-accuracy + physical-validity gates (§10) | **No** | **No** (read-only) | new trusted read-only aggregator adapter (MISSING; §13) | verdict + leakage cert + per-family table | CPU/local, seconds | — | verdict=generalizes / teacher-limited | localize residual OOD; propose iteration-2 lever | data/label defect → re-carve or re-label |
| *(later, NOT Run 1)* P2 | carve iteration-2 candidates at the identified OOD target | data-curator | `detect_atomic_overlap`+carve (MISSING adapter) | geometry/min-dist/domain gates | No | No (read-only carve) | — | candidate cells | CPU | mins | — | — | — |
| *(later)* P3 | DFT(SCAN) label the carved iteration-2 cells | simulation | `run_dft` | convergence + label validity | optional | **YES** | HPC/VASP | DFT E/F labels | HPC scheduler | hours–days | — | — | — |
| *(later)* P4 | teacher DFT-anchored fine-tune at clustered defects | ml-trainer | `fine_tune_teacher` (OUT_OF_CURRENT_SCOPE) | before/after error(a) at defects | optional | **YES** | GPU | fine-tuned teacher | GPU | hours | — | — | — |
| *(later)* P5 | re-distill + committee adopt-gate | ml-trainer | `train_committee`→adopt | error(c)/u_max vs original | No | **YES** | GPU | new committee | GPU | hours | — | — | — |

## 8. Inputs and SHAs

- **FACT** Teacher `teacher/model.nequip.pth` SHA `b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57` (== deployed).
- **FACT** v5 committee bundle `gpu_return_v5_committee/v5_committee_bundle/seed0{1..4}/potential_saved_bestmodel` — 4 md5-distinct (verified by the held-out script's own md5 diversity check).
- **FACT** Held-out scored cells: `heldout_dft_batch/analysis/per_cell/cell_ho_{02,03,04,05,06,07}_baseline.csv`; aggregate `heldout_baseline_errord.csv`.
- **DERIVED** Per-file SHA-256 of the above are computed and recorded by the executor at run time (not precomputed here). **UNRESOLVED** ho_01 and ho_08 are absent from `analysis/` (2 of 8 held-out cells unscored).

## 9. Compute / resource plan

- **Run 1 (P1):** CPU only, read-only aggregation of existing CSV/OUTCAR/JSON; **seconds**; no GPU, no
  scheduler, no external network; no checkpoint/restart needed. **No approval** (no side effect).
- **Later phases (NOT authorized here):** P3 DFT = HPC VASP/SCAN, ~40–80-atom cells, KPAR=1, hours/cell,
  scheduler; P4 teacher fine-tune = 1 GPU, hours, checkpointed; P5 re-distill = GPU committee train,
  hours. Each would carry its own approval + resource proposal.

## 10. Deterministic scientific gates (four distinct axes — §item 10; NOT reusing Stage-D thresholds)

1. **ARTIFACT VALIDITY** (authoritative): per-cell CSVs parse; atom counts match `manifest_heldout.csv`;
   all force/energy values finite; DFT OUTCARs already judge-gate-PASSED (`coordination_log.csv`); v5
   bundle = 4 md5-distinct models; teacher SHA == `b56e20ff…`.
2. **LEAKAGE** (authoritative): each held-out cell's source config/frame is **disjoint** from the v5
   training/augmentation sources. Grounded in `HANDOFF_v6.md` train/val seed manifests +
   `manifest_heldout.csv` provenance. **UNRESOLVED offline** if the v5 augmentation frame list lives on
   KISTI — then this gate returns REVISE (cannot certify), not PASS.
3. **MODEL ACCURACY** (grounded in real stats, not Stage-D): v5 held-out error(c)_core is judged against
   (a) the original committee's held-out error(c) [improvement claim], and (b) the teacher's error(a)_core
   in the same family [distillation-gap ceiling]. Reference scale: teacher global error(a)=0.190,
   near-ambient student 0.145, 11-AL-cell 0.368 (all eV/Å). **Attribution rule:** if
   error(c)_core ≈ error(a)_core (gap error(b) small) → *teacher-limited*; else *distillation-limited*.
4. **PHYSICAL VALIDITY** (defect-domain band, grounded — NOT the equilibrium [−11,−8]): held-out DFT
   E/atom must lie in the **relaxed O-deficient SiO₂₋ₓ band derived from the DFT-labelled clustered cells**
   (`coordination_log.csv`: −9.41 … −9.80 eV/atom) → proposed band **[−10.0, −9.0] eV/atom**. (These are
   relaxed/annealed DFT cells, unlike the C3 strained input; §12.)

## 11. Optional semantic gate

**None.** All four axes are quantitative/structural and deterministic. Per the architecture rule, **no LLM
Judge is added when deterministic criteria suffice**, and the LLM never owns these thresholds. A semantic
interpretation could *later* summarize the verdict for humans, but it is **not** part of Run 1 and never
binds the gates.

## 12. Human approval points

- **Run 1 (P1): none.** It is read-only over existing artifacts with no scientific side effect; approval is
  the mandatory boundary only for **costly/side-effecting** actions (teacher inference, DFT, MD, training,
  scheduler) — none occur in Run 1.
- **C3 lesson carried forward (§item 12):** every held-out cell in the verdict records explicit
  provenance + domain — source config, frame, center CN, `local_x`, n_atoms, and **whether it lies inside
  the teacher/student training domain**. A structure is never called "representative"/"in-domain" from a
  filename; the DFT E/atom band used is the **defect-domain** band, not the equilibrium band that C3's
  Axis-B (mis)applied.

## 13. Architecture-issue classification (document, do NOT fix — §item 13)

- **SCIENTIFIC_WORKFLOW_ISSUE** — gate-log (v5 ADOPTED) vs STATUS prose (keep original) contradiction on
  the current best student. Resolved *by* Run 1's verdict, not by code.
- **PRODUCTION_CONFIGURATION_ISSUE** — ho_01/ho_08 unscored; no merged labelled extxyz for al_iter3;
  v5 augmentation frame list not in-repo (KISTI). Data packaging, not architecture.
- **MISSING_ADAPTER_FOR_NEW_SCIENTIFIC_ACTION** — no trusted executor/adapter yet exists for
  `evaluate_heldout_fidelity` (a read-only aggregator), analogous to C3 needing a teacher adapter. The
  **action_type already exists** in the frozen taxonomy (`ML_TRAINER_ACTIONS`), so this is an adapter to
  *build later under the frozen contracts*, not an architecture change.
- **GENUINE_ARCHITECTURE_BLOCKER** — **none found.** The plan expresses entirely within the frozen
  architecture.

## 14. (This document)

## 15. Provenance strategy

Append-only run dir `runs/production_run1/<run_id>/` with: the validated proposal snapshot; input SHAs;
`leakage_certificate.json` (immutable); `per_family_error_table.csv`; `heldout_generalization_verdict.json`;
`criterion_results.json` (bound deterministic verdict); `provenance.json` (package HEAD, tool identity,
per-artifact SHA-256). No existing research artifact is mutated; the verdict is additive.

## 16. Stop conditions

- **ADVANCE** (verdict emitted) iff all four gates evaluate and leakage is certified → record whether v5
  generalizes and the attribution (teacher- vs distillation-limited), and recommend the iteration-2 lever.
- **REVISE** if leakage cannot be certified offline (KISTI frame list absent) or ho_01/ho_08 must be scored
  first — the verdict is emitted with an explicit coverage caveat.
- **STOP/FAIL** if any artifact-validity criterion fails (non-finite, atom-count mismatch, SHA mismatch,
  DFT non-converged) or a physical-validity value is outside the defect band → localize and halt.
- Run 1 **stops at the verdict.** It does **not** trigger P2–P5.

## 17. Risks / unresolved scientific questions

- **UNRESOLVED** Is v5 truly the production student, or is the "keep original" prose current? (Run 1
  resolves via leakage-clean held-out numbers.)
- **UNRESOLVED** v5 augmentation frame list (leakage ground truth) may be KISTI-only → leakage gate may
  REVISE.
- **UNRESOLVED** Only 6/8 held-out cells scored; ho_01/ho_08 coverage gap.
- **INFERENCE** The residual clustered-core gap is teacher-limited (erra≈errd) → iteration-2 should be a
  **teacher DFT-anchored fine-tune at clustered defects**, not more distillation. Run 1 confirms/refutes
  this quantitatively before any HPC.
- **Risk** the defect-domain E/atom band [−10.0,−9.0] is derived from only ~7 clustered DFT cells; widen
  only if new DFT justifies it (do not invent).

## 18. Actions explicitly NOT yet authorized

Teacher inference; DFT/VASP; any MD (teacher or student); committee training / re-distillation; teacher
fine-tune; scheduler submission; semantic Judge; carving new candidate cells; starting C2; modifying any
Stage-D artifact or threshold; push/PR/merge.

## 19. Proposed first executable action

**`evaluate_heldout_fidelity`** (role **ml-trainer**, `dry_run=true`, **not approval-gated**): a read-only,
no-compute deterministic aggregation of the **existing** held-out v5+teacher-vs-DFT per-cell errors into a
leakage-checked, domain-tagged generalization verdict with teacher-vs-distillation attribution. Full typed
proposal: `examples/production_run1/action_proposal.json`; contract: `examples/production_run1/CONTRACT.md`.
It produces **new artifacts only under the run dir**, mutates nothing, and requires no approval.

## 20. Why that first action is scientifically necessary

It resolves the single open item at the workflow frontier — whether the **adopted** v5 improvement is real
(leakage-clean) on held-out OOD data — and localizes the residual error to the teacher vs the distillation
step. That verdict is the **decision gate** that determines whether the next (expensive, HPC) iteration
should fine-tune the **teacher** at clustered defects or re-distill the **student**, and whether v5 or the
original is the production student. It is the smallest step that is fully interpretable **before** any large
HPC workload, uses only existing artifacts, and carries no scientific side effect — exactly the
"one validated state transition at a time" discipline the production phase requires.
