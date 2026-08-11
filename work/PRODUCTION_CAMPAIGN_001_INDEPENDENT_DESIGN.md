# Production Campaign 001 — Independent Design from Raw Assets

Design + offline data analysis ONLY. **No teacher/student inference, DFT, MD, training, scheduler,
network, or semantic Judge.** Conclusions are re-derived from raw CSV/model artifacts, **not** inherited
from historical decisions. Claims tagged **FACT / DERIVED / INFERENCE / UNRESOLVED**.
`RES = research-sio2-allegro-simplenn-distillation/`. Reproducible diagnosis:
`work/production_campaign_001_diagnosis.py` → `runs/production_campaign_001/pc001-independent-diagnosis/`.

## 1. Reset rationale

Prior Run-1 work (R1.P1/R1.P1R/R1.P1R2) drifted into **auditing/justifying the historical human workflow**
(v5 adoption, leakage provenance). This campaign restarts from the scientific question and the raw assets,
treating historical decisions as non-authoritative reference only. Prior R1 artifacts are preserved as
**HISTORICAL_EXPLORATORY_PRODUCTION_AUDITS** (not deleted, not rewritten, not the starting state).

## 2. Frozen architecture baseline

`0c5892e5…` — PydanticAI roles, controller as durable state authority, trusted executors, deterministic
validators, human-approval boundaries, advisory Judge, append-only provenance, Stage A–D results, and all
R1 audits are preserved and unchanged. Not reopened. (FACT)

## 3. Scientific objective (DERIVED from primary materials)

A deployable **SIMPLE-NN** potential distilled from an **Allegro** teacher for **amorphous SiO₂ and
oxygen-deficient SiO₂₋ₓ** (dilute + clustered vacancies, void surfaces / under-coordinated Si) at
production density (~2.2 g/cm³) and ambient–moderate T, good enough for **production MD + physical-property
validation**. (`PROJECT.md`, `DISTILLATION_RECIPE.md`, production-MD assets.)

## 4. Raw scientific asset inventory (FACT; source paths)

- **DFT/high-fidelity:** 39 SCAN cells (11 original AL + 28 al_iter3), SiO₂ + SiO₂₋ₓ, local_x −0.09…0.60
  (`dft_labeling/`, `al_iter3/{dft_validation_11A,v6_dft_batch,heldout_dft_batch}`); teacher held-out test
  `test_set.xyz` 1155 frames; DFT E/F in eV, eV/Å (`data_provenance/PROVENANCE.md`).
- **Teacher:** base Allegro `b56e20ff` (deployed); fine-tuned `b3be4d2a` (not deployed).
- **Students:** original/deployed committee (`SIMPLE_NN_DISTILLATION_CE`, 4 seeds); v5 committee
  (`gpu_return_v5_committee`, 4 seeds md5 `42a85cd0/52507ee6/bab2e896/92f669b4`).
- **Error tables (numerical source of truth):** `teacher_diag/error_{a,b,c}.csv` (shared idx+config_type
  over the 1155 test_set).
- **EOS:** `teacher_diag/eos_{teacher,student}_bm_summary.csv`.
- **Simulation/physical validation:** 12288-atom glass + vacancy sweeps (`production_12288/`,
  `sio2x_production/`); PASS on RDF/ADF/CN/S(Q)/FSDP/MSD/NVE/mechanics/PH.
- **Uncertainty/AL:** committee `u_α` production study (`sio2x_production/committee_u_out/`).

## 5. Target deployment domain (DERIVED)

**In-domain:** amorphous SiO₂; SiO₂₋ₓ dilute + clustered vacancies; void-surface under-coordinated Si;
ρ≈2.2; ambient–moderate T; long MD. **Out-of-domain:** high-pressure dense polymorphs (stishovite/pyrite/
PbO₂), elemental-Si bulk, isolated clusters. → **Model ranking must be domain-aware** (a global RMSE would
wrongly penalize the student for out-of-domain phases).

## 6. Teacher candidates (independent)

- **base Allegro `b56e20ff`** — the only teacher with broad DFT validation: global force MAE **0.190 eV/Å**,
  E-MAE(shift) **35.8 meV/atom** (recomputed from `error_a`, n=1155). **SELECTED (DERIVED).**
- fine-tuned `b3be4d2a` — not deployed; gain didn't survive distillation (`DISTILLATION_DIAGNOSIS`). Not
  selected on independent evidence.

## 7. Student candidates (independent)

Two materially-distinct 4-seed committees (original, v5). **CURRENT_STUDENT = UNRESOLVED** — see §10/§14.

## 8. DFT reference data

39 SCAN cells (dilute+clustered SiO₂₋ₓ) + 1155-frame `test_set`. The **28 al_iter3 cells postdate both
students** → a candidate common CLEAN set for a fair comparison (FACT: al_iter3 feeds v6, postdates v5;
original predates all).

## 9. Simulation / physical-validation assets

12288-atom glass validation PASS. **EOS bulk modulus (FACT):** in-domain phases match (α-quartz teacher
B0 201.9 vs student 200.7; coesite 228 vs 226); **out-of-domain high-pressure phases the student
over-stiffens badly** (stishovite 344→**526**, PbO₂ 355→**550**, pyrite 352→**450**). Consistent with the
high-pressure force gap; domain-irrelevant for deployment.

## 10. Independent quantitative model comparison (recomputed from raw)

Global force MAE (eV/Å): **teacher-vs-DFT 0.190**, **original-student-vs-DFT 0.233**,
**student-vs-teacher 0.175** (n=1155 / 881 / 881). Energy MAE(shift): teacher 35.8, student 27.1 meV/atom.

Domain-aware (`per_domain_model_comparison.csv`); attribution rule declared in code
(TEACHER_LIMITED iff errC≈errA & errB<errA; DISTILLATION_LIMITED iff errB>errA):

| domain | n | errA teach-DFT | errC stud-DFT | errB stud-teach | attribution |
|---|---|---|---|---|---|
| SiOx_defect | 96 | 0.324 | 0.432 | 0.309 | MIXED |
| bulk_amorphous/quench | 152 | 0.202 | 0.279 | 0.206 | DISTILLATION_LIMITED |
| bulk_crystal | 328 | 0.031 | 0.067 | 0.062 | DISTILLATION_LIMITED (tiny) |
| vacancy_SiOx | 93 | 0.187 | 0.218 | 0.121 | MIXED |
| surfaces | 69 | 0.216 | 0.317 | 0.221 | DISTILLATION_LIMITED |
| liquid | 34 | 0.211 | 0.283 | 0.219 | DISTILLATION_LIMITED |
| high_pressure* | 58 | 0.193 | 0.459 | 0.421 | DISTILLATION_LIMITED (*out-of-domain) |
| cluster† | 68 | 1.182 | 0.404 | 0.280 | TEACHER_LIMITED† (†cc001 artifact) |
| elemental_Si* | 257 | 0.063 | — | — | student not evaluated (*out-of-domain) |

**Fairness (FACT):** the above is the **original** student. **v5 has DFT numbers only on 6 held-out cells**
(R1). **No common original-vs-v5 DFT set exists** → a v5-vs-original ranking is **not** computable offline.

## 11. Error decomposition (DERIVED)

The only teacher force outlier >5 eV/Å is **cluster cc001 = 56.65** — the known atom-overlap **data
artifact**, not model error. Excluding it, the teacher is accurate across every domain (≤0.32/family).
Therefore, in deployment-relevant domains the larger error source is the **distillation step** (errB ≈
errA; the original student is *worse than the teacher* in bulk-amorphous/surfaces/liquid), or **MIXED**
(SiOx-defect/vacancy). **No domain is genuinely teacher-limited.**

## 12. Domain-specific failure analysis

- **SiOx-defect / clustered-vacancy (in-domain, hardest):** errC 0.43 / errA 0.32 / errB 0.31 → both
  teacher error and a large distillation gap contribute (MIXED).
- **bulk amorphous / surfaces / liquid (in-domain):** DISTILLATION-limited (student ~0.28–0.32 vs teacher
  ~0.20).
- **high-pressure (out-of-domain):** student badly off (force + EOS B0) — irrelevant to deployment.
- **elemental Si / cluster:** out of scope / data-artifact.

## 13. Data-coverage analysis

DFT covers dilute + clustered SiO₂₋ₓ (39 cells, local_x −0.09…0.60) — good defect coverage. The
deployment domain is well-represented by the 28 al_iter3 cells. **UNRESOLVED:** whether the *student
augmentation* covers clustered/void-surface motifs (v5's whole purpose) — cannot be answered without
either a common DFT comparison (first action) or the KISTI train manifest.

## 14. Historical decisions quarantined from inference

Not used to form any conclusion above: "v5 ADOPTED", "keep original", the AL-iteration-3 plan, gate/
STATUS/coordination recommendations, past bottleneck picks. (Rule 8/2.)

## 15. Historical comparison — NON-AUTHORITATIVE

- **Independent:** CURRENT_STUDENT = UNRESOLVED (no common DFT set).
- **coordination_log:** v5 ADOPTED (beat original on deployment distribution + error(c) 0.337<0.368).
- **STATUS prose:** keep original.
- **Classification: HISTORICAL_RECORD_INTERNALLY_CONFLICTED.** The independent analysis neither confirms
  nor refutes v5 adoption — the two students were never put on a common DFT set. The v5 claim rests on a
  *deployment-distribution committee-uncertainty* comparison, not a DFT-fidelity comparison, and its
  error(c) carried a self-declared in-distribution caveat.

## 16. Current scientific state

`ARCHITECTURE = FROZEN_COMPLETE`. `CURRENT_TEACHER = base Allegro b56e20ff` (SELECTED). `CURRENT_STUDENT =
UNRESOLVED` (original vs v5, no common DFT set). Teacher is broadly accurate (0.19); the deployment-domain
error is distillation-dominated or MIXED; the clustered-SiOx-defect domain is the hardest in-domain region.

## 17. Primary bottleneck — **MODEL_SELECTION**

Two student candidates, **no common DFT evaluation set** → CURRENT_STUDENT undeterminable. This gates every
downstream lever (is distillation or teacher the limit for the *deployed* student; did v5's defect
re-augment actually help in-domain). It is the smallest unbiased next step and quarantines the historical
"v5 ADOPTED" claim. (Chosen from evidence — NOT because leakage was the last audit.)

## 18. Production Campaign 001 objective

Establish, on a **single common DFT set spanning the deployment domain**, a fair **domain-aware fidelity
ranking of the original vs v5 student committees**, to resolve `CURRENT_STUDENT` and localize the
in-domain (SiOx-defect/vacancy) error to teacher vs distillation — the prerequisite for any teacher-fix
vs re-distill decision.

## 19. First scientifically necessary action (PROPOSED; not executed)

**`fair_domain_aware_student_comparison_on_common_dft_set`** — evaluate **both** committees (original + v5)
on the **28 al_iter3 SCAN cells** (postdate both students) with identical force-RMSE metric + domain masks,
producing per-domain original-vs-v5 fidelity. Typed proposal:
`examples/production_campaign_001/action_proposal.json`. **Requires new compute** (student committee
inference, CPU/LAMMPS, minutes; NO teacher/DFT/MD/training) → proposed, not executed.

## 20. Expected artifacts

`common_set_manifest.json`, `per_domain_student_comparison.csv` (original vs v5, per family),
`student_selection_verdict.json`, `criterion_results.json`, `provenance.json`, `run_manifest.json`.

## 21. Deterministic gates (four axes, grounded)

- **Artifact validity:** both committees = 4 distinct members; the 28 DFT cells parse; forces finite; the
  same cells/mask used for both students.
- **Fairness:** identical evaluation set + metric + mask for original and v5 (else REVISE).
- **Model accuracy:** per-domain force MAE (original vs v5), reference scale = teacher error_a per family.
- **Selection:** v5 preferred iff it lowers in-domain (SiOx-defect + vacancy) force MAE vs original without
  regressing bulk-amorphous; else original retained; if within noise → UNRESOLVED.

## 22. Human approval boundary

**Not required** — student CPU inference on existing structures/models is cheap and read-only-of-models
with outputs confined to a run dir; no teacher inference, DFT, MD, training, or scheduler. (Approval
remains mandatory for those later, costly actions.)

## 23. Resource estimate

2 committees × 4 seeds × 28 cells (~40–130 atoms) via LAMMPS `pair_style nn`, CPU, **minutes**; <1 GB;
no scheduler, no network.

## 24. Stop conditions

ADVANCE → emit the per-domain original-vs-v5 ranking + a CURRENT_STUDENT verdict (or UNRESOLVED-within-
noise). REVISE → fairness broken (mismatched cells/metric) or a committee member missing. FAIL → non-finite
predictions / atom-count mismatch. Stops at the verdict; triggers no downstream phase.

## 25. Remaining uncertainties

- **BLOCKING:** which student is best on a common in-domain DFT set (the first action).
- **NONBLOCKING:** v5 KISTI train manifest; absolute leakage-clean v5 generalization; ho_01/ho_08;
  original-student predictions on the 6 R1 cells.
- **V5_HELDOUT_LEAKAGE_PROVENANCE = IMPORTANT_BUT_NONBLOCKING** (§26 / rule 21): the first action is a
  *relative* comparison on the 28 al_iter3 cells that postdate both students → held-out from both by
  timing; the relative ranking is valid without the KISTI train_list. Leakage would block only an
  *absolute* leakage-clean generalization claim, which is not this action.

## 26. Reused existing assets vs proposed new work

- **Reused (no recompute):** teacher `b56e20ff`; error_a/b/c CSVs; the 28 al_iter3 DFT labels; EOS
  summaries; physical-validation results; both student committee checkpoints; R1 held-out artifacts (as
  reference).
- **Proposed new compute (first action only):** student committee inference of **both** committees on the
  28 common DFT cells (CPU, minutes). Nothing else is regenerated.

### Architecture-issue classification (rule 23)

- Fair student comparison via `evaluate_heldout_fidelity` (ml-trainer) = **MISSING_ACTION_ADAPTER** (a
  read-only student-eval adapter, action_type already in the frozen taxonomy) — build later under frozen
  contracts.
- No **GENUINE_ARCHITECTURE_BLOCKER**. No architecture change in this task.

## 27. Pipeline-order correction — Teacher validation is PC001's gate (EXECUTED)

The canonical scientific pipeline is: **scope / DFT reference → TEACHER VALIDATION → [gate: teacher
ACCEPTED?] → distillation dataset/labeling → student training → student validation → physical/MD →
coverage-failure / active learning.** Production Campaign 001 sits at **Teacher validation**; all
student-level work (including the original-vs-v5 comparison in §17–§19) is **downstream of the teacher
acceptance gate**, not the first action. §17's "primary bottleneck = MODEL_SELECTION" is therefore a
**student-validation-stage** concern; the actual PC001 first action is the **teacher-acceptance gate**.

**Teacher-validation gate — EXECUTED offline** (read-only from `error_a` — the teacher forward was already
run to produce it; NO new teacher inference), `work/production_campaign_001_teacher_validation.py` →
`runs/production_campaign_001/pc001-teacher-validation/`:

| Metric (in-scope: SiO2 + SiO2-x defects/vacancies/surfaces/liquid/ambient-crystal; cc001 artifact excluded) | Value |
|---|---|
| in-scope frames | 727 |
| teacher force MAE | **0.152 eV/Å** (≤ 0.20 bar; and < deployed-student in-domain ~0.23 ⇒ not the dominant limiter) |
| teacher energy MAE | **18.8 meV/atom** (≤ 50) |
| EOS ambient (α-quartz/coesite) | SMOOTH; B0 202/228 GPa |
| flagged sub-regions (force MAE > 1.5× in-scope mean) | SiOx_defect_dilute (0.23), SiOx_defect_clustered (0.35) |

**TEACHER_VERDICT = `ACCEPT_CONDITIONAL`** (FACT/DERIVED): accepted as the distillation teacher for the
in-scope domain (A1 absolute + A2 EOS + A3 relative-not-dominant-limiter all pass), **conditional** on the
elevated SiOx-defect sub-region (teacher force ~0.23–0.35 eV/Å) — flagged for monitoring and, if student
validation later shows it limiting, for the pipeline's coverage-failure / active-learning loop. The
verdict **does not block** proceeding; it bounds achievable student accuracy in the clustered-defect region.

**Re-sequenced next action (downstream, gated by this ACCEPT):** the original-vs-v5 student comparison
(`examples/production_campaign_001/action_proposal.json`) is the **student-validation** stage, to run
after the teacher-accept gate — not before. It remains prepared, not executed.
