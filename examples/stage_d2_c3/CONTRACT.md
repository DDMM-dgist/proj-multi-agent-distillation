# Stage D-2 C3 — teacher single-point (E/F) execution contract (PREPARATION; not executed)

One real teacher (Allegro) single-point on one scientifically valid structure → a genuinely NEW E/F
artifact, through proposal → approval → trusted execution → deterministic validation → provenance.
**Nothing is executed; no model is loaded; no result files exist; no run dir is created.**

## Selected structure

- `…/teacher_diag/nve_drift/mini216_nvt_fixed.data` · SHA256 `3d2dd2464d83ca144e2c6d51382b83546b1152da06a386bfe3672550f4348364` · 18,098 bytes
- **216 atoms** — O ×144, Si ×72 (SiO2, 2:1); cubic box L = 14.8355 Å; LAMMPS `atomic` data
- Gate-confirmed **representative amorphous a-SiO2** (error(d) mini-cell). **Not already teacher/Allegro-labeled** (search empty) → the artifact is genuinely new. No new MD/DFT required (structure already exists).

## Selected teacher model

- `…/gpu_finetune_handoff/models/teacher_current_compiled.nequip.pth` · SHA256 `b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57` · 4,905,990 bytes
- nequip/allegro (compiled `.nequip.pth`), cutoff 5.0 Å, chemical_symbols `[O, Si]` (matches type 1→O, 2→Si). **Software env:** nequip 0.15 / allegro 0.7.1 + torch.
- Version note (for the approver): this is `teacher_current_compiled`; confirm base-vs-finetuned teacher identity before execution if it matters to the scientific claim.

## Compute / runtime / GPU

One forward pass on 216 atoms: **< 1 GB GPU, seconds** (ceiling ≤ 1 min). One explicitly selected RTX 6000 Ada. **No scheduler** (direct), no training, no MD, no DFT, no paid API, no network.

## Output E/F schema (`teacher_ef.json` + `forces.csv`)

`source_structure` + `source_sha256`; `teacher_model` + `model_sha256`; `n_atoms`; `composition`;
`predicted_total_energy_eV`; `energy_per_atom_eV`; `forces_artifact` (forces.csv, id,fx,fy,fz N×3) +
`max_force_eV_A`; `units` (eV, eV/atom, eV/Å, Å); `inference_metadata` (cutoff, type_symbol_map,
one_forward_pass, device, runtime_s); provenance in `provenance.json`.

## A. Deterministic ARTIFACT/COMPUTATION validity (authoritative)

`criteria/teacher_ef_validity.json`, frozen operators, evaluated by the frozen `criterion_eval` (bound
verdict; LLM owns nothing). Invalidating → FAIL: input/model SHA match, structure parsed, atom count
preserved (216), energy finite, E/atom finite, forces finite, **force shape N×3**, max|F| finite,
source+model unchanged, writes only under the run dir. Completeness → REVISE: artifact hashes recorded.

## B. Physical validity — REUSED frozen SiO2 criteria (provenance documented)

Reused **verbatim** from the Stage D-1 DFT-label physical-validity criteria
(`examples/stage_d1_replay/criteria/d1-dft-*.json`): **`E_per_atom_eV ∈ [-11, -8]`** and
**`max|F| ≤ 50 eV/Å`** (both invalidating).

**Provenance for the reuse:** these are physical-validity ranges for **DFT-scale SiO2 total energy per
atom and force**. The Allegro teacher was trained to reproduce **DFT** energies/forces for SiO2, so its
predicted E/atom is on the same DFT scale; the DFT-labeled AL cells sit at **−9.4…−9.9 eV/atom**, well
inside `[-11,-8]`, and `max|F| ≤ 50` is an atom-overlap sanity bound independent of method. The ranges
are therefore scientifically applicable to the teacher's output on the same SiO2 domain. **No threshold
was invented to obtain PASS.** Artifact validity (A) is kept separate from physical validity (B); a
higher-level scientific reading is (C) advisory only.

## C. Optional advisory semantic Judge (`deterministic_authoritative=false`)

`judge_interpretation_task.json` — reads ONLY `runs/stage_d2_c3/<run_id>/teacher_ef.json` (full
repo-relative path; no bare filename; no manifest): is the predicted E/atom consistent with a
representative a-SiO2 single-point (comparable to the DFT-labeled AL cells) and are the forces
physically reasonable? Grounded only in the artifact; no invented threshold; never binds A/B. Optional.

## Artifacts — PLANNED vs GENERATED

PLANNED (committed): `action_proposal.json` (validated vs frozen `DataCuratorActionProposal`),
`input_manifest.json`, `model_manifest.json`, `criteria/teacher_ef_validity.json`,
`judge_interpretation_task.json`, `run_manifest.template.json`, `CONTRACT.md`.
GENERATED at execution (not created): `approval.json`, `teacher_ef.json`, `forces.csv`,
`criterion_results.json`, `provenance.json`, `run_manifest.json` under `runs/stage_d2_c3/<run_id>/`.

## Authorization / trusted executor

- Proposal role **data-curator**; `action_type = label_with_teacher` (frozen; existing allow-listed
  approval-gated action), `approval_boundary = costly_teacher_labeling`, `parameters.subtype =
  teacher_single_point`. **Explicit human approval required**; preparing ≠ approving.
- Trusted executor `runtimes/pydantic_ai/stage_d2_c3_teacher_executor.py::run_teacher_single_point` —
  model-agnostic; enforces approval, fresh run dir (no overwrite), source+model allow-list + SHA256,
  N×3 force shape + finite checks, writes only under the run dir. The Allegro forward pass is an
  **injected `forward_fn`** supplied only at approved execution. **Not invoked in preparation** (default
  `forward_fn=None` raises).

## Idempotency / mutation / rollback

Source structure + teacher model + entire Stage D-1 tree **read-only**; the executor refuses an existing
run dir; the structure is confirmed to carry **no existing teacher E/F artifact**; no writes outside the
run dir; on any failure the run dir is removed and no partial artifact is accepted as final.

## Stop conditions

- **ADVANCE (artifact accepted)** only if: authoritative A+B validity = PASS, within the ceiling,
  source+model unchanged, all writes isolated, provenance complete. (Advisory C, if run, is descriptive.)
- **REVISE** if the validity result is REVISE-class (a completeness gap).
- **STOP/FAIL** if: any invalidating A/B criterion fails (non-finite, wrong shape, unphysical E/F, SHA
  mismatch); run dir exists; source/model mutated; approval absent; write attempted outside the run dir.

## Exact action after explicit approval

Write `runs/stage_d2_c3/d2c3-teacher-sp-mini216/approval.json`, then run `run_teacher_single_point`
once on one selected GPU: verify approval + fresh run dir + source/model SHA → parse mini216 (216
atoms) → **one Allegro forward pass** (injected `forward_fn` wrapping the compiled teacher) → write
`teacher_ef.json` + `forces.csv` → record A+B validity → evaluate the authoritative validity gate
(frozen `criterion_eval`, bound verdict) → optionally the advisory Judge → write `criterion_results.json`,
`provenance.json`, `run_manifest.json`. No scheduler, no MD/DFT/training, no automatic follow-up labeling.
