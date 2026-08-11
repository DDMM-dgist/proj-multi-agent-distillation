# Stage D-2 C3 — real-model PREFLIGHT (no forward pass; no E/F artifact)

Preflight before the first real teacher inference. **No structure was fed through the model; no
forward pass; no `teacher_ef.json`.** Model-load-only + source-grounded identity/energy checks.

## 1. Teacher model identity (source-grounded)

```
D2_C3_TEACHER_IDENTITY_UNRESOLVED = false
```

- **What it is:** a **compiled/deployed TorchScript, deploy-only** copy of the **CURRENT teacher,
  PRE-fine-tune** — i.e. the **base/original KISTI Allegro teacher** (NOT fine-tuned, NOT a training
  checkpoint).
- **Evidence (not the filename):**
  - `gpu_finetune_handoff/HANDOFF.md`: "`teacher_current_compiled.nequip.pth` ← CURRENT deployed
    teacher (**pre-fine-tune**)"; "a compiled TorchScript deploy-only artifact — it is NOT the training
    checkpoint and CANNOT be used [for fine-tuning restart]."
  - `gpu_finetune_handoff/models/MODEL_NOTES.txt`: "COMPILED/DEPLOYED NequIP model (TorchScript). Used
    by LAMMPS (pair_allegro) for production MD." Source = `materials-ml-kit/teacher/model.nequip.pth`.
- **Training-data identity (config template):** `config/base_allegro_train.yaml` →
  `training_subset_fixed_energy_key_fixed.xyz` (KISTI SiO2 set); loss `EnergyForceLoss` with
  `per_atom_energy: true`, `total_energy` + `forces` targets; `per_type_energy_shifts =
  training_data_stats:per_atom_energy_mean` (trainable=false). (The exact KISTI `.hydra/config.yaml`
  is a KISTI-side asset; the compiled model's own metadata is authoritative below.)
- **Architecture / metadata (from the MODEL itself, torch.jit metadata + buffers):**
  `allegro.model.AllegroModel`; **type_names = [O, Si]**; **r_max (cutoff) = 5.0 Å**;
  **model_dtype = float32**; buffers show bessel radial embed + 4 Allegro latent MLP layers + tps(w3j).
- **Creation:** file mtime 2026-06-19 23:27.

## 2. Model SHA / immutability

```
path  = …/gpu_finetune_handoff/models/teacher_current_compiled.nequip.pth
size  = 4,905,990 bytes
SHA256 = b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57   (unchanged)
```
Read-only; the trusted adapter re-verifies this SHA (fail-closed) before every load/execution.

## 3–4. Loader / adapter + model-load preflight (NO forward)

- **Trusted path:** `trusted executor → TrustedAllegroAdapter → compiled model → one structure → E/F`.
  `runtimes/pydantic_ai/stage_d2_c3_teacher_adapter.py::TrustedAllegroAdapter` is committed,
  deterministic, sha+allow-list guarded; it **constructs** the forward callable
  (`build_forward_fn()`) — no arbitrary agent-supplied Python. torch/nequip are imported lazily; no
  forward at import/preflight.
- **Model-load-only preflight** (`work/stage_d2_c3_model_load_preflight.py`, run in env `allegro`):

  | field | value |
  |---|---|
  | env | conda `allegro` |
  | python | 3.10.16 |
  | torch | 2.5.1+cu124 |
  | nequip | 0.16.1 |
  | allegro | 0.2.0 |
  | cuda_available (this host) | false (CPU-only preflight host) |
  | model_load | **SUCCESS** (`torch.jit.load`, cpu, **no forward**) |
  | type_names | **[O, Si]** |
  | r_max | 5.0 |
  | model_dtype | float32 |
  | required_input_fields | pos, cell, pbc, atom_types, edge_index |

  (On the GPU host, re-run with `--device cuda:<id>` to also confirm GPU load — still no forward pass.)

## 5. Species / type mapping (explicit; fail-closed)

Confirmed from the **model's own** `type_names = [O, Si]` (index 0=O, 1=Si) — not assumed from numeric
LAMMPS types. The C3 conversion maps by **chemical symbol**:
```
LAMMPS type 1 -> "O"  -> model index 0   (verified)
LAMMPS type 2 -> "Si" -> model index 1   (verified)
```
`TrustedAllegroAdapter.map_lammps_types` fails closed on: unknown LAMMPS type, unexpected species,
atom-count mismatch; a reversed symbol map is applied literally so the reversal is auditable in the
recorded `atom_type_index`.

## 6. Structure conversion contract (`mini216_nvt_fixed.data` → model input)

Recorded by `TrustedAllegroAdapter.structure_conversion_contract` (no relaxation / no coordinate
modification): **positions** as read (ordered by LAMMPS id ascending); **cell** = cubic
`diag(L, L, L)`, **L ≈ 14.8355 Å**; **PBC** = [True, True, True]; **species** via the mapping above;
**units** Å / eV / eV·Å⁻¹. Preserves **N = 216, O = 144, Si = 72**.

## 7. Energy-reference compatibility (source-grounded → range retained)

From `teacher_diag/error_a_allegro_vs_dft.csv` (direct `E_allegro` vs `E_dft`, 1155 frames):
- mean **raw** offset `(E_allegro − E_dft)/atom = −0.0160 eV/atom` (~16 meV — negligible);
  raw per-atom |ΔE| ≈ 27 meV; `E_allegro`/atom mean −9.802 vs `E_dft`/atom −9.786.
- The deployed teacher's **raw** total energy is on the **DFT reference** (`per_type_energy_shifts`
  reconstruct the DFT-scale absolute energy; no baseline REMOVAL — the shifts are added back). Training
  target is total energy in eV; deployed output retains it. The `_shifted` error column removes only the
  ~16 meV residual (it does NOT indicate a large incomparable offset).

**Conclusion:** absolute-energy comparability **CONFIRMED**. Retain the reused frozen criteria as
authoritative for this deployed model output:
- `E_per_atom_eV ∈ [-11, -8]` (DFT-scale; a-SiO2 sits at ~−9.7, well inside);
- `max|F| ≤ 50 eV/Å` (forces are reference-independent; LAMMPS/pair_allegro + error_a use eV/Å →
  unit-confirmed).
No conversion invented; no threshold invented to force PASS.

## 8. GPU resource preflight

This preflight host is **CPU-only** (`cuda_available=false`), so model-load memory on GPU was not
measured here — the model **loads successfully on CPU** (4.9 MB TorchScript). Expected at execution
(216 atoms, one forward pass): **< 1 GB GPU**, seconds. The eventual execution selects **one explicit
GPU** (e.g. `cuda:1`), measures free memory before load, and **must not disturb or signal existing VASP
processes**; **no scheduler**. GPU model-load memory is measured on the GPU host at execution
(`--device cuda:<id>`, still no forward).

## 9. Tests

`tests/test_stage_d2_c3_adapter.py` (+ existing `tests/test_stage_d2_c3_teacher_prep.py`) — network-free:
allow-list + sha identity, type1=O/type2=Si mapping, fail-closed species, cell/PBC/atom-count
preservation, forward only from a loaded trusted adapter (no arbitrary callable; the generic executor's
default `forward_fn=None` raises), one-forward/no-side-jobs execution contract, and (torch-gated) the
real model-load metadata. No real mini216 prediction.

## Verdict

- `D2_C3_TEACHER_IDENTITY_UNRESOLVED = false` (base/pre-fine-tune KISTI Allegro, compiled deploy-only).
- `[-11,-8] eV/atom` and `max|F| ≤ 50 eV/Å` are **scientifically valid** for this deployed model output
  (energy reference + force units source-confirmed) → kept authoritative.
- Loader/adapter, species mapping, and conversion contract validated **without** a forward pass.
- Remaining execution-time item: select one GPU, measure model-load memory, then one Allegro forward
  pass (still gated on the existing explicit `costly_teacher_labeling` approval).

## Exact eventual single-forward command (run only after approval; NOT run now)

```
# on the GPU host, env=allegro, one selected GPU, NO scheduler, does not disturb VASP:
conda run -n allegro python work/stage_d2_c3_model_load_preflight.py --device cuda:1   # (load-mem check, still NO forward)
# then the approved one-forward execution (writes runs/stage_d2_c3/d2c3-teacher-sp-mini216/*), using
# TrustedAllegroAdapter.build_forward_fn() as the trusted forward source injected into
# runtimes.pydantic_ai.stage_d2_c3_teacher_executor.run_teacher_single_point (one structure, one pass).
```
