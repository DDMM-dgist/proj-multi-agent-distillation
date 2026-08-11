# Stage D-2 C3 — Teacher Single-Point Energy-Mismatch Audit (offline; scientific diagnosis only)

**Scope.** Explain why the deployed teacher predicts **−7.5394 eV/atom** for the Attempt-3 structure
`mini216_nvt_fixed.data` when the prior teacher/DFT reference sits near **−9.8 eV/atom**. Architecture is
FROZEN and unchanged. Axis-B thresholds unchanged. No teacher inference, DFT, MD, training, or semantic
Judge was run. Every claim is tagged **FACT / DERIVED / INFERENCE / UNRESOLVED** with a source path.

Evidence root `RES = research-sio2-allegro-simplenn-distillation/`.

---

## 1. Executive diagnosis

**Primary cause — H1 (structure genuinely high-energy), SUPPORTED.** The exact file fed to C3
(`RES/teacher_diag/nve_drift/mini216_nvt_fixed.data`, SHA `3d2dd246…`) is **not** an equilibrium
amorphous a-SiO₂ glass. It is a **strained / distorted, high-symmetry, non-equilibrium** configuration:
every Si sits in a *split* coordination of **2 short Si–O (mean 1.54 Å) + 2 long Si–O (~2.3 Å)** — bond
lengths spanning 1.40–2.58 Å — versus the equilibrium glass's uniform **1.620 Å, CN(Si)=4.000**. The
teacher's own forces on this structure are **far from zero** (median **6.18**, max **17.09** eV/Å; 44 % of
atoms > 10 eV/Å), which is only possible for a structure **far from an energy minimum**. A configuration
that far from equilibrium legitimately carries a multi-eV/atom energy penalty, so the teacher's
−7.5394 eV/atom (≈ **+2.3 eV/atom** above equilibrium a-SiO₂) is a **physically correct** response.

**Contributing — H6 (provenance / wrong-file), SUPPORTED.** The on-disk file used for C3 is **not** the
structure STATUS.md validated as "REPRESENTATIVE amorphous a-SiO2" (which has Si–O 1.620 Å, CN 4.000,
near-zero pressure, forces→0 after 935 ps NVT). It shares the name `mini216_nvt_fixed.data` but differs
structurally, and it does **not** match the parent `glass216_300K_equil.data` in the same directory (only
4 / 216 atoms coincide). The distortion is **in the file on disk**, not introduced by the adapter.

**Retrospective Axis-B — H7, SUPPORTED.** The frozen range `−11 ≤ E/atom ≤ −8` was derived from
**equilibrium / relaxed DFT-labelled** cells and is an *equilibrium-structure* band. A strained
non-equilibrium cell correctly falls outside it. The deterministic **FAIL is scientifically appropriate**:
the gate flagged an out-of-equilibrium (out-of-family) artifact. **The historical FAIL stands.**

**Reference/convention mismatch — H2/H3/H4/H5, PLAUSIBLE-BUT-UNPROVEN and NOT required.** error_a's
`E_allegro` was produced from the **byte-identical** teacher (SHA `b56e20ff…`) via
`NequIPCalculator.get_potential_energy()` and matches DFT to **16 meV/atom**. The C3 path reads the raw
`out["total_energy"]` from a direct `torch.jit` call. A residual offset between the two extraction paths
cannot be *fully* excluded offline, but it is **unnecessary** to explain −7.5394: the structural strain
and large forces already account for the elevation.

---

## 2. Attempt-3 fixed facts (immutable; unchanged)  — **FACT**

Source: `runs/stage_d2_c3/d2c3-teacher-sp-mini216-attempt3/{teacher_ef,provenance,criterion_results}.json`.

| Quantity | Value |
|---|---|
| Teacher | `teacher_current_compiled.nequip.pth`, SHA `b56e20ff…`, device cuda:1, torch 2.6.0 / nequip 0.16.1 |
| Structure | `teacher_diag/nve_drift/mini216_nvt_fixed.data`, SHA `3d2dd246…`, N=216 (O144/Si72) |
| E_total | −1628.5079406630 eV |
| E/atom | −7.5393886142 eV/atom |
| force shape / max\|F\| | [216, 3] / 17.0860959523 eV/Å |
| inference wall time | 0.835 s |
| model_forward_invoked / completed / valid_prediction | true / true / true |
| Axis-A / Axis-B / verdict | **PASS / FAIL / FAIL** |

**Axis-B FAIL is preserved verbatim and is not reinterpreted.**

---

## 3. mini216 provenance  — **FACT** unless noted

- `RES/teacher_diag/nve_drift/` holds: `mini216_nvt_fixed.data` (SHA `3d2dd246…`, "**written by ASE**"),
  `glass216_300K_equil.data` (SHA `3f3fade2…`, LAMMPS `write_data`, timestep 935000), plus
  `nvt_equil_done.data`, `nve_final.data`.
- `nve_drift.in` (**FACT**): the NVE-drift run reads **`glass216_300K_equil.data`**, uses
  **`pair_style nn`** (the **SIMPLE-NN student** `01_potential_saved_bestmodel`, type 1=O, 2=Si) — **not**
  pair_allegro. So the LAMMPS energies in this directory are the **student's**, not the teacher's.
- STATUS.md (**FACT**): the "representative" structure lives at
  `05_SMALL_CELL_ERRORD/run_nvt_fixed/mini216_nvt_fixed.data`, validated by RDF/ADF/CN/MSD:
  **Si–O first peak 1.620 Å (FWHM 0.12), CN(Si)=4.000, O–Si–O 109.41°±5.24, near-zero pressure** →
  "REPRESENTATIVE amorphous a-SiO2". This is a **935 ps NVT-quenched equilibrium glass**, ρ = 2.20 g/cm³.
- **DERIVED:** the on-disk C3 file (SHA `3d2dd246…`) is **not** structurally equal to that validated glass
  (§9), and is **not** the parent `glass216_300K_equil.data` (only 4/216 atoms within 0.5 Å; 44 atoms
  >2 Å from any same-type parent atom).
- Existing recorded energies for the C3 file: **none** — no DFT, no direct teacher/Allegro, no
  pair_allegro energy exists for `mini216_nvt_fixed.data` anywhere in `RES` (grep). The only LAMMPS energy
  in this dir is the **student** on the *parent* glass: PotEng = −2253.33 eV → **−10.43 eV/atom**
  (`nve_drift/log.lammps`, `nve_thermo.txt`). **UNRESOLVED:** exact construction of the distorted C3 file.

> "Representative structure" was established for an **equilibrium glass**; it does **not** license the
> assumption that the specific on-disk C3 file is at a representative *absolute energy*.

---

## 4. Historical energy-reference dataset (~ −9.8 eV/atom)  — **FACT / DERIVED**

Source: `RES/teacher_diag/error_a_allegro_vs_dft.csv`, generator `RES/teacher_diag/run_task_a.py`.

- Columns: `idx, natoms, E_allegro_eV, E_dft_eV, dE_per_atom_meV, dE_per_atom_meV_shifted, Fmae_eV_A, config_type`.
- **1155 structures**, natoms 1–432, 24 config families (SiOx AL, bulk_cryst, bulk_amo, quench, liquid,
  surfaces, silicon_*, cluster, vacancy, …). Energies are **totals in eV**; per-atom = total/natoms.
- **Recomputed from raw values (DERIVED):**

| Quantity | mean | min | max |
|---|---|---|---|
| E_allegro/atom (eV) | **−9.8023** | −10.5459 | −0.7701 |
| E_dft/atom (eV) | −9.7862 | −10.5493 | −0.6649 |
| dE_per_atom RAW (meV) | **−16.05** (= −0.016 eV/atom) | −9112 | +410 |
| dE_per_atom SHIFTED (meV) | −0.000 | −9096 | +426 |

- **`E_allegro` is the raw column and already sits on the DFT absolute reference** (mean per-atom offset
  only **−16 meV/atom**; the "shifted" column merely removes that mean → 0). This is the exact basis of the
  earlier "raw teacher ≈ DFT reference" conclusion, **confirmed by recomputation**, not prose.
- Equilibrium family means (E_allegro/atom): bulk_cryst −10.40, bulk_amo −10.39, quench −10.11,
  vacancy_int_AL −10.16, liquid −9.79, SiOx_int_AL −9.84. **No family mean is near −7.54.** Only extreme
  `*_max_AL` tails and tiny clusters reach −7.5 (e.g. SiOx_max_AL max −7.557; cluster max −0.770).
- **DERIVED:** −7.5394 is **+2.30 eV/atom** above SiOx_int_AL and **+2.86** above bulk_cryst — i.e. deep in
  the high-energy tail that AL selection reserves for *deliberately non-equilibrium* configurations.

---

## 5. Teacher training / deployment energy semantics  — **FACT**

Source: `RES/gpu_finetune_handoff/config/base_allegro_train.yaml`.

- Loss `nequip.train.EnergyForceLoss`, **`per_atom_energy: true`**, target key **`total_energy`** (weight 1.0);
  metric `total_energy_mae`.
- Model head:
  - `per_type_energy_shifts: ${training_data_stats:per_atom_energy_mean}` (**baked**, `trainable: false`)
  - `per_type_energy_scales: ${training_data_stats:forces_rms}` (`trainable: false`)
  - `avg_num_neighbors: ${training_data_stats:num_neighbors_mean}`
- **INFERENCE:** the per-type shift equals the *training per-atom energy mean* (≈ −9.8 eV/atom scale).
  Because it is non-trainable and part of the model, a correctly-deployed graph's `total_energy` should be
  on the DFT absolute reference. Exact numeric per-type shift values are computed at train time from the
  training set statistics; **UNRESOLVED** — the resolved per-O / per-Si numbers are not stored in the
  committed config (they live in the training checkpoint on the GPU side, per `MODEL_NOTES.txt`).

---

## 6. Direct TorchScript output path  — **FACT**

Source: `runtimes/pydantic_ai/stage_d2_c3_teacher_adapter.py::build_forward_fn`.

```
data = build_model_input(...)              # pos/cell/pbc/atom_types + compute_neighborlist_ (nequip 0.16.1)
out  = self._model(data)                   # ONE torch.jit forward
energy = float(out[TOTAL_ENERGY_KEY].reshape(-1)[0].item())   # key = "total_energy"
forces = out[FORCE_KEY].detach().cpu().tolist()               # key = "forces"
E_per_atom = E_total / N                    # executor: -1628.508 / 216
```

- Output key used: **`total_energy`** (scalar), read directly; **no** summation, normalization,
  baseline add/subtract, or unit conversion in the adapter or executor. E/atom = E_total / N only.
- **INFERENCE:** this is the *raw graph* `total_energy`. error_a instead uses
  `NequIPCalculator.from_compiled_model(...).get_potential_energy()` (`run_task_a.py:30,61`) on the
  **same** model file. Both *should* read `total_energy`; whether the ASE calculator applies any extra
  handling vs the bare graph is **UNRESOLVED offline** (would need a same-structure two-path comparison, §13).

---

## 7. LAMMPS pair_allegro comparison  — **FACT**

- The NVE-drift LAMMPS run uses **`pair_style nn` (SIMPLE-NN student)**, not `pair_allegro`
  (`nve_drift.in`). There is therefore **no historical teacher-through-LAMMPS energy** for mini216 to
  compare against the direct TorchScript value. **No pair_allegro energy exists for this structure**
  (grep over `RES`).
- The only LAMMPS energy here is the **student on the parent equilibrium glass**: PotEng −10.43 eV/atom
  (`log.lammps`, `nve_thermo.txt` pe_per_atom −10.428). **DERIVED:** consistent with the −9.8…−10.4
  equilibrium a-SiO₂ scale — and **2.9 eV/atom below** the strained C3 file, reinforcing that the C3 file
  is the outlier, not the energy scale.
- **A direct-vs-pair_allegro comparison cannot be made offline** (no such record; a new run is forbidden).

---

## 8. Quantitative shift / reference analysis  — **DERIVED**

- Gap to explain: reference −9.80 → Attempt-3 −7.5394 ⇒ **ΔE ≈ +2.26 eV/atom** (less negative).
- A *fully missing* `per_atom_energy_mean` shift (≈ −9.8 eV/atom) would drive the output toward **≈ 0
  eV/atom**, not −7.54. The observed −7.54 is **inconsistent** with a cleanly missing/doubled global shift.
- Composition-weighted per-type shift for Si72O144: offset = (144·s_O + 72·s_Si)/216 = (2·s_O + s_Si)/3.
  The resolved s_O / s_Si are **not in the committed config** (§5) ⇒ **cannot be evaluated offline**; per
  the task rule, a shift explanation is admissible only if independently present in provenance — it is
  **not**, so H4 stays **UNPROVEN**.
- **INFERENCE:** the +2.26 eV/atom is better explained by **structural strain** (§1, §9, §10) than by any
  recorded energy-shift convention.

---

## 9. mini216 structural sanity  — **FACT** (network-free; no teacher)

Computed by MIC on the exact C3 file (SHA `3d2dd246…`); parent/validated references shown alongside.

| Metric | **C3 file `mini216_nvt_fixed.data`** | parent `glass216_300K_equil` | STATUS-validated glass |
|---|---|---|---|
| N / O / Si | 216 / 144 / 72 | 216 / 144 / 72 | 216 |
| box L / density | 14.8355 Å / **2.200 g/cm³** | 14.8355 Å / 2.200 | 2.20 |
| min dist any / Si–O / O–O | **1.404 / 1.404 / 1.768** | 1.552 / 1.552 / 2.394 | — |
| CN(Si) < 2.0 Å | **2.00** (all Si) | 4.00 | 4.000 |
| CN(Si) < 2.6 Å | 4.00 (**split 2 short + 2 long**) | 4.00 | 4.000 |
| Si–O bond mean (<2.6 Å) | **1.923** (range **1.404–2.582**) | ~1.60 | **1.620** (FWHM 0.12) |
| unique x / y / z values | **30 / 18 / 36** (54 columns) | 216 / 216 / 216 | (thermal) |
| symmetry-equivalent Si | **yes** (e.g. ids 146=147=149=150 identical NN sets) | no | no |
| atoms matching parent (<0.5 Å) | **4 / 216** | — | — |

- **Density and composition are correct**, but the **network is distorted**: split Si–O coordination
  (2×~1.54 Å + 2×~2.3 Å), short contacts (Si–O 1.40 Å, O–O 1.77 Å), and only 30/18/36 unique
  coordinate values with symmetry-equivalent Si groups (a thermal MD snapshot would have 216 distinct
  values per axis). Folding into (L/2)³ collapses to 63 unique positions ⇒ **partial internal symmetry**
  (not a clean 2×2×2, but far from disordered).
- **DERIVED:** the C3 file is a **strained, high-symmetry, non-equilibrium** structure — categorically
  different from the equilibrium glass STATUS validated (uniform 1.620 Å, CN 4.000). **This is not an
  adapter artifact:** the adapter fed the on-disk coordinates verbatim and the input-build preflight
  confirmed pos[216,3], cell diag 14.8355, PBC true, edge_index [2, 7872] (36.4 neigh/atom). **H6-in-code
  (conversion altered the config) is RULED OUT.**

---

## 10. Force-distribution analysis  — **FACT** (from the Attempt-3 `forces.csv`, no new run)

| stat | value (eV/Å) |
|---|---|
| min | 2.348 |
| median | **6.183** |
| mean | 9.427 |
| p90 / p95 / p99 | 17.086 / 17.086 / 17.086 |
| max | 17.086 (atom id 88, species **O**) |
| \|F\|>10 / \|F\|>5 | 96/216 (44 %) / 144/216 (67 %) |
| distinct \|F\| magnitudes | **8** (matches the structure's high symmetry) |

- **DERIVED:** an equilibrated 300 K glass has per-atom forces ~1–2 eV/Å. Median **6.18** and 44 % of atoms
  **>10 eV/Å** are **incompatible with equilibrium** and diagnostic of a strongly **strained/unrelaxed**
  structure. Only **8 distinct magnitudes** reflect the symmetry seen in §9.
- Because energy and forces come from the **same forward**, they share the model's internal scale: large
  forces ⇔ far from a minimum ⇔ **elevated energy**. This is an **offline** confirmation of H1 that does
  **not** depend on any reference-convention question.

---

## 11. Retrospective Axis-B applicability  — **assessment only; threshold unchanged**

- Provenance of `−11 ≤ E/atom ≤ −8`: reused verbatim from the Stage D-1 **DFT-labelled** physical-validity
  criteria; motivated by **relaxed/equilibrium** DFT cells (bulk_cryst −10.40, bulk_amo −10.39, AL cells
  −9.4…−10.5). It is an **equilibrium-structure energy band** used as **corruption / broad physical-sanity
  / out-of-family** detection.
- mini216 (the C3 file) is a **non-equilibrium** structure (§9–10) ⇒ it is **not** in the same
  thermodynamic population the band was built for. **POST-HOC applicability assessment:** the band is
  *equilibrium-specific*; applying it to a strained cell is a **domain stretch**, **but** the gate's
  outcome is still **correct** — it rejected a genuinely non-equilibrium, out-of-family artifact.
- **HISTORICAL_GATE_RESULT = FAIL (preregistered, authoritative, UNCHANGED).**
  **POST-HOC_CRITERION_APPLICABILITY = the band is equilibrium-scoped; the FAIL correctly identifies a
  non-equilibrium structure, so it is scientifically valid as a sanity/out-of-family flag.** No threshold
  change is made or recommended here.

---

## 12. Ranked root-cause hypotheses

| # | Hypothesis | Verdict | Key evidence |
|---|---|---|---|
| **H1** | mini216 is genuinely much higher energy (strained / non-equilibrium) | **SUPPORTED** | split Si–O (1.54/2.3 Å, range 1.40–2.58), CN 2 at 1.8 Å, forces median 6.18 / max 17.09 / 44 % >10 eV/Å (§9–10) |
| **H6** | wrong/distorted file selected — not the STATUS-validated equilibrium glass | **SUPPORTED** | C3 file ≠ validated glass (1.620 Å, CN 4.000) and ≠ parent (4/216 match); adapter did not alter it (§3, §9) |
| **H7** | Axis-B range valid for a narrower (equilibrium) domain than mini216 | **SUPPORTED** | band derived from relaxed DFT cells; mini216 non-equilibrium (§11) |
| **H2** | direct TorchScript uses a different energy baseline than NequIPCalculator | **PLAUSIBLE-BUT-UNPROVEN** | same model file; error_a uses `NequIPCalculator`, C3 uses raw `total_energy`; not separable offline (§6, §13) |
| **H3** | historical −9.8 were post-processed; Attempt-3 stored raw | **PLAUSIBLE-BUT-UNPROVEN** | error_a `E_allegro` is the *raw* column already on DFT scale (offset 16 meV) — argues **against**, not resolved (§4) |
| **H4** | per-type energy shift missing / double-applied | **UNPROVEN (not in provenance)** | a fully-missing shift → ≈0 eV/atom, not −7.54; resolved shift values absent from committed config (§5, §8) |
| **H5** | pair_allegro vs direct TorchScript convention differ | **RULED OUT (for this case)** | NVE run used `pair_style nn` (student); no teacher pair_allegro energy for mini216 exists (§7) |
| **H6-code** | adapter conversion / type-map / cell handling altered the config | **RULED OUT** | preflight verified pos[216,3]/cell/PBC/edge_index; positions read verbatim (§9) |

**No single reference-convention cause is forced.** The structural evidence (H1) is sufficient and offline-proven; H2/H3/H4 remain open but unnecessary.

---

## 13. Smallest decisive next test (if required) — **PROPOSAL ONLY; not executed**

The structural + force evidence already establishes H1 offline. The *only* residual question is whether a
raw-`total_energy` vs `NequIPCalculator` **path offset (H2)** also exists. Smallest test to separate them:

- **Hypothesis tested:** does the C3 direct adapter's raw `total_energy` agree with error_a's
  `NequIPCalculator.get_potential_energy()` on the **same equilibrium structure**?
- **Exact input:** one already-DFT-labelled **equilibrium** cell from error_a with a recorded
  `E_allegro/atom` ≈ −10.4 (e.g. a `quench` or `bulk_amo` frame), evaluated once through the committed
  TrustedAllegroAdapter (teacher `b56e20ff`, cuda:1).
- **Discriminating outcome:** raw ≈ −10.4 (within meV) ⇒ **no path offset**, H1 fully confirmed, H2 ruled
  out. raw ≈ −10.4 + 2.3 ⇒ a per-atom path offset exists ⇒ H2 operative.
- **Cost:** one forward, seconds, <1 GB GPU. **Human approval:** **required** (teacher inference class,
  identical to C3). *(An even cheaper offline half-step: evaluate the STATUS-validated equilibrium
  mini216 — but that still needs a forward and approval.)*
- **Recommendation:** **optional / confirmatory only.** Not needed to accept the primary diagnosis.

---

## 14. Final scientific conclusion

**The teacher is behaving correctly.** `mini216_nvt_fixed.data` as used by C3 is a **strained,
high-symmetry, non-equilibrium** a-SiO₂ configuration (split Si–O coordination 1.54/2.3 Å; forces median
6.18, up to 17.09 eV/Å; not the STATUS-validated equilibrium glass and not its parent frame). A structure
that far from a minimum genuinely carries a ~+2.3 eV/atom penalty, so the deployed teacher's
**−7.5394 eV/atom** is a **physically correct** prediction for *this* structure. The −9.8 eV/atom
reference is the **equilibrium** a-SiO₂/DFT scale; it is not the expected value for a distorted cell.

The dominant root cause is **H1 (genuinely higher-energy, non-equilibrium structure)**, with **H6
(a distorted / mis-selected file, not the validated glass)** as its origin and **H7 (an
equilibrium-scoped Axis-B band applied to a non-equilibrium cell)** explaining why the correct high energy
tripped the gate. A reference/convention offset (H2/H3/H4) is **not required and unproven**; H5 and the
adapter-conversion hypothesis are **ruled out**.

- **Main diagnosis:** non-equilibrium/strained structure → correct high teacher energy → correct Axis-B FAIL.
- **Unresolved:** (a) the exact origin of the distorted C3 file vs the validated equilibrium glass;
  (b) whether a residual raw-vs-NequIPCalculator path offset also exists (§13).
- **Further calculation needed?** **No** for the primary conclusion. **Optional** one-shot equilibrium
  re-evaluation (§13) to close H2, human-approved.

---

## 15. Architecture & Axis-B status  — **unchanged**

- **The Attempt-3 historical Axis-B result remains FAIL** under the preregistered criterion. Nothing in
  this audit alters it, the thresholds, the deterministic criteria, or any Attempt-1/2/3 artifact.
- **Architecture stays FROZEN.** The workflow already demonstrated proposal → approval → trusted real GPU
  inference → new scientific artifact → deterministic gate → append-only provenance, and here it
  **correctly rejected** an out-of-family artifact. This is a **scientific/reference-convention audit of
  the artifact**, not an architecture-design failure. No PydanticAI role, controller, authorization,
  trusted-executor, provenance, or Judge component is modified.
