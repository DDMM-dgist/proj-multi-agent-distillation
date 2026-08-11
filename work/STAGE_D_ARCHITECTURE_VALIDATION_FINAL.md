# Stage D — Architecture Validation: Final Closure

**Status:**
`ARCHITECTURE_DESIGN = COMPLETE` ·
`ARCHITECTURE_VALIDATION = COMPLETE` ·
`REPRESENTATIVE_REAL_ACTION_VALIDATION = COMPLETE`

This document is the final closure of the PydanticAI multi-agent scientific-workflow **architecture**
validation. It is documentation only: no scientific computation, no teacher inference, no threshold
change, and no modification of the frozen architecture accompany it.

---

## 1. Objective

Validate that the multi-agent MLIP-distillation **workflow architecture** — typed/action-bounded
proposals, explicit human approval, trusted committed executors, deterministic (authoritative) vs
semantic (advisory) decision separation, and append-only provenance — is **complete and behaves as
specified on real scientific state advancement**, including at least one real GPU teacher-model
inference. The objective is **architecture** validation, explicitly **not** completion of the
production-scale distillation campaign (§12–13).

---

## 2. Frozen architecture

The architecture is pinned by a SHA-256 freeze guard (`tests/test_architecture_freeze.py`, revision
"deterministic-verdict-ownership-refactor") over the core files: `criterion_eval.py` (deterministic
criterion evaluator + bound verdict), role `judge.md`, `orchestration/exchange.py`, `models.py`,
`production_router.py`, `actions.py`, `tool_registry.py`, `controller.py`, and the role output schemas.
Verdict ownership is authoritative: **policy owns the verdict; the LLM only interprets**. No frozen file
was modified in Stage D-2 C3 or in this closure.

---

## 3. Stage A–C summary

- **A** — common runtime hardening: typed IO, provider config, failure provenance, retry, CLI.
- **B** — seven-role local smoke package (fixtures + runner + tests), OpenAI-compatible/vLLM local
  provider.
- **C** — role-specific typed tool registry (7 roles, allow-lists, read-only tools), typed
  `ActionProposal` + controller integration, tests/CI, golden shadow comparison, readiness gate.

These established the typed-message, role-scoped-tool, approval-gated substrate that Stage D exercises on
real artifacts.

---

## 4. Stage D-1 — replay / holdout validation — **COMPLETE**

Frozen scientific-workflow **shadow-replay** package: the deterministic gate + role decisions were
replayed against recorded coordination decisions. Dev-set authoritative replay reproduced the recorded
outcomes (7/7 AGREE at the frozen revision); the holdout remains gated. Stage D-1 frozen files are
unchanged by later stages (freeze guard green).

---

## 5. Stage D-2 C1 — real-action validation — **COMPLETE**

First real *state-advancing* scientific action through the full loop (proposal → approval → trusted
execution → deterministic validation → provenance): a **post-hoc MSD** computation.

- `AXIS_A = PASS` (deterministic artifact/computation validity)
- `SEMANTIC_JUDGE = REVISE` (advisory, non-authoritative)
- `FINAL_TRANSITION = REVISE`

C1 demonstrated the deterministic/advisory split on a real artifact: the authoritative axis passed while
the advisory Judge independently returned REVISE, and the two were not collapsed. C1 frozen artifacts are
unchanged.

---

## 6. Stage D-2 C3 — real-teacher validation — **COMPLETE**

Exactly one real teacher (Allegro/NequIP) single-point inference on one scientifically valid structure,
through proposal → explicit human approval → trusted committed adapter → exact compiled teacher → one real
GPU forward → new E/F artifact → deterministic Axis-A/B validation → append-only provenance.

- `real_teacher_gpu_prediction = COMPLETED` · `valid_prediction_generated = true`
- `model_forward_invoked = true` · `model_forward_completed = true`
- `AXIS_A = PASS` · `AXIS_B = FAIL`
- **Historical deterministic result: `D2_C3_TEACHER_SINGLE_POINT = FAIL` (unchanged).**

| Attempt-3 fact | value |
|---|---|
| teacher | `teacher_current_compiled.nequip.pth`, SHA `b56e20ff…`, cuda:1, torch 2.6.0 / nequip 0.16.1 |
| structure | `mini216_nvt_fixed.data`, SHA `3d2dd246…`, N=216 (O144/Si72) |
| E_total / E/atom | −1628.5079406630 eV / −7.5393886142 eV/atom |
| max\|F\| / wall time | 17.0860959523 eV/Å / 0.835 s |
| Axis-A / Axis-B / accepted | PASS / FAIL / false |

**Why the FAIL does not invalidate the architecture.** Architecture validation does **not** require every
scientific artifact to PASS. The C3 deterministic FAIL is positive evidence that the validation system
**rejects a scientifically suspect / out-of-domain artifact** instead of rubber-stamping a successful
computation. The full chain still executed correctly: typed/action-bounded proposal → explicit human
approval → trusted committed executor → exact scientific input/model identity → real GPU execution → new
scientific artifact → authoritative deterministic validation → **rejection when the preregistered
criterion failed** → append-only provenance → post-hoc scientific diagnosis without rewriting history.

---

## 7. Failure-mode chronology (nothing hidden)

The path to a completed real inference included real failures, each preserved:

| # | Event | Phase / class | Model state |
|---|---|---|---|
| 0 | GPU model materialization (FileNotFoundError) | failed closed **before load** | not loaded |
| 1 | Missing `pydantic` runtime dependency | failed **before scientific execution** (`ALLEGRO_ENV_MISSING_PYDANTIC`) | not invoked |
| 2 | Attempt 1 — NequIP 0.16.1 input-API mismatch (`with_edge_vectors` absent) | `EXECUTION_FAILED_BEFORE_FORWARD` | **not invoked** |
| 3 | Attempt 2 — CPU/CUDA device mismatch (`cuda:1` vs `cpu`) | raw coarse `BEFORE_FORWARD` preserved byte-identical **+ additive** `EXECUTION_FAILED_DURING_FORWARD` correction | **invoked, not completed** |
| 4 | Attempt 3 — success | `OK`; Axis-A PASS / Axis-B FAIL | **invoked & completed; valid E/F** |

Demonstrated across this chronology:
- **No automatic retry rewrote history** (`automatic_retry=false` throughout).
- **Each real execution attempt used a fresh run identity** (`…mini216`, `…-attempt2`, `…-attempt3`); the
  wrapper refuses reuse of any prior attempt's id/dir.
- **Approvals remained traceable** — each external approval validated read-only and snapshotted immutably
  into its run dir (attempt-3 snapshot SHA `0328b190…` = the external approval used).
- **Failed runs were preserved** append-only; Attempt-2's coarse raw classification was **corrected
  additively** (`CORRECTED_INTERPRETATION.json`), never rewritten.
- **Scientific artifacts are append-only**; **deterministic criteria and Axis-B thresholds were not
  changed after seeing any result.**

---

## 8. Safety / authorization properties demonstrated

- Costly/scientific actions are **approval-gated**: `approved=false` is never active; the wrapper
  fail-closes on wrong action/subtype, structure/teacher SHA mismatch, `authorizes_subsequent_actions ≠
  false`, or any limit violation (1 structure / 1 forward / 1 GPU, ≤60 s, no scheduler/MD/DFT/training/
  extra-labeling/paid-API/external-network/overwrite).
- Model + structure are **allow-listed and SHA-pinned** (immutability), verified before and after the run
  (`source_model_unchanged=true`).
- The forward comes **only** from the trusted committed adapter — no CLI/agent-supplied callable, no
  `eval`/`exec`, no `--forward`.
- Environment + model-load + input-build + **device-consistency** preflights all run **with zero model
  calls** and fail closed.
- Writes are **confined to the fresh run directory**; the executor refuses a pre-existing run dir.

---

## 9. Deterministic vs semantic decision separation

- **Deterministic (authoritative):** `criterion_eval` evaluates frozen criteria and **binds** the verdict;
  the LLM owns nothing. Axis-A (artifact/computation validity) and Axis-B (physical sanity) are computed
  from stored values with recorded provenance per criterion.
- **Semantic (advisory):** the Judge interpretation task is explicitly `deterministic_authoritative=false`
  and **never binds** Axis-A/B.
- Both real cases exercised the split: **C1** (Axis-A PASS, Judge REVISE) and **C3** (Axis-A PASS,
  Axis-B FAIL) — the authoritative verdict governed the transition; the advisory layer was recorded, not
  obeyed.

---

## 10. Provenance / append-only guarantees demonstrated

- Every run records `provenance.json` + `run_manifest.json` with package HEAD, external-approval path +
  SHA, immutable approval snapshot, load provenance (versions/device/type_names/r_max/dtype), per-artifact
  SHA-256, and the durable counters `model_forward_invoked` / `model_forward_completed` /
  `valid_prediction_generated`.
- Attempt-3 artifact hashes were independently re-verified against `provenance.json`; `forces.csv` holds
  exactly 216 rows in ascending atom order.
- History is **append-only**: attempts 1/2/3 coexist immutably; corrections are additive files.

---

## 11. C3 scientific mismatch audit — summary

Reference: `work/STAGE_D2_C3_ENERGY_MISMATCH_AUDIT.md`.

**Conclusion:** the exact C3 `mini216` structure is a **strained / non-equilibrium, high-symmetry**
configuration, **not** the equilibrium glass population that motivated the −11…−8 eV/atom reference band.
Supporting offline observations: **split Si–O coordination** (2 short ~1.54 Å + 2 long ~2.3 Å; bond range
1.40–2.58 Å), **distorted/long Si–O bonds**, a **high force distribution** (median 6.18, max 17.09 eV/Å,
44 % of atoms > 10), **limited coordinate diversity / symmetry** (30/18/36 unique x/y/z; symmetry-
equivalent Si groups), **mismatch with the expected parent equilibrium glass** (4/216 atoms coincide), and
an Attempt-3 E/atom (−7.54) **≈ +2.3 eV/atom above** the equilibrium reference population. Because energy
and forces come from the same forward, the large forces are an offline proof that the structure is far
from a minimum, so the teacher's high energy is physically correct and the Axis-B FAIL is scientifically
appropriate.

**The primary diagnosis (`NON_EQUILIBRIUM_HIGH_ENERGY_INPUT_STRUCTURE`) is sufficiently supported
offline.** The optional direct-adapter equilibrium-cell comparison (to also exclude a raw-`total_energy`
vs `NequIPCalculator` path offset) is **not required for architecture closure** and is retained only as
future optional scientific confirmation. **The historical Axis-B FAIL is not converted to PASS.**

---

## 12. Architecture limitations

- Validation is **representative, not exhaustive**: one MSD action (C1) and one teacher single-point (C3),
  plus replay/holdout (D-1). It demonstrates the contracts function; it does not certify exhaustive
  correctness of every role, tool, or edge case.
- Axis-B physical bounds are **equilibrium-scoped** (reused DFT-labelled equilibrium ranges); applying
  them to non-equilibrium inputs yields correct-but-domain-narrow rejections (see §11). Selecting
  scientifically in-domain inputs remains an application responsibility.
- The C3 input exposed a **data-provenance risk** (a distorted file sharing a name with a validated
  equilibrium structure). Input identity is SHA-pinned, but *scientific* representativeness of a chosen
  input is not automatically enforced by the architecture.
- The semantic Judge is advisory; its quality is not part of the authoritative guarantee.

---

## 13. Production-scale work explicitly OUTSIDE architecture validation

The following are **application / scaling** activities that **use** the completed architecture and are
**not** blockers to architecture completion — none is claimed validated here:

- large-scale teacher labeling (many structures / batches);
- student training at production scale;
- long / many MD campaigns;
- production active-learning loops (closed-loop iteration);
- large DFT batches;
- scheduler-scale (HPC queue) campaigns;
- any multi-artifact scientific-accuracy claim about the distilled potential.

Reaching these does not require further architecture changes; they are downstream usage.

---

## 14. Final completion statement

> The PydanticAI-based multi-agent scientific-workflow architecture and its execution contracts are
> complete and have been validated through replay, holdout, deterministic/advisory decision gates, and
> representative real scientific state advancement, including a real GPU teacher-model inference. This
> validation does **not** constitute completion of the full production-scale MLIP distillation campaign,
> and no claim of exhaustive correctness is made.

`ARCHITECTURE_DESIGN = COMPLETE` · `ARCHITECTURE_VALIDATION = COMPLETE` ·
`REPRESENTATIVE_REAL_ACTION_VALIDATION = COMPLETE`. Historical `D2_C3_TEACHER_SINGLE_POINT = FAIL`
preserved; architecture frozen.
