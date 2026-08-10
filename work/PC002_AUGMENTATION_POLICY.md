# PC002 augmentation policy (augment-atoms) — reference + future-final wiring

**Status for the DEV campaigns:** `AUGMENTATION_STATUS = BYPASSED_FOR_SMALL_DEVELOPMENT_CAMPAIGN`.
The 400-structure DEV campaigns (`SIO2_DISTILLATION_DEV_V6_SMALL_00{1,2}`) label frozen PC002-derived
SOURCE structures directly with v6; they do **not** run augment-atoms and must not be represented as
reproducing the full scientific distillation-data-generation protocol.

## 1. augment-atoms already exists in the runtime (Data Curator layer)
- `adapters/acquisition.py`: `run_augment_atoms(cfg, seed_path, out_path)` (version-agnostic wrapper that
  runs a configured command) + `acquire()` dispatch on `kind == "augment-atoms"`, followed by
  `validate_lineage()` requiring `parent_structure_id`.
- Config surface: `configs/templates/acquisition.yaml`; invoked as a Data-Curator **acquisition workflow
  stage** (see `examples/mock/workflow.yaml`; `distill-start` skill lists augment-atoms / teacher-md).
- **Gap (optional, not built now):** it is not yet a typed `ActionProposal` action in
  `runtimes/pydantic_ai/actions.py` / `tool_registry.py`. Wiring it there is a later enhancement; the
  acquisition-stage path is sufficient for the pipeline. (Do not redesign now.)

## 2. HISTORICAL_AUGMENTATION_POLICY (reference only — NOT an auto-authoritative target)
From `data_provenance/PROVENANCE.md §3`:
- method: **augment-atoms** (jla-gardner, arXiv:2506.10956, *rattle-relax-repeat*)
- seeds: `teacher/input_small.xyz` (50 normal-cell) + `input_large.xyz` (10 large-cell), from the DFT corpus
- output: **~8,000 normal + ~2,000 large = ~10,000** augmented frames
- labels: energy + forces (teacher/Allegro); **no stress, no dft_energy**; **DFT-anchor count = 0 by design**
  (augment-atoms relabels seeds with the teacher before augmenting)
- split ~90/10
- **not locally recoverable:** exact rattle amplitude / n_iterations / augment-atoms version (generated on
  KISTI; PROVENANCE flags "record augment-atoms version + exact config" as an open item)

## 3. FUTURE FINAL scientific campaign — explicit augmentation decision BEFORE labeling
Layered, separately-versioned dataset model (the frozen 5,552 source record is never modified):
```
PC002_SOURCE_ENSEMBLE            (5,552 frozen source structures)
  -> PC002_AUGMENTATION_DECISION (typed; Data Curator proposes, controller validates/dispatches)
  -> PC002_FINAL_DISTILLATION_ENSEMBLE  (source/replay + selectively-augmented + selected-MD)
  -> Final-Teacher labeling
```
Typed `augmentation_policy` decision:
```
augmentation_policy:
  mode: NONE | SELECTIVE | BROAD
  tool: augment-atoms
  target_domains: [...]        source_groups: [...]
  perturbation_parameters: {...}   requested_output_count: N
  rationale: "coverage/diversity/redundancy/size-driven; not 'augment everything'"
  provenance: {augment_atoms_version, config_sha, seed}
```
Per-augmented-structure record + deterministic pre-label validation:
`parent_structure_id, parent_structure_hash, augmentation_tool, augmentation_parameters, augmentation_seed,
derived_structure_hash, domain` ; checks: composition preserved, natoms (unless intended), valid cell,
**no pathological overlap / min-distance**, no NaN, exact-duplicate detection.

## 4. DFT policy unchanged (§7)
The 11 SCAN DFT cells (`PC004_INDEPENDENT_DFT_REFERENCE`) must **never** be augment-atoms parents,
never enter Student training, never be modified. If DFT training anchors are ever needed, select **new**
structures for DFT (do not consume the 11 benchmark cells).
