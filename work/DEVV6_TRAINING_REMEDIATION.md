# SMALL_002 training remediation (ML Trainer) — outcome

Real §16 remediation of the `training` stage failure. Fixes are PROVEN by smoke; two genuine
blockers to a literal same-campaign resume are reported honestly.

## A. Root-cause fix (PROVEN)
- Failure: `adapters/student.py train_student -> NotImplementedError: student kind='simple-nn-v2'`.
- Fix: use the supported `kind: simple-nn` + a real `train:` block (config recovered from
  `configs/examples/student.simple-nn.yaml`): `config_template=templates/student/simple-nn.input.yaml.template`,
  `descriptor_params` -> historical `gpu_return_v5/seed01_bundle_preview/params_{Si,O}` (70 SF = 16 G2 + 54 G4,
  cutoff 5.0), `nodes 30-30`, `runner.module=adapters.simple_nn_v2_wrapper`.
- **SMOKE #1 PASSED** (3 × 80-atom frames, epochs=1, `simple-nn` env): real `generate_features + preprocess
  (PCA/scale) + train` executed and produced `potential_saved_bestmodel`. The frozen PC003 science is
  unchanged; only the adapter schema mapping was fixed.

## B. struct_weight fix (§6) — PROVEN, was a real silent bug
- `adapters/simple_nn_v2_wrapper.py` `_write_structure_list` wrote a SINGLE unweighted tag -> per-structure
  `struct_weight` was silently dropped to 1.0.
- Fix (adapter wiring, no SIMPLE-NN core change): `--struct-weight-policy c_size_normalized_bounded` emits
  ONE weighted tag PER FRAME `[tag-i : w_i] / <dataset> i`, `w_i = clip((1/N_i)/geomean(1/N), 1/sqrt8, sqrt8)`;
  wired through `adapters/student.py _train_simple_nn` from `cfg.struct_weight_policy.name`.
- **ROUND-TRIP VERIFIED**: 2-atom frame -> 2.828, 141-atom -> 0.671, 3000-atom -> 0.354 (`struct_weights.json`
  ratio = 8.000). C_SIZE_NORMALIZED_BOUNDED now genuinely reaches SIMPLE-NN's structure_weights; validation
  stays unweighted (SIMPLE-NN `weighted=False if valid`).

## C. Environment (§7) — classified
- Intended `distill-student-simplenn` env: **ABSENT**. Only `simple-nn` env (torch 1.10 **CPU-only**;
  sklearn 1.6.1/scipy 1.13.1 vs SIMPLE-NN 2.0.0 pins). `braceexpand` installed.
- Drift classification from SMOKE #1: **NONBLOCKING** for generate_features + preprocess(PCA) + train.

## D. New data-quality issue (§16 -> Data Curator)
- SMOKE #2 (deliberately including a 2-atom frame) failed feature generation:
  `'Atom' object has no attribute 'cell'`. The PC002/DEV pool contains 2-atom corpus frames -> degenerate
  small cells break SIMPLE-NN feature generation. Remediation: Data Curator min-atoms / valid-cell filter
  before feature generation (does not change PC002 selection science; a preprocessing guard).

## E. Blockers to literal same-campaign resume (HONEST; earlier resume claim RETRACTED)
1. **Controller has no surgical resume-after-fix.** `rebind_inputs` explicitly "invalidates all prior stage
   results" -> fixing the student config (a hash-bound input) would rerun teacher_labeling/label_validation/
   dataset_split. The reproducibility model requires full re-derivation on any input/code change.
2. **git code-revision pin broken.** SMALL_002 pinned `git_commit 560c73c8` at init; persisting deliverables
   changed HEAD -> `verify_inputs` git guard now blocks every SMALL_002 `run-stage`.
3. The struct_weight fix is a CODE change (adapter) -> also triggers re-derivation.
=> A literal "resume SMALL_002, skip upstream, retry only training" is NOT supported by this controller's
   design; the sanctioned path is a fresh re-init of the same campaign identity on a clean committed tree
   (a new run_dir). RESUME_TEST(literal, in-place) = NOT_ACHIEVABLE (reported per §17, not faked).

## F. Remaining to run the full 4-seed DEV committee
config fix (done) + struct_weight fix (done) + Data-Curator degenerate-frame filter (new) + a controller
re-derivation (or direct dispatch) + CPU compute for 10 × ~2900-atom cells × 4 seeds × 200 epochs.
Real SIMPLE-NN training belongs on a version-correct/GPU backend via the trusted executor.
