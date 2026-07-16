# Worked example: SiO2, Allegro → SIMPLE-NN

This directory documents the SiO2 reference case from which the first workflow
recipes were generalized. It is a case record, not a default for new runs.

## What's here vs. what isn't

The concrete configs for this instance are `configs/examples/teacher.allegro.yaml`,
`configs/examples/student.simple-nn.yaml`, `configs/examples/uncertainty.yaml`,
`configs/examples/md_backend.yaml`, `configs/examples/reference_dft.yaml`, and
this directory's `configs/validation_profile.yaml`. Use them as inputs when
bootstrapping a run-specific config directory; do not treat them as generic
defaults.

**Not duplicated here** (too large / instance-specific / licensed):
- The teacher checkpoint, student committee checkpoints.
- The full dataset (10k+ augmented structures, the DFT-labeled seed pool, the
  uncertainty-selected DFT anchor cells).
- Production MD trajectories and the full result-CSV audit trail.
- `POTCAR` files (see `NOTICE.md`).

## Porting to your own system

1. Start through `/distill-start`. Reuse only the adapter examples that match
   the selected tools and create a validation profile from the target material
   and deployment domain rather than modifying the silica profile in place.
2. If the selected teacher or student is not a built-in recipe, provide the
   config-selected factory/callable/command described in `configs/README.md`;
   do not add a model-name branch to the controller.
