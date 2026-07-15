# Mock end-to-end smoke test

This lightweight example uses ASE EMT as both teacher and mock student. It does
not represent a scientific distillation result. Its only purpose is to verify
fresh-clone package imports, teacher labeling, leakage-resistant splitting,
committee artifacts, held-out evaluation, gates, hashes, and resume behavior
without MACE, GRACE/FS, LAMMPS, or DFT.

The mock calculator loads each generated checkpoint. Tests mutate one checkpoint
after a PASS and confirm that evaluation is blocked by both the committee tree
hash and the per-model integrity record.

The automated test drives the controller. Normal users start through Claude
Code with `/distill-start`; they do not need to run these commands manually.
