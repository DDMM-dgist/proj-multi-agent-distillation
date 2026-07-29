# Mock end-to-end smoke test

This lightweight example uses ASE EMT as both teacher and mock student. It does
not represent a scientific distillation result. Its only purpose is to verify
fresh-clone package imports, teacher labeling, leakage-resistant splitting,
committee artifacts, held-out evaluation, gates, hashes, and resume behavior
without MACE, GRACE/FS, LAMMPS, or DFT.

It also validates a Teacher-first baseline and an explicit
`teacher_training_data_access: unavailable` coverage report. The mock therefore
tests the contract and audit path without claiming quantitative scientific
coverage.

The final mock stage also emits and validates a common structure-validation
report from the evaluated frames.

The mock calculator loads each generated checkpoint. Tests mutate one checkpoint
after a PASS and confirm that evaluation is blocked by both the committee tree
hash and the per-model integrity record.

The automated test drives the controller. Normal users start through Claude
Code with `/distill-start`; they do not need to run these commands manually.
