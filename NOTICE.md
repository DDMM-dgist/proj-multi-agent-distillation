# Notice — what you must supply yourself

This toolkit is code, prompts, and config **templates**. A few things are
deliberately **not** included because they are either licensed, proprietary, or
too instance-specific to ship:

## VASP `POTCAR` files
`templates/dft/INCAR.scan.template` is an `INCAR` recipe only. VASP
pseudopotential (`POTCAR`) files are licensed per-user by VASP Software GmbH
and **must not be redistributed**. Obtain them through your own VASP license
and place them alongside the rendered `INCAR`/`POSCAR`/`KPOINTS`.

If you use a different DFT code (QE, CP2K, ...), write a new
`configs/reference_dft.<yours>.yaml` and a corresponding template; see
`configs/README.md` for the interface.

## Trained model weights
No teacher checkpoint or student `potential_saved_bestmodel` is included.
- **Teacher weights**: bring your own (a pretrained foundation MLIP, a model you
  trained, or one obtained from its original authors under their license/terms).
  If you are adapting the SiO₂/Allegro reference case, the teacher checkpoint
  used there was obtained from a prior study under its own terms — confirm
  redistribution rights before sharing it further.
- **Student weights**: the whole point of this workflow is that `ml-trainer`
  trains these for you from your teacher + data; there's nothing to ship.

## Raw datasets / full run artifacts
The full SiO₂ dataset, DFT-labeled pool, production trajectories, and result
CSVs from the reference run are **not duplicated here** — they live in the
original research repository referenced in `examples/sio2_allegro_simplenn/README.md`.
This toolkit ships the *recipes* (configs, templates, scripts) to regenerate an
analogous set for your own system, not the SiO₂ data itself.

## If you redistribute a fork of this toolkit
Keep this file, update it with anything *you* choose not to include (e.g. your
own teacher checkpoint's license terms), and do not commit `POTCAR`,
`*.pth`/`*.pt` weight files, or large trajectory dumps — see `.gitignore`.
