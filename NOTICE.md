# Notice — what you must supply yourself

This toolkit is code, prompts, and config **templates**. A few things are
deliberately **not** included because they are either licensed, proprietary, or
too instance-specific to ship:

## Licensed reference-calculation assets
Pseudopotentials, basis sets, and proprietary executables are not distributed.
Supply them under the terms of the selected reference backend. The built-in
VASP example renders an INCAR only; it never fetches or writes a `POTCAR`.
Other codes connect through the reference adapter contract in
`configs/README.md`.

## Trained model weights
No teacher checkpoint or student `potential_saved_bestmodel` is included.
- **Teacher weights**: bring your own (a pretrained foundation MLIP, a model you
  trained, or one obtained from its original authors under their license/terms).
  Each case may have different redistribution terms; confirm them before
  sharing a checkpoint.
- **Student weights**: the whole point of this workflow is that `ml-trainer`
  trains these for you from your teacher + data; there's nothing to ship.

## Raw datasets / full run artifacts
Production datasets, reference labels, trajectories, and run artifacts are not
bundled. Case READMEs describe any separately managed research data. This
repository ships workflow code, contracts, and small examples only.

## If you redistribute a fork of this toolkit
Keep this file, update it with anything *you* choose not to include (e.g. your
own teacher checkpoint's license terms), and do not commit `POTCAR`,
`*.pth`/`*.pt` weight files, or large trajectory dumps — see `.gitignore`.
