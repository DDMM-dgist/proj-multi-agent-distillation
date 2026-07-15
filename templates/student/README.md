# templates/student/

`simple-nn.input.yaml.template` — training config template for the `kind:
simple-nn` student adapter (`adapters/student.py:_train_simple_nn`). Verify
its keys against your installed SIMPLE-NN version before relying on it (the
schema has changed across releases).

## `params_Si` / `params_O` (descriptor/symmetry-function basis) — not included here

These are per-element numeric symmetry-function definitions (radial + angular
Gaussian basis parameters, ~70 functions/element in the reference case) — they
are a genuine hyperparameter choice, not licensed data, but they are also
fairly specific to the element set and cutoff you're targeting, so a generic
placeholder here would be misleading rather than useful.

To get a working pair for your own elements: either (a) regenerate them with
SIMPLE-NN's own descriptor-generation tool for your element set and desired
cutoff, or (b) if you are reproducing the SiO2 reference case specifically,
obtain them from the original research repository (see
`examples/sio2_allegro_simplenn/README.md`).
