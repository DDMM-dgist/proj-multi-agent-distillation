# Student training templates

These are built-in adapter examples, not universal defaults.

- `simple-nn.input.yaml.template`: rendered by the SIMPLE-NN recipe; descriptor
  parameters and wrapper compatibility remain case/environment inputs.
- `grace-fs.input.yaml.template`: placeholder for a version-matched input
  generated and reviewed with the installed GRACE/FS tooling.

An external student can use `adapter.train` or `train.command` and does not
need a template in this directory. Keep material-specific descriptors and
version-specific complete inputs under the run config or corresponding case.
