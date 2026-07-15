"""MD backend adapter: render an input deck from a template + run it.
`kind: lammps` is implemented; `kind: ase-md` would be a pure-python
alternative for small cells and is a documented stub.
"""
import subprocess
from pathlib import Path

from adapters.student import lammps_pair_style_block


def render_lammps_input(md_cfg, student_cfg, checkpoint_path, template_name, context, out_path):
    """Fill a templates/lammps/*.in.template with the pair_style block for the
    active student plus run-specific context (temperature, timestep, ...).

    template_name: e.g. "prod_md.in.template"
    context: dict of {placeholder: value} for simple str.format substitution —
        keep templates simple (str.format, not a full templating engine) so
        they stay readable as plain LAMMPS input files with a few {slots}.
    """
    template_path = Path(md_cfg["template_dir"]) / template_name
    text = template_path.read_text()
    pair_block = lammps_pair_style_block(student_cfg, checkpoint_path)
    text = text.format(PAIR_STYLE_BLOCK=pair_block, **context)
    Path(out_path).write_text(text)
    return out_path


def run(md_cfg, input_path, run_dir, mpi_ranks=1):
    kind = md_cfg["kind"]
    if kind != "lammps":
        raise NotImplementedError(f"md_backend kind={kind!r} is not implemented (only 'lammps' is).")
    binary = md_cfg.get("binary", "lmp_mpi")
    cmd = ["mpirun", "-np", str(mpi_ranks), binary, "-in", str(input_path)] if mpi_ranks > 1 else [binary, "-in", str(input_path)]
    print(f"[md_backend:lammps] {' '.join(cmd)} (cwd={run_dir})")
    subprocess.run(cmd, check=True, cwd=run_dir)
