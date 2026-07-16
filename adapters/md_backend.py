"""Config-selected MD input rendering and execution adapters."""
import importlib
import subprocess
from pathlib import Path

from adapters import resolve_config_path
from adapters.student import lammps_pair_style_block


def _callable(path):
    module_name, name = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), name, None)
    if not callable(value):
        raise TypeError(f"configured callable is invalid: {path}")
    return value


def render_input(md_cfg, student_cfg, checkpoint_path, template_name, context, out_path):
    renderer = md_cfg.get("adapter", {}).get("renderer")
    if renderer:
        return _callable(renderer)(md_cfg, student_cfg, checkpoint_path, template_name,
                                   context, out_path)
    if md_cfg["kind"] == "lammps":
        return render_lammps_input(md_cfg, student_cfg, checkpoint_path, template_name,
                                   context, out_path)
    raise NotImplementedError(f"MD backend {md_cfg['kind']!r} requires adapter.renderer")


def render_lammps_input(md_cfg, student_cfg, checkpoint_path, template_name, context, out_path):
    """Fill a templates/lammps/*.in.template with the pair_style block for the
    active student plus run-specific context (temperature, timestep, ...).

    template_name: e.g. "prod_md.in.template"
    context: dict of {placeholder: value} for simple str.format substitution —
        keep templates simple (str.format, not a full templating engine) so
        they stay readable as plain LAMMPS input files with a few {slots}.
    """
    template_path = resolve_config_path(md_cfg, md_cfg["template_dir"]) / template_name
    text = template_path.read_text()
    pair_block = lammps_pair_style_block(student_cfg, checkpoint_path)
    text = text.format(PAIR_STYLE_BLOCK=pair_block, **context)
    Path(out_path).write_text(text)
    return out_path


def run(md_cfg, input_path, run_dir, mpi_ranks=1):
    kind = md_cfg["kind"]
    runner = md_cfg.get("adapter", {}).get("runner")
    if runner:
        return _callable(runner)(md_cfg, input_path, run_dir, mpi_ranks=mpi_ranks)
    if kind != "lammps":
        raise NotImplementedError(f"MD backend {kind!r} requires adapter.runner")
    binary = md_cfg.get("binary", "lmp_mpi")
    cmd = ["mpirun", "-np", str(mpi_ranks), binary, "-in", str(input_path)] if mpi_ranks > 1 else [binary, "-in", str(input_path)]
    print(f"[md_backend:lammps] {' '.join(cmd)} (cwd={run_dir})")
    subprocess.run(cmd, check=True, cwd=run_dir)
