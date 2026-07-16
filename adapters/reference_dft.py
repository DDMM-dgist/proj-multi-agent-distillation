"""Reference-calculation input adapters selected by config."""
import importlib
from pathlib import Path

from adapters import resolve_config_path


def _callable(path):
    module_name, name = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), name, None)
    if not callable(value):
        raise TypeError(f"configured callable is invalid: {path}")
    return value


def render_reference_input(dft_cfg, out_path, overrides=None):
    renderer = dft_cfg.get("adapter", {}).get("renderer")
    if renderer:
        return _callable(renderer)(dft_cfg, out_path, overrides=overrides)
    if dft_cfg["kind"] == "vasp":
        return render_incar(dft_cfg, out_path, overrides)
    raise NotImplementedError(
        f"reference backend {dft_cfg['kind']!r} requires adapter.renderer"
    )


def render_incar(dft_cfg, out_path, overrides=None):
    """Built-in VASP input renderer; never fetches or writes POTCAR."""

    template_path = resolve_config_path(dft_cfg, dft_cfg["incar_template"])
    text = template_path.read_text()
    ctx = {
        "ENCUT": dft_cfg["encut_ev"],
        "KSPACING": dft_cfg["kspacing_inv_angstrom"],
        "ISMEAR": dft_cfg["smearing"]["ismear"],
        "SIGMA": dft_cfg["smearing"]["sigma"],
        "NSW": dft_cfg["relaxation"]["nsw"],
        "IBRION": dft_cfg["relaxation"]["ibrion"],
    }
    ctx.update(dft_cfg.get("template_variables", {}))
    if overrides:
        ctx.update(overrides)
    text = text.format(**ctx)
    Path(out_path).write_text(text)
    return out_path
