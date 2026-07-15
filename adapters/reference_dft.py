"""DFT reference adapter: render an INCAR (or equivalent) from a config +
overrides. Only renders the input — never fetches or writes a POTCAR (see
NOTICE.md); the caller supplies their own licensed copy.
"""
from pathlib import Path

from adapters import resolve_config_path


def render_incar(dft_cfg, out_path, overrides=None):
    kind = dft_cfg["kind"]
    if kind != "vasp":
        raise NotImplementedError(f"reference_dft kind={kind!r} is not implemented (only 'vasp' is).")

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
    if overrides:
        ctx.update(overrides)
    text = text.format(**ctx)
    Path(out_path).write_text(text)
    return out_path
