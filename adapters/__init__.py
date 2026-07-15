"""Thin adapter layer: dispatch on a config's `kind` to the concrete model /
backend implementation. Agents and validation scripts import from here rather
than importing a specific teacher/student package directly — that's the whole
point of the adapter layer (see configs/README.md).

Each module exposes a small function surface; adding support for a new `kind`
means adding a branch here, not touching agents/ or validation/.
"""
from pathlib import Path

import yaml


def load_config(path):
    """Load one of the configs/*.yaml files."""
    path = Path(path).resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if "kind" not in cfg:
        raise ValueError(f"{path}: config must declare a `kind` (see configs/README.md)")
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    project = next((p for p in (path.parent, *path.parents)
                    if (p / "pyproject.toml").is_file()), path.parent)
    cfg["_project_dir"] = str(project)
    return cfg


def resolve_config_path(cfg, value):
    """Resolve repository-relative config paths independently of process cwd."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(cfg.get("_project_dir", cfg.get("_config_dir", "."))) / path).resolve()
