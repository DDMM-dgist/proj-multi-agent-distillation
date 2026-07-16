"""Thin adapter layer for config-selected model and backend implementations.
Agents and validation scripts import from here rather than importing a specific
teacher/student package directly (see configs/README.md).

Each interface accepts configured callables/commands. Built-in recipes may use
local registries, but a new external `kind` does not require a controller or
agent-prompt branch.
"""
from pathlib import Path

import yaml


def load_config(path):
    """Load one of the configs/*.yaml files."""
    path = Path(path).resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must contain a YAML mapping")
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
