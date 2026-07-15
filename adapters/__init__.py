"""Thin adapter layer: dispatch on a config's `kind` to the concrete model /
backend implementation. Agents and validation scripts import from here rather
than importing a specific teacher/student package directly — that's the whole
point of the adapter layer (see configs/README.md).

Each module exposes a small function surface; adding support for a new `kind`
means adding a branch here, not touching agents/ or validation/.
"""
import yaml


def load_config(path):
    """Load one of the configs/*.yaml files."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "kind" not in cfg:
        raise ValueError(f"{path}: config must declare a `kind` (see configs/README.md)")
    return cfg
