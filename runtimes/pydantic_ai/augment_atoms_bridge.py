"""Bridge callables imported by augment-atoms native data2objects configs."""
from __future__ import annotations

from pathlib import Path

from adapters import load_config
from adapters.teacher import load_teacher


def teacher_calculator(teacher_config: str):
    """Return the exact ASE calculator bound by a workflow Teacher config."""
    path = Path(teacher_config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Teacher config is missing: {path}")
    return load_teacher(load_config(path))
