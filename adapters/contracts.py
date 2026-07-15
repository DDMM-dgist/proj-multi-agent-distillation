"""Architecture-neutral values exchanged between workflow stages."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ModelArtifact:
    """A trained model plus the minimum provenance needed by later stages."""

    kind: str
    path: Path
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def require_exists(self) -> "ModelArtifact":
        if not self.path.exists():
            raise FileNotFoundError(f"model artifact does not exist: {self.path}")
        return self


@dataclass(frozen=True)
class PredictionBatch:
    """Common prediction payload consumed by validation and label writers."""

    energies: np.ndarray
    forces: List[np.ndarray]
    stresses: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        if len(self.energies) != len(self.forces):
            raise ValueError("energies and forces must contain the same number of structures")
        if self.stresses is not None and len(self.stresses) != len(self.energies):
            raise ValueError("stresses must contain the same number of structures as energies")
