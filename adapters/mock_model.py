"""Checkpoint-aware calculator used only by the lightweight smoke test."""
import json
from pathlib import Path

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT


class MockCheckpointCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, checkpoint, **kwargs):
        super().__init__(**kwargs)
        payload = json.loads(Path(checkpoint).read_text())
        self.seed = int(payload["seed"])

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        probe = atoms.copy()
        probe.calc = EMT()
        self.results = {"energy": float(probe.get_potential_energy()) + self.seed * 1e-9,
                        "forces": probe.get_forces()}
