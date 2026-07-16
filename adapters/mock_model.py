"""Checkpoint-aware calculator used only by the lightweight smoke test."""
import json
from pathlib import Path

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.io import read, write

from adapters.contracts import ModelArtifact


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


def train_external_adapter(cfg, dataset_path, out_dir, seed):
    """Test adapter proving an unknown student kind needs no core branch."""
    path = Path(out_dir) / "external-model.json"
    path.write_text(json.dumps({"seed": int(seed)}))
    return ModelArtifact(kind=cfg["kind"], path=path, seed=int(seed))


def load_external_adapter(cfg, checkpoint):
    return Path(checkpoint)


def deploy_external_adapter(cfg, checkpoint):
    elements = " ".join(cfg["deploy"]["elements"])
    return f"pair_style external\npair_coeff * * {checkpoint} {elements}\n"


def preflight_external_adapter(cfg, check_files=True, require_ready=False):
    if cfg.get("adapter_test_token") != "ok":
        raise ValueError("external adapter test token is missing")
    return ["external adapter ready"]


def render_external_md(md_cfg, student_cfg, checkpoint, template_name, context, out_path):
    Path(out_path).write_text(f"checkpoint={checkpoint}\n")
    return out_path


def run_external_md(md_cfg, input_path, run_dir, mpi_ranks=1):
    return {"input": str(input_path), "run_dir": str(run_dir), "mpi_ranks": mpi_ranks}


def render_external_reference(cfg, out_path, overrides=None):
    Path(out_path).write_text(json.dumps({"kind": cfg["kind"],
                                         "overrides": overrides or {}}))
    return out_path


def acquire_external_adapter(cfg, teacher_cfg, seed_path, out_path):
    frames = read(seed_path, index=":")
    for index, atoms in enumerate(frames):
        atoms.info.setdefault("parent_structure_id", f"external-{index}")
    write(out_path, frames)
    return out_path


def validate_external_manifest(manifest_path, expected_value=None):
    payload = json.loads(Path(manifest_path).read_text())
    if expected_value is not None and payload.get("value") != expected_value:
        raise ValueError("external validation value mismatch")
    return payload
