"""Structure acquisition and teacher pseudo-labeling backends."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from adapters import load_config, resolve_config_path
from adapters.teacher import load_teacher, teacher_model_reference


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_augment_atoms(cfg, seed_path, out_path):
    """Run a configured augment-atoms wrapper without assuming its CLI version."""
    context = {"seed_path": str(Path(seed_path).resolve()), "out_path": str(Path(out_path).resolve())}
    command = [str(part).format(**context) for part in cfg["command"]]
    workdir = resolve_config_path(cfg, cfg["workdir"]) if cfg.get("workdir") else None
    subprocess.run(command, check=True, cwd=workdir)
    if not Path(out_path).exists():
        raise FileNotFoundError(f"augment-atoms command produced no output: {out_path}")
    return Path(out_path)


def run_teacher_md(cfg, teacher_cfg, seed_path, out_path):
    """Generate snapshots by Langevin MD under the teacher ASE calculator."""
    seeds = read(seed_path, index=":")
    calc = load_teacher(teacher_cfg)
    snapshots = []
    for seed_index, source in enumerate(seeds):
        parent_id = source.info.get("parent_structure_id",
                                    source.info.get("structure_id", f"seed-{seed_index:08d}"))
        atoms = source.copy()
        atoms.calc = calc
        temperature = float(cfg["temperature_K"])
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                     rng=np.random.default_rng(int(cfg.get("seed", 0)) + seed_index))
        dyn = Langevin(atoms, float(cfg.get("timestep_fs", 1.0)) * units.fs,
                       temperature_K=temperature, friction=float(cfg.get("friction", 0.01)))
        stride = int(cfg.get("snapshot_interval", 100))
        n_steps = int(cfg["n_steps"])

        def capture():
            frame = atoms.copy()
            frame.info.update(acquisition="teacher-md", seed_structure_index=seed_index,
                              temperature_K=temperature, parent_structure_id=str(parent_id))
            snapshots.append(frame)

        dyn.attach(capture, interval=stride)
        dyn.run(n_steps)
    write(out_path, snapshots)
    return Path(out_path)


def acquire(acquisition_cfg, teacher_cfg, seed_path, out_path):
    kind = acquisition_cfg["kind"]
    if kind == "augment-atoms":
        result = run_augment_atoms(acquisition_cfg, seed_path, out_path)
        validate_lineage(result)
        return result
    if kind == "teacher-md":
        result = run_teacher_md(acquisition_cfg, teacher_cfg, seed_path, out_path)
        validate_lineage(result)
        return result
    raise NotImplementedError(f"acquisition kind={kind!r} is not implemented")


def validate_lineage(structures_path, grouping_key="parent_structure_id"):
    frames = read(structures_path, index=":")
    missing = [index for index, atoms in enumerate(frames) if grouping_key not in atoms.info]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"acquired structures are missing {grouping_key!r} at frames: {preview}")
    return len(frames)


def label_with_teacher(teacher_cfg, structures_path, out_path, manifest_path, include_stress=False):
    """Attach teacher labels to ASE-readable structures and write provenance."""
    frames = read(structures_path, index=":")
    calc = load_teacher(teacher_cfg)
    for index, atoms in enumerate(frames):
        atoms.calc = calc
        atoms.info["teacher_energy"] = float(atoms.get_potential_energy())
        atoms.arrays["teacher_forces"] = np.asarray(atoms.get_forces())
        if include_stress:
            atoms.info["teacher_stress"] = np.asarray(atoms.get_stress()).tolist()
        atoms.info.setdefault("structure_id", f"frame-{index:08d}")
        atoms.info["label_source"] = "teacher"
        atoms.calc = None
    write(out_path, frames)
    model_value = teacher_model_reference(teacher_cfg)
    model_path = Path(model_value).expanduser() if model_value else None
    config_path = teacher_cfg.get("_config_path")
    packages = {}
    for package in ("ase", "numpy", "mace-torch", "nequip"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema_version": 1,
        "teacher_kind": teacher_cfg["kind"],
        "teacher_model": model_value,
        "teacher_model_sha256": _sha256(model_path) if model_path and model_path.is_file() else None,
        "teacher_head": teacher_cfg.get("calculator", {}).get("kwargs", {}).get("head"),
        "calculator": teacher_cfg.get("calculator", {}),
        "teacher_config_sha256": _sha256(config_path) if config_path else None,
        "source": str(Path(structures_path).resolve()),
        "source_sha256": _sha256(structures_path),
        "output": str(Path(out_path).resolve()),
        "n_frames": len(frames),
        "labels": ["energy", "forces"] + (["stress"] if include_stress else []),
        "units": {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"},
        "sha256": _sha256(out_path),
        "environment": {"python": platform.python_version(), "packages": packages},
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    acq = sub.add_parser("acquire")
    acq.add_argument("acquisition_config")
    acq.add_argument("teacher_config")
    acq.add_argument("seed_structures")
    acq.add_argument("output")
    label = sub.add_parser("label")
    label.add_argument("teacher_config")
    label.add_argument("structures")
    label.add_argument("output")
    label.add_argument("manifest")
    label.add_argument("--stress", action="store_true")
    args = p.parse_args()
    teacher_cfg = load_config(args.teacher_config)
    if args.action == "acquire":
        acquire(load_config(args.acquisition_config), teacher_cfg, args.seed_structures, args.output)
    else:
        label_with_teacher(teacher_cfg, args.structures, args.output, args.manifest, args.stress)


if __name__ == "__main__":
    main()
