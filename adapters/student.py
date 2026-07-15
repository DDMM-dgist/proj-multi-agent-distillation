"""Student adapter: train, load, and deploy a student model from
configs/student.<name>.yaml. This is the adapter with the most real work in
it, because "student" bundles three different concerns: training recipe,
checkpoint loading, and MD deployment (LAMMPS pair_style).

The `simple-nn` path is the extracted reference runner, but its wrapper module
must be verified for the installed SIMPLE-NN version. `mtp`/`ace`/`compact-gnn`
are documented stubs — see configs/README.md
for what implementing one of them actually involves (usually: a training
subprocess call + a checkpoint loader + a one-line LAMMPS pair_style string).
"""
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from adapters import resolve_config_path
from adapters.contracts import ModelArtifact, PredictionBatch


def train_student(cfg, dataset_path, out_dir, seed):
    """Train one committee member.

    cfg: configs/student.<name>.yaml, already loaded.
    dataset_path: path to the (already merged + energy-aligned, if mixing
        teacher-labeled + DFT-anchor data — see agents/data-curator.md) training set.
    out_dir: where to write the checkpoint + logs for this seed.
    seed: int, the committee member's random seed.
    """
    kind = cfg["kind"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "simple-nn":
        checkpoint = _train_simple_nn(cfg, dataset_path, out_dir, seed)
        return ModelArtifact(kind=kind, path=checkpoint, seed=seed).require_exists()
    elif kind == "grace-fs":
        checkpoint = _train_grace_fs(cfg, dataset_path, out_dir, seed)
        return ModelArtifact(kind=kind, path=checkpoint, seed=seed).require_exists()
    elif kind == "mock":
        checkpoint = out_dir / "mock-model.json"
        checkpoint.write_text(f'{{"seed": {int(seed)}}}\n')
        return ModelArtifact(kind=kind, path=checkpoint, seed=seed).require_exists()
    else:
        raise NotImplementedError(
            f"student kind={kind!r} training is not implemented in adapters/student.py. "
            f"Add a _train_<kind>(...) function following the _train_simple_nn shape below."
        )


def _train_simple_nn(cfg, dataset_path, out_dir, seed):
    """SIMPLE-NN v2 training.

    NOTE: this shells out to a driver script rather than calling SIMPLE-NN's
    python API directly, because the exact API differs across SIMPLE-NN
    versions. Point `driver_script` at a small wrapper in your own SIMPLE-NN
    install that reads `input.yaml` + `params_Si`/`params_O` and trains one
    seed — adjust the command below to match how you actually invoke SIMPLE-NN
    v2 in your environment (this is a template, verify before relying on it).
    """
    train_cfg = cfg["train"]
    runner = train_cfg.get("runner", {})
    module = runner.get("module", "simple_nn.driver")
    env = train_cfg.get("env")
    prefix = ["conda", "run", "-n", env, "python"] if env else [sys.executable]
    cmd = prefix + [
        "-m", module,  # override train.runner.module for the installed SIMPLE-NN wrapper
        "--config", str(resolve_config_path(cfg, train_cfg["config_template"])),
    ]
    for element, path in train_cfg["descriptor_params"].items():
        cmd += ["--descriptor-param", f"{element}={resolve_config_path(cfg, path)}"]
    cmd += [
        "--dataset", str(dataset_path),
        "--out", str(out_dir),
        "--seed", str(seed),
        "--epochs", str(train_cfg["total_epoch"]),
        "--precision", "double" if train_cfg.get("double_precision") else "single",
        "--batch-size", str(train_cfg["batch_size"]),
    ]
    if train_cfg.get("use_stress"):
        cmd += ["--use-stress", "--stress-loss-weight", str(train_cfg.get("stress_loss_weight", 0.1))]

    print(f"[train_student:simple-nn] seed={seed} -> {out_dir}")
    print("  ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir / "potential_saved_bestmodel"


def _train_grace_fs(cfg, dataset_path, out_dir, seed):
    """Run gracemaker from a user-reviewed input template and export GRACE/FS."""
    train_cfg = cfg["train"]
    template = resolve_config_path(cfg, train_cfg["config_template"])
    rendered = template.read_text().replace("{{DATASET_PATH}}", str(Path(dataset_path).resolve()))
    input_path = out_dir / "input.yaml"
    input_path.write_text(rendered)
    env = train_cfg.get("env")
    prefix = ["conda", "run", "-n", env] if env else []
    binary = train_cfg.get("binary", "gracemaker")
    subprocess.run(prefix + [binary, "--seed", str(seed), str(input_path)], check=True, cwd=out_dir)
    subprocess.run(prefix + [binary, "--seed", str(seed), "-r", "-s", "-sf"], check=True, cwd=out_dir)
    return out_dir / "seed" / str(seed) / "FS_model.yaml"


def load_student(cfg, checkpoint):
    """Return whatever handle downstream validation needs (path is often enough
    for LAMMPS deployment; extend if a script needs an in-process calculator)."""
    kind = cfg["kind"]
    if isinstance(checkpoint, ModelArtifact):
        return checkpoint.require_exists()
    checkpoint = Path(checkpoint)
    if kind == "simple-nn":
        path = checkpoint if checkpoint.name == "potential_saved_bestmodel" else checkpoint / "potential_saved_bestmodel"
        return ModelArtifact(kind=kind, path=path).require_exists()
    if kind == "grace-fs":
        path = checkpoint if checkpoint.name == "FS_model.yaml" else checkpoint / "FS_model.yaml"
        return ModelArtifact(kind=kind, path=path).require_exists()
    if kind == "mock":
        path = checkpoint if checkpoint.name == "mock-model.json" else checkpoint / "mock-model.json"
        return ModelArtifact(kind=kind, path=path).require_exists()
    raise NotImplementedError(f"student kind={kind!r} loading is not implemented.")


def _calculator_from_predict_config(cfg, artifact):
    """Construct an ASE calculator through a config-supplied factory.

    This avoids adding an adapter branch for every student architecture. The
    callable receives ``checkpoint=<path>`` plus optional ``kwargs``.
    """
    pred = cfg.get("predict", {})
    factory_path = pred.get("factory")
    if not factory_path:
        raise NotImplementedError(
            "student prediction requires predict.factory='package.module.callable' in the config"
        )
    module_name, callable_name = factory_path.rsplit(".", 1)
    factory = getattr(importlib.import_module(module_name), callable_name)
    kwargs = dict(pred.get("kwargs", {}))
    checkpoint_arg = pred.get("checkpoint_arg", "checkpoint")
    if not checkpoint_arg:
        return factory(**kwargs)
    if checkpoint_arg == "__positional__":
        return factory(str(artifact.path), **kwargs)
    kwargs[checkpoint_arg] = str(artifact.path)
    return factory(**kwargs)


def predict_student(cfg, model_artifact, structures, include_stress=False):
    """Predict through a common ASE-based interface for any architecture."""
    artifact = load_student(cfg, model_artifact)
    calculator = _calculator_from_predict_config(cfg, artifact)
    energies, forces, stresses = [], [], []
    for source in structures:
        atoms = source.copy()
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        forces.append(np.asarray(atoms.get_forces()))
        if include_stress:
            stresses.append(np.asarray(atoms.get_stress(voigt=False)))
    return PredictionBatch(
        energies=np.asarray(energies),
        forces=forces,
        stresses=stresses if include_stress else None,
    )


def lammps_pair_style_block(cfg, checkpoint_path):
    """Return the LAMMPS input lines needed to deploy this student.

    Used by adapters/md_backend.py when rendering templates/lammps/*.in.template.
    """
    kind = cfg["kind"]
    if isinstance(checkpoint_path, ModelArtifact):
        checkpoint_path = checkpoint_path.require_exists().path
    deploy = cfg.get("deploy", {})
    elements = deploy.get("elements")
    if not elements:
        raise ValueError("student config deploy.elements must list the LAMMPS atom-type order")
    element_order = " ".join(elements)
    if kind == "simple-nn":
        style = deploy.get("lammps_pair_style", "nn")
        return f"pair_style {style}\npair_coeff * * {checkpoint_path} {element_order}\n"
    elif kind == "grace-fs":
        style = deploy.get("lammps_pair_style", "grace/fs")
        return f"pair_style {style}\npair_coeff * * {checkpoint_path} {element_order}\n"
    elif kind == "mtp":
        return f"pair_style mlip load_from={checkpoint_path}\npair_coeff * *\n"
    else:
        raise NotImplementedError(
            f"student kind={kind!r} has no LAMMPS pair_style recipe yet — add one "
            f"(usually 1-2 lines) here and in configs/README.md."
        )
