"""Config-driven student training, loading, prediction, and deployment adapters."""
import importlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from adapters import resolve_config_path
from adapters.contracts import ModelArtifact, PredictionBatch


def _callable(path):
    module_name, name = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), name, None)
    if not callable(value):
        raise TypeError(f"configured callable is invalid: {path}")
    return value


def _artifact(value, kind, seed=None):
    if isinstance(value, ModelArtifact):
        return value.require_exists()
    return ModelArtifact(kind=kind, path=Path(value), seed=seed).require_exists()


def train_student(cfg, dataset_path, out_dir, seed):
    """Train one committee member.

    cfg: configs/student.<name>.yaml, already loaded.
    dataset_path: path to the reviewed training set. If multiple label sources
        are combined, any reference transformation must already have been
        performed by an explicit project-specific stage.
    out_dir: where to write the checkpoint + logs for this seed.
    seed: int, the committee member's random seed.
    """
    kind = cfg["kind"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = cfg.get("adapter", {})
    if adapter.get("train"):
        return _artifact(_callable(adapter["train"])(cfg, dataset_path, out_dir, int(seed)),
                         kind, int(seed))
    if cfg.get("train", {}).get("command"):
        context = {"dataset_path": str(Path(dataset_path).resolve()),
                   "out_dir": str(out_dir.resolve()), "seed": int(seed),
                   "project_dir": cfg.get("_project_dir", str(Path.cwd()))}
        command = [str(part).format(**context) for part in cfg["train"]["command"]]
        env = cfg["train"].get("env")
        if env:
            command = ["conda", "run", "--no-capture-output", "-n", env, *command]
        subprocess.run(command, check=True, cwd=out_dir)
        artifact = cfg["train"].get("artifact")
        if not artifact:
            raise ValueError("train.command requires train.artifact")
        artifact_path = Path(str(artifact).format(**context))
        if not artifact_path.is_absolute():
            artifact_path = out_dir / artifact_path
        return _artifact(artifact_path, kind, int(seed))
    trainers = {"simple-nn": _train_simple_nn, "grace-fs": _train_grace_fs,
                "mock": _train_mock}
    if kind not in trainers:
        raise NotImplementedError(
            f"student kind={kind!r} requires adapter.train or train.command"
        )
    return _artifact(trainers[kind](cfg, dataset_path, out_dir, seed), kind, int(seed))


def _train_mock(cfg, dataset_path, out_dir, seed):
    checkpoint = Path(out_dir) / "mock-model.json"
    checkpoint.write_text(f'{{"seed": {int(seed)}}}\n')
    return checkpoint


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
    rendered_config = _render_simple_nn_config(cfg, out_dir)
    runner = train_cfg.get("runner", {})
    module = runner.get("module", "simple_nn.driver")
    env = train_cfg.get("env")
    prefix = ["conda", "run", "-n", env, "python"] if env else [sys.executable]
    cmd = prefix + [
        "-m", module,  # override train.runner.module for the installed SIMPLE-NN wrapper
        "--config", str(rendered_config),
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


def _render_simple_nn_config(cfg, out_dir):
    """Render the packaged SIMPLE-NN wrapper template without silent placeholders."""
    train_cfg = cfg["train"]
    text = resolve_config_path(cfg, train_cfg["config_template"]).read_text()
    replacements = {
        "NODES": train_cfg["nodes"],
        "BATCH_SIZE": train_cfg["batch_size"],
        "TOTAL_EPOCH": train_cfg["total_epoch"],
        "LEARNING_RATE": train_cfg.get("learning_rate", 1e-4),
        "DOUBLE_PRECISION": str(bool(train_cfg.get("double_precision"))).lower(),
        "USE_STRESS": str(bool(train_cfg.get("use_stress"))).lower(),
        "STRESS_LOSS_WEIGHT": train_cfg.get("stress_loss_weight", 0.0),
    }
    for element, path in train_cfg["descriptor_params"].items():
        token = re.sub(r"[^A-Za-z0-9]", "_", element).upper() + "_PARAMS_PATH"
        replacements[token] = str(resolve_config_path(cfg, path))
    for key, value in replacements.items():
        text = text.replace("{" + key + "}", str(value))
    unresolved = sorted(set(re.findall(r"\{[A-Z][A-Z0-9_]*\}", text)))
    if unresolved:
        raise ValueError("unresolved SIMPLE-NN template placeholders: " + ", ".join(unresolved))
    output = Path(out_dir) / "input.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return output


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
    loader = cfg.get("adapter", {}).get("load")
    if loader:
        return _artifact(_callable(loader)(cfg, checkpoint), kind)
    checkpoint = Path(checkpoint)
    names = {"simple-nn": "potential_saved_bestmodel", "grace-fs": "FS_model.yaml",
             "mock": "mock-model.json"}
    if kind in names:
        path = checkpoint if checkpoint.name == names[kind] else checkpoint / names[kind]
        return ModelArtifact(kind=kind, path=path).require_exists()
    # For callable/command adapters the committee manifest already stores the
    # exact checkpoint path, so no architecture-specific path convention is needed.
    return ModelArtifact(kind=kind, path=checkpoint).require_exists()


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
    if isinstance(checkpoint_path, ModelArtifact):
        checkpoint_path = checkpoint_path.require_exists().path
    deploy = cfg.get("deploy", {})
    renderer = cfg.get("adapter", {}).get("deploy") or deploy.get("renderer")
    if renderer:
        return str(_callable(renderer)(cfg, Path(checkpoint_path)))
    elements = deploy.get("elements")
    if not elements:
        raise ValueError("student config deploy.elements must list the LAMMPS atom-type order")
    element_order = " ".join(elements)
    style = deploy.get("lammps_pair_style")
    if not style:
        raise ValueError("deployment requires deploy.lammps_pair_style or adapter.deploy")
    context = {"checkpoint": str(checkpoint_path), "elements": element_order,
               "pair_style": style}
    pair_style_line = deploy.get("pair_style_template", "pair_style {pair_style}")
    pair_coeff_line = deploy.get(
        "pair_coeff_template", "pair_coeff * * {checkpoint} {elements}"
    )
    return pair_style_line.format(**context) + "\n" + pair_coeff_line.format(**context) + "\n"
