"""Config-driven student training, loading, prediction, and deployment adapters."""
import importlib
import re
import subprocess
import sys
import tempfile
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
    """Analytic mock committee member: writes a seed-only checkpoint.

    This trainer performs no numerical optimization. The returned artifact
    carries honest provenance describing how the checkpoint was produced so
    downstream manifests (e.g. student_committee.manifest.json) and gates can
    explain each committee member without fabricating epochs, losses, or
    wall-times that were never measured.
    """
    checkpoint = Path(out_dir) / "mock-model.json"
    checkpoint.write_text(f'{{"seed": {int(seed)}}}\n')
    return ModelArtifact(
        kind=cfg["kind"],
        path=checkpoint,
        seed=int(seed),
        metadata={
            "trainer_kind": "analytic_mock",
            "training_mode": "no_optimization",
            "seed": int(seed),
            "epochs": "not_applicable",
            "optimizer": "not_applicable",
            "loss": "not_applicable",
            "adapter": "adapters.student._train_mock",
            "checkpoint_contents": (
                "seed-only JSON consumed by "
                "adapters.mock_model.MockCheckpointCalculator"
            ),
            "notes": (
                "Synthetic committee member; the checkpoint stores only the "
                "seed and no learned parameters, so no training curve, "
                "optimizer state, or wall-time was measured."
            ),
        },
    )


def _train_simple_nn(cfg, dataset_path, out_dir, seed):
    """Run one SIMPLE-NN v2 seed through the configured CLI wrapper."""
    train_cfg = cfg["train"]
    rendered_config = _render_simple_nn_config(cfg, out_dir)
    runner = train_cfg.get("runner", {})
    module = runner.get("module", "adapters.simple_nn_v2_wrapper")
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
    policy = (cfg.get("struct_weight_policy") or {}).get("name", "").strip().lower()
    if policy:
        cmd += ["--struct-weight-policy", policy]

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
        "SUBPROCESSES": train_cfg.get("subprocesses", 0),
        "ACCURATE_TRAIN_RMSE": str(bool(
            train_cfg.get("accurate_train_rmse", True)
        )).lower(),
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


def _validated_prediction_batch(value, structures, include_stress=False):
    if not isinstance(value, PredictionBatch):
        raise TypeError("student prediction adapter must return PredictionBatch")
    if len(value.energies) != len(structures):
        raise ValueError("student prediction count does not match the requested structures")
    if not np.all(np.isfinite(np.asarray(value.energies, dtype=float))):
        raise ValueError("student prediction energies contain non-finite values")
    for index, (atoms, forces) in enumerate(zip(structures, value.forces)):
        forces = np.asarray(forces, dtype=float)
        if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
            raise ValueError(f"student prediction forces are invalid at frame {index}")
    if include_stress:
        if value.stresses is None:
            raise ValueError("student prediction did not return requested stresses")
        for index, stress in enumerate(value.stresses):
            stress = np.asarray(stress, dtype=float)
            if stress.shape not in {(6,), (3, 3)} or not np.all(np.isfinite(stress)):
                raise ValueError(f"student prediction stress is invalid at frame {index}")
    return value


def _predict_with_command(cfg, artifact, structures, include_stress=False):
    """Run a config-selected batch predictor through a temporary extxyz exchange."""
    from ase.io import read, write

    pred = cfg.get("predict", {})
    raw_command = pred.get("command")
    if not isinstance(raw_command, list) or not raw_command:
        raise ValueError("predict.command must be a non-empty list")
    with tempfile.TemporaryDirectory(prefix="student-predict-") as tmp:
        work_dir = Path(tmp).resolve()
        input_path = work_dir / "structures.extxyz"
        output_path = work_dir / "predictions.extxyz"
        write(input_path, list(structures), format="extxyz")
        context = {
            "checkpoint": str(artifact.path.resolve()),
            "structures": str(input_path),
            "output": str(output_path),
            "work_dir": str(work_dir),
            "project_dir": cfg.get("_project_dir", str(Path.cwd())),
            "include_stress": "true" if include_stress else "false",
        }
        command = [str(part).format(**context) for part in raw_command]
        env = pred.get("env")
        if env:
            command = ["conda", "run", "--no-capture-output", "-n", env, *command]
        subprocess.run(command, check=True, cwd=work_dir)
        if not output_path.is_file():
            raise FileNotFoundError("predict.command produced no predictions.extxyz")
        predicted = read(output_path, index=":")
        if len(predicted) != len(structures):
            raise ValueError("predict.command output frame count does not match the input")
        energy_key = pred.get("energy_key", "student_energy")
        forces_key = pred.get("forces_key", "student_forces")
        stress_key = pred.get("stress_key", "student_stress")
        energies, forces, stresses = [], [], []
        for index, (source, result) in enumerate(zip(structures, predicted)):
            if (not np.array_equal(source.numbers, result.numbers) or
                    not np.allclose(source.positions, result.positions, atol=1e-12, rtol=0) or
                    not np.allclose(source.cell.array, result.cell.array, atol=1e-12, rtol=0) or
                    not np.array_equal(source.pbc, result.pbc)):
                raise ValueError(f"predict.command changed or reordered structure {index}")
            if energy_key not in result.info or forces_key not in result.arrays:
                raise ValueError(
                    f"predict.command output frame {index} is missing {energy_key}/{forces_key}"
                )
            energies.append(float(result.info[energy_key]))
            forces.append(np.asarray(result.arrays[forces_key]))
            if include_stress:
                if stress_key not in result.info:
                    raise ValueError(
                        f"predict.command output frame {index} is missing {stress_key}"
                    )
                stresses.append(np.asarray(result.info[stress_key]))
    return PredictionBatch(np.asarray(energies), forces,
                           stresses if include_stress else None)


def predict_student(cfg, model_artifact, structures, include_stress=False):
    """Predict through a callable, command, or ASE-calculator interface."""
    artifact = load_student(cfg, model_artifact)
    structures = list(structures)
    adapter = cfg.get("adapter", {})
    if adapter.get("predict"):
        value = _callable(adapter["predict"])(cfg, artifact, structures, include_stress)
        return _validated_prediction_batch(value, structures, include_stress)
    if cfg.get("predict", {}).get("command"):
        value = _predict_with_command(cfg, artifact, structures, include_stress)
        return _validated_prediction_batch(value, structures, include_stress)
    calculator = _calculator_from_predict_config(cfg, artifact)
    energies, forces, stresses = [], [], []
    for source in structures:
        atoms = source.copy()
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        forces.append(np.asarray(atoms.get_forces()))
        if include_stress:
            stresses.append(np.asarray(atoms.get_stress(voigt=False)))
    return _validated_prediction_batch(PredictionBatch(
        energies=np.asarray(energies),
        forces=forces,
        stresses=stresses if include_stress else None,
    ), structures, include_stress)


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
