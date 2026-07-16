"""Cheap validation of a run configuration before expensive work starts."""
import importlib
import math
import os
import shutil
import string
import subprocess
from pathlib import Path

import yaml

from adapters import load_config, resolve_config_path
from adapters.teacher import teacher_model_reference


def _require(cfg, dotted):
    value = cfg
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing required config field: {dotted}")
        value = value[key]
    return value


def _require_positive(cfg, dotted, integer=False, allow_zero=False):
    value = _require(cfg, dotted)
    if isinstance(value, bool):
        raise ValueError(f"config field {dotted} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"config field {dotted} must be finite")
    if integer and not numeric.is_integer():
        raise ValueError(f"config field {dotted} must be an integer")
    parsed = int(numeric) if integer else numeric
    if (parsed < 0 if allow_zero else parsed <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"config field {dotted} must be {qualifier}")
    return parsed


def _require_binary(binary, env=None):
    if env and env != os.environ.get("CONDA_DEFAULT_ENV"):
        if shutil.which("conda") is None:
            raise FileNotFoundError(f"Conda is required to inspect environment: {env}")
        code = f"import shutil,sys; sys.exit(0 if shutil.which({str(binary)!r}) else 1)"
        result = subprocess.run(["conda", "run", "-n", str(env), "python", "-c", code],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError(f"binary is not available in Conda environment {env}: {binary}")
    elif shutil.which(str(binary)) is None:
        raise FileNotFoundError(f"binary is not available: {binary}")


def _require_import_path(path):
    module_name, callable_name = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), callable_name, None)
    if not callable(value):
        raise TypeError(f"configured callable is invalid: {path}")
    return value


def _dotted_callable(value, field):
    if not isinstance(value, str) or "." not in value:
        raise ValueError(f"config field {field} must be a dotted callable path")
    return value


def check_teacher_config(cfg, check_files=True):
    kind = _require(cfg, "kind")
    calculator = _require(cfg, "calculator")
    if not isinstance(calculator, dict) or not (calculator.get("factory") or
                                                (calculator.get("module") and calculator.get("class"))):
        raise ValueError("teacher calculator requires factory or module/class")
    model = teacher_model_reference(cfg)
    model_arg = calculator.get("model_arg", "model")
    if model_arg and not model:
        raise ValueError("teacher config requires model or checkpoint")
    if (check_files and model is not None and
            calculator.get("model_is_path", True) and not Path(model).exists()):
        raise FileNotFoundError(f"teacher model is missing: {model}")
    if check_files:
        if calculator.get("factory"):
            _require_import_path(calculator["factory"])
        else:
            module = importlib.import_module(calculator["module"])
            cls = getattr(module, calculator["class"], None)
            if not callable(cls):
                raise TypeError("teacher calculator class is not callable")
            if calculator.get("constructor") and not callable(
                    getattr(cls, calculator["constructor"], None)):
                raise TypeError("teacher calculator constructor is not callable")
    required_kwargs = calculator.get("required_kwargs", [])
    if not isinstance(required_kwargs, list) or any(not isinstance(name, str)
                                                    for name in required_kwargs):
        raise ValueError("calculator.required_kwargs must be a list of names")
    missing_kwargs = [name for name in required_kwargs
                      if calculator.get("kwargs", {}).get(name) in (None, "")]
    if missing_kwargs:
        raise ValueError("teacher calculator is missing required kwargs: " +
                         ", ".join(missing_kwargs))
    constructor = calculator.get("factory") or (
        f"{calculator['module']}.{calculator['class']}"
        if calculator.get("module") and calculator.get("class") else "unresolved"
    )
    return [f"teacher kind={kind}", f"teacher calculator={constructor}"]


def check_acquisition_config(cfg):
    kind = _require(cfg, "kind")
    adapter = cfg.get("adapter", {})
    if adapter.get("acquire"):
        _dotted_callable(adapter["acquire"], "adapter.acquire")
        if adapter.get("preflight"):
            _dotted_callable(adapter["preflight"], "adapter.preflight")
        return [f"acquisition kind={kind}", "acquisition adapter=configured"]
    if kind == "augment-atoms":
        command = _require(cfg, "command")
        if not isinstance(command, list) or not command:
            raise ValueError("augment-atoms command must be a non-empty list")
        if not isinstance(command[0], str) or not command[0].strip():
            raise ValueError("augment-atoms command executable must be a non-empty string")
    elif kind == "teacher-md":
        _require_positive(cfg, "temperature_K")
        _require_positive(cfg, "timestep_fs")
        n_steps = _require_positive(cfg, "n_steps", integer=True)
        interval = _require_positive(cfg, "snapshot_interval", integer=True)
        if interval > n_steps:
            raise ValueError("teacher-md snapshot_interval must not exceed n_steps")
        if "friction_per_fs" not in cfg and "friction_ase_time_inverse" not in cfg:
            raise ValueError("teacher-md requires an explicit friction unit")
        if "friction" in cfg:
            raise ValueError("teacher-md field friction is ambiguous")
        friction_key = "friction_per_fs" if "friction_per_fs" in cfg else "friction_ase_time_inverse"
        _require_positive(cfg, friction_key, allow_zero=True)
        if "fix_center_of_mass" in cfg and not isinstance(cfg["fix_center_of_mass"], bool):
            raise ValueError("teacher-md fix_center_of_mass must be true or false")
    else:
        raise ValueError(f"unsupported acquisition kind: {kind}")
    return [f"acquisition kind={kind}"]


def check_acquisition_files(cfg):
    """Check external acquisition paths separately from its portable schema."""
    adapter = cfg.get("adapter", {})
    if adapter.get("acquire"):
        _require_import_path(adapter["acquire"])
        if adapter.get("preflight"):
            _require_import_path(adapter["preflight"])(cfg, check_files=True)
        return []
    if cfg["kind"] == "augment-atoms":
        _require_binary(cfg["command"][0], cfg.get("env"))
        if cfg.get("workdir") and not resolve_config_path(cfg, cfg["workdir"]).is_dir():
            raise FileNotFoundError("augment-atoms workdir is missing")
    return []


def check_validation_profile(cfg, require_ready=False):
    checks = _require(cfg, "checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("validation profile checks must be a non-empty list")
    if require_ready:
        def null_paths(value, prefix=""):
            if isinstance(value, dict):
                return [item for key, child in value.items()
                        for item in null_paths(child, f"{prefix}.{key}" if prefix else key)]
            return [prefix] if value is None else []
        def threshold_nulls(value, prefix=""):
            if not isinstance(value, dict):
                return []
            found = []
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                found += null_paths(child, path) if key == "thresholds" else threshold_nulls(child, path)
            return found
        missing = threshold_nulls(cfg)
        if missing:
            raise ValueError("validation thresholds are unresolved: " + ", ".join(missing))
    return ["validation checks=" + ",".join(checks)]


def check_uncertainty_config(cfg, check_files=True):
    kind = _require(cfg, "kind")
    adapter = cfg.get("adapter", {})
    if adapter.get("preflight"):
        _dotted_callable(adapter["preflight"], "adapter.preflight")
        if check_files:
            _require_import_path(adapter["preflight"])(cfg, check_files=check_files)
        return [f"uncertainty kind={kind}", "uncertainty adapter=configured"]
    if kind != "committee-force-std":
        raise ValueError(f"unsupported uncertainty kind: {kind}")
    aggregate = cfg.get("aggregate", "mean")
    if aggregate not in {"mean", "max"}:
        raise ValueError("uncertainty aggregate must be mean or max")
    top_fraction = float(cfg.get("top_fraction", 0.1))
    if not 0 < top_fraction <= 1:
        raise ValueError("uncertainty top_fraction must be in (0, 1]")
    if "require_complete" in cfg and not isinstance(cfg["require_complete"], bool):
        raise ValueError("uncertainty require_complete must be true or false")
    return [f"uncertainty kind={kind}", f"uncertainty aggregate={aggregate}"]


def check_md_config(cfg, check_files=True):
    kind = _require(cfg, "kind")
    adapter = cfg.get("adapter", {})
    if adapter:
        for field in ("renderer", "runner"):
            path = _dotted_callable(_require(cfg, f"adapter.{field}"), f"adapter.{field}")
            if check_files:
                _require_import_path(path)
        if adapter.get("preflight") and check_files:
            _require_import_path(adapter["preflight"])(cfg, check_files=check_files)
        return [f"MD backend kind={kind}", "MD backend adapter=configured"]
    if kind != "lammps":
        raise ValueError(f"MD backend {kind!r} requires adapter.renderer and adapter.runner")
    template_dir = resolve_config_path(cfg, _require(cfg, "template_dir"))
    binary = cfg.get("binary", "lmp_mpi")
    if check_files and not template_dir.is_dir():
        raise FileNotFoundError(f"MD template directory is missing: {template_dir}")
    if check_files:
        _require_binary(binary, cfg.get("env"))
    return [f"MD backend kind={kind}", f"MD binary={binary}"]


def check_dft_config(cfg, check_files=True):
    kind = _require(cfg, "kind")
    theory = _require(cfg, "reference_theory")
    if theory in (None, ""):
        raise ValueError("reference_theory must be explicitly identified")
    adapter = cfg.get("adapter", {})
    if adapter:
        renderer = _dotted_callable(_require(cfg, "adapter.renderer"), "adapter.renderer")
        if check_files:
            _require_import_path(renderer)
        if adapter.get("preflight") and check_files:
            _require_import_path(adapter["preflight"])(cfg, check_files=check_files)
        return [f"DFT backend kind={kind}", f"DFT reference theory={theory}",
                "DFT renderer adapter=configured"]
    if kind != "vasp":
        raise ValueError(f"DFT backend {kind!r} requires adapter.renderer")
    template = resolve_config_path(cfg, _require(cfg, "incar_template"))
    for field in ("encut_ev", "kspacing_inv_angstrom", "smearing.ismear",
                  "smearing.sigma", "relaxation.nsw", "relaxation.ibrion"):
        _require(cfg, field)
    _require_positive(cfg, "encut_ev")
    _require_positive(cfg, "kspacing_inv_angstrom")
    _require_positive(cfg, "smearing.sigma", allow_zero=True)
    if check_files:
        if not template.is_file():
            raise FileNotFoundError(f"DFT template is missing: {template}")
        fields = {name for _, name, _, _ in string.Formatter().parse(template.read_text()) if name}
        provided = {"ENCUT", "KSPACING", "ISMEAR", "SIGMA", "NSW", "IBRION"}
        provided.update(cfg.get("template_variables", {}))
        unresolved = fields - provided
        if unresolved:
            raise ValueError("DFT template variables are unresolved: " + ", ".join(sorted(unresolved)))
    return [f"DFT backend kind={kind}", f"DFT reference theory={theory}"]


def check_student_config(cfg, check_files=True, require_ready=False):
    """Return a list of human-readable checks; raise on an invalid contract."""
    kind = _require(cfg, "kind")
    deploy = cfg.get("deploy", {})
    elements = deploy.get("elements")
    if elements is not None and (not isinstance(elements, list) or not elements or
            any(not isinstance(element, str) or not element.strip() for element in elements) or
            len(set(elements)) != len(elements)):
        raise ValueError("deploy.elements must be a non-empty list of unique element symbols")
    if (deploy.get("lammps_pair_style") or deploy.get("pair_coeff_template")) and not elements:
        raise ValueError("LAMMPS deployment requires deploy.elements")
    if require_ready and elements and any(element in {"A", "B", "C"} for element in elements):
        raise ValueError("replace placeholder deploy.elements before a pilot")

    checks = [f"student kind={kind}"]
    if elements:
        checks.append(f"element order={','.join(elements)}")
    if "n_seeds" in cfg.get("committee", {}):
        try:
            n_seeds = _require_positive(cfg, "committee.n_seeds", integer=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("committee.n_seeds must be an integer of at least 1") from exc
    else:
        n_seeds = 4
    checks.append(f"committee seeds={n_seeds}")
    adapter = cfg.get("adapter", {})
    for field in ("train", "load", "deploy", "preflight"):
        if adapter.get(field):
            path = _dotted_callable(adapter[field], f"adapter.{field}")
            if check_files:
                _require_import_path(path)
    if adapter.get("preflight") and check_files:
        _require_import_path(adapter["preflight"])(cfg, check_files=check_files,
                                                    require_ready=require_ready)

    train_command = cfg.get("train", {}).get("command")
    if train_command:
        if not isinstance(train_command, list) or not train_command:
            raise ValueError("train.command must be a non-empty list")
        if not cfg["train"].get("artifact"):
            raise ValueError("train.command requires train.artifact")
        if check_files and "{" not in str(train_command[0]):
            _require_binary(train_command[0], cfg["train"].get("env"))
        checks.append("training adapter=command")
    elif adapter.get("train"):
        checks.append(f"training adapter={adapter['train']}")
    elif kind in {"simple-nn", "grace-fs"}:
        _require(cfg, "train.config_template")
        if check_files:
            paths = [cfg["train"]["config_template"]]
            if kind == "simple-nn":
                paths += list(_require(cfg, "train.descriptor_params").values())
            missing = [str(resolve_config_path(cfg, p)) for p in paths
                       if not resolve_config_path(cfg, p).exists()]
            if missing:
                raise FileNotFoundError("missing student inputs: " + ", ".join(missing))
            if kind == "grace-fs":
                template = resolve_config_path(cfg, cfg["train"]["config_template"]).read_text()
                if "Paste the reviewed GRACE/FS" in template:
                    raise ValueError("GRACE/FS template is still the packaged placeholder; generate a version-matched input")
                if template.count("{{DATASET_PATH}}") != 1:
                    raise ValueError("GRACE/FS template requires exactly one {{DATASET_PATH}} placeholder")
                yaml.safe_load(template.replace("{{DATASET_PATH}}", "/tmp/dataset.extxyz"))
        env = cfg["train"].get("env")
        if check_files:
            if kind == "grace-fs":
                _require_binary(cfg["train"].get("binary", "gracemaker"), env)
            elif kind == "simple-nn":
                module = cfg["train"].get("runner", {}).get("module", "simple_nn.driver")
                if env and env != os.environ.get("CONDA_DEFAULT_ENV"):
                    _require_binary("python", env)
                else:
                    importlib.import_module(module)
        checks.append(f"training environment={env or 'current interpreter'}")
    elif kind != "mock":
        raise ValueError("student training requires adapter.train or train.command")

    factory_path = cfg.get("predict", {}).get("factory")
    if factory_path:
        if check_files:
            _require_import_path(factory_path)
        checks.append(f"prediction factory={factory_path}")
    else:
        checks.append("prediction factory=not configured (training/deployment only)")
    if deploy.get("renderer") and check_files:
        _require_import_path(_dotted_callable(deploy["renderer"], "deploy.renderer"))
    return checks


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_config", nargs="?", help="student config (backward-compatible positional form)")
    parser.add_argument("--teacher-config")
    parser.add_argument("--acquisition-config")
    parser.add_argument("--validation-profile")
    parser.add_argument("--uncertainty-config")
    parser.add_argument("--md-config")
    parser.add_argument("--dft-config")
    parser.add_argument("--skip-files", action="store_true", help="validate schema only")
    parser.add_argument("--require-ready", action="store_true",
                        help="reject unresolved elements, templates, and validation thresholds")
    args = parser.parse_args()
    checks = []
    try:
        student_cfg = load_config(args.student_config) if args.student_config else None
        student_checks = (check_student_config(student_cfg,
                                               check_files=not args.skip_files,
                                               require_ready=args.require_ready)
                          if student_cfg else [])
        if args.teacher_config:
            checks += check_teacher_config(load_config(args.teacher_config), check_files=not args.skip_files)
        if args.acquisition_config:
            acquisition_cfg = load_config(args.acquisition_config)
            checks += check_acquisition_config(acquisition_cfg)
            if not args.skip_files:
                checks += check_acquisition_files(acquisition_cfg)
        if args.validation_profile:
            checks += check_validation_profile(load_config(args.validation_profile), args.require_ready)
        if args.uncertainty_config:
            checks += check_uncertainty_config(load_config(args.uncertainty_config),
                                               check_files=not args.skip_files)
            if student_cfg and int(student_cfg.get("committee", {}).get("n_seeds", 4)) < 2:
                raise ValueError("committee uncertainty requires committee.n_seeds >= 2")
        if args.md_config:
            checks += check_md_config(load_config(args.md_config), check_files=not args.skip_files)
        if args.dft_config:
            checks += check_dft_config(load_config(args.dft_config), check_files=not args.skip_files)
        if student_cfg:
            checks += student_checks
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, TypeError,
            yaml.YAMLError) as exc:
        parser.error(str(exc))
    if not checks:
        parser.error("provide a student config or one of the named config options")
    for check in checks:
        print(f"PASS: {check}")


if __name__ == "__main__":
    main()
