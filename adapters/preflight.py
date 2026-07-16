"""Cheap validation of a run configuration before expensive work starts."""
import importlib
import shutil
from pathlib import Path

from adapters import load_config, resolve_config_path
from adapters.teacher import teacher_model_reference


def _require(cfg, dotted):
    value = cfg
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing required config field: {dotted}")
        value = value[key]
    return value


def check_teacher_config(cfg, check_files=True):
    kind = _require(cfg, "kind")
    calculator = _require(cfg, "calculator")
    if not isinstance(calculator, dict) or not (calculator.get("factory") or
                                                (calculator.get("module") and calculator.get("class"))):
        raise ValueError("teacher calculator requires factory or module/class")
    model = teacher_model_reference(cfg)
    if not model:
        raise ValueError("teacher config requires model or checkpoint")
    if check_files and calculator.get("model_is_path", True) and not Path(model).exists():
        raise FileNotFoundError(f"teacher model is missing: {model}")
    if "mace-mh" in kind and not calculator.get("kwargs", {}).get("head"):
        raise ValueError("MACE-MH teacher config requires calculator.kwargs.head")
    return [f"teacher kind={kind}", f"teacher head={calculator.get('kwargs', {}).get('head', 'n/a')}"]


def check_acquisition_config(cfg):
    kind = _require(cfg, "kind")
    if kind == "augment-atoms":
        command = _require(cfg, "command")
        if not isinstance(command, list) or not command:
            raise ValueError("augment-atoms command must be a non-empty list")
    elif kind == "teacher-md":
        for field in ("temperature_K", "timestep_fs", "n_steps", "snapshot_interval"):
            _require(cfg, field)
        if "friction_per_fs" not in cfg and "friction_ase_time_inverse" not in cfg:
            raise ValueError("teacher-md requires an explicit friction unit")
        if "friction" in cfg:
            raise ValueError("teacher-md field friction is ambiguous")
    else:
        raise ValueError(f"unsupported acquisition kind: {kind}")
    return [f"acquisition kind={kind}"]


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


def check_student_config(cfg, check_files=True, require_ready=False):
    """Return a list of human-readable checks; raise on an invalid contract."""
    kind = _require(cfg, "kind")
    elements = _require(cfg, "deploy.elements")
    if not isinstance(elements, list) or not elements or len(set(elements)) != len(elements):
        raise ValueError("deploy.elements must be a non-empty list of unique element symbols")
    if require_ready and any(element in {"A", "B", "C"} for element in elements):
        raise ValueError("replace placeholder deploy.elements before a pilot")

    checks = [f"student kind={kind}", f"element order={','.join(elements)}"]
    if kind in {"simple-nn", "grace-fs"}:
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
        env = cfg["train"].get("env")
        if check_files and env and shutil.which("conda") is None:
            raise RuntimeError("train.env is set but conda is not available")
        checks.append(f"training environment={env or 'current interpreter'}")

    factory_path = cfg.get("predict", {}).get("factory")
    if factory_path:
        if check_files:
            module_name, callable_name = factory_path.rsplit(".", 1)
            factory = getattr(importlib.import_module(module_name), callable_name, None)
            if not callable(factory):
                raise TypeError(f"predict.factory is not callable: {factory_path}")
        checks.append(f"prediction factory={factory_path}")
    else:
        checks.append("prediction factory=not configured (training/deployment only)")
    return checks


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_config", nargs="?", help="student config (backward-compatible positional form)")
    parser.add_argument("--teacher-config")
    parser.add_argument("--acquisition-config")
    parser.add_argument("--validation-profile")
    parser.add_argument("--skip-files", action="store_true", help="validate schema only")
    parser.add_argument("--require-ready", action="store_true",
                        help="reject unresolved elements, templates, and validation thresholds")
    args = parser.parse_args()
    checks = []
    try:
        if args.teacher_config:
            checks += check_teacher_config(load_config(args.teacher_config), check_files=not args.skip_files)
        if args.acquisition_config:
            checks += check_acquisition_config(load_config(args.acquisition_config))
        if args.validation_profile:
            checks += check_validation_profile(load_config(args.validation_profile), args.require_ready)
        if args.student_config:
            checks += check_student_config(load_config(args.student_config),
                                           check_files=not args.skip_files,
                                           require_ready=args.require_ready)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, TypeError) as exc:
        parser.error(str(exc))
    if not checks:
        parser.error("provide a student config or one of the named config options")
    for check in checks:
        print(f"PASS: {check}")


if __name__ == "__main__":
    main()
