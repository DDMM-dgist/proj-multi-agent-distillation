"""Cheap validation of a run configuration before expensive work starts."""
import importlib
import shutil
from pathlib import Path

from adapters import load_config, resolve_config_path


def _require(cfg, dotted):
    value = cfg
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing required config field: {dotted}")
        value = value[key]
    return value


def check_student_config(cfg, check_files=True):
    """Return a list of human-readable checks; raise on an invalid contract."""
    kind = _require(cfg, "kind")
    elements = _require(cfg, "deploy.elements")
    if not isinstance(elements, list) or not elements or len(set(elements)) != len(elements):
        raise ValueError("deploy.elements must be a non-empty list of unique element symbols")

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
    parser.add_argument("student_config")
    parser.add_argument("--skip-files", action="store_true", help="validate schema only")
    args = parser.parse_args()
    for check in check_student_config(load_config(args.student_config), check_files=not args.skip_files):
        print(f"PASS: {check}")


if __name__ == "__main__":
    main()
