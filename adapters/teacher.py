"""Teacher adapter: given configs/teacher.<name>.yaml, return a usable ASE
calculator. Any teacher exposing an ASE Calculator satisfies this interface —
that already covers NequIP/Allegro, MACE, GAP (via quippy), ACE (via pyace),
and foundation models (MACE-MP-0, MatterSim, Orb, ...).

Adding a new `kind` normally needs only a config with `calculator.factory` or
`module`/`class` plus optional `constructor` and `model_arg`; the core does not
dispatch on a teacher name.
"""
import importlib

from adapters import resolve_config_path


def teacher_model_reference(cfg):
    """Resolve a model/checkpoint path, unless config marks it as a named model."""
    value = cfg.get("model", cfg.get("checkpoint"))
    if value is None:
        return None
    if cfg.get("calculator", {}).get("model_is_path", True):
        return str(resolve_config_path(cfg, value))
    return value


def load_teacher(cfg):
    """Instantiate the teacher's ASE calculator from its config.

    cfg: a dict loaded from configs/teacher.<name>.yaml (see adapters.load_config).
    Returns an ase.calculators.calculator.Calculator instance.
    """
    calc_cfg = cfg["calculator"]
    if "factory" in calc_cfg:
        module_name, callable_name = calc_cfg["factory"].rsplit(".", 1)
        factory = getattr(importlib.import_module(module_name), callable_name)
        kwargs = dict(calc_cfg.get("kwargs", {}))
        model_arg = calc_cfg.get("model_arg", "model")
        model = teacher_model_reference(cfg)
        if model_arg == "__positional__":
            return factory(model, **kwargs)
        if model_arg:
            kwargs[model_arg] = model
        return factory(**kwargs)
    module = importlib.import_module(calc_cfg["module"])
    calc_cls = getattr(module, calc_cfg["class"])
    constructor = getattr(calc_cls, calc_cfg["constructor"]) \
        if calc_cfg.get("constructor") else calc_cls
    kwargs = dict(calc_cfg.get("kwargs", {}))
    model_arg = calc_cfg.get("model_arg", "model")
    model = teacher_model_reference(cfg)
    if model_arg == "__positional__":
        return constructor(model, **kwargs)
    if model_arg:
        kwargs[model_arg] = model
    return constructor(**kwargs)


def check_stress_support(cfg, test_atoms):
    """Empirically confirm whether a teacher checkpoint emits stress/virial.

    Do NOT assume `emits_stress` in the config is correct — verify against the
    actual compiled/deployed checkpoint (different builds of the same
    architecture may or may not include the stress head). Confirm it against
    the actual checkpoint before using stress-derived observables.

    test_atoms: one ase.Atoms with a sensible cell (periodic), for a quick probe.
    Returns True/False and prints what it found.
    """
    calc = load_teacher(cfg)
    test_atoms.calc = calc
    try:
        stress = test_atoms.get_stress(voigt=False)
        ok = stress is not None and stress.shape == (3, 3)
    except Exception as e:  # noqa: BLE001 — this IS the check; report and return False
        print(f"[check_stress_support] teacher {cfg.get('checkpoint')} does NOT expose stress: {e}")
        return False
    print(f"[check_stress_support] teacher {cfg.get('checkpoint')} stress OK: {stress}")
    return ok
