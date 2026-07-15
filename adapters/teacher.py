"""Teacher adapter: given configs/teacher.<name>.yaml, return a usable ASE
calculator. Any teacher exposing an ASE Calculator satisfies this interface —
that already covers NequIP/Allegro, MACE, GAP (via quippy), ACE (via pyace),
and foundation models (MACE-MP-0, MatterSim, Orb, ...).

Adding a new `kind`: import the calculator class dynamically (already generic
below) and add any `kind`-specific construction quirks as a branch in
`load_teacher`. Most teachers need NO new code here at all — only a new
configs/teacher.<name>.yaml with the right `calculator.module`/`calculator.class`.
"""
import importlib


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
        kwargs[model_arg] = cfg.get("model", cfg.get("checkpoint"))
        return factory(**kwargs)
    module = importlib.import_module(calc_cfg["module"])
    calc_cls = getattr(module, calc_cfg["class"])

    kind = cfg["kind"]
    if kind == "allegro":
        # nequip/allegro calculators are constructed from a compiled checkpoint path.
        return calc_cls.from_deployed_model(cfg["checkpoint"])
    elif kind == "mace":
        # MACE's calculator takes the checkpoint path + configurable kwargs.
        kwargs = dict(calc_cfg.get("kwargs", {}))
        kwargs.setdefault("device", "cpu")
        return calc_cls(model_paths=cfg["checkpoint"], **kwargs)
    else:
        raise NotImplementedError(
            f"teacher kind={kind!r} has no construction recipe in adapters/teacher.py yet. "
            f"Add one here (usually a few lines — see the `allegro`/`mace` branches above) "
            f"and update configs/README.md."
        )


def check_stress_support(cfg, test_atoms):
    """Empirically confirm whether a teacher checkpoint emits stress/virial.

    Do NOT assume `emits_stress` in the config is correct — verify against the
    actual compiled/deployed checkpoint (different builds of the same
    architecture may or may not include the stress head). See the Pass-1
    handoff's open item 1.

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
