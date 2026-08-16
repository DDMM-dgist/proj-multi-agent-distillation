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


def _species_mapping_evidence(declared_kwargs: dict, runtime_kwargs: dict,
                              fallback_reason) -> dict:
    """Deterministic evidence of the actual runtime species/type mapping a Teacher calculator was
    constructed with, and whether the identity-mapping fallback/reconciliation below was applied.

    Derived only from the real, already-resolved calculator kwargs at construction time (never an
    LLM's interpretation, and never a hardcoded material-specific mapping introduced here -- the
    mapping's actual contents, whatever species it names, come entirely from the config/runtime
    state being reported on).
    """
    return {
        "declared_chemical_symbols": declared_kwargs.get("chemical_symbols"),
        "declared_chemical_species_to_atom_type_map":
            declared_kwargs.get("chemical_species_to_atom_type_map"),
        "runtime_chemical_species_to_atom_type_map":
            runtime_kwargs.get("chemical_species_to_atom_type_map"),
        "fallback_applied": fallback_reason is not None,
        "fallback_reason": fallback_reason,
    }


def species_mapping_is_attested(evidence: dict) -> bool:
    """Whether `_species_mapping_evidence`'s output for one Teacher construction is internally
    consistent and, where the calculator's own config actually declares a chemical_symbols /
    chemical_species_to_atom_type_map convention (or the fallback below was applied), backed by a
    real resolved runtime mapping -- never manufactured for calculator kinds (e.g. EMT, foundation
    models that auto-detect species) that use no such convention at all, since requiring one
    universally would itself be a hardcoded, architecture-specific assumption."""
    if not isinstance(evidence, dict) or not isinstance(evidence.get("fallback_applied"), bool):
        return False
    declares_mapping_convention = bool(
        evidence.get("declared_chemical_symbols")
        or evidence.get("declared_chemical_species_to_atom_type_map")
        or evidence.get("fallback_applied")
    )
    if not declares_mapping_convention:
        return True
    runtime_mapping = evidence.get("runtime_chemical_species_to_atom_type_map")
    return isinstance(runtime_mapping, dict) and bool(runtime_mapping)


def load_teacher_with_species_evidence(cfg):
    """Instantiate the teacher's ASE calculator from its config, exactly like `load_teacher`, but
    also return deterministic evidence (see `_species_mapping_evidence`) of the actual runtime
    species/type mapping used and whether the identity-mapping fallback below was applied --
    needed so a caller (e.g. `adapters.acquisition.label_with_teacher`) can attest this in a
    provenance manifest, rather than only ever recording the declared config.

    Returns (calculator, species_mapping_evidence).
    """
    calc_cfg = cfg["calculator"]
    if "factory" in calc_cfg:
        module_name, callable_name = calc_cfg["factory"].rsplit(".", 1)
        factory = getattr(importlib.import_module(module_name), callable_name)
        kwargs = dict(calc_cfg.get("kwargs", {}))
        declared_kwargs = dict(kwargs)
        model_arg = calc_cfg.get("model_arg", "model")
        model = teacher_model_reference(cfg)
        if model_arg == "__positional__":
            calc = factory(model, **kwargs)
        else:
            if model_arg:
                kwargs[model_arg] = model
            calc = factory(**kwargs)
        return calc, _species_mapping_evidence(declared_kwargs, kwargs, None)
    module = importlib.import_module(calc_cfg["module"])
    calc_cls = getattr(module, calc_cfg["class"])
    constructor = getattr(calc_cls, calc_cfg["constructor"]) \
        if calc_cfg.get("constructor") else calc_cls
    kwargs = dict(calc_cfg.get("kwargs", {}))
    declared_kwargs = dict(kwargs)
    model_arg = calc_cfg.get("model_arg", "model")
    model = teacher_model_reference(cfg)
    fallback_reason = None
    if model_arg == "__positional__":
        try:
            calc = constructor(model, **kwargs)
        except ValueError as exc:
            if "chemical_symbols" not in kwargs or "chemical_species_to_atom_type_map" not in str(exc):
                raise
            symbols = list(kwargs.pop("chemical_symbols"))
            kwargs["chemical_species_to_atom_type_map"] = {symbol: symbol for symbol in symbols}
            fallback_reason = str(exc)
            calc = constructor(model, **kwargs)
        return calc, _species_mapping_evidence(declared_kwargs, kwargs, fallback_reason)
    if model_arg:
        kwargs[model_arg] = model
    try:
        calc = constructor(**kwargs)
    except ValueError as exc:
        if "chemical_symbols" not in kwargs or "chemical_species_to_atom_type_map" not in str(exc):
            raise
        symbols = list(kwargs.pop("chemical_symbols"))
        kwargs["chemical_species_to_atom_type_map"] = {symbol: symbol for symbol in symbols}
        fallback_reason = str(exc)
        calc = constructor(**kwargs)
    return calc, _species_mapping_evidence(declared_kwargs, kwargs, fallback_reason)


def load_teacher(cfg):
    """Instantiate the teacher's ASE calculator from its config.

    cfg: a dict loaded from configs/teacher.<name>.yaml (see adapters.load_config).
    Returns an ase.calculators.calculator.Calculator instance.
    """
    calc, _evidence = load_teacher_with_species_evidence(cfg)
    return calc


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
