"""Structure acquisition and teacher pseudo-labeling backends."""
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import yaml
from ase import units
from ase.constraints import FixCom
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from adapters import load_config, resolve_config_path
from adapters.teacher import (load_teacher, load_teacher_with_species_evidence,
                              teacher_model_reference)
from workflow.integrity import artifact_digest
from workflow.subprocess_runner import run_bounded


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class AcquisitionTimeoutError(TimeoutError):
    """Raised when the augment-atoms subprocess exceeds its configured ``timeout_s`` wall-clock
    budget. Deliberately a ``TimeoutError`` subclass (name ends in "TimeoutError") so the generic,
    forward-compatible detection in ``runtimes.pydantic_ai.cli.run_production_stage`` recognizes it
    from the ``EXECUTOR_ERROR`` reason string and routes the stage through the Controller's own
    ``timeout_stage_execution`` + FAIL-gate recovery path instead of leaving it silently hung --
    the exact R28 regression class this module's bounded execution now makes representable."""


class AcquisitionFeasibilityError(ValueError):
    """Raised by ``check_acquisition_feasibility`` when a configured augment-atoms
    rejection-sampling parameter combination is structurally infeasible -- i.e. the R28
    pathology (see runs/sio2-sox-allegro-simplenn-r28/artifacts/r28_workflow_failure_report.json):
    ``augment_atoms.generate_structures()``'s outer while loop has no cap on rejected attempts,
    and only accepts a candidate once its cumulative per-atom displacement from every existing
    pool member exceeds ``similarity_threshold``. If the largest displacement the sampler can
    plausibly ever produce does not clear that threshold by a comfortable margin, acceptance is a
    rare-to-impossible tail event and the subprocess can run unboundedly by design. Raised BEFORE
    any subprocess is dispatched, so it is an ordinary (non-timeout) exception -- it takes the
    same fast, deterministic ``defer_stage_execution``-to-pending path as any other pre-dispatch
    validation failure, never the timeout/FAIL path."""


def _harmonic_number(n):
    return sum(1.0 / i for i in range(1, int(n) + 1))


def check_acquisition_feasibility(config_section, *, margin=1.5):
    """Pre-execution sanity check on augment-atoms's own resolved ``config`` parameters.

    Deliberately generic: it never hard-codes a specific ``similarity_threshold`` value (or any
    other parameter value) -- it only checks that whatever ``sigma_range``/``max_relax_steps``/
    ``similarity_threshold`` a plan actually declares are mutually feasible, so a legitimately
    tighter or looser plan is free to use any values it likes as long as they remain feasible.

    Per-atom displacement from the rattle+relax walk is bounded above by
    ``max(sigma_range) * sum(1/i for i in 1..max_relax_steps)`` (the worst-case cumulative
    relaxation-step walk at the largest sampled perturbation scale). Returns the computed bound
    when all three parameters are present and feasible; returns ``None`` (no-op) when any of them
    is absent, since there is then nothing concrete enough to evaluate.
    """
    sigma_range = config_section.get("sigma_range")
    similarity_threshold = config_section.get("similarity_threshold")
    max_relax_steps = config_section.get("max_relax_steps")
    if not sigma_range or similarity_threshold is None or not max_relax_steps:
        return None
    sigma_max = float(max(sigma_range))
    harmonic = _harmonic_number(max_relax_steps)
    reach = sigma_max * harmonic
    threshold = float(similarity_threshold)
    if reach < margin * threshold:
        raise AcquisitionFeasibilityError(
            "acquisition parameters are structurally infeasible for the rejection-sampling loop: "
            f"max(sigma_range)={sigma_max:g} * harmonic(max_relax_steps={max_relax_steps})="
            f"{harmonic:.4f} gives an estimated maximum plausible cumulative per-atom displacement "
            f"of {reach:.4f} A, which does not clear similarity_threshold={threshold:g} by the "
            f"required {margin:g}x safety margin. Under this combination, augment_atoms's "
            "too_similar() rejection is expected to fire on effectively every candidate, so its "
            "uncapped outer while loop can run for a practically unbounded time (the R28 "
            "regression). Revise sigma_range, max_relax_steps, or similarity_threshold so the "
            "bound comfortably exceeds the threshold, then resubmit for approval."
        )
    return reach


def langevin_friction(cfg):
    """Return friction in inverse ASE time units from an explicitly unit-tagged config."""
    if "friction_per_fs" in cfg:
        return float(cfg["friction_per_fs"]) / units.fs
    if "friction_ase_time_inverse" in cfg:
        return float(cfg["friction_ase_time_inverse"])
    if "friction" in cfg:
        raise ValueError(
            "ambiguous teacher-MD field 'friction'; use friction_per_fs or "
            "friction_ase_time_inverse"
        )
    raise ValueError("teacher-MD config requires friction_per_fs")


def run_augment_atoms(cfg, seed_path, out_path, *, progress_cb=None):
    """Run a configured augment-atoms wrapper without assuming its CLI version.

    Bounded via ``workflow.subprocess_runner.run_bounded`` (R28 forensic-defect correction): an
    optional ``cfg["timeout_s"]`` wall-clock budget (``None`` means unbounded, exactly as before
    this fix) terminates ONLY this subprocess's own process group on expiry and raises
    ``AcquisitionTimeoutError`` -- never silently hangs the calling Controller-dispatched attempt.
    ``progress_cb``, if given, is called with ``{"pid": <int or None>}`` at most every few seconds
    while the subprocess is still running -- the same liveness mechanism every other trusted
    executor uses via ``dispatch.authorize_and_execute``'s ``progress_cb`` passthrough, so a
    long-running acquisition attempt is no longer silent between dispatch and return.
    """
    context = {
        "config_path": str(Path(cfg["config_path"]).resolve()) if cfg.get("config_path") else "",
        "seed_path": str(Path(seed_path).resolve()),
        "out_path": str(Path(out_path).resolve()),
    }
    command = cfg.get("command")
    if command:
        command = [str(part).format(**context) for part in command]
    else:
        cli = cfg.get("cli") or {}
        invocation = cli.get("invocation")
        if not isinstance(invocation, list) or not invocation:
            raise ValueError("augment-atoms acquisition requires command or cli.invocation")
        command = [str(part).format(**context) for part in invocation]
        executable = Path(command[0]).expanduser()
        if executable.is_absolute():
            if not executable.is_file():
                raise FileNotFoundError(f"augment-atoms executable is missing: {executable}")
            command[0] = str(executable)
        elif cfg.get("env"):
            command = ["conda", "run", "-n", str(cfg["env"]), *command]
        else:
            raise ValueError(
                "augment-atoms executable must be absolute or acquisition env must be configured"
            )
    workdir = resolve_config_path(cfg, cfg["workdir"]) if cfg.get("workdir") else None
    config_path = cfg.get("config_path")
    if config_path and Path(config_path).exists():
        native_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        check_acquisition_feasibility(native_cfg.get("config", {}) or {})
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (str(repo_root) if not existing_pythonpath
                         else str(repo_root) + os.pathsep + existing_pythonpath)
    timeout_s = cfg.get("timeout_s")
    pid_box: dict = {}

    def _on_start(pid):
        pid_box["pid"] = pid

    def _heartbeat():
        if progress_cb is not None:
            progress_cb({"pid": pid_box.get("pid")})

    # stdout/stderr left as None (inherited) to preserve run_augment_atoms's pre-existing
    # passthrough behavior exactly -- only the timeout/process-group/heartbeat wiring is new.
    bounded = run_bounded(command, cwd=workdir, env=env, stdout=None, stderr=None,
                         timeout_s=timeout_s, on_start=_on_start, heartbeat_cb=_heartbeat)
    if bounded.timed_out:
        raise AcquisitionTimeoutError(
            f"augment-atoms command timed out after {timeout_s}s (pid={bounded.pid})")
    if bounded.returncode != 0:
        raise subprocess.CalledProcessError(bounded.returncode, command)
    if not Path(out_path).exists():
        raise FileNotFoundError(f"augment-atoms command produced no output: {out_path}")
    return Path(out_path)


def run_teacher_md(cfg, teacher_cfg, seed_path, out_path, capture_labels=False):
    """Generate snapshots by Langevin MD under the teacher ASE calculator.

    ``capture_labels=True`` additionally records ``teacher_energy``/``teacher_forces`` on each
    snapshot (computed while the live MD ``atoms`` object still has its calculator attached, since
    ``atoms.copy()`` drops it) -- used by the teacher_baseline dynamic sanity check, never by the
    default acquisition/sampling path, so acquisition output is unchanged by default.
    """
    seeds = read(seed_path, index=":")
    calc = load_teacher(teacher_cfg)
    snapshots = []
    for seed_index, source in enumerate(seeds):
        parent_id = source.info.get("parent_structure_id",
                                    source.info.get("structure_id", f"seed-{seed_index:08d}"))
        atoms = source.copy()
        # Snapshot constraint policy: FixCom is a whole-system center-of-mass
        # dynamics device, not per-atom static data, and ASE 3.26's extxyz writer
        # cannot serialize it. We therefore omit EVERY FixCom (seed-provided or
        # runtime-added) from the recorded snapshots, while preserving all other
        # original constraints (FixAtoms, FixCartesian, ...) exactly. Deep-copied so
        # the snapshot does not depend on object identity surviving atoms.copy().
        original_constraints = copy.deepcopy(source.constraints)
        source_had_fixcom = any(isinstance(item, FixCom)
                                for item in original_constraints)
        snapshot_constraints = [copy.deepcopy(item) for item in original_constraints
                                if not isinstance(item, FixCom)]
        fix_center_of_mass = cfg.get("fix_center_of_mass", True)
        runtime_fixcom_added = False
        if fix_center_of_mass and not source_had_fixcom:
            # Added only to the live MD atoms object, never to the snapshot copies.
            atoms.set_constraint([*atoms.constraints, FixCom()])
            runtime_fixcom_added = True
        atoms.calc = calc
        temperature = float(cfg["temperature_K"])
        random_seed = int(cfg.get("seed", 0)) + seed_index
        rng = np.random.default_rng(random_seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                     rng=rng)
        friction = langevin_friction(cfg)
        dyn = Langevin(atoms, float(cfg.get("timestep_fs", 1.0)) * units.fs,
                       temperature_K=temperature, friction=friction, rng=rng,
                       fixcm=False)
        stride = int(cfg.get("snapshot_interval", 100))
        n_steps = int(cfg["n_steps"])

        def capture():
            if capture_labels:
                label_energy = float(atoms.get_potential_energy())
                label_forces = np.asarray(atoms.get_forces(), dtype=float)
            frame = atoms.copy()
            # Apply the FixCom-free snapshot constraints (never mutating the live MD
            # atoms), and record scalar provenance so an omitted FixCom is auditable
            # rather than silently dropped. Scalars only (bool) — list/dict info keys
            # are unstable across ASE extxyz versions.
            frame.set_constraint(copy.deepcopy(snapshot_constraints))
            frame.info["source_had_fixcom"] = bool(source_had_fixcom)
            frame.info["runtime_fixcom_applied"] = bool(runtime_fixcom_added)
            frame.info["snapshot_fixcom_omitted"] = bool(
                source_had_fixcom or runtime_fixcom_added)
            frame.info.update(acquisition="teacher-md", seed_structure_index=seed_index,
                              temperature_K=temperature, parent_structure_id=str(parent_id),
                              random_seed=random_seed,
                              fix_center_of_mass=bool(fix_center_of_mass),
                              timestep_fs=float(cfg.get("timestep_fs", 1.0)),
                              n_steps=n_steps, snapshot_interval=stride)
            if "friction_per_fs" in cfg:
                frame.info["friction_per_fs"] = float(cfg["friction_per_fs"])
            else:
                frame.info["friction_ase_time_inverse"] = float(
                    cfg["friction_ase_time_inverse"]
                )
            if capture_labels:
                frame.info["teacher_energy"] = label_energy
                frame.arrays["teacher_forces"] = label_forces
            snapshots.append(frame)

        dyn.attach(capture, interval=stride)
        dyn.run(n_steps)
    write(out_path, snapshots)
    return Path(out_path)


def acquire(acquisition_cfg, teacher_cfg, seed_path, out_path, *, progress_cb=None):
    """``progress_cb``, if given, is only wired to the built-in ``augment-atoms`` recipe (see
    ``run_augment_atoms``) -- a configured ``adapter.acquire`` callable's signature is not
    controlled by this module, so it is invoked exactly as before, unaffected."""
    kind = acquisition_cfg["kind"]
    adapter = acquisition_cfg.get("adapter", {}).get("acquire")
    if adapter:
        module_name, name = adapter.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), name, None)
        if not callable(function):
            raise TypeError(f"configured acquisition callable is invalid: {adapter}")
        result = Path(function(acquisition_cfg, teacher_cfg, seed_path, out_path))
        validate_lineage(result)
        return result
    if kind == "augment-atoms":
        result = run_augment_atoms(acquisition_cfg, seed_path, out_path, progress_cb=progress_cb)
        if not acquisition_cfg.get("defer_lineage_validation"):
            validate_lineage(result)
        return result
    if kind == "teacher-md":
        result = run_teacher_md(acquisition_cfg, teacher_cfg, seed_path, out_path)
        validate_lineage(result)
        return result
    raise NotImplementedError(
        f"acquisition kind={kind!r} requires adapter.acquire or a built-in recipe"
    )


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
    calc, species_mapping_evidence = load_teacher_with_species_evidence(teacher_cfg)
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
    package_names = list(dict.fromkeys(
        ["ase", "numpy", *teacher_cfg.get("provenance", {}).get("packages", [])]
    ))
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema_version": 1,
        "teacher_kind": teacher_cfg["kind"],
        "teacher_model": model_value,
        "teacher_model_integrity": (artifact_digest(model_path)
                                    if model_path and model_path.exists() else None),
        "teacher_model_sha256": (_sha256(model_path)
                                 if model_path and model_path.is_file() else None),
        "teacher_head": teacher_cfg.get("calculator", {}).get("kwargs", {}).get("head"),
        "calculator": teacher_cfg.get("calculator", {}),
        # Actual runtime-resolved species/type mapping the calculator was constructed with (and
        # whether the identity-mapping fallback in adapters.teacher.load_teacher was applied) --
        # distinct from `calculator` above, which is only the DECLARED config, never the resolved
        # runtime state.
        "species_mapping_evidence": species_mapping_evidence,
        "teacher_config_integrity": artifact_digest(config_path) if config_path else None,
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
