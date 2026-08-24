"""Deterministic Stage-10/11 deployment resolution and NVE-segment protocol derivation.

This module removes the last hand-authored inputs from the production MD / physical-validation
path:

  * ``resolve_selected_checkpoint`` resolves the SINGLE canonical deployed Student checkpoint
    from a committee manifest by a GOVERNED seed selection -- the checkpoint *path* and its
    sha256 are derived from the manifest, never typed into a proposal by hand. It also carries
    the semantic guard that the committee-manifest sha256 is NOT the checkpoint sha256.

  * ``derive_nve_segment_protocol`` derives the dedicated NVE energy-conservation segment
    (warm-up + microcanonical run + thermo sampling) autonomously from the frozen
    ``shared_md_protocol`` of the run's validation profile. The NVE segment is DISTINCT from the
    thermostatted NVT production trajectory: its total-energy drift is a valid microcanonical
    energy-conservation metric, whereas the NVT production run's ``etotal`` is not.

  * ``build_deployment_context`` renders the placeholder context consumed by the LAMMPS
    templates (``prod_md.in.template`` for NVT production, ``nve_drift.in.template`` for the NVE
    segment) purely from the frozen protocol -- no hand-tuned step counts.

Every quantity is derived structurally from a frozen contract; where a defensible engineering
choice exists (thermostat damping constant, NVE observation-window length when the policy does
not name one explicitly) the choice is the minimal standard one and its rationale is recorded in
the returned payload rather than silently baked in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow.integrity import sha256_file


# The framework-canonical filename a training stage publishes its committee (per-seed checkpoint
# identity + integrity) under. This is an artifact-KIND name, never a material/campaign identifier;
# it is the single name the run-binding resolver and the training producer agree on.
CANONICAL_COMMITTEE_MANIFEST_NAME = "student_committee.manifest.json"


# --------------------------------------------------------------------------- checkpoint resolution


def _load_committee_manifest(committee_manifest) -> tuple[dict, str | None]:
    """Return (manifest_dict, manifest_sha256_or_None). Accepts a path or an in-memory dict."""
    if isinstance(committee_manifest, dict):
        return committee_manifest, None
    import json

    path = Path(committee_manifest)
    return json.loads(path.read_text()), sha256_file(path)


def resolve_published_committee_manifest(controller, *,
                                         manifest_name=CANONICAL_COMMITTEE_MANIFEST_NAME):
    """Resolve the canonical committee manifest PUBLISHED by an upstream training stage in THIS run.

    This is the run-binding that makes it impossible for a downstream deployment stage to resolve a
    checkpoint the training stage never published: the expected committee-manifest identity (path +
    sha256) is DERIVED from the run's own artifact registry, never taken from a proposal. A
    deployment stage feeds the returned ``sha256`` to ``resolve_selected_checkpoint`` as
    ``expected_manifest_sha256`` so a hand-named or foreign manifest is rejected.

    Accepts a ``RunController`` (reads ``controller.state``) or a state mapping directly, so it is
    unit-testable without a full controller. Fails closed if:
      * no artifact named ``manifest_name`` has been registered (training never published one);
      * the sole such artifact's producing stage has not completed;
      * more than one distinct committee manifest is active (ambiguous canonical set);
      * the registered manifest is missing on disk or its on-disk sha256 has drifted.
    """
    import json

    state = getattr(controller, "state", controller)
    artifacts = state.get("artifacts") or []
    stages = {s.get("name"): s for s in (state.get("stages") or [])}
    matches = [rec for rec in artifacts
               if Path(rec.get("path", "")).name == manifest_name]
    if not matches:
        raise ValueError(
            "no Student committee manifest has been published by any stage in this run; a "
            "deployment checkpoint cannot be resolved for a committee the training stage never "
            f"published (expected a registered artifact named {manifest_name!r})")
    published = []
    for rec in matches:
        stage = stages.get(rec.get("stage"))
        if stage is not None and stage.get("status") != "completed":
            continue
        published.append(rec)
    if not published:
        raise ValueError(
            "a committee manifest artifact exists but its producing (training) stage has not "
            "completed; the canonical published checkpoint set is not available yet")
    distinct = {rec.get("sha256") for rec in published}
    if len(distinct) != 1:
        raise ValueError(
            "more than one distinct committee manifest is active in this run; the canonical "
            f"published checkpoint set is ambiguous (sha256s: {sorted(distinct)})")
    record = published[-1]
    path = Path(record["path"])
    if not path.is_file():
        raise ValueError(
            f"published committee manifest is registered but missing on disk: {path}")
    on_disk = sha256_file(path)
    if record.get("sha256") is not None and record["sha256"] != on_disk:
        raise ValueError(
            "published committee manifest on disk does not match its registered sha256 -- artifact "
            "drift, refusing to bind a deployment checkpoint")
    manifest = json.loads(path.read_text())
    seeds = sorted(int(m["seed"]) for m in (manifest.get("models") or [])
                   if isinstance(m.get("seed"), int))
    return {
        "path": str(path.resolve()),
        "sha256": on_disk,
        "stage": record.get("stage"),
        "published_seeds": seeds,
    }


def resolve_selected_checkpoint(committee_manifest, *, selected_seed=None, select_by=None,
                                expected_manifest_sha256=None):
    """Resolve the canonical deployed Student checkpoint from a committee manifest.

    Selecting WHICH committee member is deployed is a governed scientific decision, so it must be
    made explicitly -- either by naming ``selected_seed`` or by a declared ``select_by`` policy
    that is fully determined by evidence already in the manifest. This function never invents a
    seed and never hand-types a checkpoint path: given the governed seed it derives the member's
    path + sha256 from the manifest, verifies the on-disk checkpoint, and returns an identity
    block suitable for ``deployment_provenance.json``.

    select_by:
      * ``"min_validation_loss"`` -- the member with the smallest numeric ``metadata.loss``.
        Fails closed if any member lacks a numeric loss (so a mock committee with
        ``loss: not_applicable`` cannot be silently ranked).
    """
    manifest, manifest_sha256 = _load_committee_manifest(committee_manifest)

    # Run-binding guard: when the deployment stage supplies the sha256 of the committee manifest the
    # training stage PUBLISHED in this run (derived by resolve_published_committee_manifest), the
    # manifest actually consumed here MUST be byte-identical to it. This is what makes it impossible
    # to deploy a checkpoint the training stage never published.
    if expected_manifest_sha256 is not None:
        if manifest_sha256 is None:
            raise ValueError(
                "expected_manifest_sha256 was supplied but the committee manifest was passed as an "
                "in-memory dict with no file identity; cannot verify it is the run-published "
                "manifest")
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "committee manifest consumed at deployment is not the one the training stage "
                "published in this run (sha256 mismatch); refusing to deploy a checkpoint the "
                "training stage never published")

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("committee manifest has no models to select a deployment checkpoint from")

    if selected_seed is not None and select_by is not None:
        raise ValueError("supply exactly one of selected_seed or select_by, not both")

    if selected_seed is not None:
        seed = int(selected_seed)
        derivation = f"governed explicit selected_seed={seed}"
    elif select_by == "min_validation_loss":
        losses = {}
        for model in models:
            meta = model.get("metadata") or {}
            loss = meta.get("loss")
            if not isinstance(loss, (int, float)):
                raise ValueError(
                    "select_by=min_validation_loss requires every committee member to carry a "
                    "numeric metadata.loss; member seed "
                    f"{model.get('seed')!r} has loss={loss!r}")
            losses[int(model["seed"])] = float(loss)
        seed = min(losses, key=lambda s: (losses[s], s))
        derivation = (
            f"governed select_by=min_validation_loss -> seed={seed} "
            f"(loss={losses[seed]!r}; ranked over {sorted(losses)})")
    else:
        raise ValueError(
            "deployment checkpoint selection is a governed decision: supply selected_seed "
            "(explicit) or a select_by policy determined by the committee manifest evidence")

    match = next((m for m in models if int(m.get("seed", -1)) == seed), None)
    if match is None:
        raise ValueError(
            f"selected_seed={seed} is not present in the committee manifest "
            f"(available seeds: {sorted(int(m.get('seed', -1)) for m in models)})")

    checkpoint_path = Path(match["path"])
    if not checkpoint_path.is_file():
        raise ValueError(
            f"resolved deployment checkpoint for seed={seed} does not exist on disk: "
            f"{checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)

    recorded = (match.get("integrity") or {}).get("sha256")
    if recorded is not None and recorded != checkpoint_sha256:
        raise ValueError(
            f"resolved deployment checkpoint for seed={seed} does not match the sha256 recorded "
            "in the committee manifest -- checkpoint drift, refusing to deploy")

    return {
        "selected_seed": seed,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_kind": match.get("kind"),
        "committee_manifest_sha256": manifest_sha256,
        "committee_manifest_sha_is_not_the_checkpoint_sha": (
            manifest_sha256 != checkpoint_sha256 if manifest_sha256 is not None else None),
        "cross_check_match": recorded is not None and recorded == checkpoint_sha256,
        "selection_derivation": derivation,
        "expected_manifest_sha256": expected_manifest_sha256,
        "published_manifest_binding_verified": (
            expected_manifest_sha256 is not None
            and manifest_sha256 == expected_manifest_sha256),
    }


# ---------------------------------------------------------------------- protocol derivation


def _require_number(protocol: dict, key: str) -> float:
    value = protocol.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"shared_md_protocol is missing a numeric {key!r}; cannot derive the deployment "
            "protocol without it (fail closed rather than invent a value)")
    return float(value)


def load_shared_md_protocol(validation_profile) -> dict:
    """Read the frozen ``shared_md_protocol`` block from a validation profile (path or dict)."""
    if isinstance(validation_profile, dict):
        profile = validation_profile
    else:
        import yaml

        profile = yaml.safe_load(Path(validation_profile).read_text())
    protocol = (profile or {}).get("shared_md_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            "validation profile has no shared_md_protocol block; the deployment MD protocol is "
            "not derivable without the frozen shared protocol")
    return protocol


def derive_nve_segment_protocol(shared_md_protocol: dict, *, nve_segment_ps=None):
    """Derive the dedicated NVE energy-conservation segment protocol from the frozen protocol.

    The microcanonical (NVE) segment is a validation artifact SEPARATE from the thermostatted NVT
    production trajectory. Standard deterministic MLIP energy-conservation validation is: reach
    the target temperature under a thermostat, then remove the thermostat and integrate NVE while
    logging total energy vs. time, then fit the drift. Every quantity below is derived from the
    frozen ``shared_md_protocol``; the two engineering choices (thermostat damping, and the NVE
    observation window when the policy names no NVE-specific length) use the minimal standard rule
    and record their rationale.
    """
    timestep_fs = _require_number(shared_md_protocol, "timestep_fs")
    temperature_K = _require_number(shared_md_protocol, "temperature_K")
    equilibration_ps = _require_number(shared_md_protocol, "nvt_equilibration_ps")
    sampling_interval_fs = _require_number(shared_md_protocol, "sampling_interval_fs")
    if timestep_fs <= 0 or sampling_interval_fs <= 0 or equilibration_ps <= 0:
        raise ValueError("shared_md_protocol timestep/sampling/equilibration must be positive")

    timestep_ps = timestep_fs / 1000.0
    rationale = {}

    # NVE observation window: prefer an explicitly frozen NVE length; otherwise mirror the
    # equilibration window (observe microcanonical conservation over a window at least as long as
    # the time spent reaching the target state -- the minimal defensible choice).
    if nve_segment_ps is None:
        nve_segment_ps = float(shared_md_protocol.get("nve_production_ps") or 0.0) or None
    if nve_segment_ps is None:
        nve_segment_ps = equilibration_ps
        rationale["nve_segment_ps"] = (
            "no NVE-specific window frozen in shared_md_protocol; autonomously set equal to "
            f"nvt_equilibration_ps ({equilibration_ps} ps) so the microcanonical observation "
            "window is at least as long as the equilibration window")
    else:
        rationale["nve_segment_ps"] = f"frozen NVE window = {nve_segment_ps} ps"

    # Thermostat damping for the warm-up: the standard Nose-Hoover rule of thumb is ~100 timesteps.
    tdamp_ps = 100.0 * timestep_ps
    rationale["tdamp_ps"] = (
        "warm-up thermostat damping set to the standard ~100*timestep Nose-Hoover rule of thumb "
        f"({tdamp_ps} ps); used only during the thermostatted warm-up, not the NVE segment")

    warmup_steps = int(round(equilibration_ps / timestep_ps))
    nve_steps = int(round(nve_segment_ps / timestep_ps))
    thermo_every_steps = int(round(sampling_interval_fs / timestep_fs))
    if warmup_steps < 1 or nve_steps < 2 or thermo_every_steps < 1:
        raise ValueError("derived NVE protocol is degenerate (need warmup>=1, nve>=2 steps)")
    n_expected_samples = nve_steps // thermo_every_steps + 1
    if n_expected_samples < 2:
        raise ValueError(
            "derived NVE protocol samples fewer than 2 energy points; drift is not fittable")

    return {
        "ensemble_role": "nve_energy_conservation_segment",
        "distinct_from_production": (
            "this NVE segment is separate from the thermostatted NVT production trajectory; only "
            "this segment's total-energy drift is a valid microcanonical conservation metric"),
        "temperature_K": temperature_K,
        "timestep_fs": timestep_fs,
        "timestep_ps": timestep_ps,
        "tdamp_ps": tdamp_ps,
        "equilibration_ps": equilibration_ps,
        "nve_segment_ps": nve_segment_ps,
        "warmup_steps": warmup_steps,
        "nve_steps": nve_steps,
        "thermo_every_steps": thermo_every_steps,
        "sampling_interval_fs": sampling_interval_fs,
        "n_expected_energy_samples": n_expected_samples,
        "autonomous_choice_rationale": rationale,
    }


def build_deployment_context(shared_md_protocol: dict, ensemble: str, datafile, *,
                             velocity_seed: int, mpi_ranks: int = 1, dump_file: str | None = None,
                             energy_log: str | None = None, nve_segment_ps=None) -> dict:
    """Build the LAMMPS template context dict for a deployment run.

    ensemble:
      * ``"nvt"`` -> production trajectory context (prod_md.in.template placeholders).
      * ``"nve"`` -> dedicated energy-conservation segment (nve_drift.in.template placeholders).
    """
    timestep_fs = _require_number(shared_md_protocol, "timestep_fs")
    temperature_K = _require_number(shared_md_protocol, "temperature_K")
    timestep_ps = timestep_fs / 1000.0
    common: dict[str, Any] = {
        "DATAFILE": str(datafile),
        "TIMESTEP_PS": timestep_ps,
        "TEMPERATURE_K": temperature_K,
        "SEED": int(velocity_seed),
        "MPI_RANKS": int(mpi_ranks),
    }

    if ensemble == "nvt":
        production_ps = _require_number(shared_md_protocol, "nvt_production_ps")
        sampling_interval_fs = _require_number(shared_md_protocol, "sampling_interval_fs")
        tdamp_ps = 100.0 * timestep_ps
        thermo_every = int(round(sampling_interval_fs / timestep_fs))
        n_steps = int(round(production_ps / timestep_ps))
        common.update({
            "TDAMP_PS": tdamp_ps,
            "THERMO_EVERY_STEPS": thermo_every,
            "DUMP_EVERY_STEPS": thermo_every,
            "DUMP_FILE": dump_file or "trajectory.dump",
            "N_STEPS": n_steps,
        })
        return common

    if ensemble == "nve":
        nve = derive_nve_segment_protocol(shared_md_protocol, nve_segment_ps=nve_segment_ps)
        common.update({
            "TDAMP_PS": nve["tdamp_ps"],
            "WARMUP_STEPS": nve["warmup_steps"],
            "THERMO_EVERY_STEPS": nve["thermo_every_steps"],
            "N_STEPS": nve["nve_steps"],
            "ENERGY_LOG": energy_log or "nve_energy.csv",
        })
        common["_nve_protocol"] = nve
        return common

    raise ValueError(f"unknown deployment ensemble {ensemble!r}; expected 'nvt' or 'nve'")


def build_deployment_provenance(student_identity: dict, *, starting_structure: dict,
                                ensemble_role: str, shared_md_protocol: dict | None = None,
                                nve_protocol: dict | None = None, extra: dict | None = None) -> dict:
    """Assemble the deterministic core of ``deployment_provenance.json``.

    Records the resolved Student checkpoint identity, the starting-structure identity, the
    ensemble role, and (for NVE segments) the derived NVE protocol + rationale. Backend/preflight
    provenance is attached by the MD run itself; this is the pre-run, resolvable core.
    """
    provenance = {
        "schema_version": 1,
        "ensemble_role": ensemble_role,
        "student": student_identity,
        "starting_structure": starting_structure,
    }
    if shared_md_protocol is not None:
        provenance["shared_md_protocol"] = shared_md_protocol
    if nve_protocol is not None:
        provenance["nve_protocol"] = nve_protocol
    if extra:
        provenance.update(extra)
    return provenance
