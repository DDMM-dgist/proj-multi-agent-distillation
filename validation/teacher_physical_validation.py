"""Objective/profile-conditioned Teacher physical (dynamical) validation.

This module upgrades Teacher-side physical validation into a first-class,
provenance-bound stage WITHOUT reimplementing any observable: every physical
observable is computed by the SAME model-independent kernels in
``validation.structure_dynamics`` that the Student physical-validation path
(``runtimes.pydantic_ai.executors._exec_build_physical_validation_report`` /
Stage 11) already uses. The single shared dispatcher below,
``evaluate_observable``, is what both the Teacher (Stage 2, target
establishment) and the Student (Stage 11, target reproduction) call — so "the
Teacher and the Student measured the same thing the same way" is guaranteed by
construction, not by convention.

Two Teacher-side entry points produce one frozen artifact:

  * COMPUTE mode (``compute_teacher_validation_target``) drives the Teacher PES
    under a declared MD protocol — reusing the SAME ASE integrator reference
    (``framework_v2.acquisition.generators.teacher_dynamics``) already used for
    Teacher-driven acquisition — samples a trajectory, computes the profile's
    observables, and freezes a hash-bound ``TeacherValidationTarget``.

  * INGEST mode (``ingest_teacher_validation_target``) accepts a pre-existing
    Teacher trajectory (and/or externally computed observable evidence),
    validates its provenance/hashes, re-derives the observables through the same
    engine when the raw trajectory is available, and freezes the SAME target —
    never re-running expensive Teacher MD unnecessarily and never constructing a
    Teacher calculator.

Stage 11 consumes the frozen target through
``compare_student_to_teacher_target``: it re-evaluates the frozen observable
DEFINITIONS on the Student trajectory (the Student may supply only its own
trajectory, never redefine the Teacher's observables or thresholds) and reports
per-observable reproduction. Deterministic first; a Judge evaluates the bounded
scientific evidence afterwards.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from validation import structure_dynamics as sd
from validation.report import criterion_passes
from workflow.integrity import sha256_file

# Status vocabulary for a single observable measurement. COMPUTED = a real number
# was produced from real data; UNAVAILABLE = the data required for this observable
# is not present in this trajectory/context (never fabricated); NOT_APPLICABLE =
# the profile/objective did not select this observable for this run.
COMPUTED = "COMPUTED"
UNAVAILABLE = "UNAVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"

TARGET_SCHEMA_VERSION = 1
_HASH_EXCLUDED_FIELDS = ("target_sha256", "frozen_at")


# --------------------------------------------------------------------------------------------
# Shared observable dispatch — the ONE implementation both Teacher and Student go through.
# --------------------------------------------------------------------------------------------

def _species_present(frames, symbol):
    symbol = str(symbol)
    return any(symbol in atoms.get_chemical_symbols() for atoms in frames)


def _require(params, key, observable_name):
    if key not in params:
        raise ValueError(f"observable {observable_name!r} is missing required param {key!r}")
    return params[key]


def _result(spec, status, *, value=None, details=None, reason=None):
    row = {
        "name": spec.get("name"),
        "kind": spec.get("kind"),
        "status": status,
        "value": value,
        "units": spec.get("units"),
    }
    if details is not None:
        row["details"] = details
    if reason:
        row["reason"] = reason
    return row


def evaluate_observable(spec, frames, *, context=None):
    """Compute ONE observable defined by ``spec`` on ``frames``.

    ``spec`` is a plain dict: ``{name, kind, units?, params?, applicable_objectives?,
    comparison_criterion?}``. This is the single shared entry point used for BOTH
    the Teacher target and the Student reproduction, so the two are guaranteed to
    use byte-identical observable code. Returns a result row whose ``status`` is
    one of COMPUTED / UNAVAILABLE / NOT_APPLICABLE. It never fabricates a value: a
    missing prerequisite yields UNAVAILABLE with a reason, a malformed spec raises.
    """
    context = context or {}
    kind = spec.get("kind")
    name = spec.get("name")
    if not kind or not name:
        raise ValueError("observable spec requires 'name' and 'kind'")
    params = spec.get("params") or {}

    # Objective conditioning: a spec that declares applicable_objectives is only
    # evaluated when the run's active objectives intersect them.
    applicable = spec.get("applicable_objectives")
    active = context.get("objectives")
    if applicable and active is not None and not (set(applicable) & set(active)):
        return _result(spec, NOT_APPLICABLE,
                       reason=f"objectives {sorted(set(active))} do not select this observable")

    if not frames:
        raise ValueError("evaluate_observable requires at least one frame")

    if kind in ("rdf_peak_position", "rdf_peak_height"):
        center = _require(params, "center_species", name)
        neighbor = _require(params, "neighbor_species", name)
        if not _species_present(frames, center) or not _species_present(frames, neighbor):
            return _result(spec, UNAVAILABLE,
                           reason=f"species {center}/{neighbor} not present in trajectory")
        rdf = sd.compute_rdf_v2(frames, center, neighbor,
                                r_max=float(params.get("r_max", 6.0)),
                                nbins=int(params.get("nbins", 200)))
        peakmin = sd.rdf_first_peak_and_minimum(
            rdf["r_A"], rdf["g_of_r"],
            smoothing_window=int(params.get("smoothing_window", 5)),
            min_r_A=params.get("min_r_A"), max_r_A=params.get("max_r_A"))
        if kind == "rdf_peak_height":
            value = peakmin["g_first_peak"]
        else:
            value = peakmin["r_first_min_A"] if params.get("position") == "first_min" \
                else peakmin["r_first_peak_A"]
        return _result(spec, COMPUTED, value=float(value),
                       details={"peakmin": peakmin, "bin_width_A": rdf["bin_width_A"]})

    if kind in ("species_coordination", "coordination_distribution"):
        center = _require(params, "center_species", name)
        neighbor = _require(params, "neighbor_species", name)
        cutoff_A = _require(params, "cutoff_A", name)
        if not _species_present(frames, center):
            return _result(spec, UNAVAILABLE,
                           reason=f"center species {center} not present in trajectory")
        cc = sd.compute_species_coordination(
            frames, center, neighbor, float(cutoff_A),
            cutoff_source_ref=params.get("cutoff_source_ref"),
            cutoff_frozen_before_student=params.get("cutoff_frozen_before_student"),
            max_topology=int(params.get("max_topology", 8)))
        if kind == "species_coordination":
            return _result(spec, COMPUTED,
                           value=float(cc["aggregate_mean_coordination"]), details=cc)
        target_cn = params.get("target_coordination")
        if target_cn is None:
            return _result(spec, COMPUTED, value=None, details=cc)
        frac = cc["coordination_fractions"].get(int(target_cn), 0.0)
        return _result(spec, COMPUTED, value=float(frac), details=cc)

    if kind == "density":
        mean_rho, std_rho = sd.compute_density(frames)
        return _result(spec, COMPUTED, value=float(mean_rho),
                       details={"standard_deviation": float(std_rho)})

    if kind == "msd":
        msd = sd.compute_msd(frames)
        species = params.get("species")
        if species is not None:
            if str(species) not in {str(k) for k in msd}:
                return _result(spec, UNAVAILABLE,
                               reason=f"species {species} not present for MSD")
            series = next(v for k, v in msd.items() if str(k) == str(species))
            return _result(spec, COMPUTED, value=float(series[-1]),
                           details={"final_msd_by_species":
                                    {str(k): float(v[-1]) for k, v in msd.items()}})
        # aggregate final MSD weighted by atom count of the first frame
        syms = frames[0].get_chemical_symbols()
        counts = {str(k): syms.count(str(k)) for k in msd}
        total = sum(counts.values()) or 1
        agg = sum(float(v[-1]) * counts[str(k)] for k, v in msd.items()) / total
        return _result(spec, COMPUTED, value=float(agg),
                       details={"final_msd_by_species":
                                {str(k): float(v[-1]) for k, v in msd.items()}})

    if kind == "diffusivity":
        species = _require(params, "species", name)
        if not _species_present(frames, species):
            return _result(spec, UNAVAILABLE,
                           reason=f"species {species} not present for diffusivity")
        timestep_fs = params.get("timestep_fs", context.get("timestep_fs"))
        sample_interval = params.get("sample_interval_steps",
                                     context.get("sample_interval_steps", 1))
        if timestep_fs is None:
            return _result(spec, UNAVAILABLE,
                           reason="diffusivity requires timestep_fs in params or context")
        msd = sd.compute_msd(frames)
        series_map = {str(k): v for k, v in msd.items()}
        if str(species) not in series_map:
            return _result(spec, UNAVAILABLE, reason=f"no MSD series for {species}")
        d = sd.compute_diffusivity(
            {str(species): series_map[str(species)]}, float(timestep_fs),
            fit_start_frame=int(_require(params, "fit_start_frame", name)),
            fit_end_frame=int(_require(params, "fit_end_frame", name)),
            sample_interval_steps=int(sample_interval),
            n_dims=int(params.get("n_dims", 3)))
        rec = d[str(species)]
        return _result(spec, COMPUTED, value=float(rec["diffusivity_A2_per_ps"]), details=rec)

    if kind == "adf":
        center = _require(params, "center_species", name)
        neighbor = _require(params, "neighbor_species", name)
        if not _species_present(frames, center):
            return _result(spec, UNAVAILABLE,
                           reason=f"center species {center} not present for ADF")
        adf = sd.compute_adf(
            frames, center, neighbor,
            r_cut_A=float(_require(params, "r_cut_A", name)),
            nbins=int(params.get("nbins", 180)),
            angle_min_deg=float(params.get("angle_min_deg", 0.0)),
            angle_max_deg=float(params.get("angle_max_deg", 180.0)))
        if adf["n_triplets"] == 0:
            return _result(spec, UNAVAILABLE,
                           reason="no qualifying triplets within r_cut_A", details=adf)
        summary = params.get("summary", "mean_angle")
        if summary == "peak_angle":
            k = int(np.argmax(adf["counts"]))
            value = adf["bin_centers_deg"][k]
        else:
            value = adf["mean_angle_deg"]
        return _result(spec, COMPUTED, value=float(value), details=adf)

    if kind == "nve_drift":
        energies = params.get("energies", context.get("energies"))
        if energies is None or any(e is None for e in energies):
            return _result(spec, UNAVAILABLE,
                           reason="nve_drift requires a complete per-frame energy series")
        timestep_fs = params.get("timestep_fs", context.get("timestep_fs"))
        if timestep_fs is None:
            return _result(spec, UNAVAILABLE, reason="nve_drift requires timestep_fs")
        n_atoms = int(params.get("n_atoms", context.get("n_atoms", len(frames[0]))))
        drift, resid = sd.compute_nve_drift(
            [float(x) for x in energies], float(timestep_fs), n_atoms,
            sample_interval_steps=int(params.get("sample_interval_steps",
                                                 context.get("sample_interval_steps", 1))),
            steps=params.get("steps"))
        return _result(spec, COMPUTED, value=float(drift),
                       details={"residual_std_meV_per_atom": float(resid),
                                "n_atoms": n_atoms})

    plugin = sd.observable_plugin(kind)
    if plugin is not None:
        out = plugin(frames, params, context)
        if not isinstance(out, dict):
            raise ValueError(f"observable plugin {kind!r} must return a dict")
        status = out.get("status", COMPUTED)
        return _result(spec, status, value=out.get("value"),
                       details=out.get("details"), reason=out.get("reason"))

    raise ValueError(f"unknown observable kind: {kind!r} (register a plugin to add it)")


def evaluate_observables(specs, frames, *, context=None):
    """Evaluate a list of observable specs, preserving order. Duplicate names fail closed."""
    seen = set()
    results = []
    for spec in specs:
        name = spec.get("name")
        if name in seen:
            raise ValueError(f"duplicate observable name: {name}")
        seen.add(name)
        results.append(evaluate_observable(spec, frames, context=context))
    return results


# --------------------------------------------------------------------------------------------
# Frozen, hash-bound TeacherValidationTarget.
# --------------------------------------------------------------------------------------------

def _canonical_bytes(payload):
    reduced = {k: v for k, v in payload.items() if k not in _HASH_EXCLUDED_FIELDS}
    return json.dumps(reduced, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _target_sha256(payload):
    import hashlib
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_teacher_identity(teacher_identity):
    if not isinstance(teacher_identity, dict) or not teacher_identity.get("model_sha256"):
        raise ValueError("teacher_identity must bind at least model_sha256")


def _validate_md_protocol(md_protocol):
    if not isinstance(md_protocol, dict):
        raise ValueError("md_protocol must be a mapping")
    required = ("ensemble", "timestep_fs", "n_steps", "sample_stride", "seed")
    missing = [k for k in required if k not in md_protocol]
    if missing:
        raise ValueError("md_protocol is missing required fields: " + ", ".join(missing))


def _freeze_target(*, objective_profile_sha256, teacher_identity, md_protocol,
                   trajectory, observable_definitions, observable_results, evidence,
                   mode, extra=None):
    from datetime import datetime, timezone
    payload = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "artifact": "teacher_validation_target",
        "mode": mode,
        "objective_profile_sha256": objective_profile_sha256,
        "teacher_identity": teacher_identity,
        "md_protocol": md_protocol,
        "trajectory": trajectory,
        "observable_definitions": observable_definitions,
        "observable_results": observable_results,
        "observable_status": {r["name"]: r["status"] for r in observable_results},
        "units": {r["name"]: r["units"] for r in observable_results},
        "evidence": evidence,
    }
    if extra:
        payload.update(extra)
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
    payload["target_sha256"] = _target_sha256(payload)
    return payload


def verify_teacher_validation_target(target):
    """Recompute the canonical hash of a frozen target and reject any post-freeze mutation.

    ``target`` may be a dict or a path to the frozen JSON. Returns the loaded dict on success;
    raises ValueError if the recomputed hash does not match the bound ``target_sha256`` (i.e.
    the frozen target was edited after it was established)."""
    if isinstance(target, (str, Path)):
        target = json.loads(Path(target).read_text(encoding="utf-8"))
    if not isinstance(target, dict) or "target_sha256" not in target:
        raise ValueError("not a frozen TeacherValidationTarget (no target_sha256)")
    recomputed = _target_sha256(target)
    if recomputed != target["target_sha256"]:
        raise ValueError(
            "TeacherValidationTarget integrity mismatch: it was mutated after being frozen "
            f"({recomputed} != {target['target_sha256']})")
    return target


def _write_target(target, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(target, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return target_path


# --------------------------------------------------------------------------------------------
# COMPUTE mode — run the Teacher PES under a declared protocol, freeze the target.
# --------------------------------------------------------------------------------------------

def _default_md_engine():
    from framework_v2.acquisition.generators.teacher_dynamics import _ase_md_engine
    return _ase_md_engine


def _drive_teacher_md(start_atoms, calc, md_protocol, *, md_engine):
    """Reuse the shared ASE integrator reference to produce a sampled trajectory + per-frame
    energies. ``md_engine`` has the teacher_dynamics signature
    ``(atoms, calc, params, seed, sample_fn)``; ``sample_fn(atoms, step)`` records a frame."""
    collected = []
    energies = []

    def _sample(atoms, step):
        geom = atoms.copy()
        e = None
        try:
            e = float(atoms.get_potential_energy())
        except Exception:
            e = None
        geom.calc = None
        geom.info["md_step"] = int(step)
        if e is not None:
            geom.info["potential_energy"] = e
        collected.append(geom)
        energies.append(e)

    params = dict(md_protocol)
    params.setdefault("temperature_K", md_protocol.get("temperature_K", 0.0))
    md_engine(start_atoms.copy(), calc, params, int(md_protocol["seed"]), _sample)
    if not collected:
        raise RuntimeError("Teacher MD produced no sampled frames")
    return collected, energies


def compute_teacher_validation_target(*, objective_profile_sha256, teacher_identity,
                                      md_protocol, start_structures_path, observable_specs,
                                      target_path, trajectory_out_path,
                                      teacher_calculator_provider, md_engine=None,
                                      context=None):
    """COMPUTE mode: drive the Teacher PES, sample a trajectory, compute the profile's
    observables through the shared engine, and freeze a hash-bound TeacherValidationTarget.

    ``teacher_calculator_provider`` exposes ``make_ase_calculator()`` (the exact contract the
    Teacher-driven-acquisition backend already uses); ``md_engine`` defaults to the shared ASE
    integrator reference but is injectable so the whole flow is provable with a fake Teacher +
    fake integrator (no real Teacher inference)."""
    from ase.io import read, write
    _validate_teacher_identity(teacher_identity)
    _validate_md_protocol(md_protocol)
    if teacher_calculator_provider is None:
        raise ValueError("COMPUTE mode requires a teacher_calculator_provider")
    engine = md_engine or _default_md_engine()
    starts = read(str(start_structures_path), index=":")
    if not starts:
        raise ValueError("start_structures_path contains no frames")
    calc = teacher_calculator_provider.make_ase_calculator()
    frames, energies = _drive_teacher_md(starts[0], calc, md_protocol, md_engine=engine)

    trajectory_out_path = Path(trajectory_out_path)
    trajectory_out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(trajectory_out_path), frames, format="extxyz")
    traj_sha = sha256_file(trajectory_out_path)

    eng_context = dict(context or {})
    eng_context.setdefault("energies", energies)
    eng_context.setdefault("timestep_fs", float(md_protocol["timestep_fs"]))
    eng_context.setdefault("sample_interval_steps", int(md_protocol["sample_stride"]))
    eng_context.setdefault("n_atoms", len(frames[0]))
    results = evaluate_observables(observable_specs, frames, context=eng_context)

    trajectory = {"path": str(trajectory_out_path.resolve()), "sha256": traj_sha,
                  "n_frames": len(frames), "source": "teacher_md_compute"}
    evidence = [{"role": "teacher_trajectory", "path": str(trajectory_out_path.resolve()),
                 "sha256": traj_sha}]
    target = _freeze_target(
        objective_profile_sha256=objective_profile_sha256, teacher_identity=teacher_identity,
        md_protocol=md_protocol, trajectory=trajectory,
        observable_definitions=list(observable_specs), observable_results=results,
        evidence=evidence, mode="COMPUTE")
    _write_target(target, target_path)
    return target


# --------------------------------------------------------------------------------------------
# INGEST mode — accept an existing Teacher trajectory / evidence, freeze the same target.
# --------------------------------------------------------------------------------------------

def _validate_precomputed_observables(precomputed):
    if not isinstance(precomputed, list) or not precomputed:
        raise ValueError("precomputed_observables must be a non-empty list of result rows")
    for row in precomputed:
        if not isinstance(row, dict) or not row.get("name") or not row.get("kind"):
            raise ValueError("each precomputed observable requires name and kind")
        if row.get("status") not in (COMPUTED, UNAVAILABLE, NOT_APPLICABLE):
            raise ValueError(f"precomputed observable {row.get('name')!r} has invalid status")
    return precomputed


def ingest_teacher_validation_target(*, objective_profile_sha256, teacher_identity,
                                     md_protocol, trajectory_path, trajectory_sha256,
                                     observable_specs, target_path, context=None,
                                     precomputed_observables=None,
                                     recompute_from_trajectory=True):
    """INGEST mode: validate a pre-existing Teacher trajectory's provenance/hash and freeze the
    SAME target. When the raw trajectory is available and ``recompute_from_trajectory`` is set,
    the observables are RE-DERIVED through the shared engine (byte-identical to COMPUTE mode and
    to Student Stage 11) — no Teacher calculator is ever constructed, so no Teacher inference is
    run. Otherwise externally computed ``precomputed_observables`` evidence is accepted as-is
    (its shape validated) and marked as externally ingested, so expensive Teacher MD is never
    re-run just to satisfy the freeze."""
    from ase.io import read
    _validate_teacher_identity(teacher_identity)
    _validate_md_protocol(md_protocol)
    trajectory_path = Path(trajectory_path)
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    actual_sha = sha256_file(trajectory_path)
    if actual_sha != trajectory_sha256:
        raise ValueError(
            f"ingested Teacher trajectory hash mismatch: {actual_sha} != {trajectory_sha256}")

    provenance = None
    if recompute_from_trajectory:
        frames = read(str(trajectory_path), index=":")
        if not frames:
            raise ValueError("ingested trajectory contains no frames")
        eng_context = dict(context or {})
        if "energies" not in eng_context:
            infos = [a.info.get("potential_energy") for a in frames]
            eng_context["energies"] = infos if all(e is not None for e in infos) else None
        eng_context.setdefault("timestep_fs", float(md_protocol["timestep_fs"]))
        eng_context.setdefault("sample_interval_steps", int(md_protocol["sample_stride"]))
        eng_context.setdefault("n_atoms", len(frames[0]))
        results = evaluate_observables(observable_specs, frames, context=eng_context)
        n_frames = len(frames)
        provenance = "INGESTED_RECOMPUTED"
    else:
        results = _validate_precomputed_observables(precomputed_observables)
        n_frames = None
        provenance = "INGESTED_EXTERNAL"

    trajectory = {"path": str(trajectory_path.resolve()), "sha256": trajectory_sha256,
                  "n_frames": n_frames, "source": "teacher_trajectory_ingest",
                  "provenance": provenance}
    evidence = [{"role": "teacher_trajectory", "path": str(trajectory_path.resolve()),
                 "sha256": trajectory_sha256}]
    target = _freeze_target(
        objective_profile_sha256=objective_profile_sha256, teacher_identity=teacher_identity,
        md_protocol=md_protocol, trajectory=trajectory,
        observable_definitions=list(observable_specs), observable_results=results,
        evidence=evidence, mode="INGEST", extra={"ingest_provenance": provenance})
    _write_target(target, target_path)
    return target


# --------------------------------------------------------------------------------------------
# Stage 11 — Student reproduction against the frozen Teacher target.
# --------------------------------------------------------------------------------------------

_ALLOWED_POLICY_KEYS = {"observable_criteria"}


def compare_student_to_teacher_target(target, student_frames, *, comparison_policy=None,
                                      student_context=None):
    """Compare a Student trajectory against the frozen Teacher target (Stage 11).

    The Student supplies ONLY its own trajectory: every observable DEFINITION (kind, params,
    units) comes from the frozen target and is re-evaluated on the Student frames through the
    SAME ``evaluate_observable`` the Teacher used. The Student can never redefine an observable
    or its threshold — ``comparison_policy`` may carry acceptance criteria under
    ``observable_criteria`` ONLY for observables the frozen target did not already bind one for,
    and may never carry observable definitions/params. The frozen target is never mutated.
    """
    target = verify_teacher_validation_target(target)
    target = copy.deepcopy(target)  # never mutate the caller's / frozen artifact
    before_sha = target["target_sha256"]

    policy = comparison_policy or {}
    illegal = set(policy) - _ALLOWED_POLICY_KEYS
    if illegal:
        raise ValueError(
            "comparison_policy may not redefine Teacher observables; illegal keys: "
            + ", ".join(sorted(illegal)))
    policy_criteria = policy.get("observable_criteria") or {}

    definitions = {d["name"]: d for d in target["observable_definitions"]}
    teacher_results = {r["name"]: r for r in target["observable_results"]}

    comparisons = []
    for name, spec in definitions.items():
        teacher_row = teacher_results.get(name)
        if teacher_row is None or teacher_row["status"] != COMPUTED:
            comparisons.append({
                "name": name, "kind": spec["kind"], "status": NOT_APPLICABLE,
                "reason": f"teacher observable status is "
                          f"{teacher_row['status'] if teacher_row else 'MISSING'}",
                "teacher_value": teacher_row["value"] if teacher_row else None,
                "student_value": None, "units": spec.get("units"),
            })
            continue
        student_row = evaluate_observable(spec, student_frames, context=student_context)
        if student_row["status"] != COMPUTED:
            comparisons.append({
                "name": name, "kind": spec["kind"], "status": UNAVAILABLE,
                "reason": f"student observable {student_row['status']}: "
                          f"{student_row.get('reason')}",
                "teacher_value": teacher_row["value"], "student_value": None,
                "units": spec.get("units"),
            })
            continue

        t_val = float(teacher_row["value"])
        s_val = float(student_row["value"])
        abs_dev = abs(s_val - t_val)
        rel_dev = abs_dev / abs(t_val) if t_val != 0 else None
        criterion = spec.get("comparison_criterion") or policy_criteria.get(name)
        status = "RECORDED"
        if criterion is not None:
            operator = criterion.get("operator")
            if operator == "max_abs_deviation":
                status = "PASS" if abs_dev <= float(criterion["threshold"]) else "FAIL"
            elif operator == "max_relative_deviation":
                status = ("PASS" if (rel_dev is not None
                                     and rel_dev <= float(criterion["threshold"]))
                          else "FAIL")
            elif operator in ("max_abs", "max", "min", "target_tolerance", "equals"):
                # criterion applied directly to the Student value (Teacher-frozen absolute bound)
                status = "PASS" if criterion_passes(s_val, criterion) else "FAIL"
            else:
                raise ValueError(f"unknown comparison operator: {operator!r}")
        comparisons.append({
            "name": name, "kind": spec["kind"], "status": status,
            "teacher_value": t_val, "student_value": s_val,
            "abs_deviation": abs_dev, "relative_deviation": rel_dev,
            "criterion": criterion, "units": spec.get("units"),
        })

    statuses = {c["status"] for c in comparisons}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "PASS" in statuses:
        overall = "PASS"
    else:
        overall = "RECORDED"

    # Integrity guard: the frozen target must be untouched by comparison.
    if _target_sha256(target) != before_sha:
        raise RuntimeError("internal error: comparison mutated the frozen target")

    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "target_sha256": before_sha,
        "objective_profile_sha256": target.get("objective_profile_sha256"),
        "comparisons": comparisons,
        "overall_status": overall,
    }
