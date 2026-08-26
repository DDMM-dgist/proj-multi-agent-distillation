"""Deterministic trusted executors — thin wrappers over EXISTING repository primitives / ASE.

No scientific method is invented here. Each function reads a proposal's parameters, calls an
existing repository function or a standard ASE primitive, and returns a bounded, typed result
(optionally hashed to an artifact). Units are explicit where relevant. All are safe/zero-cost and
run in sandbox-primary integration tests with synthetic input, through the same
dispatch -> controller -> executor path as production.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def _p(proposal) -> dict:
    if hasattr(proposal, "parameters"):
        return getattr(proposal, "parameters") or {}
    if isinstance(proposal, dict):
        return proposal.get("parameters", {}) or {}
    return {}


def _write(obj, out_path):
    result = {"metrics": obj}
    if out_path:
        from workflow.integrity import sha256_file
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, default=float, indent=2))
        result["path"] = str(p)
        result["sha256"] = sha256_file(p)
    return result


def _read_frames(path):
    from ase.io import read
    return read(path, index=":")


# --- Data Curator ---------------------------------------------------------------

def inspect_dataset(proposal) -> dict:
    p = _p(proposal)
    frames = _read_frames(p["frames_path"])
    counts = [len(a) for a in frames]
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    first = frames[0] if frames else None
    return _write({
        "n_frames": len(frames),
        "elements": elements,
        "atom_count_min": min(counts) if counts else 0,
        "atom_count_max": max(counts) if counts else 0,
        "has_energy": bool(first and "teacher_energy" in getattr(first, "info", {})),
        "has_forces": bool(first and "teacher_forces" in getattr(first, "arrays", {})),
        "has_lineage": bool(first and "parent_structure_id" in getattr(first, "info", {})),
    }, p.get("out_path"))


def summarize_source_categories(proposal) -> dict:
    """Category counts from frame metadata (key configurable; default 'source')."""
    p = _p(proposal)
    key = p.get("category_key", "source")
    frames = _read_frames(p["frames_path"])
    counts: dict = {}
    for a in frames:
        cat = str(a.info.get(key, "unknown"))
        counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values()) or 1
    fractions = {k: v / total for k, v in counts.items()}
    return _write({"category_key": key, "counts": counts, "fractions": fractions,
                   "n_frames": len(frames)}, p.get("out_path"))


def _group_by_lineage(frames, group_key="parent_structure_id"):
    from adapters.acquisition import validate_lineage  # noqa: F401 (import proves the primitive)
    groups: dict = {}
    for i, a in enumerate(frames):
        gid = str(a.info.get(group_key, a.info.get("structure_id", f"frame-{i}")))
        groups.setdefault(gid, []).append(a.info.get("structure_id", f"frame-{i}"))
    return groups


def reconstruct_lineage(proposal) -> dict:
    p = _p(proposal)
    key = p.get("group_key", "parent_structure_id")
    frames = _read_frames(p["frames_path"])
    groups = _group_by_lineage(frames, key)
    return _write({"group_key": key, "n_groups": len(groups),
                   "group_sizes": {g: len(v) for g, v in groups.items()}}, p.get("out_path"))


def detect_duplicates(proposal) -> dict:
    """Exact-geometry duplicate detection via the existing steps._structure_fingerprint."""
    from workflow.steps import _structure_fingerprint
    p = _p(proposal)
    frames = _read_frames(p["frames_path"])
    seen: dict = {}
    dup_indices = []
    for i, a in enumerate(frames):
        fp = _structure_fingerprint(a)
        if fp in seen:
            dup_indices.append(i)
        else:
            seen[fp] = i
    return _write({"n_frames": len(frames), "n_unique": len(seen),
                   "n_duplicates": len(dup_indices), "duplicate_indices": dup_indices},
                  p.get("out_path"))


def _min_distances(frames):
    import numpy as np
    mins = []
    for a in frames:
        d = a.get_all_distances(mic=True)
        n = len(a)
        if n < 2:
            mins.append(float("inf"))
            continue
        iu = np.triu_indices(n, k=1)
        mins.append(float(np.min(d[iu])))
    return mins


def compute_minimum_distance(proposal) -> dict:
    """Minimum interatomic distance per frame (Angstrom) via ASE mic distances."""
    p = _p(proposal)
    frames = _read_frames(p["frames_path"])
    mins = _min_distances(frames)
    finite = [m for m in mins if m != float("inf")]
    return _write({"unit": "Angstrom", "min_distance_per_frame": mins,
                   "global_min": (min(finite) if finite else None)}, p.get("out_path"))


def detect_atomic_overlap(proposal) -> dict:
    """Flag frames whose minimum interatomic distance is below a threshold (default 0.5 A)."""
    p = _p(proposal)
    threshold = float(p.get("min_distance_threshold", 0.5))
    frames = _read_frames(p["frames_path"])
    mins = _min_distances(frames)
    overlapping = [i for i, m in enumerate(mins) if m < threshold]
    return _write({"unit": "Angstrom", "threshold": threshold,
                   "overlapping_frame_indices": overlapping,
                   "n_overlapping": len(overlapping)}, p.get("out_path"))


def validate_label_preservation(proposal) -> dict:
    """Confirm labeled frames retain teacher labels + lineage, and count matches the source."""
    p = _p(proposal)
    labeled = _read_frames(p["labeled_path"])
    n_source = int(p["n_source_frames"]) if "n_source_frames" in p else len(labeled)
    missing_energy = sum(1 for a in labeled if "teacher_energy" not in a.info)
    missing_forces = sum(1 for a in labeled if "teacher_forces" not in a.arrays)
    missing_lineage = sum(1 for a in labeled if "parent_structure_id" not in a.info)
    ok = (missing_energy == 0 and missing_forces == 0 and missing_lineage == 0
          and len(labeled) == n_source)
    result = {"ok": ok, "n_labeled": len(labeled), "n_source": n_source,
              "missing_energy": missing_energy, "missing_forces": missing_forces,
              "missing_lineage": missing_lineage}
    if not ok:
        raise _ValidationFailure(json.dumps(result))
    return _write(result, p.get("out_path"))


def validate_species_mapping_consistency(proposal) -> dict:
    """Expose the concrete element/species -> 0-based model-type-index mapping recorded in a
    Teacher labeling manifest's ``species_mapping_evidence`` and DETERMINISTICALLY confirm it is
    internally consistent across every independently-sourced mapping the labeling run already
    recorded -- the declared Teacher config (``declared_chemical_symbols``), the constructed
    calculator's own runtime state, and the compiled model's embedded metadata -- reusing the same
    ``adapters.teacher`` cross-check primitives ``label_with_teacher`` itself used, so no new
    species-mapping science is invented here. Fails closed (``_ValidationFailure`` /
    ``SpeciesMappingConflictError``) when the mapping is not attested or any two sources disagree.

    Optional ``teacher_config``: an INDEPENDENT fourth source read fresh off disk. When supplied,
    its sha256 must match the manifest's recorded ``teacher_config_sha256`` (else fail closed), and
    its declared ``calculator.kwargs.chemical_symbols`` are re-derived into a mapping and
    cross-checked against the manifest's runtime/compiled maps -- validating the manifest against the
    hashed active Teacher configuration rather than trusting the manifest's own self-report.

    Parameters: ``manifest_path`` (required); optional ``teacher_config``,
    ``expected_manifest_sha256``, ``out_path``.
    """
    from adapters.teacher import (
        _cross_check_species_mappings, _ordered_symbols_to_type_map,
        species_mapping_is_attested)
    from workflow.integrity import sha256_file
    p = _p(proposal)
    manifest_path = Path(p["manifest_path"])
    manifest_sha256 = sha256_file(manifest_path)
    expected_manifest_sha256 = p.get("expected_manifest_sha256")
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise _ValidationFailure(json.dumps({
            "ok": False, "reason": "manifest_sha256_mismatch",
            "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256,
            "expected_manifest_sha256": expected_manifest_sha256}))
    manifest = json.loads(manifest_path.read_text())
    evidence = manifest.get("species_mapping_evidence")
    if not isinstance(evidence, dict):
        raise _ValidationFailure(json.dumps({
            "ok": False, "reason": "manifest_has_no_species_mapping_evidence",
            "manifest_path": str(manifest_path)}))

    declared_symbols = evidence.get("declared_chemical_symbols")
    declared_map = _ordered_symbols_to_type_map(declared_symbols) if declared_symbols else None
    runtime_map = evidence.get("runtime_chemical_species_to_atom_type_map")
    compiled_map = evidence.get("compiled_model_type_names_map")

    config_binding = None
    if p.get("teacher_config") is not None:
        import yaml
        config_path = Path(p["teacher_config"])
        config_sha256 = sha256_file(config_path)
        manifest_config_sha256 = manifest.get("teacher_config_sha256")
        if manifest_config_sha256 is not None and config_sha256 != manifest_config_sha256:
            raise _ValidationFailure(json.dumps({
                "ok": False, "reason": "teacher_config_sha256_mismatch",
                "teacher_config": str(config_path), "teacher_config_sha256": config_sha256,
                "manifest_teacher_config_sha256": manifest_config_sha256}))
        cfg = yaml.safe_load(config_path.read_text()) or {}
        cfg_symbols = (((cfg.get("calculator") or {}).get("kwargs") or {})
                       .get("chemical_symbols"))
        config_map = _ordered_symbols_to_type_map(cfg_symbols) if cfg_symbols else None
        config_binding = {"teacher_config": str(config_path.resolve()),
                          "teacher_config_sha256": config_sha256,
                          "sha256_matches_manifest": manifest_config_sha256 is None
                          or config_sha256 == manifest_config_sha256,
                          "declared_chemical_symbols": cfg_symbols,
                          "config_species_to_type_index_map": config_map}
    else:
        config_map = None

    # Fail closed on any disagreement among the independently-sourced mappings (declared config,
    # constructed-calculator runtime, compiled-model metadata, and the fresh on-disk config when
    # supplied). Raises adapters.teacher.SpeciesMappingConflictError on conflict.
    sources = {"declared_config": declared_map,
               "constructed_calculator_runtime": runtime_map,
               "compiled_model_metadata": compiled_map,
               "reread_teacher_config": config_map}
    _cross_check_species_mappings(sources)

    attested = species_mapping_is_attested(evidence)
    exposed_map = runtime_map or declared_map or compiled_map or config_map
    result = {
        "ok": bool(attested),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "species_to_type_index_map": exposed_map,
        "declared_config_map": declared_map,
        "runtime_map": runtime_map,
        "compiled_model_map": compiled_map,
        "attested": bool(attested),
        "fallback_applied": bool(evidence.get("fallback_applied")),
        "sources_cross_checked": sorted(name for name, m in sources.items() if m),
        "teacher_config_binding": config_binding,
    }
    if not attested:
        result["reason"] = "species_mapping_not_attested"
        raise _ValidationFailure(json.dumps(result))
    return _write(result, p.get("out_path"))


def sample_seed_pool(proposal) -> dict:
    """Deterministic seed-pool selection policy v1 (no new scientific sampling method):
    seeded shuffle of an id-sorted frame list, selecting ``count`` frames, deduplicated, with
    category counts and spanned lineage groups reported so a downstream split can keep groups
    intact. Same source + same seed => identical selection (regression-tested)."""
    from workflow.integrity import sha256_file
    p = _p(proposal)
    frames = _read_frames(p["frames_path"])
    count = int(p["count"])
    seed = int(p.get("seed", 2026))
    cat_key = p.get("category_key", "source")
    group_key = p.get("group_key", "parent_structure_id")
    items = [{"id": str(a.info.get("structure_id", f"frame-{i}")),
              "cat": str(a.info.get(cat_key, "unknown")),
              "grp": str(a.info.get(group_key, a.info.get("structure_id", f"frame-{i}")))}
             for i, a in enumerate(frames)]
    ordered = sorted(items, key=lambda x: x["id"])          # stable base order
    random.Random(seed).shuffle(ordered)                     # deterministic given the seed
    selected, seen_ids = [], set()
    for it in ordered:
        if len(selected) >= count:
            break
        if it["id"] not in seen_ids:                         # duplicate prevention
            selected.append(it)
            seen_ids.add(it["id"])
    selected_ids = sorted(seen_ids)
    manifest = {
        "schema_version": 1, "policy": "seed_pool_v1", "seed": seed,
        "requested_count": count, "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "category_counts": {c: sum(1 for it in selected if it["cat"] == c)
                            for c in sorted({it["cat"] for it in selected})},
        "lineage_groups": sorted({it["grp"] for it in selected}),
        "source_sha256": sha256_file(Path(p["frames_path"])),
    }
    out = p.get("manifest_path")
    result = {"metrics": manifest}
    if out:
        Path(out).write_text(json.dumps(manifest, indent=2))
        result["path"] = out
        result["sha256"] = sha256_file(Path(out))
    return result


def build_dataset_manifest(proposal) -> dict:
    """Assemble a hash-bound dataset manifest via the existing workflow.integrity.artifact_digest."""
    from workflow.integrity import artifact_digest
    p = _p(proposal)
    dataset = Path(p["dataset"])
    frames = _read_frames(str(dataset))
    manifest = {
        "schema_version": 1, "dataset": str(dataset.resolve()),
        "integrity": artifact_digest(dataset), "n_frames": len(frames),
        "elements": sorted({s for a in frames for s in a.get_chemical_symbols()}),
    }
    out = p.get("manifest_path")
    if out:
        Path(out).write_text(json.dumps(manifest, indent=2))
    return {"path": out, "manifest": manifest, "sha256": manifest["integrity"]["sha256"]}


# --- ML Trainer -----------------------------------------------------------------

def prepare_student_inputs(proposal) -> dict:
    """Render the student input config via the existing adapters.student.render_student_inputs
    (config/adapter-selected by ``cfg["kind"]`` -- this executor never picks a model family)."""
    from adapters import load_config
    from adapters.student import render_student_inputs
    p = _p(proposal)
    cfg = load_config(p["student_config"])
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_student_inputs(cfg, out_dir)
    return {"path": str(rendered), "rendered_config": str(rendered)}


def collect_checkpoints(proposal) -> dict:
    """Collect committee checkpoint paths + integrity from an existing committee manifest."""
    p = _p(proposal)
    manifest = json.loads(Path(p["committee_manifest"]).read_text())
    models = manifest.get("models", [])
    return _write({"n_checkpoints": len(models),
                   "checkpoints": [{"seed": m.get("seed"), "path": m.get("path"),
                                    "sha256": (m.get("integrity", {}) or {}).get("sha256", "")}
                                   for m in models]}, p.get("out_path"))


def summarize_seed_variation(proposal) -> dict:
    """Per-atom committee force-std summary via the existing adapters.uncertainty.committee_force_std."""
    from adapters.uncertainty import committee_force_std
    import numpy as np
    p = _p(proposal)
    per_atom, frame = committee_force_std(p["forces_per_seed"], aggregate=p.get("aggregate", "max"))
    arr = np.asarray(per_atom, dtype=float)
    return _write({"u_frame": float(frame), "u_mean": float(arr.mean()),
                   "u_max": float(arr.max()), "n_atoms": int(arr.size)}, p.get("out_path"))


def validate_training_completion(proposal) -> dict:
    """Completeness validator: committee manifest has the expected seeds, each checkpoint exists
    and its integrity matches the recorded digest (uses workflow.integrity.artifact_digest)."""
    from workflow.integrity import artifact_digest
    p = _p(proposal)
    manifest = json.loads(Path(p["committee_manifest"]).read_text())
    expected = int(p.get("expected_seeds", 4))
    models = manifest.get("models", [])
    problems = []
    if len(models) != expected:
        problems.append(f"expected {expected} seeds, found {len(models)}")
    for m in models:
        path = Path(m.get("path", ""))
        if not path.exists():
            problems.append(f"missing checkpoint: {path}")
            continue
        recorded = (m.get("integrity", {}) or {}).get("sha256")
        if recorded and artifact_digest(path).get("sha256") != recorded:
            problems.append(f"integrity mismatch: {path}")
    result = {"ok": not problems, "n_models": len(models), "expected_seeds": expected,
              "problems": problems}
    if problems:
        raise _ValidationFailure(json.dumps(result))
    return _write(result, p.get("out_path"))


# --- Simulation -----------------------------------------------------------------

def detect_force_spike(proposal) -> dict:
    """Flag frames whose max |force| exceeds a threshold (eV/Angstrom)."""
    import numpy as np
    p = _p(proposal)
    threshold = float(p.get("force_threshold", 50.0))
    frames = _read_frames(p["frames_path"])
    fmax = []
    for a in frames:
        f = a.get_forces() if a.calc is not None else a.arrays.get("forces")
        fmax.append(float(np.max(np.linalg.norm(np.asarray(f), axis=1))) if f is not None else 0.0)
    spikes = [i for i, v in enumerate(fmax) if v > threshold]
    return _write({"unit": "eV/Angstrom", "threshold": threshold, "fmax_per_frame": fmax,
                   "spike_frame_indices": spikes, "n_spikes": len(spikes)}, p.get("out_path"))


def validate_simulation_completion(proposal) -> dict:
    """Completeness validator: MD manifest present, trajectory exists, energies finite."""
    import numpy as np
    p = _p(proposal)
    problems = []
    manifest_path = Path(p["md_manifest"])
    if not manifest_path.exists():
        problems.append(f"missing MD manifest: {manifest_path}")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
    traj = p.get("trajectory_path")
    if traj and not Path(traj).exists():
        problems.append(f"missing trajectory: {traj}")
    energies = p.get("energies")
    if energies is not None and not np.all(np.isfinite(np.asarray(energies, dtype=float))):
        problems.append("non-finite energies present")
    result = {"ok": not problems, "problems": problems,
              "has_manifest": manifest_path.exists()}
    if problems:
        raise _ValidationFailure(json.dumps(result))
    return _write(result, p.get("out_path"))


# --- Analyst-support deterministic analysis -------------------------------------

def summarize_md_stability(proposal) -> dict:
    """Deterministic MD-stability summary composed from existing checks: NVE drift, minimum
    distance, and force spikes (no new physics)."""
    import numpy as np
    from validation.structure_dynamics import compute_nve_drift
    p = _p(proposal)
    summary: dict = {}
    if "energies" in p:
        drift, _ = compute_nve_drift([float(x) for x in p["energies"]],
                                     float(p.get("timestep_fs", 1.0)), int(p.get("n_atoms", 1)))
        summary["nve_drift"] = float(drift)
    if "frames_path" in p:
        frames = _read_frames(p["frames_path"])
        mins = _min_distances(frames)
        finite = [m for m in mins if m != float("inf")]
        summary["global_min_distance"] = (min(finite) if finite else None)
    summary["stable"] = (
        abs(summary.get("nve_drift", 0.0)) < float(p.get("nve_drift_tol", 1.0))
        and (summary.get("global_min_distance") is None
             or summary["global_min_distance"] > float(p.get("min_distance_tol", 0.5))))
    return _write(summary, p.get("out_path"))


class _ValidationFailure(Exception):
    """Raised by completion/preservation validators so the dispatcher marks the action INVALID
    (fail-closed) rather than returning a passing artifact."""
