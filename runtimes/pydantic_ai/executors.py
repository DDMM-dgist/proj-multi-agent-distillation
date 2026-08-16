"""Trusted executors: bind in-scope actions to EXISTING versioned scientific implementations.

No scientific logic is re-implemented — each safe binding is a thin adapter that reads the
proposal's parameters and calls the existing repository function (adapters/steps/validation) or a
standard ASE primitive. Readiness status per action (see the matrix doc):

- READY_EXECUTOR                       : bound to real code; runs in sandbox-primary tests.
- READY_HPC_APPROVAL_GATED             : real Teacher/Student/MD; approval-gated; never run in tests.
- READY_REASONING_OUTPUT               : an Analyst typed reasoning output, not a deterministic executor.
- READY_INTERFACE_BACKEND_NOT_CONFIGURED: typed interface + sandbox adapter; no HPC backend yet.
- OUT_OF_CURRENT_SCOPE / NOT_IMPLEMENTED: excluded / no backing.

An action with no inline executor yields DRY_RUN from the dispatcher — never a fake EXECUTED.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import deterministic_executors as de
from .actions import APPROVAL_GATED_ACTIONS
from .dispatch import ActionDescriptor

ExecStatus = str  # see module docstring


@dataclass
class ExecutorBinding:
    action_type: str
    role: str
    status: ExecStatus
    backing: str
    input_contract: str
    output_artifact: str
    validator: str
    cost_class: str
    real_execution_required_later: bool
    fn: Optional[Callable] = None


def _params(proposal) -> dict:
    if hasattr(proposal, "parameters"):
        return getattr(proposal, "parameters") or {}
    if isinstance(proposal, dict):
        return proposal.get("parameters", {}) or {}
    return {}


def _artifact(obj, out_path=None) -> dict:
    result = {"metrics": obj}
    if out_path:
        from workflow.integrity import sha256_file
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, default=float, indent=2))
        result["path"] = str(p)
        result["sha256"] = sha256_file(p)
    return result


# --- inline safe adapters (kept here; siblings live in deterministic_executors.py) -----------

def _exec_compute_nve_drift(proposal):
    from validation.structure_dynamics import compute_nve_drift
    p = _params(proposal)
    drift, _ = compute_nve_drift([float(x) for x in p["energies"]],
                                 float(p.get("timestep_fs", 1.0)), int(p["n_atoms"]),
                                 sample_interval_steps=int(p.get("sample_interval_steps", 1)))
    return _artifact({"nve_drift": float(drift), "n_atoms": int(p["n_atoms"])}, p.get("out_path"))


def _exec_committee_disagreement(proposal):
    from adapters.uncertainty import committee_force_std
    p = _params(proposal)
    per_atom, frame = committee_force_std(p["forces_per_seed"], aggregate=p.get("aggregate", "max"))
    return _artifact({"u_per_atom": list(map(float, per_atom)), "u_frame": float(frame)},
                     p.get("out_path"))


def _exec_build_uncertainty_report(proposal):
    """Composite ml-trainer driver for the ``uncertainty`` production stage: reuses the existing
    authoritative ``adapters.uncertainty.committee_force_std`` executor over per-seed forces
    ALREADY embedded on the declared held-out/deployment population by
    ``workflow.steps.evaluate_committee`` (``student_forces_seed<NN>``) -- no new inference, no
    new uncertainty science. Self-validates via ``validation.uncertainty.validate_uncertainty_report``
    before returning, and never claims calibration without explicit calibration evidence.
    """
    import numpy as np
    from ase.io import read
    from adapters.uncertainty import committee_force_std
    from validation.uncertainty import validate_uncertainty_report
    from workflow.integrity import artifact_digest, sha256_file
    p = _params(proposal)
    committee_manifest = Path(p["committee_manifest"]).resolve()
    committee = json.loads(committee_manifest.read_text())
    seeds = sorted(int(model["seed"]) for model in committee["models"])
    if len(seeds) < 2:
        raise ValueError("uncertainty requires a committee of at least two seeds")
    population_path = Path(p["population_frames"]).resolve()
    frames = read(str(population_path), index=":")
    if not frames:
        raise ValueError("uncertainty population_frames is empty")
    aggregate = p.get("aggregate", "max")
    frame_scores = []
    u_values = []
    for index, atoms in enumerate(frames):
        per_seed = []
        for seed in seeds:
            key = f"{seed:02d}"
            field = f"student_forces_seed{key}"
            if field not in atoms.arrays:
                raise ValueError(f"population frame {index} is missing committee forces: {field}")
            per_seed.append(np.asarray(atoms.arrays[field], dtype=float))
        _, frame_score = committee_force_std(np.stack(per_seed), aggregate=aggregate)
        frame_id = str(atoms.info.get("structure_id", index))
        frame_scores.append({"frame_id": frame_id, "u_frame": float(frame_score)})
        u_values.append(float(frame_score))
    calibration_evidence = p.get("calibration_evidence")
    if calibration_evidence:
        calibration = {"status": "calibrated", "caveat": p.get(
            "calibration_caveat", "calibrated against the cited calibration_evidence")}
    else:
        calibration = {
            "status": "uncalibrated",
            "caveat": ("committee force disagreement (sigma_F) is treated as a committee "
                      "disagreement / fidelity-ranking signal only; no calibration evidence "
                      "(e.g. a held-out DFT-error regression) has been supplied for this run"),
        }
    evidence = [_evidence("committee_manifest", committee_manifest),
               _evidence("population", population_path)]
    if calibration_evidence:
        evidence.append(_evidence("calibration_evidence", calibration_evidence))
    report = {
        "schema_version": 1,
        "population": {"role": p.get("population_role", "held_out_evaluation_population"),
                       "path": str(population_path), "n_frames": len(frames)},
        "committee_manifest_path": str(committee_manifest),
        "committee_manifest_sha256": sha256_file(committee_manifest),
        "seeds": seeds, "aggregate": aggregate, "frame_scores": frame_scores,
        "u_frame_summary": {"mean": sum(u_values) / len(u_values), "max": max(u_values)},
        "calibration": calibration,
        "identified_gaps": list(p.get("identified_gaps") or []),
        "limitations": list(p.get("limitations") or []),
        "evidence": evidence,
    }
    report_path = Path(p["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    validate_uncertainty_report(report_path)
    return {"path": str(report_path.resolve()), "report": report,
            "integrity": artifact_digest(report_path)}


def _exec_generate_group_split(proposal):
    from workflow.steps import prepare_student_distillation_dataset, split_dataset
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    dataset = p["dataset"]
    merge_manifest = None
    protection_audit = p.get("protection_audit_path")
    generated = []
    try:
        if p.get("base_dataset") or p.get("augmentation_dataset"):
            required = ["base_dataset", "augmentation_dataset", "merged_dataset",
                        "merge_manifest", "protection_audit_path", "reference_yaml",
                        "base_label_manifest", "augmentation_label_manifest", "run_dir"]
            missing = [key for key in required if not p.get(key)]
            if missing:
                raise ValueError("base-plus-augmentation dataset route missing: " + ", ".join(missing))
            merge_manifest = prepare_student_distillation_dataset(
                p["base_dataset"], p["augmentation_dataset"], p["merged_dataset"],
                p["merge_manifest"], p["protection_audit_path"], p["reference_yaml"],
                p["base_label_manifest"], p["augmentation_label_manifest"], p["run_dir"],
                grouping_key=p.get("grouping_key", "parent_structure_id"),
                duplicate_policy=p.get("duplicate_policy", "deduplicate-identical-labels"),
            )
            dataset = p["merged_dataset"]
            generated.extend([p["merged_dataset"], p["merge_manifest"], p["protection_audit_path"]])
        manifest = split_dataset(dataset, p["output_dir"], p["manifest"],
                                 seed=int(p.get("seed", 2026)),
                                 validation_fraction=float(p.get("validation_fraction", 0.1)),
                                 test_fraction=float(p.get("test_fraction", 0.1)),
                                 grouping_key=p.get("grouping_key", "parent_structure_id"),
                                 allow_unique_parent_fallback=bool(p.get("allow_unique_parent_fallback", False)),
                                 validation_contract_path=p.get("validation_contract_path"))
        generated.extend([p["manifest"], *(record["path"] for record in manifest.get("splits", {}).values())])
        if protection_audit and p.get("reference_yaml"):
            from ase.io import read as _read_frames
            from validation.protected_reference import (
                assert_dataset_geometry_disjoint,
                assert_parent_lineage_allowed,
                assert_source_indices_allowed,
                validate_reference_config,
            )
            protection = validate_reference_config(p["reference_yaml"])
            split_records = sorted(manifest.get("splits", {}).items())
            selected_source_indices = set()
            for _, record in split_records:
                split_path = record["path"]
                assert_dataset_geometry_disjoint(split_path, protection["reference_fingerprints"])
                assert_parent_lineage_allowed(split_path, protection["protected_source_indices"])
                for atoms in _read_frames(str(split_path), index=":"):
                    parent = str(atoms.info["parent_structure_id"])
                    selected_source_indices.add(int(parent.split(":", 1)[1]))
            selected_source_indices = sorted(selected_source_indices)
            assert_source_indices_allowed(selected_source_indices, protection["protected_source_indices"])
            audit = {
                "schema_version": 1,
                "stage": "dataset_split",
                "selected_source_indices": selected_source_indices,
                "datasets": [
                    {"role": name, "path": record["path"]}
                    for name, record in split_records
                ],
                "checks": {
                    "protected_source_indices": "PASS",
                    "protected_logical_geometries": "PASS",
                    "protected_parent_lineage": "PASS",
                },
            }
            Path(protection_audit).parent.mkdir(parents=True, exist_ok=True)
            Path(protection_audit).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return {"path": p["manifest"], "manifest": manifest,
                "integrity": artifact_digest(p["manifest"]),
                "merge_manifest": merge_manifest,
                "protection_audit_path": protection_audit}
    except Exception:
        for raw in generated:
            try:
                path = Path(raw)
                if path.exists() and path.is_file():
                    path.unlink()
            except Exception:
                pass
        raise


def _exec_compute_rdf(proposal):
    from ase.io import read
    from validation.structure_dynamics import compute_rdf
    p = _params(proposal)
    frames = read(p["frames_path"], index=":")
    distances, partial = compute_rdf(frames, p["elements"], r_max=float(p.get("r_max", 6.0)),
                                     nbins=int(p.get("nbins", 200)))
    return _artifact({"rdf_peaks": {k: float(max(v)) for k, v in partial.items()},
                      "n_bins": len(distances)}, p.get("out_path"))


def _exec_force_error_channel(proposal):
    from ase.io import read
    from validation.four_channel_audit import channel
    p = _params(proposal)
    frames = read(p["frames_path"], index=":")
    metrics = channel(frames, p["ref_prefix"], p["pred_prefix"],
                      per_config_type=bool(p.get("per_config_type", False)))
    return _artifact({"channel": metrics}, p.get("out_path"))


def _exec_generate_run_summary(proposal):
    """Composite analyst driver for the ``analysis`` production stage: reads the state snapshot
    ``runtimes.pydantic_ai.cli._assemble_run_summary_state`` mechanically assembled from the CURRENT
    ``RunController`` (never re-derived here), computes ``campaign_outcome`` deterministically from
    that snapshot's own stage gates, and self-validates via
    ``validation.run_summary.validate_run_summary_report``. No LLM narrates or invents a stage
    outcome, gate verdict, or artifact hash here.
    """
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    state_path = Path(p["run_state_path"]).resolve()
    state = json.loads(state_path.read_text())

    stages = state["stages"]
    if all(stage["gate"] == "PASS" for stage in stages):
        campaign_outcome = "ALL_STAGES_PASSED"
    elif any(stage["gate"] in ("REVISE", "FAIL") for stage in stages):
        campaign_outcome = "RECOVERY_IN_PROGRESS_OR_REQUIRED"
    else:
        campaign_outcome = "IN_PROGRESS"

    identified_gaps = list(p.get("identified_gaps") or [])
    limitations = list(p.get("limitations") or
                       ["Generated mechanically from the recorded Controller state snapshot only; "
                        "does not independently re-verify scientific conclusions"])

    evidence = [_evidence("run_state_snapshot", state_path)]
    report = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "stages": stages,
        "gate_history": state["gate_history"],
        "recoveries": state["recoveries"],
        "campaign_outcome": campaign_outcome,
        "identified_gaps": identified_gaps,
        "limitations": limitations,
        "evidence": evidence,
    }
    report_path = Path(p["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    from validation.run_summary import validate_run_summary_report
    validate_run_summary_report(report_path)
    return {"path": str(report_path.resolve()), "report": report, "integrity": artifact_digest(report_path)}


def _exec_compute_coordination(proposal):
    from ase.io import read
    from validation.structure_dynamics import compute_coordination
    p = _params(proposal)
    frames = read(p["frames_path"], index=":")
    coord = compute_coordination(frames, p["elements"], p["cutoffs"])
    return _artifact({"coordination": {k: float(v) for k, v in coord.items()}}, p.get("out_path"))


def _exec_build_physical_validation_report(proposal):
    """Composite simulation driver for the ``physical_validation`` production stage: composes the
    EXISTING validation.structure_dynamics RDF/coordination/density/MSD/NVE-drift computations into
    one validate_validation_report-conformant report. Every required observable and its pass/fail
    criterion comes ONLY from the frozen validation_profile.yaml ``checks[].threshold`` field --
    this executor never invents or softens a threshold; profile entries with ``threshold: null``
    (descriptive Teacher-Student comparisons, per validation_profile.yaml's own preregistration)
    are recorded, never forced into a synthetic PASS/FAIL.
    """
    import yaml
    from ase.io import read
    from validation.structure_dynamics import (compute_rdf, compute_coordination,
                                                compute_density, compute_msd, compute_nve_drift)
    from validation.report import make_check, validate_validation_report
    from validation.species_mapping import requires_specorder, validate_specorder
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    profile_path = Path(p["validation_profile"]).resolve()
    profile = yaml.safe_load(profile_path.read_text())
    checks_cfg = {c["name"]: c for c in (profile.get("checks") or [])}
    if not checks_cfg:
        raise ValueError("validation_profile has no declared checks")

    frames_path = Path(p["frames_path"]).resolve()
    species_mapping = p.get("species_mapping")
    read_kwargs = {}
    if requires_specorder(frames_path):
        # A raw LAMMPS dump's integer atom types are meaningless without an
        # authoritative mapping (see validation.species_mapping) -- fail closed rather
        # than let ASE silently reinterpret them as atomic numbers.
        if not species_mapping or not species_mapping.get("specorder"):
            raise ValueError(
                "physical_validation frames_path is a LAMMPS dump with integer atom types "
                "and no resolved species_mapping.specorder was supplied; bind a student_config "
                "(deploy.elements) to this stage so the Student deployment's authoritative "
                "type ordering can be resolved before dispatch")
        read_kwargs["specorder"] = validate_specorder(species_mapping["specorder"])
    frames = read(str(frames_path), index=":", **read_kwargs)
    if not frames:
        raise ValueError("physical_validation frames_path is empty")
    elements = p.get("elements") or sorted({s for atoms in frames for s in atoms.get_chemical_symbols()})
    r_max = float(p.get("r_max", 6.0))
    nbins = int(p.get("nbins", 200))
    cutoffs = p.get("cutoffs") or {}

    checks = []

    def emit(name, domain, value, unit, details=None):
        cfg = checks_cfg.get(name)
        if cfg is None or not cfg.get("required", False):
            return
        criterion = cfg.get("threshold")
        checks.append(make_check(domain, name, value=value, unit=unit, criterion=criterion,
                                 details=details))

    _, partial = compute_rdf(frames, elements, r_max=r_max, nbins=nbins)
    for pair, values in partial.items():
        e1, e2 = pair.split("-")
        candidates = [f"rdf_{e1}_{e2}", f"rdf_{e2}_{e1}"]
        name = next((c for c in candidates if c in checks_cfg), candidates[0])
        emit(name, "structure", float(max(values)), "peak_g(r)")

    coordination = compute_coordination(frames, elements, cutoffs)
    for element, value in coordination.items():
        emit(f"coordination_{element}", "structure", float(value), "count")

    mean_density, std_density = compute_density(frames)
    emit("density", "structure", float(mean_density), "g/cm3", details={"standard_deviation": std_density})

    if "msd_selfdiffusion" in checks_cfg:
        msd_series = compute_msd(frames)
        syms = frames[0].get_chemical_symbols()
        counts = {el: syms.count(el) for el in msd_series}
        total_atoms = sum(counts.values())
        mean_final_msd = sum(float(series[-1]) * counts[el] for el, series in msd_series.items()) / total_atoms
        emit("msd_selfdiffusion", "dynamics", mean_final_msd, "Angstrom^2",
             details={"per_element_final_msd": {el: float(series[-1]) for el, series in msd_series.items()}})

    if "energies" in p:
        drift, resid = compute_nve_drift([float(x) for x in p["energies"]],
                                         float(p.get("timestep_fs", 1.0)), int(p["n_atoms"]),
                                         sample_interval_steps=int(p.get("sample_interval_steps", 1)))
        emit("nve_drift", "dynamics", float(drift), "meV/atom/ns", details={"residual_std": resid})

    declared_required = {name for name, cfg in checks_cfg.items() if cfg.get("required")}
    produced = {c["observable"] for c in checks}
    missing = declared_required - produced
    if missing:
        raise ValueError("physical_validation did not produce required observables: " +
                         ", ".join(sorted(missing)))

    evidence = [_evidence("validation_profile", profile_path), _evidence("frames", frames_path)]
    report = {
        "schema_version": 1,
        "profile": profile.get("kind", "physical_validation"),
        "checks": checks,
        "evidence": evidence,
    }
    if species_mapping:
        report["species_mapping"] = species_mapping
    report_path = Path(p["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    validate_validation_report(report_path)
    return {"path": str(report_path.resolve()), "report": report, "integrity": artifact_digest(report_path)}


def _exec_compare_coverage(proposal):
    from validation.data_coverage import validate_data_coverage_report
    p = _params(proposal)
    report = validate_data_coverage_report(
        p["manifest_path"], required_source_categories=p.get("required_source_categories"))
    return {"path": p["manifest_path"], "report": report,
            "sha256": (report.get("integrity", {}) or {}).get("sha256", "")}


def _exec_build_data_coverage_report(proposal):
    """Composite data-curator driver for the ``data_coverage`` production stage: enforces
    protected-reference exclusion (``_protect_dataset``) and acquisition-lineage consistency
    against the registered ``acquisition_manifest`` BEFORE computing real, per-config_type frame
    counts, then self-validates via
    ``validation.data_coverage.validate_data_coverage_report``. No LLM invents a coverage metric,
    threshold, or acquisition count here; unassessable access is reported NOT_ASSESSABLE, never
    filled in.
    """
    from ase.io import read
    from validation.data_coverage import validate_data_coverage_report
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    candidate_dataset = Path(p["candidate_dataset"]).resolve()
    _protect_dataset(candidate_dataset, p.get("reference_yaml"), p.get("selected_source_indices"),
                     require_lineage=bool(p.get("require_lineage", False)))
    frames = read(str(candidate_dataset), index=":")
    if not frames:
        raise ValueError("data_coverage candidate_dataset is empty")
    grouping_key = p.get("grouping_key", "parent_structure_id")
    parents = set()
    for index, atoms in enumerate(frames):
        if grouping_key not in atoms.info:
            raise ValueError(f"candidate_dataset frame {index} is missing grouping key {grouping_key!r}")
        parents.add(str(atoms.info[grouping_key]))

    acquisition_manifest = Path(p["acquisition_manifest"]).resolve()
    acquisition = json.loads(acquisition_manifest.read_text())
    if not isinstance(acquisition.get("n_frames"), int) or not isinstance(acquisition.get("elements"), list):
        raise ValueError("acquisition_manifest is missing n_frames/elements")
    if len(frames) > acquisition["n_frames"]:
        raise ValueError(
            "data_coverage candidate_dataset has more frames than the acquisition manifest "
            "declares -- lineage does not match acquisition.manifest.json"
        )
    candidate_elements = {s for atoms in frames for s in atoms.get_chemical_symbols()}
    if not candidate_elements.issubset(set(acquisition["elements"])):
        raise ValueError(
            "data_coverage candidate_dataset contains elements outside the acquisition "
            "manifest's declared elements -- lineage does not match acquisition.manifest.json"
        )

    label_source_field = p.get("label_source_field", "label_source")
    label_sources = sorted({str(atoms.info.get(label_source_field, "unlabeled")) for atoms in frames})
    category = p.get("category", "proposed_acquisition")
    evidence_role = p.get("evidence_role", "proposed_distillation_structures")
    source = {
        "category": category, "n_parents": len(parents), "n_frames": len(frames),
        "fraction": 1.0, "label_sources": label_sources, "evidence_role": evidence_role,
        "statistics": {"kind": "ase", "grouping_key": grouping_key,
                      "label_source_field": label_source_field},
    }

    config_types = sorted({str(atoms.info.get("config_type", "unlabeled")) for atoms in frames})
    counts = {ct: sum(1 for atoms in frames if str(atoms.info.get("config_type", "unlabeled")) == ct)
             for ct in config_types}
    dimensions = {"config_type_coverage": {"method": "frame_count_by_config_type",
                                           "config_types": config_types, "counts": counts}}

    dataset_policy = p.get("dataset_policy")
    if not dataset_policy:
        import yaml
        dataset_policy = str(Path(p["report_path"]).with_name("dataset_policy.yaml"))
        Path(dataset_policy).parent.mkdir(parents=True, exist_ok=True)
        Path(dataset_policy).write_text(yaml.safe_dump(
            {"provenance": {"note": "auto-generated default dataset policy"}}))
    dataset_policy = str(Path(dataset_policy).resolve())

    teacher_training_data_access = p.get("teacher_training_data_access", "representative")
    coverage_status = p.get("coverage_status") or (
        "NOT_ASSESSABLE" if teacher_training_data_access == "unavailable" else "PARTIAL")
    identified_gaps = list(p.get("identified_gaps") or (
        [] if teacher_training_data_access == "full" else
        ["Teacher training distribution is not independently re-verified in this run"]))
    limitations = list(p.get("limitations") or
                       ["Coverage computed per config_type frame counts only; no density-manifold "
                        "coverage metric is computed"])

    evidence = [_evidence("dataset_policy", dataset_policy),
               _evidence(evidence_role, candidate_dataset),
               _evidence("acquisition_manifest", acquisition_manifest)]
    report = {
        "schema_version": 1,
        "teacher_training_data_access": teacher_training_data_access,
        "coverage_status": coverage_status,
        "deployment_domain": p.get("deployment_domain") or {"structure_classes": ["default"]},
        "dataset_sources": [source],
        "coverage_dimensions": dimensions,
        "replay_policy": p.get("replay_policy") or {"enabled": False},
        "identified_gaps": identified_gaps, "limitations": limitations,
        "dataset_policy": dataset_policy, "evidence": evidence,
    }
    report_path = Path(p["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    validate_data_coverage_report(report_path)
    return {"path": str(report_path.resolve()), "report": report,
            "integrity": artifact_digest(report_path)}


# --- binding helpers ------------------------------------------------------------



# --- production guard helpers ---------------------------------------------------


def _protect_dataset(path, reference_yaml=None, selected_source_indices=None, require_lineage=True):
    if not reference_yaml:
        return
    from validation.protected_reference import (
        assert_dataset_geometry_disjoint,
        assert_parent_lineage_allowed,
        assert_source_indices_allowed,
        validate_reference_config,
    )
    protection = validate_reference_config(reference_yaml)
    assert_source_indices_allowed(selected_source_indices or [], protection["protected_source_indices"])
    assert_dataset_geometry_disjoint(path, protection["reference_fingerprints"])
    if require_lineage:
        assert_parent_lineage_allowed(path, protection["protected_source_indices"])


class _AcquisitionPlanError(ValueError):
    """Fail-closed acquisition planning/contract violation before adapter dispatch."""


_REQUIRED_ACQUISITION_PLAN_FIELDS = {
    "schema_version", "eligible_source_categories", "selected_parent_structure_ids",
    "selected_source_global_indices", "n_parents", "n_per_structure", "T_K", "beta",
    "sigma_range_A", "cell_sigma", "seed", "expected_output_count", "duplicate_handling",
    "protected_reference_exclusion_report",
}
_CATEGORY_KEYS = ("source_category", "category", "config_type", "structural_domain", "domain")
_INDEX_KEYS = ("source_global_index", "global_index", "seed_pool_index")


def _load_structured_file(path):
    import yaml
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise _AcquisitionPlanError(f"AcquisitionPlan is missing: {p}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _canonical_json_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _sha256_file_local(path):
    from workflow.integrity import sha256_file
    return sha256_file(path)


def _acquisition_plan_payload(raw):
    if isinstance(raw, dict):
        payload = dict(raw)
        payload.setdefault("_plan_path", None)
        payload["_plan_sha256"] = _sha256_bytes(_canonical_json_bytes({k: v for k, v in payload.items() if not str(k).startswith("_")}))
        return payload
    if not raw:
        raise _AcquisitionPlanError("AcquisitionPlan is required before acquire_structures execution")
    payload = _load_structured_file(raw)
    if not isinstance(payload, dict):
        raise _AcquisitionPlanError("AcquisitionPlan must be a JSON/YAML mapping")
    plan_path = Path(raw).expanduser().resolve()
    payload["_plan_path"] = str(plan_path)
    payload["_plan_sha256"] = _sha256_file_local(plan_path)
    return payload


def acquisition_plan_sha256_from_proposal(proposal):
    p = _params(proposal)
    plan = _acquisition_plan_payload(p.get("acquisition_plan") or p.get("acquisition_plan_path"))
    declared = p.get("acquisition_plan_sha256")
    if declared is not None and declared != plan["_plan_sha256"]:
        raise _AcquisitionPlanError("proposal acquisition_plan_sha256 does not match AcquisitionPlan")
    return plan["_plan_sha256"]


def _nonempty_list(plan, key):
    value = plan.get(key)
    if not isinstance(value, list) or not value:
        raise _AcquisitionPlanError(f"AcquisitionPlan.{key} must be a non-empty list")
    return value


def _frame_source_index(atoms, fallback):
    for key in _INDEX_KEYS:
        if key in atoms.info:
            value = atoms.info[key]
            if isinstance(value, bool):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                raise _AcquisitionPlanError(f"source frame has malformed {key}: {value!r}")
    return int(fallback)


def _frame_structure_id(atoms, source_index):
    for key in ("structure_id", "parent_structure_id", "id"):
        value = atoms.info.get(key)
        if value not in (None, ""):
            return str(value)
    return f"seed-pool:{int(source_index)}"


def _frame_categories(atoms):
    cats = set()
    for key in _CATEGORY_KEYS:
        value = atoms.info.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple, set)):
            cats.update(str(v) for v in value)
        else:
            cats.add(str(value))
    return cats


def _selected_source_records(seed_structures, selected_indices):
    from ase.io import read
    frames = read(str(seed_structures), index=":")
    by_index = {}
    for offset, atoms in enumerate(frames):
        source_index = _frame_source_index(atoms, offset)
        if source_index in by_index:
            raise _AcquisitionPlanError(f"source dataset has duplicate source index: {source_index}")
        by_index[source_index] = {
            "source_index": source_index,
            "structure_id": _frame_structure_id(atoms, source_index),
            "categories": _frame_categories(atoms),
            "atoms": atoms,
        }
    records = []
    for selected in selected_indices:
        if selected not in by_index:
            raise _AcquisitionPlanError(f"selected source index is absent from frozen source dataset: {selected}")
        records.append(by_index[selected])
    return records


def _assert_selected_sources_geometry_disjoint(records, reference_yaml):
    if not reference_yaml:
        return
    from validation.protected_reference import _structure_fingerprint, validate_reference_config
    protection = validate_reference_config(reference_yaml)
    overlaps = [record["source_index"] for record in records
                if _structure_fingerprint(record["atoms"]) in protection["reference_fingerprints"]]
    if overlaps:
        raise _AcquisitionPlanError(
            "protected logical reference geometry selected as acquisition parent: " +
            ", ".join(map(str, overlaps[:20])))


def _public_source_records(records):
    public = []
    for record in records:
        public.append({
            "source_index": int(record["source_index"]),
            "structure_id": str(record["structure_id"]),
            "categories": sorted(str(x) for x in record.get("categories", [])),
        })
    return public


def _validate_selected_sources(plan, *, seed_structures, reference_yaml=None):
    selected = list(plan["selected_source_global_indices"])
    parents = [str(x) for x in plan["selected_parent_structure_ids"]]
    records = _selected_source_records(seed_structures, selected)
    _assert_selected_sources_geometry_disjoint(records, reference_yaml)
    eligible = {str(x) for x in plan["eligible_source_categories"]}
    for expected_parent, record in zip(parents, records):
        if expected_parent != record["structure_id"]:
            raise _AcquisitionPlanError(
                "selected_parent_structure_ids must match actual source frame identities at selected_source_global_indices"
            )
        if not record["categories"]:
            raise _AcquisitionPlanError(f"selected source index {record['source_index']} has no category metadata")
        if record["categories"].isdisjoint(eligible):
            raise _AcquisitionPlanError(
                f"selected source index {record['source_index']} categories {sorted(record['categories'])} "
                f"do not match eligible_source_categories {sorted(eligible)}"
            )
    return records


def _validate_acquisition_plan(plan, *, reference_yaml=None, seed_structures=None,
                               proposal_selected_source_indices=None):
    missing = sorted(_REQUIRED_ACQUISITION_PLAN_FIELDS - set(plan))
    if missing:
        raise _AcquisitionPlanError("AcquisitionPlan missing required fields: " + ", ".join(missing))
    if plan.get("schema_version") != 1:
        raise _AcquisitionPlanError("AcquisitionPlan requires schema_version=1")
    parents = _nonempty_list(plan, "selected_parent_structure_ids")
    selected = _nonempty_list(plan, "selected_source_global_indices")
    categories = _nonempty_list(plan, "eligible_source_categories")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in selected):
        raise _AcquisitionPlanError("AcquisitionPlan.selected_source_global_indices must contain integers")
    if proposal_selected_source_indices is not None and list(proposal_selected_source_indices) != selected:
        raise _AcquisitionPlanError("proposal selected_source_indices must exactly match AcquisitionPlan selected_source_global_indices")
    n_parents = int(plan.get("n_parents"))
    n_per = int(plan.get("n_per_structure"))
    expected = int(plan.get("expected_output_count"))
    if n_parents <= 0 or n_per <= 0 or expected <= 0:
        raise _AcquisitionPlanError("AcquisitionPlan parent/count fields must be positive")
    if n_parents != len(parents) or n_parents != len(selected):
        raise _AcquisitionPlanError("AcquisitionPlan n_parents must match selected parent/source counts")
    if expected != n_parents * n_per:
        raise _AcquisitionPlanError("AcquisitionPlan expected_output_count must equal n_parents * n_per_structure")
    sigma = plan.get("sigma_range_A")
    if (not isinstance(sigma, list) or len(sigma) != 2 or
            any(not isinstance(x, (int, float)) for x in sigma) or sigma[0] > sigma[1]):
        raise _AcquisitionPlanError("AcquisitionPlan.sigma_range_A must be [min, max]")
    if not all(isinstance(x, str) and x.strip() for x in parents + categories):
        raise _AcquisitionPlanError("AcquisitionPlan parent/category values must be non-empty strings")
    report = plan.get("protected_reference_exclusion_report")
    if not isinstance(report, dict) or report.get("status") != "PASS":
        raise _AcquisitionPlanError("AcquisitionPlan protected_reference_exclusion_report must be a PASS record")
    if report.get("dft_labels_used_as_selection_scores") is not False:
        raise _AcquisitionPlanError("AcquisitionPlan must record DFT labels were not used as selection scores")
    if reference_yaml:
        from validation.protected_reference import assert_source_indices_allowed, validate_reference_config
        protection = validate_reference_config(reference_yaml)
        assert_source_indices_allowed(selected, protection["protected_source_indices"])
        if report.get("reference_id") and report.get("reference_id") != protection["reference_id"]:
            raise _AcquisitionPlanError("AcquisitionPlan protection report reference_id does not match run-bound reference")
    source_records = []
    if seed_structures:
        source_records = _validate_selected_sources(plan, seed_structures=seed_structures,
                                                   reference_yaml=reference_yaml)
    return {**plan, "selected_source_global_indices": selected,
            "selected_parent_structure_ids": parents, "eligible_source_categories": categories,
            "n_parents": n_parents, "n_per_structure": n_per, "expected_output_count": expected,
            "_selected_source_records": _public_source_records(source_records)}


def _executable_config_path(params, plan):
    raw = params.get("executable_config_path") or plan.get("executable_config_path")
    if raw:
        return Path(raw).expanduser().resolve()
    manifest = params.get("manifest_path") or params.get("out_path")
    if not manifest:
        raise _AcquisitionPlanError("acquisition execution requires manifest_path or executable_config_path")
    return Path(manifest).resolve().with_name("acquisition_augment_atoms.native.yaml")


def _schema_default(acquisition_cfg, key, default):
    value = ((acquisition_cfg.get("installed_schema") or {}).get("config") or {}).get(key)
    if isinstance(value, dict) and "default" in value:
        return value["default"]
    return default


def _framework_plan_envelope(plan):
    envelope = {
        "schema_version": int(plan["schema_version"]),
        "selected_parent_structure_ids": list(plan["selected_parent_structure_ids"]),
        "selected_source_global_indices": list(plan["selected_source_global_indices"]),
        "eligible_source_categories": list(plan["eligible_source_categories"]),
        "n_parents": int(plan["n_parents"]),
        "n_per_structure": int(plan["n_per_structure"]),
        "T_K": float(plan["T_K"]),
        "beta": float(plan["beta"]),
        "sigma_range_A": [float(plan["sigma_range_A"][0]), float(plan["sigma_range_A"][1])],
        "cell_sigma": plan.get("cell_sigma"),
        "seed": int(plan["seed"]),
        "expected_output_count": int(plan["expected_output_count"]),
        "duplicate_handling": plan["duplicate_handling"],
        "protected_reference_exclusion_report": plan["protected_reference_exclusion_report"],
    }
    if plan.get("executable_config_path"):
        envelope["executable_config_path"] = str(Path(plan["executable_config_path"]).expanduser().resolve())
    return envelope


def _write_executable_augment_config(path, acquisition_cfg, plan, *, seed_path, out_path, teacher_config):
    import copy
    import yaml
    from adapters import load_config
    teacher_cfg = load_config(teacher_config)
    calculator = copy.deepcopy(teacher_cfg.get("calculator"))
    if not isinstance(calculator, dict) or not calculator:
        raise _AcquisitionPlanError("Teacher config must bind an ASE calculator for augment-atoms")
    cfg = {
        "data": {
            "input": str(Path(seed_path).resolve()),
            "output": str(Path(out_path).resolve()),
        },
        "model": {
            "calculator": {
                "+runtimes.pydantic_ai.augment_atoms_bridge.teacher_calculator": {
                    "teacher_config": str(Path(teacher_config).resolve()),
                },
            },
        },
        "config": {
            "n_per_structure": int(plan["n_per_structure"]),
            "T": float(plan["T_K"]),
            "beta": float(plan["beta"]),
            "sigma_range": [float(plan["sigma_range_A"][0]), float(plan["sigma_range_A"][1])],
            "seed": int(plan["seed"]),
            "units": str(plan.get("units", _schema_default(acquisition_cfg, "units", "eV"))),
            "cell_sigma": plan.get("cell_sigma"),
            "max_force": float(plan.get("max_force", _schema_default(acquisition_cfg, "max_force", 30.0))),
            "min_separation": float(plan.get("min_separation", _schema_default(acquisition_cfg, "min_separation", 0.5))),
            "max_relax_steps": int(plan.get("max_relax_steps", _schema_default(acquisition_cfg, "max_relax_steps", 20))),
            "similarity_threshold": float(plan.get("similarity_threshold", _schema_default(acquisition_cfg, "similarity_threshold", 0.1))),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return cfg


def _translate_acquisition_cli(acquisition_cfg, plan, *, seed_path, out_path, executable_config_path):
    if acquisition_cfg.get("kind") != "augment-atoms":
        return acquisition_cfg
    invocation = ((acquisition_cfg.get("cli") or {}).get("invocation") or [])
    if not invocation:
        executable = ((acquisition_cfg.get("cli") or {}).get("executable") or "augment-atoms")
        invocation = [executable, "{config_path}"]
    context = {
        "config_path": str(Path(executable_config_path).resolve()),
        "seed_path": str(Path(seed_path).resolve()),
        "out_path": str(Path(out_path).resolve()),
    }
    cfg = dict(acquisition_cfg)
    cfg["config_path"] = context["config_path"]
    cfg["defer_lineage_validation"] = True
    command = [str(part).format(**context) for part in invocation]
    executable = Path(command[0]).expanduser()
    if executable.is_absolute():
        if not executable.is_file():
            raise _AcquisitionPlanError(f"augment-atoms executable is missing: {executable}")
        command[0] = str(executable)
    elif cfg.get("env"):
        command = ["conda", "run", "-n", str(cfg["env"]), *command]
    else:
        raise _AcquisitionPlanError(
            "augment-atoms executable must be absolute or acquisition env must be configured"
        )
    cfg["command"] = command
    return cfg


def _apply_acquisition_lineage(result_path, plan):
    from ase.io import read, write
    frames = read(str(result_path), index=":")
    parents = [str(x) for x in plan["selected_parent_structure_ids"]]
    changed = False
    for atoms in frames:
        if atoms.info.get("parent_structure_id"):
            continue
        if "starting-structure" in atoms.info:
            native_start = atoms.info["starting-structure"]
            try:
                start_index = int(native_start)
            except (TypeError, ValueError) as exc:
                raise _AcquisitionPlanError(
                    f"malformed augment-atoms starting-structure value: {native_start!r}"
                ) from exc
            if start_index < 0 or start_index >= len(parents):
                raise _AcquisitionPlanError(
                    f"augment-atoms starting-structure index out of range: {start_index}"
                )
            parent = parents[start_index]
        elif len(parents) == 1:
            parent = parents[0]
        else:
            raise _AcquisitionPlanError(
                "acquired structure lacks parent_structure_id and no deterministic native starting-structure field is available"
            )
        atoms.info["parent_structure_id"] = str(parent)
        changed = True
    if changed:
        write(str(result_path), frames)
    return len(frames)


def _validate_acquisition_output(result_path, plan):
    n_frames = _apply_acquisition_lineage(result_path, plan)
    if n_frames != int(plan["expected_output_count"]):
        raise _AcquisitionPlanError(
            f"acquisition output count mismatch: expected {plan['expected_output_count']}, got {n_frames}")
    return n_frames


def _write_acquisition_protection_audit(path, *, reference_yaml, result_path, selected_source_indices):
    from validation.protected_reference import write_protection_audit, validate_protection_audit_report
    write_protection_audit("acquisition", reference_yaml,
                           [f"acquisition_candidates={Path(result_path).resolve()}"],
                           path, selected_source_indices=selected_source_indices)
    return validate_protection_audit_report(path, reference_yaml=reference_yaml,
                                            submitted_artifacts=[Path(path).resolve(), Path(result_path).resolve()])


def _quarantine_acquisition_outputs(paths):
    import shutil
    import uuid
    existing = [Path(x).resolve() for x in paths if x and Path(x).exists()]
    if not existing:
        return []
    quarantine = existing[0].parent / ".quarantine" / f"acquisition-{uuid.uuid4().hex[:12]}"
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in existing:
        target = quarantine / path.name
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return moved


def _evidence(role, path):
    from validation.report import evidence_record
    return evidence_record(role, path)


def _exec_build_teacher_baseline(proposal):
    import math
    import numpy as np
    from ase.io import read
    from adapters import load_config
    from adapters.acquisition import label_with_teacher
    from validation.teacher_baseline import validate_teacher_baseline_report
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    structures = p["structures_path"]
    reference_yaml = p.get("reference_yaml")
    _protect_dataset(structures, reference_yaml, p.get("selected_source_indices"),
                     require_lineage=bool(p.get("require_lineage", False)))
    labeled_output = p.get("labeled_output") or str(Path(p["report_path"]).with_suffix(".extxyz"))
    label_manifest = p.get("label_manifest_path") or str(
        Path(p["report_path"]).with_name("teacher_baseline_labels.manifest.json"))
    label_manifest_payload = label_with_teacher(
        load_config(p["teacher_config"]), structures, labeled_output, label_manifest,
        bool(p.get("include_stress", False)))
    frames = read(labeled_output, index=":")
    energies = [float(a.info["teacher_energy"]) for a in frames]
    fmax = []
    for atoms in frames:
        forces = np.asarray(atoms.arrays["teacher_forces"], dtype=float)
        if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
            raise ValueError("Teacher baseline produced non-finite or malformed forces")
        fmax.append(float(np.max(np.linalg.norm(forces, axis=1))))
    if not energies or not all(math.isfinite(e) for e in energies):
        raise ValueError("Teacher baseline produced no finite energies")
    report_path = Path(p["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    required = ["deployment_domain", "applicability_status", "applicability_limitations"]
    missing = [name for name in required if name not in p]
    if missing:
        raise ValueError("Teacher baseline requires explicit deployment/applicability evidence: " + ", ".join(missing))
    deployment_domain = p["deployment_domain"]
    applicability_status = p["applicability_status"]
    limitations = list(p["applicability_limitations"])
    report = {
        "schema_version": 1,
        "profile": p.get("profile", "teacher_baseline"),
        "teacher": {"kind": label_manifest_payload["teacher_kind"],
                    "config": str(Path(p["teacher_config"]).resolve()),
                    "model_sha256": label_manifest_payload.get("teacher_model_sha256")},
        "distillation_scope": str(Path(p["distillation_scope"]).resolve()),
        "validation_profile": str(Path(p["validation_profile"]).resolve()),
        "deployment_domain": deployment_domain,
        "applicability": {"status": applicability_status, "limitations": limitations},
        "checks": [{
            "domain": "operational_teacher_inference",
            "observable": "fresh_teacher_energy_force_finiteness",
            "status": "PASS",
            "value": float(max(fmax)),
            "unit": "eV/Angstrom",
            "criterion": {"operator": "max", "threshold": float(p.get("force_finite_threshold", 1.0e12))},
            "purpose": "deployment_stability",
            "reference_source": "teacher",
            "protocol": "fresh Teacher inference on declared operational structures",
            "details": {"n_frames": len(frames), "energy_min": min(energies),
                        "energy_max": max(energies),
                        "fresh_label_output": str(Path(labeled_output).resolve()),
                        "fresh_label_output_integrity": artifact_digest(labeled_output),
                        "fresh_label_manifest": str(Path(label_manifest).resolve()),
                        "fresh_label_manifest_integrity": artifact_digest(label_manifest)},
        }],
        "evidence": [
            _evidence("teacher_config", p["teacher_config"]),
            _evidence("distillation_scope", p["distillation_scope"]),
            _evidence("validation_profile", p["validation_profile"]),
            _evidence("operational_structures", structures),
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    validate_teacher_baseline_report(report_path)
    return {"path": str(report_path.resolve()), "report": report,
            "integrity": artifact_digest(report_path),
            "labeled_output": str(Path(labeled_output).resolve()),
            "label_manifest": str(Path(label_manifest).resolve())}


def _exec_acquire_structures(proposal):
    from adapters import load_config
    from adapters.acquisition import acquire
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    plan = _validate_acquisition_plan(
        _acquisition_plan_payload(p.get("acquisition_plan") or p.get("acquisition_plan_path")),
        reference_yaml=p.get("reference_yaml"),
        seed_structures=p.get("seed_structures"),
        proposal_selected_source_indices=p.get("selected_source_indices"),
    )
    if p.get("acquisition_plan_sha256") and p["acquisition_plan_sha256"] != plan["_plan_sha256"]:
        raise _AcquisitionPlanError("proposal acquisition_plan_sha256 does not match AcquisitionPlan")
    acquisition_cfg = load_config(p["acquisition_config"])
    executable_path = _executable_config_path(p, plan)
    executable_payload = _write_executable_augment_config(
        executable_path, acquisition_cfg, plan, seed_path=p["seed_structures"],
        out_path=p["out_path"], teacher_config=p["teacher_config"])
    executable_cfg = _translate_acquisition_cli(
        acquisition_cfg, plan, seed_path=p["seed_structures"], out_path=p["out_path"],
        executable_config_path=executable_path)
    result = None
    audit_path = Path(p.get("protection_audit_path") or Path(p["manifest_path"]).with_name("acquisition_protection_audit.json"))
    created_paths = [p.get("out_path"), p.get("manifest_path"), str(audit_path), str(executable_path)]
    try:
        result = acquire(executable_cfg, load_config(p["teacher_config"]),
                         p["seed_structures"], p["out_path"])
        n_frames = _validate_acquisition_output(result, plan)
        audit_result = _write_acquisition_protection_audit(
            audit_path, reference_yaml=p["reference_yaml"], result_path=result,
            selected_source_indices=plan["selected_source_global_indices"])
        _protect_dataset(result, p.get("reference_yaml"), require_lineage=True)
        artifact = {"path": str(Path(result).resolve()), "integrity": artifact_digest(result)}
        manifest_path = p.get("manifest_path")
        if manifest_path:
            manifest = {
                "schema_version": 1,
                "operation": "acquire_structures",
                "stage": _params(proposal).get("stage", "acquisition"),
                "acquisition_plan": str(Path(plan["_plan_path"]).resolve()) if plan.get("_plan_path") else None,
                "acquisition_plan_sha256": plan["_plan_sha256"],
                "acquisition_config": str(Path(p["acquisition_config"]).resolve()),
                "acquisition_config_integrity": artifact_digest(p["acquisition_config"]),
                "teacher_config": str(Path(p["teacher_config"]).resolve()),
                "teacher_config_integrity": artifact_digest(p["teacher_config"]),
                "seed_structures": str(Path(p["seed_structures"]).resolve()),
                "seed_structures_integrity": artifact_digest(p["seed_structures"]),
                "reference_yaml": str(Path(p["reference_yaml"]).resolve()) if p.get("reference_yaml") else None,
                "reference_yaml_integrity": artifact_digest(p["reference_yaml"]) if p.get("reference_yaml") else None,
                "selected_parent_structure_ids": list(plan["selected_parent_structure_ids"]),
                "selected_source_global_indices": list(plan["selected_source_global_indices"]),
                "eligible_source_categories": list(plan["eligible_source_categories"]),
                "selected_source_records": list(plan.get("_selected_source_records") or []),
                "expected_output_count": int(plan["expected_output_count"]),
                "actual_output_count": int(n_frames),
                "duplicate_handling": plan["duplicate_handling"],
                "dft_labels_used_as_selection_scores": False,
                "executable_config": str(executable_path),
                "framework_plan_envelope": _framework_plan_envelope(plan),
                "native_executable_config_payload": executable_payload,
                "executable_config_integrity": artifact_digest(executable_path),
                "translated_command": list(executable_cfg.get("command", [])),
                "protection_audit": str(audit_path.resolve()),
                "protection_audit_integrity": artifact_digest(audit_path),
                "protection_audit_result": audit_result,
                "output": artifact["path"],
                "output_integrity": artifact["integrity"],
            }
            Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
            Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            artifact["manifest_path"] = str(Path(manifest_path).resolve())
            artifact["manifest_integrity"] = artifact_digest(manifest_path)
            artifact["protection_audit_path"] = str(audit_path.resolve())
            artifact["protection_audit_integrity"] = artifact_digest(audit_path)
        return artifact
    except Exception:
        _quarantine_acquisition_outputs(created_paths)
        raise


def _exec_label_with_teacher(proposal):
    from adapters import load_config
    from adapters.acquisition import label_with_teacher
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    _protect_dataset(p.get("structures_path", p.get("structures")), p.get("reference_yaml"),
                     p.get("selected_source_indices"), require_lineage=True)
    manifest = label_with_teacher(
        load_config(p["teacher_config"]),
        p.get("structures_path", p.get("structures")),
        p.get("out_path", p.get("labeled_output")),
        p["manifest_path"],
        bool(p.get("include_stress", False)),
    )
    return {"path": manifest["output"], "sha256": manifest["sha256"],
            "manifest_path": str(Path(p["manifest_path"]).resolve()),
            "manifest_integrity": artifact_digest(p["manifest_path"])}



def _exec_validate_teacher_reference(proposal):
    import tempfile
    from ase.io import read
    from adapters import load_config
    from adapters.acquisition import label_with_teacher
    from adapters.teacher import teacher_model_reference
    from validation.four_channel_audit import channel
    from validation.protected_reference import validate_reference_config
    from validation.reference_validation import validate_reference_validation_report
    from workflow.integrity import artifact_digest, sha256_file
    p = _params(proposal)
    reference_yaml = p["reference_yaml"]
    teacher_config = p["teacher_config"]
    protection = validate_reference_config(reference_yaml)
    predictions_path = Path(p["predictions_path"]).resolve()
    report_path = Path(p["report_path"]).resolve()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_cfg = load_config(teacher_config)
    with tempfile.TemporaryDirectory(prefix="teacher-reference-labels-") as tmp:
        label_manifest_path = Path(tmp) / "teacher_labels.manifest.json"
        label_with_teacher(teacher_cfg, protection["reference_path"], predictions_path,
                           label_manifest_path, bool(p.get("include_stress", False)))
        label_manifest = json.loads(label_manifest_path.read_text())
    frames = read(str(predictions_path), index=":")
    metrics_raw = channel(frames, "dft", "teacher", per_config_type=True, require_complete=True)

    def metric_subset(src):
        return {
            "n_frames": int(src["n_frames"]),
            "n_atoms": int(src["n_atoms"]),
            "energy_mae": float(src["e_raw_mae_meV"]),
            "energy_rmse": float(src["e_raw_rmse_meV"]),
            "force_component_mae": float(src["f_mae"]),
            "force_component_rmse": float(src["f_rmse"]),
        }

    by_config_type = {key: metric_subset(value) for key, value in metrics_raw.items() if key != "all"}
    model_value = teacher_model_reference(teacher_cfg)
    model_path = Path(model_value).expanduser().resolve() if model_value else None
    teacher_config_path = Path(teacher_config).expanduser().resolve()
    domain_fields = {"config_type": "present" if by_config_type else "absent"}
    for field in p.get("domain_fields", ["structural_domain"]):
        if field != "config_type":
            domain_fields[field] = "present" if any(field in atoms.info for atoms in frames) else "absent"
    report = {
        "schema_version": 1,
        "profile": "teacher_reference_validation",
        "stage": "reference_validation",
        "protected_reference_use": "teacher_vs_dft_reference_validation_only",
        "historical_prediction_policy": "PROVENANCE_ONLY_NOT_USED_AS_FRESH_RESULT",
        "teacher": {
            "config": str(teacher_config_path),
            "config_integrity": artifact_digest(teacher_config_path),
            "model": str(model_path) if model_path else model_value,
            "model_integrity": artifact_digest(model_path) if model_path and model_path.exists() else None,
            "model_sha256": sha256_file(model_path) if model_path and model_path.is_file() else None,
            "label_manifest": label_manifest,
        },
        "reference": {
            "reference_id": protection["reference_id"],
            "reference_yaml": str(Path(reference_yaml).expanduser().resolve()),
            "structures_path": str(protection["reference_path"]),
            "logical_frames": int(protection["logical_frames"]),
            "protected_source_rows": int(protection["protected_source_rows"]),
            "structures_integrity": artifact_digest(protection["reference_path"]),
        },
        "prediction_artifact": {
            "path": str(predictions_path),
            "integrity": artifact_digest(predictions_path),
            "n_frames": len(frames),
            "labels": ["teacher_energy", "teacher_forces", "dft_energy", "dft_forces"],
        },
        "metrics": {
            "energy_normalization": "per_atom",
            "energy_unit": "meV/atom",
            "force_unit": "eV/Angstrom",
            "global": metric_subset(metrics_raw["all"]),
            "by_config_type": by_config_type,
            "domain_fields": domain_fields,
        },
        "checks": [
            {"domain": "teacher_reference", "observable": "logical_frame_count", "status": "RECORDED", "value": int(protection["logical_frames"]), "unit": "frames", "criterion": None},
            {"domain": "teacher_reference", "observable": "protected_source_row_count", "status": "RECORDED", "value": int(protection["protected_source_rows"]), "unit": "rows", "criterion": None},
            {"domain": "teacher_reference", "observable": "energy_mae", "status": "RECORDED", "value": metric_subset(metrics_raw["all"])["energy_mae"], "unit": "meV/atom", "criterion": None},
            {"domain": "teacher_reference", "observable": "energy_rmse", "status": "RECORDED", "value": metric_subset(metrics_raw["all"])["energy_rmse"], "unit": "meV/atom", "criterion": None},
            {"domain": "teacher_reference", "observable": "force_component_mae", "status": "RECORDED", "value": metric_subset(metrics_raw["all"])["force_component_mae"], "unit": "eV/Angstrom", "criterion": None},
            {"domain": "teacher_reference", "observable": "force_component_rmse", "status": "RECORDED", "value": metric_subset(metrics_raw["all"])["force_component_rmse"], "unit": "eV/Angstrom", "criterion": None},
        ],
        "evidence": [
            _evidence("teacher_config", teacher_config),
            _evidence("protected_reference_config", reference_yaml),
            _evidence("protected_reference_structures", protection["reference_path"]),
            _evidence("teacher_reference_predictions", predictions_path),
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    validate_reference_validation_report(report_path, reference_yaml=reference_yaml,
                                         teacher_config=teacher_config,
                                         submitted_artifacts=[report_path, predictions_path])
    return {"path": str(report_path), "report": report, "integrity": artifact_digest(report_path),
            "predictions_path": str(predictions_path),
            "predictions_integrity": artifact_digest(predictions_path)}

def _exec_run_teacher_md(proposal):
    from adapters import load_config
    from adapters.acquisition import run_teacher_md
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    out = run_teacher_md(load_config(p["md_config"]), load_config(p["teacher_config"]),
                         p["seed_structures"], p["out_path"])
    return {"path": str(Path(out).resolve()), "integrity": artifact_digest(out)}


def _exec_train_committee(proposal):
    from workflow.steps import train_committee
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    _protect_dataset(p["dataset"], p.get("reference_yaml"),
                     p.get("selected_source_indices"), require_lineage=True)
    manifest = train_committee(p["student_config"], p["dataset"], p["output_dir"],
                               p["manifest_path"])
    return {"path": str(Path(p["manifest_path"]).resolve()), "manifest": manifest,
            "integrity": artifact_digest(p["manifest_path"])}


def _exec_evaluate_committee(proposal):
    from workflow.steps import evaluate_committee
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    report = evaluate_committee(
        p["student_config"], p["committee_manifest"], p["frames_path"],
        p["labeled_output"], p["report_path"], p.get("required_channels"))
    return {"path": str(Path(p["report_path"]).resolve()), "report": report,
            "integrity": artifact_digest(p["report_path"]),
            "labeled_output": str(Path(p["labeled_output"]).resolve())}


def _exec_run_student_md(proposal):
    from workflow.steps import run_md
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    manifest = run_md(
        p["md_config"], p["student_config"], p["checkpoint"], p["template_name"],
        p["context_yaml"], p["input_path"], p["run_dir"], p["manifest_path"],
        p.get("committee_manifest"), p.get("selected_seed"), p.get("evidence_paths"))
    return {"path": str(Path(p["manifest_path"]).resolve()), "manifest": manifest,
            "integrity": artifact_digest(p["manifest_path"])}


def _ready(action, role, backing, contract, out, fn, validator="", cost="light"):
    return ExecutorBinding(action, role, "READY_EXECUTOR", backing, contract, out, validator,
                           cost, False, fn=fn)


def _hpc(action, role, backing, contract, out, validator=""):
    return ExecutorBinding(action, role, "READY_HPC_APPROVAL_GATED", backing, contract, out,
                           validator, "hpc", True, fn=None)


def _interface(action, role, contract):
    return ExecutorBinding(action, role, "READY_INTERFACE_BACKEND_NOT_CONFIGURED",
                           "scheduler interface (no HPC backend configured)", contract,
                           "job identity / status / collected artifact reference", "", "hpc",
                           True, fn=None)


def _reasoning(action, role, out):
    return ExecutorBinding(action, role, "READY_REASONING_OUTPUT",
                           "Analyst typed reasoning output (not a deterministic executor)",
                           "deterministic analysis artifacts", out, "", "light", False, fn=None)


BINDINGS: dict[str, ExecutorBinding] = {b.action_type: b for b in [
    # --- Data Curator ---
    _ready("inspect_dataset", "data-curator", "ase.io.read + metadata", "frames_path",
           "dataset summary", de.inspect_dataset),
    _ready("summarize_source_categories", "data-curator",
           "frame metadata (cf. validation.data_coverage)", "frames_path[,category_key]",
           "category counts + fractions", de.summarize_source_categories),
    _ready("sample_seed_pool", "data-curator", "deterministic policy seed_pool_v1",
           "frames_path,count,seed", "selection manifest (ids + hashes)", de.sample_seed_pool),
    _ready("reconstruct_lineage", "data-curator", "adapters.acquisition.validate_lineage grouping",
           "frames_path[,group_key]", "lineage groups", de.reconstruct_lineage),
    _ready("generate_group_split", "data-curator", "workflow.steps.split_dataset",
           "dataset,output_dir,manifest", "split manifest (sha256-bound)",
           _exec_generate_group_split, validator="workflow.steps split integrity"),
    _hpc("acquire_structures", "data-curator", "adapters.acquisition.acquire",
         "acquisition_config,teacher_config,seed_structures,out_path",
         "acquired extxyz + acquisition manifest", "lineage validation"),
    _hpc("label_with_teacher", "data-curator", "adapters.acquisition.label_with_teacher",
         "teacher_config,structures_path,out_path,manifest_path",
         "labeled extxyz + labeling manifest", "labeling manifest integrity"),
    _ready("validate_label_preservation", "data-curator",
           "ase.io.read + acquisition.validate_lineage", "labeled_path[,n_source_frames]",
           "label-preservation report", de.validate_label_preservation,
           validator="artifact completeness"),
    _ready("build_dataset_manifest", "data-curator", "workflow.integrity.artifact_digest",
           "dataset[,manifest_path]", "hash-bound dataset manifest", de.build_dataset_manifest),
    _ready("compare_deployment_coverage", "data-curator",
           "validation.data_coverage.validate_data_coverage_report",
           "manifest_path[,required_source_categories]", "validated coverage report",
           _exec_compare_coverage, validator="data_coverage validator"),
    _ready("detect_duplicates", "data-curator", "workflow.steps._structure_fingerprint",
           "frames_path", "duplicate indices", de.detect_duplicates),
    _ready("detect_atomic_overlap", "data-curator", "ASE get_all_distances(mic=True)",
           "frames_path[,min_distance_threshold]", "overlapping frame indices",
           de.detect_atomic_overlap),
    _ready("build_data_coverage_report", "data-curator",
           "protected-reference guard + acquisition-lineage check + "
           "validation.data_coverage.validate_data_coverage_report",
           "candidate_dataset,acquisition_manifest[,reference_yaml,dataset_policy,...]",
           "data coverage report (schema_version=1, hash-bound)", _exec_build_data_coverage_report,
           validator="validation.data_coverage.validate_data_coverage_report"),
    # --- ML Trainer ---
    _ready("prepare_student_inputs", "ml-trainer", "adapters.student.render_student_inputs",
           "student_config,out_dir", "rendered student input config", de.prepare_student_inputs),
    _hpc("train_committee", "ml-trainer", "workflow.steps.train_committee",
         "student_config,dataset,output_dir,manifest_path", "committee manifest + checkpoints", ""),
    _ready("collect_checkpoints", "ml-trainer", "committee manifest convention",
           "committee_manifest", "checkpoint paths + integrity", de.collect_checkpoints),
    _hpc("evaluate_heldout_fidelity", "ml-trainer", "workflow.steps.evaluate_committee",
         "student_config,committee_manifest,frames_path,labeled_output,report_path",
         "3-channel fidelity report", "four_channel_audit"),
    _ready("summarize_seed_variation", "ml-trainer", "adapters.uncertainty.committee_force_std",
           "forces_per_seed", "seed-variation summary", de.summarize_seed_variation),
    _ready("compute_committee_disagreement", "ml-trainer",
           "adapters.uncertainty.committee_force_std", "forces_per_seed[,aggregate]",
           "committee u_per_atom + u_frame", _exec_committee_disagreement),
    _ready("validate_training_completion", "ml-trainer", "workflow.integrity.artifact_digest",
           "committee_manifest[,expected_seeds]", "training-completeness report",
           de.validate_training_completion, validator="artifact completeness"),
    _ready("build_uncertainty_report", "ml-trainer",
           "adapters.uncertainty.committee_force_std over registered per-seed committee forces",
           "committee_manifest,population_frames[,aggregate,population_role,calibration_evidence]",
           "uncertainty report (hash-bound to committee manifest)", _exec_build_uncertainty_report,
           validator="validation.uncertainty.validate_uncertainty_report"),
    # --- Simulation ---
    _hpc("build_teacher_baseline", "simulation",
         "adapters.acquisition.label_with_teacher + validation.teacher_baseline",
         "teacher_config,structures_path,distillation_scope,validation_profile,report_path",
         "TeacherBaselineReport", "validation.teacher_baseline"),
    _hpc("validate_teacher_reference", "simulation",
         "adapters.acquisition.label_with_teacher + validation.four_channel_audit + validation.reference_validation",
         "teacher_config,reference_yaml,predictions_path,report_path",
         "reference validation report + fresh Teacher reference predictions",
         "validation.reference_validation"),
    _hpc("run_teacher_md", "simulation", "adapters.acquisition.run_teacher_md",
         "md_config,teacher_config,seed_structures,out_path", "teacher MD snapshots", ""),
    _hpc("run_student_md", "simulation", "workflow.steps.run_md / adapters.md_backend.run",
         "md_cfg,student_cfg,checkpoint,...", "MD trajectory + manifest", ""),
    _ready("compute_rdf", "simulation", "validation.structure_dynamics.compute_rdf",
           "frames_path,elements[,r_max,nbins]", "partial RDF peaks", _exec_compute_rdf),
    _ready("compute_coordination", "simulation", "validation.structure_dynamics.compute_coordination",
           "frames_path,elements,cutoffs", "mean coordination", _exec_compute_coordination),
    _ready("compute_minimum_distance", "simulation", "ASE get_all_distances(mic=True)",
           "frames_path", "min distance per frame (A)", de.compute_minimum_distance),
    _ready("detect_force_spike", "simulation", "ASE forces + norm",
           "frames_path[,force_threshold]", "force-spike frame indices (eV/A)",
           de.detect_force_spike),
    _ready("compute_nve_drift", "simulation", "validation.structure_dynamics.compute_nve_drift",
           "energies,timestep_fs,n_atoms", "NVE drift meV/atom/ns", _exec_compute_nve_drift),
    _ready("validate_simulation_completion", "simulation",
           "artifact existence + finiteness", "md_manifest[,trajectory_path,energies]",
           "simulation-completeness report", de.validate_simulation_completion,
           validator="artifact completeness"),
    _ready("build_physical_validation_report", "simulation",
           "validation.structure_dynamics.{compute_rdf,compute_coordination,compute_density,"
           "compute_msd,compute_nve_drift} against the frozen validation_profile.yaml",
           "validation_profile,frames_path[,elements,cutoffs,energies,n_atoms,timestep_fs]",
           "physical validation report (schema_version=1, threshold-bound)",
           _exec_build_physical_validation_report,
           validator="validation.report.validate_validation_report"),
    _interface("submit_scheduler_job", "simulation", "SchedulerSubmissionProposal (protocol+config hash, idempotency)"),
    _interface("query_scheduler_job", "simulation", "job identity"),
    _interface("collect_scheduler_artifact", "simulation", "job identity -> artifact reference"),
    # --- Analyst ---
    _ready("compare_force_errors", "analyst", "validation.four_channel_audit.channel",
           "frames_path,ref_prefix,pred_prefix", "force error channel metrics",
           _exec_force_error_channel),
    _ready("compare_energy_errors", "analyst", "validation.four_channel_audit.channel",
           "frames_path,ref_prefix,pred_prefix", "energy error channel metrics",
           _exec_force_error_channel),
    _ready("summarize_committee_disagreement", "analyst",
           "adapters.uncertainty.committee_force_std", "forces_per_seed", "u summary",
           _exec_committee_disagreement),
    _ready("compare_rdf", "analyst", "validation.structure_dynamics.compute_rdf",
           "frames_path,elements", "RDF peaks", _exec_compute_rdf),
    _ready("compare_coordination", "analyst", "validation.structure_dynamics.compute_coordination",
           "frames_path,elements,cutoffs", "coordination", _exec_compute_coordination),
    _ready("fit_nve_drift", "analyst", "validation.structure_dynamics.compute_nve_drift",
           "energies,timestep_fs,n_atoms", "NVE drift fit", _exec_compute_nve_drift),
    _ready("summarize_md_stability", "analyst",
           "compose NVE drift + min distance (validation.structure_dynamics)",
           "energies|frames_path", "MD-stability summary", de.summarize_md_stability),
    _reasoning("classify_root_cause", "analyst", "RootCauseClassification (typed)"),
    _ready("generate_run_summary", "analyst",
           "runtimes.pydantic_ai.cli._assemble_run_summary_state Controller-state snapshot",
           "run_state_path[,identified_gaps,limitations]",
           "run summary report (schema_version=1, hash-bound to Controller state)",
           _exec_generate_run_summary, validator="validation.run_summary.validate_run_summary_report"),
]}


def build_executor_registry() -> dict:
    reg: dict = {}
    for action, b in BINDINGS.items():
        executor = b.fn
        if action == "build_teacher_baseline":
            executor = _exec_build_teacher_baseline
        elif action == "validate_teacher_reference":
            executor = _exec_validate_teacher_reference
        elif action == "acquire_structures":
            executor = _exec_acquire_structures
        elif action == "label_with_teacher":
            executor = _exec_label_with_teacher
        elif action == "run_teacher_md":
            executor = _exec_run_teacher_md
        elif action == "train_committee":
            executor = _exec_train_committee
        elif action == "evaluate_heldout_fidelity":
            executor = _exec_evaluate_committee
        elif action == "run_student_md":
            executor = _exec_run_student_md
        reg[action] = ActionDescriptor(
            action_type=action, role=b.role, cost_class=b.cost_class,
            approval_boundary=APPROVAL_GATED_ACTIONS.get(action), executor=executor)
    return reg


def executor_status(action: str) -> Optional[str]:
    b = BINDINGS.get(action)
    return b.status if b else None
