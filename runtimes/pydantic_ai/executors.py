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


def _exec_generate_group_split(proposal):
    from workflow.steps import split_dataset
    p = _params(proposal)
    manifest = split_dataset(p["dataset"], p["output_dir"], p["manifest"],
                             seed=int(p.get("seed", 2026)),
                             validation_fraction=float(p.get("validation_fraction", 0.1)))
    return {"path": p["manifest"], "manifest": manifest,
            "sha256": manifest.get("integrity", {}).get("sha256", "")}


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


def _exec_compute_coordination(proposal):
    from ase.io import read
    from validation.structure_dynamics import compute_coordination
    p = _params(proposal)
    frames = read(p["frames_path"], index=":")
    coord = compute_coordination(frames, p["elements"], p["cutoffs"])
    return _artifact({"coordination": {k: float(v) for k, v in coord.items()}}, p.get("out_path"))


def _exec_compare_coverage(proposal):
    from validation.data_coverage import validate_data_coverage_report
    p = _params(proposal)
    report = validate_data_coverage_report(
        p["manifest_path"], required_source_categories=p.get("required_source_categories"))
    return {"path": p["manifest_path"], "report": report,
            "sha256": (report.get("integrity", {}) or {}).get("sha256", "")}


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
    _protect_dataset(p["seed_structures"], p.get("reference_yaml"),
                     p.get("selected_source_indices"), require_lineage=False)
    result = acquire(load_config(p["acquisition_config"]), load_config(p["teacher_config"]),
                     p["seed_structures"], p["out_path"])
    _protect_dataset(result, p.get("reference_yaml"), require_lineage=True)
    artifact = {"path": str(Path(result).resolve()), "integrity": artifact_digest(result)}
    manifest_path = p.get("manifest_path")
    if manifest_path:
        manifest = {
            "schema_version": 1,
            "operation": "acquire_structures",
            "acquisition_config": str(Path(p["acquisition_config"]).resolve()),
            "teacher_config": str(Path(p["teacher_config"]).resolve()),
            "seed_structures": str(Path(p["seed_structures"]).resolve()),
            "output": artifact["path"],
            "output_integrity": artifact["integrity"],
        }
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
        artifact["manifest_path"] = str(Path(manifest_path).resolve())
        artifact["manifest_integrity"] = artifact_digest(manifest_path)
    return artifact


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
    # --- ML Trainer ---
    _ready("prepare_student_inputs", "ml-trainer", "adapters.student._render_simple_nn_config",
           "student_config,out_dir", "rendered SIMPLE-NN config", de.prepare_student_inputs),
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
    # --- Simulation ---
    _hpc("build_teacher_baseline", "simulation",
         "adapters.acquisition.label_with_teacher + validation.teacher_baseline",
         "teacher_config,structures_path,distillation_scope,validation_profile,report_path",
         "TeacherBaselineReport", "validation.teacher_baseline"),
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
]}


def build_executor_registry() -> dict:
    reg: dict = {}
    for action, b in BINDINGS.items():
        executor = b.fn
        if action == "build_teacher_baseline":
            executor = _exec_build_teacher_baseline
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
