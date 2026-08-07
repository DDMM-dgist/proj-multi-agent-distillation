"""Trusted executors: bind in-scope actions to EXISTING versioned scientific implementations.

No scientific logic is re-implemented here — each safe binding is a thin adapter that reads the
proposal's parameters and calls the existing repository function (adapters/steps/validation).
Actions with no backing implementation are marked NOT_IMPLEMENTED (executor=None) and are never
mocked; costly actions (real Teacher/Student/MD) are bound but flagged AVAILABLE_HPC and are only
run behind approval — never in tests. build_executor_registry() turns these bindings into the
ActionDescriptor map the dispatcher enforces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .actions import APPROVAL_GATED_ACTIONS
from .dispatch import ActionDescriptor

# executor status independent of the capability registry (which covers out-of-scope actions).
ExecStatus = str  # AVAILABLE | AVAILABLE_HPC | NOT_IMPLEMENTED


@dataclass
class ExecutorBinding:
    action_type: str
    role: str
    status: ExecStatus
    backing: str                 # "module.function" of the existing implementation, or ""
    input_contract: str
    output_artifact: str
    validator: str               # existing deterministic validator, or ""
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
    """Serialize a small result and hash it if a path is given (uses the repo digest)."""
    result = {"metrics": obj}
    if out_path:
        from workflow.integrity import sha256_file
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, default=float, indent=2))
        result["path"] = str(p)
        result["sha256"] = sha256_file(p)
    return result


# --- Safe trusted executors (call existing code; zero-cost with synthetic input) ------------

def _exec_compute_nve_drift(proposal) -> dict:
    from validation.structure_dynamics import compute_nve_drift
    p = _params(proposal)
    drift, _ = compute_nve_drift([float(x) for x in p["energies"]],
                                 float(p.get("timestep_fs", 1.0)), int(p["n_atoms"]),
                                 sample_interval_steps=int(p.get("sample_interval_steps", 1)))
    return _artifact({"nve_drift": float(drift), "n_atoms": int(p["n_atoms"])}, p.get("out_path"))


def _exec_committee_disagreement(proposal) -> dict:
    from adapters.uncertainty import committee_force_std
    p = _params(proposal)
    per_atom, frame = committee_force_std(p["forces_per_seed"], aggregate=p.get("aggregate", "max"))
    return _artifact({"u_per_atom": list(map(float, per_atom)), "u_frame": float(frame)},
                     p.get("out_path"))


def _exec_generate_group_split(proposal) -> dict:
    from workflow.steps import split_dataset
    p = _params(proposal)
    manifest = split_dataset(p["dataset"], p["output_dir"], p["manifest"],
                             seed=int(p.get("seed", 2026)),
                             validation_fraction=float(p.get("validation_fraction", 0.1)))
    return {"path": p["manifest"], "manifest": manifest,
            "sha256": manifest.get("integrity", {}).get("sha256", "")}


def _exec_compute_rdf(proposal) -> dict:
    from ase.io import read
    from validation.structure_dynamics import compute_rdf
    p = _params(proposal)
    frames = read(p["frames_path"], index=":")
    distances, partial = compute_rdf(frames, p["elements"], r_max=float(p.get("r_max", 6.0)),
                                     nbins=int(p.get("nbins", 200)))
    peaks = {k: float(max(v)) for k, v in partial.items()}
    return _artifact({"rdf_peaks": peaks, "n_bins": len(distances)}, p.get("out_path"))


def _exec_force_error_channel(proposal) -> dict:
    from ase.io import read
    from validation.four_channel_audit import channel
    p = _params(proposal)
    frames = read(p["frames_path"], index=":")
    metrics = channel(frames, p["ref_prefix"], p["pred_prefix"],
                      per_config_type=bool(p.get("per_config_type", False)))
    return _artifact({"channel": metrics}, p.get("out_path"))


def _exec_compute_coordination(proposal) -> dict:
    from ase.io import read
    from validation.structure_dynamics import compute_coordination
    p = _params(proposal)
    frames = read(p["frames_path"], index=":")
    coord = compute_coordination(frames, p["elements"], p["cutoffs"])
    return _artifact({"coordination": {k: float(v) for k, v in coord.items()}}, p.get("out_path"))


def _exec_compare_coverage(proposal) -> dict:
    from validation.data_coverage import validate_data_coverage_report
    p = _params(proposal)
    report = validate_data_coverage_report(
        p["manifest_path"], required_source_categories=p.get("required_source_categories"))
    return {"path": p["manifest_path"], "report": report,
            "sha256": (report.get("integrity", {}) or {}).get("sha256", "")}


# --- Backing table (audited against the repo; honest statuses) ------------------------------

def _hpc(action, role, backing, contract, out, validator=""):
    return ExecutorBinding(action, role, "AVAILABLE_HPC", backing, contract, out, validator,
                           "hpc", True, fn=None)  # bound conceptually; NOT executed in tests


def _not_impl(action, role, reason):
    return ExecutorBinding(action, role, "NOT_IMPLEMENTED", "", reason, "", "", "light", False,
                           fn=None)


BINDINGS: dict[str, ExecutorBinding] = {b.action_type: b for b in [
    # Data Curator
    _not_impl("inspect_dataset", "data-curator", "no standalone dataset-inspection function"),
    _not_impl("summarize_source_categories", "data-curator",
              "only internal data_coverage._source_statistics; no public producer"),
    _not_impl("sample_seed_pool", "data-curator", "no deterministic seed-pool sampler exists"),
    _not_impl("reconstruct_lineage", "data-curator",
              "validate_lineage only asserts presence; no reconstruction"),
    ExecutorBinding("generate_group_split", "data-curator", "AVAILABLE",
                    "workflow.steps.split_dataset", "dataset,output_dir,manifest",
                    "split manifest (sha256-bound)", "workflow.steps split integrity", "light",
                    False, fn=_exec_generate_group_split),
    _hpc("label_with_teacher", "data-curator", "adapters.acquisition.label_with_teacher",
         "teacher_cfg,structures,out,manifest", "labeled extxyz + labeling manifest",
         "labeling manifest integrity"),
    _not_impl("validate_label_preservation", "data-curator", "no label-preservation validator"),
    _not_impl("build_dataset_manifest", "data-curator",
              "manifests are byproducts of split/merge; no standalone builder"),
    ExecutorBinding("compare_deployment_coverage", "data-curator", "AVAILABLE",
                    "validation.data_coverage.validate_data_coverage_report",
                    "coverage manifest + required categories", "validated coverage report",
                    "data_coverage validator", "light", False, fn=_exec_compare_coverage),
    _not_impl("detect_duplicates", "data-curator",
              "exact dedup only inside steps.merge_datasets; no standalone action"),
    _not_impl("detect_atomic_overlap", "data-curator",
              "no minimum-distance / atomic-overlap function exists"),
    # ML Trainer
    _not_impl("prepare_student_inputs", "ml-trainer",
              "only internal student._render_simple_nn_config; no standalone action"),
    _hpc("train_committee", "ml-trainer", "workflow.steps.train_committee",
         "student_config,dataset,output_dir,manifest", "committee manifest + checkpoints",
         "training-completion (n/a)"),
    _not_impl("collect_checkpoints", "ml-trainer",
              "checkpoints are train_committee output; no standalone collector"),
    _hpc("evaluate_heldout_fidelity", "ml-trainer", "workflow.steps.evaluate_committee",
         "student_config,committee_manifest,frames", "3-channel fidelity report",
         "four_channel_audit"),
    _not_impl("summarize_seed_variation", "ml-trainer",
              "committee_force_std gives per-seed std; no seed-variation summarizer action"),
    ExecutorBinding("compute_committee_disagreement", "ml-trainer", "AVAILABLE",
                    "adapters.uncertainty.committee_force_std", "forces_per_seed[,aggregate]",
                    "committee u_per_atom + u_frame", "", "light", False,
                    fn=_exec_committee_disagreement),
    _not_impl("validate_training_completion", "ml-trainer",
              "no training-completion validator function"),
    # Simulation
    _hpc("run_teacher_md", "simulation", "adapters.acquisition.run_teacher_md",
         "cfg,teacher_cfg,seed,out", "teacher MD snapshots", ""),
    _hpc("run_student_md", "simulation", "workflow.steps.run_md / adapters.md_backend.run",
         "md_cfg,student_cfg,checkpoint,...", "MD trajectory + manifest", ""),
    ExecutorBinding("compute_rdf", "simulation", "AVAILABLE",
                    "validation.structure_dynamics.compute_rdf", "frames_path,elements[,r_max,nbins]",
                    "partial RDF peaks", "", "light", False, fn=_exec_compute_rdf),
    ExecutorBinding("compute_coordination", "simulation", "AVAILABLE",
                    "validation.structure_dynamics.compute_coordination",
                    "frames_path,elements,cutoffs", "mean coordination", "", "light", False,
                    fn=_exec_compute_coordination),
    _not_impl("compute_minimum_distance", "simulation",
              "no minimum-distance function in validation/structure_dynamics"),
    _not_impl("detect_force_spike", "simulation", "no force-spike detector function"),
    ExecutorBinding("compute_nve_drift", "simulation", "AVAILABLE",
                    "validation.structure_dynamics.compute_nve_drift",
                    "energies,timestep_fs,n_atoms", "NVE drift meV/atom/ns", "", "light", False,
                    fn=_exec_compute_nve_drift),
    _not_impl("validate_simulation_completion", "simulation",
              "structure_dynamics CLI emits a report; no standalone completion validator action"),
    _not_impl("submit_scheduler_job", "simulation",
              "typed scheduler bridge interface only; no scheduler backend (approval-gated)"),
    _not_impl("query_scheduler_job", "simulation", "no scheduler backend"),
    _not_impl("collect_scheduler_artifact", "simulation", "no scheduler backend"),
    # Analyst
    ExecutorBinding("compare_force_errors", "analyst", "AVAILABLE",
                    "validation.four_channel_audit.channel", "frames_path,ref_prefix,pred_prefix",
                    "force error channel metrics", "", "light", False, fn=_exec_force_error_channel),
    ExecutorBinding("compare_energy_errors", "analyst", "AVAILABLE",
                    "validation.four_channel_audit.channel", "frames_path,ref_prefix,pred_prefix",
                    "energy error channel metrics", "", "light", False, fn=_exec_force_error_channel),
    ExecutorBinding("summarize_committee_disagreement", "analyst", "AVAILABLE",
                    "adapters.uncertainty.committee_force_std", "forces_per_seed", "u summary", "",
                    "light", False, fn=_exec_committee_disagreement),
    ExecutorBinding("compare_rdf", "analyst", "AVAILABLE",
                    "validation.structure_dynamics.compute_rdf", "frames_path,elements", "RDF peaks",
                    "", "light", False, fn=_exec_compute_rdf),
    ExecutorBinding("compare_coordination", "analyst", "AVAILABLE",
                    "validation.structure_dynamics.compute_coordination",
                    "frames_path,elements,cutoffs", "coordination", "", "light", False,
                    fn=_exec_compute_coordination),
    ExecutorBinding("fit_nve_drift", "analyst", "AVAILABLE",
                    "validation.structure_dynamics.compute_nve_drift", "energies,timestep_fs,n_atoms",
                    "NVE drift fit", "", "light", False, fn=_exec_compute_nve_drift),
    _not_impl("summarize_md_stability", "analyst",
              "composition of nve/msd; no standalone stability-summary action"),
    _not_impl("classify_root_cause", "analyst",
              "root-cause is a reasoning output, not a deterministic executor"),
]}


def build_executor_registry() -> dict:
    """ActionDescriptor map for the dispatcher: AVAILABLE bindings carry their trusted executor;
    AVAILABLE_HPC carry the approval boundary but no inline executor (run only behind approval,
    never in tests); NOT_IMPLEMENTED carry no executor."""
    reg: dict = {}
    for action, b in BINDINGS.items():
        reg[action] = ActionDescriptor(
            action_type=action, role=b.role, cost_class=b.cost_class,
            approval_boundary=APPROVAL_GATED_ACTIONS.get(action),
            executor=b.fn)  # None for HPC/NOT_IMPLEMENTED -> dispatcher yields DRY_RUN, never fakes
    return reg


def executor_status(action: str) -> Optional[str]:
    b = BINDINGS.get(action)
    return b.status if b else None
