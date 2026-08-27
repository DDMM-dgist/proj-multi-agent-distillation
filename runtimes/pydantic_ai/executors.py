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
    aggregate = p.get("aggregate", "max")

    def _frame_disagreement(frames_list):
        scores = []
        values = []
        for index, atoms in enumerate(frames_list):
            per_seed = []
            for seed in seeds:
                field = f"student_forces_seed{seed:02d}"
                if field not in atoms.arrays:
                    raise ValueError(f"population frame {index} is missing committee forces: {field}")
                per_seed.append(np.asarray(atoms.arrays[field], dtype=float))
            _, frame_score = committee_force_std(np.stack(per_seed), aggregate=aggregate)
            frame_id = str(atoms.info.get("structure_id", index))
            scores.append({"frame_id": frame_id, "u_frame": float(frame_score)})
            values.append(float(frame_score))
        return scores, values

    def _force_error_records(frames_list, u_values):
        if len(frames_list) != len(u_values):
            raise ValueError("force-error records require one uncertainty value per frame")
        records = []
        for index, (atoms, u_frame) in enumerate(zip(frames_list, u_values)):
            if "dft_forces" not in atoms.arrays:
                raise ValueError(f"calibration frame {index} is missing dft_forces")
            per_seed = []
            for seed in seeds:
                field = f"student_forces_seed{seed:02d}"
                if field not in atoms.arrays:
                    raise ValueError(f"calibration frame {index} is missing committee forces: {field}")
                per_seed.append(np.asarray(atoms.arrays[field], dtype=float))
            pred = np.mean(np.stack(per_seed), axis=0)
            ref = np.asarray(atoms.arrays["dft_forces"], dtype=float)
            if pred.shape != ref.shape or pred.ndim != 2 or pred.shape[1] != 3:
                raise ValueError(f"calibration frame {index} has incompatible force shapes")
            err = float(np.max(np.abs(pred - ref)))
            records.append({
                "frame_id": str(atoms.info.get("structure_id", index)),
                "u_frame": float(u_frame),
                "actual_force_component_error_eV_per_angstrom": err,
            })
        return records

    def _rank_correlation(xs, ys):
        if len(xs) < 2 or len(xs) != len(ys):
            return None
        def ranks(values):
            order = sorted(range(len(values)), key=lambda i: (values[i], i))
            out = [0.0] * len(values)
            for rank, idx in enumerate(order):
                out[idx] = float(rank)
            return np.asarray(out, dtype=float)
        rx = ranks([float(x) for x in xs])
        ry = ranks([float(y) for y in ys])
        if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
            return None
        return float(np.corrcoef(rx, ry)[0, 1])

    def _load_uncertainty_policy():
        path = p.get("uncertainty_policy") or p.get("uncertainty_policy_path")
        if not path:
            return None, None, None
        policy_path = Path(path).resolve()
        policy = json.loads(policy_path.read_text())
        note = policy.get("_scientific_semantics_note") or {}
        target_percent = note.get("nominal_coverage_target_percent")
        if not isinstance(target_percent, (int, float)):
            raise ValueError("uncertainty policy lacks nominal_coverage_target_percent")
        return policy_path, policy, float(target_percent) / 100.0

    # Governed calibration/eval isolation: when an access-partition contract is bound, the
    # uncertainty report's PRIMARY population is the disjoint ``uncertainty_calibration_fit``
    # role, and the disjoint ``uncertainty_calibration_eval`` role is summarized separately as an
    # independent held-out disagreement check. Both roles are access-enforced and their
    # disjointness is re-verified from the replayed contract -- calibration-fit and
    # calibration-eval can never share a frame.
    governed_partition = None
    access_partition_path = p.get("access_partition_path")
    if access_partition_path:
        from validation.access_partition import (
            ROLE_CALIBRATION_FIT, ROLE_CALIBRATION_EVAL, enforce_and_materialize,
        )
        source_population = Path(p["population_frames"]).resolve()
        report_dir = Path(p["report_path"]).parent
        fit_role = p.get("calibration_fit_role", ROLE_CALIBRATION_FIT)
        eval_role = p.get("calibration_eval_role", ROLE_CALIBRATION_EVAL)
        if fit_role == eval_role:
            raise ValueError("calibration fit and eval roles must be distinct")
        fit = enforce_and_materialize(
            access_partition_path, "uncertainty", fit_role, source_population,
            p.get("calibration_fit_out", report_dir / "uncertainty_calibration_fit.extxyz"),
            require_committee_seeds=seeds,
            expected_reference_id=p.get("expected_reference_id"),
            expected_structures_sha256=p.get("expected_structures_sha256"))
        ev = enforce_and_materialize(
            access_partition_path, "uncertainty", eval_role, source_population,
            p.get("calibration_eval_out", report_dir / "uncertainty_calibration_eval.extxyz"),
            require_committee_seeds=seeds,
            expected_reference_id=p.get("expected_reference_id"),
            expected_structures_sha256=p.get("expected_structures_sha256"))
        if fit["frame_fingerprints_sha256"] == ev["frame_fingerprints_sha256"]:
            raise ValueError("calibration fit and eval slices are identical; partition is not disjoint")
        population_path = Path(fit["path"]).resolve()
        frames = read(str(population_path), index=":")
        eval_frames = read(str(Path(ev["path"]).resolve()), index=":")
        eval_scores, eval_values = _frame_disagreement(eval_frames)
        governed_partition = {
            "access_partition_path": str(Path(access_partition_path).resolve()),
            "partition_assignment_sha256": fit["contract"].get("partition_assignment_sha256"),
            "calibration_fit": {"role": fit_role, "path": fit["path"], "n_frames": fit["n_frames"],
                                "frame_fingerprints_sha256": fit["frame_fingerprints_sha256"]},
            "calibration_eval": {"role": eval_role, "path": ev["path"], "n_frames": ev["n_frames"],
                                 "frame_fingerprints_sha256": ev["frame_fingerprints_sha256"],
                                 "holdout_disagreement_summary": {
                                     "mean": sum(eval_values) / len(eval_values),
                                     "max": max(eval_values)}},
            "fit_eval_disjoint": True,
        }
    else:
        population_path = Path(p["population_frames"]).resolve()
        frames = read(str(population_path), index=":")
    if not frames:
        raise ValueError("uncertainty population_frames is empty")
    frame_scores, u_values = _frame_disagreement(frames)
    policy_path, uncertainty_policy, nominal_coverage = _load_uncertainty_policy()
    calibration_evidence = p.get("calibration_evidence")
    if governed_partition is not None and uncertainty_policy is not None:
        epsilon = float(p.get("conformal_epsilon", 1e-12))
        if epsilon < 0:
            raise ValueError("conformal_epsilon must be non-negative")
        fit_records = _force_error_records(frames, u_values)
        fit_scores = [r["actual_force_component_error_eV_per_angstrom"] / (r["u_frame"] + epsilon)
                      for r in fit_records]
        if not fit_scores:
            raise ValueError("calibration FIT population is empty")
        ordered = sorted(float(x) for x in fit_scores)
        rank = int(np.ceil((len(ordered) + 1) * nominal_coverage))
        qhat = ordered[min(max(rank, 1), len(ordered)) - 1]
        eval_records = _force_error_records(eval_frames, eval_values)
        covered = [r["actual_force_component_error_eV_per_angstrom"] <= qhat * (r["u_frame"] + epsilon)
                   for r in eval_records]
        covered_count = int(sum(bool(x) for x in covered))
        total_count = int(len(covered))
        observed = covered_count / total_count if total_count else 0.0
        coverage_acceptance = p.get("coverage_acceptance")
        if coverage_acceptance:
            min_cov = coverage_acceptance.get("min_observed_coverage")
            if not isinstance(min_cov, (int, float)):
                raise ValueError("coverage_acceptance.min_observed_coverage must be numeric")
            decision = "PASS" if observed >= float(min_cov) else "FAIL"
        else:
            decision = "HUMAN_SCIENTIFIC_INPUT_REQUIRED"
        calibration = {
            "status": "calibrated",
            "quantity": "frame_max_abs_force_component_error_of_committee_mean_vs_dft",
            "uncertainty_signal": f"committee_force_std_sigma_F_{aggregate}",
            "nominal_coverage": nominal_coverage,
            "policy_id": uncertainty_policy.get("policy_id"),
            "fit": governed_partition["calibration_fit"],
            "eval": governed_partition["calibration_eval"],
            "conformal": {
                "estimator": "split_conformal_normalized_force_error_radius",
                "radius_formula": "qhat * (u_frame + epsilon)",
                "qhat": float(qhat),
                "epsilon": float(epsilon),
                "score_count": len(fit_scores),
                "finite_sample_rank": rank,
            },
            "coverage_eval": {
                "observed_coverage": float(observed),
                "covered_count": covered_count,
                "total_count": total_count,
            },
            "association_diagnostics": {
                "fit_spearman_rank_correlation_sigma_vs_error": _rank_correlation(
                    [r["u_frame"] for r in fit_records],
                    [r["actual_force_component_error_eV_per_angstrom"] for r in fit_records]),
                "eval_spearman_rank_correlation_sigma_vs_error": _rank_correlation(
                    [r["u_frame"] for r in eval_records],
                    [r["actual_force_component_error_eV_per_angstrom"] for r in eval_records]),
            },
            "decision": decision,
        }
        if coverage_acceptance:
            calibration["coverage_acceptance"] = coverage_acceptance
        else:
            calibration["human_scientific_input_required"] = (
                "nominal coverage is frozen, but no preregistered binomial/coverage acceptance "
                "test parameters are bound; calibrated evidence is reported without fabricating "
                "a PASS threshold")
    elif calibration_evidence:
        calibration = {"status": "calibrated", "caveat": p.get(
            "calibration_caveat", "calibrated against the cited calibration_evidence")}
    else:
        if p.get("require_calibrated"):
            raise ValueError("calibrated uncertainty requires access_partition_path and uncertainty_policy")
        calibration = {
            "status": "uncalibrated",
            "caveat": ("committee force disagreement (sigma_F) is treated as a committee "
                      "disagreement / fidelity-ranking signal only; no calibration evidence "
                      "(e.g. a held-out DFT-error regression) has been supplied for this run"),
        }
    evidence = [_evidence("committee_manifest", committee_manifest),
               _evidence("population", population_path)]
    if governed_partition is not None:
        evidence.append(_evidence("calibration_eval_population", governed_partition["calibration_eval"]["path"]))
    if policy_path is not None:
        evidence.append(_evidence("uncertainty_policy", policy_path))
    if calibration_evidence:
        evidence.append(_evidence("calibration_evidence", calibration_evidence))
    default_role = (governed_partition["calibration_fit"]["role"] if governed_partition
                    else "held_out_evaluation_population")
    report = {
        "schema_version": 1,
        "population": {"role": p.get("population_role", default_role),
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
    if governed_partition is not None:
        report["governed_partition"] = governed_partition
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
        # Deterministic locked-policy propagation (mirrors data_coverage/teacher_baseline). When
        # this run has a frozen validation contract, the split parameters are sourced VERBATIM from
        # the locked dataset_split_policy.value -- they are never re-authored by a proposal or a
        # later recovery attempt. split_dataset still hash-checks them, so a drift fails closed.
        split_kwargs = {
            "seed": int(p.get("seed", 2026)),
            "validation_fraction": float(p.get("validation_fraction", 0.1)),
            "test_fraction": float(p.get("test_fraction", 0.1)),
            "grouping_key": p.get("grouping_key", "parent_structure_id"),
        }
        split_contract_path = _resolve_validation_contract_path(p, p["manifest"])
        if split_contract_path is not None and split_contract_path.is_file():
            locked_policy = (json.loads(split_contract_path.read_text()).get("components")
                             or {}).get("dataset_split_policy")
            if not isinstance(locked_policy, dict) or "value" not in locked_policy:
                raise ValueError(
                    "validation contract is bound but has no dataset_split_policy.value; "
                    "cannot deterministically source dataset split parameters")
            locked_value = locked_policy["value"]
            for key in split_kwargs:
                if key in locked_value:
                    split_kwargs[key] = locked_value[key]
        manifest = split_dataset(dataset, p["output_dir"], p["manifest"],
                                 allow_unique_parent_fallback=bool(p.get("allow_unique_parent_fallback", False)),
                                 validation_contract_path=(str(split_contract_path)
                                                           if split_contract_path is not None
                                                           and split_contract_path.is_file() else None),
                                 **split_kwargs)
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


def _exec_build_split_membership_population(proposal):
    from workflow.integrity import artifact_digest
    from workflow.steps import build_split_membership_population
    p = _params(proposal)
    result = build_split_membership_population(
        p["source_dataset"], p["split_source_manifest"], p["target_split"],
        p["output_path"], manifest_path=p.get("manifest_path"))
    integrity_path = p.get("manifest_path") or result["structures"]["path"]
    return {"path": result["structures"]["path"], "manifest": result,
            "integrity": artifact_digest(integrity_path)}


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
    if all(stage["gate"] in ("PASS", "NOT_APPLICABLE") for stage in stages):
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


def _load_physical_validation_policy_v2(params):
    """Return the bound PhysicalValidationPolicyV2 dict, or None if the caller
    is not opting into the typed path. Accepts either
    ``physical_validation_policy_v2_dict`` (inline) or
    ``physical_validation_policy_v2_ref`` (JSON path on disk).
    """
    policy = params.get("physical_validation_policy_v2_dict")
    if policy is None and params.get("physical_validation_policy_v2_ref"):
        policy = json.loads(Path(params["physical_validation_policy_v2_ref"]).read_text())
    return policy


def _emit_typed_observable(obs_spec, frames, params, energies_ctx):
    """Given one ObservableSpec-shaped dict, compute the requested typed
    observable and return a check-row (validation.report.make_check-shaped
    dict). Fails-closed with a make_check(reason=...) row when the current
    implementation cannot compute a specific ObservableSpec (no silent
    fallback).
    """
    from validation.structure_dynamics import (
        compute_rdf_v2, rdf_first_peak_and_minimum,
        compute_species_coordination, compute_density,
    )
    from validation.report import make_check
    kind = obs_spec.get("kind")
    name = obs_spec.get("name")
    units = obs_spec.get("units")
    role = obs_spec.get("role")
    thresholded = role == "thresholded"

    def _threshold_from_spec():
        # THRESHOLDED observables are expected to carry a numerical threshold
        # via the policy's own criterion binding at gate time; the executor
        # itself does not synthesize thresholds. For nve_drift, use the
        # existing param binding.
        return None

    if kind == "rdf_peak_position":
        center = obs_spec.get("center_species")
        neighbor = obs_spec.get("neighbor_species")
        r_max = float(obs_spec.get("_r_max_A", 6.0))
        nbins = int(obs_spec.get("_nbins", 200))
        rdf = compute_rdf_v2(frames, center, neighbor, r_max=r_max, nbins=nbins)
        peakmin = rdf_first_peak_and_minimum(
            rdf["r_A"], rdf["g_of_r"],
            smoothing_window=int(obs_spec.get("_smoothing_window", 5)))
        # Decide which position this observable requests: peak or first-min.
        method = obs_spec.get("computation_method", "")
        if "min" in method:
            value = peakmin["r_first_min_A"]
        else:
            value = peakmin["r_first_peak_A"]
        return make_check("structure", name, value=float(value), unit=units,
                          criterion=_threshold_from_spec() if thresholded else None,
                          details={"peakmin": peakmin,
                                   "bin_width_A": rdf["bin_width_A"],
                                   "r_max_A": rdf["r_max_A"], "nbins": rdf["nbins"]})
    if kind == "rdf_peak_height":
        center = obs_spec.get("center_species")
        neighbor = obs_spec.get("neighbor_species")
        r_max = float(obs_spec.get("_r_max_A", 6.0))
        nbins = int(obs_spec.get("_nbins", 200))
        rdf = compute_rdf_v2(frames, center, neighbor, r_max=r_max, nbins=nbins)
        peakmin = rdf_first_peak_and_minimum(
            rdf["r_A"], rdf["g_of_r"],
            smoothing_window=int(obs_spec.get("_smoothing_window", 5)))
        return make_check("structure", name, value=float(peakmin["g_first_peak"]),
                          unit=units, criterion=None,
                          details={"g_first_peak_raw": peakmin["g_first_peak_raw"],
                                   "r_A": rdf["r_A"], "bin_width_A": rdf["bin_width_A"]})
    if kind == "species_coordination":
        center = obs_spec.get("center_species")
        neighbor = obs_spec.get("neighbor_species")
        cutoff_A = obs_spec.get("_cutoff_A")
        if cutoff_A is None:
            # Fail-closed: cutoff must have been resolved from the policy's
            # cutoff_source_ref before this executor was invoked (the policy
            # already asserted cutoff_frozen_before_student=True).
            return make_check("structure", name, value=None, unit=units,
                              criterion=None,
                              reason=("cutoff_A not resolved from cutoff_source_ref; "
                                       "policy must supply a numeric cutoff in "
                                       "obs_spec._cutoff_A before executor invocation"))
        method = obs_spec.get("computation_method", "")
        cc = compute_species_coordination(frames, center, neighbor, float(cutoff_A),
                                          cutoff_source_ref=obs_spec.get("cutoff_source_ref"),
                                          cutoff_frozen_before_student=obs_spec.get("cutoff_frozen_before_student"))
        # Emit either aggregate mean OR the requested topology-fraction target.
        if "fraction_of_" in method and "_coord_eq_" in method:
            # e.g. "fraction_of_si_atoms_with_coord_eq_4"
            try:
                target_cn = int(method.split("_coord_eq_")[-1])
            except ValueError:
                return make_check("structure", name, value=None, unit=units,
                                  criterion=None,
                                  reason=f"cannot parse target coordination from method {method!r}")
            frac = cc["coordination_fractions"].get(target_cn, 0.0)
            return make_check("structure", name, value=float(frac), unit=units,
                              criterion=None, details=cc)
        return make_check("structure", name,
                          value=float(cc["aggregate_mean_coordination"]),
                          unit=units, criterion=None, details=cc)
    if kind == "density":
        from validation.structure_dynamics import compute_density
        mean_rho, std_rho = compute_density(frames)
        return make_check("structure", name, value=float(mean_rho), unit=units,
                          criterion=None,
                          details={"standard_deviation": float(std_rho),
                                   "ensemble_applicability": obs_spec.get("ensemble_applicability"),
                                   "ensemble_interpretation":
                                       ("inherited_state_variable"
                                        if "NVT" in (obs_spec.get("ensemble_applicability") or [])
                                        and "NPT" not in (obs_spec.get("ensemble_applicability") or [])
                                        else "variable_cell_or_diagnostic")})
    if kind == "nve_drift":
        # Delegate to the existing energy_log_path / energies branch by
        # returning a sentinel that tells the caller to consume that path.
        return "__NVE_DRIFT_HANDLED_BY_EXISTING_ENERGY_LOG_BRANCH__"
    # Unknown kind: fail-closed
    return make_check("structure", name, value=None, unit=units,
                      criterion=None,
                      reason=f"executor cannot compute observable of kind {kind!r}")


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
    pv_policy_v2 = _load_physical_validation_policy_v2(p)
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

    # TYPED-OBSERVABLE PATH when PhysicalValidationPolicyV2 is bound.
    if pv_policy_v2 is not None:
        for obs in (pv_policy_v2.get("observables") or []):
            row = _emit_typed_observable(obs, frames, p, energies_ctx=None)
            if row == "__NVE_DRIFT_HANDLED_BY_EXISTING_ENERGY_LOG_BRANCH__":
                continue  # NVE drift is emitted below by the existing branch
            if isinstance(row, dict):
                checks.append(row)
    else:
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

    # NVE drift observable. Two accepted param shapes:
    #   (a) inline energies list bound directly in params (legacy path); or
    #   (b) energy_log_path pointing at an authoritative NVE energy log written
    #       by a preceding physical_validation_nve LAMMPS run — read via
    #       validation.structure_dynamics.read_energy_log so the executor never
    #       re-implements the log parser or reinterprets sampling.
    nve_energy_log_evidence = None
    # Stage-11 automatic consumption: when a dedicated NVE-segment MD manifest is bound, resolve
    # its authoritative energy log from the manifest's own evidence (role ``nve_energy_log``)
    # rather than requiring the log path to be hand-wired. This keeps the NVE energy-conservation
    # metric sourced from the SEPARATE microcanonical segment, never the NVT production run.
    if "energy_log_path" not in p and p.get("nve_md_manifest"):
        nve_manifest = json.loads(Path(p["nve_md_manifest"]).read_text())
        log_entry = next((e for e in (nve_manifest.get("evidence") or [])
                          if e.get("role") == "nve_energy_log"), None)
        if log_entry is None:
            raise ValueError(
                "nve_md_manifest carries no evidence entry with role 'nve_energy_log'; the NVE "
                "segment run did not record its energy log")
        from workflow.integrity import verify_artifact
        verify_artifact(log_entry["path"], log_entry.get("integrity", {}))
        p = {**p, "energy_log_path": log_entry["path"]}
    if "energies" in p:
        energies_for_drift = [float(x) for x in p["energies"]]
        drift_steps = None
        nve_sample_interval = int(p.get("sample_interval_steps", 1))
    elif "energy_log_path" in p:
        from validation.structure_dynamics import read_energy_log as _read_energy_log
        energy_log_path = Path(p["energy_log_path"]).resolve()
        step_arr, energy_arr = _read_energy_log(energy_log_path)
        if len(step_arr) < 2:
            raise ValueError("physical_validation energy_log_path yielded fewer than 2 samples")
        energies_for_drift = [float(x) for x in energy_arr]
        drift_steps = [int(x) for x in step_arr]
        # sample_interval_steps in compute_nve_drift is used only when the caller
        # does not supply `steps`; when we pass real steps, we still pass the
        # nominal interval for reporting.
        nve_sample_interval = int(p.get("sample_interval_steps",
                                        int(step_arr[1] - step_arr[0]) if len(step_arr) >= 2 else 1))
        nve_energy_log_evidence = _evidence("nve_energy_log", energy_log_path)
    else:
        energies_for_drift = None
    if energies_for_drift is not None:
        n_atoms_for_drift = int(p.get("n_atoms") or len(frames[0]))
        timestep_fs = float(p.get("timestep_fs", 1.0))
        drift_kwargs = {"sample_interval_steps": nve_sample_interval}
        if drift_steps is not None:
            drift_kwargs["steps"] = drift_steps
        # compute_nve_drift returns slope in meV/atom/ns (see t_ns = steps*fs*1e-6).
        # The pre-registered validation_profile threshold is declared in
        # meV/atom/ps. Convert to meV/atom/ps here so the criterion evaluator
        # (validation.report.criterion_passes) compares like-unit to like-unit;
        # the threshold value 1.0 is not modified.
        drift_ns, resid_ns = compute_nve_drift(energies_for_drift, timestep_fs, n_atoms_for_drift,
                                               **drift_kwargs)
        drift_ps = float(drift_ns) / 1000.0
        emit("nve_drift", "dynamics", drift_ps, "meV/atom/ps", details={
            "residual_std_meV_per_atom": float(resid_ns),
            "n_atoms": n_atoms_for_drift,
            "timestep_fs": timestep_fs,
            "sample_interval_steps": nve_sample_interval,
            "n_samples": len(energies_for_drift),
            "signed_slope_meV_per_atom_per_ps": drift_ps,
            "raw_slope_meV_per_atom_per_ns": float(drift_ns),
        })

    declared_required = {name for name, cfg in checks_cfg.items() if cfg.get("required")}
    produced = {c["observable"] for c in checks}
    missing = declared_required - produced
    if missing:
        raise ValueError("physical_validation did not produce required observables: " +
                         ", ".join(sorted(missing)))

    evidence = [_evidence("validation_profile", profile_path), _evidence("frames", frames_path)]
    if nve_energy_log_evidence is not None:
        evidence.append(nve_energy_log_evidence)
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


def _resolve_validation_contract_path(p, report_path):
    """Locate the run's frozen validation contract deterministically.

    Prefers an explicitly wired ``validation_contract_path`` param; otherwise derives the
    canonical ``{run_dir}/validation_contract.json`` from ``run_dir`` (if provided) or from the
    report path's grandparent (``{run_dir}/artifacts/<report>.json``). Returns a resolved Path
    (which may not exist -- callers check ``.is_file()``) or None when no location is derivable.
    This reads the run's OWN authoritative contract; it is not a fallback to a guessed domain.
    """
    contract_value = p.get("validation_contract_path")
    if contract_value:
        candidate = Path(contract_value).expanduser()
        return candidate.resolve()
    run_dir = p.get("run_dir")
    if run_dir:
        return (Path(run_dir).expanduser() / "validation_contract.json").resolve()
    resolved_report = Path(report_path).expanduser().resolve()
    if resolved_report.parent.name == "artifacts":
        return (resolved_report.parent.parent / "validation_contract.json").resolve()
    return None


def _discover_acquisition_plan(explicit_path, acquisition_manifest_path):
    """Locate the run's single bound AcquisitionPlan so its AUTHORITATIVE FE-037 exclusion report
    can be surfaced into Stage 4 (never recomputed here). Prefers an explicit param; otherwise the
    canonical ``{run_dir}/acquisition/plans/*.acquisition_plan.json`` (run_dir = the manifest's
    grandparent, since the manifest lives under ``{run_dir}/artifacts/``). Returns the parsed plan
    or ``None`` when no plan is bound (an unprotected / plan-less run)."""
    if explicit_path:
        candidate = Path(explicit_path).resolve()
        return json.loads(candidate.read_text()) if candidate.is_file() else None
    run_dir = Path(acquisition_manifest_path).resolve().parent.parent
    plans = sorted((run_dir / "acquisition" / "plans").glob("*.acquisition_plan.json"))
    if len(plans) == 1:
        return json.loads(plans[0].read_text())
    return None


def _resolve_frozen_structure_class_label_map(p, deployment_domain):
    """Resolve the run's FROZEN, human-authored config_type->canonical structure-class ``label_map``
    for the FE-039 gap-based Stage-4 occupancy gate, or ``None`` when none is deterministically bound
    (per-class support then stays honestly NOT_ASSESSABLE -- never fabricated).

    Two provenance-faithful sources, in priority: (1) the locked deployment domain may carry the map
    inline as ``structure_class_label_map`` (a fresh contract that embeds its own scope classification);
    (2) a bound frozen ``DeploymentScopeContractV2`` JSON referenced by ``scope_classification_evidence_path``
    (the same ``{raw_label, canonical_domain, claim_role}`` label_map ``acquisition_readiness`` resolves).
    This function INVENTS no mapping: it only reads a frozen artifact's own entries."""
    inline = (deployment_domain.get("structure_class_label_map")
              if isinstance(deployment_domain, dict) else None)
    if isinstance(inline, list) and inline:
        return inline
    evidence_path = p.get("scope_classification_evidence_path")
    if evidence_path:
        scope_path = Path(evidence_path)
        if not scope_path.is_absolute():
            scope_path = Path(str(p.get("project_dir") or Path.cwd())) / scope_path
        scope_path = scope_path.resolve()
        if scope_path.is_file():
            scope_doc = json.loads(scope_path.read_text())
            label_map = scope_doc.get("label_map")
            if isinstance(label_map, list) and label_map:
                return label_map
    return None


def _coverage_assessment_block(*, p, counts, deployment_domain, acquisition_manifest,
                               acquisition, candidate_elements, n_candidate_frames,
                               teacher_training_data_access, limitations):
    """Build the typed FE-038 ``coverage_assessment`` block: per-declared-dimension PASS/FAIL/
    NOT_ASSESSABLE records under the never-fabricate invariants, an explicit acquisition-lineage
    equality result, and FE-037 protected-reference exclusion provenance sourced through the ONE
    canonical resolver + the existing AcquisitionPlan exclusion report (no duplicated protected-set
    interpretation)."""
    from validation.coverage_assessment import (build_coverage_assessment, make_dimension,
                                                validate_coverage_assessment)
    from validation.protected_reference import (resolve_protected_population,
                                                assert_source_indices_allowed)
    from workflow.integrity import sha256_file

    requirement = (deployment_domain.get("coverage_requirement")
                   if isinstance(deployment_domain, dict) else None)
    min_by_ct = (requirement.get("min_frames_by_config_type")
                 if isinstance(requirement, dict) else None)

    dimensions = []
    # config_type_coverage: assessable ONLY when a frozen per-config_type minimum exists.
    if isinstance(min_by_ct, dict) and min_by_ct:
        unmet = {ct: {"required_min_frames": m, "observed_frames": counts.get(ct, 0)}
                 for ct, m in min_by_ct.items()
                 if not (counts.get(ct, 0) >= m)}
        criterion = {"min_frames_by_config_type": min_by_ct, "unmet": unmet, "met": not unmet}
        dimensions.append(make_dimension(
            dimension_id="config_type_coverage",
            declared_target={"config_types": sorted(min_by_ct)},
            metric="frame_count_by_config_type", criterion_provenance="frozen_deployment_domain",
            criterion=criterion, observed_support={"counts": counts},
            reason=("all frozen per-config_type minimums met" if not unmet
                    else f"config_types below frozen minimum: {sorted(unmet)}")))
    else:
        dimensions.append(make_dimension(
            dimension_id="config_type_coverage",
            declared_target={"config_types": sorted(counts)},
            metric="frame_count_by_config_type", criterion_provenance="absent",
            observed_support={"counts": counts},
            reason=("no frozen coverage_requirement.min_frames_by_config_type in the locked "
                    "deployment domain; per-config_type frame counts are surfaced but adequacy is "
                    "not assessable without an evaluable criterion")))
    # One record per DECLARED deployment structure class (FE-039 gap-based Stage-4 gate). When the
    # run carries the FROZEN, human-authored config_type->canonical structure-class label_map, each
    # declared class gets a definitional occupancy PRESENCE criterion (provenance
    # frozen_deployment_domain): PASS iff >=1 acquired frame maps to it, FAIL (zero-occupancy =>
    # structurally UNSUPPORTED) otherwise. This is presence/absence, NOT an invented min-N or quota.
    # Absent a frozen label_map, per-class support is honestly NOT_ASSESSABLE (unchanged behavior) --
    # never a fabricated PASS, never a false insufficiency.
    from validation.coverage_gap_assessment import build_structure_class_dimensions
    declared_structure_classes = (deployment_domain.get("structure_classes") or []
                                  if isinstance(deployment_domain, dict) else [])
    structure_class_label_map = _resolve_frozen_structure_class_label_map(p, deployment_domain)
    dimensions.extend(build_structure_class_dimensions(
        declared_structure_classes, counts, structure_class_label_map))

    manifest_sha = sha256_file(Path(acquisition_manifest))
    lineage = {
        "acquisition_manifest_path": str(acquisition_manifest),
        "acquisition_manifest_sha256": manifest_sha,
        "expected_identity": manifest_sha,
        "observed_identity": manifest_sha,
        "equality_result": "PASS",
        "checks": {
            "candidate_elements_subset_of_manifest": sorted(candidate_elements),
            "manifest_elements": sorted(acquisition.get("elements") or []),
            "candidate_frames": n_candidate_frames,
            "manifest_n_frames": acquisition.get("n_frames"),
            "frame_count_within_manifest": n_candidate_frames <= acquisition.get("n_frames", 0),
        },
    }

    reference_yaml = p.get("reference_yaml")
    plan = _discover_acquisition_plan(p.get("acquisition_plan_path"), acquisition_manifest)
    plan_report = (plan or {}).get("protected_reference_exclusion_report") or {}
    if reference_yaml:
        resolved = resolve_protected_population(reference_yaml)
        reference_id = resolved["reference_id"]
        protected_candidate_count = int(resolved["protected_source_rows"])
        protected_indices = resolved["protected_source_indices"]
        # cross-check the plan's own exclusion report names the SAME run-bound reference.
        if plan_report and plan_report.get("reference_id") not in (None, reference_id):
            raise ValueError(
                "acquisition plan protected_reference_exclusion_report names reference_id "
                f"{plan_report.get('reference_id')!r} but the run-bound reference resolves to "
                f"{reference_id!r} -- refusing to surface a mismatched protection provenance")
        selected = ((plan or {}).get("selected_source_global_indices")
                    or p.get("selected_source_indices") or [])
        overlap = sorted(set(int(x) for x in selected) & set(int(x) for x in protected_indices))
        assert_source_indices_allowed(selected, protected_indices)  # defense in depth
        protected_excluded_count = plan_report.get("protected_excluded_count")
        eligible_after = plan_report.get("eligible_population_after_exclusion")
    else:
        reference_id = "no_protected_reference"
        protected_candidate_count = 0
        protected_excluded_count = 0
        eligible_after = 0
        overlap = []

    protection = {
        "reference_id": reference_id,
        "protected_candidate_count": protected_candidate_count,
        "protected_excluded_count": (protected_excluded_count
                                     if isinstance(protected_excluded_count, int) else 0),
        "eligible_population_after_exclusion": (eligible_after
                                                if isinstance(eligible_after, int) else 0),
        "post_selection_overlap_count": len(overlap),
        "result": "PASS" if not overlap else "FAIL",
        "provenance": ("acquisition_plan_exclusion_report+canonical_resolver" if plan_report
                       else ("canonical_resolver_only" if reference_yaml else "unprotected_run")),
    }
    if reference_yaml and not plan_report:
        limitations = list(limitations) + [
            "Pool-specific protected_excluded_count / eligible_population_after_exclusion were not "
            "surfaced (no AcquisitionPlan exclusion report bound to data_coverage); protected "
            "candidate count and post-selection overlap are canonical-resolver-derived."]

    assessment = build_coverage_assessment(
        teacher_training_data_access=teacher_training_data_access,
        teacher_access_limitations=limitations, dimensions=dimensions,
        acquisition_lineage=lineage, protected_reference_exclusion=protection)
    validate_coverage_assessment(assessment)
    return assessment, limitations


def _resolve_teacher_training_data_access(p, dataset_policy_path):
    """FE-041: resolve the typed ``teacher_training_data_access`` from the AUTHORITATIVE frozen,
    run-bound scientific input(s) -- the bound ``dataset_policy`` and, when a path is bound, the
    ``distillation_scope`` -- so the truthful Teacher-data provenance (e.g.
    ``representative_geometry_only``) is preserved end-to-end and never re-authored by an LLM
    proposal or collapsed to a coarser mode.

    Contract: every frozen source that declares the field must agree exactly; distinct values fail
    closed (``TEACHER_ACCESS_CONFLICT``) rather than guessing. When NO frozen source declares it,
    fall back to the historical proposal-parameter path (preserving pre-FE-041 behavior for callers
    that pass it directly), and finally to the historical default ``representative``. Returns
    ``(mode, provenance)`` where provenance records each source path + sha256 + the resolved value.
    """
    import yaml
    from workflow.integrity import sha256_file
    candidate_paths = []
    if dataset_policy_path:
        candidate_paths.append(("dataset_policy", Path(dataset_policy_path)))
    scope_path = p.get("distillation_scope")
    if scope_path:
        candidate_paths.append(("distillation_scope", Path(scope_path)))
    declarations = []
    for role, path in candidate_paths:
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text())
        value = doc.get("teacher_training_data_access") if isinstance(doc, dict) else None
        if isinstance(value, str) and value.strip():
            declarations.append({"source_role": role, "source_path": str(path.resolve()),
                                 "source_sha256": sha256_file(path), "value": value})
    if declarations:
        distinct = sorted({d["value"] for d in declarations})
        if len(distinct) > 1:
            raise ValueError(
                "TEACHER_ACCESS_CONFLICT: authoritative frozen inputs declare conflicting "
                f"teacher_training_data_access values {distinct}; refusing to guess -- reconcile "
                "the frozen inputs before re-running data_coverage. Declarations: "
                + json.dumps(declarations, sort_keys=True))
        return distinct[0], {"resolved_from": "frozen_authoritative_input",
                             "resolved_value": distinct[0], "declarations": declarations}
    param = p.get("teacher_training_data_access")
    if isinstance(param, str) and param.strip():
        return param, {"resolved_from": "proposal_parameter", "resolved_value": param,
                       "declarations": []}
    return "representative", {"resolved_from": "historical_default",
                             "resolved_value": "representative", "declarations": []}


def _exec_build_data_coverage_report(proposal):
    """Composite data-curator driver for the ``data_coverage`` production stage: enforces
    protected-reference exclusion (``_protect_dataset``) and acquisition-lineage consistency
    against the registered ``acquisition_manifest`` BEFORE computing real, per-config_type frame
    counts, then self-validates via
    ``validation.data_coverage.validate_data_coverage_report``. No LLM invents a coverage metric,
    threshold, or acquisition count here; unassessable access is reported NOT_ASSESSABLE, never
    filled in. It also emits the typed FE-038 ``coverage_assessment`` block (per-dimension
    SUFFICIENT/INSUFFICIENT/NOT_ASSESSABLE, explicit lineage-equality, FE-037 protection provenance).
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

    # FE-041: the truthful Teacher-data access mode is sourced from the AUTHORITATIVE frozen,
    # run-bound input(s) (bound dataset_policy / distillation_scope), never re-authored by the LLM
    # proposal and never collapsed to a coarser mode; conflicting frozen declarations fail closed.
    teacher_training_data_access, teacher_access_provenance = (
        _resolve_teacher_training_data_access(p, dataset_policy))
    # A proposal may declare ONLY a conservative status (PARTIAL / NOT_ASSESSABLE); it can
    # never self-assert COMPLETE. Adequacy (COMPLETE) is earned below, deterministically,
    # solely by satisfying a FROZEN coverage_requirement -- no LLM fabricates completeness.
    proposed_status = p.get("coverage_status")
    if proposed_status is not None and proposed_status not in ("PARTIAL", "NOT_ASSESSABLE"):
        raise ValueError(
            f"coverage_status may not be self-asserted as {proposed_status!r}; COMPLETE is "
            "earned only by deterministically satisfying a frozen coverage_requirement")
    identified_gaps = list(p.get("identified_gaps") or (
        [] if teacher_training_data_access == "full" else
        ["Teacher training distribution is not independently re-verified in this run"]))
    limitations = list(p.get("limitations") or
                       ["Coverage computed per config_type frame counts only; no density-manifold "
                        "coverage metric is computed"])

    evidence = [_evidence("dataset_policy", dataset_policy),
               _evidence(evidence_role, candidate_dataset),
               _evidence("acquisition_manifest", acquisition_manifest)]
    report_path = Path(p["report_path"])

    # Deterministic locked-domain propagation. When this run has a frozen validation
    # contract, data_coverage.deployment_domain is sourced VERBATIM from the locked
    # teacher_applicability_domain.value -- it is never re-authored by a proposal or by
    # this run's workflow.yaml. This executor is the single deterministic producer of the
    # field, so the strict contract hash-check in validate_data_coverage_report matches by
    # construction. No scientific domain is invented here; a frozen value is copied.
    contract_path = _resolve_validation_contract_path(p, report_path)
    locked_deployment_domain = None
    if contract_path is not None and contract_path.is_file():
        contract_doc = json.loads(contract_path.read_text())
        component = (contract_doc.get("components") or {}).get("teacher_applicability_domain")
        if not isinstance(component, dict) or "value" not in component:
            raise ValueError(
                "validation contract is bound but has no teacher_applicability_domain.value; "
                "cannot deterministically source data_coverage.deployment_domain"
            )
        locked_deployment_domain = component["value"]
    if locked_deployment_domain is not None:
        deployment_domain = locked_deployment_domain
    else:
        deployment_domain = p.get("deployment_domain") or {"structure_classes": ["default"]}

    # Deterministic coverage adequacy. COMPLETE is earned ONLY by satisfying a FROZEN
    # coverage_requirement carried by the (locked) deployment_domain -- a per-config_type
    # minimum-frame map. The threshold is frozen scientific scope; this code merely evaluates
    # the real per-config_type counts against it. Absent a frozen requirement, completeness is
    # not assessable, so the status stays fail-closed: NOT_ASSESSABLE when the Teacher training
    # distribution is unavailable, otherwise a proposal-declared conservative status or PARTIAL.
    coverage_requirement = (deployment_domain.get("coverage_requirement")
                            if isinstance(deployment_domain, dict) else None)
    if teacher_training_data_access == "unavailable":
        coverage_status = "NOT_ASSESSABLE"
    elif isinstance(coverage_requirement, dict) and coverage_requirement.get("min_frames_by_config_type") is not None:
        required = coverage_requirement["min_frames_by_config_type"]
        if not isinstance(required, dict) or not required:
            raise ValueError(
                "coverage_requirement.min_frames_by_config_type must be a non-empty mapping")
        unmet = []
        for config_type, minimum in required.items():
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
                raise ValueError(
                    f"coverage_requirement minimum for {config_type!r} must be a "
                    "non-negative integer")
            have = counts.get(config_type, 0)
            if have < minimum:
                unmet.append(
                    f"config_type {config_type!r} has {have} frames; frozen coverage_requirement "
                    f"needs >= {minimum}")
        if unmet:
            coverage_status = "PARTIAL"
            identified_gaps = list(identified_gaps) + unmet
        else:
            coverage_status = "COMPLETE"
    else:
        coverage_status = proposed_status or "PARTIAL"

    coverage_assessment, limitations = _coverage_assessment_block(
        p=p, counts=counts, deployment_domain=deployment_domain,
        acquisition_manifest=acquisition_manifest, acquisition=acquisition,
        candidate_elements=candidate_elements, n_candidate_frames=len(frames),
        teacher_training_data_access=teacher_training_data_access, limitations=limitations)

    # FE-039 Stage-4 gap gate -> recovery routing. When the typed occupancy gate finds a declared
    # deployment structure class with ZERO acquired representatives, the coverage is structurally
    # INSUFFICIENT: name each UNSUPPORTED class as a concrete identified gap (so a recovery RootCause
    # names exactly the classes the gate found unsupported, never a re-derived guess) and never let the
    # legacy coverage_status assert COMPLETE over an unsupported region. This adds no quota and invents
    # no frame count; it only reports the definitional zero-occupancy absences the gate already found.
    from validation.coverage_gap_assessment import unsupported_structure_classes
    structurally_unsupported = unsupported_structure_classes(coverage_assessment.get("dimensions") or [])
    if coverage_assessment.get("assessment_status") == "COVERAGE_INSUFFICIENT" and structurally_unsupported:
        if coverage_status == "COMPLETE":
            coverage_status = "PARTIAL"
        identified_gaps = list(identified_gaps) + [
            f"declared deployment structure class {sc!r} has zero acquired representatives "
            "(structurally UNSUPPORTED; targeted reacquisition required)"
            for sc in structurally_unsupported]

    report = {
        "schema_version": 1,
        "teacher_training_data_access": teacher_training_data_access,
        "teacher_training_data_access_provenance": teacher_access_provenance,
        "coverage_status": coverage_status,
        "coverage_assessment": coverage_assessment,
        "deployment_domain": deployment_domain,
        "dataset_sources": [source],
        "coverage_dimensions": dimensions,
        "replay_policy": p.get("replay_policy") or {"enabled": False},
        "identified_gaps": identified_gaps, "limitations": limitations,
        "dataset_policy": dataset_policy, "evidence": evidence,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    # Self-validate against the SAME locked contract the controller enforces, so a domain or
    # split-policy drift fails fast in-process rather than only at the external contract gate.
    validate_data_coverage_report(
        report_path,
        validation_contract_path=(str(contract_path) if contract_path is not None
                                  and contract_path.is_file() else None),
    )
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
        from validation.protected_reference import (
            assert_source_indices_allowed, resolve_protected_population)
        protection = resolve_protected_population(reference_yaml)
        assert_source_indices_allowed(selected, protection["protected_source_indices"])
        if protection["protected_source_indices"]:
            report_reference_id = report.get("reference_id")
            if not report_reference_id:
                raise _AcquisitionPlanError(
                    "AcquisitionPlan protection report omits reference_id while the run binds a "
                    "protected reference; an anonymous exclusion report cannot be authoritative")
            if report_reference_id != protection["reference_id"]:
                raise _AcquisitionPlanError(
                    "AcquisitionPlan protection report reference_id does not match run-bound reference")
    source_records = []
    if seed_structures:
        source_records = _validate_selected_sources(plan, seed_structures=seed_structures,
                                                   reference_yaml=reference_yaml)
    return {**plan, "selected_source_global_indices": selected,
            "selected_parent_structure_ids": parents, "eligible_source_categories": categories,
            "n_parents": n_parents, "n_per_structure": n_per, "expected_output_count": expected,
            "_selected_source_records": _public_source_records(source_records)}


# --------------------------------------------------------------------------------------------
# FE-028 -- EXISTING_POOL_SELECTION execution (SELECT an existing subset, do NOT generate frames)
# --------------------------------------------------------------------------------------------
_REQUIRED_EXISTING_POOL_PLAN_FIELDS = {
    "schema_version", "pool_path", "eligible_source_categories",
    "selected_parent_structure_ids", "selected_source_global_indices", "n_selected",
    "expected_output_count", "duplicate_handling", "labeling_population_sizing",
    "protected_reference_exclusion_report",
}


def _is_existing_pool_plan(plan) -> bool:
    """An acquisition plan is the existing-pool projection iff it carries a ``pool_path`` (the
    perturbation legacy projection never does). This is how the single ``acquire_structures`` stage
    executor discriminates the two projections without a new action_type (the workflow is authored
    before the plan is autonomously designed)."""
    return isinstance(plan, dict) and bool(plan.get("pool_path"))


def _validate_existing_pool_plan(plan, *, reference_yaml=None,
                                 proposal_selected_source_indices=None):
    """Deterministically validate the EXISTING_POOL_SELECTION projection before execution.

    Mirrors the fail-closed rigor of ``_validate_acquisition_plan`` for the perturbation path, but
    for a selection (not generation) plan: unique non-negative global indices, equal-length parent
    ids, counts derived from the deterministic labeling-population sizing, and a PASS
    protected-reference exclusion report with DFT labels never used as selection scores."""
    missing = sorted(_REQUIRED_EXISTING_POOL_PLAN_FIELDS - set(plan))
    if missing:
        raise _AcquisitionPlanError(
            "existing-pool AcquisitionPlan missing required fields: " + ", ".join(missing))
    if plan.get("schema_version") != 1:
        raise _AcquisitionPlanError("existing-pool AcquisitionPlan requires schema_version=1")
    parents = _nonempty_list(plan, "selected_parent_structure_ids")
    selected = _nonempty_list(plan, "selected_source_global_indices")
    categories = _nonempty_list(plan, "eligible_source_categories")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in selected):
        raise _AcquisitionPlanError(
            "existing-pool selected_source_global_indices must be non-negative integers")
    if len(set(selected)) != len(selected):
        raise _AcquisitionPlanError(
            "existing-pool selected_source_global_indices must be unique")
    if proposal_selected_source_indices is not None and list(proposal_selected_source_indices) != selected:
        raise _AcquisitionPlanError(
            "proposal selected_source_indices must exactly match existing-pool "
            "selected_source_global_indices")
    n_selected = int(plan.get("n_selected"))
    expected = int(plan.get("expected_output_count"))
    if n_selected <= 0:
        raise _AcquisitionPlanError("existing-pool n_selected must be positive")
    if n_selected != len(selected) or n_selected != len(parents):
        raise _AcquisitionPlanError(
            "existing-pool n_selected must match selected index/parent counts")
    if expected != n_selected:
        raise _AcquisitionPlanError(
            "existing-pool expected_output_count must equal n_selected")
    if not all(isinstance(x, str) and x.strip() for x in parents + categories):
        raise _AcquisitionPlanError(
            "existing-pool parent/category values must be non-empty strings")
    sizing = plan.get("labeling_population_sizing")
    if not isinstance(sizing, dict) or not sizing:
        raise _AcquisitionPlanError(
            "existing-pool plan missing labeling_population_sizing evidence")
    rec = sizing.get("recommended_population_size")
    if rec is not None and int(rec) != n_selected:
        raise _AcquisitionPlanError(
            "existing-pool sizing recommended_population_size must equal n_selected")
    report = plan.get("protected_reference_exclusion_report")
    if not isinstance(report, dict) or report.get("status") != "PASS":
        raise _AcquisitionPlanError(
            "existing-pool protected_reference_exclusion_report must be a PASS record")
    if report.get("dft_labels_used_as_selection_scores") is not False:
        raise _AcquisitionPlanError(
            "existing-pool plan must record DFT labels were not used as selection scores")
    if reference_yaml:
        # Defense in depth: resolve the protected population through the ONE canonical resolver the
        # autonomous planner also consumes, then INDEPENDENTLY re-verify disjointness of the selected
        # rows here (this guard is what fail-closed on the ffv4o leak). Keep this even though the
        # planner now excludes up front -- the two must agree, and this proves it at bind time.
        from validation.protected_reference import (
            assert_source_indices_allowed, resolve_protected_population)
        protection = resolve_protected_population(reference_yaml)
        assert_source_indices_allowed(selected, protection["protected_source_indices"])
        # When the run is genuinely protected, the plan's exclusion report must NAME the same
        # reference it excluded against -- an anonymous PASS can no longer masquerade as an
        # authoritative exclusion (ffv4o emitted a PASS with no reference_id and excluded nothing).
        if protection["protected_source_indices"]:
            report_reference_id = report.get("reference_id")
            if not report_reference_id:
                raise _AcquisitionPlanError(
                    "existing-pool protection report omits reference_id while the run binds a "
                    "protected reference; an anonymous exclusion report cannot be authoritative")
            if report_reference_id != protection["reference_id"]:
                raise _AcquisitionPlanError(
                    "existing-pool protection report reference_id does not match run-bound reference")
    return {**plan, "selected_source_global_indices": selected,
            "selected_parent_structure_ids": [str(x) for x in parents],
            "eligible_source_categories": [str(x) for x in categories],
            "n_selected": n_selected, "expected_output_count": expected}


def _load_pool_frames_global_order(pool_path):
    """Reproduce the pre-campaign planner's GLOBAL pool ordering from the source-pool manifest.

    The autonomous planner sizes/selects over ``framework_v2.acquisition.generic_representation.
    load_pool`` order: manifest ``categories`` order x per-category ASE read order. This reads the
    SAME manifest and concatenates the SAME category files in the SAME order, so the plan's
    ``selected_source_global_indices`` name exactly the frames the planner scored. Returns a list of
    ``(global_index, category, atoms)`` and the manifest dict."""
    from ase.io import read as ase_read
    manifest_path = Path(pool_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise _AcquisitionPlanError(f"existing-pool manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cats = manifest.get("categories")
    if not isinstance(cats, list) or not cats:
        raise _AcquisitionPlanError(
            f"existing-pool manifest {manifest_path} has no categories list")
    manifest_dir = manifest_path.parent
    ordered = []
    gidx = 0
    for cat in cats:
        category = str(cat["category"])
        rel = str(cat["sanitized_file"])
        fpath = (manifest_dir / rel)
        if not fpath.is_file():
            raise _AcquisitionPlanError(
                f"existing-pool manifest references a missing structure file: {fpath}")
        atoms_list = ase_read(str(fpath), index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        for atoms in atoms_list:
            ordered.append((gidx, category, atoms))
            gidx += 1
    return ordered, manifest


def _exec_select_existing_pool(proposal, progress_cb=None):
    """Execute an EXISTING_POOL_SELECTION acquisition plan: SELECT (never generate) a representative
    existing subset for canonical Teacher labeling.

    No Teacher inference happens here -- selection is descriptor/geometry work whose size and members
    were fixed deterministically by the pre-campaign planner. Any prior energies/forces/stress are
    STRIPPED so a historical label can never leak into training; the ONLY labels come from the
    downstream canonical teacher_labeling stage. The manifest emits the same n_frames/elements
    lineage contract the data_coverage reader consumes, so the existing-pool path is indistinguishable
    downstream from the perturbation path."""
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    plan = _validate_existing_pool_plan(
        _acquisition_plan_payload(p.get("acquisition_plan") or p.get("acquisition_plan_path")),
        reference_yaml=p.get("reference_yaml"),
        proposal_selected_source_indices=p.get("selected_source_indices"))
    if p.get("acquisition_plan_sha256") and p["acquisition_plan_sha256"] != plan["_plan_sha256"]:
        raise _AcquisitionPlanError("proposal acquisition_plan_sha256 does not match AcquisitionPlan")

    out_path = p.get("out_path")
    if not out_path:
        raise _AcquisitionPlanError("existing-pool selection requires out_path")
    manifest_path = p.get("manifest_path")
    reference_yaml = p.get("reference_yaml")
    audit_path = Path(p.get("protection_audit_path")
                      or Path(manifest_path or out_path).with_name("acquisition_protection_audit.json"))
    created_paths = [out_path, manifest_path, str(audit_path)]
    try:
        from ase.io import write as ase_write

        ordered, pool_manifest = _load_pool_frames_global_order(plan["pool_path"])
        pool_size = len(ordered)
        selected = list(plan["selected_source_global_indices"])
        parents = list(plan["selected_parent_structure_ids"])
        eligible = {str(x) for x in plan["eligible_source_categories"]}

        out_atoms = []
        selected_records = []
        for parent_id, gindex in zip(parents, selected):
            if gindex < 0 or gindex >= pool_size:
                raise _AcquisitionPlanError(
                    f"existing-pool selected_source_global_index {gindex} out of range for a pool "
                    f"of {pool_size} frames")
            gidx, category, atoms = ordered[gindex]
            geom = atoms.copy()
            geom.calc = None
            for k in ("energy", "forces", "stress", "free_energy", "dft_energy", "dft_forces"):
                geom.info.pop(k, None)
            geom.arrays.pop("forces", None)
            # Lineage: an existing-pool frame IS its own source row, so its top-level parent is that
            # global seed-pool index (the format the protected-reference lineage guard requires).
            geom.info["parent_structure_id"] = f"seed-pool:{int(gindex)}"
            geom.info["source_global_index"] = int(gindex)
            geom.info["pool_item_id"] = str(parent_id)
            geom.info["source_category"] = category
            geom.info["generation_backend"] = "existing_pool_selection.ase"
            geom.info["exploration_only"] = True
            out_atoms.append(geom)
            selected_records.append({
                "source_index": int(gindex),
                "structure_id": str(parent_id),
                "categories": [category]})

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(out_path), out_atoms, format="extxyz")

        # Protected-reference exclusion: selection is Teacher-free, so protection reduces to
        # (a) none of the selected global indices is a protected source row and (b) no selected
        # geometry matches a protected reference fingerprint. Lineage-descendant checking does not
        # apply (these are source frames, not perturbation children), so require_lineage is False.
        _protect_dataset(out_path, reference_yaml, selected_source_indices=selected,
                         require_lineage=False)

        elements = sorted({s for atoms in out_atoms for s in atoms.get_chemical_symbols()})
        n_frames = len(out_atoms)
        if n_frames != int(plan["expected_output_count"]):
            raise _AcquisitionPlanError(
                f"existing-pool output count mismatch: expected {plan['expected_output_count']}, "
                f"got {n_frames}")

        audit_result = None
        if reference_yaml:
            audit_result = _write_acquisition_protection_audit(
                audit_path, reference_yaml=reference_yaml, result_path=out_path,
                selected_source_indices=selected)

        artifact = {"path": str(Path(out_path).resolve()),
                    "integrity": artifact_digest(out_path)}
        if manifest_path:
            manifest = {
                "schema_version": 1,
                "operation": "select_existing_pool",
                "stage": p.get("stage", "acquisition"),
                "acquisition_plan": str(Path(plan["_plan_path"]).resolve()) if plan.get("_plan_path") else None,
                "acquisition_plan_sha256": plan["_plan_sha256"],
                "pool_path": str(Path(plan["pool_path"]).expanduser().resolve()),
                "pool_manifest_sha256": pool_manifest.get("sanitized_pool_manifest_sha256"),
                "reference_yaml": str(Path(reference_yaml).resolve()) if reference_yaml else None,
                "reference_yaml_integrity": artifact_digest(reference_yaml) if reference_yaml else None,
                "selected_parent_structure_ids": list(parents),
                "selected_source_global_indices": list(selected),
                "eligible_source_categories": sorted(eligible),
                "selected_source_records": selected_records,
                "expected_output_count": int(plan["expected_output_count"]),
                "actual_output_count": int(n_frames),
                "n_frames": int(n_frames),
                "elements": elements,
                "duplicate_handling": plan["duplicate_handling"],
                "dft_labels_used_as_selection_scores": False,
                "labeling_population_sizing": plan["labeling_population_sizing"],
                "protected_reference_exclusion_report": plan["protected_reference_exclusion_report"],
                "performs_teacher_inference": False,
                "output": artifact["path"],
                "output_integrity": artifact["integrity"],
            }
            if reference_yaml:
                manifest["protection_audit"] = str(audit_path.resolve())
                manifest["protection_audit_integrity"] = artifact_digest(audit_path)
                manifest["protection_audit_result"] = audit_result
            Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
            Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            artifact["manifest_path"] = str(Path(manifest_path).resolve())
            artifact["manifest_integrity"] = artifact_digest(manifest_path)
            if reference_yaml:
                artifact["protection_audit_path"] = str(audit_path.resolve())
                artifact["protection_audit_integrity"] = artifact_digest(audit_path)
        return artifact
    except Exception:
        _quarantine_acquisition_outputs(created_paths)
        raise


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


def _teacher_md_sanity_checks(cfg, teacher_config, frames, report_path):
    """Bounded Teacher-driven MD dynamic sanity check (scope: teacher acceptance, not a new
    top-level stage). Runs a small deterministic Langevin trajectory, under the SAME bound
    Teacher calculator used for the operational labels above, seeded from a handful of already-
    approved operational structures (no new sampling/acquisition policy). Purpose is only to
    catch an obviously broken Teacher dynamical response (energy/force blow-up, atoms colliding
    or the structure collapsing) before the Teacher is trusted as a labeling oracle -- this is
    not a physical-accuracy benchmark and makes no Teacher-vs-DFT claim.

    "No unphysical minimum interatomic distance" and "no obvious structural collapse" are both
    detected by the same minimum-pairwise-distance metric: a collapsing structure and an
    unphysical close contact are the same observable at this coarse a resolution, and inventing a
    second, materially different collapse metric would exceed the bounded scope of this check.
    """
    import math
    import numpy as np
    from adapters.acquisition import run_teacher_md
    from ase.io import read, write
    from validation.report import make_check
    from workflow.integrity import artifact_digest

    n_structures = int(cfg.get("n_structures", 3))
    if n_structures < 1:
        raise ValueError("teacher_md_sanity.n_structures must be >= 1")
    stride = max(1, len(frames) // n_structures)
    seeds = frames[::stride][:n_structures]
    if not seeds:
        raise ValueError("teacher_md_sanity found no operational structures to seed the MD sanity check")

    seed_path = cfg.get("seed_structures_path") or str(
        Path(report_path).with_name("teacher_md_sanity_seed.extxyz"))
    trajectory_path = cfg.get("trajectory_path") or str(
        Path(report_path).with_name("teacher_md_sanity_trajectory.extxyz"))
    write(seed_path, [s.copy() for s in seeds])
    run_teacher_md(cfg, teacher_config, seed_path, trajectory_path, capture_labels=True)

    md_frames = read(trajectory_path, index=":")
    if not md_frames:
        raise ValueError("teacher_md_sanity trajectory produced no snapshots")
    counts_by_seed: dict = {}
    for atoms in md_frames:
        idx = atoms.info["seed_structure_index"]
        counts_by_seed[idx] = counts_by_seed.get(idx, 0) + 1
    if (len(counts_by_seed) != len(seeds) or len(set(counts_by_seed.values())) != 1
            or next(iter(counts_by_seed.values())) < 1):
        raise ValueError(
            "teacher_md_sanity trajectory is incomplete: expected every seed structure to "
            f"produce an equal, non-zero number of snapshots, got {counts_by_seed}")

    md_energies, md_fmax, md_min_distance = [], [], []
    for atoms in md_frames:
        energy = float(atoms.info["teacher_energy"])
        forces = np.asarray(atoms.arrays["teacher_forces"], dtype=float)
        if (not math.isfinite(energy) or forces.shape != (len(atoms), 3)
                or not np.all(np.isfinite(forces))):
            raise ValueError("teacher_md_sanity produced non-finite or malformed energy/forces")
        md_energies.append(energy)
        md_fmax.append(float(np.max(np.linalg.norm(forces, axis=1))))
        n = len(atoms)
        if n < 2:
            md_min_distance.append(float("inf"))
        else:
            d = atoms.get_all_distances(mic=True)
            iu = np.triu_indices(n, k=1)
            md_min_distance.append(float(np.min(d[iu])))

    force_threshold = float(cfg.get("force_spike_threshold_eV_per_angstrom", 50.0))
    distance_threshold = float(cfg.get("min_distance_threshold_angstrom", 0.5))
    finite_min_distances = [m for m in md_min_distance if math.isfinite(m)]
    details = {
        "n_seed_structures": len(seeds), "n_snapshots": len(md_frames),
        "snapshots_per_seed": next(iter(counts_by_seed.values())),
        "temperature_K": float(cfg["temperature_K"]), "timestep_fs": float(cfg.get("timestep_fs", 1.0)),
        "n_steps": int(cfg["n_steps"]), "snapshot_interval": int(cfg.get("snapshot_interval", 100)),
        "energy_min": min(md_energies), "energy_max": max(md_energies),
        "trajectory": str(Path(trajectory_path).resolve()),
        "trajectory_integrity": artifact_digest(trajectory_path),
        "seed_structures": str(Path(seed_path).resolve()),
        "seed_structures_integrity": artifact_digest(seed_path),
    }
    checks = [
        make_check("teacher_dynamics_sanity", "teacher_md_sanity_no_force_spike", max(md_fmax),
                   "eV/Angstrom", {"operator": "max", "threshold": force_threshold}, details=details),
    ]
    if finite_min_distances:
        checks.append(make_check(
            "teacher_dynamics_sanity", "teacher_md_sanity_no_collapse", min(finite_min_distances),
            "Angstrom", {"operator": "min", "threshold": distance_threshold}, details=details))
    else:
        checks.append(make_check(
            "teacher_dynamics_sanity", "teacher_md_sanity_no_collapse", details=details,
            reason="every sanity-MD snapshot has fewer than 2 atoms; a pairwise minimum-"
                   "distance collapse check is not applicable to single-atom structures"))
    for check in checks:
        check["purpose"] = "deployment_stability"
        check["reference_source"] = "teacher"
        check["protocol"] = ("short deterministic Langevin MD under the bound Teacher calculator, "
                             "seeded from already-approved operational structures -- a dynamical "
                             "sanity gate, not a physical-accuracy or sampling protocol")
    return checks


def _exec_build_teacher_baseline(proposal):
    import math
    import numpy as np
    from ase.io import read
    from adapters import load_config
    from adapters.acquisition import label_with_teacher
    from adapters.teacher import load_teacher_with_species_evidence, species_mapping_is_attested
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
    teacher_cfg = load_config(p["teacher_config"])
    # Fail-fast: attest the actual constructed-calculator species/type mapping immediately after
    # calculator construction, BEFORE the expensive per-frame Teacher inference (label_with_teacher,
    # below) or Teacher-MD sanity checks run -- an unattested mapping must never be discovered only
    # after both have already been paid for.
    _, preflight_species_mapping = load_teacher_with_species_evidence(teacher_cfg)
    if not species_mapping_is_attested(preflight_species_mapping):
        raise ValueError(
            "Teacher baseline species_mapping is not attested: the declared config names a "
            "chemical_symbols/chemical_species_to_atom_type_map convention (or the identity-"
            "mapping fallback was applied) but the constructed calculator's own runtime state "
            "does not carry a resolved, non-empty species/type mapping -- refusing to run Teacher "
            f"inference or Teacher-MD against an unattested mapping (evidence: "
            f"{preflight_species_mapping})"
        )
    label_manifest_payload = label_with_teacher(
        teacher_cfg, structures, labeled_output, label_manifest,
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
    required = ["applicability_status", "applicability_limitations", "teacher_md_sanity"]
    missing = [name for name in required if name not in p]
    if missing:
        raise ValueError("Teacher baseline requires explicit deployment/applicability evidence: " + ", ".join(missing))
    # Deterministic locked-domain propagation (mirrors data_coverage). When this run has a frozen
    # validation contract, teacher_baseline.deployment_domain is sourced VERBATIM from the locked
    # teacher_applicability_domain.value -- it is never re-authored by a proposal. A re-executed
    # teacher_baseline (e.g. via recovery) may only run under the SAME frozen Teacher applicability
    # domain, never a redefined one; a genuine domain change requires a new run.
    contract_path = _resolve_validation_contract_path(p, report_path)
    locked_deployment_domain = None
    if contract_path is not None and contract_path.is_file():
        contract_doc = json.loads(contract_path.read_text())
        component = (contract_doc.get("components") or {}).get("teacher_applicability_domain")
        if not isinstance(component, dict) or "value" not in component:
            raise ValueError(
                "validation contract is bound but has no teacher_applicability_domain.value; "
                "cannot deterministically source teacher_baseline.deployment_domain"
            )
        locked_deployment_domain = component["value"]
    if locked_deployment_domain is not None:
        deployment_domain = locked_deployment_domain
    elif "deployment_domain" in p:
        deployment_domain = p["deployment_domain"]
    else:
        raise ValueError(
            "Teacher baseline requires explicit deployment/applicability evidence: deployment_domain")
    applicability_status = p["applicability_status"]
    limitations = list(p["applicability_limitations"])
    species_mapping = label_manifest_payload.get("species_mapping_evidence") or {}
    species_mapping_attested = species_mapping_is_attested(species_mapping)
    md_sanity_checks = _teacher_md_sanity_checks(
        p["teacher_md_sanity"], load_config(p["teacher_config"]), frames, report_path)
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
        "species_mapping": species_mapping,
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
        }, {
            "domain": "operational_teacher_inference",
            "observable": "runtime_species_type_mapping_attested",
            "status": "PASS" if species_mapping_attested else "FAIL",
            "value": 1 if species_mapping_attested else 0,
            "unit": "boolean",
            "criterion": {"operator": "equals", "target": 1},
            "purpose": "deployment_stability",
            "reference_source": "teacher",
            "protocol": "deterministic capture of the calculator kwargs actually bound at "
                       "construction time (adapters.teacher.load_teacher_with_species_evidence), "
                       "never an LLM's interpretation of the declared config",
            "details": {"fallback_applied": bool(species_mapping.get("fallback_applied"))},
        }, *md_sanity_checks],
        "evidence": [
            _evidence("teacher_config", p["teacher_config"]),
            _evidence("distillation_scope", p["distillation_scope"]),
            _evidence("validation_profile", p["validation_profile"]),
            _evidence("operational_structures", structures),
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    # Self-validate against the SAME locked contract the controller enforces, so a re-authored or
    # drifted deployment_domain fails fast in-process rather than only at the external contract gate.
    validate_teacher_baseline_report(
        report_path,
        validation_contract_path=(str(contract_path) if contract_path is not None
                                  and contract_path.is_file() else None),
    )
    return {"path": str(report_path.resolve()), "report": report,
            "integrity": artifact_digest(report_path),
            "labeled_output": str(Path(labeled_output).resolve()),
            "label_manifest": str(Path(label_manifest).resolve())}


def _exec_acquire_structures(proposal, progress_cb=None):
    from adapters import load_config
    from adapters.acquisition import acquire
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    # Discriminate the two executable acquisition projections on the plan's own shape: an
    # EXISTING_POOL_SELECTION plan carries a ``pool_path`` (SELECT an existing subset, no Teacher
    # inference); the legacy perturbation plan does not (GENERATE new frames via the Teacher).
    _raw_plan = _acquisition_plan_payload(p.get("acquisition_plan") or p.get("acquisition_plan_path"))
    if _is_existing_pool_plan(_raw_plan):
        return _exec_select_existing_pool(proposal, progress_cb=progress_cb)
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
                         p["seed_structures"], p["out_path"], progress_cb=progress_cb)
        n_frames = _validate_acquisition_output(result, plan)
        # Emit the data-coverage lineage contract (n_frames + real elements) directly from the
        # produced artifact, so the acquisition manifest is self-sufficient for the downstream
        # data_coverage reader. Derived from the actual accepted output frames -- never asserted,
        # never element-list hardcoded -- so it generalizes to any chemistry.
        from ase.io import read as _ase_read
        _produced_frames = _ase_read(str(result), index=":")
        acquisition_elements = sorted(
            {s for atoms in _produced_frames for s in atoms.get_chemical_symbols()})
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
                "n_frames": int(n_frames),
                "elements": acquisition_elements,
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
    structures_path = p.get("structures_path", p.get("structures"))
    reference_yaml = p.get("reference_yaml")
    selected_source_indices = p.get("selected_source_indices")
    _protect_dataset(structures_path, reference_yaml, selected_source_indices, require_lineage=True)
    manifest = label_with_teacher(
        load_config(p["teacher_config"]),
        structures_path,
        p.get("out_path", p.get("labeled_output")),
        p["manifest_path"],
        bool(p.get("include_stress", False)),
    )
    result = {"path": manifest["output"], "sha256": manifest["sha256"],
              "manifest_path": str(Path(p["manifest_path"]).resolve()),
              "manifest_integrity": artifact_digest(p["manifest_path"])}
    # Persist the protected-reference exclusion audit that this stage declares as an output and
    # gates on: the in-memory _protect_dataset check above proves the labeled population, but the
    # durable audit artifact is what record_gate's validation_manifest contract re-verifies. The
    # labeled output (not the pre-label input) is the audited dataset -- labeling only attaches
    # Teacher energies/forces, preserving every frame's geometry and parent lineage.
    if reference_yaml:
        labeled_output = Path(manifest["output"]).resolve()
        audit_path = Path(p.get("protection_audit_path")
                          or Path(p["manifest_path"]).with_name("teacher_labeling_protection_audit.json"))
        _write_teacher_labeling_protection_audit(
            audit_path, reference_yaml=reference_yaml, labeled_output=labeled_output,
            selected_source_indices=selected_source_indices)
        result["protection_audit_path"] = str(audit_path.resolve())
        result["protection_audit_integrity"] = artifact_digest(audit_path)
    return result


def _write_teacher_labeling_protection_audit(path, *, reference_yaml, labeled_output,
                                             selected_source_indices=None):
    from validation.protected_reference import write_protection_audit, validate_protection_audit_report
    write_protection_audit("teacher_labeling", reference_yaml,
                           [f"teacher_labeled={Path(labeled_output).resolve()}"],
                           path, selected_source_indices=[int(x) for x in (selected_source_indices or [])])
    return validate_protection_audit_report(
        path, reference_yaml=reference_yaml,
        submitted_artifacts=[Path(path).resolve(), Path(labeled_output).resolve()])



class _ReferenceReuseBlocked(RuntimeError):
    """Raised when VERIFIED_HISTORICAL_REUSE cannot deterministically verify the
    historical Teacher-vs-reference evidence. Fail-closed: never fall back to fresh
    Teacher inference during a no-Teacher-inference campaign."""


def _reference_validation_verified_reuse(proposal, p, protection, teacher_cfg,
                                         teacher_config, reference_yaml,
                                         predictions_path, report_path,
                                         historical_report_path, historical_predictions_path):
    """Complete the historical Stage-1/2 evidence reuse path for reference_validation.

    Deterministically verify that an ALREADY-EXISTING Teacher-vs-reference artifact is
    identity-, provenance-, and scope-compatible with THIS run's authoritative reference
    contract and Teacher identity, then re-derive the Teacher-vs-DFT metrics from that
    verified artifact WITHOUT any fresh Teacher inference. Every required identity /
    provenance / condition / compatibility check must pass or the run is BLOCKED with the
    exact mismatch."""
    import shutil
    from ase.io import read
    from adapters.teacher import teacher_model_reference
    from validation.four_channel_audit import channel
    from validation.reference_validation import validate_reference_validation_report
    from workflow.integrity import artifact_digest, sha256_file

    if not historical_predictions_path:
        raise _ReferenceReuseBlocked(
            "AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: verified reuse requires "
            "'historical_predictions' alongside 'historical_report'")
    hist_report_path = Path(historical_report_path).expanduser().resolve()
    hist_pred_path = Path(historical_predictions_path).expanduser().resolve()
    if not hist_report_path.is_file():
        raise _ReferenceReuseBlocked(
            f"AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: historical report not found: {hist_report_path}")
    if not hist_pred_path.is_file():
        raise _ReferenceReuseBlocked(
            f"AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: historical predictions not found: {hist_pred_path}")
    hist = json.loads(hist_report_path.read_text())

    verified = []

    def require(cond_id, description, expected, actual):
        if expected != actual:
            raise _ReferenceReuseBlocked(
                "AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: reference_validation "
                f"verified-reuse check {cond_id} failed ({description}): "
                f"expected {expected!r} != actual {actual!r}")
        verified.append({"check": cond_id, "description": description,
                         "expected": expected, "actual": actual, "status": "VERIFIED"})

    teacher_config_path = Path(teacher_config).expanduser().resolve()
    cur_config_sha = sha256_file(teacher_config_path)
    model_value = teacher_model_reference(teacher_cfg)
    model_path = Path(model_value).expanduser().resolve() if model_value else None
    if not (model_path and model_path.is_file()):
        raise _ReferenceReuseBlocked(
            "AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: current Teacher model file "
            f"is not resolvable for identity verification: {model_value!r}")
    cur_model_sha = sha256_file(model_path)
    hist_teacher = hist.get("teacher") or {}
    hist_lm = hist_teacher.get("label_manifest") or {}
    hist_ref = hist.get("reference") or {}
    hist_pred_meta = hist.get("prediction_artifact") or {}

    # 1. exact Teacher model identity/hash + config identity match
    require("1a", "Teacher model sha256 identity",
            (hist_teacher.get("model_sha256")), cur_model_sha)
    require("1b", "Teacher config sha256 identity",
            (hist_teacher.get("config_integrity") or {}).get("sha256"), cur_config_sha)
    # 2. exact protected-reference structure hash match
    cur_ref_struct = artifact_digest(protection["reference_path"])
    require("2", "protected reference structures sha256 identity",
            (hist_ref.get("structures_integrity") or {}).get("sha256"),
            cur_ref_struct.get("sha256"))
    # 3. exact reference population + membership match
    require("3a", "reference_id identity", hist_ref.get("reference_id"), protection["reference_id"])
    require("3b", "logical frame count identity", int(hist_ref.get("logical_frames")),
            int(protection["logical_frames"]))
    require("3c", "protected source row count identity",
            int(hist_ref.get("protected_source_rows")), int(protection["protected_source_rows"]))
    # 4. historical Teacher prediction artifact identity/hash is known + matches
    cur_hist_pred_sha = sha256_file(hist_pred_path)
    require("4a", "historical prediction sha256 recorded in report",
            (hist_pred_meta.get("integrity") or {}).get("sha256"), cur_hist_pred_sha)
    declared_hist_sha = ((_load_reference_yaml(reference_yaml).get("historical_teacher_prediction") or {})
                         .get("sha256"))
    require("4b", "historical prediction sha256 declared in reference contract",
            declared_hist_sha, cur_hist_pred_sha)
    # 5. original Teacher-inference provenance known and valid
    require("5a", "label manifest Teacher model provenance",
            hist_lm.get("teacher_model_sha256"), cur_model_sha)
    require("5b", "label manifest Teacher config provenance",
            hist_lm.get("teacher_config_sha256"), cur_config_sha)
    require("5c", "label manifest frame count provenance",
            int(hist_lm.get("n_frames")), int(protection["logical_frames"]))
    if not hist_lm.get("source_sha256"):
        raise _ReferenceReuseBlocked(
            "AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: reference_validation "
            "verified-reuse check 5d failed (original inference source provenance missing)")
    verified.append({"check": "5d", "description": "original inference source provenance present",
                     "expected": "non-empty", "actual": hist_lm.get("source_sha256"),
                     "status": "VERIFIED"})
    # 6. DFT/reference labels present + complete (re-derive channel metrics; no Teacher call)
    frames = read(str(hist_pred_path), index=":")
    require("6a", "prediction frame count identity", len(frames), int(protection["logical_frames"]))
    metrics_raw = channel(frames, "dft", "teacher", per_config_type=True, require_complete=True)
    if not metrics_raw or "all" not in metrics_raw:
        raise _ReferenceReuseBlocked(
            "AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: reference_validation "
            "verified-reuse check 6b failed (Teacher-vs-DFT metrics not recomputable)")
    verified.append({"check": "6b", "description": "Teacher-vs-DFT labels complete + metrics recomputed",
                     "expected": "complete", "actual": "complete", "status": "VERIFIED"})
    # 7. scientific conditions compatible (units/normalization)
    hist_metrics = hist.get("metrics") or {}
    require("7a", "energy normalization compatibility", hist_metrics.get("energy_normalization"), "per_atom")
    require("7b", "energy unit compatibility", hist_metrics.get("energy_unit"), "meV/atom")
    require("7c", "force unit compatibility", hist_metrics.get("force_unit"), "eV/Angstrom")
    # 8. current DeploymentScope/ValidationProfile does not make it inapplicable
    #    (current reference contract resolved AVAILABLE + frame identity above)
    require("8", "protected reference use compatibility",
            hist.get("protected_reference_use"), "teacher_vs_dft_reference_validation_only")
    # 9. no blind/reference access rule violated (validate_reference_config already enforced
    #    protection; reference_validation is an allowed consumer)
    verified.append({"check": "9", "description": "protected reference access rules enforced",
                     "expected": "enforced", "actual": "enforced", "status": "VERIFIED"})

    def metric_subset(src):
        return {
            "n_frames": int(src["n_frames"]), "n_atoms": int(src["n_atoms"]),
            "energy_mae": float(src["e_raw_mae_meV"]), "energy_rmse": float(src["e_raw_rmse_meV"]),
            "force_component_mae": float(src["f_mae"]), "force_component_rmse": float(src["f_rmse"]),
        }

    by_config_type = {k: metric_subset(v) for k, v in metrics_raw.items() if k != "all"}
    # Materialize this run's own artifact as a byte-identical copy (SHA preserved).
    shutil.copyfile(hist_pred_path, predictions_path)
    if sha256_file(predictions_path) != cur_hist_pred_sha:
        raise _ReferenceReuseBlocked(
            "AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED: reference_validation "
            "verified-reuse artifact copy did not preserve SHA")
    domain_fields = {"config_type": "present" if by_config_type else "absent"}
    for field in p.get("domain_fields", ["structural_domain"]):
        if field != "config_type":
            domain_fields[field] = "present" if any(field in a.info for a in frames) else "absent"
    m_all = metric_subset(metrics_raw["all"])
    report = {
        "schema_version": 1,
        "profile": "teacher_reference_validation",
        "stage": "reference_validation",
        "protected_reference_use": "teacher_vs_dft_reference_validation_only",
        "evidence_source": "VERIFIED_HISTORICAL_REUSE",
        "historical_prediction_policy": "VERIFIED_HISTORICAL_REUSE",
        "reuse_verification": {
            "historical_report": {"path": str(hist_report_path),
                                  "sha256": sha256_file(hist_report_path)},
            "historical_predictions": {"path": str(hist_pred_path), "sha256": cur_hist_pred_sha},
            "conditions": verified,
        },
        "teacher": {
            "config": str(teacher_config_path),
            "config_integrity": artifact_digest(teacher_config_path),
            "model": str(model_path),
            "model_integrity": artifact_digest(model_path),
            "model_sha256": cur_model_sha,
            "label_manifest": hist_lm,
        },
        "reference": {
            "reference_id": protection["reference_id"],
            "reference_yaml": str(Path(reference_yaml).expanduser().resolve()),
            "structures_path": str(protection["reference_path"]),
            "logical_frames": int(protection["logical_frames"]),
            "protected_source_rows": int(protection["protected_source_rows"]),
            "structures_integrity": cur_ref_struct,
        },
        "prediction_artifact": {
            "path": str(predictions_path),
            "integrity": artifact_digest(predictions_path),
            "n_frames": len(frames),
            "labels": ["teacher_energy", "teacher_forces", "dft_energy", "dft_forces"],
        },
        "metrics": {
            "energy_normalization": "per_atom", "energy_unit": "meV/atom", "force_unit": "eV/Angstrom",
            "global": m_all, "by_config_type": by_config_type, "domain_fields": domain_fields,
        },
        "checks": [
            {"domain": "teacher_reference", "observable": "verified_historical_reuse",
             "status": "VERIFIED", "value": len(verified), "unit": "conditions", "criterion": None},
            {"domain": "teacher_reference", "observable": "logical_frame_count", "status": "RECORDED",
             "value": int(protection["logical_frames"]), "unit": "frames", "criterion": None},
            {"domain": "teacher_reference", "observable": "energy_mae", "status": "RECORDED",
             "value": m_all["energy_mae"], "unit": "meV/atom", "criterion": None},
            {"domain": "teacher_reference", "observable": "energy_rmse", "status": "RECORDED",
             "value": m_all["energy_rmse"], "unit": "meV/atom", "criterion": None},
            {"domain": "teacher_reference", "observable": "force_component_mae", "status": "RECORDED",
             "value": m_all["force_component_mae"], "unit": "eV/Angstrom", "criterion": None},
            {"domain": "teacher_reference", "observable": "force_component_rmse", "status": "RECORDED",
             "value": m_all["force_component_rmse"], "unit": "eV/Angstrom", "criterion": None},
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
                                         submitted_artifacts=[report_path, predictions_path],
                                         reuse_verified_historical=True)
    return {"path": str(report_path), "report": report, "integrity": artifact_digest(report_path),
            "predictions_path": str(predictions_path),
            "predictions_integrity": artifact_digest(predictions_path),
            "evidence_source": "VERIFIED_HISTORICAL_REUSE",
            "reuse_verification": report["reuse_verification"]}


def _load_reference_yaml(reference_yaml):
    import yaml
    return yaml.safe_load(Path(reference_yaml).expanduser().read_text(encoding="utf-8")) or {}


def _exec_validate_teacher_reference(proposal):
    import tempfile
    from ase.io import read
    from adapters import load_config
    from adapters.acquisition import label_with_teacher
    from adapters.teacher import teacher_model_reference
    from validation.four_channel_audit import channel
    from validation.protected_reference import assert_reference_provides_dft_labels
    from validation.reference_validation import validate_reference_validation_report
    from workflow.integrity import artifact_digest, sha256_file
    p = _params(proposal)
    reference_yaml = p["reference_yaml"]
    teacher_config = p["teacher_config"]
    # Teacher-vs-DFT reference_validation reads the reference's own DFT/Teacher labels, so it is
    # an EVALUATION-reference consumer. Fail closed (rather than surface an obscure
    # missing-label error later) if a PROTECTION-ONLY structure-identity reference -- which must
    # never expose label truth to the early Student route -- is misrouted to this label path.
    protection = assert_reference_provides_dft_labels(reference_yaml)
    predictions_path = Path(p["predictions_path"]).resolve()
    report_path = Path(p["report_path"]).resolve()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_cfg = load_config(teacher_config)
    if p.get("historical_report"):
        return _reference_validation_verified_reuse(
            proposal, p, protection, teacher_cfg, teacher_config, reference_yaml,
            predictions_path, report_path, p["historical_report"], p.get("historical_predictions"))
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
                               p["manifest_path"],
                               continue_from=p.get("continue_from"),
                               total_epoch_override=p.get("total_epoch_override"))
    return {"path": str(Path(p["manifest_path"]).resolve()), "manifest": manifest,
            "integrity": artifact_digest(p["manifest_path"])}


def _exec_evaluate_committee(proposal):
    from workflow.steps import evaluate_committee, evaluate_multi_population
    from workflow.integrity import artifact_digest
    p = _params(proposal)
    # Multi-population Stage-8: when the stage binds a role-bound
    # MultiPopulationEvaluationPlan (inline ``evaluation_plan`` or a bound
    # ``evaluation_plan_path``), evaluate each role-bound population separately
    # and emit one channel-separated, provenance-bound report. The single-
    # population path below stays the default when no plan is bound.
    plan = p.get("evaluation_plan")
    plan_path = p.get("evaluation_plan_path")
    if plan is None and plan_path:
        plan = json.loads(Path(plan_path).read_text())
    if plan is not None:
        report = evaluate_multi_population(
            p["student_config"], p["committee_manifest"], plan,
            p["training_frames_path"], p["labeled_output"], p["report_path"],
            code_revision=p.get("code_revision"))
        return {"path": str(Path(p["report_path"]).resolve()), "report": report,
                "integrity": artifact_digest(p["report_path"]),
                "labeled_output": str(Path(p["labeled_output"]).resolve())}
    # Governed protected-reference isolation: when an access-partition contract is bound, the
    # Stage-8 fidelity claim is restricted to the ``protected_stage8_evaluation`` role, while
    # committee predictions are still embedded on the full evaluated population so Stage-9 can
    # slice the disjoint calibration roles from the same labeled artifact.
    report_fingerprints = None
    partition_provenance = None
    access_partition_path = p.get("access_partition_path")
    if access_partition_path:
        from validation.access_partition import (
            ROLE_STUDENT_FINAL_EVALUATION, validate_access_partition_contract,
            assert_stage_partition_access, resolve_partition_fingerprints,
        )
        role = p.get("partition_role", ROLE_STUDENT_FINAL_EVALUATION)
        contract = validate_access_partition_contract(
            access_partition_path,
            expected_reference_id=p.get("expected_reference_id"),
            expected_structures_sha256=p.get("expected_structures_sha256"),
        )
        assert_stage_partition_access(contract, "evaluation", role)
        report_fingerprints = resolve_partition_fingerprints(contract, role)
        partition_provenance = {
            "access_partition_path": str(Path(access_partition_path).resolve()),
            "partition_assignment_sha256": contract.get("partition_assignment_sha256"),
            "role": role,
            "role_n_frames": (contract.get("partitions") or {}).get(role, {}).get("n_frames"),
        }
    report = evaluate_committee(
        p["student_config"], p["committee_manifest"], p["frames_path"],
        p["labeled_output"], p["report_path"], p.get("required_channels"),
        report_fingerprints=report_fingerprints)
    result = {"path": str(Path(p["report_path"]).resolve()), "report": report,
              "integrity": artifact_digest(p["report_path"]),
              "labeled_output": str(Path(p["labeled_output"]).resolve())}
    if partition_provenance is not None:
        result["governed_partition"] = partition_provenance
    return result


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


def _exec_resolve_deployment_checkpoint(proposal):
    """Resolve the canonical deployed Student checkpoint (Stage-10) from the committee manifest.

    Selecting the deployed member is a governed decision (``selected_seed`` explicit, or a
    ``select_by`` policy fully determined by the manifest); the checkpoint path and sha256 are
    then DERIVED from the manifest -- never hand-typed into a proposal. Writes a
    ``deployment_provenance.json`` carrying the resolved Student identity (+ starting-structure
    identity + ensemble role) so Stage-10 C2b identity checks can bind approved-vs-realized.

    When ``expected_committee_manifest_sha256`` is supplied (the sha256 of the committee manifest
    the training stage PUBLISHED in this run, derived by
    ``validation.deployment_resolution.resolve_published_committee_manifest``), the consumed
    manifest must be byte-identical to it -- this is the run-binding that makes it impossible to
    deploy a checkpoint the training stage never published.
    """
    from validation.deployment_resolution import (
        resolve_selected_checkpoint, build_deployment_provenance)
    from workflow.integrity import sha256_file, artifact_digest
    p = _params(proposal)
    student_identity = resolve_selected_checkpoint(
        p["committee_manifest"], selected_seed=p.get("selected_seed"),
        select_by=p.get("select_by"),
        expected_manifest_sha256=p.get("expected_committee_manifest_sha256"))
    datafile = Path(p["starting_structure"]).resolve()
    if not datafile.is_file():
        raise ValueError(f"deployment starting_structure does not exist: {datafile}")
    starting_structure = {
        "path": str(datafile),
        "sha256": sha256_file(datafile),
        "provenance_role": p.get("starting_structure_role", "deployment_starting_structure"),
        "leakage_check": p.get("leakage_check"),
    }
    provenance = build_deployment_provenance(
        student_identity, starting_structure=starting_structure,
        ensemble_role=p.get("ensemble_role", "deployment_md"),
        shared_md_protocol=p.get("shared_md_protocol"),
        extra=p.get("provenance_extra"))
    out_path = Path(p["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return {"path": str(out_path.resolve()), "student": student_identity,
            "provenance": provenance, "integrity": artifact_digest(out_path)}


def _exec_build_deployment_context(proposal):
    """Produce the LAMMPS deployment context.yaml (Stage-10) from the frozen shared MD protocol.

    ``ensemble='nvt'`` renders the thermostatted production-trajectory context;
    ``ensemble='nve'`` renders the dedicated microcanonical energy-conservation segment context
    (distinct from production) whose total-energy drift is a valid NVE metric. Every step count
    is derived from the frozen ``shared_md_protocol`` -- no hand-tuned numbers. The derived NVE
    protocol + autonomous-choice rationale is returned and, for NVE, also embedded as a
    ``_nve_protocol`` comment-safe key inside the returned payload (NOT written into the plain
    template context).
    """
    import yaml
    from validation.deployment_resolution import (
        load_shared_md_protocol, build_deployment_context)
    from workflow.integrity import sha256_file
    p = _params(proposal)
    if p.get("shared_md_protocol") is not None:
        shared = p["shared_md_protocol"]
    else:
        shared = load_shared_md_protocol(p["validation_profile"])
    ensemble = p["ensemble"]
    context = build_deployment_context(
        shared, ensemble, p["starting_structure"],
        velocity_seed=int(p["velocity_seed"]), mpi_ranks=int(p.get("mpi_ranks", 1)),
        dump_file=p.get("dump_file"), energy_log=p.get("energy_log"),
        nve_segment_ps=p.get("nve_segment_ps"))
    nve_protocol = context.pop("_nve_protocol", None)
    out_path = Path(p["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(context, sort_keys=True))
    result = {"path": str(out_path.resolve()), "context": context,
              "ensemble": ensemble, "sha256": sha256_file(out_path)}
    if nve_protocol is not None:
        result["nve_protocol"] = nve_protocol
    return result


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
    _ready("build_split_membership_population", "data-curator",
           "workflow.steps.build_split_membership_population",
           "source_dataset,split_source_manifest,target_split,output_path[,manifest_path]",
           "recovered split-membership population (sha256-bound)",
           _exec_build_split_membership_population,
           validator="validation.protected_reference recovered-original-holdout kind"),
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
    _ready("validate_species_mapping_consistency", "data-curator",
           "adapters.teacher species-mapping cross-check",
           "manifest_path[,teacher_config,expected_manifest_sha256,out_path]",
           "species-mapping consistency evidence", de.validate_species_mapping_consistency,
           validator="adapters.teacher species mapping attestation + cross-check"),
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
    _ready("resolve_deployment_checkpoint", "simulation",
           "validation.deployment_resolution.resolve_selected_checkpoint",
           "committee_manifest,starting_structure,out_path[,selected_seed,select_by,ensemble_role,"
           "expected_committee_manifest_sha256]",
           "deployment_provenance.json (resolved Student + starting-structure identity)",
           _exec_resolve_deployment_checkpoint,
           validator="committee-manifest checkpoint cross-check"),
    _ready("build_deployment_context", "simulation",
           "validation.deployment_resolution.build_deployment_context",
           "ensemble,starting_structure,velocity_seed,out_path[,validation_profile,"
           "shared_md_protocol,mpi_ranks,dump_file,energy_log,nve_segment_ps]",
           "LAMMPS deployment context.yaml (NVT production or dedicated NVE segment)",
           _exec_build_deployment_context,
           validator="frozen shared_md_protocol derivation"),
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
           "validation_profile,frames_path[,elements,cutoffs,energies,n_atoms,timestep_fs,"
           "energy_log_path,nve_md_manifest]",
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


def required_parameters_for_action(action_type: str) -> Optional[frozenset]:
    """The top-level ``parameters`` keys a deterministic READY executor unconditionally reads,
    parsed single-source from that action's ``input_contract`` (``"req1,req2[,opt1,opt2]"`` --
    tokens before the optional ``[...]`` group are required, tokens inside it are optional). Used
    by recovery-plan acceptance validation to reject a corrective_action whose parameters would
    make its executor raise ``KeyError`` at dispatch, BEFORE a human approves it.

    Returns ``None`` (meaning "no parseable parameter contract -- do not fail closed on this")
    for any action that is not a real deterministic READY executor (HPC/interface/reasoning
    bindings, no ``fn``) or whose contract is free-text rather than a comma-separated parameter
    list -- so this never manufactures a spurious requirement it cannot actually prove.
    """
    b = BINDINGS.get(action_type)
    if b is None or b.fn is None or b.status != "READY_EXECUTOR":
        return None
    contract = (b.input_contract or "").strip()
    required_part = contract.split("[", 1)[0]
    tokens = [t.strip() for t in required_part.split(",") if t.strip()]
    if not tokens or any((" " in t or not t.replace("_", "").isalnum()) for t in tokens):
        return None  # free-text contract, not a parameter list
    return frozenset(tokens)


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
