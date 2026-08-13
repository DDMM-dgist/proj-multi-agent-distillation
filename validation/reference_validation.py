"""Deterministic validator for Teacher-vs-DFT protected reference validation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from ase.io import read

from adapters import load_config
from adapters.teacher import teacher_model_reference
from validation.four_channel_audit import channel
from validation.protected_reference import validate_reference_config
from validation.report import validate_evidence
from workflow.integrity import artifact_digest, sha256_file, verify_artifact

REQUIRED_LOGICAL_FRAMES = 1155
REQUIRED_PROTECTED_SOURCE_ROWS = 1156
GLOBAL_METRIC_KEYS = {
    "n_frames", "n_atoms", "energy_mae", "energy_rmse",
    "force_component_mae", "force_component_rmse",
}


def _resolve(raw, base):
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_subset(src):
    return {
        "n_frames": int(src["n_frames"]),
        "n_atoms": int(src["n_atoms"]),
        "energy_mae": float(src["e_raw_mae_meV"]),
        "energy_rmse": float(src["e_raw_rmse_meV"]),
        "force_component_mae": float(src["f_mae"]),
        "force_component_rmse": float(src["f_rmse"]),
    }


def _assert_metric_close(observed, expected, label):
    for key in GLOBAL_METRIC_KEYS:
        if key not in observed:
            raise ValueError(f"{label} metric is missing {key}")
        if key in {"n_frames", "n_atoms"}:
            if int(observed[key]) != int(expected[key]):
                raise ValueError(f"{label} metric mismatch for {key}: {observed[key]} != {expected[key]}")
        elif not _finite(observed[key]) or not math.isclose(float(observed[key]), float(expected[key]), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{label} metric mismatch for {key}: {observed[key]} != {expected[key]}")


def _teacher_identity(teacher_config):
    cfg = load_config(teacher_config)
    model = teacher_model_reference(cfg)
    model_path = Path(model).expanduser().resolve() if model else None
    config_path = Path(teacher_config).expanduser().resolve()
    return {
        "config": str(config_path),
        "config_integrity": artifact_digest(config_path),
        "model": str(model_path) if model_path else model,
        "model_integrity": artifact_digest(model_path) if model_path and model_path.exists() else None,
        "model_sha256": sha256_file(model_path) if model_path and model_path.is_file() else None,
    }


def _historical_prediction_sha(reference_yaml):
    import yaml
    cfg = yaml.safe_load(Path(reference_yaml).read_text(encoding="utf-8")) or {}
    hist = cfg.get("historical_teacher_prediction") or {}
    path = hist.get("path")
    return (str(Path(path).expanduser().resolve()) if path else None, hist.get("sha256"))


def validate_reference_validation_report(
    manifest_path,
    reference_yaml=None,
    teacher_config=None,
    submitted_artifacts=None,
    allowed_evidence=None,
    enforce_required_pass=False,
):
    """Validate Teacher-vs-DFT reference validation without relying on report claims."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("reference validation report requires schema_version=1")
    if payload.get("profile") != "teacher_reference_validation":
        raise ValueError("reference validation report has wrong profile")
    if payload.get("stage") != "reference_validation":
        raise ValueError("reference validation report has wrong stage")
    if payload.get("status") == "PASS":
        raise ValueError("reference validation report must not self-declare PASS")

    reference_yaml = reference_yaml or (payload.get("reference") or {}).get("reference_yaml")
    teacher_config = teacher_config or (payload.get("teacher") or {}).get("config")
    if not reference_yaml or not teacher_config:
        raise ValueError("reference_yaml and teacher_config are required")
    reference_yaml = str(Path(reference_yaml).expanduser().resolve())
    teacher_config = str(Path(teacher_config).expanduser().resolve())

    protection = validate_reference_config(reference_yaml)
    if protection["logical_frames"] != REQUIRED_LOGICAL_FRAMES:
        raise ValueError("protected reference logical-frame count is not 1155")
    if protection["protected_source_rows"] != REQUIRED_PROTECTED_SOURCE_ROWS:
        raise ValueError("protected source-row count is not 1156")

    ref = payload.get("reference")
    if not isinstance(ref, dict):
        raise ValueError("reference validation report requires reference block")
    expected_ref = {
        "reference_id": protection["reference_id"],
        "reference_yaml": reference_yaml,
        "structures_path": str(protection["reference_path"]),
        "logical_frames": REQUIRED_LOGICAL_FRAMES,
        "protected_source_rows": REQUIRED_PROTECTED_SOURCE_ROWS,
    }
    for key, expected in expected_ref.items():
        if ref.get(key) != expected:
            raise ValueError(f"reference block mismatch for {key}: {ref.get(key)!r} != {expected!r}")

    prediction = payload.get("prediction_artifact")
    if not isinstance(prediction, dict) or not prediction.get("path") or not isinstance(prediction.get("integrity"), dict):
        raise ValueError("reference validation report requires prediction_artifact")
    pred_path = _resolve(prediction["path"], manifest_path.parent)
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    verify_artifact(pred_path, prediction["integrity"])
    if artifact_digest(pred_path) != prediction["integrity"]:
        raise ValueError("prediction artifact integrity mismatch")
    hist_path, hist_sha = _historical_prediction_sha(reference_yaml)
    if hist_path and pred_path == Path(hist_path):
        raise ValueError("historical Teacher prediction path cannot be accepted as fresh output")
    if hist_sha and prediction["integrity"].get("sha256") == hist_sha:
        raise ValueError("historical Teacher prediction SHA cannot be accepted as fresh output")

    frames = read(str(pred_path), index=":")
    if len(frames) != REQUIRED_LOGICAL_FRAMES:
        raise ValueError(f"prediction frame count mismatch: {len(frames)} != {REQUIRED_LOGICAL_FRAMES}")
    for index, atoms in enumerate(frames):
        for key in ("dft_energy", "teacher_energy"):
            if key not in atoms.info or not _finite(float(atoms.info[key])):
                raise ValueError(f"frame {index} missing finite {key}")
        for key in ("dft_forces", "teacher_forces"):
            if key not in atoms.arrays:
                raise ValueError(f"frame {index} missing {key}")
            arr = np.asarray(atoms.arrays[key], dtype=float)
            if arr.shape != (len(atoms), 3) or not np.all(np.isfinite(arr)):
                raise ValueError(f"frame {index} has invalid {key}")

    teacher = payload.get("teacher")
    expected_teacher = _teacher_identity(teacher_config)
    if not isinstance(teacher, dict):
        raise ValueError("reference validation report requires teacher block")
    for key, expected in expected_teacher.items():
        if teacher.get(key) != expected:
            raise ValueError(f"teacher block mismatch for {key}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("reference validation report requires metrics")
    if metrics.get("energy_normalization") != "per_atom":
        raise ValueError("energy normalization must be per_atom")
    if metrics.get("energy_unit") != "meV/atom" or metrics.get("force_unit") != "eV/Angstrom":
        raise ValueError("metric units are incorrect")
    recomputed = channel(frames, "dft", "teacher", per_config_type=True, require_complete=True)
    if not recomputed or "all" not in recomputed:
        raise ValueError("Teacher-vs-DFT metrics could not be recomputed")
    _assert_metric_close(metrics.get("global") or {}, _metric_subset(recomputed["all"]), "global")
    by_ct = metrics.get("by_config_type")
    if not isinstance(by_ct, dict) or not by_ct:
        raise ValueError("config_type metrics are required")
    for key, value in recomputed.items():
        if key == "all":
            continue
        if key not in by_ct:
            raise ValueError(f"missing config_type metrics for {key}")
        _assert_metric_close(by_ct[key], _metric_subset(value), f"config_type {key}")

    domain_fields = metrics.get("domain_fields")
    if not isinstance(domain_fields, dict) or domain_fields.get("config_type") != "present":
        raise ValueError("domain_fields must record config_type as present")

    forbidden = {"student_training", "student_dataset", "student_tuning", "student_recovery"}
    if forbidden & set(payload):
        raise ValueError("reference validation report claims Student-training semantics")
    if payload.get("protected_reference_use") != "teacher_vs_dft_reference_validation_only":
        raise ValueError("protected reference use is not validation-only")
    if payload.get("historical_prediction_policy") != "PROVENANCE_ONLY_NOT_USED_AS_FRESH_RESULT":
        raise ValueError("historical prediction policy is not preserved")

    roles = validate_evidence(manifest_path, payload.get("evidence"), submitted_artifacts,
                              False, allowed_evidence, label="reference validation")
    required_roles = {"teacher_config", "protected_reference_config", "protected_reference_structures", "teacher_reference_predictions"}
    missing = required_roles - roles
    if missing:
        raise ValueError("reference validation evidence missing roles: " + ", ".join(sorted(missing)))
    if submitted_artifacts is not None and pred_path not in {Path(p).resolve() for p in submitted_artifacts}:
        raise ValueError("prediction artifact was not submitted as a stage artifact")
    return payload
