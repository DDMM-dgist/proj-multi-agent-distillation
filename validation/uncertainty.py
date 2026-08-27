"""Deterministic contract for committee-disagreement (uncertainty) reports.

This is the smallest structural contract needed to validate the shape and provenance of a
committee force-disagreement report before Judge review -- it does not invent a calibrated-
uncertainty claim. ``calibration.status`` defaults to (and, absent calibration evidence, must
be) "uncalibrated": sigma_F is a committee disagreement / fidelity-ranking signal only, per
``adapters.uncertainty.committee_force_std``'s own docstring, unless genuine calibration
evidence is supplied and cited.
"""
import json
import math
from pathlib import Path

from validation.report import validate_evidence
from workflow.integrity import sha256_file

CALIBRATION_STATUSES = {"uncalibrated", "calibrated"}
CALIBRATION_DECISIONS = {"PASS", "FAIL", "HUMAN_SCIENTIFIC_INPUT_REQUIRED"}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _nonnegative_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_uncertainty_report(manifest_path, submitted_artifacts=None, allowed_evidence=None,
                                enforce_required_pass=False):
    """Validate a committee force-disagreement report's shape and provenance.

    Requires: a declared population (role/path/n_frames), a hash-verified pointer to the exact
    Student committee manifest actually used (never an asserted hash), at least two committee
    seeds, a non-empty per-frame score list whose mean/max match the declared summary, and an
    explicit calibration status that must carry a caveat unless real calibration evidence is
    cited.
    """
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("uncertainty report requires schema_version=1")

    population = payload.get("population")
    if not isinstance(population, dict):
        raise ValueError("uncertainty report requires a population block")
    for field in ("role", "path"):
        if not isinstance(population.get(field), str) or not population[field].strip():
            raise ValueError(f"uncertainty report population.{field} must be a non-empty string")
    if not _nonnegative_integer(population.get("n_frames")):
        raise ValueError("uncertainty report population.n_frames must be a non-negative integer")

    manifest_value = payload.get("committee_manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("uncertainty report requires committee_manifest_path")
    committee_path = Path(manifest_value).expanduser()
    committee_path = (committee_path.resolve() if committee_path.is_absolute() else
                      (manifest_path.parent / committee_path).resolve())
    if not committee_path.is_file():
        raise ValueError(f"uncertainty report committee_manifest_path does not exist: {committee_path}")
    recorded_hash = payload.get("committee_manifest_sha256")
    if not isinstance(recorded_hash, str) or not recorded_hash.strip():
        raise ValueError("uncertainty report requires committee_manifest_sha256")
    if sha256_file(committee_path) != recorded_hash:
        raise ValueError(
            "uncertainty report committee_manifest_sha256 does not match the actual "
            f"committee_manifest_path file ({committee_path}); the report must cite the exact "
            "Student committee manifest hash, never an asserted one"
        )

    seeds = payload.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) < 2 or
            any(not isinstance(s, int) or isinstance(s, bool) for s in seeds)):
        raise ValueError("uncertainty report requires at least two integer committee seeds")

    frame_scores = payload.get("frame_scores")
    if not isinstance(frame_scores, list) or not frame_scores:
        raise ValueError("uncertainty report requires a non-empty frame_scores list")
    seen_frames = set()
    u_values = []
    for index, entry in enumerate(frame_scores):
        if not isinstance(entry, dict):
            raise ValueError(f"frame_scores[{index}] must be an object")
        frame_id = entry.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError(f"frame_scores[{index}].frame_id must be a non-empty string")
        if frame_id in seen_frames:
            raise ValueError(f"frame_scores frame_id is duplicated: {frame_id}")
        seen_frames.add(frame_id)
        u_frame = entry.get("u_frame")
        if not _finite(u_frame) or u_frame < 0:
            raise ValueError(f"frame_scores[{index}].u_frame must be a non-negative finite number")
        u_values.append(float(u_frame))

    summary = payload.get("u_frame_summary")
    if not isinstance(summary, dict):
        raise ValueError("uncertainty report requires u_frame_summary")
    for field in ("mean", "max"):
        if not _finite(summary.get(field)):
            raise ValueError(f"uncertainty report u_frame_summary.{field} must be finite")
    if not math.isclose(summary["mean"], sum(u_values) / len(u_values), rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("uncertainty report u_frame_summary.mean is inconsistent with frame_scores")
    if not math.isclose(summary["max"], max(u_values), rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("uncertainty report u_frame_summary.max is inconsistent with frame_scores")

    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("status") not in CALIBRATION_STATUSES:
        raise ValueError(
            "uncertainty report requires calibration.status in " + str(sorted(CALIBRATION_STATUSES))
        )
    if calibration["status"] != "calibrated":
        caveat = calibration.get("caveat")
        if not isinstance(caveat, str) or not caveat.strip():
            raise ValueError(
                "uncertainty report calibration requires a non-empty caveat unless status is "
                "'calibrated' -- committee force disagreement must stay disclosed as a ranking/"
                "fidelity signal, not silently presented as calibrated uncertainty"
            )
    else:
        for field in ("quantity", "nominal_coverage", "fit", "eval", "conformal"):
            if field not in calibration:
                raise ValueError(f"calibrated uncertainty report requires calibration.{field}")
        nominal = calibration.get("nominal_coverage")
        if not _finite(nominal) or not (0.0 < float(nominal) < 1.0):
            raise ValueError("calibration.nominal_coverage must be a finite probability")
        quantity = calibration.get("quantity")
        if not isinstance(quantity, str) or not quantity.strip():
            raise ValueError("calibration.quantity must name the calibrated quantity")
        for block_name in ("fit", "eval"):
            block = calibration.get(block_name)
            if not isinstance(block, dict):
                raise ValueError(f"calibration.{block_name} must be an object")
            if not _nonnegative_integer(block.get("n_frames")) or block["n_frames"] <= 0:
                raise ValueError(f"calibration.{block_name}.n_frames must be positive")
            if not isinstance(block.get("path"), str) or not block["path"].strip():
                raise ValueError(f"calibration.{block_name}.path is required")
            if not isinstance(block.get("frame_fingerprints_sha256"), str) or not block["frame_fingerprints_sha256"].strip():
                raise ValueError(f"calibration.{block_name}.frame_fingerprints_sha256 is required")
        if calibration["fit"]["frame_fingerprints_sha256"] == calibration["eval"]["frame_fingerprints_sha256"]:
            raise ValueError("calibration FIT and EVAL fingerprints must be disjoint")
        conformal = calibration.get("conformal")
        if not isinstance(conformal, dict):
            raise ValueError("calibration.conformal must be an object")
        for field in ("qhat", "epsilon", "score_count"):
            if not _finite(conformal.get(field)) or float(conformal[field]) < 0:
                raise ValueError(f"calibration.conformal.{field} must be non-negative finite")
        coverage = calibration.get("coverage_eval")
        if not isinstance(coverage, dict):
            raise ValueError("calibrated uncertainty report requires calibration.coverage_eval")
        for field in ("observed_coverage", "covered_count", "total_count"):
            if field not in coverage:
                raise ValueError(f"calibration.coverage_eval.{field} is required")
        if not _finite(coverage["observed_coverage"]) or not (0.0 <= float(coverage["observed_coverage"]) <= 1.0):
            raise ValueError("calibration.coverage_eval.observed_coverage must be a probability")
        if not _nonnegative_integer(coverage["covered_count"]) or not _nonnegative_integer(coverage["total_count"]):
            raise ValueError("calibration.coverage_eval counts must be non-negative integers")
        if coverage["total_count"] <= 0 or coverage["covered_count"] > coverage["total_count"]:
            raise ValueError("calibration.coverage_eval counts are inconsistent")
        decision = calibration.get("decision")
        if decision not in CALIBRATION_DECISIONS:
            raise ValueError("calibration.decision must be one of " + str(sorted(CALIBRATION_DECISIONS)))
        if decision == "PASS" and not isinstance(calibration.get("coverage_acceptance"), dict):
            raise ValueError("calibration cannot PASS without bound coverage_acceptance parameters")

    validate_evidence(manifest_path, payload.get("evidence"), submitted_artifacts, False,
                      allowed_evidence, label="uncertainty")
    for field in ("identified_gaps", "limitations"):
        value = payload.get(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"uncertainty report {field} must be a list of strings")
    return payload
