"""Regression tests for the Stage-9 uncertainty_report bounded-evidence adapter.

Bound by the authorized Option-2 Stage-9 evidence-serialization fix (session
2026-08-21). Proves that the adapter surfaces every criterion-relevant VALUE
from the authoritative `uncertainty_report.json` inline in the Judge-facing
bounded packet, that the source artifact is not modified, and that a missing
required field remains an explicit evidence gap.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from runtimes.pydantic_ai.bounded_evidence import (
    _is_uncertainty_report,
    _uncertainty_report_summary,
)


# Framework-generic minimal uncertainty_report fixture. No campaign-, model-, or
# chemistry-specific values; every field the adapter reads is populated with a
# deterministic literal so the tests do not depend on any particular run's data.
def _fixture() -> dict:
    return {
        "schema_version": 1,
        "population": {
            "role": "test_held_out_population",
            "path": "/fake/artifacts/evaluated.extxyz",
            "n_frames": 100,
        },
        "committee_manifest_path": "/fake/artifacts/student_committee.manifest.json",
        "committee_manifest_sha256": "b" * 64,
        "seeds": [1, 2, 3, 4],
        "aggregate": "max",
        "frame_scores": [
            {"frame_id": f"frame-{i:08d}", "u_frame": float(i) / 100.0}
            for i in range(5)
        ],
        "u_frame_summary": {"mean": 0.02, "max": 0.04},
        "calibration": {
            "status": "uncalibrated",
            "caveat": (
                "committee force disagreement is treated as a committee disagreement "
                "/ fidelity-ranking signal only; no calibration evidence has been supplied"
            ),
        },
        "identified_gaps": [],
        "limitations": [],
        "evidence": [
            {"role": "committee_manifest",
             "path": "/fake/artifacts/student_committee.manifest.json",
             "integrity": {"kind": "file", "size": 2048, "sha256": "b" * 64}},
            {"role": "population",
             "path": "/fake/artifacts/evaluated.extxyz",
             "integrity": {"kind": "file", "size": 4096, "sha256": "c" * 64}},
        ],
    }


def test_predicate_recognizes_by_required_key_signature():
    assert _is_uncertainty_report(_fixture()) is True


@pytest.mark.parametrize(
    "missing_key",
    ["schema_version", "committee_manifest_sha256", "seeds", "aggregate",
     "u_frame_summary", "calibration"],
)
def test_predicate_rejects_when_any_required_key_missing(missing_key):
    payload = _fixture()
    payload.pop(missing_key)
    assert _is_uncertainty_report(payload) is False


def test_summary_surfaces_criterion1_population_declaration():
    """Criterion 1 — held-out / deployment-relevant population is declared."""
    summary = _uncertainty_report_summary(_fixture())
    pop = summary["population"]
    assert pop["role"] == "test_held_out_population"
    assert pop["n_frames"] == 100
    assert pop["artifact_sha256"] == "c" * 64  # from evidence.population.integrity
    assert pop["artifact_size"] == 4096


def test_summary_surfaces_criterion2_calibration_status_and_caveat():
    """Criterion 2 — sigma_F is treated as a disagreement / ranking signal unless
    calibration evidence is supplied."""
    summary = _uncertainty_report_summary(_fixture())
    assert summary["calibration"]["status"] == "uncalibrated"
    assert ("committee disagreement" in summary["calibration"]["caveat"] or
            "fidelity-ranking" in summary["calibration"]["caveat"])
    # The `disagreement.aggregate_rule` + per-frame summary must also be present so
    # a Judge can cite the actual statistic, not only its narrative interpretation.
    assert summary["disagreement"]["aggregate_rule"] == "max"
    assert summary["disagreement"]["per_frame_summary"] == {"mean": 0.02, "max": 0.04}


def test_summary_surfaces_criterion3_committee_manifest_hash_and_seeds():
    """Criterion 3 — the report cites the exact Student committee manifest hash."""
    summary = _uncertainty_report_summary(_fixture())
    comm = summary["committee"]
    assert comm["committee_manifest_sha256"] == "b" * 64
    assert comm["committee_manifest_size"] == 2048
    assert comm["seeds"] == [1, 2, 3, 4]
    assert comm["n_seeds"] == 4


def test_summary_surfaces_criterion4_protected_reference_exclusion_state():
    """Criterion 4 — any acquisition or recovery proposal triggered by uncertainty
    continues to respect the protected-reference exclusion policy. The bounded
    packet must let a Judge read whether any such proposal was emitted."""
    summary = _uncertainty_report_summary(_fixture())
    prx = summary["protected_reference_exclusion"]
    assert prx["identified_gaps"] == []
    assert prx["limitations"] == []
    # Population identity is shared between the report's `population.path` and its
    # own `evidence[population].path` when the report is internally consistent.
    assert prx["population_shared_with_stage8_evaluation"] is True


def test_summary_notes_absence_of_percentiles_and_domain_breakdown_explicitly():
    """Do NOT invent statistics that aren't in the report; expose their absence."""
    summary = _uncertainty_report_summary(_fixture())
    assert summary["disagreement"]["per_frame_percentiles_present"] is False
    assert summary["disagreement"]["domain_resolved_present"] is False


def test_summary_does_not_mutate_source_payload():
    """The adapter must not modify the payload it reads."""
    payload = _fixture()
    snapshot = json.dumps(payload, sort_keys=True)
    _ = _uncertainty_report_summary(payload)
    assert json.dumps(payload, sort_keys=True) == snapshot


def test_summary_never_invents_a_field_absent_from_the_report():
    """Missing keys/blocks yield None (never a fabricated value)."""
    payload = _fixture()
    payload["population"] = {}       # empty population block
    payload["evidence"] = []          # empty evidence list
    payload["calibration"] = {}       # empty calibration block
    payload["u_frame_summary"] = {}   # empty summary
    summary = _uncertainty_report_summary(payload)
    assert summary["population"]["role"] is None
    assert summary["population"]["n_frames"] is None
    assert summary["population"]["artifact_sha256"] is None
    assert summary["calibration"]["status"] is None
    assert summary["calibration"]["caveat"] is None
    assert summary["disagreement"]["per_frame_summary"] == {"mean": None, "max": None}


def test_summary_reaches_the_bounded_json_summary_that_feeds_the_judge_packet():
    """End-to-end: the adapter fires from `_json_summary` and its output is nested
    under the `uncertainty_report` key (the registered `summary_key`)."""
    from runtimes.pydantic_ai.bounded_evidence import _json_summary
    payload = _fixture()
    p = Path("/tmp/_test_uncertainty_report_fixture.json")
    p.write_text(json.dumps(payload))
    summary = _json_summary(p)
    assert "uncertainty_report" in summary, (
        "the `uncertainty_report` adapter must nest its output under its registered "
        f"summary_key; got summary keys={sorted(summary.keys())}")
    us = summary["uncertainty_report"]
    assert us["committee"]["committee_manifest_sha256"] == "b" * 64
    assert us["calibration"]["status"] == "uncalibrated"


# ---------- Live-artifact-untouched fixture ------------------------------------------------
# The uncertainty_report artifact this recovery re-gates against must be byte-identical
# to its pre-recovery pin.

_CANONICAL_STAGE9_ARTIFACT_SHA = {
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/uncertainty_report.json":
        "a46f763d4ccc35effd7823d398e60f8236202e501953d9123b1e94c98f38ed97",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relpath, expected_sha", sorted(_CANONICAL_STAGE9_ARTIFACT_SHA.items()))
def test_stage9_uncertainty_report_untouched_by_evidence_fix(relpath, expected_sha):
    from workflow.integrity import sha256_file
    path = _project_root() / relpath
    if not path.is_file():
        pytest.skip(f"Stage-9 artifact not present in this checkout: {relpath}")
    assert sha256_file(path) == expected_sha, (
        f"{relpath!r} sha256 drifted from pre-Option-2-fix pin — the Stage-9 evidence "
        "serialization recovery must NOT modify the uncertainty_report artifact")
