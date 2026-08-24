"""Regression tests for the Stage-10 versioned gate-criteria rebound
(deployment_md_gate_spec v1 → v2 = C2a/C2b/C2c). Session 2026-08-21.

Guarantees:
  * the ACTIVE gate criteria in the C12F run contain C2a/C2b/C2c and no longer
    contain the original v1 C2 line as an active criterion;
  * the original v1 C2 text remains recoverable from the immutable audit history;
  * the gate-spec version identity is visible on the state.stages record;
  * a ``gate_criteria_rebound`` event with root_cause=
    ``CASE_B_stage_review_contract_semantics`` exists;
  * ``global_domain_claim_supported=false`` is NOT interpreted as a Stage-10
    execution failure;
  * no numerical T/P bounds were introduced by this rebound;
  * every Stage-10 scientific artifact remains byte-identical.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest


_C12F_RUN = Path(__file__).resolve().parents[1] / "runs" / "sio2-sox-allegro-simplenn-c12f"

# The original C2 v1 text — byte-for-byte immutable audit reference.
ORIGINAL_C2_V1 = "MD run stayed inside the frozen deployment domain (composition, T, P)"


def _load_manifest():
    if not (_C12F_RUN / "manifest.json").is_file():
        pytest.skip("C12F run not present")
    return json.loads((_C12F_RUN / "manifest.json").read_text())


def _deployment_stage(st):
    for s in st.get("stages", []):
        if s.get("name") == "deployment_md":
            return s
    raise AssertionError("deployment_md stage missing")


def _rebound_event(st):
    for ev in st.get("events", []):
        if (ev.get("type") == "gate_criteria_rebound"
                and ev.get("stage") == "deployment_md"
                and ev.get("new_gate_spec_identity") == "deployment_md_gate_spec_v2"):
            return ev
    return None


# ------------------------------------------------------------------
# ACTIVE gate now contains C2a/b/c and no longer contains the raw v1 C2
# ------------------------------------------------------------------

def test_active_gate_criteria_contain_c2a_c2b_c2c():
    st = _load_manifest()
    stage = _deployment_stage(st)
    criteria = stage.get("gate_criteria") or []
    joined = "\n".join(criteria)
    assert "C2a — composition scope" in joined
    assert "C2b — pre-approved deployment point" in joined
    assert "C2c — ensemble-aware pressure semantics" in joined


def test_active_gate_criteria_no_longer_ask_v1_c2_as_active_criterion():
    st = _load_manifest()
    stage = _deployment_stage(st)
    criteria = stage.get("gate_criteria") or []
    # No active criterion should equal the v1 C2 line VERBATIM.
    assert ORIGINAL_C2_V1 not in criteria, (
        f"v1 C2 is still an active criterion — versioned rebound incomplete: {criteria!r}")


def test_active_gate_criteria_count_is_six_not_four():
    st = _load_manifest()
    stage = _deployment_stage(st)
    assert len(stage.get("gate_criteria") or []) == 6, (
        "expected 6 active criteria: C1, C2a, C2b, C2c, C3=required_evidence, C4=approval")


# ------------------------------------------------------------------
# v1 C2 text recoverable from immutable audit history
# ------------------------------------------------------------------

def test_v1_c2_text_preserved_verbatim_in_stage_metadata():
    st = _load_manifest()
    stage = _deployment_stage(st)
    assert stage.get("gate_spec_v1_c2_text_immutable_audit") == ORIGINAL_C2_V1


def test_v1_c2_text_preserved_verbatim_in_rebound_event():
    st = _load_manifest()
    ev = _rebound_event(st)
    assert ev is not None, "gate_criteria_rebound event for deployment_md v1→v2 must exist"
    assert ev.get("original_c2_v1_text_preserved_verbatim") == ORIGINAL_C2_V1
    # And the old_gate_criteria snapshot must contain the v1 C2 line.
    assert ORIGINAL_C2_V1 in (ev.get("old_gate_criteria") or [])


# ------------------------------------------------------------------
# Gate-spec version identity visible on the stage record
# ------------------------------------------------------------------

def test_gate_spec_version_identity_present_on_stage_record():
    st = _load_manifest()
    stage = _deployment_stage(st)
    assert stage.get("gate_spec_version") == "deployment_md_gate_spec_v2"
    assert stage.get("gate_spec_previous_version") == "deployment_md_gate_spec_v1"


def test_rebound_event_carries_case_b_root_cause():
    st = _load_manifest()
    ev = _rebound_event(st)
    assert ev.get("root_cause") == "CASE_B_stage_review_contract_semantics"


def test_rebound_event_references_recovery_id_6():
    st = _load_manifest()
    ev = _rebound_event(st)
    ref = ev.get("recovery_reference") or {}
    assert ref.get("recovery_id") == 6
    assert ref.get("canonical_plan_hash") == \
        "98c01c834601c48a420819a2c40761a5dd0d22f2fd79ebc97c1578fdb3ac7d23"


def test_rebound_event_declares_no_scientific_result_change():
    st = _load_manifest()
    ev = _rebound_event(st)
    decl = ev.get("scientific_result_change_declaration") or ""
    assert "No scientific result" in decl
    assert "trajectory" in decl and "model" in decl


def test_rebound_event_asserts_no_new_numerical_bounds():
    st = _load_manifest()
    ev = _rebound_event(st)
    assert ev.get("no_new_numerical_bounds_introduced") is True


def test_rebound_event_is_not_a_result_override():
    st = _load_manifest()
    ev = _rebound_event(st)
    note = ev.get("not_a_result_override") or ""
    assert "not a human adjudication" in note.lower() or "distinct prospective" in note
    # And REVISE outcomes stand.
    assert "REMAIN VALID" in note or "stand as-is" in note or "REVISEs against v1" in note


# ------------------------------------------------------------------
# C2c interpretation semantic — NOT_EVALUABLE ≠ FAIL
# ------------------------------------------------------------------

def test_c2c_criterion_text_explicitly_defines_not_evaluable_as_not_a_failure():
    st = _load_manifest()
    stage = _deployment_stage(st)
    joined = "\n".join(stage.get("gate_criteria") or [])
    assert "NOT_EVALUABLE" in joined
    assert "NOT_EVALUABLE is not a failure" in joined
    # NVT pressure is explicitly not a controlled setpoint per C2c
    assert "not a controlled setpoint" in joined


# ------------------------------------------------------------------
# No numerical T/P bounds introduced anywhere in the rebound
# ------------------------------------------------------------------

def test_no_numerical_temperature_or_pressure_bounds_added_to_gate_criteria():
    st = _load_manifest()
    stage = _deployment_stage(st)
    text = "\n".join(stage.get("gate_criteria") or [])
    # The rebound must not have introduced numerical T/P envelope bounds.
    import re
    # A "numerical envelope" would look like e.g. "T_min:", "K bound", "pressure_GPa: 5",
    # etc. Reject any pattern like <number> K / GPa / bar / Pa used as a threshold.
    # (Explicit exception for 300 K temperature_setpoint value which is the pre-approved
    # POINT, not a bound.)
    bounds_pattern = re.compile(
        r"(temperature.*(?:<|<=|>|>=|min|max)\s*[0-9]|"
        r"pressure.*(?:<|<=|>|>=|min|max)\s*[0-9]|"
        r"[0-9]+\s*(?:GPa|bar)\s*(?:min|max|<|>))"
    )
    assert not bounds_pattern.search(text), (
        f"rebound leaked a numerical T/P bound into the active gate criteria: {text!r}")


# ------------------------------------------------------------------
# Scientific-artifact byte-identity preserved
# ------------------------------------------------------------------

_CANONICAL_STAGE10_SHAS = {
    "artifacts/md.manifest.json":
        "6541c3a1da04e038b3cbb05b0b9c36efda8b05806bcb941887a2660a2f7c46a0",
    "artifacts/deployment_md/trajectory.dump":
        "6eec4a0e90bc4c63ad2def8b081c0b1fdbec3e8358186a58bff7045d77988a4d",
    "artifacts/deployment_md/thermo.log":
        "3ed87bcec0beaea44726de04f90c0a38730101a2059c58ab35954d421c0983cc",
    "artifacts/deployment_md/input.lmp":
        "63e3438068ad26a04a15abcef02d3fdeb33afbe74eef291608eb1707c743aa53",
    "artifacts/deployment_md/context.yaml":
        "af0bc999434bf66c242d131cf38818d55a560e7ecf739929a54b90b5eb3d4931",
    "artifacts/deployment_md/deployment_provenance.json":
        "6cae634f29fd2599d537a208dd6be7cf0fd6bbf9c4a553c7b43430adb2b3302c",
}


@pytest.mark.parametrize("relpath, expected", sorted(_CANONICAL_STAGE10_SHAS.items()))
def test_stage10_artifacts_untouched_by_v2_rebound(relpath, expected):
    from workflow.integrity import sha256_file
    path = _C12F_RUN / relpath
    if not path.is_file():
        pytest.skip(f"Stage-10 artifact not present: {relpath}")
    assert sha256_file(path) == expected, (
        f"{relpath!r} sha256 drifted — the versioned gate-criteria rebound must NOT "
        "modify any Stage-10 scientific artifact")
