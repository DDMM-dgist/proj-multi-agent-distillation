"""Framework V2 closure — Judge reproducibility L1 + L2 (directive Section AG).

L1 = one Judge attempt is re-runnable from its provenance (packet SHA, decision
SHA, sampling temperature + seed all pinned). L2 = the three mutually-blind
lenses form a committee that is each L1-reproducible AND provably reasoned over
the identical packet/decision bytes.

Tests run against both a plain-dict provenance shape and the real
RuntimeInvocationRecord so the auditor is verified against the actual on-disk
record it will consume.
"""
from __future__ import annotations

from framework_v2.judge_reproducibility import (
    verify_l1,
    verify_l2,
    L1_REQUIRED_FIELDS,
)
from framework_v2.review_spec import CANONICAL_LENS_IDS
from runtimes.pydantic_ai.models import RuntimeInvocationRecord


def _rec(**over):
    base = dict(packet_sha256="pkt", decision_sha256="dec",
                temperature=0.0, seed=7)
    base.update(over)
    return base


def _provenance(**over):
    """A minimal-but-valid RuntimeInvocationRecord carrying the repro fields."""
    base = dict(
        attempt_id="a", task_id="t", agent="judge", provider="mock",
        model_id="mock", runtime_version="v", prompt_sha256="p",
        tool_manifest_sha256="tm", raw_response="{}",
        packet_sha256="pkt", decision_sha256="dec", temperature=0.0, seed=7)
    base.update(over)
    return RuntimeInvocationRecord(**base)


# --- L1 ---------------------------------------------------------------------
def test_l1_reproducible_when_all_fields_pinned():
    assert verify_l1(_rec()).reproducible


def test_l1_reproducible_on_real_provenance_record():
    r = verify_l1(_provenance())
    assert r.reproducible and r.packet_sha256 == "pkt" and r.seed == 7


def test_l1_fails_when_seed_missing():
    r = verify_l1(_rec(seed=None))
    assert not r.reproducible and "seed" in r.missing_fields


def test_l1_fails_when_temperature_missing():
    r = verify_l1(_rec(temperature=None))
    assert not r.reproducible and "temperature" in r.missing_fields


def test_l1_fails_when_packet_sha_absent():
    r = verify_l1(_rec(packet_sha256=None))
    assert not r.reproducible and "packet_sha256" in r.missing_fields


def test_l1_fails_when_packet_sha_blank():
    r = verify_l1(_rec(packet_sha256="   "))
    assert not r.reproducible and "packet_sha256" in r.missing_fields


def test_l1_required_fields_are_the_four_repro_fields():
    assert set(L1_REQUIRED_FIELDS) == {
        "packet_sha256", "decision_sha256", "temperature", "seed"}


# --- L2 ---------------------------------------------------------------------
def _committee(**per_lens_over):
    """Three lens records sharing packet/decision by default."""
    recs = {lens: _rec() for lens in CANONICAL_LENS_IDS}
    for lens, over in per_lens_over.items():
        recs[lens] = _rec(**over)
    return recs


def test_l2_reproducible_when_identical_and_all_l1():
    r = verify_l2(_committee(), expected_lens_ids=CANONICAL_LENS_IDS)
    assert r.reproducible
    assert r.shared_packet_sha256 == "pkt"
    assert r.shared_decision_sha256 == "dec"
    assert all(r.per_lens_l1.values())


def test_l2_fails_on_divergent_packet_sha():
    recs = _committee()
    recs["scientific_validity"] = _rec(packet_sha256="DIFFERENT")
    r = verify_l2(recs, expected_lens_ids=CANONICAL_LENS_IDS)
    assert not r.reproducible
    assert any("different packet_sha256" in e for e in r.errors)


def test_l2_fails_on_divergent_decision_sha():
    recs = _committee()
    recs["evidence_provenance"] = _rec(decision_sha256="OTHER")
    r = verify_l2(recs, expected_lens_ids=CANONICAL_LENS_IDS)
    assert not r.reproducible
    assert any("different decision_sha256" in e for e in r.errors)


def test_l2_fails_when_a_lens_not_l1_reproducible():
    recs = _committee()
    recs["reproducibility_deployment"] = _rec(seed=None)
    r = verify_l2(recs, expected_lens_ids=CANONICAL_LENS_IDS)
    assert not r.reproducible
    assert r.per_lens_l1["reproducibility_deployment"] is False


def test_l2_fails_when_not_three_lenses():
    recs = {"scientific_validity": _rec(), "evidence_provenance": _rec()}
    r = verify_l2(recs)
    assert not r.reproducible
    assert any("exactly three" in e for e in r.errors)


def test_l2_fails_when_lens_ids_unexpected():
    recs = {"a": _rec(), "b": _rec(), "c": _rec()}
    r = verify_l2(recs, expected_lens_ids=CANONICAL_LENS_IDS)
    assert not r.reproducible
    assert any("do not match expected" in e for e in r.errors)


def test_l2_reproducible_over_real_provenance_records():
    recs = {lens: _provenance() for lens in CANONICAL_LENS_IDS}
    r = verify_l2(recs, expected_lens_ids=CANONICAL_LENS_IDS)
    assert r.reproducible
