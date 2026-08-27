"""V2-H06: one authoritative training-eligibility invariant.

Protected evaluation/test data may never enter Student training, recovery
candidates, replay, or augmentation parents.  Eligibility is derived from the
frozen hash-pinned authoritative split; a candidate that is protected, carries a
non-TRAIN role, or is absent from the split fails closed.
"""
import pytest

from framework_v2.contracts import PartitionRole
from framework_v2.protected_eligibility import (
    EligibilityCheckResult,
    derive_training_eligible_candidates,
)


def _roles(**kw):
    return {k: PartitionRole(v) for k, v in kw.items()}


def test_train_role_candidates_are_eligible():
    result = derive_training_eligible_candidates(
        candidate_ids=["c1", "c2"],
        candidate_roles=_roles(c1="TRAIN", c2="TRAIN"),
        protected_ids=set(),
        training_split_sha256="split",
        expected_training_split_sha256="split",
        protected_evidence_sha256="prot",
        expected_protected_evidence_sha256="prot",
    )
    assert isinstance(result, EligibilityCheckResult)
    assert result.eligible_candidate_ids == ["c1", "c2"]
    assert result.rejected_candidate_ids == {}


def test_protected_candidate_rejected():
    with pytest.raises(ValueError, match="protected"):
        derive_training_eligible_candidates(
            candidate_ids=["c1", "p1"],
            candidate_roles=_roles(c1="TRAIN", p1="TRAIN"),
            protected_ids={"p1"},
            training_split_sha256="split",
            expected_training_split_sha256="split",
            protected_evidence_sha256="prot",
            expected_protected_evidence_sha256="prot",
        )


def test_non_train_role_rejected():
    with pytest.raises(ValueError, match="non-training role"):
        derive_training_eligible_candidates(
            candidate_ids=["c1", "v1"],
            candidate_roles=_roles(c1="TRAIN", v1="VALIDATION"),
            protected_ids=set(),
            training_split_sha256="split",
            expected_training_split_sha256="split",
            protected_evidence_sha256="prot",
            expected_protected_evidence_sha256="prot",
        )


def test_candidate_missing_from_split_fails_closed():
    with pytest.raises(ValueError, match="missing from authoritative split"):
        derive_training_eligible_candidates(
            candidate_ids=["c1", "unknown"],
            candidate_roles=_roles(c1="TRAIN"),
            protected_ids=set(),
            training_split_sha256="split",
            expected_training_split_sha256="split",
            protected_evidence_sha256="prot",
            expected_protected_evidence_sha256="prot",
        )


def test_split_hash_mismatch_fails():
    with pytest.raises(ValueError, match="training split hash mismatch"):
        derive_training_eligible_candidates(
            candidate_ids=["c1"],
            candidate_roles=_roles(c1="TRAIN"),
            protected_ids=set(),
            training_split_sha256="drifted",
            expected_training_split_sha256="split",
            protected_evidence_sha256="prot",
            expected_protected_evidence_sha256="prot",
        )


def test_protected_evidence_hash_mismatch_fails():
    with pytest.raises(ValueError, match="protected evidence hash mismatch"):
        derive_training_eligible_candidates(
            candidate_ids=["c1"],
            candidate_roles=_roles(c1="TRAIN"),
            protected_ids=set(),
            training_split_sha256="split",
            expected_training_split_sha256="split",
            protected_evidence_sha256="drifted",
            expected_protected_evidence_sha256="prot",
        )
