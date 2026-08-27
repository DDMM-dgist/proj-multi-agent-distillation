"""One authoritative training-eligibility invariant for V2.

Recovery candidate selection, DFT replay, supercell/augmentation parents, and
final training admission must all pass through the *same* check: a candidate is
training-eligible only if it is not protected and carries the TRAIN split role in
the frozen, hash-pinned authoritative split.  Protected evaluation/test data can
never enter Student training, recovery candidates, replay, or augmentation
parents.  The check fails closed: any ineligible candidate raises rather than
being silently dropped.
"""
from __future__ import annotations

from typing import Mapping

from pydantic import Field

from framework_v2.contracts import ContractBase, PartitionRole


class EligibilityCheckResult(ContractBase):
    eligible_candidate_ids: list[str]
    rejected_candidate_ids: dict[str, str] = Field(default_factory=dict)
    training_split_sha256: str
    protected_evidence_sha256: str


def derive_training_eligible_candidates(
    *,
    candidate_ids: list[str],
    candidate_roles: Mapping[str, PartitionRole],
    protected_ids: set[str],
    training_split_sha256: str,
    expected_training_split_sha256: str,
    protected_evidence_sha256: str,
    expected_protected_evidence_sha256: str,
    train_role: PartitionRole = PartitionRole.TRAIN,
) -> EligibilityCheckResult:
    if training_split_sha256 != expected_training_split_sha256:
        raise ValueError("training split hash mismatch")
    if protected_evidence_sha256 != expected_protected_evidence_sha256:
        raise ValueError("protected evidence hash mismatch")

    eligible: list[str] = []
    rejected: dict[str, str] = {}
    for cid in candidate_ids:
        if cid in protected_ids:
            rejected[cid] = "protected"
            continue
        role = candidate_roles.get(cid)
        if role is None:
            raise ValueError(f"candidate {cid!r} missing from authoritative split")
        if role != train_role:
            rejected[cid] = f"non-training role {role.value}"
            continue
        eligible.append(cid)

    if rejected:
        raise ValueError(
            "ineligible candidates: "
            + ", ".join(f"{k}:{v}" for k, v in sorted(rejected.items()))
        )
    return EligibilityCheckResult(
        eligible_candidate_ids=eligible,
        rejected_candidate_ids=rejected,
        training_split_sha256=training_split_sha256,
        protected_evidence_sha256=protected_evidence_sha256,
    )


__all__ = [
    "EligibilityCheckResult",
    "derive_training_eligible_candidates",
]
