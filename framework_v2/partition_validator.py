"""Framework V2 -- dataset-partition validator (Section 8, CASE B).

R31 could produce a train/validation/blind-test split that was *feasible*
(no parent-family leakage across partitions) yet *scientifically
unrepresentative* (a major deployment regime absent from the training
partition). The framework had no capability that could distinguish those
two things, so an unrepresentative-but-leakage-free split passed silently.

This module makes the two requirements independent and both mandatory:

  1. LINEAGE SAFETY -- no ``lineage_key`` value may appear in more than one
     partition role. A violation is fail-closed: the split is invalid and
     cannot be used, regardless of representativeness.

  2. REPRESENTATIVENESS -- every regime the ``DomainRepresentation`` marks
     as PRIMARY_DEPLOYMENT ("a major deployment regime") must be present in
     each required partition role (TRAIN by default). A major deployment
     regime absent from a required role, even under zero leakage, yields
     ``REVISE_SPLIT`` -- the exact R31 CASE-B failure the framework now
     refuses to wave through.

The validator produces ``DeterministicFact`` records (Section 13) so the
outcome is authoritative and an LLM Judge cannot negate it, and a single
top-level ``verdict`` in {``PASS_SPLIT``, ``REVISE_SPLIT``,
``LINEAGE_LEAKAGE``}. Nothing here is element-, composition-, or
campaign-specific: the regimes come from the generic ``DomainRepresentation``
and the roles come from the ``DatasetPartitionPlan``.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

from framework_v2.contracts import (
    DatasetPartitionPlan,
    DomainRepresentation,
    PartitionRole,
    ScopeCategory,
)
from framework_v2.facts import DeterministicFact, FactVerdict

_VALIDATOR = "framework_v2.partition_validator.validate_partition"

PASS_SPLIT = "PASS_SPLIT"
REVISE_SPLIT = "REVISE_SPLIT"
LINEAGE_LEAKAGE = "LINEAGE_LEAKAGE"


@dataclasses.dataclass(frozen=True)
class PartitionedItem:
    """One assigned dataset item.

    ``lineage_key`` is the value of the plan's ``lineage_key`` variable for
    this item (e.g. its parent-structure identity) -- the thing that must not
    straddle partitions. ``regime_ids`` are the ``DomainRegime.regime_id``
    values this item belongs to (an item may belong to more than one regime).
    """
    item_id: str
    lineage_key: str
    role: PartitionRole
    regime_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class PartitionValidationReport:
    verdict: str
    facts: tuple[DeterministicFact, ...]
    lineage_violations: tuple[dict, ...]
    missing_regime_coverage: tuple[dict, ...]
    role_counts: dict
    plan_sha256: str
    representation_sha256: str
    required_roles: tuple[str, ...]

    def gate_ok(self) -> bool:
        """A partition may be accepted only on ``PASS_SPLIT``."""
        return self.verdict == PASS_SPLIT

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "profile": "dataset_partition_validation_report",
            "verdict": self.verdict,
            "gate_ok": self.gate_ok(),
            "required_roles": list(self.required_roles),
            "role_counts": self.role_counts,
            "lineage_violations": [dict(v) for v in self.lineage_violations],
            "missing_regime_coverage": [dict(m) for m in self.missing_regime_coverage],
            "plan_sha256": self.plan_sha256,
            "representation_sha256": self.representation_sha256,
            "facts": [f.model_dump(mode="json") for f in self.facts],
        }


def _fact(fact_id: str, kind: str, observed, expected, verdict: FactVerdict,
          rationale: str) -> DeterministicFact:
    return DeterministicFact(
        fact_id=fact_id,
        kind=kind,
        observed=observed,
        expected=expected,
        verdict=verdict,
        validator=_VALIDATOR,
        rationale=rationale,
    )


def validate_partition(
    *,
    plan: DatasetPartitionPlan,
    items: Sequence[PartitionedItem],
    representation: DomainRepresentation,
    required_roles: Sequence[PartitionRole] = (PartitionRole.TRAIN,),
) -> PartitionValidationReport:
    """Validate an actual split against its plan and the discovered domain.

    ``items`` is the realized assignment (one record per dataset item).
    ``representation`` supplies the regimes; those overlapping
    PRIMARY_DEPLOYMENT are the "major deployment regimes" that must appear in
    every role listed in ``required_roles``.

    Binding: the plan must reference the same scope as the representation.
    If ``plan.scope_contract_sha256`` differs from
    ``representation.linked_scope_contract_sha256`` the validator fails closed
    with ``LINEAGE_LEAKAGE``-severity mismatch fact rather than silently
    validating an unrelated pairing.
    """
    plan_sha = plan.content_sha256()
    rep_sha = representation.content_sha256()
    required = tuple(required_roles)
    facts: list[DeterministicFact] = []

    # --- 0. scope binding consistency (fail closed on mismatch) --------------
    scope_consistent = (
        plan.scope_contract_sha256 == representation.linked_scope_contract_sha256
    )
    facts.append(_fact(
        fact_id=f"partition-scope-binding-{plan_sha[:12]}",
        kind="partition_scope_binding_consistent",
        observed={
            "plan_scope_sha256": plan.scope_contract_sha256,
            "representation_scope_sha256": representation.linked_scope_contract_sha256,
        },
        expected="equal",
        verdict=FactVerdict.PASS if scope_consistent else FactVerdict.FAIL,
        rationale=("the partition plan and the domain representation must be "
                   "bound to the same DeploymentScopeContract"),
    ))

    # --- role tallies --------------------------------------------------------
    role_counts: dict[str, int] = {}
    for it in items:
        role_counts[it.role.value] = role_counts.get(it.role.value, 0) + 1

    # --- 1. lineage safety ---------------------------------------------------
    lineage_roles: dict[str, set] = {}
    for it in items:
        lineage_roles.setdefault(it.lineage_key, set()).add(it.role.value)
    lineage_violations = tuple(
        {"lineage_key": key, "roles": sorted(roles)}
        for key, roles in sorted(lineage_roles.items())
        if len(roles) > 1
    )
    facts.append(_fact(
        fact_id=f"partition-lineage-safe-{plan_sha[:12]}",
        kind="partition_lineage_leakage",
        observed=len(lineage_violations),
        expected=0,
        verdict=FactVerdict.PASS if not lineage_violations else FactVerdict.FAIL,
        rationale=(f"lineage_key={plan.lineage_key!r} must not appear in more "
                   "than one partition role"),
    ))

    # --- 2. representativeness of major (PRIMARY_DEPLOYMENT) regimes ---------
    primary_regimes = [
        r for r in representation.regimes
        if ScopeCategory.PRIMARY_DEPLOYMENT in r.within_scope_categories
    ]
    # regime_id -> set of roles it appears in
    regime_roles: dict[str, set] = {}
    for it in items:
        for rid in it.regime_ids:
            regime_roles.setdefault(rid, set()).add(it.role.value)

    missing: list[dict] = []
    for regime in primary_regimes:
        present_in = regime_roles.get(regime.regime_id, set())
        absent_from = [role.value for role in required if role.value not in present_in]
        if absent_from:
            missing.append({
                "regime_id": regime.regime_id,
                "label": regime.label,
                "absent_from_roles": absent_from,
                "present_in_roles": sorted(present_in),
            })
    facts.append(_fact(
        fact_id=f"partition-representative-{plan_sha[:12]}",
        kind="partition_major_regime_coverage",
        observed={
            "n_primary_regimes": len(primary_regimes),
            "n_missing": len(missing),
            "missing": missing,
        },
        expected={"n_missing": 0},
        verdict=FactVerdict.PASS if not missing else FactVerdict.FAIL,
        rationale=("every PRIMARY_DEPLOYMENT regime must be present in each "
                   f"required partition role {[r.value for r in required]}; "
                   f"requirement={plan.representativeness_requirement!r}"),
    ))

    # --- verdict -------------------------------------------------------------
    if not scope_consistent or lineage_violations:
        verdict = LINEAGE_LEAKAGE
    elif missing:
        verdict = REVISE_SPLIT
    else:
        verdict = PASS_SPLIT

    return PartitionValidationReport(
        verdict=verdict,
        facts=tuple(facts),
        lineage_violations=lineage_violations,
        missing_regime_coverage=tuple(missing),
        role_counts=role_counts,
        plan_sha256=plan_sha,
        representation_sha256=rep_sha,
        required_roles=tuple(r.value for r in required),
    )


__all__ = [
    "PASS_SPLIT",
    "REVISE_SPLIT",
    "LINEAGE_LEAKAGE",
    "PartitionedItem",
    "PartitionValidationReport",
    "validate_partition",
]
