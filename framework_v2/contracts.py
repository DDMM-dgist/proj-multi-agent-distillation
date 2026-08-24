"""Framework V2 — typed, versioned, serializable, artifact-addressable contracts.

R31 exposed that many scientifically-critical decisions were carried only in
prose prompts, ad-hoc YAML keys, or executor internals. Framework V2 puts
them in explicit typed contracts so:

  * Every scientific choice has a machine-checkable schema.
  * Every contract has a canonical SHA that binds artifacts to identity.
  * Downstream stages consume a specific artifact SHA -- silent scope drift
    across stages becomes impossible.
  * Auditors can answer "why this X?" from the contract + DecisionLedger,
    not from terminal logs.

Contracts implemented (Section 2 of the V2 directive):

  A. DeploymentScopeContract    -- 5-way scope categorisation, single source
                                   of truth consumed by every downstream
                                   stage.
  B. ScientificDecisionRecord   -- one row of the DecisionLedger.
  C. DomainRepresentation       -- discovered regimes + kind (continuous /
                                   categorical / hierarchical / hybrid).
  D. CoveragePlan               -- descriptor, distance, stopping criterion.
  E. ParentSelectionPlan        -- selector + selected identities.
  F. AugmentationPlan           -- PER-PARENT policies (first-class); the
                                   plan lists ``required_capabilities`` that
                                   the executor must advertise.
  G. DatasetPartitionPlan       -- lineage key + stratification variables +
                                   representativeness requirement.
  H. StudentRecipePlan          -- every scientific-critical parameter as a
                                   RecipeParameter with provenance_class.
  I. ConvergencePolicy          -- thresholds for the ConvergencePolicy
                                   classifier (no hard-coded numbers here).
  J. EvaluationPolicy           -- primary vs. diagnostic metrics,
                                   ``reject_mixed_aggregate_as_primary``.
  K. UncertaintyPolicy          -- method + metrics + calibration evidence.
  L. DeploymentMDPolicy         -- ensembles, stability checks, max wall
                                   time.
  M. PhysicalValidationPolicy   -- observables + tolerance config.

Every contract inherits from ``ContractBase`` and therefore has:

  * ``schema_version``   -- integer for forward migration.
  * ``.content_sha256()``-- canonical SHA over ``model_dump(mode="json")``
                            with sorted keys / compact separators. This is
                            the identity that downstream contracts bind to.

These are *only* the schemas. Executors, validators, ledger writers and
enforcement rules live in the sibling V2 modules (``convergence``,
``capability``, ``evaluation``, ``blind_test``, ``decision_ledger`` etc.).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =====================================================================
# BASE
# =====================================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ContractBase(BaseModel):
    """Immutable, validated, artifact-addressable pydantic model."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int = 1

    def content_sha256(self) -> str:
        """Deterministic SHA-256 of the canonical JSON serialization.

        Uses ``sort_keys=True`` and ``separators=(",", ":")`` so
        identical semantic content across processes/machines yields
        identical digests. This is the identity downstream contracts
        bind to (``linked_*_sha256`` fields)."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =====================================================================
# ENUMS (shared)
# =====================================================================
class ScopeCategory(str, Enum):
    """The single categorisation of a deployment region.

    Consumed by every downstream stage (acquisition/coverage/split/
    evaluation/uncertainty/MD/PV). A stage may act only within categories
    it is explicitly allowed to touch (Blind-test enforcement, Section 4).

    Categories are generic and material-agnostic; a campaign assigns
    concrete regions/subpopulations to them via evidence, never the core.
    """
    PRIMARY_DEPLOYMENT = "PRIMARY_DEPLOYMENT"
    AUXILIARY_SUPPORT = "AUXILIARY_SUPPORT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    PROTECTED_REFERENCE = "PROTECTED_REFERENCE"
    BLIND_TEST = "BLIND_TEST"
    HISTORICAL_BENCHMARK = "HISTORICAL_BENCHMARK"


class ProvenanceClass(str, Enum):
    """Where a scientific value came from.

    LEGACY_REUSED requires explicit ``rationale`` per StudentRecipePlan
    validation (Section 9). TOOL_DEFAULT means "the training tool's own
    default was accepted without inspection" -- also flagged unless
    justified. FRAMEWORK_CONSTRAINT means the framework itself imposes
    the value (e.g. a hard schema type).
    """
    EVIDENCE_DERIVED = "EVIDENCE_DERIVED"
    HUMAN_FIXED = "HUMAN_FIXED"
    AGENT_HEURISTIC = "AGENT_HEURISTIC"
    LEGACY_REUSED = "LEGACY_REUSED"
    FRAMEWORK_CONSTRAINT = "FRAMEWORK_CONSTRAINT"
    TOOL_DEFAULT = "TOOL_DEFAULT"


class PartitionRole(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    BLIND_TEST = "BLIND_TEST"


class ConvergenceStatus(str, Enum):
    CONVERGED_EARLY = "CONVERGED_EARLY"
    CONVERGED_AT_MAX = "CONVERGED_AT_MAX"
    NOT_CONVERGED = "NOT_CONVERGED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# =====================================================================
# A. DeploymentScopeContract
# =====================================================================
class ScopeRegion(ContractBase):
    """One region of deployment space with a category assignment.

    ``membership_rule`` is a deterministic, human-readable expression
    (parseable by domain-discovery code -- e.g. "x = 2 - O/Si in [0,2],
    density in [1.9, 2.4] g/cm^3, coordination(Si-O) = 4"). Evidence
    for why this region is in this category is listed in
    ``membership_evidence``.
    """
    region_id: str
    category: ScopeCategory
    membership_rule: str
    membership_evidence: list[str] = Field(default_factory=list)
    rationale: str = ""


class DeploymentScopeContract(ContractBase):
    """The single source of truth for what the run is trying to be good at.

    Section 3: a stage that tries to reinterpret scope must produce a
    contract violation, not silently proceed. Every downstream contract
    binds to ``.content_sha256()`` (or an equivalent explicit reference)
    so scope drift is detectable.
    """
    contract_id: str
    objective: str
    regions: list[ScopeRegion]
    established_at: str = Field(default_factory=utc_now_iso)

    def region(self, region_id: str) -> Optional[ScopeRegion]:
        for r in self.regions:
            if r.region_id == region_id:
                return r
        return None

    def regions_of(self, category: ScopeCategory) -> list[ScopeRegion]:
        return [r for r in self.regions if r.category == category]

    @model_validator(mode="after")
    def _at_least_one_primary(self):
        if not self.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT):
            raise ValueError(
                "DeploymentScopeContract must declare at least one "
                "PRIMARY_DEPLOYMENT region -- the objective must have a target"
            )
        return self

    @model_validator(mode="after")
    def _unique_region_ids(self):
        ids = [r.region_id for r in self.regions]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate ScopeRegion.region_id values: {dupes}")
        return self


# =====================================================================
# B. ScientificDecisionRecord (one row of DecisionLedger)
# =====================================================================
class ScientificDecisionRecord(ContractBase):
    """Section 16: one auditable row.

    Answers "why this X?" without terminal-log archaeology.
    ``deterministic_facts`` lists identifiers of ``DeterministicFact``
    records that support this decision (see ``framework_v2.facts``).
    ``downstream_dependencies`` lists ``decision_id`` values that this
    decision constrains.
    """
    decision_id: str
    stage: str
    decision: str
    selected: Any
    alternatives_considered: list[Any] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    deterministic_facts: list[str] = Field(default_factory=list)
    provenance_class: ProvenanceClass
    rationale: str
    sensitivity: Optional[dict[str, Any]] = None
    at: str = Field(default_factory=utc_now_iso)
    actor: str
    downstream_dependencies: list[str] = Field(default_factory=list)


# =====================================================================
# C. DomainRepresentation (Section 5)
# =====================================================================
class DomainRegime(ContractBase):
    """One discovered regime.

    ``membership_rule`` is deterministic (a code path or an explicit
    predicate). ``within_scope_categories`` lists which scope categories
    this regime overlaps -- a single regime may be simultaneously in
    PRIMARY_DEPLOYMENT and AUXILIARY_SUPPORT if the regime happens to
    straddle categories, which itself is auditable evidence."""
    regime_id: str
    label: str
    membership_rule: str
    membership_evidence_refs: list[str] = Field(default_factory=list)
    within_scope_categories: list[ScopeCategory]


class DomainRepresentation(ContractBase):
    representation_id: str
    kind: Literal["continuous", "categorical", "hierarchical", "hybrid"]
    descriptor: str
    regimes: list[DomainRegime]
    sensitivity_report: Optional[dict[str, Any]] = None
    linked_scope_contract_sha256: str

    @model_validator(mode="after")
    def _regime_ids_unique(self):
        ids = [r.regime_id for r in self.regimes]
        if len(set(ids)) != len(ids):
            raise ValueError("DomainRegime.regime_id must be unique")
        return self


# =====================================================================
# D. CoveragePlan (Section 6)
# =====================================================================
class CoveragePlan(ContractBase):
    plan_id: str
    representation_sha256: str
    distance_metric: str
    diminishing_return_threshold: Optional[float] = None
    stopping_criterion: str
    max_selected: Optional[int] = None


# =====================================================================
# E. ParentSelectionPlan (Section 6)
# =====================================================================
class ParentSelectionPlan(ContractBase):
    plan_id: str
    coverage_plan_sha256: str
    selector: str
    selector_config: dict[str, Any] = Field(default_factory=dict)
    selected_ids: list[str] = Field(default_factory=list)
    selection_evidence_refs: list[str] = Field(default_factory=list)


# =====================================================================
# F. AugmentationPlan (Section 7) -- PER-PARENT is first-class
# =====================================================================
class PerParentAugPolicy(ContractBase):
    """The policy for augmenting ONE parent structure.

    ``n_samples`` may legitimately be 0 (a parent that need not be
    augmented). ``amplitude_range`` is a closed interval [lo, hi]. The
    executor must consume this record exactly -- if it can't represent
    any field it must produce a FRAMEWORK_CAPABILITY_BLOCKER, never
    silently round or drop.
    """
    parent_id: str
    n_samples: int
    method: str
    displacement_distribution: str = "gaussian"
    amplitude_range: tuple[float, float]
    cell_perturbation: Optional[dict[str, Any]] = None
    relaxation: Optional[dict[str, Any]] = None
    proposal_temperature: Optional[float] = None
    acceptance_constraints: dict[str, Any] = Field(default_factory=dict)
    force_max_per_atom: Optional[float] = None
    min_atomic_separation: Optional[float] = None
    stopping_condition: str = "reach_n_samples"


class AugmentationPlan(ContractBase):
    plan_id: str
    parent_selection_plan_sha256: str
    per_parent: list[PerParentAugPolicy]
    required_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parent_ids_unique(self):
        ids = [p.parent_id for p in self.per_parent]
        if len(set(ids)) != len(ids):
            raise ValueError("AugmentationPlan: per_parent.parent_id must be unique")
        return self

    def is_heterogeneous(self) -> bool:
        """True iff parents differ in any field beyond ``parent_id``.

        Used by the capability negotiator (``framework_v2.capability``)
        to decide whether the plan strictly requires per-parent
        executor capability, or whether a global executor would produce
        the same result."""
        if len(self.per_parent) < 2:
            return False
        first = self.per_parent[0].model_dump(exclude={"parent_id"})
        return any(
            p.model_dump(exclude={"parent_id"}) != first for p in self.per_parent[1:]
        )


# =====================================================================
# G. DatasetPartitionPlan (Section 8)
# =====================================================================
class DatasetPartitionPlan(ContractBase):
    """Lineage safety + scientific representativeness are separate
    requirements. Both must PASS."""
    plan_id: str
    scope_contract_sha256: str
    lineage_key: str
    stratification_variables: list[str]
    fractions: dict[PartitionRole, float]
    representativeness_requirement: str  # a deterministic rule expression

    @model_validator(mode="after")
    def _fractions_sum_close_to_one(self):
        s = sum(self.fractions.values())
        if not 0.999 <= s <= 1.001:
            raise ValueError(f"DatasetPartitionPlan.fractions must sum to 1.0 (got {s})")
        return self


# =====================================================================
# H. StudentRecipePlan (Section 9)
# =====================================================================
class RecipeParameter(ContractBase):
    """One scientifically-critical Student parameter.

    ``LEGACY_REUSED`` and ``TOOL_DEFAULT`` are allowed but require a
    non-empty ``rationale`` (validated externally by
    ``framework_v2.decision_ledger.validate_recipe_provenance``)."""
    name: str
    value: Any
    provenance_class: ProvenanceClass
    evidence: list[str] = Field(default_factory=list)
    alternatives_considered: list[Any] = Field(default_factory=list)
    rationale: str


class StudentRecipePlan(ContractBase):
    """Every scientifically-critical Student parameter, each with full
    provenance. Additional parameters go in ``additional``."""
    plan_id: str
    descriptor: RecipeParameter
    architecture: RecipeParameter
    optimizer: RecipeParameter
    learning_rate: RecipeParameter
    batch_size: RecipeParameter
    energy_force_loss_weighting: RecipeParameter
    normalization: RecipeParameter
    initial_training_budget: RecipeParameter
    numerical_precision: RecipeParameter
    additional: list[RecipeParameter] = Field(default_factory=list)

    def all_parameters(self) -> list[RecipeParameter]:
        core = [self.descriptor, self.architecture, self.optimizer,
                self.learning_rate, self.batch_size,
                self.energy_force_loss_weighting, self.normalization,
                self.initial_training_budget, self.numerical_precision]
        return core + list(self.additional)


# =====================================================================
# I. ConvergencePolicy (Section 10)
# =====================================================================
class ConvergencePolicy(ContractBase):
    """Thresholds for the convergence classifier.

    NO NUMBERS ARE HARD-CODED. The caller (workflow.yaml or
    StudentRecipePlan.additional) supplies the values and stamps their
    provenance in ``provenance_class`` + ``provenance_source``.
    ``framework_v2.convergence.classify_seed_convergence`` and
    ``build_convergence_report`` consume this contract."""
    policy_id: str
    trailing_window: int
    projection_window: int
    min_relative_improvement: float
    boundary_tolerance: int
    metrics: list[str]
    provenance_class: ProvenanceClass
    provenance_source: str


# =====================================================================
# J. EvaluationPolicy (Section 11)
# =====================================================================
class EvaluationPolicy(ContractBase):
    """Section 11: primary metrics only over PRIMARY_DEPLOYMENT.

    ``reject_mixed_aggregate_as_primary`` MUST be true for any
    production evaluation; ``False`` is allowed only in explicit
    diagnostic modes."""
    policy_id: str
    scope_contract_sha256: str
    primary_metrics: list[str]
    diagnostic_metrics: list[str] = Field(default_factory=list)
    normalization: str = "absolute"
    reject_mixed_aggregate_as_primary: bool = True
    reference_scale_context_required_for_r2: bool = True

    @model_validator(mode="after")
    def _non_empty_primary(self):
        if not self.primary_metrics:
            raise ValueError("EvaluationPolicy.primary_metrics must be non-empty")
        return self


# =====================================================================
# K. UncertaintyPolicy (Section 12)
# =====================================================================
class UncertaintyPolicy(ContractBase):
    policy_id: str
    method: str
    metrics: list[str]
    calibration_evidence_ref: Optional[str] = None


# =====================================================================
# L. DeploymentMDPolicy (Section 12)
# =====================================================================
class DeploymentMDPolicy(ContractBase):
    policy_id: str
    scope_contract_sha256: str
    ensembles: list[dict[str, Any]]
    stability_checks: list[str]
    max_wall_time_s: Optional[float] = None


# =====================================================================
# M. PhysicalValidationPolicy (Section 12)
# =====================================================================
class PhysicalValidationPolicy(ContractBase):
    policy_id: str
    scope_contract_sha256: str
    observables: list[str]
    reference_values: dict[str, Any] = Field(default_factory=dict)
    tolerance_config: dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Public exports
# =====================================================================
__all__ = [
    # base
    "ContractBase", "utc_now_iso",
    # enums
    "ScopeCategory", "ProvenanceClass", "PartitionRole", "ConvergenceStatus",
    # A
    "ScopeRegion", "DeploymentScopeContract",
    # B
    "ScientificDecisionRecord",
    # C
    "DomainRegime", "DomainRepresentation",
    # D
    "CoveragePlan",
    # E
    "ParentSelectionPlan",
    # F
    "PerParentAugPolicy", "AugmentationPlan",
    # G
    "DatasetPartitionPlan",
    # H
    "RecipeParameter", "StudentRecipePlan",
    # I
    "ConvergencePolicy",
    # J
    "EvaluationPolicy",
    # K
    "UncertaintyPolicy",
    # L
    "DeploymentMDPolicy",
    # M
    "PhysicalValidationPolicy",
]
