"""Framework V2 -- generic autonomous acquisition contracts.

Prior to this evolution, Stage-3 acquisition fail-closed with
``PLAN_INPUT_REQUIRED`` because the 14-field ``AcquisitionPlan`` consumed by
``executors._validate_acquisition_plan`` had no framework-derivable source: a
human had to hand-author ``selected_parent_structure_ids``, ``n_parents``,
``T_K``, ``beta``, ``sigma_range_A`` and the rest. That made the *low-level
acquisition knobs* a human authorization boundary, which is the wrong boundary:
the human should authorize only the *high-level objective* (target, claim
scope, optional compute ceiling); everything below that should be designed
autonomously by the framework from evidence, exactly as the Teacher-validation
planner already does for its own stage.

This module defines the typed, immutable, content-addressable contracts the
autonomous acquisition pipeline produces:

    CampaignObjective                (the human boundary -- high-level only)
      -> SourceAndCapabilityInventory  (what sources + backends actually exist)
      -> TargetRegimeModel             (what regimes the target implies)
      -> RegionResolution              (DECLARED vs DISCOVERED, tiered trust)
      -> CoverageGapAnalysis           (where the gaps are, per regime)
      -> AcquisitionStrategy           (which backend(s), decided from evidence)
      -> CandidateGenerationResult     (candidates + yield/rejection diagnostics)
      -> CandidateSelectionResult      (diversity selection + disjointness)
      -> AcquisitionPlanV2             (the bound, executable, auditable plan)

Nothing here hard-codes an element, a composition, a temperature, a sigma, a
seed, a backend, or a regime count. Every material-specific choice is supplied
by evidence (inventory + regime model + coverage gaps) or by a caller-owned
plugin. The core owns only generic mechanics.

Provenance is kept strictly separated (Section K): candidate *generation*
provenance, candidate *selection* provenance, and canonical Teacher *labeling*
provenance are three independently-auditable records. An MD trajectory used for
exploration must never be conflated with a training label; selected frames are
re-labeled canonically under the frozen Teacher.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso


# =====================================================================
# ENUMS
# =====================================================================
class RelevanceRole(str, Enum):
    """A candidate/region's role *relative to the campaign target*.

    These are target-relative (not absolute scope categories): the same
    physical region can be CORE_TARGET for one campaign and BOUNDARY_GUARDRAIL
    for another. The planner assigns roles from evidence, never from a
    material-specific rule baked into the core.
    """
    CORE_TARGET = "CORE_TARGET"
    ADJACENT_PHYSICS = "ADJACENT_PHYSICS"
    GENERATION_PATHWAY = "GENERATION_PATHWAY"
    BOUNDARY_GUARDRAIL = "BOUNDARY_GUARDRAIL"
    OUT_OF_TARGET_ACQUISITION = "OUT_OF_TARGET_ACQUISITION"


class AcquisitionStrategyKind(str, Enum):
    """How candidates are produced. Selected autonomously by the planner
    from the objective + inventory + coverage evidence; never fixed by the
    core."""
    EXISTING_POOL_SELECTION = "EXISTING_POOL_SELECTION"
    LOCAL_PERTURBATION = "LOCAL_PERTURBATION"
    TEACHER_DRIVEN_MD = "TEACHER_DRIVEN_MD"
    STRUCTURE_GENERATION = "STRUCTURE_GENERATION"
    HYBRID = "HYBRID"


class RegionResolutionMode(str, Enum):
    """How the target region structure was resolved.

    DECLARED -- metadata was present and passed a mandatory lightweight
      consistency audit, so declared regions were trusted.
    DISCOVERED -- no metadata; regions were discovered from descriptors.
    DECLARED_ESCALATED_TO_DISCOVERED -- metadata was present but the audit
      flagged a mismatch/weak-provenance/unmapped region, so the resolver
      escalated to a full discovered pass. Metadata is evidence, not ground
      truth.
    """
    DECLARED = "DECLARED"
    DISCOVERED = "DISCOVERED"
    DECLARED_ESCALATED_TO_DISCOVERED = "DECLARED_ESCALATED_TO_DISCOVERED"


class MetadataAuditVerdict(str, Enum):
    PASS = "PASS"
    ESCALATE = "ESCALATE"


class AcquisitionPhase(str, Enum):
    """Which evidence contract applies.

    INITIAL -- no trained Student / no calibrated uncertainty exists yet.
      May use ONLY coverage/novelty/diversity/redundancy/saturation/
      source-coverage/generator-yield evidence + optional compute ceiling.
      Must NOT use model uncertainty or expected-information-gain.
    MODEL_INFORMED -- a trained, calibrated Student exists. May ADDITIONALLY
      use calibrated uncertainty / admissible model disagreement / model-based
      expected information gain.
    """
    INITIAL = "INITIAL"
    MODEL_INFORMED = "MODEL_INFORMED"


# =====================================================================
# 0. CampaignObjective -- the human authorization boundary
# =====================================================================
class ComputeCeiling(ContractBase):
    """Optional bound on how much acquisition may cost.

    All fields optional; a null field means "unbounded on this axis". The
    planner treats these as hard caps -- it may design a *smaller* campaign
    but never exceed a stated ceiling."""
    max_candidates_generated: Optional[int] = None
    max_candidates_selected: Optional[int] = None
    max_teacher_label_calls: Optional[int] = None
    max_md_steps_total: Optional[int] = None
    rationale: str = ""


class CampaignObjective(ContractBase):
    """The high-level, human-authorized boundary for a campaign.

    This is the ONLY thing a human must supply for acquisition. Everything
    downstream (parent selection, perturbation physics, backend choice,
    counts, seeds) is designed autonomously by the planner from evidence.

    ``primary_target`` is a human-readable statement of the distillation
    target (e.g. "amorphous SiO2-x"). ``scope_contract_sha256`` binds this
    objective to the frozen DeploymentScopeContract(V2) that formalizes the
    target regions; the planner reads its PRIMARY regions as CORE_TARGET.
    """
    objective_id: str
    primary_target: str
    claim_scope: str
    scope_contract_sha256: str
    phase: AcquisitionPhase = AcquisitionPhase.INITIAL
    compute_ceiling: Optional[ComputeCeiling] = None
    established_at: str = Field(default_factory=utc_now_iso)


# =====================================================================
# 1. SourceAndCapabilityInventory
# =====================================================================
class SourceCategoryRecord(ContractBase):
    """One category of available source material.

    ``has_metadata`` records whether the source carries descriptor/domain
    metadata (drives the tiered-trust region resolution). ``provenance_class``
    is a freeform provenance label (e.g. "sanitized_pool", "generated"). The
    inventory records *what exists*, never *what to use*.
    """
    category: str
    n_items: int
    has_metadata: bool
    provenance_class: str
    metadata_keys: list[str] = Field(default_factory=list)
    eligible_for_target: bool = True
    rationale: str = ""


class BackendCapabilityRecord(ContractBase):
    """One candidate-generation backend that is actually available in this
    environment, with the capabilities it advertises.

    ``feasible`` is the result of a deterministic capability probe (e.g. the
    generation library imports, the Teacher exposes an ASE calculator). An
    infeasible backend is recorded (absence is evidence) but the planner may
    not select it."""
    backend_id: str
    strategy_kind: AcquisitionStrategyKind
    feasible: bool
    supported_capabilities: list[str] = Field(default_factory=list)
    infeasible_reason: str = ""


class TeacherCapabilityRecord(ContractBase):
    """What the frozen Teacher can do, probed deterministically.

    ``can_label`` -- can produce canonical energy/force labels (required for
    any acquisition; without it acquisition fails closed).
    ``can_drive_dynamics`` -- exposes an ASE-compatible calculator usable to
    drive MD (required for TEACHER_DRIVEN_MD; its absence simply removes that
    backend from the admissible set)."""
    teacher_id: str
    can_label: bool
    can_drive_dynamics: bool
    identity_sha256: str = ""
    rationale: str = ""


class SourceAndCapabilityInventory(ContractBase):
    """The single evidence record of what sources and backends actually
    exist for a campaign. Content-addressable so the planner's decisions
    can be audited against the exact inventory in effect."""
    inventory_id: str
    objective_sha256: str
    sources: list[SourceCategoryRecord]
    backends: list[BackendCapabilityRecord]
    teacher: TeacherCapabilityRecord
    established_at: str = Field(default_factory=utc_now_iso)

    def feasible_backends(self) -> list[BackendCapabilityRecord]:
        return [b for b in self.backends if b.feasible]

    def any_metadata(self) -> bool:
        return any(s.has_metadata for s in self.sources)

    @model_validator(mode="after")
    def _teacher_must_label(self):
        if not self.teacher.can_label:
            raise ValueError(
                "SourceAndCapabilityInventory: frozen Teacher cannot produce "
                "canonical labels; acquisition cannot proceed (fail-closed)"
            )
        return self


# =====================================================================
# 2. TargetRegimeModel
# =====================================================================
class TargetRegime(ContractBase):
    """One regime the target implies, with its target-relative role.

    ``membership_rule`` is a deterministic, human-readable predicate on the
    descriptor space (same convention as DomainRegime.membership_rule)."""
    regime_id: str
    label: str
    relevance_role: RelevanceRole
    membership_rule: str
    evidence_refs: list[str] = Field(default_factory=list)


class TargetRegimeModel(ContractBase):
    """The regime structure implied by the campaign target, derived from the
    scope contract's regions and the descriptor space -- not from a
    material-specific rule in the core."""
    model_id: str
    objective_sha256: str
    descriptor: str
    regimes: list[TargetRegime]
    established_at: str = Field(default_factory=utc_now_iso)

    def core_regimes(self) -> list[TargetRegime]:
        return [r for r in self.regimes if r.relevance_role == RelevanceRole.CORE_TARGET]

    @model_validator(mode="after")
    def _unique_ids_and_has_core(self):
        ids = [r.regime_id for r in self.regimes]
        if len(set(ids)) != len(ids):
            raise ValueError("TargetRegimeModel: regime_id must be unique")
        if not self.core_regimes():
            raise ValueError(
                "TargetRegimeModel must contain at least one CORE_TARGET regime "
                "-- the objective must have a target"
            )
        return self


# =====================================================================
# 3. RegionResolution (tiered metadata trust)
# =====================================================================
class MetadataConsistencyCheck(ContractBase):
    """One deterministic, material-agnostic consistency check applied to
    declared metadata. ``passed`` False on any check triggers escalation."""
    check_id: str
    description: str
    passed: bool
    observed: str = ""
    expected: str = ""


class MetadataConsistencyAudit(ContractBase):
    """The mandatory lightweight audit applied whenever metadata is present.

    If ``verdict`` is ESCALATE the resolver must run a full DISCOVERED pass;
    metadata is evidence, never unquestioned ground truth. The audit logic
    is material-agnostic and deterministically reproducible from
    ``checks``."""
    metadata_present: bool
    audited: bool
    verdict: MetadataAuditVerdict
    checks: list[MetadataConsistencyCheck] = Field(default_factory=list)
    escalation_reason: str = ""

    @model_validator(mode="after")
    def _verdict_consistent_with_checks(self):
        if self.audited and any(not c.passed for c in self.checks):
            if self.verdict != MetadataAuditVerdict.ESCALATE:
                raise ValueError(
                    "MetadataConsistencyAudit: a failed check requires "
                    "verdict=ESCALATE (metadata is evidence, not ground truth)"
                )
        return self


class RegionResolution(ContractBase):
    """How the target region structure was resolved for this campaign, with
    the metadata audit that justified the mode. Binds to the produced
    DomainRepresentation by content-SHA."""
    resolution_id: str
    mode: RegionResolutionMode
    domain_representation_sha256: str
    metadata_audit: MetadataConsistencyAudit
    established_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _mode_consistent_with_audit(self):
        if self.metadata_audit.verdict == MetadataAuditVerdict.ESCALATE:
            if self.mode != RegionResolutionMode.DECLARED_ESCALATED_TO_DISCOVERED:
                raise ValueError(
                    "RegionResolution: an ESCALATE audit must resolve as "
                    "DECLARED_ESCALATED_TO_DISCOVERED"
                )
        if not self.metadata_audit.metadata_present:
            if self.mode != RegionResolutionMode.DISCOVERED:
                raise ValueError(
                    "RegionResolution: absent metadata must resolve as DISCOVERED"
                )
        return self


# =====================================================================
# 4. CoverageGapAnalysis
# =====================================================================
class RegimeCoverage(ContractBase):
    """Coverage evidence for one target regime.

    All fields are INITIAL-phase-admissible (coverage/novelty/diversity/
    redundancy/saturation) -- no model uncertainty or EIG. ``gap_score`` is a
    non-negative deterministic function of the deficit; higher means more
    acquisition is warranted. ``saturated`` True means additional acquisition
    yields diminishing returns for this regime."""
    regime_id: str
    relevance_role: RelevanceRole
    current_count: int
    target_count: Optional[int] = None
    saturation: float
    novelty_headroom: float
    gap_score: float
    saturated: bool

    @model_validator(mode="after")
    def _non_negative(self):
        if self.gap_score < 0 or self.saturation < 0 or self.current_count < 0:
            raise ValueError("RegimeCoverage: counts/scores must be non-negative")
        return self


class CoverageGapAnalysis(ContractBase):
    """Per-regime coverage gap evidence for the whole target. This is the
    evidence the strategy planner reasons over. INITIAL phase must not carry
    uncertainty/EIG fields (enforced by the validator, not the type)."""
    analysis_id: str
    phase: AcquisitionPhase
    target_regime_model_sha256: str
    region_resolution_sha256: str
    per_regime: list[RegimeCoverage]
    available_source_coverage: dict[str, int] = Field(default_factory=dict)
    established_at: str = Field(default_factory=utc_now_iso)

    def unsaturated_core_gaps(self) -> list[RegimeCoverage]:
        return [
            c for c in self.per_regime
            if c.relevance_role == RelevanceRole.CORE_TARGET
            and not c.saturated and c.gap_score > 0.0
        ]


# =====================================================================
# 5. AcquisitionStrategy
# =====================================================================
class AcquisitionStrategy(ContractBase):
    """The backend(s) the planner chose, and why. Bound by content-SHA to the
    coverage-gap analysis and inventory it was derived from."""
    strategy_id: str
    kind: AcquisitionStrategyKind
    selected_backend_ids: list[str]
    coverage_gap_sha256: str
    inventory_sha256: str
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_backend(self):
        if not self.selected_backend_ids:
            raise ValueError("AcquisitionStrategy must select at least one backend")
        return self


# =====================================================================
# 6. Candidate generation (provenance-separated)
# =====================================================================
class GenerationProvenance(ContractBase):
    """How a candidate was generated -- STRICTLY distinct from any label.

    For TEACHER_DRIVEN_MD, ``exploration_energy`` / ``exploration_forces_ref``
    (if recorded at all) are the *exploration* PES used to drive dynamics and
    MUST NOT be used as training labels. Selected frames are re-labeled
    canonically under the frozen Teacher (see CanonicalLabelingRequest)."""
    candidate_id: str
    strategy_kind: AcquisitionStrategyKind
    backend_id: str
    parent_id: Optional[str] = None
    generation_params_sha256: str = ""
    exploration_only: bool = True
    notes: str = ""


class CandidateGenerationResult(ContractBase):
    """Output of a candidate-generation backend, with yield/rejection
    diagnostics (INITIAL-phase-admissible generator evidence)."""
    result_id: str
    strategy_sha256: str
    backend_id: str
    candidate_ids: list[str]
    provenance: list[GenerationProvenance]
    n_requested: int
    n_generated: int
    n_rejected: int
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    artifact_ref: str = ""

    @model_validator(mode="after")
    def _counts_and_provenance_align(self):
        if self.n_generated != len(self.candidate_ids):
            raise ValueError(
                "CandidateGenerationResult: n_generated must equal "
                "len(candidate_ids)"
            )
        prov_ids = {p.candidate_id for p in self.provenance}
        if prov_ids != set(self.candidate_ids):
            raise ValueError(
                "CandidateGenerationResult: every candidate must have exactly "
                "one generation-provenance record"
            )
        if self.n_generated < 0 or self.n_rejected < 0 or self.n_requested < 0:
            raise ValueError("CandidateGenerationResult: counts must be non-negative")
        return self


# =====================================================================
# 7. Candidate selection (diversity + fail-closed disjointness)
# =====================================================================
class ProtectedDisjointnessReport(ContractBase):
    """Result of the mandatory protected-reference disjointness check applied
    to selected candidates. ``status`` must be PASS for the plan to bind;
    ``dft_labels_used_as_selection_scores`` must be False in INITIAL phase."""
    status: str
    n_checked: int
    n_overlaps: int
    dft_labels_used_as_selection_scores: bool = False
    rationale: str = ""

    @model_validator(mode="after")
    def _pass_means_no_overlap(self):
        if self.status == "PASS" and self.n_overlaps != 0:
            raise ValueError(
                "ProtectedDisjointnessReport: status=PASS requires n_overlaps==0"
            )
        return self


class CandidateSelectionResult(ContractBase):
    """The diversity-selected subset that will be canonically labeled, with
    its own selection provenance (distinct from generation provenance)."""
    selection_id: str
    generation_result_sha256: str
    selector: str
    selector_config: dict[str, Any] = Field(default_factory=dict)
    selected_candidate_ids: list[str]
    diversity_evidence: dict[str, Any] = Field(default_factory=dict)
    disjointness_report: ProtectedDisjointnessReport

    @model_validator(mode="after")
    def _selected_nonempty_and_disjoint(self):
        if not self.selected_candidate_ids:
            raise ValueError("CandidateSelectionResult: nothing selected")
        if len(set(self.selected_candidate_ids)) != len(self.selected_candidate_ids):
            raise ValueError("CandidateSelectionResult: duplicate selected ids")
        if self.disjointness_report.status != "PASS":
            raise ValueError(
                "CandidateSelectionResult: protected-reference disjointness must "
                "PASS before selection can bind (fail-closed)"
            )
        return self


# =====================================================================
# 8. Canonical Teacher labeling (provenance-separated)
# =====================================================================
class CanonicalLabelingRequest(ContractBase):
    """Instruction to (re-)label the selected candidates canonically under
    the frozen Teacher. This is the ONLY provenance from which training
    labels may originate. Any exploration PES from generation is discarded
    for labeling purposes."""
    request_id: str
    selection_result_sha256: str
    teacher_identity_sha256: str
    candidate_ids: list[str]
    relabel_from_scratch: bool = True

    @model_validator(mode="after")
    def _must_relabel(self):
        if not self.relabel_from_scratch:
            raise ValueError(
                "CanonicalLabelingRequest: training labels MUST be produced "
                "canonically under the frozen Teacher, never reused from "
                "exploration dynamics (provenance separation, Section K)"
            )
        return self


# =====================================================================
# 9. AcquisitionPlanV2 -- the bound, executable, auditable plan
# =====================================================================
class AcquisitionPlanV2(ContractBase):
    """The autonomous acquisition plan the planner emits and binds to the run.

    It chains every upstream evidence artifact by content-SHA so the whole
    decision path is auditable, and carries exactly ONE execution projection:

      * ``legacy_projection`` -- the 14-field ``AcquisitionPlan`` dict consumed
        unchanged by ``executors._validate_acquisition_plan`` for the legacy
        LOCAL_PERTURBATION executor;
      * ``dynamics_protocol_sha256`` -- for strategies executed by driving the
        Teacher PES (TEACHER_DRIVEN_MD);
      * ``existing_pool_projection`` -- for EXISTING_POOL_SELECTION, where no new
        configurations are generated: a representative subset of the EXISTING
        eligible pool is SELECTED (by deterministic descriptor-space sizing) for
        canonical Teacher labeling. It carries the pool file, the selected global
        indices + parent ids, the deterministic labeling-population sizing
        evidence, and the protected-disjointness report.

    Exactly one execution projection must be present.
    """
    plan_id: str
    objective_sha256: str
    inventory_sha256: str
    target_regime_model_sha256: str
    region_resolution_sha256: str
    coverage_gap_sha256: str
    strategy_sha256: str
    generation_result_sha256: str
    selection_result_sha256: str
    labeling_request_sha256: str
    phase: AcquisitionPhase
    legacy_projection: Optional[dict[str, Any]] = None
    dynamics_protocol_sha256: Optional[str] = None
    existing_pool_projection: Optional[dict[str, Any]] = None
    established_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _exactly_one_projection(self):
        present = sum((
            self.legacy_projection is not None,
            self.dynamics_protocol_sha256 is not None,
            self.existing_pool_projection is not None,
        ))
        if present != 1:
            raise ValueError(
                "AcquisitionPlanV2: exactly one of legacy_projection / "
                "dynamics_protocol_sha256 / existing_pool_projection must be set"
            )
        return self
