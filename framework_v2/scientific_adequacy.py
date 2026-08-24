"""Framework V2 -- SCIENTIFIC-ADEQUACY LAYER (Section 11-bis).

Session 2026-08-21. Governance closure of the C12F diagnosis.

This module implements the **scientific-adequacy layer** that sits atop the
existing procedural/reproducibility gates. It is deliberately generic and
material-agnostic:

  * no MLIP is named,
  * no numerical thresholds (energy MAE, force RMSE, R^2, density, cutoffs)
    are hard-coded here,
  * every numerical criterion is a CONTRACT INPUT with typed provenance,
  * every criterion has an admissible ``ThresholdSourceClass`` and a
    ``frozen_before_evaluation`` boolean; if that boolean is false the
    adequacy verdict fails closed to ``NOT_EVALUABLE`` (not PASS, not FAIL).

The separating principle (Section 1 of the closure directive):

    procedural PASS  !=  scientific adequacy

A stage that satisfies its procedural gate (correct population, valid hashes,
required artifacts exist, deterministic validator passes) has only established
that its evidence was *generated correctly*. Whether the evidence *supports the
scientific claim* is a separate typed decision handled here.

Everything in this module is a Pydantic contract or a pure adjudicator
function. Nothing writes files, calls executors, or mutates run state.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso


# =====================================================================
# Enums
# =====================================================================
class AdequacyStatus(str, Enum):
    """Verdict of a scientific-adequacy adjudication.

    ``NOT_EVALUABLE`` is the fail-closed default whenever no admissible
    pre-registered criterion exists. It is distinct from ``FAIL`` (a
    pre-registered criterion existed and was not met) and from ``PASS``
    (a pre-registered criterion existed and was met)."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class ThresholdSourceClass(str, Enum):
    """Admissible provenance classes for a numerical acceptance criterion.

    A historical achieved metric alone is NOT an admissible source. The
    threshold must trace to an *independent* rationale that predates the
    evidence it is applied to.
    """
    APPLICATION_ERROR_BUDGET = "application_error_budget"
    LITERATURE_OR_COMMUNITY = "literature_or_community"
    EXTERNALLY_VALIDATED_PROJECT_CRITERION = "externally_validated_project_criterion"
    PREREGISTERED_RELATIVE_TO_REFERENCE_MODEL = "preregistered_relative_to_reference_model"
    PREREGISTERED_HISTORICAL_BENCHMARK = "preregistered_historical_benchmark"


class CalibrationStatus(str, Enum):
    """Uncertainty calibration status (Section 6 of the closure directive)."""
    UNCALIBRATED = "uncalibrated"
    CALIBRATED_PARTIAL = "calibrated_partial"
    CALIBRATED = "calibrated"


class ClaimRole(str, Enum):
    """The role a domain plays in the campaign's scientific claim."""
    PRIMARY_CLAIM = "primary_claim"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


class DeploymentStateRole(str, Enum):
    """Role of one physical-validation representative deployment point.

    ``AMBIENT_REPRESENTATIVE_POINT`` means the point is chosen as a stand-in
    for ambient-condition behavior; ``HIGH_PRESSURE_REPRESENTATIVE_POINT``
    and similar labels do NOT stand in for ambient behavior. The framework
    must not silently promote one role to another.
    """
    AMBIENT_REPRESENTATIVE_POINT = "ambient_representative_point"
    HIGH_PRESSURE_REPRESENTATIVE_POINT = "high_pressure_representative_point"
    HIGH_TEMPERATURE_REPRESENTATIVE_POINT = "high_temperature_representative_point"
    MELT_POINT = "melt_point"
    DEFECT_RICH_POINT = "defect_rich_point"
    USER_DEFINED_OTHER = "user_defined_other"


class EnsembleKind(str, Enum):
    """MD ensemble labels used by StatePreparationPolicy."""
    NVE = "NVE"
    NVT = "NVT"
    NPT = "NPT"
    NPH = "NPH"


class ObservableRole(str, Enum):
    """Role an observable plays in the physical-validation policy."""
    DESCRIPTIVE = "descriptive"
    THRESHOLDED = "thresholded"


class RootCauseClass(str, Enum):
    """Diagnosed root cause of a scientific-adequacy failure.

    Each root-cause class carries an admissible return-stage set (see
    ``recovery_return_stages`` below); routing therefore follows the cause,
    not a fixed stage number.
    """
    FIDELITY_INADEQUACY = "fidelity_inadequacy"
    DEPLOYMENT_STATE_MISMATCH = "deployment_state_mismatch"
    PHYSICAL_OBSERVABLE_IMPLEMENTATION_DEFECT = "physical_observable_implementation_defect"
    REFERENCE_INADEQUACY = "reference_inadequacy"
    UNCERTAINTY_CALIBRATION_FAILURE = "uncertainty_calibration_failure"
    FRAMEWORK_EVIDENCE_READABILITY_DEFECT = "framework_evidence_readability_defect"


# =====================================================================
# AdequacyCriterion: one numerical acceptance rule with typed provenance
# =====================================================================
class AdequacyCriterion(ContractBase):
    """A single numerical acceptance rule.

    Every field is required. There is no default value that lets a caller
    silently omit provenance. A caller supplying an inadmissible or
    unprovenanced criterion cannot induce a PASS; the adequacy adjudicator
    will refuse to score the observable and return ``NOT_EVALUABLE``.

    ``frozen_before_evaluation`` MUST be true for the criterion to
    contribute to a PASS/FAIL adjudication on a specific evidence artifact.
    A criterion frozen after the fact is admissible only in the sense that
    it may govern a *future* evaluation; it may not be back-applied.
    """
    criterion_id: str
    observable: str                         # e.g. "student_vs_teacher::f_rmse"
    domain: Optional[str] = None            # None => aggregate; else a domain id
    operator: str                            # "max_abs", "max", "min", "target_tolerance"
    value: float                             # threshold value
    unit: str                                # required, no default
    rationale: str                           # WHY this number, independent of the current result
    source_class: ThresholdSourceClass
    source_reference: str                    # path / DOI / decision-id
    frozen_timestamp: str = Field(default_factory=utc_now_iso)
    frozen_before_evaluation: bool           # HARD requirement for scoring
    applicable_domains: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_shape(self):
        allowed = {"max_abs", "max", "min", "target_tolerance", "target_max_diff"}
        if self.operator not in allowed:
            raise ValueError(f"AdequacyCriterion operator {self.operator!r} not in {sorted(allowed)}")
        if not self.unit:
            raise ValueError("AdequacyCriterion.unit is required")
        if not self.rationale:
            raise ValueError("AdequacyCriterion.rationale is required")
        if not self.source_reference:
            raise ValueError("AdequacyCriterion.source_reference is required")
        return self


def criterion_passes(value: Any, criterion: AdequacyCriterion,
                     *, tolerance: Optional[float] = None,
                     target: Optional[float] = None) -> bool:
    """Deterministic comparison. Mirrors validation.report.criterion_passes but
    on the typed AdequacyCriterion so a caller cannot supply a value in the
    wrong unit and quietly get a PASS -- that is caught upstream by the
    observable's own emit; here we only perform the numeric compare."""
    v = float(value)
    if criterion.operator == "max_abs":
        return abs(v) <= float(criterion.value)
    if criterion.operator == "max":
        return v <= float(criterion.value)
    if criterion.operator == "min":
        return v >= float(criterion.value)
    if criterion.operator == "target_tolerance":
        if target is None or tolerance is None:
            raise ValueError("target_tolerance requires explicit target and tolerance")
        return abs(v - float(target)) <= float(tolerance)
    if criterion.operator == "target_max_diff":
        if target is None:
            raise ValueError("target_max_diff requires explicit target")
        return abs(v - float(target)) <= float(criterion.value)
    raise ValueError(f"unknown operator {criterion.operator!r}")


# =====================================================================
# Deployment scope V2: label -> canonical domain -> claim role
# =====================================================================
class DomainMapping(ContractBase):
    """One raw-label => canonical scientific domain mapping.

    ``claim_role`` MUST be non-ambiguous. If the campaign truly does not
    have a decision about the label, the mapping shape supports
    ``ClaimRole.AMBIGUOUS`` explicitly; the aggregator then refuses to
    treat this label as either in-scope or out-of-scope for PASS/FAIL
    adjudication and reports it as an unresolved scope decision.
    """
    raw_label: str
    canonical_domain: str
    claim_role: ClaimRole
    rationale: str


class DeploymentScopeContractV2(ContractBase):
    """A scope contract with explicit label mapping and representative-point
    separation.

    ``primary_domains`` lists canonical domain identifiers claimed for the
    primary deployment domain (the fidelity claim).
    ``representative_deployment_points`` lists the specific physical-validation
    MD points; each point has its own ``DeploymentStateRole`` and does NOT
    automatically validate the full primary domain.
    """
    contract_id: str
    objective: str
    primary_domains: list[str]
    label_map: list[DomainMapping]
    representative_deployment_points: list[str] = Field(default_factory=list)
    established_at: str = Field(default_factory=utc_now_iso)

    def role_of(self, raw_label: str) -> ClaimRole:
        for m in self.label_map:
            if m.raw_label == raw_label:
                return m.claim_role
        # missing label != implicit in-scope
        return ClaimRole.AMBIGUOUS

    @model_validator(mode="after")
    def _at_least_one_primary_domain(self):
        if not self.primary_domains:
            raise ValueError("DeploymentScopeContractV2 must declare at least one primary_domain")
        return self


# =====================================================================
# Evaluation adequacy policy V2
# =====================================================================
class EvaluationAdequacyPolicyV2(ContractBase):
    """A policy that ADD a scientific-adequacy layer atop the procedural
    Stage-8 evaluation.

    ``per_domain_criteria`` and ``worst_domain_criteria`` are the two most
    important knobs. The aggregation semantic is captured in
    ``aggregate_role``: an aggregate metric that hides in-scope worst-domain
    failure is a policy defect, not a PASS.

    ``preregistration_witness_ref`` is a required back-reference (decision
    ledger id / signed frozen artifact hash) proving that the policy was
    bound BEFORE the evidence it evaluates. If a caller supplies a policy
    without such a witness, ``score`` refuses and returns NOT_EVALUABLE.
    """
    policy_id: str
    scope_contract_ref: str
    per_domain_criteria: list[AdequacyCriterion] = Field(default_factory=list)
    worst_domain_criteria: list[AdequacyCriterion] = Field(default_factory=list)
    aggregate_criteria: list[AdequacyCriterion] = Field(default_factory=list)
    relative_to_reference_criteria: list[AdequacyCriterion] = Field(default_factory=list)
    outlier_tail_criteria: list[AdequacyCriterion] = Field(default_factory=list)
    aggregate_role: str = "diagnostic_only"    # aggregate must not hide domain failure
    preregistration_witness_ref: Optional[str] = None


class EvaluationAdequacyVerdict(ContractBase):
    """The typed outcome of scoring an accuracy_report against a
    EvaluationAdequacyPolicyV2."""
    policy_id: str
    status: AdequacyStatus
    unmet_criteria: list[str] = Field(default_factory=list)
    per_domain_status: dict[str, AdequacyStatus] = Field(default_factory=dict)
    not_evaluable_reasons: list[str] = Field(default_factory=list)


def evaluate_adequacy(
    policy: EvaluationAdequacyPolicyV2,
    metrics: dict[str, dict[str, float]],
    *,
    in_scope_domains: list[str],
) -> EvaluationAdequacyVerdict:
    """Adjudicate an accuracy report against a bound adequacy policy.

    ``metrics`` shape:
        {domain_id: {observable_id: value, ...}, ..., "__aggregate__": {...}}

    Fail-closed semantics:
      * missing preregistration witness => NOT_EVALUABLE
      * any criterion with ``frozen_before_evaluation=False`` => NOT_EVALUABLE
      * an in-scope domain missing a required per-domain criterion evidence
        value => NOT_EVALUABLE for that domain
      * ambiguous scope role => NOT_EVALUABLE (never silently in-scope)
    """
    not_evaluable: list[str] = []
    unmet: list[str] = []
    per_domain: dict[str, AdequacyStatus] = {}

    if not policy.preregistration_witness_ref:
        return EvaluationAdequacyVerdict(
            policy_id=policy.policy_id,
            status=AdequacyStatus.NOT_EVALUABLE,
            not_evaluable_reasons=["preregistration_witness_ref is required to score"],
        )

    def _scoreable(crit: AdequacyCriterion) -> bool:
        return crit.frozen_before_evaluation

    for domain in in_scope_domains:
        domain_ok = True
        domain_metrics = metrics.get(domain, {})
        matched_any = False
        for crit in policy.per_domain_criteria:
            if crit.domain not in (None, domain) and domain not in crit.applicable_domains:
                continue
            if crit.domain is not None and crit.domain != domain:
                continue
            if not _scoreable(crit):
                not_evaluable.append(
                    f"criterion {crit.criterion_id!r} not frozen before evaluation")
                domain_ok = False
                continue
            observable = crit.observable
            if observable not in domain_metrics:
                not_evaluable.append(
                    f"domain {domain!r} lacks required metric {observable!r}")
                domain_ok = False
                continue
            matched_any = True
            if not criterion_passes(domain_metrics[observable], crit):
                unmet.append(f"{domain}::{observable} vs criterion {crit.criterion_id}")
                domain_ok = False
        if not matched_any:
            per_domain[domain] = AdequacyStatus.NOT_EVALUABLE
            not_evaluable.append(f"domain {domain!r} has no bound per-domain criterion")
        else:
            per_domain[domain] = AdequacyStatus.PASS if domain_ok else AdequacyStatus.FAIL

    # Worst-domain criteria: must be scored across in-scope domains.
    for crit in policy.worst_domain_criteria:
        if not _scoreable(crit):
            not_evaluable.append(
                f"worst-domain criterion {crit.criterion_id!r} not frozen before evaluation")
            continue
        vals = []
        for d in in_scope_domains:
            v = metrics.get(d, {}).get(crit.observable)
            if v is None:
                not_evaluable.append(
                    f"worst-domain check {crit.criterion_id!r}: domain {d!r} lacks {crit.observable}")
                continue
            vals.append(v)
        if not vals:
            continue
        worst = max(abs(v) for v in vals) if crit.operator == "max_abs" \
            else max(vals) if crit.operator == "max" else min(vals)
        if not criterion_passes(worst, crit):
            unmet.append(f"worst-domain {crit.observable} exceeds {crit.criterion_id}")

    # Aggregate criteria may only *support* a claim, not overturn a per-domain
    # failure (Section 2 of the closure directive: aggregate metric must not
    # hide in-scope domain failure).
    for crit in policy.aggregate_criteria:
        if not _scoreable(crit):
            not_evaluable.append(
                f"aggregate criterion {crit.criterion_id!r} not frozen before evaluation")
            continue
        agg = metrics.get("__aggregate__", {}).get(crit.observable)
        if agg is None:
            not_evaluable.append(
                f"aggregate check {crit.criterion_id!r}: __aggregate__ lacks {crit.observable}")
            continue
        if not criterion_passes(agg, crit):
            unmet.append(f"aggregate {crit.observable} vs criterion {crit.criterion_id}")

    # Final status
    if not_evaluable:
        return EvaluationAdequacyVerdict(
            policy_id=policy.policy_id,
            status=AdequacyStatus.NOT_EVALUABLE,
            unmet_criteria=unmet,
            per_domain_status=per_domain,
            not_evaluable_reasons=not_evaluable,
        )
    status = AdequacyStatus.PASS if not unmet else AdequacyStatus.FAIL
    return EvaluationAdequacyVerdict(
        policy_id=policy.policy_id,
        status=status,
        unmet_criteria=unmet,
        per_domain_status=per_domain,
    )


# =====================================================================
# State preparation policy (Stage-10 governance)
# =====================================================================
class StatePreparationPolicy(ContractBase):
    """A typed policy binding *what* the pre-MD state must be and *how* it
    was prepared.

    The framework must reject any interpretation of the executed MD as
    validating the intended-state claim unless ``realized_state_matches`` is
    ``True``. The realized-vs-intended comparison lives outside this policy
    (it is a deterministic ``verify_state_realization`` step in Stage 10);
    but this contract is what the check binds to.
    """
    policy_id: str
    scope_contract_ref: str
    state_role: DeploymentStateRole
    intended_composition_ref: str
    intended_temperature_K: Optional[float] = None
    intended_pressure_GPa: Optional[float] = None
    intended_density_g_per_cm3: Optional[float] = None
    intended_density_justification: Optional[str] = None
    intended_phase_or_structure_class: Optional[str] = None
    preparation_method: str                             # e.g. "melt_quench", "validated_ambient_reference", "npt_equilibration"
    starting_structure_provenance_ref: str
    ensemble: EnsembleKind
    equilibration_protocol_ref: str
    production_protocol_ref: str
    realized_state_criteria: list[str] = Field(default_factory=list)
    frozen_before_md: bool = False

    @model_validator(mode="after")
    def _density_needs_justification(self):
        if self.intended_density_g_per_cm3 is not None and not self.intended_density_justification:
            raise ValueError(
                "intended_density_g_per_cm3 requires intended_density_justification "
                "(the framework must not silently accept a density as a target)")
        return self

    @model_validator(mode="after")
    def _ensemble_semantics(self):
        # In NVT the reported density is inherited from the cell, not predicted.
        if self.ensemble == EnsembleKind.NVT and self.intended_density_g_per_cm3 is not None:
            # allowed, but caller must acknowledge inheritance explicitly
            if "inherited_from_cell" not in (self.intended_density_justification or "").lower():
                raise ValueError(
                    "NVT ensemble: intended_density_g_per_cm3 must be justified with "
                    "'inherited_from_cell' acknowledgment (NVT density is not predicted)")
        return self


# =====================================================================
# Physical validation policy V2
# =====================================================================
class ObservableSpec(ContractBase):
    """One typed physical-validation observable.

    Distinguishes observable *shapes* (e.g. RDF peak position vs peak height,
    species-specific coordination vs total-neighbor count) so a generic
    physical-validation pipeline cannot silently emit an all-species neighbor
    count and call it 'Si-O coordination'.
    """
    name: str
    kind: str                              # e.g. "rdf_peak_position", "rdf_peak_height", "species_coordination", "density", "nve_drift"
    center_species: Optional[str] = None
    neighbor_species: Optional[str] = None
    computation_method: str
    units: str
    ensemble_applicability: list[EnsembleKind]
    reference_source: str                  # "teacher", "dft", "experiment", "other"
    reference_thermodynamic_state_ref: Optional[str] = None
    comparison_method: str                  # e.g. "peak_position_within_A", "descriptive", "max_abs_threshold"
    role: ObservableRole
    cutoff_source_ref: Optional[str] = None
    cutoff_frozen_before_student: Optional[bool] = None
    frozen_before_student_results: bool = False

    @model_validator(mode="after")
    def _cutoff_needs_freeze_flag(self):
        if self.kind == "species_coordination" and self.cutoff_source_ref is None:
            raise ValueError(
                "species_coordination requires cutoff_source_ref (must not default silently)")
        if self.cutoff_source_ref is not None and self.cutoff_frozen_before_student is not True:
            raise ValueError(
                "any cutoff derived from a reference/Teacher structure MUST be frozen "
                "before Student structural results if it is used as an acceptance definition")
        return self


class PhysicalValidationPolicyV2(ContractBase):
    """A physical-validation contract typed at the observable-shape level.

    A caller cannot silently substitute a total-neighbor count for a
    species-specific coordination, nor a peak height for a peak position.
    """
    policy_id: str
    scope_contract_ref: str
    representative_point_ref: str
    observables: list[ObservableSpec]
    reference_at_matched_state: bool = True

    @model_validator(mode="after")
    def _no_unfrozen_reference_cutoff(self):
        for o in self.observables:
            if o.role == ObservableRole.THRESHOLDED and not o.frozen_before_student_results:
                raise ValueError(
                    f"THRESHOLDED observable {o.name!r} must be frozen before Student results")
        return self


# =====================================================================
# Uncertainty policy V2
# =====================================================================
class UncertaintyPolicyV2(ContractBase):
    """Uncertainty policy that separates disagreement from calibrated
    predictive uncertainty.

    ``required_status`` lets a campaign declare what level of calibration is
    required by its scientific claim. If ``UNCALIBRATED`` is required, the
    Stage-9 procedural PASS is scientifically sufficient. If
    ``CALIBRATED_PARTIAL`` or ``CALIBRATED`` is required, an uncalibrated
    committee-disagreement report is scientifically inadequate.
    """
    policy_id: str
    scope_contract_ref: str
    method: str
    metrics: list[str]
    required_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    calibration_evidence_ref: Optional[str] = None
    independent_holdout_ref: Optional[str] = None

    @model_validator(mode="after")
    def _calibrated_needs_evidence(self):
        if self.required_status != CalibrationStatus.UNCALIBRATED and self.calibration_evidence_ref is None:
            raise ValueError(
                f"required_status={self.required_status} requires calibration_evidence_ref")
        return self


def adjudicate_uncertainty(policy: UncertaintyPolicyV2,
                           observed_status: CalibrationStatus) -> AdequacyStatus:
    """Non-substitutable check: an uncalibrated report cannot satisfy a
    calibrated requirement."""
    order = [CalibrationStatus.UNCALIBRATED,
             CalibrationStatus.CALIBRATED_PARTIAL,
             CalibrationStatus.CALIBRATED]
    if order.index(observed_status) >= order.index(policy.required_status):
        return AdequacyStatus.PASS
    return AdequacyStatus.FAIL


# =====================================================================
# Judge-facing scientific-question contract
# =====================================================================
class ScientificQuestion(ContractBase):
    """Generic Judge-facing question. Does not encode numbers.

    The Judge receives:
        (question_text)
      + (frozen policy content, hashes, provenance references)
      + (evidence values from the deterministic layer)

    That composition is what lets Judges adjudicate scientific adequacy
    without encoding SiO2-specific (or any other material-specific) values
    in the Judge prompt itself.
    """
    question_id: str
    stage: str
    question_text: str
    policy_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


DEFAULT_SCIENTIFIC_QUESTIONS: dict[str, ScientificQuestion] = {
    "evaluation": ScientificQuestion(
        question_id="scientific_adequacy::evaluation",
        stage="evaluation",
        question_text=(
            "Does the Student satisfy the pre-registered adequacy requirements "
            "throughout the declared in-scope deployment domain, per the bound "
            "EvaluationAdequacyPolicyV2?"),
    ),
    "deployment_md": ScientificQuestion(
        question_id="scientific_adequacy::deployment_md",
        stage="deployment_md",
        question_text=(
            "Did the realized simulation correspond to the pre-registered "
            "physical deployment point (state role, T, P, phase, "
            "starting-structure provenance, ensemble, protocol), and was "
            "execution numerically stable there?"),
    ),
    "physical_validation": ScientificQuestion(
        question_id="scientific_adequacy::physical_validation",
        stage="physical_validation",
        question_text=(
            "Do the scientifically defined observables at the matched "
            "physical state satisfy their pre-registered reference / "
            "comparison policy?"),
    ),
    "uncertainty": ScientificQuestion(
        question_id="scientific_adequacy::uncertainty",
        stage="uncertainty",
        question_text=(
            "Does the committee-disagreement report satisfy the calibration "
            "status required by the campaign's UncertaintyPolicyV2?"),
    ),
}


# =====================================================================
# Root-cause classifier + return-stage set
# =====================================================================
class RootCauseDiagnosis(ContractBase):
    """A typed root-cause diagnosis with an admissible return-stage set.

    The return-stage set is an ORDERED list of candidates -- the earliest
    stage from which recovery is scientifically justified given the
    evidence. The framework does not select a return stage on the basis of
    the failing-stage number alone.
    """
    diagnosis_id: str
    root_cause: RootCauseClass
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    admissible_return_stages: list[str] = Field(default_factory=list)
    forbidden_recovery_actions: list[str] = Field(default_factory=list)


DEFAULT_ROOT_CAUSE_ROUTING: dict[RootCauseClass, list[str]] = {
    RootCauseClass.FIDELITY_INADEQUACY: [
        "diagnose_domain_error",
        "data_coverage_replay_if_supported",
        "acquisition_if_new_structures_required",
        "teacher_labeling",
        "dataset_split",
        "training",
        "evaluation",
    ],
    RootCauseClass.DEPLOYMENT_STATE_MISMATCH: [
        "state_preparation_recovery",
        "deployment_md",
        "physical_validation",
    ],
    RootCauseClass.PHYSICAL_OBSERVABLE_IMPLEMENTATION_DEFECT: [
        "validation_method_recovery",
        "physical_validation",
    ],
    RootCauseClass.REFERENCE_INADEQUACY: [
        "reference_validation_recovery",
    ],
    RootCauseClass.UNCERTAINTY_CALIBRATION_FAILURE: [
        "calibration_reference_data_recovery",
        "uncertainty",
    ],
    RootCauseClass.FRAMEWORK_EVIDENCE_READABILITY_DEFECT: [
        "governance_or_framework_recovery",
    ],
}


def route_by_root_cause(diagnosis: RootCauseDiagnosis) -> list[str]:
    """Return the admissible return-stage list. A caller MUST NOT hard-code a
    stage number; it must ask this function.
    """
    if diagnosis.admissible_return_stages:
        return list(diagnosis.admissible_return_stages)
    return list(DEFAULT_ROOT_CAUSE_ROUTING.get(diagnosis.root_cause, []))


# =====================================================================
# Public exports
# =====================================================================
__all__ = [
    # enums
    "AdequacyStatus", "ThresholdSourceClass", "CalibrationStatus", "ClaimRole",
    "DeploymentStateRole", "EnsembleKind", "ObservableRole", "RootCauseClass",
    # contracts
    "AdequacyCriterion", "DomainMapping", "DeploymentScopeContractV2",
    "EvaluationAdequacyPolicyV2", "EvaluationAdequacyVerdict",
    "StatePreparationPolicy",
    "ObservableSpec", "PhysicalValidationPolicyV2",
    "UncertaintyPolicyV2",
    "ScientificQuestion", "RootCauseDiagnosis",
    # functions
    "criterion_passes", "evaluate_adequacy", "adjudicate_uncertainty",
    "route_by_root_cause",
    # defaults (data, not values)
    "DEFAULT_SCIENTIFIC_QUESTIONS", "DEFAULT_ROOT_CAUSE_ROUTING",
]
