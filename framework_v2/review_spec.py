"""Framework V2 — StageReviewSpec (Sections E, G): predeclared, frozen review criteria.

Scientific acceptance criteria must be defined BEFORE the evidence they judge is
inspected. This module provides the typed, versioned StageReviewSpec and a
generic default spec for every canonical stage. The default questions are the
*generic scientific responsibilities* of each stage (directive Section G) — they
contain NO material observables, cutoffs, or ontology. A campaign may extend a
spec with additional application-specific criteria, but doing so mints a new
``spec_version`` (Section E: changing any criterion invalidates prior votes).

Each criterion is owned by exactly one of the three mutually-blind review lenses
(``framework_v2`` binds to the run's lens ids; the canonical three are
``scientific_validity`` (J1), ``evidence_provenance`` (J2),
``reproducibility_deployment`` (J3)). A well-formed spec must give all three
lenses at least one criterion so no required review responsibility is empty.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from .contracts import ContractBase
from .stages import CanonicalStage
from .states import SemanticState

# Canonical lens ids (must match workflow.review_lenses.DEFAULT_REVIEW_LENSES).
LENS_SCIENTIFIC_VALIDITY = "scientific_validity"          # J1
LENS_EVIDENCE_PROVENANCE = "evidence_provenance"          # J2
LENS_REPRODUCIBILITY_DEPLOYMENT = "reproducibility_deployment"  # J3
CANONICAL_LENS_IDS = (
    LENS_SCIENTIFIC_VALIDITY,
    LENS_EVIDENCE_PROVENANCE,
    LENS_REPRODUCIBILITY_DEPLOYMENT,
)

_ALLOWED_FAILURE_STATES = {
    SemanticState.REVISE, SemanticState.FAIL,
    SemanticState.REPRESENTATION_INSUFFICIENT, SemanticState.REVISE_SPLIT,
    SemanticState.NOT_CONVERGED, SemanticState.EVIDENCE_INSUFFICIENT,
}


class ReviewCriterion(ContractBase):
    """One predeclared scientific question a Judge lens must answer."""
    criterion_id: str
    lens_id: str
    question: str
    kind: Literal["quantitative", "qualitative"]
    required_evidence_classes: list[str] = Field(default_factory=list)
    quantitative_rule: Optional[str] = None      # human/machine-readable threshold
    severity: Literal["blocking", "advisory"] = "blocking"
    failure_state: SemanticState = SemanticState.REVISE
    failure_code: str = "unknown"                # registered recovery-taxonomy code
    target_stage: Optional[str] = None           # canonical stage recovery returns to

    @model_validator(mode="after")
    def _valid_failure_state(self):
        if self.failure_state not in _ALLOWED_FAILURE_STATES:
            raise ValueError(
                f"criterion {self.criterion_id}: failure_state {self.failure_state} "
                f"is not a recovery-bearing verdict"
            )
        return self

    @model_validator(mode="after")
    def _failure_code_registered(self):
        from workflow.recovery_taxonomy import resolve_failure_code
        try:
            resolve_failure_code(self.failure_code)   # fail closed on unregistered
        except KeyError as exc:
            # surface as a ValueError so pydantic reports a clean ValidationError
            # (consistent with framework_v2.recovery.RecoveryPlan)
            raise ValueError(str(exc)) from exc
        return self


class StageReviewSpec(ContractBase):
    """Frozen, versioned review criteria for one stage (Section E)."""
    spec_id: str
    stage: str
    validation_profile_version: int
    lens_ids: tuple[str, str, str] = CANONICAL_LENS_IDS
    criteria: list[ReviewCriterion]
    allowed_verdicts: tuple[str, ...] = ("PASS", "REVISE", "FAIL")
    spec_version: int = 1

    def criteria_for_lens(self, lens_id: str) -> list[ReviewCriterion]:
        return [c for c in self.criteria if c.lens_id == lens_id]

    def blocking_criteria(self) -> list[ReviewCriterion]:
        return [c for c in self.criteria if c.severity == "blocking"]

    @model_validator(mode="after")
    def _unique_criteria(self):
        ids = [c.criterion_id for c in self.criteria]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate ReviewCriterion.criterion_id: {dupes}")
        return self

    @model_validator(mode="after")
    def _lenses_covered(self):
        """Every declared lens must own at least one criterion, and every
        criterion's lens must be a declared lens (Section F: all three review
        responsibilities must be satisfiable)."""
        declared = set(self.lens_ids)
        used = {c.lens_id for c in self.criteria}
        stray = sorted(used - declared)
        if stray:
            raise ValueError(f"criteria reference undeclared lens ids: {stray}")
        empty = sorted(declared - used)
        if empty:
            raise ValueError(f"these review lenses have no criterion: {empty}")
        return self


# =====================================================================
# Generic default specs for the 12 canonical stages (Section G).
# Questions are generic scientific responsibilities — no material specifics.
# =====================================================================
def _c(cid, lens, question, *, kind="qualitative", evidence=(), rule=None,
       severity="blocking", state=SemanticState.REVISE, code="unknown",
       target=None) -> ReviewCriterion:
    return ReviewCriterion(
        criterion_id=cid, lens_id=lens, question=question, kind=kind,
        required_evidence_classes=list(evidence), quantitative_rule=rule,
        severity=severity, failure_state=state, failure_code=code, target_stage=target)


def default_stage_review_specs(validation_profile_version: int = 1) -> dict[str, StageReviewSpec]:
    """Return the 12 generic StageReviewSpecs keyed by canonical stage value."""
    S = CanonicalStage
    V = LENS_SCIENTIFIC_VALIDITY
    E = LENS_EVIDENCE_PROVENANCE
    R = LENS_REPRODUCIBILITY_DEPLOYMENT
    St = SemanticState

    specs: dict[str, list[ReviewCriterion]] = {
        S.TEACHER_BASELINE.value: [
            _c("tb-applicability", V,
               "Is this Teacher applicable to the intended deployment domain, and are its "
               "known limits compatible with the proposed use?",
               evidence=("teacher_identity", "teacher_applicability"),
               code="teacher_applicability", target=S.TEACHER_BASELINE.value),
            _c("tb-provenance", E,
               "Is Teacher identity/version/hash and any reused historical Teacher evidence "
               "verified and transferable to this campaign scope?",
               evidence=("teacher_identity", "provenance"),
               code="teacher_applicability", target=S.TEACHER_BASELINE.value),
            _c("tb-downstream", R,
               "Could relying on this Teacher baseline create downstream fidelity or scope "
               "failures given its training distribution?",
               evidence=("teacher_applicability",),
               code="teacher_distribution_coverage", target=S.TEACHER_BASELINE.value),
        ],
        S.REFERENCE_VALIDATION.value: [
            _c("rv-anchors", V,
               "Are validation targets and reference anchors sufficient and representative "
               "of the intended claim?",
               evidence=("reference_data", "validation_profile"),
               code="reference_disagreement", target=S.REFERENCE_VALIDATION.value),
            _c("rv-disagreement", E,
               "Is Teacher-vs-reference disagreement quantified and understood well enough "
               "for downstream claims?",
               evidence=("reference_data",),
               code="reference_disagreement", target=S.REFERENCE_VALIDATION.value),
            _c("rv-scope", R,
               "Do the reference structures stay within, and adequately span, the deployment "
               "scope they will be used to certify?",
               evidence=("reference_data", "scope_contract"),
               code="reference_disagreement", target=S.REFERENCE_VALIDATION.value),
        ],
        S.ACQUISITION.value: [
            _c("aq-representation", V,
               "Is the chosen representation itself adequate for the relevant configuration "
               "space, or does good internal coverage merely reflect an inadequate "
               "representation?",
               evidence=("representation_adequacy",),
               state=St.REPRESENTATION_INSUFFICIENT,
               code="representation_insufficient", target=S.ACQUISITION.value),
            _c("aq-evidence", E,
               "Does the acquisition proposal (parents, method, budget) rest on cited "
               "evidence and consider meaningful alternatives?",
               evidence=("parent_selection", "coverage_plan"),
               code="dataset_coverage", target=S.ACQUISITION.value),
            _c("aq-diversity", R,
               "Does the acquisition target scientifically relevant diversity rather than "
               "redundant regions, and is the replay/data-mixing strategy justified?",
               evidence=("coverage_plan", "replay_strategy"),
               code="replay_strategy_unjustified", target=S.ACQUISITION.value),
            _c("aq-objective-consistency", V,
               "Is the autonomously-designed acquisition plan actually consistent with the "
               "campaign objective — do the targeted regimes, selected strategy/backends, and "
               "generated-then-selected candidates serve the declared primary target and claim "
               "scope rather than acquiring out-of-target configurations?",
               evidence=("scope_contract", "coverage_plan", "parent_selection"),
               code="dataset_coverage", target=S.ACQUISITION.value),
        ],
        S.DATA_COVERAGE.value: [
            _c("dc-meaningful", V,
               "Does measured coverage reflect scientifically meaningful configuration space, "
               "and are rare/boundary/sparse environments sufficiently treated?",
               evidence=("coverage_plan", "representation_adequacy"),
               code="dataset_coverage", target=S.DATA_COVERAGE.value),
            _c("dc-representation", E,
               "Could excellent numerical coverage simply reflect an unjustified "
               "representation (is representation adequacy evidenced)?",
               evidence=("representation_adequacy",),
               state=St.REPRESENTATION_INSUFFICIENT,
               code="representation_insufficient", target=S.ACQUISITION.value),
            _c("dc-teacher-cov", R,
               "Is Student distillation coverage assessed against the Teacher training "
               "distribution where data are available, and is stopping justified?",
               evidence=("teacher_distribution_coverage",),
               code="teacher_distribution_coverage", target=S.DATA_COVERAGE.value),
        ],
        S.TEACHER_LABELING.value: [
            _c("tl-applicability", V,
               "Are the selected structures within justified Teacher applicability?",
               evidence=("teacher_applicability", "scope_contract"),
               code="teacher_applicability", target=S.TEACHER_LABELING.value),
            _c("tl-usable", E,
               "Are the labels scientifically usable for the stated distillation purpose?",
               evidence=("labeling_manifest",),
               code="teacher_applicability", target=S.TEACHER_LABELING.value),
            _c("tl-scope", R,
               "Are systematically unsupported populations being labeled (scope leakage)?",
               evidence=("scope_contract", "labeling_manifest"),
               code="dataset_coverage", target=S.TEACHER_LABELING.value),
        ],
        S.DATASET_SPLIT.value: [
            _c("ds-representative", V,
               "Is the split scientifically representative of the deployment manifold as well "
               "as lineage-safe?",
               evidence=("partition_plan", "partition_validation"),
               state=St.REVISE_SPLIT,
               code="split_unrepresentative", target=S.DATASET_SPLIT.value),
            _c("ds-lineage", E,
               "Is the split lineage-safe with zero train/val/blind leakage?",
               evidence=("partition_validation",),
               code="lineage_or_leakage", target=S.DATASET_SPLIT.value),
            _c("ds-bias", R,
               "Could model selection be biased by the partition (continuous/sparse regions "
               "absent or distorted)?",
               evidence=("partition_validation",),
               state=St.REVISE_SPLIT,
               code="split_unrepresentative", target=S.DATASET_SPLIT.value),
        ],
        S.TRAINING.value: [
            _c("tr-recipe", V,
               "Is the Student recipe (architecture/hyperparameters/loss) appropriate to the "
               "target and data, with justified provenance?",
               evidence=("student_recipe",),
               code="student_fidelity", target=S.TRAINING.value),
            _c("tr-convergence", E,
               "Is convergence established from train/val trajectories (not epoch==max), with "
               "no undertraining/overfitting/seed instability unresolved?",
               evidence=("convergence_report",),
               state=St.NOT_CONVERGED,
               code="training_instability", target=S.TRAINING.value),
            _c("tr-stability", R,
               "Is the training numerically stable and reproducible across seeds?",
               evidence=("convergence_report",),
               code="training_instability", target=S.TRAINING.value),
        ],
        S.EVALUATION.value: [
            _c("ev-scope", V,
               "Does the evaluated population match deployment scope, and are Student<->Teacher, "
               "Teacher<->reference, Student<->reference channels distinguished correctly?",
               evidence=("evaluation_report", "scope_contract"),
               code="student_fidelity", target=S.EVALUATION.value),
            _c("ev-metrics", E,
               "Are metrics sufficient (absolute and normalized scales) and not hiding "
               "scientifically important failure behind aggregates?",
               evidence=("evaluation_report",),
               code="student_fidelity", target=S.EVALUATION.value),
            _c("ev-mixed", R,
               "Are mixed-scope aggregates prevented from becoming the primary metric?",
               evidence=("evaluation_report", "evaluation_policy"),
               code="data_coverage", target=S.EVALUATION.value),
        ],
        S.UNCERTAINTY.value: [
            _c("un-detects", V,
               "Does uncertainty detect relevant extrapolation/failure and is it "
               "scientifically interpretable/calibrated?",
               evidence=("uncertainty_report",),
               code="student_fidelity", target=S.UNCERTAINTY.value),
            _c("un-evidence", E,
               "Is calibration evidence present rather than merely a computed number?",
               evidence=("uncertainty_report",),
               code="missing_evidence", target=S.UNCERTAINTY.value),
            _c("un-confident-wrong", R,
               "Could confidently wrong predictions remain invisible in deployment?",
               evidence=("uncertainty_report",),
               code="student_fidelity", target=S.UNCERTAINTY.value),
        ],
        S.DEPLOYMENT_MD.value: [
            _c("md-usecase", V,
               "Does the MD protocol actually test the declared production use case?",
               evidence=("md_report", "md_policy", "validation_profile"),
               code="simulation_protocol", target=S.DEPLOYMENT_MD.value),
            _c("md-stability-vs-accuracy", E,
               "Is numerical MD stability being confused with physical accuracy?",
               evidence=("md_report",),
               code="simulation_instability", target=S.DEPLOYMENT_MD.value),
            _c("md-domain", R,
               "Does trajectory sampling leave the supported domain (extrapolation)?",
               evidence=("md_report", "uncertainty_report"),
               code="simulation_protocol", target=S.DEPLOYMENT_MD.value),
        ],
        S.PHYSICAL_VALIDATION.value: [
            _c("pv-sufficient", V,
               "Are the chosen observables sufficient for the ValidationProfile, and are the "
               "required behaviors tested?",
               evidence=("physical_validation_report", "validation_profile"),
               code="physical_validation", target=S.PHYSICAL_VALIDATION.value),
            _c("pv-evidence", E,
               "Are results traceable to registered artifacts and deterministic checks?",
               evidence=("physical_validation_report",),
               code="physical_validation", target=S.PHYSICAL_VALIDATION.value),
            _c("pv-unsupported", R,
               "Are unsupported properties being incorrectly used as acceptance criteria "
               "(which would fail a narrower valid claim)?",
               evidence=("validation_profile", "physical_validation_report"),
               code="physical_validation", target=S.PHYSICAL_VALIDATION.value),
        ],
        S.ANALYSIS.value: [
            _c("an-bounded", V,
               "Are final claims bounded by validated evidence and are limitations propagated?",
               evidence=("run_summary", "validation_profile"),
               code="missing_evidence", target=S.ANALYSIS.value),
            _c("an-traceable", E,
               "Is every claim traceable through the DecisionLedger without terminal "
               "archaeology?",
               evidence=("run_summary", "decision_ledger"),
               code="missing_evidence", target=S.ANALYSIS.value),
            _c("an-loop-claim", R,
               "Is the workflow called closed-loop / active-learning only when an actual "
               "corrective loop was executed?",
               evidence=("run_summary", "recovery_history"),
               code="evidence_gap", target=S.ANALYSIS.value),
        ],
    }

    out: dict[str, StageReviewSpec] = {}
    for stage_value, crits in specs.items():
        out[stage_value] = StageReviewSpec(
            spec_id=f"review-spec-{stage_value}",
            stage=stage_value,
            validation_profile_version=validation_profile_version,
            criteria=crits,
        )
    return out
