"""Framework V2 — the single authoritative ordered list of canonical stages.

The Controller is data-driven from each run's ``workflow.yaml`` (a run may omit
or reorder stages with typed justification), but the *canonical* scientific
lifecycle is a framework invariant, not a per-run opinion. This module is the
one place that invariant is declared, so review specs, dependency graphs, and
portability audits all refer to the same ordered set of 12 stages.

Nothing here is material-specific: these are the generic phases of any MLIP
knowledge-distillation campaign.
"""
from __future__ import annotations

from enum import Enum


class CanonicalStage(str, Enum):
    TEACHER_BASELINE = "teacher_baseline"
    REFERENCE_VALIDATION = "reference_validation"
    ACQUISITION = "acquisition"
    DATA_COVERAGE = "data_coverage"
    TEACHER_LABELING = "teacher_labeling"
    DATASET_SPLIT = "dataset_split"
    TRAINING = "training"
    EVALUATION = "evaluation"
    UNCERTAINTY = "uncertainty"
    DEPLOYMENT_MD = "deployment_md"
    PHYSICAL_VALIDATION = "physical_validation"
    ANALYSIS = "analysis"


# Canonical scientific order (Section B). teacher_baseline/reference_validation
# come first so the Teacher reference is established BEFORE Student design
# (Section C: Teacher-first).
CANONICAL_STAGE_ORDER: tuple[CanonicalStage, ...] = (
    CanonicalStage.TEACHER_BASELINE,
    CanonicalStage.REFERENCE_VALIDATION,
    CanonicalStage.ACQUISITION,
    CanonicalStage.DATA_COVERAGE,
    CanonicalStage.TEACHER_LABELING,
    CanonicalStage.DATASET_SPLIT,
    CanonicalStage.TRAINING,
    CanonicalStage.EVALUATION,
    CanonicalStage.UNCERTAINTY,
    CanonicalStage.DEPLOYMENT_MD,
    CanonicalStage.PHYSICAL_VALIDATION,
    CanonicalStage.ANALYSIS,
)

# The stage boundary before which no Teacher inference (expensive labeling) may
# run without human approval. Everything strictly before this in the canonical
# order is "cheap planning" (Section AL).
FIRST_EXPENSIVE_STAGE: CanonicalStage = CanonicalStage.TEACHER_LABELING


def stage_index(stage: str) -> int:
    """Position of a stage in the canonical order, or -1 if not canonical."""
    for i, s in enumerate(CANONICAL_STAGE_ORDER):
        if s.value == stage:
            return i
    return -1


def is_cheap_planning_stage(stage: str) -> bool:
    """True if the stage is strictly before the first expensive (Teacher
    inference) stage in the canonical order."""
    idx = stage_index(stage)
    if idx < 0:
        return False
    return idx < stage_index(FIRST_EXPENSIVE_STAGE.value)
