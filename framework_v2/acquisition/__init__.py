"""Framework V2 -- generic autonomous acquisition planner.

This package replaces the human-supplied 14-field ``AcquisitionPlan`` with an
objective-conditioned autonomous pipeline: given only a high-level
``CampaignObjective`` (target, claim scope, optional compute ceiling), it
inventories sources/backends, models the target regimes, resolves regions under
a tiered-trust metadata policy, analyzes coverage gaps, autonomously selects a
generation strategy, generates + diversity-selects candidates, and emits a
content-addressable ``AcquisitionPlanV2`` that chains the whole decision path
for audit. Low-level acquisition knobs are Agent-autonomous, checked by
deterministic validators (+ a downstream objective-consistency Judge criterion
and bounded semantic-correction retry).
"""
from framework_v2.acquisition.contracts import (
    AcquisitionPhase,
    AcquisitionPlanV2,
    AcquisitionStrategy,
    AcquisitionStrategyKind,
    BackendCapabilityRecord,
    CampaignObjective,
    CandidateGenerationResult,
    CandidateSelectionResult,
    CanonicalLabelingRequest,
    ComputeCeiling,
    CoverageGapAnalysis,
    GenerationProvenance,
    MetadataAuditVerdict,
    MetadataConsistencyAudit,
    MetadataConsistencyCheck,
    ProtectedDisjointnessReport,
    RegimeCoverage,
    RegionResolution,
    RegionResolutionMode,
    RelevanceRole,
    SourceAndCapabilityInventory,
    SourceCategoryRecord,
    TargetRegime,
    TargetRegimeModel,
    TeacherCapabilityRecord,
)

__all__ = [
    "AcquisitionPhase",
    "AcquisitionPlanV2",
    "AcquisitionStrategy",
    "AcquisitionStrategyKind",
    "BackendCapabilityRecord",
    "CampaignObjective",
    "CandidateGenerationResult",
    "CandidateSelectionResult",
    "CanonicalLabelingRequest",
    "ComputeCeiling",
    "CoverageGapAnalysis",
    "GenerationProvenance",
    "MetadataAuditVerdict",
    "MetadataConsistencyAudit",
    "MetadataConsistencyCheck",
    "ProtectedDisjointnessReport",
    "RegimeCoverage",
    "RegionResolution",
    "RegionResolutionMode",
    "RelevanceRole",
    "SourceAndCapabilityInventory",
    "SourceCategoryRecord",
    "TargetRegime",
    "TargetRegimeModel",
    "TeacherCapabilityRecord",
]
