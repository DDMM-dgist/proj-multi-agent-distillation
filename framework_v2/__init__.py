"""Framework V2 -- typed contracts + generic scientific capabilities.

Public surface. Everything that stages, executors, validators, judges,
and the Controller import lives here (or the specific submodule).

Sub-module map (Sections of the V2 directive in parentheses):

  contracts        -- (Section 2)  the 13 typed contracts A..M
  facts            -- (Section 13) DeterministicFact vs ScientificJudgment,
                                    JUDGE_CONTRADICTION classifier
  decision_ledger  -- (Section 16) append-only, per-decision provenance
  convergence      -- (Section 10) ConvergencePolicy classifier producing
                                    NOT_CONVERGED / CONVERGED_* status
  capability       -- (Section 17) FRAMEWORK_CAPABILITY_BLOCKER negotiator
  evaluation       -- (Section 11) scope-aware evaluation partitioner
  blind_test       -- (Section  4) fail-closed blind-test access guard
  recipe           -- (Section  9) StudentRecipe provenance validator
"""
from __future__ import annotations

from framework_v2.contracts import (  # noqa: F401
    ContractBase,
    utc_now_iso,
    ScopeCategory,
    ProvenanceClass,
    PartitionRole,
    ConvergenceStatus,
    ScopeRegion,
    DeploymentScopeContract,
    ScientificDecisionRecord,
    DomainRegime,
    DomainRepresentation,
    CoveragePlan,
    ParentSelectionPlan,
    PerParentAugPolicy,
    AugmentationPlan,
    DatasetPartitionPlan,
    RecipeParameter,
    StudentRecipePlan,
    ConvergencePolicy,
    EvaluationPolicy,
    UncertaintyPolicy,
    DeploymentMDPolicy,
    PhysicalValidationPolicy,
)

from framework_v2.facts import (  # noqa: F401
    FactVerdict,
    DeterministicFact,
    JudgeClaim,
    ScientificJudgment,
    JudgeContradiction,
    detect_judge_contradictions,
    judgment_usability,
)

from framework_v2.decision_ledger import (  # noqa: F401
    DecisionLedger,
    DecisionLedgerError,
)

from framework_v2.convergence import (  # noqa: F401
    classify_seed_convergence,
    build_convergence_report,
    convergence_gate_ok,
    CONVERGED_EARLY,
    CONVERGED_AT_MAX,
    NOT_CONVERGED,
    INSUFFICIENT_DATA,
)

from framework_v2.capability import (  # noqa: F401
    FRAMEWORK_CAPABILITY_BLOCKER,
    ExecutorCapabilities,
    PlanRequirements,
    FrameworkCapabilityBlocker,
    check_capabilities,
    augmentation_capability_requirements,
)

from framework_v2.evaluation import (  # noqa: F401
    FrameClassification,
    PartitionMetrics,
    EvaluationReport,
    partition_frames,
    count_by_category,
    build_evaluation_report,
    cross_stage_scope_consistent,
)

from framework_v2.evaluation_population import (  # noqa: F401
    EvaluationPopulation,
    EvaluationPopulationRole,
    MultiPopulationEvaluationPlan,
    EvaluationLeakageError,
    assert_no_training_leakage,
    ROLE_ALLOWED_CHANNELS,
    STUDENT_VS_TEACHER,
    STUDENT_VS_DFT,
    TEACHER_VS_DFT,
)

from framework_v2.blind_test import (  # noqa: F401
    BlindTestBoundary,
    BlindTestAccessAttempt,
    BlindTestAccessLog,
    BlindTestAccessViolation,
    guard_blind_access,
    ALLOW,
    DENY,
)

from framework_v2.recipe import (  # noqa: F401
    RecipeProvenanceViolation,
    validate_recipe_provenance,
)

from framework_v2.domain_discovery import (  # noqa: F401
    SourceItem,
    DiscoveryConfig,
    discover_domain,
    primary_regimes_present,
)

from framework_v2.partition_validator import (  # noqa: F401
    PartitionedItem,
    PartitionValidationReport,
    validate_partition,
    PASS_SPLIT,
    REVISE_SPLIT,
    LINEAGE_LEAKAGE,
)

from framework_v2.policy_validators import (  # noqa: F401
    PolicyValidationReport,
    validate_uncertainty,
    validate_deployment_md,
    validate_physical_validation,
)

from framework_v2.evidence_compiler import (  # noqa: F401
    EvidenceCompiler,
    DEFAULT_MAX_EVIDENCE_BYTES,
    fact_to_validation_outcome,
)


__version__ = "2.0.0-alpha"
