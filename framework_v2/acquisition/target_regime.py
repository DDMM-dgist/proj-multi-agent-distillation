"""Framework V2 -- TargetRegimeModel builder (generic).

The target-regime model expresses *what regimes the campaign target implies*
and each regime's target-relative relevance role. It is derived deterministically
from the frozen ``DeploymentScopeContract`` bound to the objective -- the core
does not invent regimes, elements, or compositions.

The mapping from a generic scope *category* to a target-relative *relevance
role* is itself generic and material-agnostic:

    PRIMARY_DEPLOYMENT   -> CORE_TARGET
    AUXILIARY_SUPPORT    -> ADJACENT_PHYSICS
    HISTORICAL_BENCHMARK -> BOUNDARY_GUARDRAIL
    OUT_OF_SCOPE         -> OUT_OF_TARGET_ACQUISITION

PROTECTED_REFERENCE and BLIND_TEST regions are deliberately excluded from the
target-regime model: they are protected populations acquisition must stay
disjoint from, never regimes to acquire into. GENERATION_PATHWAY is a role the
strategy planner assigns to a regime when a chosen backend *reaches* the target
through it (e.g. a melt/quench pathway); it is not derivable from the scope
category alone, so it is never assigned here.
"""
from __future__ import annotations

from framework_v2.acquisition.contracts import (
    RelevanceRole,
    TargetRegime,
    TargetRegimeModel,
)
from framework_v2.contracts import DeploymentScopeContract, ScopeCategory


# Generic, material-agnostic category -> role mapping.
_CATEGORY_TO_ROLE: dict[ScopeCategory, RelevanceRole] = {
    ScopeCategory.PRIMARY_DEPLOYMENT: RelevanceRole.CORE_TARGET,
    ScopeCategory.AUXILIARY_SUPPORT: RelevanceRole.ADJACENT_PHYSICS,
    ScopeCategory.HISTORICAL_BENCHMARK: RelevanceRole.BOUNDARY_GUARDRAIL,
    ScopeCategory.OUT_OF_SCOPE: RelevanceRole.OUT_OF_TARGET_ACQUISITION,
}

# Categories that are protected populations, never acquisition target regimes.
_EXCLUDED_CATEGORIES = frozenset(
    {ScopeCategory.PROTECTED_REFERENCE, ScopeCategory.BLIND_TEST}
)


def build_target_regime_model(
    *,
    model_id: str,
    objective_sha256: str,
    descriptor: str,
    scope_contract: DeploymentScopeContract,
) -> TargetRegimeModel:
    """Derive the target-regime model from the bound scope contract.

    Deterministic. Fails closed (via TargetRegimeModel's own validator) if the
    scope contract yields no CORE_TARGET regime."""
    regimes: list[TargetRegime] = []
    for region in scope_contract.regions:
        if region.category in _EXCLUDED_CATEGORIES:
            continue
        role = _CATEGORY_TO_ROLE[region.category]
        regimes.append(
            TargetRegime(
                regime_id=region.region_id,
                label=region.category.value,
                relevance_role=role,
                membership_rule=region.membership_rule,
                evidence_refs=list(region.membership_evidence),
            )
        )

    return TargetRegimeModel(
        model_id=model_id,
        objective_sha256=objective_sha256,
        descriptor=descriptor,
        regimes=regimes,
    )
