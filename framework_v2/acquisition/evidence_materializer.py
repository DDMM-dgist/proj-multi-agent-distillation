"""Framework V2 -- AUTO-MATERIALIZED acquisition coverage evidence.

Composes the generic deterministic acquisition pipeline around a plugin-supplied
``DescriptorSpaceEvidence`` bundle and freezes the whole evidence chain into ONE typed,
content-addressed artifact (``AutoMaterializedAcquisitionEvidence``). This is the framework-owned
boundary the framework-default acquisition provider consumes: the material-specific descriptor work
lives entirely in the plugin, the generic pipeline (inventory / target-regime / region / coverage /
strategy) + the freeze/hash live here, and neither depends on any particular material.

Fails closed with a typed ``AcquisitionCapabilityGap`` when the campaign yields no admissible
parents or no admissible strategy backend -- never a silent no-op, never a fabricated fallback.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

from pydantic import Field

from framework_v2.acquisition.contracts import (
    AcquisitionStrategy,
    BackendCapabilityRecord,
    CampaignObjective,
    CoverageGapAnalysis,
    RegionResolution,
    RegionResolutionMode,
    SourceAndCapabilityInventory,
    TargetRegimeModel,
    TeacherCapabilityRecord,
)
from framework_v2.acquisition.coverage_gap import build_coverage_gap_analysis
from framework_v2.acquisition.descriptor_plugins import (
    AcquisitionCapabilityGap,
    DescriptorSpaceEvidence,
)
from framework_v2.acquisition.region_resolution import resolve_regions
from framework_v2.acquisition.strategy import StrategyUndecidable, select_strategy
from framework_v2.acquisition.target_regime import build_target_regime_model
from framework_v2.contracts import ContractBase, DeploymentScopeContract


class AutoMaterializedAcquisitionEvidence(ContractBase):
    """The frozen, content-addressed record of one auto-materialization: it binds every deterministic
    upstream artifact by content-SHA and records the admissible decision space the Agent must plan
    within. Its ``content_sha256()`` is the single freeze hash the default provider + the exact
    record/replay machinery pin the autonomous acquisition planning invocation to."""
    evidence_id: str
    material_id: str
    objective_sha256: str
    inventory_sha256: str
    target_regime_model_sha256: str
    region_resolution_sha256: str
    region_resolution_mode: str
    coverage_gap_sha256: str
    strategy_sha256: str
    strategy_kind: str
    descriptor: str
    teacher_identity_sha256: str
    admissible_parent_ids: list[str] = Field(default_factory=list)
    required_param_keys: list[str] = Field(default_factory=list)
    param_bounds: dict[str, list[float]] = Field(default_factory=dict)
    eligible_source_categories: list[str] = Field(default_factory=list)
    selected_source_global_indices: list[int] = Field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class MaterializedAcquisitionEvidence:
    """The live evidence-chain objects (built ONCE so their content-SHAs are stable) plus the frozen
    artifact. The default provider maps this straight onto its ``AcquisitionPlanningContext``."""
    objective: CampaignObjective
    inventory: SourceAndCapabilityInventory
    target_regime_model: TargetRegimeModel
    region_resolution: RegionResolution
    coverage: CoverageGapAnalysis
    strategy: AcquisitionStrategy
    admissible_parent_ids: tuple[str, ...]
    teacher_identity_sha256: str
    required_param_keys: tuple[str, ...]
    param_bounds: dict[str, tuple[float, float]]
    eligible_source_categories: tuple[str, ...]
    selected_source_global_indices: tuple[int, ...]
    descriptor_evidence: DescriptorSpaceEvidence
    frozen_artifact: AutoMaterializedAcquisitionEvidence


def materialize_acquisition_evidence(
    *,
    id_prefix: str,
    material_id: str,
    objective: CampaignObjective,
    scope_contract: DeploymentScopeContract,
    descriptor_evidence: DescriptorSpaceEvidence,
    backend_records: Sequence[BackendCapabilityRecord],
    teacher_record: TeacherCapabilityRecord,
) -> MaterializedAcquisitionEvidence:
    """Deterministically assemble + freeze the full acquisition evidence chain from plugin evidence.

    Every object is constructed exactly once and reused so its content-SHA is stable (each contract
    carries an ``established_at`` default that would otherwise drift on reconstruction)."""
    if not descriptor_evidence.admissible_parent_ids:
        raise AcquisitionCapabilityGap(
            f"descriptor provider {material_id!r} yielded an empty admissible parent pool; the "
            "framework will not fabricate parents or prompt for them",
            gap_kind="NO_ADMISSIBLE_PARENTS",
        )

    obj_sha = objective.content_sha256()
    inventory = SourceAndCapabilityInventory(
        inventory_id=f"{id_prefix}-inventory",
        objective_sha256=obj_sha,
        sources=list(descriptor_evidence.source_records),
        backends=list(backend_records),
        teacher=teacher_record,
    )
    target_regime_model = build_target_regime_model(
        model_id=f"{id_prefix}-target-regime",
        objective_sha256=obj_sha,
        descriptor=descriptor_evidence.descriptor,
        scope_contract=scope_contract,
    )
    region_resolution = resolve_regions(
        resolution_id=f"{id_prefix}-region-resolution",
        metadata_present=descriptor_evidence.metadata_present,
        discovered_representation_builder=descriptor_evidence.discovered_representation_builder,
        auditor=descriptor_evidence.metadata_auditor,
        declared_representation_builder=descriptor_evidence.declared_representation_builder,
    )
    coverage = build_coverage_gap_analysis(
        analysis_id=f"{id_prefix}-coverage",
        phase=objective.phase,
        target_regime_model_sha256=target_regime_model.content_sha256(),
        region_resolution_sha256=region_resolution.content_sha256(),
        regime_inputs=list(descriptor_evidence.regime_coverage_inputs),
        saturation_threshold=descriptor_evidence.saturation_threshold,
        available_source_coverage=descriptor_evidence.available_source_coverage,
    )
    try:
        strategy = select_strategy(
            strategy_id=f"{id_prefix}-strategy",
            inventory=inventory,
            coverage=coverage,
            evidence=descriptor_evidence.strategy_evidence,
            evidence_refs=[coverage.content_sha256(), region_resolution.domain_representation_sha256],
        )
    except StrategyUndecidable as exc:
        raise AcquisitionCapabilityGap(
            f"no admissible acquisition strategy for material {material_id!r}: {exc}",
            gap_kind="STRATEGY_UNDECIDABLE",
        ) from exc

    frozen = AutoMaterializedAcquisitionEvidence(
        evidence_id=f"{id_prefix}-auto-materialized-evidence",
        material_id=material_id,
        objective_sha256=obj_sha,
        inventory_sha256=inventory.content_sha256(),
        target_regime_model_sha256=target_regime_model.content_sha256(),
        region_resolution_sha256=region_resolution.content_sha256(),
        region_resolution_mode=region_resolution.mode.value,
        coverage_gap_sha256=coverage.content_sha256(),
        strategy_sha256=strategy.content_sha256(),
        strategy_kind=strategy.kind.value,
        descriptor=descriptor_evidence.descriptor,
        teacher_identity_sha256=teacher_record.identity_sha256,
        admissible_parent_ids=list(descriptor_evidence.admissible_parent_ids),
        required_param_keys=list(descriptor_evidence.required_param_keys),
        param_bounds={k: [float(v[0]), float(v[1])]
                      for k, v in descriptor_evidence.param_bounds.items()},
        eligible_source_categories=list(descriptor_evidence.eligible_source_categories),
        selected_source_global_indices=list(descriptor_evidence.selected_source_global_indices),
    )
    return MaterializedAcquisitionEvidence(
        objective=objective,
        inventory=inventory,
        target_regime_model=target_regime_model,
        region_resolution=region_resolution,
        coverage=coverage,
        strategy=strategy,
        admissible_parent_ids=tuple(descriptor_evidence.admissible_parent_ids),
        teacher_identity_sha256=teacher_record.identity_sha256,
        required_param_keys=tuple(descriptor_evidence.required_param_keys),
        param_bounds=dict(descriptor_evidence.param_bounds),
        eligible_source_categories=tuple(descriptor_evidence.eligible_source_categories),
        selected_source_global_indices=tuple(descriptor_evidence.selected_source_global_indices),
        descriptor_evidence=descriptor_evidence,
        frozen_artifact=frozen,
    )
