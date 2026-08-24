"""Framework V2 -- AcquisitionPlanV2 assembly + legacy projection.

Ties every upstream evidence artifact together by content-SHA into the bound
``AcquisitionPlanV2`` and, for the LOCAL_PERTURBATION strategy, projects the
plan into the exact 14-field legacy ``AcquisitionPlan`` dict that
``executors._validate_acquisition_plan`` consumes unchanged -- so the existing
fail-closed executor validation is preserved verbatim, not weakened. For
strategies that the legacy perturbation executor cannot run (e.g.
TEACHER_DRIVEN_MD) the plan carries the dynamics protocol's content-SHA instead
of a legacy projection.
"""
from __future__ import annotations

from typing import Any, Optional

from framework_v2.acquisition.contracts import (
    AcquisitionPhase,
    AcquisitionPlanV2,
    AcquisitionStrategy,
    CandidateGenerationResult,
    CandidateSelectionResult,
    CanonicalLabelingRequest,
    CampaignObjective,
    CoverageGapAnalysis,
    RegionResolution,
    SourceAndCapabilityInventory,
    TargetRegimeModel,
)
from framework_v2.acquisition.generators.base import GenerationProtocol

# The 14 required legacy fields (mirrors executors._REQUIRED_ACQUISITION_PLAN_FIELDS).
LEGACY_REQUIRED_FIELDS = (
    "schema_version",
    "eligible_source_categories",
    "selected_parent_structure_ids",
    "selected_source_global_indices",
    "n_parents",
    "n_per_structure",
    "T_K",
    "beta",
    "sigma_range_A",
    "cell_sigma",
    "seed",
    "expected_output_count",
    "duplicate_handling",
    "protected_reference_exclusion_report",
)


def build_legacy_projection(
    *,
    protocol: GenerationProtocol,
    selection_result: CandidateSelectionResult,
    eligible_source_categories: list[str],
    selected_source_global_indices: list[int],
    duplicate_handling: str,
) -> dict[str, Any]:
    """Project a LOCAL_PERTURBATION plan into the 14-field legacy dict.

    All scientific values come from the autonomously-designed protocol +
    selection evidence -- nothing is invented here. The disjointness report is
    carried straight through from the selector (status must be PASS)."""
    p = protocol.params
    parent_ids = list(protocol.parent_ids)
    n_parents = len(parent_ids)
    n_per_structure = int(p["n_per_structure"])
    report = selection_result.disjointness_report
    projection = {
        "schema_version": 1,
        "eligible_source_categories": list(eligible_source_categories),
        "selected_parent_structure_ids": parent_ids,
        "selected_source_global_indices": list(selected_source_global_indices),
        "n_parents": n_parents,
        "n_per_structure": n_per_structure,
        "T_K": float(p["T_K"]),
        "beta": float(p["beta"]),
        "sigma_range_A": [float(p["sigma_range_A"][0]), float(p["sigma_range_A"][1])],
        "cell_sigma": None if p["cell_sigma"] is None else float(p["cell_sigma"]),
        "seed": int(p["seed"]),
        "expected_output_count": n_parents * n_per_structure,
        "duplicate_handling": duplicate_handling,
        "protected_reference_exclusion_report": {
            "status": report.status,
            "n_checked": report.n_checked,
            "n_overlaps": report.n_overlaps,
            "dft_labels_used_as_selection_scores": (
                report.dft_labels_used_as_selection_scores
            ),
        },
    }
    return projection


# The required fields of an EXISTING_POOL_SELECTION projection (mirrors
# executors._REQUIRED_EXISTING_POOL_PLAN_FIELDS).
EXISTING_POOL_REQUIRED_FIELDS = (
    "schema_version",
    "pool_path",
    "eligible_source_categories",
    "selected_parent_structure_ids",
    "selected_source_global_indices",
    "n_selected",
    "duplicate_handling",
    "labeling_population_sizing",
    "protected_reference_exclusion_report",
)


def build_existing_pool_projection(
    *,
    pool_path: str,
    eligible_source_categories: list[str],
    selected_parent_structure_ids: list[str],
    selected_source_global_indices: list[int],
    labeling_population_sizing: dict[str, Any],
    selection_result: CandidateSelectionResult,
    duplicate_handling: str,
) -> dict[str, Any]:
    """Project an EXISTING_POOL_SELECTION plan into its executable dict.

    No new configurations are generated: the selected EXISTING frames are named by
    their global indices into ``pool_path`` (stable pool order) and their parent ids.
    Every count is DERIVED from the deterministic ``labeling_population_sizing``
    evidence, never invented. The disjointness report is carried straight through
    from the selector (status must be PASS)."""
    report = selection_result.disjointness_report
    n_selected = len(selected_source_global_indices)
    if len(selected_parent_structure_ids) != n_selected:
        raise ValueError(
            "build_existing_pool_projection: selected_parent_structure_ids and "
            "selected_source_global_indices must have equal length")
    return {
        "schema_version": 1,
        "pool_path": str(pool_path),
        "eligible_source_categories": list(eligible_source_categories),
        "selected_parent_structure_ids": list(selected_parent_structure_ids),
        "selected_source_global_indices": [int(i) for i in selected_source_global_indices],
        "n_selected": n_selected,
        "expected_output_count": n_selected,
        "duplicate_handling": duplicate_handling,
        "labeling_population_sizing": dict(labeling_population_sizing),
        "protected_reference_exclusion_report": {
            "status": report.status,
            "n_checked": report.n_checked,
            "n_overlaps": report.n_overlaps,
            "dft_labels_used_as_selection_scores": (
                report.dft_labels_used_as_selection_scores
            ),
        },
    }


def assemble_plan_v2(
    *,
    plan_id: str,
    objective: CampaignObjective,
    inventory: SourceAndCapabilityInventory,
    target_regime_model: TargetRegimeModel,
    region_resolution: RegionResolution,
    coverage: CoverageGapAnalysis,
    strategy: AcquisitionStrategy,
    generation_result: CandidateGenerationResult,
    selection_result: CandidateSelectionResult,
    labeling_request: CanonicalLabelingRequest,
    legacy_projection: Optional[dict[str, Any]] = None,
    dynamics_protocol_sha256: Optional[str] = None,
    existing_pool_projection: Optional[dict[str, Any]] = None,
) -> AcquisitionPlanV2:
    """Bind the full evidence chain into an AcquisitionPlanV2.

    Exactly one execution projection (legacy_projection / dynamics_protocol_sha256 /
    existing_pool_projection) must be supplied -- enforced by the contract."""
    return AcquisitionPlanV2(
        plan_id=plan_id,
        objective_sha256=objective.content_sha256(),
        inventory_sha256=inventory.content_sha256(),
        target_regime_model_sha256=target_regime_model.content_sha256(),
        region_resolution_sha256=region_resolution.content_sha256(),
        coverage_gap_sha256=coverage.content_sha256(),
        strategy_sha256=strategy.content_sha256(),
        generation_result_sha256=generation_result.content_sha256(),
        selection_result_sha256=selection_result.content_sha256(),
        labeling_request_sha256=labeling_request.content_sha256(),
        phase=objective.phase,
        legacy_projection=legacy_projection,
        dynamics_protocol_sha256=dynamics_protocol_sha256,
        existing_pool_projection=existing_pool_projection,
    )
