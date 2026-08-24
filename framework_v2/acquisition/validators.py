"""Framework V2 -- deterministic acquisition-plan validators.

These mirror the Teacher-validation planner's ``proposal_validator``: a purely
deterministic gate the autonomously-designed ``AcquisitionPlanV2`` must pass
before it can bind to a run. The Judge's objective-consistency criterion is a
*separate* semantic check; this module owns only the mechanical, reproducible
invariants:

  1. Full evidence-chain integrity -- every ``*_sha256`` on the plan matches
     the content-SHA of the artifact it claims to reference (no dangling or
     forged references).
  2. INITIAL-phase evidence contract (binding #3) -- an INITIAL plan may not
     carry model-uncertainty / expected-information-gain / Student-derived
     evidence anywhere in its evidence dicts or rationale.
  3. Provenance separation (Section K) -- all generation provenance is marked
     exploration-only; selected ids are a subset of generated ids; the
     labeling request re-labels canonically and targets exactly the selected
     ids.
  4. Fail-closed protected disjointness -- selection disjointness is PASS with
     zero overlaps and DFT labels were not used as selection scores.
  5. Execution-projection consistency -- exactly one of the three projections
     (legacy / dynamics-protocol-sha / existing-pool); a legacy projection
     carries all 14 fields with a consistent expected_output_count and a PASS
     exclusion report; an existing-pool projection carries unique non-negative
     global indices whose count equals n_selected, the recommended sizing, and
     the parent-id list, with a PASS exclusion report.
  6. Strategy admissibility -- every selected backend is feasible in the
     inventory.

The validator never mutates or "repairs" the plan; it returns a list of issues
(empty iff valid), so the bounded semantic-correction retry loop can hand a
fresh, corrected proposal back through the same gate.
"""
from __future__ import annotations

import json

from framework_v2.acquisition.contracts import (
    AcquisitionPhase,
    AcquisitionPlanV2,
    AcquisitionStrategy,
    CandidateGenerationResult,
    CandidateSelectionResult,
    CampaignObjective,
    CanonicalLabelingRequest,
    CoverageGapAnalysis,
    RegionResolution,
    SourceAndCapabilityInventory,
    TargetRegimeModel,
)
from framework_v2.acquisition.plan_assembly import (
    EXISTING_POOL_REQUIRED_FIELDS,
    LEGACY_REQUIRED_FIELDS,
)

# Tokens that betray model-informed evidence leaking into an INITIAL plan.
_FORBIDDEN_INITIAL_TOKENS = (
    "uncertainty",
    "expected_information_gain",
    "expected information gain",
    "eig",
    "model_disagreement",
    "committee_disagreement",
    "student_error",
    "student_uncertainty",
    "acquisition_score_from_model",
)


def _scan_forbidden(blob: object) -> list[str]:
    """Return forbidden INITIAL-phase tokens found anywhere in a JSON-able
    structure (case-insensitive)."""
    text = json.dumps(blob, sort_keys=True, default=str).lower()
    return [tok for tok in _FORBIDDEN_INITIAL_TOKENS if tok in text]


def validate_acquisition_plan_v2(
    plan: AcquisitionPlanV2,
    *,
    objective: CampaignObjective,
    inventory: SourceAndCapabilityInventory,
    target_regime_model: TargetRegimeModel,
    region_resolution: RegionResolution,
    coverage: CoverageGapAnalysis,
    strategy: AcquisitionStrategy,
    generation_result: CandidateGenerationResult,
    selection_result: CandidateSelectionResult,
    labeling_request: CanonicalLabelingRequest,
) -> list[str]:
    """Deterministic validation. Returns issues; empty iff the plan binds."""
    issues: list[str] = []

    # 1. Evidence-chain integrity.
    chain = {
        "objective_sha256": objective.content_sha256(),
        "inventory_sha256": inventory.content_sha256(),
        "target_regime_model_sha256": target_regime_model.content_sha256(),
        "region_resolution_sha256": region_resolution.content_sha256(),
        "coverage_gap_sha256": coverage.content_sha256(),
        "strategy_sha256": strategy.content_sha256(),
        "generation_result_sha256": generation_result.content_sha256(),
        "selection_result_sha256": selection_result.content_sha256(),
        "labeling_request_sha256": labeling_request.content_sha256(),
    }
    for field, expected in chain.items():
        if getattr(plan, field) != expected:
            issues.append(f"evidence-chain mismatch: {field}")

    # cross-artifact binding integrity
    if strategy.coverage_gap_sha256 != coverage.content_sha256():
        issues.append("strategy not bound to the provided coverage analysis")
    if strategy.inventory_sha256 != inventory.content_sha256():
        issues.append("strategy not bound to the provided inventory")
    if generation_result.strategy_sha256 != strategy.content_sha256():
        issues.append("generation_result not bound to the provided strategy")
    if selection_result.generation_result_sha256 != generation_result.content_sha256():
        issues.append("selection_result not bound to the provided generation_result")
    if labeling_request.selection_result_sha256 != selection_result.content_sha256():
        issues.append("labeling_request not bound to the provided selection_result")

    # 2. INITIAL-phase evidence contract.
    if plan.phase == AcquisitionPhase.INITIAL:
        if objective.phase != AcquisitionPhase.INITIAL:
            issues.append("plan phase INITIAL but objective phase differs")
        if coverage.phase != AcquisitionPhase.INITIAL:
            issues.append("coverage analysis phase must be INITIAL for an INITIAL plan")
        for label, blob in (
            ("coverage", coverage.model_dump(mode="json")),
            ("strategy.rationale", strategy.rationale),
            ("selection.diversity_evidence", selection_result.diversity_evidence),
            ("coverage.available_source_coverage", coverage.available_source_coverage),
        ):
            found = _scan_forbidden(blob)
            if found:
                issues.append(
                    f"INITIAL plan carries forbidden model-informed evidence in "
                    f"{label}: {found}"
                )

    # 3. Provenance separation.
    if not all(p.exploration_only for p in generation_result.provenance):
        issues.append("some generation provenance not marked exploration_only")
    gen_ids = set(generation_result.candidate_ids)
    if not set(selection_result.selected_candidate_ids).issubset(gen_ids):
        issues.append("selected candidates are not a subset of generated candidates")
    if not labeling_request.relabel_from_scratch:
        issues.append("labeling request must relabel canonically from scratch")
    if set(labeling_request.candidate_ids) != set(
        selection_result.selected_candidate_ids
    ):
        issues.append("labeling request candidates != selected candidates")

    # 4. Fail-closed disjointness.
    rep = selection_result.disjointness_report
    if rep.status != "PASS":
        issues.append(f"protected disjointness not PASS: {rep.status}")
    if rep.n_overlaps != 0:
        issues.append("protected disjointness reports overlaps")
    if rep.dft_labels_used_as_selection_scores:
        issues.append("DFT labels were used as selection scores (forbidden)")

    # 5. Execution-projection consistency -- exactly one of the three.
    has_legacy = plan.legacy_projection is not None
    has_dyn = plan.dynamics_protocol_sha256 is not None
    has_pool = plan.existing_pool_projection is not None
    if sum((has_legacy, has_dyn, has_pool)) != 1:
        issues.append("exactly one execution projection must be present")
    if has_legacy:
        proj = plan.legacy_projection
        for f in LEGACY_REQUIRED_FIELDS:
            if f not in proj:
                issues.append(f"legacy projection missing field: {f}")
        if "n_parents" in proj and "n_per_structure" in proj:
            exp = int(proj["n_parents"]) * int(proj["n_per_structure"])
            if int(proj.get("expected_output_count", -1)) != exp:
                issues.append("legacy expected_output_count != n_parents*n_per_structure")
        pre = proj.get("protected_reference_exclusion_report", {})
        if pre.get("status") != "PASS":
            issues.append("legacy exclusion report status not PASS")
        if pre.get("dft_labels_used_as_selection_scores", False):
            issues.append("legacy exclusion report used DFT labels as scores")
        if not proj.get("duplicate_handling"):
            issues.append("legacy projection missing duplicate_handling")
    if has_pool:
        proj = plan.existing_pool_projection
        for f in EXISTING_POOL_REQUIRED_FIELDS:
            if f not in proj:
                issues.append(f"existing-pool projection missing field: {f}")
        idxs = proj.get("selected_source_global_indices", [])
        parents = proj.get("selected_parent_structure_ids", [])
        if not isinstance(idxs, list) or len(idxs) == 0:
            issues.append("existing-pool projection selected_source_global_indices empty")
        else:
            if any((not isinstance(i, int)) or i < 0 for i in idxs):
                issues.append("existing-pool indices must be non-negative ints")
            if len(set(idxs)) != len(idxs):
                issues.append("existing-pool indices contain duplicates")
        if len(parents) != len(idxs):
            issues.append(
                "existing-pool selected_parent_structure_ids and indices differ in length")
        if int(proj.get("n_selected", -1)) != len(idxs):
            issues.append("existing-pool n_selected != len(selected_source_global_indices)")
        if int(proj.get("expected_output_count", -1)) != len(idxs):
            issues.append(
                "existing-pool expected_output_count != len(selected_source_global_indices)")
        sizing = proj.get("labeling_population_sizing")
        if not isinstance(sizing, dict) or not sizing:
            issues.append("existing-pool projection missing labeling_population_sizing evidence")
        else:
            rec = sizing.get("recommended_population_size")
            if rec is not None and int(rec) != len(idxs):
                issues.append(
                    "existing-pool sizing recommended_population_size != number of selected frames")
        pre = proj.get("protected_reference_exclusion_report", {})
        if pre.get("status") != "PASS":
            issues.append("existing-pool exclusion report status not PASS")
        if pre.get("dft_labels_used_as_selection_scores", False):
            issues.append("existing-pool exclusion report used DFT labels as scores")
        if not proj.get("duplicate_handling"):
            issues.append("existing-pool projection missing duplicate_handling")

    # 6. Strategy admissibility.
    feasible_ids = {b.backend_id for b in inventory.feasible_backends()}
    for bid in strategy.selected_backend_ids:
        if bid not in feasible_ids:
            issues.append(f"strategy selected an infeasible backend: {bid}")

    return issues
