"""Framework V2 -- the GENERIC fallback ``StructuralDescriptorProvider`` (FE-027).

This is the framework-provided, material-agnostic descriptor provider that removes the need for
a per-material plugin just to make autonomous acquisition *usable*. It builds its representation
from raw structural facts alone (see :mod:`framework_v2.acquisition.generic_representation`) and
is registered into the SEPARATE generic-fallback slot (see
``descriptor_plugins.register_generic_descriptor_provider``) so a specialized plugin always wins
when one applies.

Construction is deliberately INCREMENTAL across the FE-027 phases, matching the layered gap:

  * P1 (this file's initial surface): ``applies`` gating from the run's own frozen inputs +
    ``build_representation_result`` -- the generic raw-structure representation with a first-class,
    comparative adequacy assessment and fail-closed ``REPRESENTATION_INSUFFICIENT`` recovery.
  * P2 adds typed executable region membership + Agent-proposed target relevance.
  * P3 adds the two coverage axes + evidence-driven acquisition sizing.
  * P4 adds the generic protocol-envelope-derived generation bounds.

With P2--P4 in place, ``build_descriptor_space_evidence`` (P6) assembles the full
``DescriptorSpaceEvidence`` the framework materializer composes the generic FE-026 pipeline
around: the P1 discovered representation supplies the DISCOVERED-path region structure, the
pool's own farthest-point saturation curve (P3) supplies the CORE_TARGET coverage evidence, and
the P4 perturbation envelope supplies the physics-bounded generation-recipe decision space. No
relevance / coverage / protocol number is fabricated ahead of its evidence; every value is
derived deterministically from the run's own raw pool.
"""
from __future__ import annotations

from typing import Optional

from framework_v2.acquisition.descriptor_plugins import DescriptorSpaceEvidence
from framework_v2.acquisition.generic_representation import (
    GenericRepresentationResult,
    build_adequate_representation,
    load_pool,
    locate_pool_manifest,
)
from framework_v2.contracts import DeploymentScopeContract, ScopeCategory

GENERIC_MATERIAL_ID = "generic-raw-structure"


class GenericStructuralDescriptorProvider:
    """The material-agnostic fallback provider.

    ``material_id`` is a fixed generic identifier (it names no material). ``applies`` is
    deliberately broad -- it admits any v2 campaign that (a) declares a PRIMARY_DEPLOYMENT scope
    region and (b) exposes a schema-detectable raw-structure pool among its frozen inputs -- so
    the framework can plan acquisition from raw structures with no per-material authoring. Because
    it lives in the generic-fallback slot, this breadth never collides with a specialized plugin.
    """

    material_id = GENERIC_MATERIAL_ID

    def __init__(self, *, max_frames_per_category: Optional[int] = None) -> None:
        # A bounded-compute cap on planning-time descriptor computation over very large pools; it
        # is a deterministic head-slice, not a scientific selection. None reads the whole pool.
        self._max_frames_per_category = max_frames_per_category

    # -- gating ------------------------------------------------------------------------------
    def applies(self, *, controller, objective, scope_contract: DeploymentScopeContract) -> bool:
        if not scope_contract.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT):
            return False
        try:
            locate_pool_manifest(controller)
        except Exception:
            return False
        return True

    # -- P1 core: generic representation + first-class adequacy ------------------------------
    def build_representation_result(
        self, *, controller, objective, scope_contract: DeploymentScopeContract,
        region_classifier=None,
    ) -> GenericRepresentationResult:
        """Load the raw pool and build an ADEQUATE generic representation.

        Fails closed with a typed ``AcquisitionCapabilityGap`` (raised from within the generic
        representation builder) if no admissible representation discriminates the pool -- it never
        asks a human for a descriptor."""
        pool = load_pool(controller, max_frames_per_category=self._max_frames_per_category)
        run_id = controller.state["run_id"]
        return build_adequate_representation(
            pool, id_prefix=run_id, scope_contract=scope_contract,
            deployment_claim=objective.claim_scope, region_classifier=region_classifier)

    # -- P6: full DescriptorSpaceEvidence from P1 representation + P3 coverage + P4 envelope ---
    def build_descriptor_space_evidence(
        self, *, controller, objective, scope_contract: DeploymentScopeContract,
    ) -> DescriptorSpaceEvidence:
        """Assemble the full material-agnostic ``DescriptorSpaceEvidence`` bundle from raw structures.

        Every value is DERIVED, never fabricated: the discovered representation + its executable
        regions come from P1/P2; the CORE_TARGET saturation/novelty comes from the pool's OWN
        farthest-point marginal-novelty curve (P3) over the chosen descriptor axes; the admissible
        generation-recipe decision space (displacement / cell-strain bounds + presence-required
        seed) comes from the pool's OWN nearest-neighbor scale via the P4 perturbation envelope.
        The scope contract's PRIMARY_DEPLOYMENT regions are the CORE_TARGET regimes (the frozen
        objective convention), matching the scope-derived TargetRegimeModel the materializer builds.
        """
        from framework_v2.acquisition.contracts import RelevanceRole, SourceCategoryRecord
        from framework_v2.acquisition.coverage_gap import RegimeCoverageInput
        from framework_v2.acquisition.generic_coverage import (
            FrameworkSizingParams, _axis_scales, compute_saturation)
        from framework_v2.acquisition.generic_protocol import (
            EnvelopeParams, build_perturbation_envelope)
        from framework_v2.acquisition.strategy import StrategyEvidence

        result = self.build_representation_result(
            controller=controller, objective=objective, scope_contract=scope_contract)
        pool = result.pool
        representation = result.representation
        run_id = controller.state["run_id"]

        # CORE_TARGET coverage: the pool's own descriptor-space saturation over the chosen axes.
        axes = list(result.spec.continuous_variables)
        sizing_params = FrameworkSizingParams()
        vectors = [{k: f.features[k] for k in axes if k in f.features} for f in pool.frames]
        scales = _axis_scales(vectors, axes)
        saturation, novelty_headroom = compute_saturation(vectors, axes, scales, sizing_params)

        primary_regions = scope_contract.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT)
        regime_coverage_inputs = tuple(
            RegimeCoverageInput(
                regime_id=region.region_id,
                relevance_role=RelevanceRole.CORE_TARGET,
                current_count=pool.total_frames,
                saturation=saturation,
                novelty_headroom=novelty_headroom,
                target_count=None)
            for region in primary_regions)

        # Deterministic strategy signals DERIVED from the coverage evidence + available sources
        # (FE-028), never hardcoded:
        #   * Case A -- a SATURATED core means the eligible existing pool already closes the gap:
        #     ``pool_covers_gaps`` is True, no new configurations are required, and
        #     EXISTING_POOL_SELECTION becomes admissible (select a representative existing subset
        #     for canonical labeling instead of raising StrategyUndecidable).
        #   * Case B -- an UNSATURATED core in the generic raw-pool path is reachable by local
        #     perturbation of the existing parents (the pool spans a continuous descriptor space),
        #     so new configurations are still not *required* (-> LOCAL_PERTURBATION).
        #   * Case C -- the stronger claim that a gap is reachable ONLY by Teacher-driven dynamics
        #     requires richer, plugin-supplied evidence the generic path does not have, so it is
        #     never fabricated here; ``gaps_require_new_configurations`` is asserted True only when
        #     the core is unsaturated AND no existing parent can reach it (e.g. no seeds at all).
        core_saturated = saturation >= sizing_params.saturation_threshold
        seeds_exist = pool.total_frames > 0
        parents_reach_gaps = seeds_exist
        gaps_require_new_configurations = (not core_saturated) and (not parents_reach_gaps)
        strategy_evidence = StrategyEvidence(
            pool_covers_gaps=core_saturated,
            parents_reach_gaps=parents_reach_gaps,
            gaps_require_new_configurations=gaps_require_new_configurations,
            seed_structures_exist=seeds_exist)

        # Physics-bounded generation-recipe decision space from the pool's own nearest-neighbor scale.
        envelope = build_perturbation_envelope(
            pool, params=EnvelopeParams(),
            envelope_id=f"{run_id}-perturbation-envelope",
            evidence_ref=f"pool_manifest:{pool.manifest_sha256}")

        categories = sorted(pool.per_category_counts.items())
        source_records = tuple(
            SourceCategoryRecord(
                category=category, n_items=int(count), has_metadata=False,
                provenance_class="sanitized_pool")
            for category, count in categories)

        # ``resolve_regions`` consumes the DISCOVERED-path builder; return the already-built P1
        # representation so its content-SHA is stable across calls.
        def _discovered_representation():
            return representation

        return DescriptorSpaceEvidence(
            descriptor=result.spec.descriptor,
            source_records=source_records,
            discovered_representation_builder=_discovered_representation,
            regime_coverage_inputs=regime_coverage_inputs,
            strategy_evidence=strategy_evidence,
            admissible_parent_ids=tuple(f.item_id for f in pool.frames),
            required_param_keys=tuple(envelope.required_param_keys),
            param_bounds=dict(envelope.param_bounds),
            eligible_source_categories=tuple(category for category, _ in categories),
            selected_source_global_indices=tuple(range(len(categories))),
            duplicate_handling="reject",
            saturation_threshold=sizing_params.saturation_threshold,
            metadata_present=False)
