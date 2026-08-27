"""V2-H08: explicit / discovered / hybrid region providers share one pipeline.

A region manifest is provider-neutral downstream: whatever produced the regions
(explicit metadata, structural discovery, or a hybrid of the two), it flows
through the same sampler, the same evaluation-population binding, the same
metric aggregation, and the same error ledger.  We assert equal API and
invariants -- not equal selected IDs, since the providers legitimately group
frames differently.
"""
from framework_v2.error_tracking import (
    ErrorLedger,
    RawEfficiencyRecord,
    build_error_ledger_iteration,
)
from framework_v2.region_discovery import (
    DiscoveryMethod,
    RegionDiscoveryConfig,
    discover_structural_regions,
)
from framework_v2.region_evaluation import (
    EvaluationPopulationRegionBinding,
    FrameEvaluationRecord,
    RegionEvaluationRecord,
    aggregate_region_metrics,
    bind_evaluation_population_to_regions,
)
from framework_v2.structural_regions import (
    StructuralRegionManifest,
    StructuralRegionProviderType,
    explicit_regions_from_membership,
    hybrid_regions_from_parent_and_subregions,
)
from framework_v2.structural_representation import (
    CompositionRepresentationAdapter,
    StructureRecord,
)
from framework_v2.v2_sampling import (
    CriterionBindingStatus,
    CriterionComparator,
    CriterionRole,
    RegionClosureState,
    RegionStoppingPolicy,
    SamplerKind,
    SamplerRequest,
    SignalCriterion,
    sample_candidates,
)


def _structures():
    return [
        StructureRecord(structure_id="a", species_counts={"Si": 1, "O": 2}),
        StructureRecord(structure_id="b", species_counts={"Si": 1, "O": 1}),
        StructureRecord(structure_id="c", species_counts={"Si": 3}),
        StructureRecord(structure_id="d", species_counts={"O": 2}),
    ]


def _representation():
    return CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        _structures(), representation_id="rep"
    )


def _closure_policy():
    return RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            SignalCriterion(
                signal="force.component_rmse_eV_per_angstrom",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.BOUND,
                comparator=CriterionComparator.LE,
                value=0.3,
                units="eV/Angstrom",
                provenance=["synthetic"],
            )
        ],
    )


def _run_pipeline(manifest: StructuralRegionManifest, rep):
    req = SamplerRequest(
        sampler=SamplerKind.FPS,
        candidate_ids=["a", "b", "c", "d"],
        n_select=2,
        protected_candidate_ids=["d"],
    )
    sres = sample_candidates(req, representation=rep)

    eval_frames = ["a", "b"]
    binding = bind_evaluation_population_to_regions(
        region_manifest=manifest,
        evaluation_frame_ids=eval_frames,
        evaluation_population_sha256="protected_eval",
        binding_id="binding",
    )
    revs = aggregate_region_metrics(
        binding,
        [
            FrameEvaluationRecord(
                frame_id=f,
                n_atoms=2,
                reference_channel="student_vs_teacher",
                force_component_errors=[0.1],
            )
            for f in eval_frames
        ],
    )
    ledger = build_error_ledger_iteration(
        ledger=ErrorLedger(ledger_id="ledger", campaign_id="mock"),
        iteration=0,
        evaluation_binding=binding,
        region_evaluations=revs,
        closure_policy=_closure_policy(),
        target_validation_sha256=None,
        training_population_sha256="train0",
        efficiency=RawEfficiencyRecord(
            selected_structures=2,
            measurement_provenance={"selected_structures": ["mock"]},
        ),
    )
    return sres, binding, revs, ledger


def _assert_shared_invariants(sres, binding, revs, ledger):
    assert len(sres.selected_ids) == 2
    assert "d" not in sres.selected_ids
    assert isinstance(binding, EvaluationPopulationRegionBinding)
    assert all(isinstance(r, RegionEvaluationRecord) for r in revs)
    assert revs
    # bound force criterion passes -> every region closes, none deficient
    assert ledger.deficient_regions(0) == []
    assert all(
        r.state == RegionClosureState.CLOSED for r in ledger.records_for_iteration(0)
    )


def test_explicit_provider_flows_through_shared_pipeline():
    rep = _representation()
    explicit = explicit_regions_from_membership(
        manifest_id="explicit",
        frame_to_region={"a": "A", "b": "A", "c": "B", "d": "B"},
        source_sha256="source",
        membership_manifest_sha256="membership",
    )
    assert explicit.provider_type == StructuralRegionProviderType.EXPLICIT_METADATA
    _assert_shared_invariants(*_run_pipeline(explicit, rep))


def test_discovered_provider_flows_through_shared_pipeline():
    rep = _representation()
    discovered = discover_structural_regions(
        rep,
        RegionDiscoveryConfig(method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2),
        manifest_id="discovered",
        source_sha256="source",
    )
    assert discovered.provider_type == StructuralRegionProviderType.STRUCTURAL_DISCOVERY
    _assert_shared_invariants(*_run_pipeline(discovered, rep))


def test_hybrid_provider_has_parent_linkage_and_shared_pipeline():
    rep = _representation()
    explicit = explicit_regions_from_membership(
        manifest_id="explicit",
        frame_to_region={"a": "A", "b": "A", "c": "B", "d": "B"},
        source_sha256="source",
        membership_manifest_sha256="membership",
    )
    discovered = discover_structural_regions(
        rep,
        RegionDiscoveryConfig(method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2),
        manifest_id="discovered",
        source_sha256="source",
    )
    hybrid = hybrid_regions_from_parent_and_subregions(
        manifest_id="hybrid",
        parent_manifest=explicit,
        subregion_manifest=discovered,
        source_sha256="source",
    )
    assert hybrid.provider_type == StructuralRegionProviderType.HYBRID
    hybrid_children = [
        r
        for r in hybrid.regions
        if r.provider_type == StructuralRegionProviderType.HYBRID
    ]
    assert hybrid_children
    assert all(r.parent_region_id is not None for r in hybrid_children)
    _assert_shared_invariants(*_run_pipeline(hybrid, rep))
