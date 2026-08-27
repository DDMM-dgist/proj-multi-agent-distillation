import pytest

from framework_v2.contracts import DomainRepresentation, DomainRegime, ScopeCategory
from framework_v2.error_tracking import ErrorLedger, RawEfficiencyRecord, RegionErrorRecord
from framework_v2.experiment_controls import (
    DFTReplayPolicy,
    ReplayEligibilityRole,
    SupercellStrategy,
    SupercellUse,
)
from framework_v2.region_discovery import (
    DiscoveryMethod,
    RegionDiscoveryConfig,
    discover_structural_regions,
)
from framework_v2.region_recovery import plan_region_recovery
from framework_v2.structural_regions import (
    StructuralRegion,
    StructuralRegionProviderType,
    explicit_regions_from_membership,
    hybrid_regions_from_parent_and_subregions,
    regions_from_domain_representation,
)
from framework_v2.structural_representation import (
    CompositionRepresentationAdapter,
    StructureRecord,
)
from framework_v2.v2_judge_policy import (
    DecisionMode,
    EvidenceSufficiency,
    V2JudgePolicy,
    choose_decision_mode,
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


def test_representation_ordering_and_hash_are_stable():
    rep1 = CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        _structures(), representation_id="rep"
    )
    rep2 = CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        _structures(), representation_id="rep"
    )

    assert rep1.structure_ids == ["a", "b", "c", "d"]
    assert rep1.content_sha256() == rep2.content_sha256()
    assert rep1.software["numpy"]


def test_explicit_discovered_and_hybrid_regions_share_api():
    explicit = explicit_regions_from_membership(
        manifest_id="explicit",
        frame_to_region={"a": "semantic_a", "b": "semantic_a", "c": "semantic_b"},
        source_sha256="source",
        membership_manifest_sha256="membership",
    )
    rep = CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        _structures(), representation_id="rep"
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

    assert explicit.region_for_frame("a").region_id == "semantic_a"
    assert discovered.region_for_frame("a") is not None
    assert hybrid.provider_type == StructuralRegionProviderType.HYBRID


def test_discovered_cluster_is_not_physical_phase():
    with pytest.raises(ValueError, match="physical phase"):
        StructuralRegion(
            region_id="structural_region_003",
            provider_type=StructuralRegionProviderType.STRUCTURAL_DISCOVERY,
            membership_provenance=["evidence"],
            population_size=3,
            membership_manifest_sha256="manifest",
            representation_sha256="rep",
            semantic_annotation="liquid",
            cluster_is_physical_phase=True,
        )


def test_old_domain_representation_adapts_to_structural_regions():
    domain = DomainRepresentation(
        representation_id="old",
        kind="categorical",
        descriptor="config_type",
        linked_scope_contract_sha256="scope",
        regimes=[
            DomainRegime(
                regime_id="oxygen_vacancy_SiO2",
                label="oxygen vacancy",
                membership_rule="metadata",
                membership_evidence_refs=["pool_manifest:pool", "frame:a"],
                within_scope_categories=[ScopeCategory.PRIMARY_DEPLOYMENT],
            )
        ],
    )

    manifest = regions_from_domain_representation(
        domain, manifest_id="compat", source_sha256="source"
    )
    assert manifest.region_for_frame("a").region_id == "oxygen_vacancy_SiO2"


def test_samplers_share_interface_and_exclude_protected():
    rep = CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        _structures(), representation_id="rep"
    )
    req = SamplerRequest(
        sampler=SamplerKind.FPS,
        candidate_ids=["a", "b", "c", "d"],
        n_select=2,
        protected_candidate_ids=["d"],
    )
    result = sample_candidates(req, representation=rep)
    assert len(result.selected_ids) == 2
    assert "d" not in result.selected_ids


def test_direct_like_stratified_path_and_recovery_samplers():
    req = SamplerRequest(
        sampler=SamplerKind.DIRECT_LIKE,
        candidate_ids=["a", "b", "c", "d"],
        n_select=3,
        region_by_candidate={"a": "r1", "b": "r1", "c": "r2", "d": "r2"},
    )
    result = sample_candidates(req)
    assert result.selected_ids == ["a", "b", "c"]

    rep = CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        _structures(), representation_id="rep"
    )
    uq_req = SamplerRequest(
        sampler=SamplerKind.UNCERTAINTY_DIVERSITY,
        candidate_ids=["a", "b", "c", "d"],
        n_select=2,
        uncertainty_by_candidate={"a": 0.1, "b": 2.0, "c": 1.5, "d": 1.0},
    )
    assert len(sample_candidates(uq_req, representation=rep).selected_ids) == 2


def test_stopping_policy_is_independent_of_sampler():
    policy = RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            SignalCriterion(
                signal="force_rmse",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.BOUND,
                comparator=CriterionComparator.LE,
                value=0.3,
                units="eV/A",
                provenance=["human_contract"],
            )
        ],
    )
    assert policy.state_for({"force_rmse": 0.2}) == RegionClosureState.CLOSED
    assert policy.state_for({"force_rmse": 0.4}) == RegionClosureState.RECOVER

    unbound = RegionStoppingPolicy(
        policy_id="unbound",
        criteria=[
            SignalCriterion(
                signal="force_rmse",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.UNBOUND,
                unbound_reason="human has not set target",
            )
        ],
    )
    assert (
        unbound.state_for({"force_rmse": 0.2})
        == RegionClosureState.HUMAN_SCIENTIFIC_INPUT_REQUIRED
    )


def test_error_ledger_persist_reload_and_recovery_before_after(tmp_path):
    record = RegionErrorRecord(
        campaign_id="c",
        iteration=0,
        region_id="r1",
        region_membership_sha256="region",
        state=RegionClosureState.RECOVER,
        failure_reason="force error high",
        before_metric={"force": 0.5},
        after_metric={"force": 0.2},
        delta={"force": -0.3},
        efficiency=RawEfficiencyRecord(added_structures=2, wall_time_seconds=1.0),
    )
    ledger = ErrorLedger(ledger_id="l", campaign_id="c").append(record)
    path = tmp_path / "ledger.json"
    ledger.save(path)
    loaded = ErrorLedger.load(path)

    assert loaded.deficient_regions(0) == ["r1"]
    assert loaded.records[0].delta["force"] == -0.3


def test_region_recovery_uses_only_deficient_regions_and_no_protected():
    ledger = ErrorLedger(
        ledger_id="l",
        campaign_id="c",
        records=[
            RegionErrorRecord(
                campaign_id="c",
                iteration=0,
                region_id="r_closed",
                region_membership_sha256="r_closed",
                state=RegionClosureState.CLOSED,
            ),
            RegionErrorRecord(
                campaign_id="c",
                iteration=0,
                region_id="r_bad",
                region_membership_sha256="r_bad",
                state=RegionClosureState.RECOVER,
                failure_reason="uncertainty high",
            ),
        ],
    )
    plan = plan_region_recovery(
        ledger,
        iteration=0,
        eligible_training_candidate_ids=["a", "b"],
        protected_candidate_ids=["p"],
        sampler=SamplerKind.UNCERTAINTY,
        n_select=1,
        rationale="recover deficient region only",
    )
    req = plan.sampler_request({"a": "r_bad", "b": "r_closed"})
    assert req.deficient_region_ids == ["r_bad"]
    assert sample_candidates(req).selected_ids == ["a"]


def test_replay_train_only_and_supercell_lineage():
    with pytest.raises(ValueError, match="TRAIN-role"):
        DFTReplayPolicy(
            policy_id="replay",
            enabled=True,
            ratio=0.1,
            eligible_frame_ids=["protected"],
            frame_roles={"protected": ReplayEligibilityRole.PROTECTED},
            provenance_refs=["split"],
        )

    policy = DFTReplayPolicy(
        policy_id="replay",
        enabled=True,
        ratio=0.1,
        eligible_frame_ids=["train"],
        frame_roles={"train": ReplayEligibilityRole.TRAIN},
        provenance_refs=["split"],
    )
    assert policy.ratio == 0.1

    with pytest.raises(ValueError, match="Teacher labeling provenance"):
        SupercellStrategy(
            strategy_id="supercell",
            use=SupercellUse.TRAINING_STRATEGY,
            parent_ids=["a"],
            replication_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        )


def test_deterministic_first_judge_policy():
    policy = V2JudgePolicy(policy_id="judge")
    assert (
        choose_decision_mode(policy, evidence_sufficiency=EvidenceSufficiency.SUFFICIENT)
        == DecisionMode.DETERMINISTIC_GATE
    )
    assert (
        choose_decision_mode(
            policy,
            evidence_sufficiency=EvidenceSufficiency.AMBIGUOUS,
            reason="scientific_ambiguity",
        )
        == DecisionMode.JUDGE_ALLOWED
    )
    assert (
        choose_decision_mode(
            policy,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            deterministic_failure=True,
            reason="scientific_ambiguity",
        )
        == DecisionMode.DETERMINISTIC_GATE
    )
