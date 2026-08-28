"""V2-H13: generic recursive structural-region discovery.

Covers the generic hierarchy hardening: deterministic child identity,
partition-preserving refinement, active-leaf semantics, refinement
assessment/authorization, downstream identity continuity (evaluation /
ErrorLedger / RECOVER), backward compatibility, and system-name independence.

No SiO2-specific logic: the same API is exercised on Si/O and on a synthetic
ternary A/B/C system.
"""
from __future__ import annotations

import pytest

from framework_v2.error_tracking import (
    ErrorLedger,
    ReferenceChannel,
    RegionErrorRecord,
)
from framework_v2.region_discovery import (
    DiscoveryMethod,
    RegionDiscoveryConfig,
    RegionRefinementAssessment,
    RegionRefinementEvidence,
    RegionRefinementRequest,
    RegionResolutionStatus,
    TargetRelevanceStatus,
    authorize_refinement,
    discover_structural_regions,
    discover_structural_subregions,
    refinement_is_authorized,
)
from framework_v2.region_evaluation import (
    FrameEvaluationRecord,
    aggregate_region_metrics,
    bind_evaluation_population_to_regions,
)
from framework_v2.region_recovery import (
    RecoveryAction,
    RegionRecoveryPlan,
    plan_region_recovery,
)
from framework_v2.structural_regions import (
    StructuralRegion,
    StructuralRegionManifest,
    StructuralRegionProviderType,
    child_region_id,
    mint_child_regions,
    refine_region_partition,
)
from framework_v2.structural_representation import (
    CompositionRepresentationAdapter,
    StructureRecord,
)
from framework_v2.v2_sampling import RegionClosureState, SamplerKind, sample_candidates


# --------------------------------------------------------------------------
# fixtures: representations whose rows are all distinct so FPS splits are
# non-degenerate (a parent with >= 2 members truly splits into non-empty
# children).
# --------------------------------------------------------------------------
def _distinct_records(species, n):
    recs = []
    for i in range(n):
        counts = {species[0]: i + 1, species[1]: n - i, species[2]: (i * 3) % (n + 1) + 1}
        recs.append(StructureRecord(structure_id=f"f{i}", species_counts=counts))
    return recs


def _rep(species=("A", "B", "C"), n=9):
    species = list(species)
    return CompositionRepresentationAdapter(species=species).compute(
        _distinct_records(species, n), representation_id="rep"
    )


def _flat(rep, k=3):
    return discover_structural_regions(
        rep,
        RegionDiscoveryConfig(method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=k),
        manifest_id="L1",
        source_sha256="src",
    )


def _largest_leaf(manifest):
    counts = {}
    for rid in manifest.frame_to_region.values():
        counts[rid] = counts.get(rid, 0) + 1
    return max(sorted(counts), key=lambda r: counts[r])


def _refine(manifest, rep, parent_id, k=2, mid="L2", auth=None):
    child = discover_structural_subregions(
        parent_manifest=manifest,
        parent_region_id=parent_id,
        representation=rep,
        config=RegionDiscoveryConfig(method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=k),
        manifest_id=f"{mid}child",
        source_sha256="src",
        authorization=auth,
    )
    refined = refine_region_partition(
        manifest_id=mid,
        current_manifest=manifest,
        parent_region_id=parent_id,
        child_manifest=child,
    )
    return child, refined


# --------------------------------------------------------------------------
# A. deterministic child ID minting
# --------------------------------------------------------------------------
def test_A_deterministic_child_ids():
    assert child_region_id("C0", 0) == "C0.0"
    assert child_region_id("C0", 1) == "C0.1"
    assert child_region_id("C0.1", 2) == "C0.1.2"
    with pytest.raises(ValueError):
        child_region_id("", 0)
    with pytest.raises(ValueError):
        child_region_id("C0", -1)

    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    c1, _ = _refine(flat, rep, parent_id)
    c2, _ = _refine(flat, rep, parent_id)
    assert sorted(r.region_id for r in c1.regions) == sorted(r.region_id for r in c2.regions)
    assert c1.frame_to_region == c2.frame_to_region


# --------------------------------------------------------------------------
# B. refine one parent -> the other two regions preserved byte-for-byte
# --------------------------------------------------------------------------
def test_B_unaffected_regions_preserved():
    rep = _rep()
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)
    others = {r.region_id: r for r in flat.regions if r.region_id != parent_id}
    other_frames = {f: r for f, r in flat.frame_to_region.items() if r != parent_id}

    _, refined = _refine(flat, rep, parent_id)

    for rid, region in others.items():
        assert refined.region(rid) == region  # StructuralRegion unchanged
    for f, r in other_frames.items():
        assert refined.frame_to_region[f] == r  # untouched assignment


# --------------------------------------------------------------------------
# C. child union exactly replaces parent membership
# --------------------------------------------------------------------------
def test_C_children_repartition_parent():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    parent_members = {f for f, r in flat.frame_to_region.items() if r == parent_id}

    child, refined = _refine(flat, rep, parent_id)
    child_ids = {r.region_id for r in child.regions}
    remapped = {f: refined.frame_to_region[f] for f in parent_members}
    assert set(remapped) == parent_members
    assert set(remapped.values()) <= child_ids
    # every original parent frame is now owned by exactly one child leaf
    assert set(child.frame_to_region) == parent_members


# --------------------------------------------------------------------------
# D. incomplete child coverage fails closed
# --------------------------------------------------------------------------
def test_D_incomplete_coverage_fails_closed():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    child, _ = _refine(flat, rep, parent_id)
    # drop one frame from the child partition
    dropped_frame = sorted(child.frame_to_region)[0]
    trimmed_map = {f: r for f, r in child.frame_to_region.items() if f != dropped_frame}
    trimmed = StructuralRegionManifest(
        manifest_id=child.manifest_id,
        provider_type=child.provider_type,
        regions=child.regions,
        frame_to_region=trimmed_map,
        source_sha256=child.source_sha256,
    )
    with pytest.raises(ValueError, match="incomplete"):
        refine_region_partition(
            manifest_id="bad",
            current_manifest=flat,
            parent_region_id=parent_id,
            child_manifest=trimmed,
        )


# --------------------------------------------------------------------------
# E. a child that claims a sibling leaf's frame (overlap) fails closed
# --------------------------------------------------------------------------
def test_E_overlap_with_sibling_fails_closed():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    sibling_frame = next(f for f, r in flat.frame_to_region.items() if r != parent_id)
    child, _ = _refine(flat, rep, parent_id)
    first_child = sorted(r.region_id for r in child.regions)[0]
    overlapped = dict(child.frame_to_region)
    overlapped[sibling_frame] = first_child  # claim a frame owned by a sibling leaf
    bad = StructuralRegionManifest(
        manifest_id=child.manifest_id,
        provider_type=child.provider_type,
        regions=child.regions,
        frame_to_region=overlapped,
        source_sha256=child.source_sha256,
    )
    with pytest.raises(ValueError, match="outside parent membership"):
        refine_region_partition(
            manifest_id="bad",
            current_manifest=flat,
            parent_region_id=parent_id,
            child_manifest=bad,
        )


# --------------------------------------------------------------------------
# F. child frame entirely outside the population fails closed
# --------------------------------------------------------------------------
def test_F_child_outside_population_fails_closed():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    child, _ = _refine(flat, rep, parent_id)
    first_child = sorted(r.region_id for r in child.regions)[0]
    extended = dict(child.frame_to_region)
    extended["ghost_frame"] = first_child
    bad = StructuralRegionManifest(
        manifest_id=child.manifest_id,
        provider_type=child.provider_type,
        regions=child.regions,
        frame_to_region=extended,
        source_sha256=child.source_sha256,
    )
    with pytest.raises(ValueError, match="outside parent membership"):
        refine_region_partition(
            manifest_id="bad",
            current_manifest=flat,
            parent_region_id=parent_id,
            child_manifest=bad,
        )


# --------------------------------------------------------------------------
# G. duplicate child IDs fail closed (manifest-level and collision with existing)
# --------------------------------------------------------------------------
def test_G_duplicate_child_ids_fail_closed():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    parent = flat.region(parent_id)

    # (g1) duplicate ids within a child manifest -> manifest _consistent rejects
    dup = StructuralRegion(
        region_id=child_region_id(parent_id, 0),
        provider_type=StructuralRegionProviderType.HYBRID,
        membership_provenance=["p"],
        population_size=1,
        membership_manifest_sha256="m",
        parent_region_id=parent_id,
        discovery_depth=parent.discovery_depth + 1,
    )
    with pytest.raises(ValueError, match="duplicate region_id"):
        StructuralRegionManifest(
            manifest_id="dup",
            provider_type=StructuralRegionProviderType.HYBRID,
            regions=[dup, dup],
            frame_to_region={sorted(flat.frame_to_region)[0]: dup.region_id},
            source_sha256="src",
        )

    # (g2) child id colliding with an existing region -> refine rejects
    members = [f for f, r in flat.frame_to_region.items() if r == parent_id]
    existing_sibling = next(r.region_id for r in flat.regions if r.region_id != parent_id)
    colliding = StructuralRegion(
        region_id=existing_sibling,  # collides with a sibling
        provider_type=StructuralRegionProviderType.HYBRID,
        membership_provenance=["p"],
        population_size=len(members),
        membership_manifest_sha256="m",
        parent_region_id=parent_id,
        discovery_depth=parent.discovery_depth + 1,
    )
    bad_child = StructuralRegionManifest(
        manifest_id="c",
        provider_type=StructuralRegionProviderType.HYBRID,
        regions=[colliding],
        frame_to_region={f: existing_sibling for f in members},
        source_sha256="src",
    )
    with pytest.raises(ValueError, match="collides"):
        refine_region_partition(
            manifest_id="bad",
            current_manifest=flat,
            parent_region_id=parent_id,
            child_manifest=bad_child,
        )


# --------------------------------------------------------------------------
# H. nested refinement to depth >= 2
# --------------------------------------------------------------------------
def test_H_nested_refinement_depth_two():
    rep = _rep(n=12)
    flat = _flat(rep, k=2)
    parent_id = _largest_leaf(flat)
    _, refined = _refine(flat, rep, parent_id, mid="L2")

    # refine a depth-1 child that still has >= 2 members
    def leaf_members(m, rid):
        return [f for f, r in m.frame_to_region.items() if r == rid]

    depth1_leaves = [
        r.region_id
        for r in refined.active_leaf_regions()
        if r.discovery_depth == 1 and len(leaf_members(refined, r.region_id)) >= 2
    ]
    assert depth1_leaves, "expected a refinable depth-1 leaf"
    target = depth1_leaves[0]
    _, refined2 = _refine(refined, rep, target, mid="L3")

    depth2 = [r for r in refined2.regions if r.discovery_depth == 2]
    assert depth2
    for r in depth2:
        assert r.parent_region_id == target
        assert r.region_id.startswith(target + ".")
    # active-leaf partition still complete
    assert set(refined2.active_leaf_frame_to_region()) == set(flat.frame_to_region)
    assert target not in set(refined2.active_leaf_region_ids())


# --------------------------------------------------------------------------
# I. independent refinement of two different leaves
# --------------------------------------------------------------------------
def test_I_independent_leaf_refinement():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    counts = {}
    for rid in flat.frame_to_region.values():
        counts[rid] = counts.get(rid, 0) + 1
    refinable = sorted([rid for rid, c in counts.items() if c >= 2])[:2]
    assert len(refinable) == 2

    _, r1 = _refine(flat, rep, refinable[0], mid="A")
    _, r2 = _refine(r1, rep, refinable[1], mid="B")

    leaves = set(r2.active_leaf_region_ids())
    assert refinable[0] not in leaves and refinable[1] not in leaves
    assert any(x.startswith(refinable[0] + ".") for x in leaves)
    assert any(x.startswith(refinable[1] + ".") for x in leaves)
    assert set(r2.active_leaf_frame_to_region()) == set(flat.frame_to_region)


# --------------------------------------------------------------------------
# J. flat legacy manifest remains valid (parent None, depth 0, all leaves)
# --------------------------------------------------------------------------
def test_J_flat_legacy_manifest_valid():
    rep = _rep()
    flat = _flat(rep)
    for r in flat.regions:
        assert r.parent_region_id is None
        assert r.discovery_depth == 0
    assert sorted(flat.active_leaf_region_ids()) == sorted(r.region_id for r in flat.regions)
    assert flat.active_leaf_frame_to_region() == flat.frame_to_region

    # a hand-built legacy region without depth/parent still validates
    legacy = StructuralRegion(
        region_id="structural_region_001",
        provider_type=StructuralRegionProviderType.STRUCTURAL_DISCOVERY,
        membership_provenance=["e"],
        population_size=1,
        membership_manifest_sha256="m",
        representation_sha256="r",
    )
    assert legacy.discovery_depth == 0 and legacy.parent_region_id is None


# --------------------------------------------------------------------------
# K. active-leaf frame mapping stays complete across refinements
# --------------------------------------------------------------------------
def test_K_active_leaf_partition_complete():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    m = flat
    for i, parent in enumerate(sorted(set(flat.frame_to_region.values()))):
        members = [f for f, r in m.frame_to_region.items() if r == parent]
        if len(members) >= 2:
            _, m = _refine(m, rep, parent, mid=f"R{i}")
    alp = m.active_leaf_frame_to_region()
    assert set(alp) == set(flat.frame_to_region)
    assert len(alp) == len(flat.frame_to_region)


# --------------------------------------------------------------------------
# L. RegionEvaluation binds + aggregates on hierarchical leaf IDs
# --------------------------------------------------------------------------
def _eval_on(manifest):
    leaf_map = manifest.active_leaf_frame_to_region()
    binding = bind_evaluation_population_to_regions(
        region_manifest=manifest,
        evaluation_frame_ids=sorted(leaf_map),
        evaluation_population_sha256="evalpop",
        binding_id="bind",
    )
    frame_records = [
        FrameEvaluationRecord(
            frame_id=f,
            n_atoms=10,
            reference_channel="student_vs_teacher",
            energy_error_eV=0.01,
            force_component_errors=[0.02, 0.03],
        )
        for f in sorted(leaf_map)
    ]
    return binding, aggregate_region_metrics(binding, frame_records)


def test_L_region_evaluation_on_hierarchical_ids():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)
    _, refined = _refine(flat, rep, parent_id)

    binding, records = _eval_on(refined)
    ids = {rec.region_id for rec in records}
    assert ids == set(refined.active_leaf_region_ids())
    assert any("." in rid for rid in ids)  # a hierarchical leaf was evaluated
    assert parent_id not in ids  # refined parent is not evaluated


# --------------------------------------------------------------------------
# M. ErrorLedger deficient leaf survives with identity unchanged
# --------------------------------------------------------------------------
def test_M_error_ledger_hierarchical_leaf():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)
    _, refined = _refine(flat, rep, parent_id)
    leaf = next(r for r in refined.active_leaf_region_ids() if "." in r)

    rec = RegionErrorRecord(
        campaign_id="camp",
        iteration=0,
        region_id=leaf,
        region_membership_sha256=refined.content_sha256(),
        state=RegionClosureState.RECOVER,
        reference_channel=ReferenceChannel.STUDENT_VS_TEACHER,
        failure_reason="under target",
    )
    ledger = ErrorLedger(ledger_id="L", campaign_id="camp").append(rec)
    assert ledger.deficient_regions(0) == [leaf]


# --------------------------------------------------------------------------
# N. RECOVER routes the same hierarchical leaf ID
# --------------------------------------------------------------------------
def test_N_recover_routes_hierarchical_leaf():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)
    _, refined = _refine(flat, rep, parent_id)
    leaf_map = refined.active_leaf_frame_to_region()
    leaf = next(r for r in refined.active_leaf_region_ids() if "." in r)

    rec = RegionErrorRecord(
        campaign_id="camp",
        iteration=0,
        region_id=leaf,
        region_membership_sha256=refined.content_sha256(),
        state=RegionClosureState.RECOVER,
        failure_reason="under target",
    )
    ledger = ErrorLedger(ledger_id="L", campaign_id="camp").append(rec)
    eligible = sorted(leaf_map)
    plan = plan_region_recovery(
        ledger,
        iteration=0,
        eligible_training_candidate_ids=eligible,
        protected_candidate_ids=[],
        sampler=SamplerKind.DIRECT_LIKE,
        n_select=1,
        rationale="recover leaf",
    )
    assert plan.deficient_region_ids == [leaf]
    req = plan.sampler_request(region_by_candidate=leaf_map)
    result = sample_candidates(req)
    assert result.selected_ids  # non-empty
    # every selected candidate belongs to the deficient hierarchical leaf
    assert all(leaf_map[c] == leaf for c in result.selected_ids)


# --------------------------------------------------------------------------
# O. refined parent does not double-trigger recovery (leaves authoritative)
# --------------------------------------------------------------------------
def test_O_refined_parent_not_double_counted():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)
    _, refined = _refine(flat, rep, parent_id)

    leaves = set(refined.active_leaf_region_ids())
    assert parent_id not in leaves  # parent retired from active tracking
    assert parent_id in {r.region_id for r in refined.regions}  # still in ancestry

    _, records = _eval_on(refined)
    # authoritative required set = active leaves; parent never yields a record
    assert parent_id not in {rec.region_id for rec in records}


# --------------------------------------------------------------------------
# P. representation / config SHA changes propagate to refined identity
# --------------------------------------------------------------------------
def test_P_sha_propagation():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)

    _, refined_a = _refine(flat, rep, parent_id, k=2, mid="A")
    _, refined_b = _refine(flat, rep, parent_id, k=3, mid="B")

    # different discovery granularity -> different tree + partition identity
    assert refined_a.region_tree_sha256() != refined_b.region_tree_sha256()
    assert (
        refined_a.active_leaf_partition_sha256()
        != refined_b.active_leaf_partition_sha256()
    )
    assert refined_a.content_sha256() != refined_b.content_sha256()

    # refined manifest identity differs from the pre-refinement flat manifest
    assert refined_a.region_tree_sha256() != flat.region_tree_sha256()
    assert refined_a.active_leaf_partition_sha256() != flat.active_leaf_partition_sha256()

    # a different representation propagates too
    rep2 = _rep(species=("A", "B", "C"), n=13)
    flat2 = _flat(rep2, k=3)
    assert flat2.active_leaf_partition_sha256() != flat.active_leaf_partition_sha256()


# --------------------------------------------------------------------------
# Q. system-name independence: identical API on a non-Si/O ternary system
# --------------------------------------------------------------------------
def test_Q_system_name_independence():
    # Synthetic ternary "X/Y/Z"; region ids come from the generic backend, no
    # material-specific naming anywhere.
    rep = _rep(species=("X", "Y", "Z"), n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)
    child, refined = _refine(flat, rep, parent_id)
    assert all(r.region_id.startswith(parent_id + ".") for r in child.regions)
    assert set(refined.active_leaf_frame_to_region()) == set(flat.frame_to_region)
    # no Si/O/amorphous/crystalline strings anywhere in the produced identities
    blob = "".join(r.region_id for r in refined.regions)
    for banned in ("Si", "O2", "amorphous", "crystalline"):
        assert banned not in blob


# --------------------------------------------------------------------------
# R. synthetic prospective lifecycle end-to-end, child ID preserved through
#    discovery -> refine -> evaluate -> RECOVER
# --------------------------------------------------------------------------
def test_R_prospective_lifecycle_preserves_child_id():
    rep = _rep(species=("A", "B", "C"), n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)

    # assessment -> authorized request -> gated recursive discovery
    assessment = RegionRefinementAssessment(
        region_id=parent_id,
        manifest_sha256=flat.content_sha256(),
        representation_sha256=rep.content_sha256(),
        resolution_status=RegionResolutionStatus.REFINE_SUPPORTED,
        target_relevance_status=TargetRelevanceStatus.TARGET_RELEVANCE_AMBIGUOUS,
        evidence=[
            RegionRefinementEvidence(channel="reproducible_substructure", supports_refinement=True)
        ],
        reasons=["stable substructure"],
    )
    assert refinement_is_authorized(assessment)
    req = authorize_refinement(
        assessment,
        discovery_config=RegionDiscoveryConfig(
            method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2
        ),
    )
    _, refined = _refine(flat, rep, parent_id, auth=req)

    leaf_map = refined.active_leaf_frame_to_region()
    leaf = next(r for r in refined.active_leaf_region_ids() if "." in r)

    # evaluate -> ledger RECOVER on the child leaf -> recovery request keeps it
    binding, records = _eval_on(refined)
    assert leaf in {rec.region_id for rec in records}

    rec = RegionErrorRecord(
        campaign_id="camp",
        iteration=0,
        region_id=leaf,
        region_membership_sha256=refined.content_sha256(),
        state=RegionClosureState.RECOVER,
        failure_reason="under target",
    )
    ledger = ErrorLedger(ledger_id="L", campaign_id="camp").append(rec)
    plan = plan_region_recovery(
        ledger,
        iteration=0,
        eligible_training_candidate_ids=sorted(leaf_map),
        protected_candidate_ids=[],
        sampler=SamplerKind.DIRECT_LIKE,
        n_select=1,
        rationale="recover child leaf",
    )
    assert plan.action == RecoveryAction.ADD_TRAINING_SIDE_CANDIDATES
    assert plan.deficient_region_ids == [leaf]
    result = sample_candidates(plan.sampler_request(region_by_candidate=leaf_map))
    assert all(leaf_map[c] == leaf for c in result.selected_ids)


# --------------------------------------------------------------------------
# authorization gate: UNRESOLVED cannot produce a request (fail closed)
# --------------------------------------------------------------------------
def test_unresolved_cannot_authorize_refinement():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    assessment = RegionRefinementAssessment(
        region_id=parent_id,
        manifest_sha256=flat.content_sha256(),
        representation_sha256=rep.content_sha256(),
        resolution_status=RegionResolutionStatus.UNRESOLVED,
        evidence=[RegionRefinementEvidence(channel="occupancy_pathology", supports_refinement=False)],
        reasons=["ambiguous, no stable substructure"],
    )
    assert not refinement_is_authorized(assessment)
    with pytest.raises(ValueError, match="not authorized"):
        authorize_refinement(
            assessment,
            discovery_config=RegionDiscoveryConfig(
                method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2
            ),
        )
    # and a request cannot be hand-forged for a non-REFINE_SUPPORTED status
    with pytest.raises(ValueError, match="REFINE_SUPPORTED"):
        RegionRefinementRequest(
            parent_region_id=parent_id,
            current_manifest_sha256=flat.content_sha256(),
            representation_sha256=rep.content_sha256(),
            resolution_status=RegionResolutionStatus.UNRESOLVED,
            refinement_reasons=["x"],
            discovery_config=RegionDiscoveryConfig(
                method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2
            ),
        )


# --------------------------------------------------------------------------
# authorization SHA-mismatch guard in discover_structural_subregions
# --------------------------------------------------------------------------
def test_authorization_sha_mismatch_fails_closed():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    req = RegionRefinementRequest(
        parent_region_id=parent_id,
        current_manifest_sha256="wrong",
        representation_sha256=rep.content_sha256(),
        resolution_status=RegionResolutionStatus.REFINE_SUPPORTED,
        refinement_reasons=["r"],
        discovery_config=RegionDiscoveryConfig(
            method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2
        ),
    )
    with pytest.raises(ValueError, match="current_manifest_sha256"):
        discover_structural_subregions(
            parent_manifest=flat,
            parent_region_id=parent_id,
            representation=rep,
            config=RegionDiscoveryConfig(
                method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2
            ),
            manifest_id="x",
            source_sha256="src",
            authorization=req,
        )


# --------------------------------------------------------------------------
# refining a non-leaf (already refined) parent fails closed
# --------------------------------------------------------------------------
def test_cannot_refine_already_refined_parent():
    rep = _rep(n=12)
    flat = _flat(rep, k=3)
    parent_id = _largest_leaf(flat)
    child, refined = _refine(flat, rep, parent_id)
    # Re-discovery against a non-leaf parent fails closed in the recursive API.
    with pytest.raises(ValueError, match="not an active leaf"):
        discover_structural_subregions(
            parent_manifest=refined,
            parent_region_id=parent_id,
            representation=rep,
            config=RegionDiscoveryConfig(
                method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2
            ),
            manifest_id="x",
            source_sha256="src",
        )
    # Re-applying the partition against the already-refined parent fails closed
    # in refine_region_partition itself (independent guard).
    with pytest.raises(ValueError, match="already refined"):
        refine_region_partition(
            manifest_id="again",
            current_manifest=refined,
            parent_region_id=parent_id,
            child_manifest=child,
        )


# --------------------------------------------------------------------------
# mint_child_regions produces namespaced, depth-bound children
# --------------------------------------------------------------------------
def test_mint_child_regions_namespacing():
    rep = _rep()
    flat = _flat(rep)
    parent_id = _largest_leaf(flat)
    parent = flat.region(parent_id)
    members = {f for f, r in flat.frame_to_region.items() if r == parent_id}
    from framework_v2.region_discovery import _subset_representation

    subset = _subset_representation(rep, members)
    sub_flat = discover_structural_regions(
        subset,
        RegionDiscoveryConfig(method=DiscoveryMethod.FARTHEST_CENTROID, n_regions=2),
        manifest_id="sf",
        source_sha256="src",
    )
    children = mint_child_regions(parent=parent, flat_subregion_manifest=sub_flat)
    for r in children.regions:
        assert r.parent_region_id == parent_id
        assert r.discovery_depth == parent.discovery_depth + 1
        assert r.region_id.startswith(parent_id + ".")
        assert r.provider_type == StructuralRegionProviderType.HYBRID
