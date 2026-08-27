"""V2-H02: stratified sampling + closure semantics.

DIRECT-like is a structural-stratified diversity selector, not published DIRECT.
Under-budget requests return a typed unresolved status (never silent region-ID
ordering).  Region-ID renaming must not change the selected set.  Closure
distinguishes UNBOUND-required / missing-evidence / evaluated-fail / pass so a
deterministic numeric gate can precede any Judge.
"""
import pytest

from framework_v2.structural_representation import (
    CompositionRepresentationAdapter,
    StructureRecord,
)
from framework_v2.v2_sampling import (
    CriterionBindingStatus,
    CriterionComparator,
    CriterionRole,
    EvidenceStatus,
    RegionClosureState,
    RegionStoppingPolicy,
    SamplerKind,
    SamplerRequest,
    SelectionStatus,
    SignalCriterion,
    criterion_passes,
    sample_candidates,
)


def _rep(ids_species):
    records = [StructureRecord(structure_id=i, species_counts=s) for i, s in ids_species]
    return CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        records, representation_id="rep"
    )


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------
def test_direct_like_under_budget_returns_unresolved_not_alphabetical():
    req = SamplerRequest(
        sampler=SamplerKind.DIRECT_LIKE,
        candidate_ids=["a", "b", "c", "d"],
        n_select=1,
        region_by_candidate={"a": "r1", "b": "r1", "c": "r2", "d": "r2"},
    )
    result = sample_candidates(req)
    assert result.status == SelectionStatus.SELECTION_BUDGET_INSUFFICIENT
    assert result.selected_ids == []
    assert "region" in result.unresolved_reason


def test_direct_like_round_robin_quota_handles_uneven_regions():
    req = SamplerRequest(
        sampler=SamplerKind.DIRECT_LIKE,
        candidate_ids=["a1", "a2", "b1"],
        n_select=3,
        region_by_candidate={"a1": "A", "a2": "A", "b1": "B"},
    )
    result = sample_candidates(req)
    assert result.status == SelectionStatus.SELECTED
    assert set(result.selected_ids) == {"a1", "a2", "b1"}


def test_direct_like_selection_invariant_under_region_rename():
    base = SamplerRequest(
        sampler=SamplerKind.DIRECT_LIKE,
        candidate_ids=["a", "b", "c", "d"],
        n_select=3,
        region_by_candidate={"a": "r1", "b": "r1", "c": "r2", "d": "r2"},
    )
    renamed = SamplerRequest(
        sampler=SamplerKind.DIRECT_LIKE,
        candidate_ids=["a", "b", "c", "d"],
        n_select=3,
        # zzz sorts after aaa; a pure region-ID ordering would flip results
        region_by_candidate={"a": "zzz", "b": "zzz", "c": "aaa", "d": "aaa"},
    )
    assert set(sample_candidates(base).selected_ids) == set(
        sample_candidates(renamed).selected_ids
    )


def test_direct_like_residual_uses_structural_fps_when_representation_present():
    rep = _rep([
        ("a", {"Si": 1, "O": 0}),
        ("b", {"Si": 1, "O": 0}),  # duplicate of a in composition space
        ("c", {"O": 1}),
        ("d", {"Si": 1, "O": 1}),
    ])
    req = SamplerRequest(
        sampler=SamplerKind.DIRECT_LIKE,
        candidate_ids=["a", "b", "c", "d"],
        n_select=3,
        region_by_candidate={"a": "r1", "b": "r1", "c": "r2", "d": "r2"},
    )
    result = sample_candidates(req, representation=rep)
    assert result.status == SelectionStatus.SELECTED
    # coverage picks a (r1) + c (r2); residual FPS prefers d (farther) over b (== a)
    assert result.provenance["residual_fill"] == "global_fps_diversity"
    assert "d" in result.selected_ids
    assert "b" not in result.selected_ids


def test_sampler_excludes_protected_from_all_kinds():
    rep = _rep([
        ("a", {"Si": 1}),
        ("b", {"O": 1}),
        ("c", {"Si": 1, "O": 1}),
        ("d", {"Si": 2, "O": 1}),
    ])
    req = SamplerRequest(
        sampler=SamplerKind.FPS,
        candidate_ids=["a", "b", "c", "d"],
        n_select=2,
        protected_candidate_ids=["d"],
    )
    result = sample_candidates(req, representation=rep)
    assert "d" not in result.selected_ids
    assert result.provenance["protected_excluded"] == 1


def test_sampler_restricts_to_deficient_recovery_regions():
    req = SamplerRequest(
        sampler=SamplerKind.DIRECT_LIKE,
        candidate_ids=["a", "b", "c", "d"],
        n_select=2,
        region_by_candidate={"a": "r1", "b": "r1", "c": "r2", "d": "r2"},
        deficient_region_ids=["r2"],
    )
    result = sample_candidates(req)
    assert set(result.selected_ids) <= {"c", "d"}


# --------------------------------------------------------------------------
# comparators
# --------------------------------------------------------------------------
def test_criterion_passes_all_comparators():
    assert criterion_passes(0.2, 0.3, CriterionComparator.LE)
    assert criterion_passes(0.3, 0.3, CriterionComparator.LE)
    assert not criterion_passes(0.3, 0.3, CriterionComparator.LT)
    assert criterion_passes(0.4, 0.3, CriterionComparator.GE)
    assert not criterion_passes(0.3, 0.3, CriterionComparator.GT)
    assert criterion_passes("amorphous", "amorphous", CriterionComparator.EQ)
    assert not criterion_passes("amorphous", "crystal", CriterionComparator.EQ)


# --------------------------------------------------------------------------
# closure semantics
# --------------------------------------------------------------------------
def _bound(signal, value):
    return SignalCriterion(
        signal=signal,
        role=CriterionRole.SCIENTIFIC_REQUIRED,
        binding_status=CriterionBindingStatus.BOUND,
        comparator=CriterionComparator.LE,
        value=value,
        provenance=["human_contract"],
    )


def test_stopping_distinguishes_unbound_missing_failed_passed():
    unbound = RegionStoppingPolicy(
        policy_id="u",
        criteria=[
            SignalCriterion(
                signal="force_rmse",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.UNBOUND,
                unbound_reason="not set",
            )
        ],
    )
    assert unbound.state_for({"force_rmse": 0.1}) == (
        RegionClosureState.HUMAN_SCIENTIFIC_INPUT_REQUIRED
    )

    bound = RegionStoppingPolicy(policy_id="b", criteria=[_bound("force_rmse", 0.3)])
    assert bound.state_for({}) == RegionClosureState.EVIDENCE_NOT_EVALUATED
    assert bound.state_for({"force_rmse": None}) == RegionClosureState.EVIDENCE_NOT_EVALUATED
    assert bound.state_for({"force_rmse": 0.4}) == RegionClosureState.RECOVER
    assert bound.state_for({"force_rmse": 0.2}) == RegionClosureState.CLOSED


def test_unbound_dominates_missing_and_failed():
    policy = RegionStoppingPolicy(
        policy_id="mixed",
        criteria=[
            _bound("force_rmse", 0.3),
            SignalCriterion(
                signal="target_property",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.UNBOUND,
                unbound_reason="human input required",
            ),
        ],
    )
    # even with a failing bound signal, an unbound required criterion escalates
    assert policy.state_for({"force_rmse": 0.9}) == (
        RegionClosureState.HUMAN_SCIENTIFIC_INPUT_REQUIRED
    )


def test_evidence_only_criterion_never_blocks_closure():
    policy = RegionStoppingPolicy(
        policy_id="ev",
        criteria=[
            _bound("force_rmse", 0.3),
            SignalCriterion(
                signal="wall_time_seconds",
                role=CriterionRole.EVIDENCE_ONLY,
                binding_status=CriterionBindingStatus.UNBOUND,
            ),
        ],
    )
    # evidence-only signal is recorded but does not prevent CLOSED
    assert policy.state_for({"force_rmse": 0.2}) == RegionClosureState.CLOSED
    evals = policy.evaluate_signals({"force_rmse": 0.2, "wall_time_seconds": 12.0})
    ev = [e for e in evals if e.signal == "wall_time_seconds"][0]
    assert ev.role == CriterionRole.EVIDENCE_ONLY
    assert ev.evidence_status == EvidenceStatus.EVALUATED
    assert ev.passed is None


def test_target_property_failure_recovers_even_when_forces_pass():
    policy = RegionStoppingPolicy(
        policy_id="tp",
        criteria=[_bound("force_rmse", 0.3), _bound("target_error", 0.05)],
    )
    assert policy.state_for({"force_rmse": 0.1, "target_error": 0.2}) == (
        RegionClosureState.RECOVER
    )


def test_bound_criterion_requires_comparator_value_provenance():
    with pytest.raises(ValueError, match="BOUND"):
        SignalCriterion(
            signal="force_rmse",
            role=CriterionRole.SCIENTIFIC_REQUIRED,
            binding_status=CriterionBindingStatus.BOUND,
            comparator=CriterionComparator.LE,
            value=0.3,
            provenance=[],  # missing provenance
        )


def test_unbound_criterion_cannot_carry_value():
    with pytest.raises(ValueError, match="UNBOUND"):
        SignalCriterion(
            signal="force_rmse",
            role=CriterionRole.SCIENTIFIC_REQUIRED,
            binding_status=CriterionBindingStatus.UNBOUND,
            comparator=CriterionComparator.LE,
            value=0.3,
        )
