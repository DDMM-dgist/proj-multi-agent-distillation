"""V2-H12: scientific target taxonomy hardening.

Three orthogonal concepts are tested:

    WHAT  -- TargetPropertyFamily (property family)
    WHERE -- EvaluationDomain (temperature / composition / structural region)
    WHY   -- CriterionRole observable role (scientific / operational / guard / evidence)

The taxonomy *supports* every family, but which observables are REQUIRED for a
campaign is an explicit selection, never an automatic consequence of taxonomy.
No acceptance thresholds are invented here.
"""
import pytest

from framework_v2.property_targets import (
    CampaignTargetSelection,
    CriterionRole,
    DomainResolvedObservable,
    EvaluationDomain,
    ObservableSelectionStatus,
    TargetObservableChannel,
    TargetPropertyFamily as F,
    default_observable_registry,
    sio2_fresh01_target_selection,
)
from framework_v2.v2_sampling import (
    CriterionBindingStatus,
    RegionClosureState,
    RegionStoppingPolicy,
    SignalCriterion,
)


def _spec(name):
    return default_observable_registry().get(name)


# ---------------------------------------------------------------------------
# A. PROPERTY FAMILY  (WHAT)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,family",
    [
        ("rdf", F.STRUCTURAL),
        ("partial_rdf", F.STRUCTURAL),
        ("adf", F.STRUCTURAL),
        ("coordination", F.STRUCTURAL),
        ("density", F.THERMODYNAMIC),
        ("diffusion_coefficient", F.TRANSPORT),
        ("msd", F.TRANSPORT),
        ("vacf", F.DYNAMICAL),
        ("vdos", F.DYNAMICAL),
    ],
)
def test_property_family_classification(name, family):
    assert _spec(name).family == family


def test_transport_and_kinetic_families_exist():
    assert F.TRANSPORT.value == "TRANSPORT"
    assert F.KINETIC.value == "KINETIC"


# ---------------------------------------------------------------------------
# B. OBSERVABLE ROLE SEPARATION  (WHY)
# ---------------------------------------------------------------------------
def test_scientific_targets_are_scientific():
    for name in ("rdf", "adf", "coordination", "density"):
        assert _spec(name).observable_role == CriterionRole.SCIENTIFIC_REQUIRED
        assert _spec(name).is_scientific_target()


def test_energy_and_force_fidelity_are_operational_not_a_target_family():
    # E/F fidelity live in the energy.*/force.* closure namespace as
    # OPERATIONAL_REQUIRED criteria; they are not target observable families.
    from framework_v2.v2_sampling import CriterionComparator

    e = SignalCriterion(
        signal="energy.rmse_meV_per_atom",
        role=CriterionRole.OPERATIONAL_REQUIRED,
        binding_status=CriterionBindingStatus.BOUND,
        comparator=CriterionComparator.LE,
        value=25.0,
        units="meV/atom",
        provenance=["01_evaluation_adequacy"],
    )
    assert e.role == CriterionRole.OPERATIONAL_REQUIRED
    reg_names = {o.name for o in default_observable_registry().observables}
    assert "energy_rmse" not in reg_names and "force_rmse" not in reg_names


def test_nve_drift_is_a_numerical_guard_not_a_scientific_target():
    d = _spec("energy_drift")
    assert d.observable_role == CriterionRole.NUMERICAL_GUARD
    assert not d.is_scientific_target()


def test_evidence_only_observable_does_not_block_closure():
    policy = RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            SignalCriterion(
                signal="target.structural.rdf",
                role=CriterionRole.EVIDENCE_ONLY,
                binding_status=CriterionBindingStatus.UNBOUND,
            )
        ],
    )
    # An evidence-only, unbound signal must not force HUMAN_SCIENTIFIC_INPUT.
    assert policy.state_for({"target.structural.rdf": 0.1}) == RegionClosureState.CLOSED


def _guard_policy(scope=None):
    from framework_v2.v2_sampling import ClosurePolicyScope, CriterionComparator

    kwargs = {} if scope is None else {"scope": scope}
    return RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            SignalCriterion(
                signal="target.dynamical.nve_drift",
                role=CriterionRole.NUMERICAL_GUARD,
                binding_status=CriterionBindingStatus.BOUND,
                comparator=CriterionComparator.LE,
                value=1.0,
                units="meV/atom/ps",
                provenance=["guard"],
            )
        ],
        **kwargs,
    )


def test_scientific_required_failure_drives_region_recover():
    from framework_v2.v2_sampling import CriterionComparator

    policy = RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            SignalCriterion(
                signal="target.structural.rdf",
                role=CriterionRole.SCIENTIFIC_REQUIRED,
                binding_status=CriterionBindingStatus.BOUND,
                comparator=CriterionComparator.LE,
                value=0.2,
                units="",
                provenance=["sci"],
            )
        ],
    )
    assert policy.state_for({"target.structural.rdf": 0.9}) == RegionClosureState.RECOVER
    assert policy.state_for({"target.structural.rdf": 0.1}) == RegionClosureState.CLOSED


def test_operational_required_failure_drives_region_recover():
    from framework_v2.v2_sampling import CriterionComparator

    policy = RegionStoppingPolicy(
        policy_id="p",
        criteria=[
            SignalCriterion(
                signal="energy.rmse_meV_per_atom",
                role=CriterionRole.OPERATIONAL_REQUIRED,
                binding_status=CriterionBindingStatus.BOUND,
                comparator=CriterionComparator.LE,
                value=25.0,
                units="meV/atom",
                provenance=["op"],
            )
        ],
    )
    assert policy.state_for({"energy.rmse_meV_per_atom": 40.0}) == RegionClosureState.RECOVER


def test_numerical_guard_does_not_auto_drive_region_recover():
    # H12.1: a failing stability guard must NOT imply that a structural region
    # needs more training data.  In the default REGION_RECOVERY scope the guard
    # is non-gating, so a guard failure does not produce RECOVER.
    policy = _guard_policy()  # default scope == REGION_RECOVERY
    assert policy.state_for({"target.dynamical.nve_drift": 5.0}) == RegionClosureState.CLOSED
    assert policy.state_for({"target.dynamical.nve_drift": 0.2}) == RegionClosureState.CLOSED
    # And it is reported as explicitly non-gating rather than passed/failed.
    ev = policy.evaluate_signals({"target.dynamical.nve_drift": 5.0})[0]
    assert ev.passed is None
    assert "does not drive region recovery" in ev.reason


def test_numerical_guard_blocks_final_validation_scope():
    # H12.1: an explicit FINAL_VALIDATION policy DOES let the guard block closure.
    from framework_v2.v2_sampling import ClosurePolicyScope

    policy = _guard_policy(ClosurePolicyScope.FINAL_VALIDATION)
    assert policy.state_for({"target.dynamical.nve_drift": 5.0}) == RegionClosureState.RECOVER
    assert policy.state_for({"target.dynamical.nve_drift": 0.2}) == RegionClosureState.CLOSED


def test_selecting_dynamical_family_does_not_recommend_nve_guard():
    # H12.1 issue 2: requesting the DYNAMICAL family for automatic candidate
    # selection must not pull in energy_drift (a NUMERICAL_GUARD), but must still
    # offer the genuine DYNAMICAL scientific targets vacf/vdos.
    reg = default_observable_registry()
    scientific = set(reg.scientific_names_for_family(F.DYNAMICAL))
    assert "energy_drift" not in scientific
    assert {"vacf", "vdos"} <= scientific
    # energy_drift is still a member of the family classification (WHAT), just
    # not a scientific target (WHY).
    assert "energy_drift" in set(reg.names_for_family(F.DYNAMICAL))


def test_broad_dynamical_request_recommends_only_scientific_targets():
    from framework_v2.property_targets import (
        HumanTargetPropertyContract,
        ObservableSelectionRole,
        operationalize_target_request,
    )

    target = HumanTargetPropertyContract(
        contract_id="t", target_property_family=F.DYNAMICAL
    )
    result = operationalize_target_request(target)
    recommended = {
        d.observable_name for d in result.decisions
        if d.role == ObservableSelectionRole.RECOMMENDED
    }
    assert "energy_drift" not in recommended
    assert {"vacf", "vdos"} <= recommended


def test_validation_contract_rejects_numerical_guard_as_scientific_target():
    from framework_v2.property_targets import (
        HumanTargetPropertyContract,
        build_target_validation_contract,
        operationalize_target_request,
    )

    target = HumanTargetPropertyContract(
        contract_id="t",
        target_property_family=F.DYNAMICAL,
        required_observables=["energy_drift"],  # a NUMERICAL_GUARD, not a target
    )
    result = operationalize_target_request(target)
    with pytest.raises(ValueError, match="non-scientific observables"):
        build_target_validation_contract(result, default_observable_registry())


# ---------------------------------------------------------------------------
# C. DOMAIN SEPARATION  (WHERE) — family invariant across domain
# ---------------------------------------------------------------------------
def test_same_rdf_evaluated_over_many_temperatures_same_family():
    o = _spec("partial_rdf")
    domains = [EvaluationDomain(temperature_K=t, composition="SiO2-x", structural_region_id="R1")
               for t in (300.0, 1000.0, 2000.0, 4000.0)]
    resolved = [DomainResolvedObservable(observable=o, domain=d) for d in domains]
    assert {r.family() for r in resolved} == {F.STRUCTURAL}
    assert len({r.resolved_signal() for r in resolved}) == 4  # domain-resolved evidence


def test_same_rdf_over_many_compositions_and_regions_same_family():
    o = _spec("partial_rdf")
    comps = [DomainResolvedObservable(observable=o, domain=EvaluationDomain(composition=c))
             for c in ("SiO2", "SiO1.8", "SiO1.5")]
    regs = [DomainResolvedObservable(observable=o, domain=EvaluationDomain(structural_region_id=r))
            for r in ("R1", "R2", "R3")]
    assert {x.family() for x in comps + regs} == {F.STRUCTURAL}
    assert o.family == F.STRUCTURAL  # observable family unchanged by domain


# ---------------------------------------------------------------------------
# D. TARGET HIERARCHY — channel identity preserved as structured metadata
# ---------------------------------------------------------------------------
def test_rdf_pair_channel_identity_preserved():
    c = TargetObservableChannel(observable_kind="rdf", family=F.STRUCTURAL,
                                channel={"pair": ["Si", "O"]})
    assert c.channel["pair"] == ["Si", "O"]
    assert c.channel_id() == "rdf::pair=Si-O"


def test_adf_angle_channel_identity_preserved():
    c = TargetObservableChannel(observable_kind="adf", family=F.STRUCTURAL,
                                channel={"angle": ["O", "Si", "O"]})
    assert c.channel["angle"] == ["O", "Si", "O"]
    assert c.channel_id() == "adf::angle=O-Si-O"


def test_coordination_species_identity_preserved():
    si = TargetObservableChannel(observable_kind="coordination", family=F.STRUCTURAL,
                                 channel={"center_species": "Si"})
    o = TargetObservableChannel(observable_kind="coordination", family=F.STRUCTURAL,
                                channel={"center_species": "O"})
    assert si.channel_id() != o.channel_id()


def test_diffusivity_species_identity_preserved():
    si = TargetObservableChannel(observable_kind="diffusivity", family=F.TRANSPORT,
                                 channel={"species": "Si"})
    o = TargetObservableChannel(observable_kind="diffusivity", family=F.TRANSPORT,
                                channel={"species": "O"})
    assert si.family == F.TRANSPORT and o.family == F.TRANSPORT
    assert si.channel_id() == "diffusivity::species=Si"
    assert o.channel_id() == "diffusivity::species=O"


# ---------------------------------------------------------------------------
# E. FIRST FRESH CAMPAIGN SELECTION — explicit, not automatic
# ---------------------------------------------------------------------------
def test_fresh01_primary_targets():
    sel = sio2_fresh01_target_selection()
    assert sel.primary_kinds() == {"rdf", "adf", "coordination", "density"}


def test_fresh01_does_not_auto_require_secondary_targets():
    sel = sio2_fresh01_target_selection()
    for kind in ("diffusivity", "vacf", "vdos", "msd"):
        assert not sel.requires(kind)
    assert sel.secondary_kinds() == {"diffusivity", "vacf", "vdos"}


def test_fresh01_primary_channels_cover_partial_rdf_and_adf_angles():
    sel = sio2_fresh01_target_selection()
    rdf_pairs = {tuple(c.channel["pair"]) for c in sel.primary if c.observable_kind == "rdf"}
    adf_angles = {tuple(c.channel["angle"]) for c in sel.primary if c.observable_kind == "adf"}
    assert rdf_pairs == {("Si", "O"), ("Si", "Si"), ("O", "O")}
    assert adf_angles == {("O", "Si", "O"), ("Si", "O", "Si")}


def test_selection_rejects_primary_secondary_overlap():
    ch = TargetObservableChannel(observable_kind="rdf", family=F.STRUCTURAL,
                                 channel={"pair": ["Si", "O"]})
    ch_sec = TargetObservableChannel(observable_kind="rdf", family=F.STRUCTURAL,
                                     channel={"pair": ["Si", "O"]},
                                     selection_status=ObservableSelectionStatus.SECONDARY_OPTIONAL)
    with pytest.raises(ValueError, match="both primary and secondary"):
        CampaignTargetSelection(campaign_id="c", primary=[ch], secondary=[ch_sec])


# ---------------------------------------------------------------------------
# F. COMPARISON SEMANTICS — declared but thresholds NOT invented
# ---------------------------------------------------------------------------
def test_channel_metric_definition_defaults_unbound():
    c = TargetObservableChannel(observable_kind="rdf", family=F.STRUCTURAL,
                                channel={"pair": ["Si", "O"]})
    assert c.metric_definition == "UNBOUND"


def test_signal_namespace_is_family_qualified():
    assert _spec("rdf").signal_namespace() == "target.structural.rdf"
    assert _spec("density").signal_namespace() == "target.thermodynamic.density"
    assert _spec("diffusion_coefficient").signal_namespace() == "target.transport.diffusivity"
