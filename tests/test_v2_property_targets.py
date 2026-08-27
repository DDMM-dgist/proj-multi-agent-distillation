import pytest

from framework_v2.property_targets import (
    AcceptanceStatus,
    HumanTargetPropertyContract,
    ObservableSpec,
    TargetPropertyFamily,
    default_observable_registry,
    operationalize_target,
)


def test_v2_target_property_required_and_human_bound():
    with pytest.raises(ValueError, match="established by the human"):
        HumanTargetPropertyContract(
            contract_id="target",
            target_property_family=TargetPropertyFamily.STRUCTURAL,
            established_by="agent",
        )


def test_target_family_cannot_be_silently_changed():
    target = HumanTargetPropertyContract(
        contract_id="target",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["msd"],
    )
    with pytest.raises(ValueError, match="human-selected STRUCTURAL"):
        operationalize_target(target)


def test_broad_family_selects_only_allowed_observables():
    target = HumanTargetPropertyContract(
        contract_id="target",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
    )
    contract = operationalize_target(target)

    assert contract.target_property_family == TargetPropertyFamily.STRUCTURAL
    assert contract.observables
    assert {b.observable.family for b in contract.observables} == {
        TargetPropertyFamily.STRUCTURAL
    }


def test_explicitly_required_observable_is_preserved():
    target = HumanTargetPropertyContract(
        contract_id="target",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["rdf", "coordination"],
    )
    contract = operationalize_target(target)

    assert [b.observable.name for b in contract.observables] == ["rdf", "coordination"]
    assert {b.selected_by for b in contract.observables} == {"human_required"}


def test_missing_threshold_remains_unbound_not_fabricated():
    target = HumanTargetPropertyContract(
        contract_id="target",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["density"],
    )
    contract = operationalize_target(target)

    assert contract.threshold_status == AcceptanceStatus.ACCEPTANCE_THRESHOLD_UNBOUND
    assert contract.missing_threshold_observables == ["density"]
    assert contract.observables[0].observable.acceptance_criterion is None


def test_bound_observable_requires_threshold_provenance():
    with pytest.raises(ValueError, match="requires criterion and provenance"):
        ObservableSpec(
            family=TargetPropertyFamily.STRUCTURAL,
            name="bad",
            kernel="plugin:bad",
            acceptance_status=AcceptanceStatus.BOUND,
            acceptance_criterion={"max_error": 1.0},
        )


def test_default_registry_hash_is_deterministic():
    assert (
        default_observable_registry().content_sha256()
        == default_observable_registry().content_sha256()
    )
