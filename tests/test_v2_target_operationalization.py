"""V2-H03: target operationalization lifecycle.

A human specifies WHAT physics matters (a target family, optionally required
observables); the agent operationalizes HOW.  A broad family is a valid but
*pending* target that yields RECOMMENDED candidates (candidate generation), not a
fabricated gate contract (selection).  A closure contract exists only once
REQUIRED observables are selected, and the human-selected family can never be
silently changed.
"""
import pytest

from framework_v2.property_targets import (
    AcceptanceStatus,
    HumanTargetPropertyContract,
    ObservableSelectionRole,
    OperationalizationStatus,
    TargetPropertyFamily,
    build_target_validation_contract,
    default_observable_registry,
    operationalize_target_request,
)


def test_broad_family_pending_not_invalid_contract():
    target = HumanTargetPropertyContract(
        contract_id="t", target_property_family=TargetPropertyFamily.STRUCTURAL
    )
    result = operationalize_target_request(target)
    assert result.status == OperationalizationStatus.PENDING_OPERATIONALIZATION
    assert all(d.role == ObservableSelectionRole.RECOMMENDED for d in result.decisions)
    with pytest.raises(ValueError, match="REQUIRED observables"):
        build_target_validation_contract(result, default_observable_registry())


def test_explicit_coordination_required_preserved():
    target = HumanTargetPropertyContract(
        contract_id="t",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["coordination"],
    )
    result = operationalize_target_request(target)
    assert [
        d.observable_name for d in result.decisions if d.role == ObservableSelectionRole.REQUIRED
    ] == ["coordination"]


def test_required_observable_cannot_change_family():
    target = HumanTargetPropertyContract(
        contract_id="t",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["msd"],  # msd is DYNAMICAL
    )
    with pytest.raises(ValueError, match="changed human target family"):
        operationalize_target_request(target)


def test_required_unbound_threshold_blocks_ready_status():
    target = HumanTargetPropertyContract(
        contract_id="t",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["density"],  # registry default is threshold-unbound
    )
    result = operationalize_target_request(target)
    assert result.status == OperationalizationStatus.ACCEPTANCE_THRESHOLD_UNBOUND
    contract = build_target_validation_contract(result, default_observable_registry())
    assert contract.threshold_status == AcceptanceStatus.ACCEPTANCE_THRESHOLD_UNBOUND
    assert contract.missing_threshold_observables == ["density"]


def test_build_contract_preserves_required_and_provenance():
    target = HumanTargetPropertyContract(
        contract_id="t",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["coordination", "density"],
    )
    result = operationalize_target_request(target)
    contract = build_target_validation_contract(result, default_observable_registry())
    assert [b.observable.name for b in contract.observables] == ["coordination", "density"]
    assert {b.selected_by for b in contract.observables} == {"REQUIRED"}
    assert contract.target_property_family == TargetPropertyFamily.STRUCTURAL
    for b in contract.observables:
        assert target.content_sha256() in b.selection_provenance


def test_request_hash_binds_human_target_and_registry():
    target = HumanTargetPropertyContract(
        contract_id="t",
        target_property_family=TargetPropertyFamily.STRUCTURAL,
        required_observables=["coordination"],
    )
    r1 = operationalize_target_request(target)
    r2 = operationalize_target_request(target)
    assert r1.request_sha256 == r2.request_sha256
    assert r1.human_target_sha256 == target.content_sha256()
