"""Property-guided distillation V2 target and observable contracts.

This module is the public V2 surface for the post-meeting workflow:

    human specifies WHAT physics matters
      -> agent operationalizes HOW to measure it

It deliberately does not run simulations or invent scientific thresholds.
Observable kernels are referenced by name so the existing validation machinery
can remain the numerical implementation.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase, utc_now_iso


class TargetPropertyFamily(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    DYNAMICAL = "DYNAMICAL"
    THERMODYNAMIC = "THERMODYNAMIC"
    MECHANICAL = "MECHANICAL"
    USER_DEFINED = "USER_DEFINED"


class ObservableRequirement(str, Enum):
    GATE_REQUIRED = "GATE_REQUIRED"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"


class AcceptanceStatus(str, Enum):
    BOUND = "BOUND"
    REFERENCE_COMPARISON_AVAILABLE = "REFERENCE_COMPARISON_AVAILABLE"
    ACCEPTANCE_THRESHOLD_UNBOUND = "ACCEPTANCE_THRESHOLD_UNBOUND"
    HUMAN_SCIENTIFIC_INPUT_REQUIRED = "HUMAN_SCIENTIFIC_INPUT_REQUIRED"


class ObservableSelectionRole(str, Enum):
    REQUIRED = "REQUIRED"
    RECOMMENDED = "RECOMMENDED"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    NOT_SELECTED = "NOT_SELECTED"


class OperationalizationStatus(str, Enum):
    PENDING_OPERATIONALIZATION = "PENDING_OPERATIONALIZATION"
    READY_FOR_VALIDATION_CONTRACT = "READY_FOR_VALIDATION_CONTRACT"
    ACCEPTANCE_THRESHOLD_UNBOUND = "ACCEPTANCE_THRESHOLD_UNBOUND"


class HumanTargetPropertyContract(ContractBase):
    """Required scientific input for every new V2 campaign."""

    contract_id: str
    target_property_family: TargetPropertyFamily
    required_observables: list[str] = Field(default_factory=list)
    simulation_conditions: dict[str, Any] = Field(default_factory=dict)
    deployment_context: dict[str, Any] = Field(default_factory=dict)
    user_defined_observables: list[dict[str, Any]] = Field(default_factory=list)
    scientific_rationale: str = ""
    acceptance_sources: list[str] = Field(default_factory=list)
    established_by: str = "human"
    established_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _human_target_is_present(self):
        if not self.contract_id.strip():
            raise ValueError("HumanTargetPropertyContract requires contract_id")
        if self.established_by.lower() != "human":
            raise ValueError("V2 target physics must be established by the human")
        if (
            self.target_property_family == TargetPropertyFamily.USER_DEFINED
            and not self.user_defined_observables
        ):
            raise ValueError("USER_DEFINED target requires user_defined_observables")
        return self


class ObservableSpec(ContractBase):
    family: TargetPropertyFamily
    name: str
    kernel: str
    required_inputs: list[str] = Field(default_factory=list)
    units: str = ""
    metric_definition: str = ""
    reference_source: str = ""
    acceptance_status: AcceptanceStatus = AcceptanceStatus.ACCEPTANCE_THRESHOLD_UNBOUND
    acceptance_criterion: dict[str, Any] | None = None
    acceptance_provenance: list[str] = Field(default_factory=list)
    requirement: ObservableRequirement = ObservableRequirement.GATE_REQUIRED

    @model_validator(mode="after")
    def _criterion_matches_status(self):
        if self.acceptance_status == AcceptanceStatus.BOUND:
            if not self.acceptance_criterion or not self.acceptance_provenance:
                raise ValueError("BOUND observable requires criterion and provenance")
        else:
            if self.acceptance_criterion is not None:
                raise ValueError("unbound observable cannot carry an acceptance criterion")
        return self


class ObservableRegistry(ContractBase):
    registry_id: str
    observables: list[ObservableSpec]

    @model_validator(mode="after")
    def _unique_names(self):
        names = [o.name for o in self.observables]
        if len(set(names)) != len(names):
            raise ValueError("ObservableRegistry contains duplicate observable names")
        return self

    def names_for_family(self, family: TargetPropertyFamily) -> list[str]:
        return [o.name for o in self.observables if o.family == family]

    def get(self, name: str) -> ObservableSpec:
        for observable in self.observables:
            if observable.name == name:
                return observable
        raise ValueError(f"unknown observable {name!r}")


class TargetObservableBinding(ContractBase):
    observable: ObservableSpec
    selected_by: str
    selection_provenance: list[str] = Field(default_factory=list)


class TargetValidationContract(ContractBase):
    contract_id: str
    human_target_sha256: str
    target_property_family: TargetPropertyFamily
    observables: list[TargetObservableBinding]
    threshold_status: AcceptanceStatus
    missing_threshold_observables: list[str] = Field(default_factory=list)
    established_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _nonempty_and_family_preserved(self):
        if not self.observables:
            raise ValueError("TargetValidationContract requires at least one observable")
        for binding in self.observables:
            if binding.observable.family != self.target_property_family:
                raise ValueError("target operationalization changed the human target family")
        return self


class TargetOperationalizationRequest(ContractBase):
    request_id: str
    human_target_sha256: str
    registry_sha256: str
    requires_human_confirmation_for_broad_family: bool = True


class OperationalizationDecision(ContractBase):
    observable_name: str
    role: ObservableSelectionRole
    rationale: str
    decision_provenance: list[str]
    acceptance_status: AcceptanceStatus
    criterion: dict[str, Any] | None = None
    criterion_provenance: list[str] = Field(default_factory=list)


class TargetOperationalizationResult(ContractBase):
    result_id: str
    request_sha256: str
    human_target_sha256: str
    target_property_family: TargetPropertyFamily
    decisions: list[OperationalizationDecision]
    status: OperationalizationStatus


def default_observable_registry() -> ObservableRegistry:
    """Registry backed by existing validation.structure_dynamics kernels."""

    specs = [
        ObservableSpec(
            family=TargetPropertyFamily.STRUCTURAL,
            name="rdf",
            kernel="validation.teacher_physical_validation.evaluate_observable:rdf_peak_position",
            required_inputs=["trajectory", "center_species", "neighbor_species"],
            units="Angstrom",
            metric_definition="partial radial-distribution peak/minimum comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.STRUCTURAL,
            name="partial_rdf",
            kernel="validation.structure_dynamics.compute_rdf_v2",
            required_inputs=["trajectory", "center_species", "neighbor_species"],
            units="dimensionless g(r)",
            metric_definition="partial radial-distribution function comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.STRUCTURAL,
            name="adf",
            kernel="validation.structure_dynamics.compute_adf",
            required_inputs=["trajectory", "center_species", "neighbor_species", "r_cut_A"],
            units="degrees",
            metric_definition="bond-angle distribution summary comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.STRUCTURAL,
            name="coordination",
            kernel="validation.structure_dynamics.compute_species_coordination",
            required_inputs=["trajectory", "center_species", "neighbor_species", "cutoff_A"],
            units="count",
            metric_definition="species coordination distribution comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.STRUCTURAL,
            name="density",
            kernel="validation.structure_dynamics.compute_density",
            required_inputs=["trajectory"],
            units="g/cm^3",
            metric_definition="mean trajectory density comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.STRUCTURAL,
            name="local_structural_descriptor",
            kernel="plugin:local_structural_descriptor",
            required_inputs=["structures"],
            metric_definition="plugin-defined local descriptor comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.DYNAMICAL,
            name="msd",
            kernel="validation.structure_dynamics.compute_msd",
            required_inputs=["trajectory"],
            units="Angstrom^2",
            metric_definition="mean-squared displacement comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.DYNAMICAL,
            name="diffusion_coefficient",
            kernel="validation.structure_dynamics.compute_diffusivity",
            required_inputs=["trajectory", "timestep_fs", "fit_start_frame", "fit_end_frame"],
            units="Angstrom^2/ps",
            metric_definition="linear MSD-slope diffusivity comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.DYNAMICAL,
            name="energy_drift",
            kernel="validation.structure_dynamics.compute_nve_drift",
            required_inputs=["trajectory", "energies", "timestep_fs"],
            units="meV/atom/ps",
            metric_definition="NVE energy drift comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.DYNAMICAL,
            name="temperature_response",
            kernel="plugin:temperature_response",
            required_inputs=["trajectory", "temperature_series"],
            metric_definition="temperature-dependent behavior comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.DYNAMICAL,
            name="vacf",
            kernel="plugin:vacf",
            required_inputs=["trajectory", "velocities"],
            metric_definition="velocity autocorrelation comparison",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.THERMODYNAMIC,
            name="thermodynamic_plugin",
            kernel="plugin:thermodynamic",
            required_inputs=["campaign_specific_evidence"],
            metric_definition="plugin-defined thermodynamic observable",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.MECHANICAL,
            name="mechanical_plugin",
            kernel="plugin:mechanical",
            required_inputs=["campaign_specific_evidence"],
            metric_definition="plugin-defined mechanical observable",
        ),
        ObservableSpec(
            family=TargetPropertyFamily.USER_DEFINED,
            name="user_defined_plugin",
            kernel="plugin:user_defined",
            required_inputs=["user_defined_evidence"],
            metric_definition="user-defined observable comparison",
        ),
    ]
    return ObservableRegistry(registry_id="v2_default_observable_registry", observables=specs)


def create_operationalization_request(
    human_target: HumanTargetPropertyContract,
    registry: ObservableRegistry,
    *,
    request_id: str,
) -> TargetOperationalizationRequest:
    return TargetOperationalizationRequest(
        request_id=request_id,
        human_target_sha256=human_target.content_sha256(),
        registry_sha256=registry.content_sha256(),
    )


def operationalize_target_request(
    human_target: HumanTargetPropertyContract,
    *,
    registry: ObservableRegistry | None = None,
    decisions: list[OperationalizationDecision] | None = None,
    request_id: str = "target_operationalization_request_v2",
    result_id: str = "target_operationalization_result_v2",
) -> TargetOperationalizationResult:
    """Lifecycle entrypoint: human required observables become REQUIRED closure
    decisions; a broad family yields RECOMMENDED candidates only (PENDING), never
    a fabricated gate contract."""

    registry = registry or default_observable_registry()
    request = create_operationalization_request(human_target, registry, request_id=request_id)

    if decisions is None and human_target.required_observables:
        decisions = [
            OperationalizationDecision(
                observable_name=name,
                role=ObservableSelectionRole.REQUIRED,
                rationale="explicitly required by human target",
                decision_provenance=[human_target.content_sha256()],
                acceptance_status=registry.get(name).acceptance_status,
            )
            for name in human_target.required_observables
        ]
    elif decisions is None:
        decisions = [
            OperationalizationDecision(
                observable_name=name,
                role=ObservableSelectionRole.RECOMMENDED,
                rationale="candidate observable for selected family; not closure-required",
                decision_provenance=[registry.content_sha256(), human_target.content_sha256()],
                acceptance_status=registry.get(name).acceptance_status,
            )
            for name in registry.names_for_family(human_target.target_property_family)
        ]

    for d in decisions:
        spec = registry.get(d.observable_name)
        if spec.family != human_target.target_property_family:
            raise ValueError("operationalization changed human target family")
        if (
            d.observable_name in human_target.required_observables
            and d.role != ObservableSelectionRole.REQUIRED
        ):
            raise ValueError("human-required observable must remain REQUIRED")

    required = [d for d in decisions if d.role == ObservableSelectionRole.REQUIRED]
    if not required:
        status = OperationalizationStatus.PENDING_OPERATIONALIZATION
    elif any(d.acceptance_status != AcceptanceStatus.BOUND for d in required):
        status = OperationalizationStatus.ACCEPTANCE_THRESHOLD_UNBOUND
    else:
        status = OperationalizationStatus.READY_FOR_VALIDATION_CONTRACT

    return TargetOperationalizationResult(
        result_id=result_id,
        request_sha256=request.content_sha256(),
        human_target_sha256=human_target.content_sha256(),
        target_property_family=human_target.target_property_family,
        decisions=decisions,
        status=status,
    )


def build_target_validation_contract(
    result: TargetOperationalizationResult,
    registry: ObservableRegistry,
    *,
    contract_id: str = "target_validation_v2",
) -> TargetValidationContract:
    required = [d for d in result.decisions if d.role == ObservableSelectionRole.REQUIRED]
    if not required:
        raise ValueError(
            "cannot build TargetValidationContract before REQUIRED observables are selected"
        )
    missing = [d.observable_name for d in required if d.acceptance_status != AcceptanceStatus.BOUND]
    return TargetValidationContract(
        contract_id=contract_id,
        human_target_sha256=result.human_target_sha256,
        target_property_family=result.target_property_family,
        observables=[
            TargetObservableBinding(
                observable=registry.get(d.observable_name),
                selected_by=d.role.value,
                selection_provenance=d.decision_provenance,
            )
            for d in required
        ],
        threshold_status=(
            AcceptanceStatus.BOUND if not missing else AcceptanceStatus.ACCEPTANCE_THRESHOLD_UNBOUND
        ),
        missing_threshold_observables=missing,
    )


def operationalize_target(
    human_target: HumanTargetPropertyContract,
    registry: ObservableRegistry | None = None,
    *,
    contract_id: str = "target_validation_v2",
) -> TargetValidationContract:
    """Backward-compatible wrapper for the explicit-required-observables path.

    A broad family (no ``required_observables``) is a valid but *pending* target:
    it has no closure-required observables yet, so we do not fabricate a gate
    contract.  Callers must drive the lifecycle via
    :func:`operationalize_target_request` and then
    :func:`build_target_validation_contract` once required observables exist.
    """

    registry = registry or default_observable_registry()
    if not human_target.required_observables:
        raise ValueError(
            "broad-family target has no REQUIRED observables; use "
            "operationalize_target_request() to obtain RECOMMENDED candidates and "
            "build_target_validation_contract() once required observables are selected"
        )
    requested = list(human_target.required_observables)

    bindings: list[TargetObservableBinding] = []
    missing_thresholds: list[str] = []
    for name in requested:
        spec = registry.get(name)
        if spec.family != human_target.target_property_family:
            raise ValueError(
                f"observable {name!r} belongs to {spec.family.value}, not the human-selected "
                f"{human_target.target_property_family.value} family"
            )
        bindings.append(
            TargetObservableBinding(
                observable=spec,
                selected_by="human_required" if name in human_target.required_observables else "agent_registry_default",
                selection_provenance=[registry.content_sha256(), human_target.content_sha256()],
            )
        )
        if spec.acceptance_status != AcceptanceStatus.BOUND:
            missing_thresholds.append(name)

    status = (
        AcceptanceStatus.BOUND
        if not missing_thresholds
        else AcceptanceStatus.ACCEPTANCE_THRESHOLD_UNBOUND
    )
    return TargetValidationContract(
        contract_id=contract_id,
        human_target_sha256=human_target.content_sha256(),
        target_property_family=human_target.target_property_family,
        observables=bindings,
        threshold_status=status,
        missing_threshold_observables=missing_thresholds,
    )


__all__ = [
    "AcceptanceStatus",
    "HumanTargetPropertyContract",
    "ObservableRegistry",
    "ObservableRequirement",
    "ObservableSelectionRole",
    "ObservableSpec",
    "OperationalizationDecision",
    "OperationalizationStatus",
    "TargetObservableBinding",
    "TargetOperationalizationRequest",
    "TargetOperationalizationResult",
    "TargetPropertyFamily",
    "TargetValidationContract",
    "build_target_validation_contract",
    "create_operationalization_request",
    "default_observable_registry",
    "operationalize_target",
    "operationalize_target_request",
]
