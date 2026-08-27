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


def operationalize_target(
    human_target: HumanTargetPropertyContract,
    registry: ObservableRegistry | None = None,
    *,
    contract_id: str = "target_validation_v2",
) -> TargetValidationContract:
    """Bind target observables without changing the human-selected family."""

    registry = registry or default_observable_registry()
    requested = list(human_target.required_observables)
    if not requested:
        requested = registry.names_for_family(human_target.target_property_family)
    if human_target.target_property_family == TargetPropertyFamily.USER_DEFINED:
        requested = requested or ["user_defined_plugin"]

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
    "ObservableSpec",
    "TargetObservableBinding",
    "TargetPropertyFamily",
    "TargetValidationContract",
    "default_observable_registry",
    "operationalize_target",
]
