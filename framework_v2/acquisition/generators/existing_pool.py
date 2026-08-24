"""Framework V2 -- EXISTING_POOL_SELECTION candidate "generator" (FE-028).

When the target regime is already covered by an eligible existing pool (no new-configuration
generation gap), Stage-3 acquisition does NOT synthesize new structures. It SELECTS a
representative subset of the EXISTING pool for canonical Teacher labeling. This backend expresses
that as a candidate generator so the strategy planner, the evidence chain, the provenance-
separation invariants, the protected-reference disjointness check, and the Controller binding all
treat it uniformly with the other backends -- no ad-hoc, material-specific path.

Provenance separation (Section K) still holds: the selected existing geometries carry NO energies
(any prior labels are stripped and never used), ``exploration_only`` is True, and the ONLY training
labels come from the downstream canonical ``CanonicalLabelingRequest`` under the frozen Teacher.

Feasibility is environment-level (ASE must be importable to read/write structures); whether a usable
seed pool actually EXISTS for a given campaign is separate evidence the strategy planner reads from
``StrategyEvidence.seed_structures_exist`` -- an infeasible-because-no-pool situation is a strategy
decision, not a backend import failure.
"""
from __future__ import annotations

import os
from typing import Optional

from framework_v2.acquisition.contracts import (
    AcquisitionStrategyKind,
    BackendCapabilityRecord,
    CandidateGenerationResult,
    GenerationProvenance,
)
from framework_v2.acquisition.generators.base import (
    CandidateGenerator,
    GenerationProtocol,
    TeacherCalculatorProvider,
)

BACKEND_ID = "existing_pool_selection.ase"

_SUPPORTED_CAPABILITIES = (
    "acquisition.existing_pool_selection",
    "acquisition.representative_subset",
    "acquisition.no_new_configuration_generation",
)


class ExistingPoolSelectionGenerator(CandidateGenerator):
    """Selects (does not synthesize) a representative subset of the existing pool for labeling."""

    @property
    def backend_id(self) -> str:
        return BACKEND_ID

    @property
    def strategy_kind(self) -> AcquisitionStrategyKind:
        return AcquisitionStrategyKind.EXISTING_POOL_SELECTION

    def probe(self) -> BackendCapabilityRecord:
        try:
            import ase.io  # noqa: F401
            feasible = True
            reason = ""
        except Exception:  # pragma: no cover - ase is a hard runtime dep here
            feasible = False
            reason = "ase.io not importable; cannot read/write existing-pool structures"
        return BackendCapabilityRecord(
            backend_id=self.backend_id,
            strategy_kind=self.strategy_kind,
            feasible=feasible,
            supported_capabilities=list(_SUPPORTED_CAPABILITIES),
            infeasible_reason=reason)

    def validate_protocol(self, protocol: GenerationProtocol) -> list[str]:
        issues: list[str] = []
        if protocol.strategy_kind != self.strategy_kind:
            issues.append(f"strategy_kind {protocol.strategy_kind} != {self.strategy_kind}")
        if protocol.n_requested <= 0:
            issues.append("n_requested must be positive")
        if not protocol.parent_ids:
            issues.append("parent_ids (the existing frames to select) must be non-empty")
        p = protocol.params
        pool_path = p.get("pool_path")
        if not pool_path:
            issues.append("missing param: pool_path (the existing pool structure file)")
        idxs = p.get("selected_source_global_indices")
        if not isinstance(idxs, (list, tuple)) or not idxs:
            issues.append("missing/empty param: selected_source_global_indices")
        elif any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in idxs):
            issues.append("selected_source_global_indices must be non-negative integers")
        elif len(set(idxs)) != len(idxs):
            issues.append("selected_source_global_indices must be unique")
        return issues

    def generate(
        self,
        protocol: GenerationProtocol,
        *,
        workdir: str,
        teacher: Optional[TeacherCalculatorProvider] = None,
    ) -> CandidateGenerationResult:
        """Materialize the selected existing frames as candidate geometries (energies stripped).

        No Teacher PES is used here -- selection is descriptor/geometry work, never labeling. The
        written geometry carries no energies so a prior label can never leak into training; the
        canonical labels come only from the downstream ``CanonicalLabelingRequest``."""
        issues = self.validate_protocol(protocol)
        if issues:
            raise ValueError(f"invalid protocol: {issues}")

        from ase.io import read, write  # local import; ase is a runtime dep

        os.makedirs(workdir, exist_ok=True)
        p = protocol.params
        frames = read(str(p["pool_path"]), index=":")
        selected_indices = list(p["selected_source_global_indices"])
        params_sha = protocol.content_sha256()

        candidate_ids: list[str] = []
        provenance: list[GenerationProvenance] = []
        out_atoms: list = []
        for pid, src_index in zip(protocol.parent_ids, selected_indices):
            if src_index < 0 or src_index >= len(frames):
                raise ValueError(
                    f"selected_source_global_index {src_index} out of range for pool of "
                    f"{len(frames)} frames")
            geom = frames[src_index].copy()
            geom.calc = None
            for k in ("energy", "forces", "stress"):
                geom.info.pop(k, None)
            geom.info["candidate_id"] = pid
            geom.info["parent_structure_id"] = pid
            geom.info["source_global_index"] = int(src_index)
            geom.info["generation_backend"] = self.backend_id
            geom.info["exploration_only"] = True
            out_atoms.append(geom)
            candidate_ids.append(pid)
            provenance.append(GenerationProvenance(
                candidate_id=pid, strategy_kind=self.strategy_kind, backend_id=self.backend_id,
                parent_id=pid, generation_params_sha256=params_sha, exploration_only=True,
                notes="existing-pool selection; prior labels stripped; canonical relabel downstream"))

        artifact = os.path.join(workdir, "candidates.xyz")
        if out_atoms:
            write(artifact, out_atoms, format="extxyz")

        return CandidateGenerationResult(
            result_id=f"gen-{protocol.protocol_id}",
            strategy_sha256=protocol.strategy_sha256,
            backend_id=self.backend_id,
            candidate_ids=candidate_ids,
            provenance=provenance,
            n_requested=protocol.n_requested,
            n_generated=len(candidate_ids),
            n_rejected=0,
            rejection_reasons={},
            artifact_ref=artifact if out_atoms else "")
