"""Framework V2 -- LOCAL_PERTURBATION candidate generator.

Wraps the ``augment_atoms`` Metropolis-rattle+relax generator: existing seed
structures are perturbed (log-uniform absolute-A per-atom displacement, optional
fractional cell perturbation), relaxed under the frozen Teacher PES, and
accepted subject to force/separation/similarity constraints. The Teacher PES
used during relaxation is *exploration only*; it is never a training label
(provenance separation, Section K) -- selected candidates are re-labeled
canonically downstream, and the geometry written here carries no energies.

The heavy ``augment_atoms`` dependency is injected (``engine``) so the wrapper's
end-to-end flow -- protocol validation, per-parent generation, provenance
separation, yield/rejection diagnostics -- is provable with a fake engine and a
fake Teacher, without requiring the generation library or a real Teacher in the
test environment. In a real campaign ``engine`` is left ``None`` and the real
``augment_atoms.generate_structures`` is imported lazily.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Optional

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

BACKEND_ID = "local_perturbation.augment_atoms"

_REQUIRED_PARAMS = (
    "parents_path",
    "n_per_structure",
    "T_K",
    "beta",
    "sigma_range_A",
    "cell_sigma",
    "seed",
    "units",
)
_SUPPORTED_CAPABILITIES = (
    "acquisition.local_perturbation",
    "acquisition.per_parent_seed",
    "acquisition.teacher_pes_relaxation",
)

# augment Config defaults for the pure numerical safety knobs (pinned by the
# frozen acquisition config, not invented here). The planner always overrides
# with the values it proposes; these are only fallbacks so the shim is complete.
_SAFETY_DEFAULTS = {
    "max_force": 30.0,
    "min_separation": 0.5,
    "max_relax_steps": 20,
    "similarity_threshold": 0.1,
}


@dataclasses.dataclass
class _PerturbConfig:
    """augment_atoms.Config-compatible shim (attribute access + get_kT)."""
    n_per_structure: int
    T: float
    beta: float
    sigma_range: tuple[float, float]
    seed: int
    units: str
    cell_sigma: Optional[float]
    max_force: float
    min_separation: float
    max_relax_steps: int
    similarity_threshold: float

    def get_kT(self) -> float:
        if self.units == "eV":
            k = 8.617333262145e-5
        elif self.units == "kcal/mol":
            k = 0.0019872041
        else:
            raise ValueError(f"Unknown units: {self.units}")
        return self.T * k


# engine signature: (starting_structure, calc, existing_pool, config) -> list[Atoms]
EngineFn = Callable[[Any, Any, list, Any], list]


class LocalPerturbationGenerator(CandidateGenerator):
    def __init__(self, *, engine: Optional[EngineFn] = None) -> None:
        self._engine = engine

    @property
    def backend_id(self) -> str:
        return BACKEND_ID

    @property
    def strategy_kind(self) -> AcquisitionStrategyKind:
        return AcquisitionStrategyKind.LOCAL_PERTURBATION

    def _resolve_engine(self) -> Optional[EngineFn]:
        if self._engine is not None:
            return self._engine
        try:
            import augment_atoms  # noqa: F401
            return augment_atoms.generate_structures
        except Exception:
            return None

    def probe(self) -> BackendCapabilityRecord:
        engine = self._resolve_engine()
        feasible = engine is not None
        return BackendCapabilityRecord(
            backend_id=self.backend_id,
            strategy_kind=self.strategy_kind,
            feasible=feasible,
            supported_capabilities=list(_SUPPORTED_CAPABILITIES),
            infeasible_reason=""
            if feasible
            else "augment_atoms not importable and no engine injected",
        )

    def validate_protocol(self, protocol: GenerationProtocol) -> list[str]:
        issues: list[str] = []
        if protocol.strategy_kind != self.strategy_kind:
            issues.append(
                f"strategy_kind {protocol.strategy_kind} != {self.strategy_kind}"
            )
        if protocol.n_requested <= 0:
            issues.append("n_requested must be positive")
        if not protocol.parent_ids:
            issues.append("parent_ids must be non-empty for local perturbation")
        p = protocol.params
        for key in _REQUIRED_PARAMS:
            if key not in p:
                issues.append(f"missing param: {key}")
        if "n_per_structure" in p and int(p["n_per_structure"]) <= 0:
            issues.append("n_per_structure must be positive")
        if "sigma_range_A" in p:
            sr = p["sigma_range_A"]
            if (not isinstance(sr, (list, tuple))) or len(sr) != 2:
                issues.append("sigma_range_A must be a 2-element [lo, hi]")
            elif not (0.0 < float(sr[0]) < float(sr[1])):
                issues.append("sigma_range_A must satisfy 0 < lo < hi")
        if "units" in p and p["units"] not in ("eV", "kcal/mol"):
            issues.append("units must be 'eV' or 'kcal/mol'")
        if "cell_sigma" in p and p["cell_sigma"] is not None:
            if float(p["cell_sigma"]) < 0.0:
                issues.append("cell_sigma must be >= 0 or null")
        return issues

    def _build_config(self, p: dict[str, Any], seed: int) -> _PerturbConfig:
        sr = p["sigma_range_A"]
        return _PerturbConfig(
            n_per_structure=int(p["n_per_structure"]),
            T=float(p["T_K"]),
            beta=float(p["beta"]),
            sigma_range=(float(sr[0]), float(sr[1])),
            seed=int(seed),
            units=str(p["units"]),
            cell_sigma=None if p["cell_sigma"] is None else float(p["cell_sigma"]),
            max_force=float(p.get("max_force", _SAFETY_DEFAULTS["max_force"])),
            min_separation=float(
                p.get("min_separation", _SAFETY_DEFAULTS["min_separation"])
            ),
            max_relax_steps=int(
                p.get("max_relax_steps", _SAFETY_DEFAULTS["max_relax_steps"])
            ),
            similarity_threshold=float(
                p.get("similarity_threshold", _SAFETY_DEFAULTS["similarity_threshold"])
            ),
        )

    def generate(
        self,
        protocol: GenerationProtocol,
        *,
        workdir: str,
        teacher: Optional[TeacherCalculatorProvider] = None,
    ) -> CandidateGenerationResult:
        issues = self.validate_protocol(protocol)
        if issues:
            raise ValueError(f"invalid protocol: {issues}")
        engine = self._resolve_engine()
        if engine is None:
            raise RuntimeError("local perturbation backend infeasible (no engine)")
        if teacher is None:
            raise ValueError("local perturbation requires a Teacher PES calculator")

        from ase.io import read, write  # local import; ase is a runtime dep

        os.makedirs(workdir, exist_ok=True)
        parents = read(protocol.params["parents_path"], index=":")
        calc = teacher.make_ase_calculator()
        params_sha = protocol.content_sha256()

        candidate_ids: list[str] = []
        provenance: list[GenerationProvenance] = []
        out_atoms: list = []
        n_rejected = 0
        rejection_reasons: dict[str, int] = {}

        for idx, parent_id in enumerate(protocol.parent_ids):
            if idx >= len(parents):
                break
            parent = parents[idx].copy()
            # deterministic per-parent seed derived from the plan seed
            cfg = self._build_config(protocol.params, int(protocol.params["seed"]) + idx)
            children = engine(parent, calc, [], cfg)
            # engine returns pool including any seeded parent; keep only children
            produced = [c for c in children if c.info.get("parent") is not None]
            if not produced:
                produced = list(children)
            n_target = cfg.n_per_structure
            n_got = len(produced)
            if n_got < n_target:
                n_rejected += n_target - n_got
                rejection_reasons["yield_below_target"] = (
                    rejection_reasons.get("yield_below_target", 0)
                    + (n_target - n_got)
                )
            for j, child in enumerate(produced):
                cid = f"{parent_id}:cand{j:04d}"
                geom = child.copy()
                geom.calc = None
                # strip any exploration PES so it can never become a label
                for k in ("energy", "forces", "stress"):
                    geom.info.pop(k, None)
                geom.info["candidate_id"] = cid
                geom.info["parent_structure_id"] = parent_id
                geom.info["generation_backend"] = self.backend_id
                geom.info["exploration_only"] = True
                out_atoms.append(geom)
                candidate_ids.append(cid)
                provenance.append(
                    GenerationProvenance(
                        candidate_id=cid,
                        strategy_kind=self.strategy_kind,
                        backend_id=self.backend_id,
                        parent_id=parent_id,
                        generation_params_sha256=params_sha,
                        exploration_only=True,
                        notes="augment-atoms Metropolis rattle+relax; PES exploration-only",
                    )
                )

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
            n_rejected=n_rejected,
            rejection_reasons=rejection_reasons,
            artifact_ref=artifact if out_atoms else "",
        )
