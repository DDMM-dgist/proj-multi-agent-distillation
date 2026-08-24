"""Framework V2 -- TEACHER_DRIVEN_MD candidate generator.

Drives molecular dynamics under the frozen Teacher PES (an ASE-compatible
calculator) to reach configurations that local perturbation of existing
structures cannot -- e.g. melt/quench pathways into an amorphous target. Frames
are sampled at a stride along the trajectory; the trajectory PES is
*exploration only* and never a training label (provenance separation,
Section K). Selected frames are re-labeled canonically downstream and the
geometry written here carries no energies.

The reference implementation uses ASE's velocity-Verlet (NVE) or Langevin (NVT)
integrators, so any frozen MLIP exposing an ASE Calculator can drive it -- no
material- or model-specific coupling. The integrator is injected (``md_engine``)
so the end-to-end flow -- capability detection, protocol validation, sampling,
provenance separation -- is provable with a fake Teacher + fake integrator in
the test environment, while a real campaign uses the built-in ASE integrator.
"""
from __future__ import annotations

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

BACKEND_ID = "teacher_driven_md.ase"

_REQUIRED_PARAMS = (
    "start_structures_path",
    "ensemble",
    "n_steps",
    "timestep_fs",
    "sample_stride",
    "seed",
)
_ENSEMBLES = ("NVE", "NVT_LANGEVIN")
_SUPPORTED_CAPABILITIES = (
    "acquisition.teacher_driven_md",
    "acquisition.ase_dynamics",
    "acquisition.trajectory_sampling",
)

# md_engine signature: (atoms, calc, protocol_params, seed, sample_fn) -> list[Atoms]
# where sample_fn(atoms, step) is called at each sampling stride to record a frame.
MDEngineFn = Callable[[Any, Any, dict, int, Callable[[Any, int], None]], None]


def _ase_md_engine(
    atoms: Any, calc: Any, params: dict, seed: int, sample_fn: Callable[[Any, int], None]
) -> None:
    """Built-in ASE integrator reference implementation."""
    import numpy as np
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    work = atoms.copy()
    work.calc = calc
    ensemble = str(params["ensemble"]).upper()
    dt = float(params["timestep_fs"]) * units.fs
    n_steps = int(params["n_steps"])
    stride = int(params["sample_stride"])
    temperature_K = float(params.get("temperature_K", 0.0))

    rng = np.random.RandomState(seed)
    if temperature_K > 0.0:
        MaxwellBoltzmannDistribution(work, temperature_K=temperature_K, rng=rng)

    if ensemble == "NVE":
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(work, timestep=dt)
    elif ensemble == "NVT_LANGEVIN":
        from ase.md.langevin import Langevin
        friction = float(params.get("friction_per_fs", 0.01)) / units.fs
        dyn = Langevin(
            work, timestep=dt, temperature_K=temperature_K, friction=friction, rng=rng
        )
    else:
        raise ValueError(f"unsupported ensemble: {ensemble}")

    step_counter = {"n": 0}

    def _maybe_sample() -> None:
        n = step_counter["n"]
        if stride > 0 and n % stride == 0:
            sample_fn(work, n)
        step_counter["n"] += 1

    dyn.attach(_maybe_sample, interval=1)
    _maybe_sample()  # sample the initial frame
    dyn.run(n_steps)


class TeacherDynamicsGenerator(CandidateGenerator):
    def __init__(self, *, md_engine: Optional[MDEngineFn] = None) -> None:
        self._md_engine = md_engine

    @property
    def backend_id(self) -> str:
        return BACKEND_ID

    @property
    def strategy_kind(self) -> AcquisitionStrategyKind:
        return AcquisitionStrategyKind.TEACHER_DRIVEN_MD

    def _resolve_engine(self) -> Optional[MDEngineFn]:
        if self._md_engine is not None:
            return self._md_engine
        try:
            import ase.md.verlet  # noqa: F401
            return _ase_md_engine
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
            infeasible_reason="" if feasible else "ase MD integrators unavailable",
        )

    def validate_protocol(self, protocol: GenerationProtocol) -> list[str]:
        issues: list[str] = []
        if protocol.strategy_kind != self.strategy_kind:
            issues.append(
                f"strategy_kind {protocol.strategy_kind} != {self.strategy_kind}"
            )
        if protocol.n_requested <= 0:
            issues.append("n_requested must be positive")
        p = protocol.params
        for key in _REQUIRED_PARAMS:
            if key not in p:
                issues.append(f"missing param: {key}")
        if "ensemble" in p and str(p["ensemble"]).upper() not in _ENSEMBLES:
            issues.append(f"ensemble must be one of {_ENSEMBLES}")
        if "n_steps" in p and int(p["n_steps"]) <= 0:
            issues.append("n_steps must be positive")
        if "timestep_fs" in p and float(p["timestep_fs"]) <= 0.0:
            issues.append("timestep_fs must be positive")
        if "sample_stride" in p and int(p["sample_stride"]) <= 0:
            issues.append("sample_stride must be positive")
        if "ensemble" in p and str(p["ensemble"]).upper() == "NVT_LANGEVIN":
            if float(p.get("temperature_K", 0.0)) <= 0.0:
                issues.append("NVT_LANGEVIN requires temperature_K > 0")
        return issues

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
            raise RuntimeError("teacher dynamics backend infeasible (no MD engine)")
        if teacher is None:
            raise ValueError("teacher dynamics requires a Teacher PES calculator")

        from ase.io import read, write

        os.makedirs(workdir, exist_ok=True)
        starts = read(protocol.params["start_structures_path"], index=":")
        calc = teacher.make_ase_calculator()
        params_sha = protocol.content_sha256()
        n_target = protocol.n_requested

        candidate_ids: list[str] = []
        provenance: list[GenerationProvenance] = []
        out_atoms: list = []

        for s_idx, start in enumerate(starts):
            if len(candidate_ids) >= n_target:
                break
            seed = int(protocol.params["seed"]) + s_idx
            collected: list = []

            def _sample(atoms: Any, step: int, _seed=seed, _sidx=s_idx) -> None:
                if len(candidate_ids) + len(collected) >= n_target:
                    return
                geom = atoms.copy()
                geom.calc = None
                for k in ("energy", "forces", "stress"):
                    geom.info.pop(k, None)
                geom.info["md_step"] = int(step)
                collected.append((geom, _sidx))

            engine(start.copy(), calc, dict(protocol.params), seed, _sample)

            for f_idx, (geom, _sidx) in enumerate(collected):
                cid = f"md:start{_sidx:03d}:step{geom.info.get('md_step', f_idx):06d}"
                geom.info["candidate_id"] = cid
                geom.info["parent_structure_id"] = f"md-start:{_sidx}"
                geom.info["generation_backend"] = self.backend_id
                geom.info["exploration_only"] = True
                out_atoms.append(geom)
                candidate_ids.append(cid)
                provenance.append(
                    GenerationProvenance(
                        candidate_id=cid,
                        strategy_kind=self.strategy_kind,
                        backend_id=self.backend_id,
                        parent_id=f"md-start:{_sidx}",
                        generation_params_sha256=params_sha,
                        exploration_only=True,
                        notes="Teacher-driven MD trajectory frame; PES exploration-only",
                    )
                )

        artifact = os.path.join(workdir, "candidates.xyz")
        if out_atoms:
            write(artifact, out_atoms, format="extxyz")

        n_generated = len(candidate_ids)
        n_rejected = max(0, n_target - n_generated)
        rejection_reasons = (
            {"trajectory_yield_below_target": n_rejected} if n_rejected else {}
        )
        return CandidateGenerationResult(
            result_id=f"gen-{protocol.protocol_id}",
            strategy_sha256=protocol.strategy_sha256,
            backend_id=self.backend_id,
            candidate_ids=candidate_ids,
            provenance=provenance,
            n_requested=n_target,
            n_generated=n_generated,
            n_rejected=n_rejected,
            rejection_reasons=rejection_reasons,
            artifact_ref=artifact if out_atoms else "",
        )
