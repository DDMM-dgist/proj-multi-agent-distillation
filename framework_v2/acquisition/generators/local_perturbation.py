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
import json
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
from framework_v2.acquisition.generators.bounded_perturbation import (
    STATUS_INADMISSIBLE_DEGENERATE,
    ParentGenerationRecord,
    StoppingPolicy,
    bounded_generate_for_parent,
)


class PerturbationExhausted(RuntimeError):
    """Raised under a ``fail_closed`` stopping policy when one or more parents finish below their
    requested child count. Carries the per-parent records so the caller can surface the deficit."""

    def __init__(self, message: str, records: list) -> None:
        super().__init__(message)
        self.records = records

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
    def __init__(
        self, *, engine: Optional[EngineFn] = None, policy: Optional[StoppingPolicy] = None,
    ) -> None:
        # ``engine`` is retained only for the capability probe (its presence signals the
        # augment_atoms runtime is installed). Generation itself uses the bounded, checkpointing
        # driver in ``bounded_perturbation`` -- never the upstream unbounded acceptance loop.
        self._engine = engine
        self._policy = policy or StoppingPolicy()

    @property
    def policy(self) -> StoppingPolicy:
        return self._policy

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

    PROGRESS_FILENAME = "generation_progress.json"
    CANDIDATES_FILENAME = "candidates.xyz"

    def _progress_path(self, workdir: str) -> str:
        return os.path.join(workdir, self.PROGRESS_FILENAME)

    def _parent_child_file(self, workdir: str, idx: int) -> str:
        return os.path.join(workdir, f"parent_{idx:05d}.children.extxyz")

    @staticmethod
    def _atomic_write_text(path: str, text: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)

    def _to_geometry(self, child, cid: str, parent_id: str):
        """Strip every exploration-PES field so a generation geometry can NEVER become a training
        label (provenance separation), and tag it with its candidate/parent identity."""
        geom = child.copy()
        geom.calc = None
        for k in ("energy", "forces", "stress"):
            geom.info.pop(k, None)
        geom.arrays.pop("forces", None)
        geom.info["candidate_id"] = cid
        geom.info["parent_structure_id"] = parent_id
        geom.info["generation_backend"] = self.backend_id
        geom.info["exploration_only"] = True
        return geom

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
        if teacher is None:
            raise ValueError("local perturbation requires a Teacher PES calculator")

        from ase.io import read, write  # local import; ase is a runtime dep

        os.makedirs(workdir, exist_ok=True)
        parents = read(protocol.params["parents_path"], index=":")
        if not isinstance(parents, list):
            parents = [parents]
        params_sha = protocol.content_sha256()
        base_seed = int(protocol.params["seed"])
        policy = self._policy

        # Resume: reuse per-parent work already checkpointed for THIS exact protocol. A recipe
        # change (different params_sha) invalidates the checkpoint so we never mix recipes.
        prior = self._load_progress(workdir)
        prior_parents = (prior.get("parents") or {}) if prior.get("params_sha256") == params_sha \
            else {}

        # Lazily build the (expensive) Teacher calculator only if some parent must be generated.
        _calc_box: list = []

        def _calc():
            if not _calc_box:
                _calc_box.append(teacher.make_ase_calculator())
            return _calc_box[0]

        records: list[ParentGenerationRecord] = []
        geoms_by_parent: list[list] = []

        for idx, parent_id in enumerate(protocol.parent_ids):
            if idx >= len(parents):
                break
            child_file = self._parent_child_file(workdir, idx)
            prior_rec = prior_parents.get(str(idx))
            if prior_rec is not None and (
                    int(prior_rec.get("accepted", 0)) == 0 or os.path.isfile(child_file)):
                # Completed on a previous invocation -> reuse, never regenerate (no duplication).
                rec = ParentGenerationRecord(
                    parent_id=prior_rec["parent_id"], parent_index=idx,
                    requested=int(prior_rec["requested"]), accepted=int(prior_rec["accepted"]),
                    attempts=int(prior_rec["attempts"]), max_attempts=int(prior_rec["max_attempts"]),
                    rejections=dict(prior_rec.get("rejections") or {}),
                    elapsed_s=float(prior_rec.get("elapsed_s", 0.0)),
                    terminal_status=prior_rec["terminal_status"],
                    admissibility_reason=prior_rec.get("admissibility_reason", ""))
                if int(rec.accepted) > 0 and os.path.isfile(child_file):
                    reloaded = read(child_file, index=":")
                    geoms = reloaded if isinstance(reloaded, list) else [reloaded]
                else:
                    geoms = []
            else:
                cfg = self._build_config(protocol.params, base_seed + idx)
                children, rec = bounded_generate_for_parent(
                    parents[idx].copy(), _calc(), config=cfg, policy=policy,
                    parent_id=parent_id, parent_index=idx, seed=base_seed + idx)
                geoms = [self._to_geometry(c, f"{parent_id}:cand{j:04d}", parent_id)
                         for j, c in enumerate(children)]
                if geoms:
                    self._checkpoint_parent(workdir, child_file, geoms, write)
                elif os.path.isfile(child_file):
                    os.remove(child_file)
            records.append(rec)
            geoms_by_parent.append(geoms)
            self._checkpoint_progress(workdir, params_sha, policy, records, len(parents))

        candidate_ids: list[str] = []
        provenance: list[GenerationProvenance] = []
        out_atoms: list = []
        for geoms in geoms_by_parent:
            for geom in geoms:
                cid = geom.info["candidate_id"]
                out_atoms.append(geom)
                candidate_ids.append(cid)
                provenance.append(GenerationProvenance(
                    candidate_id=cid, strategy_kind=self.strategy_kind, backend_id=self.backend_id,
                    parent_id=geom.info.get("parent_structure_id"),
                    generation_params_sha256=params_sha, exploration_only=True,
                    notes="bounded Metropolis rattle+relax; PES exploration-only"))

        artifact = os.path.join(workdir, self.CANDIDATES_FILENAME)
        if out_atoms:
            write(artifact, out_atoms, format="extxyz")
        elif os.path.isfile(artifact):
            os.remove(artifact)

        # Exhaustion accounting (never fabricate the missing children).
        rejection_reasons = {"force": 0, "similar": 0, "separation": 0}
        exhausted_deficit = 0
        inadmissible_children = 0
        deficient_parents: list[str] = []
        for r in records:
            for k in ("force", "similar", "separation"):
                rejection_reasons[k] += int(r.rejections.get(k, 0))
            if r.terminal_status == STATUS_INADMISSIBLE_DEGENERATE:
                inadmissible_children += int(r.requested)
                deficient_parents.append(r.parent_id)
            elif r.deficit > 0:
                exhausted_deficit += r.deficit
                deficient_parents.append(r.parent_id)
        if exhausted_deficit:
            rejection_reasons["exhausted_deficit_children"] = exhausted_deficit
        if inadmissible_children:
            rejection_reasons["inadmissible_degenerate_children"] = inadmissible_children

        if policy.exhaustion_policy == "fail_closed" and deficient_parents:
            raise PerturbationExhausted(
                "LOCAL_PERTURBATION could not produce the requested children for "
                f"{len(deficient_parents)} parent(s) within the stopping budget "
                f"(policy={policy.version}, exhaustion_policy=fail_closed); "
                f"deficient parents={deficient_parents[:8]}"
                + ("..." if len(deficient_parents) > 8 else ""),
                records=records)

        n_generated = len(candidate_ids)
        n_rejected = max(0, int(protocol.n_requested) - n_generated)
        return CandidateGenerationResult(
            result_id=f"gen-{protocol.protocol_id}",
            strategy_sha256=protocol.strategy_sha256,
            backend_id=self.backend_id,
            candidate_ids=candidate_ids,
            provenance=provenance,
            n_requested=protocol.n_requested,
            n_generated=n_generated,
            n_rejected=n_rejected,
            rejection_reasons=rejection_reasons,
            artifact_ref=artifact if out_atoms else "",
        )

    def _checkpoint_parent(self, workdir: str, child_file: str, geoms: list, write) -> None:
        tmp = f"{child_file}.tmp"
        write(tmp, geoms, format="extxyz")
        os.replace(tmp, child_file)

    def _load_progress(self, workdir: str) -> dict:
        p = self._progress_path(workdir)
        if not os.path.isfile(p):
            return {}
        try:
            return json.loads(open(p).read())
        except (ValueError, OSError):
            return {}

    def _checkpoint_progress(
        self, workdir: str, params_sha: str, policy: StoppingPolicy,
        records: list[ParentGenerationRecord], n_parents: int,
    ) -> None:
        payload = {
            "backend_id": self.backend_id,
            "params_sha256": params_sha,
            "policy": policy.to_provenance(),
            "policy_sha256": policy.content_sha256(),
            "n_parents": int(n_parents),
            "parents": {str(r.parent_index): r.to_dict() for r in records},
        }
        self._atomic_write_text(
            self._progress_path(workdir), json.dumps(payload, indent=2, sort_keys=True) + "\n")
