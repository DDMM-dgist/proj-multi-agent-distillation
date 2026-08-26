"""Framework V2 -- bounded LOCAL_PERTURBATION generation driver (FE-057).

The upstream ``augment_atoms.generate_structures`` acceptance loop is unbounded:

    while len(final_pool) < config.n_per_structure + len(existing_pool):
        ... rattle + relax + accept-or-reject ...

If a parent cannot yield ``n_per_structure`` *distinct*, force/similarity/separation-admissible
children -- because it is degenerate (e.g. a single-atom periodic cell, whose per-atom displacement
is annihilated by the center-of-mass subtraction and whose cell strain is undone by the restoring
matrix before the similarity test) or simply because almost every perturbation relaxes back to a
near-duplicate or blows past the force ceiling -- that loop spins forever. This is a generic
non-termination bug in the backend, not a property of any one structure.

This module reimplements the SAME Metropolis rattle+relax science (identical acceptance gates:
force ceiling, minimum-separation, similarity-threshold; identical log-uniform displacement, optional
cell strain, and level/energy-biased parent selection) but with:

  * a bounded, versioned per-parent STOPPING BUDGET (:class:`StoppingPolicy`) -- the loop can never
    run unbounded; the budget scales with the requested child count, is part of the backend
    capability/provenance, and is hash-bindable into a frozen plan;
  * explicit EXHAUSTION semantics -- a parent that cannot reach its target within budget terminates
    as ``EXHAUSTED_PARTIAL`` with the deficit recorded; missing children are NEVER fabricated;
  * a DEGENERATE-PARENT preflight (:func:`perturbation_admissibility`) that declares a parent
    inadmissible for this backend *before* generation instead of looping;
  * a per-parent :class:`ParentGenerationRecord` (attempts, accepted, rejections-by-reason, elapsed,
    terminal status) so the outcome is fully auditable.

It depends only on numpy + ASE + the injected Teacher calculator, so the bounded loop -- including
its termination, exhaustion and degeneracy behaviour -- is provable with a fake calculator and no
``augment_atoms``/``vesin`` in the test environment.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from typing import Any, Optional

import numpy as np

_VALID_EXHAUSTION = ("allow_partial", "fail_closed")

# Terminal statuses a parent can end in.
STATUS_COMPLETE = "COMPLETE"
STATUS_EXHAUSTED_PARTIAL = "EXHAUSTED_PARTIAL"
STATUS_INADMISSIBLE_DEGENERATE = "INADMISSIBLE_DEGENERATE"


@dataclasses.dataclass(frozen=True)
class StoppingPolicy:
    """Versioned, material-agnostic per-parent stopping budget for LOCAL_PERTURBATION.

    The budget is expressed as attempts PER REQUESTED CHILD (``attempts_per_child_factor``) so it
    scales with the request rather than being a fixed magic number: a parent is allowed up to
    ``ceil(n_per_structure * attempts_per_child_factor)`` rattle+relax trajectories (each of which
    ends in exactly one accept or one categorized rejection) before it is declared exhausted. This
    is a framework default, never a per-run/per-material constant.

    ``exhaustion_policy`` decides what happens when one or more parents finish below target:
    ``"allow_partial"`` records the deficit and continues (honest partial dataset);
    ``"fail_closed"`` refuses to finalize a partial dataset. ``max_wall_seconds_per_parent`` is an
    OPTIONAL secondary wall-clock guard; it defaults to ``None`` (disabled) so execution stays
    deterministic -- the attempt budget is the primary, reproducible bound."""

    attempts_per_child_factor: float = 40.0
    exhaustion_policy: str = "allow_partial"
    max_wall_seconds_per_parent: Optional[float] = None
    version: str = "bounded_perturbation_v1"

    def __post_init__(self) -> None:
        if self.attempts_per_child_factor < 1.0:
            raise ValueError("attempts_per_child_factor must be >= 1.0 (at least one attempt per "
                             "requested child)")
        if self.exhaustion_policy not in _VALID_EXHAUSTION:
            raise ValueError(f"exhaustion_policy must be one of {_VALID_EXHAUSTION}")
        if (self.max_wall_seconds_per_parent is not None
                and self.max_wall_seconds_per_parent <= 0):
            raise ValueError("max_wall_seconds_per_parent must be positive or None")

    def max_attempts_for(self, n_target: int) -> int:
        n_target = int(n_target)
        return max(n_target, int(np.ceil(n_target * self.attempts_per_child_factor)))

    def to_provenance(self) -> dict[str, Any]:
        return {
            "attempts_per_child_factor": self.attempts_per_child_factor,
            "exhaustion_policy": self.exhaustion_policy,
            "max_wall_seconds_per_parent": self.max_wall_seconds_per_parent,
            "version": self.version,
        }

    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_provenance(), sort_keys=True).encode("utf-8")).hexdigest()

    @classmethod
    def from_provenance(cls, d: Optional[dict[str, Any]]) -> "StoppingPolicy":
        if not d:
            return cls()
        return cls(
            attempts_per_child_factor=float(d.get("attempts_per_child_factor", 40.0)),
            exhaustion_policy=str(d.get("exhaustion_policy", "allow_partial")),
            max_wall_seconds_per_parent=(
                None if d.get("max_wall_seconds_per_parent") is None
                else float(d["max_wall_seconds_per_parent"])),
            version=str(d.get("version", "bounded_perturbation_v1")),
        )


@dataclasses.dataclass
class ParentGenerationRecord:
    """Per-parent generation outcome -- fully auditable, no fabrication."""

    parent_id: str
    parent_index: int
    requested: int
    accepted: int
    attempts: int
    max_attempts: int
    rejections: dict[str, int]
    elapsed_s: float
    terminal_status: str
    admissibility_reason: str = ""

    @property
    def deficit(self) -> int:
        return max(0, int(self.requested) - int(self.accepted))

    @property
    def complete(self) -> bool:
        return self.terminal_status == STATUS_COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "parent_index": self.parent_index,
            "requested": int(self.requested),
            "accepted": int(self.accepted),
            "deficit": self.deficit,
            "attempts": int(self.attempts),
            "max_attempts": int(self.max_attempts),
            "rejections": dict(self.rejections),
            "elapsed_s": round(float(self.elapsed_s), 6),
            "terminal_status": self.terminal_status,
            "admissibility_reason": self.admissibility_reason,
        }


def perturbation_admissibility(atoms) -> tuple[bool, str]:
    """Deterministic preflight: can LOCAL_PERTURBATION produce meaningful DISTINCT children from
    this parent at all? Returns ``(admissible, reason)``.

    A structure with fewer than two atoms is inadmissible: the rattle subtracts the center of mass
    from the displacement (``dx -= dx.mean(0)``), which for a single atom is identically zero, and
    any cell strain is removed by the restoring matrix before the similarity comparison -- so every
    candidate reads as a duplicate and the acceptance loop can never make progress. Such a parent
    must be declared inadmissible for this backend (and routed to another strategy) rather than
    entered into generation."""
    if len(atoms) < 2:
        return (False,
                "single_atom_translationally_degenerate: a structure with fewer than two atoms "
                "cannot yield distinct perturbed children (the center-of-mass-subtracted "
                "displacement is identically zero and any cell strain is undone before the "
                "similarity test); LOCAL_PERTURBATION is inadmissible for this parent")
    return (True, "")


# --------------------------------------------------------------------------------------------
# Faithful, self-contained reimplementation of the augment_atoms Metropolis primitives.
# Identical formulas to augment_atoms 0.2.0; kept local so the bounded loop is testable without
# the augment_atoms / vesin runtime dependency.
# --------------------------------------------------------------------------------------------
def _label(atoms, calc):
    s = atoms.copy()
    calc.calculate(s, ["energy", "forces"])
    s.arrays["forces"] = np.asarray(calc.results["forces"])
    s.info["energy"] = float(calc.results["energy"])
    return s


def _max_force(s) -> float:
    return float(np.linalg.norm(s.arrays["forces"], axis=1).max())


def _direction(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1)
    norm[norm < 1e-6] = 1e-6
    return v / norm[:, None]


def _too_similar(pos: np.ndarray, pool: list, threshold: float) -> bool:
    for p in pool:
        if np.linalg.norm(pos - p.positions, axis=1).mean() < threshold:
            return True
    return False


def _select_structure(structures: list, kT: float, beta: float, rand) -> Any:
    energies = np.array([s.info["energy"] / len(s) for s in structures])
    energy_probs = np.exp(-(energies - energies[0]) / kT)
    energy_probs /= energy_probs.sum()
    levels = np.array([s.info["level"] for s in structures]) + 1
    level_probs = levels / levels.sum()
    assert 0 <= beta <= 1, f"beta must be between 0 and 1 (inclusive), got {beta}"
    probs = (1 - beta) * energy_probs + beta * level_probs
    probs[np.isnan(probs)] = 0
    idx = rand.choice(len(structures), p=probs)
    return structures[idx]


def _min_separation_ok(atoms, cutoff: float) -> bool:
    """True iff no interatomic pair is closer than ``cutoff`` (angstrom). Uses vesin when present
    (parity with the real augment_atoms path), else ASE's neighbor list."""
    try:
        import vesin
        (i,) = vesin.ase_neighbor_list("i", atoms, cutoff=cutoff)
        return len(i) == 0
    except Exception:
        from ase.neighborlist import neighbor_list
        i = neighbor_list("i", atoms, cutoff)
        return len(i) == 0


def bounded_generate_for_parent(
    starting_structure,
    calc,
    *,
    config,
    policy: StoppingPolicy,
    parent_id: str,
    parent_index: int,
    seed: int,
) -> tuple[list, ParentGenerationRecord]:
    """Generate up to ``config.n_per_structure`` admissible children from ONE parent under a bounded
    attempt budget. Returns ``(children, record)``.

    Preflight-inadmissible parents return ``([], record[INADMISSIBLE_DEGENERATE])`` with no attempts.
    Otherwise each while-iteration is one attempt (one rattle+relax trajectory) that ends in exactly
    one accept or one categorized rejection; the loop stops when the target is met (COMPLETE) or the
    attempt/time budget is spent (EXHAUSTED_PARTIAL). Deterministic given ``seed``."""
    n_target = int(config.n_per_structure)
    max_attempts = policy.max_attempts_for(n_target)
    rejections = {"force": 0, "similar": 0, "separation": 0}

    admissible, reason = perturbation_admissibility(starting_structure)
    if not admissible:
        return ([], ParentGenerationRecord(
            parent_id=parent_id, parent_index=parent_index, requested=n_target, accepted=0,
            attempts=0, max_attempts=max_attempts, rejections=rejections, elapsed_s=0.0,
            terminal_status=STATUS_INADMISSIBLE_DEGENERATE, admissibility_reason=reason))

    rand = np.random.RandomState(int(seed))
    kT = config.get_kT()
    cell_sigma = config.cell_sigma
    sigma_range = (float(config.sigma_range[0]), float(config.sigma_range[1]))
    max_force_cap = float(config.max_force)
    min_sep = float(config.min_separation)
    sim_thr = float(config.similarity_threshold)
    max_relax_steps = int(config.max_relax_steps)

    seed_s = _label(starting_structure, calc)
    if seed_s.pbc.any():
        seed_s.wrap()
    seed_s.info["id"] = str(uuid.uuid4())
    seed_s.info["level"] = 0
    seed_s.info["parent"] = None
    seed_s.info["relax_steps"] = 0
    final_pool = [seed_s]

    start_time = time.time()
    attempts = 0

    def _budget_left() -> bool:
        if attempts >= max_attempts:
            return False
        if (policy.max_wall_seconds_per_parent is not None
                and (time.time() - start_time) >= policy.max_wall_seconds_per_parent):
            return False
        return True

    while (len(final_pool) - 1) < n_target and _budget_left():
        attempts += 1
        parent = _select_structure(final_pool, kT, config.beta, rand)

        child = parent.copy()
        if cell_sigma is not None:
            cell_change = (
                rand.randn(3, 3)
                * cell_sigma
                * np.linalg.norm(starting_structure.cell.array, axis=1)
            ).T
            child.set_cell(starting_structure.cell.array + cell_change.T, scale_atoms=True)
            restoring_matrix = starting_structure.cell.array @ np.linalg.inv(child.cell.array)
        else:
            restoring_matrix = np.eye(3)

        log_lo, log_hi = np.log(sigma_range)
        sigma = float(np.exp(rand.uniform(log_lo, log_hi)))
        dx = rand.randn(len(child.positions), 3) * sigma
        dx -= np.mean(dx, axis=0)
        child.positions += dx
        child.info["sigma"] = sigma
        child.info["parent"] = parent.info["id"]
        child.info["level"] = parent.info["level"] + 1
        child.info["id"] = str(uuid.uuid4())

        s = _label(child, calc)
        s.info["relax_steps"] = 0
        prev_s = s.copy()

        for i in range(1, max_relax_steps + 2):
            if _too_similar(s.positions @ restoring_matrix, final_pool, sim_thr):
                if not _too_similar(prev_s.positions @ restoring_matrix, final_pool, sim_thr):
                    if not _max_force(prev_s) < max_force_cap:
                        rejections["force"] += 1
                    elif not _min_separation_ok(prev_s, min_sep):
                        rejections["separation"] += 1
                    else:
                        final_pool.append(prev_s)
                else:
                    rejections["similar"] += 1
                break

            if i == max_relax_steps + 1:
                if _max_force(s) > max_force_cap:
                    rejections["force"] += 1
                elif not _min_separation_ok(s, min_sep):
                    rejections["separation"] += 1
                else:
                    final_pool.append(s)
                break

            dE = (s.info["energy"] - parent.info["energy"]) / len(s)
            prob = min(0.25, float(np.exp(-dE / kT)))
            if (_max_force(s) < max_force_cap
                    and rand.uniform() < prob
                    and _min_separation_ok(s, min_sep)):
                final_pool.append(s)
                break

            prev_s = s.copy()
            s.info["relax_steps"] += 1
            direction = _direction(s.arrays["forces"])
            factor = sigma / i
            s.positions += factor * direction
            s = _label(s, calc)

    children = final_pool[1:]
    accepted = len(children)
    terminal = STATUS_COMPLETE if accepted >= n_target else STATUS_EXHAUSTED_PARTIAL
    record = ParentGenerationRecord(
        parent_id=parent_id, parent_index=parent_index, requested=n_target, accepted=accepted,
        attempts=attempts, max_attempts=max_attempts, rejections=rejections,
        elapsed_s=time.time() - start_time, terminal_status=terminal)
    return (children, record)


__all__ = [
    "StoppingPolicy",
    "ParentGenerationRecord",
    "perturbation_admissibility",
    "bounded_generate_for_parent",
    "STATUS_COMPLETE",
    "STATUS_EXHAUSTED_PARTIAL",
    "STATUS_INADMISSIBLE_DEGENERATE",
]
