"""Framework V2 -- AcquisitionStrategyPlanner (generic, evidence-driven).

Which backend a campaign uses is decided *autonomously* from the objective and
the evidence -- never hard-coded to a material. A SiO2 campaign does not have to
use Teacher-driven MD; a raw-pool campaign with no seeds in the target regime
may have to. The planner applies a transparent, material-agnostic decision
procedure over:

  * the admissible backends (feasible ones from the inventory only),
  * the unsaturated CORE_TARGET coverage gaps,
  * a small set of deterministic descriptor-space evidence signals computed
    upstream (whether the pool already covers the gaps, whether existing
    structures can *reach* the gaps by local perturbation, whether the gaps
    require genuinely new configurations only dynamics/generation can produce,
    whether any seed structures exist at all).

The core never computes those signals from raw material data (that is
descriptor-space work owned by the coverage machinery); it consumes them as
evidence and records the exact decision path in ``rationale`` so the choice is
auditable. If no admissible backend can close the gaps, the planner fails
closed with ``StrategyUndecidable`` -- an irreducible gap, never a fabricated
fallback.
"""
from __future__ import annotations

import dataclasses

from framework_v2.acquisition.contracts import (
    AcquisitionStrategy,
    AcquisitionStrategyKind,
    CoverageGapAnalysis,
    SourceAndCapabilityInventory,
)


class StrategyUndecidable(RuntimeError):
    """No admissible backend can close the target coverage gaps. This is an
    irreducible gap the framework surfaces rather than papering over."""


@dataclasses.dataclass(frozen=True)
class StrategyEvidence:
    """Deterministic descriptor-space signals computed upstream from the
    coverage-gap analysis + inventory. Each is a pass/fail judgment the
    planner reasons over; none is a material-specific rule inside the core.

    ``pool_covers_gaps`` -- selecting from the existing pool alone closes the
      unsaturated CORE_TARGET gaps.
    ``parents_reach_gaps`` -- existing structures sit near enough to the gap
      regions that local perturbation can populate them.
    ``gaps_require_new_configurations`` -- at least one gap region is not
      reachable by perturbing existing structures (needs dynamics/generation).
    ``seed_structures_exist`` -- any usable seed structures exist at all.
    ``mixed_backends_required`` -- distinct gap regions need distinct backends.
    """
    pool_covers_gaps: bool
    parents_reach_gaps: bool
    gaps_require_new_configurations: bool
    seed_structures_exist: bool
    mixed_backends_required: bool = False


def select_strategy(
    *,
    strategy_id: str,
    inventory: SourceAndCapabilityInventory,
    coverage: CoverageGapAnalysis,
    evidence: StrategyEvidence,
    evidence_refs: list[str] | None = None,
) -> AcquisitionStrategy:
    """Autonomously choose the acquisition strategy from evidence.

    Deterministic. The chosen kind and the decision path are recorded on the
    returned contract; fails closed with ``StrategyUndecidable`` when no
    admissible backend fits."""
    admissible: dict[AcquisitionStrategyKind, str] = {}
    for b in inventory.feasible_backends():
        # First feasible backend per kind wins a stable id mapping.
        admissible.setdefault(b.strategy_kind, b.backend_id)

    gaps = coverage.unsaturated_core_gaps()
    path: list[str] = [f"admissible_kinds={sorted(k.value for k in admissible)}"]
    path.append(f"unsaturated_core_gaps={len(gaps)}")

    def _pick(kind: AcquisitionStrategyKind, why: str) -> AcquisitionStrategy:
        path.append(f"CHOSE {kind.value}: {why}")
        return AcquisitionStrategy(
            strategy_id=strategy_id,
            kind=kind,
            selected_backend_ids=[admissible[kind]],
            coverage_gap_sha256=coverage.content_sha256(),
            inventory_sha256=inventory.content_sha256(),
            rationale=" | ".join(path),
            evidence_refs=list(evidence_refs or []),
        )

    # HYBRID: distinct gap regions need distinct backends and both are feasible.
    if (
        evidence.mixed_backends_required
        and AcquisitionStrategyKind.LOCAL_PERTURBATION in admissible
        and AcquisitionStrategyKind.TEACHER_DRIVEN_MD in admissible
        and inventory.teacher.can_drive_dynamics
    ):
        path.append("CHOSE HYBRID: distinct gaps need perturbation + dynamics")
        return AcquisitionStrategy(
            strategy_id=strategy_id,
            kind=AcquisitionStrategyKind.HYBRID,
            selected_backend_ids=[
                admissible[AcquisitionStrategyKind.LOCAL_PERTURBATION],
                admissible[AcquisitionStrategyKind.TEACHER_DRIVEN_MD],
            ],
            coverage_gap_sha256=coverage.content_sha256(),
            inventory_sha256=inventory.content_sha256(),
            rationale=" | ".join(path),
            evidence_refs=list(evidence_refs or []),
        )

    # 1. Existing pool already closes the gaps -- OR there are no unsaturated core gaps at all
    #    (geometry/source coverage is saturated). Either way the correct, DECIDABLE outcome is to
    #    select a representative existing subset for canonical Teacher labeling. A saturated pool
    #    with an eligible existing population and a labeling population still to be chosen is a
    #    normal, fully-decidable state -- never StrategyUndecidable (FE-028).
    if (
        AcquisitionStrategyKind.EXISTING_POOL_SELECTION in admissible
        and (evidence.pool_covers_gaps or (not gaps and evidence.seed_structures_exist))
    ):
        return _pick(
            AcquisitionStrategyKind.EXISTING_POOL_SELECTION,
            "existing pool coverage closes unsaturated core gaps" if gaps else
            "no unsaturated core gaps (saturated source coverage): no new-configuration generation "
            "required; select a representative existing subset for canonical labeling",
        )

    # 2. Local perturbation of existing structures reaches the gaps.
    if (
        evidence.parents_reach_gaps
        and not evidence.gaps_require_new_configurations
        and AcquisitionStrategyKind.LOCAL_PERTURBATION in admissible
    ):
        return _pick(
            AcquisitionStrategyKind.LOCAL_PERTURBATION,
            "existing structures reach gap regions via local perturbation",
        )

    # 3. Gaps need genuinely new configurations -> Teacher-driven dynamics.
    if (
        evidence.gaps_require_new_configurations
        and AcquisitionStrategyKind.TEACHER_DRIVEN_MD in admissible
        and inventory.teacher.can_drive_dynamics
    ):
        return _pick(
            AcquisitionStrategyKind.TEACHER_DRIVEN_MD,
            "gaps require new configurations reachable only by dynamics",
        )

    # 4. No seeds at all -> de novo structure generation.
    if (
        not evidence.seed_structures_exist
        and AcquisitionStrategyKind.STRUCTURE_GENERATION in admissible
    ):
        return _pick(
            AcquisitionStrategyKind.STRUCTURE_GENERATION,
            "no seed structures exist; generate de novo",
        )

    raise StrategyUndecidable(
        "No admissible backend can close the target coverage gaps "
        f"(decision path: {' | '.join(path)}). This is an irreducible gap; "
        "the framework will not fabricate a fallback strategy."
    )
