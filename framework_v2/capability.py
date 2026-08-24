"""Framework V2 — planner/executor capability negotiation (Section 17).

R31 saw the classic silent-degradation failure: the scientific planner
requested per-parent augmentation policies while the executor supported
only one global ``n_per_structure``. Rather than surfacing this as a
capability gap, the runtime silently averaged the per-parent counts into
one number, and the executed acquisition no longer matched the plan.

Framework V2 refuses that. Every plan declares
``required_capabilities`` (a list of stable capability names). Every
executor declares ``supported_capabilities``. Before dispatch the two
sets are compared. If the planner requires anything the executor does
not advertise, dispatch fails closed with
``FRAMEWORK_CAPABILITY_BLOCKER``. The scientific plan is not modified.

The capability namespace is deliberately open (freeform strings) so new
capabilities can be introduced without changing this module. Convention:
``<domain>.<capability>[.<qualifier>]`` -- e.g.
``acquisition.per_parent_augmentation_count``,
``acquisition.per_parent_amplitude_range``,
``training.convergence_report``.

This module is *only* the negotiator. Dispatch code (e.g. the pydantic-ai
dispatch bridge) is responsible for calling ``check_capabilities`` before
executing, and for translating a returned ``FrameworkCapabilityBlocker``
into whatever error/state its stage lifecycle needs.
"""
from __future__ import annotations

from typing import Iterable

from pydantic import Field

from framework_v2.contracts import ContractBase, utc_now_iso


FRAMEWORK_CAPABILITY_BLOCKER = "FRAMEWORK_CAPABILITY_BLOCKER"


class ExecutorCapabilities(ContractBase):
    """What one executor advertises. Immutable and content-addressable so
    a dispatch log can record which capability set was in effect for a
    given run."""
    executor_id: str
    supported: list[str]

    def supports(self, name: str) -> bool:
        return name in self.supported


class PlanRequirements(ContractBase):
    """The capabilities a specific plan needs. Plans (like
    AugmentationPlan) can populate this from their own
    ``required_capabilities`` field."""
    plan_id: str
    required: list[str]


class FrameworkCapabilityBlocker(ContractBase):
    """The blocker record produced when a plan requires capabilities the
    executor does not advertise. Downstream code MUST NOT proceed with
    execution while this exists; it must either revise the plan or
    upgrade the executor."""
    status: str = Field(default=FRAMEWORK_CAPABILITY_BLOCKER, frozen=True)
    plan_id: str
    executor_id: str
    unmet_requirements: list[str]
    detected_at: str = Field(default_factory=utc_now_iso)
    rationale: str = ""


def check_capabilities(
    requirements: PlanRequirements,
    executor: ExecutorCapabilities,
    *,
    rationale: str = "",
) -> FrameworkCapabilityBlocker | None:
    """Return a blocker record if any requirement is unmet, else
    ``None``. This is the single primitive the dispatch layer calls.

    Callers must not "downgrade" the plan when a blocker is returned;
    that is precisely the R31 failure this module exists to prevent."""
    missing = [r for r in requirements.required if not executor.supports(r)]
    if not missing:
        return None
    return FrameworkCapabilityBlocker(
        plan_id=requirements.plan_id,
        executor_id=executor.executor_id,
        unmet_requirements=missing,
        rationale=rationale or (
            f"executor {executor.executor_id!r} does not advertise: "
            f"{missing}"
        ),
    )


def merge_capabilities(*sets: Iterable[str]) -> list[str]:
    """Utility to deduplicate/normalize a capability list from multiple
    sources (e.g. an executor stack that composes several sub-runtimes)."""
    seen: set[str] = set()
    out: list[str] = []
    for s in sets:
        for name in s:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


# ---------------------------------------------------------------------
# Convenience: derive per-parent-augmentation requirements from an
# AugmentationPlan. This is where R31's silent-flattening bug lived,
# so we make the derivation explicit and testable.
# ---------------------------------------------------------------------
def augmentation_capability_requirements(
    plan_id: str,
    is_heterogeneous: bool,
) -> PlanRequirements:
    """A heterogeneous augmentation plan requires per-parent capability.
    A homogeneous plan does not (a global executor can produce the same
    frames). ``AugmentationPlan.is_heterogeneous()`` is the source of
    truth; passing the boolean explicitly here keeps this module
    dependency-free of the contracts serialisation surface."""
    reqs: list[str] = []
    if is_heterogeneous:
        reqs.append("acquisition.per_parent_augmentation_count")
        reqs.append("acquisition.per_parent_amplitude_range")
        reqs.append("acquisition.per_parent_method")
    return PlanRequirements(plan_id=plan_id, required=reqs)


__all__ = [
    "FRAMEWORK_CAPABILITY_BLOCKER",
    "ExecutorCapabilities",
    "PlanRequirements",
    "FrameworkCapabilityBlocker",
    "check_capabilities",
    "merge_capabilities",
    "augmentation_capability_requirements",
]
