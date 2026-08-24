"""Framework V2 -- StudentRecipePlan provenance validator (Section 9).

R31 accepted the Student recipe (30-30 architecture, lr=1e-4, epoch=200,
double precision, ...) by silent legacy reuse: no rationale was ever
recorded for why those values were chosen. Framework V2 refuses.

Rules
-----

* Every ``RecipeParameter`` with ``provenance_class == LEGACY_REUSED``
  or ``TOOL_DEFAULT`` MUST carry a non-empty ``rationale``. This is
  the R31 lesson: legacy reuse is not forbidden, but it must be
  explicitly defended.

* Every ``RecipeParameter`` with ``provenance_class ==
  EVIDENCE_DERIVED`` MUST list at least one ``evidence`` reference.
  Claiming evidence-derivation with no evidence is worse than
  admitting legacy reuse.

* Every ``RecipeParameter`` with ``provenance_class ==
  AGENT_HEURISTIC`` MUST carry a non-empty ``rationale`` describing
  the heuristic (so an auditor can trace back what the agent thought
  it was doing).

* ``HUMAN_FIXED`` and ``FRAMEWORK_CONSTRAINT`` require no rationale
  by default (the class itself is the reason), but MAY carry one.

The validator returns a list of ``RecipeProvenanceViolation`` records;
an empty list means the recipe is valid. Callers (typically the
training-stage gate) must not PASS a recipe with any violation.
"""
from __future__ import annotations

from pydantic import Field

from framework_v2.contracts import (
    ContractBase,
    ProvenanceClass,
    RecipeParameter,
    StudentRecipePlan,
    utc_now_iso,
)


class RecipeProvenanceViolation(ContractBase):
    """One violation of the recipe-provenance rules."""
    parameter_name: str
    provenance_class: ProvenanceClass
    reason: str
    at: str = Field(default_factory=utc_now_iso)


def validate_recipe_provenance(
    recipe: StudentRecipePlan,
) -> list[RecipeProvenanceViolation]:
    """Return violation records; empty list means valid."""
    out: list[RecipeProvenanceViolation] = []
    for param in recipe.all_parameters():
        out.extend(_validate_parameter(param))
    return out


def _validate_parameter(param: RecipeParameter) -> list[RecipeProvenanceViolation]:
    violations: list[RecipeProvenanceViolation] = []
    pc = param.provenance_class

    if pc in {ProvenanceClass.LEGACY_REUSED, ProvenanceClass.TOOL_DEFAULT}:
        if not (param.rationale and param.rationale.strip()):
            violations.append(RecipeProvenanceViolation(
                parameter_name=param.name,
                provenance_class=pc,
                reason=(f"provenance_class={pc.value} requires a non-empty "
                        f"rationale explaining the reuse; parameter "
                        f"{param.name!r} silently inherited a legacy/tool "
                        f"default"),
            ))
    elif pc == ProvenanceClass.EVIDENCE_DERIVED:
        if not param.evidence:
            violations.append(RecipeProvenanceViolation(
                parameter_name=param.name,
                provenance_class=pc,
                reason=(f"provenance_class=EVIDENCE_DERIVED requires at least "
                        f"one evidence ref; parameter {param.name!r} claims "
                        f"evidence-derivation but lists none"),
            ))
    elif pc == ProvenanceClass.AGENT_HEURISTIC:
        if not (param.rationale and param.rationale.strip()):
            violations.append(RecipeProvenanceViolation(
                parameter_name=param.name,
                provenance_class=pc,
                reason=(f"provenance_class=AGENT_HEURISTIC requires a "
                        f"non-empty rationale describing the heuristic; "
                        f"parameter {param.name!r} has none"),
            ))
    # HUMAN_FIXED and FRAMEWORK_CONSTRAINT need no rationale by default.
    return violations


__all__ = [
    "RecipeProvenanceViolation",
    "validate_recipe_provenance",
]
