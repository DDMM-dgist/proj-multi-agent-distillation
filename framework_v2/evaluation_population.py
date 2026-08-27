"""Framework V2 -- role-bound multi-population Stage-8 evaluation contract.

Stage-8 fidelity is not a single number over a single population. A
distillation campaign must separate, at minimum:

  * the *distillation* claim  -- does the Student reproduce the Teacher it
    was trained to imitate? -- which is only meaningful on a population
    carrying Teacher labels (Student-vs-Teacher);
  * the *physical-accuracy* claim -- is the Student (and the Teacher it
    imitates) actually right against first-principles reference data? --
    which requires a population carrying DFT labels
    (Student-vs-DFT, Teacher-vs-DFT).

These are DIFFERENT populations with DIFFERENT label requirements, and one
must never be silently substituted for the other. This module makes the
separation a typed, hash-bound contract rather than a workflow convention.

The core is deliberately material-agnostic: it enumerates only the generic
distillation channels and the two generic population roles. Concrete file
paths, SHAs, and which channels a run actually claims are per-run inputs
bound into a ``MultiPopulationEvaluationPlan`` -- never hard-coded here.

Fail-closed guarantees enforced structurally:
  * role<->channel: a population may only carry channels its role permits
    (a Teacher-only distillation holdout can never be asked for a
    DFT channel; a DFT holdout is the only place DFT channels may come
    from). Enforced at contract construction.
  * channel uniqueness: across the whole plan each claimed channel is
    produced by exactly one population, so the channel-separated result is
    unambiguous.
  * training leakage: ``assert_no_training_leakage`` fails closed if any
    evaluation-population structure fingerprint also appears in the
    training set -- an evaluation population that leaked into training is
    not a held-out population.

Actual label-completeness ("a DFT channel requires complete DFT labels")
is enforced downstream by ``validation.four_channel_audit.channel(...,
require_complete=True)`` when the orchestrator computes each channel; this
module declares *which* channels are required so that guard is armed.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase


# ---------------------------------------------------------------------
# Generic distillation channels (the only three the framework computes)
# ---------------------------------------------------------------------
STUDENT_VS_TEACHER = "student_vs_teacher"
STUDENT_VS_DFT = "student_vs_dft"
TEACHER_VS_DFT = "teacher_vs_dft"

#: channel -> (reference-label prefix, prediction-label prefix) as consumed
#: by ``validation.four_channel_audit.channel``. The reference prefix names
#: the label a population MUST carry for the channel to be computable.
CHANNEL_REF_PRED: dict[str, tuple[str, str]] = {
    STUDENT_VS_TEACHER: ("teacher", "student"),
    STUDENT_VS_DFT: ("dft", "student"),
    TEACHER_VS_DFT: ("dft", "teacher"),
}

#: the label a channel's *reference* side requires the population to carry.
CHANNEL_REQUIRED_LABEL: dict[str, str] = {
    STUDENT_VS_TEACHER: "teacher",
    STUDENT_VS_DFT: "dft",
    TEACHER_VS_DFT: "dft",
}


class EvaluationPopulationRole(str, Enum):
    """Generic role of a Stage-8 evaluation population.

    Roles are material-agnostic. A run maps concrete files onto them; the
    core only fixes which channels each role is *allowed* to produce.
    """

    #: population carrying Teacher labels only (this run's own Student
    #: distillation holdout). Purpose: Student-vs-Teacher fidelity.
    DISTILLATION_HOLDOUT = "DISTILLATION_HOLDOUT"
    #: population carrying first-principles (DFT) labels, explicitly held
    #: out of Student training. Purpose: Student-vs-DFT + Teacher-vs-DFT.
    DFT_PROTECTED_HOLDOUT = "DFT_PROTECTED_HOLDOUT"


#: role -> the set of channels that role is permitted to produce. This is
#: the structural role<->channel firewall: a DISTILLATION_HOLDOUT can never
#: be the source of a DFT channel, and DFT channels can only come from a
#: DFT_PROTECTED_HOLDOUT.
ROLE_ALLOWED_CHANNELS: dict[EvaluationPopulationRole, frozenset[str]] = {
    EvaluationPopulationRole.DISTILLATION_HOLDOUT: frozenset({STUDENT_VS_TEACHER}),
    EvaluationPopulationRole.DFT_PROTECTED_HOLDOUT: frozenset(
        {STUDENT_VS_DFT, TEACHER_VS_DFT}
    ),
}


class EvaluationPopulation(ContractBase):
    """One role-bound Stage-8 evaluation population.

    ``structures_sha256`` binds the population to an exact on-disk artifact
    so a rebind can never silently swap the frames. ``required_channels``
    lists the channels this population is claimed to produce; each must be
    permitted by ``role`` (enforced below).
    """

    population_id: str
    role: EvaluationPopulationRole
    frames_path: str
    structures_sha256: str
    required_channels: list[str]
    # provenance of the population's label lineage (source manifest SHA,
    # e.g. the split manifest that defined a held-out DFT partition). Kept
    # optional so a purely-distillation holdout without a separate manifest
    # is still expressible; the orchestrator records whatever is bound.
    source_manifest_sha256: str | None = None
    # whether the orchestrator must prove this population is disjoint from
    # the training set before computing any channel. True for every genuine
    # held-out population; only an explicit diagnostic mode would relax it.
    forbid_training_overlap: bool = True

    @model_validator(mode="after")
    def _channels_permitted_by_role(self):
        if not self.required_channels:
            raise ValueError(
                f"EvaluationPopulation {self.population_id!r} declares no "
                "required_channels -- a population with no claimed channel is "
                "not a Stage-8 evaluation population"
            )
        allowed = ROLE_ALLOWED_CHANNELS[self.role]
        illegal = [c for c in self.required_channels if c not in allowed]
        if illegal:
            raise ValueError(
                f"EvaluationPopulation {self.population_id!r} (role {self.role.value}) "
                f"requires channel(s) {illegal} not permitted for that role; "
                f"permitted: {sorted(allowed)}. Role<->channel mismatch fails closed."
            )
        unknown = [c for c in self.required_channels if c not in CHANNEL_REF_PRED]
        if unknown:
            raise ValueError(
                f"EvaluationPopulation {self.population_id!r} requires unknown "
                f"channel(s) {unknown}; known channels: {sorted(CHANNEL_REF_PRED)}"
            )
        return self


class MultiPopulationEvaluationPlan(ContractBase):
    """The Stage-8 claim: an ordered set of role-bound populations whose
    per-channel results together constitute the run's fidelity evidence.

    Enforced: at least one population; unique population ids; unique roles
    (a run binds at most one population per role); and every claimed channel
    is produced by exactly one population (no ambiguous channel-separated
    result)."""

    plan_id: str
    populations: list[EvaluationPopulation]

    @model_validator(mode="after")
    def _well_formed(self):
        if not self.populations:
            raise ValueError("MultiPopulationEvaluationPlan requires >=1 population")
        ids = [p.population_id for p in self.populations]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate population_id in plan: {sorted(ids)}")
        roles = [p.role for p in self.populations]
        if len(set(roles)) != len(roles):
            raise ValueError(
                "a run binds at most one population per role; duplicate role in "
                f"plan: {[r.value for r in roles]}"
            )
        seen: dict[str, str] = {}
        for p in self.populations:
            for c in p.required_channels:
                if c in seen:
                    raise ValueError(
                        f"channel {c!r} is claimed by two populations "
                        f"({seen[c]!r} and {p.population_id!r}); each channel must be "
                        "produced by exactly one population for an unambiguous "
                        "channel-separated result"
                    )
                seen[c] = p.population_id
        return self

    def channel_assignments(self) -> dict[str, str]:
        """channel -> population_id that produces it (unique by construction)."""
        return {c: p.population_id for p in self.populations for c in p.required_channels}

    def population(self, population_id: str) -> EvaluationPopulation | None:
        for p in self.populations:
            if p.population_id == population_id:
                return p
        return None


class EvaluationLeakageError(RuntimeError):
    """Raised when an evaluation population overlaps the training set."""


def assert_no_training_leakage(
    population_fingerprints: Iterable[str],
    training_fingerprints: Iterable[str],
    *,
    population_id: str,
) -> None:
    """Fail closed if any evaluation-population structure fingerprint also
    appears in the training set. Fingerprints are the canonical geometry
    hashes (``validation.protected_reference._structure_fingerprint``) so
    this uses the SAME identity the rest of the framework's protection
    checks use. A population that leaked into training is, by definition,
    not held out."""
    train = set(training_fingerprints)
    overlap = sorted(fp for fp in set(population_fingerprints) if fp in train)
    if overlap:
        raise EvaluationLeakageError(
            f"evaluation population {population_id!r} leaks into the training set: "
            f"{len(overlap)} shared structure fingerprint(s) "
            f"(first: {overlap[0]}). A held-out population must be disjoint from "
            "training; fail closed."
        )


__all__ = [
    "STUDENT_VS_TEACHER",
    "STUDENT_VS_DFT",
    "TEACHER_VS_DFT",
    "CHANNEL_REF_PRED",
    "CHANNEL_REQUIRED_LABEL",
    "EvaluationPopulationRole",
    "ROLE_ALLOWED_CHANNELS",
    "EvaluationPopulation",
    "MultiPopulationEvaluationPlan",
    "EvaluationLeakageError",
    "assert_no_training_leakage",
]
