"""Framework-level failure-domain and failure-code registry.

This module is the single shared vocabulary contract between diagnosis
(the Analyst's ``runtimes.pydantic_ai.root_cause.RootCauseClassification``) and recovery
(``workflow.controller``'s ``RECOVERY_CATEGORIES``/``propose_recovery``): both consume this
registry rather than declaring their own independent category sets, so the two vocabularies
cannot silently diverge into different spellings of the same failure.

A ``FailureDomain`` is a small, fixed set of framework-level buckets a corrective action routes
on (which capability is plausibly responsible, which stages are plausible return points). A
``failure_code`` is a free-form but REGISTERED string within exactly one domain --
``resolve_failure_code`` fails closed on any unregistered code, so a caller can never invent an
ad-hoc string that silently bypasses the shared taxonomy. New campaigns register additional
codes at import time via ``register_failure_code``; nothing here encodes any one campaign's
chemistry, model family, or dataset.

The legacy codes pre-registered at the bottom of this module are exactly the union of the two
vocabularies this registry reconciles: ``workflow.controller``'s original ``RECOVERY_CATEGORIES``
(8 values) and ``runtimes.pydantic_ai.root_cause``'s original ``FailureCategory`` (11 values).
Genuinely identical concepts (``student_fidelity``, ``teacher_applicability``) are registered
once and shared; divergent-but-related concepts (e.g. ``data_coverage`` vs. ``dataset_coverage``)
are kept as distinct registered codes under a shared domain rather than silently merged, so no
historical plan or classification changes meaning -- only where it participates in downstream
routing (via ``domain_of``) is now shared and explicit.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal, Optional

FailureDomain = Literal[
    "data_coverage", "model_fitting", "simulation_protocol", "teacher_support",
    "insufficient_evidence", "operational",
]

FAILURE_DOMAINS: tuple[str, ...] = (
    "data_coverage", "model_fitting", "simulation_protocol", "teacher_support",
    "insufficient_evidence", "operational",
)


@dataclass(frozen=True)
class FailureCode:
    code: str
    domain: str
    description: str
    legacy: bool = False


_REGISTRY: dict[str, FailureCode] = {}


def register_failure_code(code: str, domain: str, description: str, *,
                          legacy: bool = False) -> FailureCode:
    """Register (or idempotently re-register identically) one failure_code within a domain.

    Fails closed: a domain outside ``FAILURE_DOMAINS``, an empty code, or a redefinition of an
    already-registered code with different domain/description is a hard error. This registry is
    the only place the diagnosis/recovery vocabulary is allowed to grow, and only by addition.
    """
    if domain not in FAILURE_DOMAINS:
        raise ValueError(f"unregistered failure domain: {domain!r}")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("failure_code must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("failure_code requires a non-empty description")
    entry = FailureCode(code=code, domain=domain, description=description, legacy=legacy)
    existing = _REGISTRY.get(code)
    if existing is not None and existing != entry:
        raise ValueError(
            f"failure_code {code!r} is already registered as {existing!r}; "
            f"cannot redefine it as {entry!r}"
        )
    _REGISTRY[code] = entry
    return entry


def resolve_failure_code(code: str) -> FailureCode:
    """Fail closed on any failure_code that was never registered."""
    try:
        return _REGISTRY[code]
    except KeyError as exc:
        raise KeyError(f"unregistered failure_code: {code!r}") from exc


def domain_of(code: str) -> str:
    return resolve_failure_code(code).domain


def registered_codes(domain: Optional[str] = None) -> tuple[str, ...]:
    if domain is not None and domain not in FAILURE_DOMAINS:
        raise ValueError(f"unregistered failure domain: {domain!r}")
    return tuple(sorted(
        c for c, entry in _REGISTRY.items() if domain is None or entry.domain == domain
    ))


_failure_category_enum_cache: Optional[type] = None


def failure_category_enum() -> type:
    """Build (once, cached) a ``str``-mixin ``Enum`` whose members are exactly the currently
    registered failure codes -- the single schema-visible source of truth for a Pydantic
    ``failure_category`` field.

    This replaces a hidden-constraint pattern (a plain ``str`` field plus a custom validator that
    secretly requires a registered code, invisible to a provider's structured-output/JSON-schema
    enforcement) with a representation the generated Pydantic JSON Schema exposes literally as
    ``"enum": [...]`` -- so a provider enforcing strict structured output can itself constrain the
    model to a registered code, not just have it rejected after the fact.

    Cached at first call: a member is added only for codes registered by then. A campaign that
    calls ``register_failure_code`` must do so before any module builds this enum (i.e. before
    importing ``runtimes.pydantic_ai.root_cause``), exactly as new legacy codes are registered at
    THIS module's own import time, above.
    """
    global _failure_category_enum_cache
    if _failure_category_enum_cache is None:
        codes = registered_codes()
        _failure_category_enum_cache = enum.Enum(
            "FailureCategory", {code: code for code in codes}, type=str)
    return _failure_category_enum_cache


# --- Legacy reconciliation ------------------------------------------------------------------
# Every value that used to be independently accepted by workflow.controller.RECOVERY_CATEGORIES
# or runtimes.pydantic_ai.root_cause.FailureCategory is registered here exactly once, tagged
# legacy=True, with an explicit domain. Both modules now derive their accepted-code sets FROM
# this registry.
_LEGACY_CODES = (
    ("data_quality", "data_coverage",
     "controller.py legacy: general data-quality defect"),
    ("dataset_coverage", "data_coverage",
     "controller.py legacy: dataset does not cover the deployment target"),
    ("data_coverage", "data_coverage",
     "root_cause.py legacy: candidate/Student population not covered by the reference"),
    ("lineage_or_leakage", "data_coverage",
     "root_cause.py legacy: provenance/lineage or train-test leakage defect"),
    ("student_fidelity", "model_fitting",
     "shared legacy: Student model does not fit the Teacher/target well enough"),
    ("training_instability", "model_fitting",
     "root_cause.py legacy: the Student training run itself was unstable"),
    ("simulation_protocol", "simulation_protocol",
     "controller.py legacy: simulation/MD protocol defect"),
    ("simulation_instability", "simulation_protocol",
     "root_cause.py legacy: a simulation run was numerically unstable"),
    ("structural_invalidity", "simulation_protocol",
     "root_cause.py legacy: generated/sampled structures are invalid"),
    ("physical_validation", "simulation_protocol",
     "controller.py legacy: physical-observable validation gate failure"),
    ("teacher_applicability", "teacher_support",
     "shared legacy: the Teacher is not applicable to the query environment"),
    ("reference_disagreement", "teacher_support",
     "root_cause.py legacy: reference/Teacher disagreement on labels"),
    ("evidence_gap", "insufficient_evidence",
     "controller.py legacy: recovery needs additional evidence, not a corrective fix"),
    ("missing_evidence", "insufficient_evidence",
     "root_cause.py legacy: the classification itself lacked supporting evidence"),
    ("unknown", "insufficient_evidence",
     "root_cause.py legacy: root cause could not be determined"),
    ("operational_failure", "operational",
     "root_cause.py legacy: infrastructure/runtime failure, not a scientific one"),
    ("other", "operational",
     "controller.py legacy: uncategorized recovery reason"),
)
for _code, _domain, _description in _LEGACY_CODES:
    register_failure_code(_code, _domain, _description, legacy=True)
del _code, _domain, _description
