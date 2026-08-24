"""Framework V2 — Judge reproducibility auditing at two levels (Section AG).

A Judge verdict is only scientifically trustworthy if the attempt that produced
it can be reproduced and if every mutually-blind lens provably reasoned over the
same evidence. The closure directive names two reproducibility levels:

  * **L1 — attempt-level reproducibility.** One Judge attempt is L1-reproducible
    iff its provenance pins everything needed to re-run it identically: the SHA
    of the CanonicalReviewPacket it reviewed, the SHA of the decision under
    review, and the sampling settings actually used (``temperature`` and
    ``seed``). If any of these is absent, the attempt is a black box and cannot
    be reproduced — that is a reproducibility failure, never silently ignored.

  * **L2 — committee-level blind-identity reproducibility.** The three
    mutually-blind lenses for one stage decision are L2-reproducible iff (a) each
    lens attempt is itself L1-reproducible and (b) all three are bound to the
    *identical* ``packet_sha256`` and ``decision_sha256`` — the Section H
    invariant that every lens saw the same bytes. Three lenses that each look
    reproducible but reasoned over different packets are NOT a valid committee.

This module is generic and material-agnostic: it reads only the provenance
fields (present on :class:`runtimes.pydantic_ai.models.RuntimeInvocationRecord`,
and readable from a plain mapping) and never inspects any scientific content. It
does not judge whether a verdict is *correct* — only whether the attempt is
reproducible and the committee is bytes-identical.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import Field

from framework_v2.contracts import ContractBase

# The four provenance fields an L1-reproducible attempt must pin.
L1_REQUIRED_FIELDS: tuple[str, ...] = (
    "packet_sha256",
    "decision_sha256",
    "temperature",
    "seed",
)


def _get(record: Any, field: str) -> Any:
    """Read a field from either a pydantic record or a plain mapping."""
    if isinstance(record, Mapping):
        return record.get(field)
    return getattr(record, field, None)


class L1ReproducibilityResult(ContractBase):
    """Whether a single Judge attempt can be reproduced from its provenance."""
    reproducible: bool
    missing_fields: list[str] = Field(default_factory=list)
    packet_sha256: Optional[str] = None
    decision_sha256: Optional[str] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None


class L2ReproducibilityResult(ContractBase):
    """Whether a three-lens committee is reproducible AND bytes-identical."""
    reproducible: bool
    lens_ids: list[str] = Field(default_factory=list)
    shared_packet_sha256: Optional[str] = None
    shared_decision_sha256: Optional[str] = None
    per_lens_l1: dict[str, bool] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


def verify_l1(record: Any) -> L1ReproducibilityResult:
    """Audit one Judge attempt for L1 (attempt-level) reproducibility.

    Reproducible iff every field in :data:`L1_REQUIRED_FIELDS` is present and
    non-empty (``None`` fails; the empty string fails for the SHA fields)."""
    missing: list[str] = []
    for field in L1_REQUIRED_FIELDS:
        value = _get(record, field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return L1ReproducibilityResult(
        reproducible=not missing,
        missing_fields=missing,
        packet_sha256=_get(record, "packet_sha256"),
        decision_sha256=_get(record, "decision_sha256"),
        temperature=_get(record, "temperature"),
        seed=_get(record, "seed"),
    )


def verify_l2(
    records_by_lens: Mapping[str, Any],
    *,
    expected_lens_ids: Optional[tuple[str, ...]] = None,
) -> L2ReproducibilityResult:
    """Audit a three-lens committee for L2 reproducibility.

    ``records_by_lens`` maps each lens id to that lens's Judge attempt
    provenance. Reproducible iff there are exactly three lenses (matching
    ``expected_lens_ids`` when supplied), each attempt is L1-reproducible, and
    all three share the identical (non-empty) ``packet_sha256`` and
    ``decision_sha256``.
    """
    errors: list[str] = []
    lens_ids = sorted(records_by_lens)

    if len(records_by_lens) != 3:
        errors.append(
            f"a committee must have exactly three mutually-blind lenses, got "
            f"{len(records_by_lens)}: {lens_ids}"
        )
    if expected_lens_ids is not None and set(records_by_lens) != set(expected_lens_ids):
        errors.append(
            f"committee lenses {lens_ids} do not match expected "
            f"{sorted(expected_lens_ids)}"
        )

    per_lens_l1: dict[str, bool] = {}
    for lens, record in records_by_lens.items():
        l1 = verify_l1(record)
        per_lens_l1[lens] = l1.reproducible
        if not l1.reproducible:
            errors.append(
                f"lens {lens!r} attempt is not L1-reproducible (missing "
                f"{l1.missing_fields})"
            )

    packet_shas = {_get(r, "packet_sha256") for r in records_by_lens.values()}
    decision_shas = {_get(r, "decision_sha256") for r in records_by_lens.values()}

    shared_packet = next(iter(packet_shas)) if len(packet_shas) == 1 else None
    shared_decision = next(iter(decision_shas)) if len(decision_shas) == 1 else None

    if len(packet_shas) != 1:
        errors.append(
            "lenses were bound to different packet_sha256 values (they did not "
            "all reason over the same CanonicalReviewPacket bytes)"
        )
    elif not shared_packet:
        errors.append("committee packet_sha256 is absent")
        shared_packet = None

    if len(decision_shas) != 1:
        errors.append("lenses were bound to different decision_sha256 values")
    elif not shared_decision:
        errors.append("committee decision_sha256 is absent")
        shared_decision = None

    return L2ReproducibilityResult(
        reproducible=not errors,
        lens_ids=lens_ids,
        shared_packet_sha256=shared_packet,
        shared_decision_sha256=shared_decision,
        per_lens_l1=per_lens_l1,
        errors=errors,
    )


__all__ = [
    "L1_REQUIRED_FIELDS",
    "L1ReproducibilityResult",
    "L2ReproducibilityResult",
    "verify_l1",
    "verify_l2",
]
