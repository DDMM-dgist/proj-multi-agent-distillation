"""Framework V2 -- tiered-trust region resolution (binding adjustment #2).

Metadata is useful evidence but NOT unquestioned ground truth. This module
resolves the target region structure under a tiered-trust policy:

  (A) No metadata present
        -> full DISCOVERED path (regions discovered from descriptors).
  (B) Metadata present
        -> a MANDATORY lightweight structural/chemical consistency audit runs
           first. If it PASSES, the declared regions are trusted (DECLARED
           path). If it flags a mismatch / suspicious structure / weak
           provenance / unmapped region, the resolver ESCALATES to the full
           DISCOVERED path (DECLARED_ESCALATED_TO_DISCOVERED).

The escalation logic here is material-agnostic and deterministically
verifiable: the core compares declared metadata against independently-computed
evidence via a caller-supplied ``MetadataAuditor`` that returns a list of
deterministic pass/fail checks. The core owns only the tiering decision; the
material-specific check *content* lives in the plugin, and the *decision* is
reproducible from the recorded checks.

Producing the DomainRepresentation for either path is delegated to
caller-supplied builders (typically wrapping ``domain_discovery.discover_domain``
for the discovered path), so this module stays independent of any particular
descriptor or discovery implementation and is trivially testable with fakes.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from framework_v2.acquisition.contracts import (
    MetadataAuditVerdict,
    MetadataConsistencyAudit,
    MetadataConsistencyCheck,
    RegionResolution,
    RegionResolutionMode,
)
from framework_v2.contracts import DomainRepresentation


@runtime_checkable
class MetadataAuditor(Protocol):
    """Runs the mandatory lightweight consistency audit on declared metadata.

    Must return a list of deterministic checks (each pass/fail with the
    observed/expected values that justify it). Any failing check causes
    escalation. The auditor must not raise on suspicious data -- it records a
    failed check so the decision remains auditable."""
    def audit(self) -> list[MetadataConsistencyCheck]: ...


def _summarize_failures(checks: list[MetadataConsistencyCheck]) -> str:
    failed = [c for c in checks if not c.passed]
    if not failed:
        return ""
    return "; ".join(f"{c.check_id}: {c.description}" for c in failed)


def resolve_regions(
    *,
    resolution_id: str,
    metadata_present: bool,
    discovered_representation_builder: Callable[[], DomainRepresentation],
    auditor: MetadataAuditor | None = None,
    declared_representation_builder: Callable[[], DomainRepresentation] | None = None,
) -> RegionResolution:
    """Resolve the region structure under the tiered-trust policy.

    Deterministic given deterministic builders/auditor. Fails closed (via
    RegionResolution's own validator) if the produced mode is inconsistent
    with the audit verdict."""
    if not metadata_present:
        representation = discovered_representation_builder()
        audit = MetadataConsistencyAudit(
            metadata_present=False,
            audited=False,
            verdict=MetadataAuditVerdict.PASS,
        )
        return RegionResolution(
            resolution_id=resolution_id,
            mode=RegionResolutionMode.DISCOVERED,
            domain_representation_sha256=representation.content_sha256(),
            metadata_audit=audit,
        )

    # Metadata present -> mandatory audit.
    if auditor is None or declared_representation_builder is None:
        raise ValueError(
            "resolve_regions: metadata_present=True requires both an auditor "
            "and a declared_representation_builder (the audit is mandatory)"
        )

    checks = list(auditor.audit())
    any_failed = any(not c.passed for c in checks)

    if any_failed:
        # Escalate: metadata is evidence, not ground truth.
        representation = discovered_representation_builder()
        audit = MetadataConsistencyAudit(
            metadata_present=True,
            audited=True,
            verdict=MetadataAuditVerdict.ESCALATE,
            checks=checks,
            escalation_reason=_summarize_failures(checks),
        )
        return RegionResolution(
            resolution_id=resolution_id,
            mode=RegionResolutionMode.DECLARED_ESCALATED_TO_DISCOVERED,
            domain_representation_sha256=representation.content_sha256(),
            metadata_audit=audit,
        )

    # Audit passed -> trust declared regions.
    representation = declared_representation_builder()
    audit = MetadataConsistencyAudit(
        metadata_present=True,
        audited=True,
        verdict=MetadataAuditVerdict.PASS,
        checks=checks,
    )
    return RegionResolution(
        resolution_id=resolution_id,
        mode=RegionResolutionMode.DECLARED,
        domain_representation_sha256=representation.content_sha256(),
        metadata_audit=audit,
    )
