"""Typed scheduler interface (Phase 6/1). Production interface WITHOUT a real Slurm/PBS backend.

Submit/query/collect are modeled as typed contracts + a controller lifecycle
(pending -> collect -> resume). No real HPC backend is attached: the status stays
READY_INTERFACE_BACKEND_NOT_CONFIGURED and nothing is ever reported as EXECUTED/COMPLETED without
a real backend. A SandboxSchedulerAdapter drives the lifecycle in network-free tests using the
SAME typed contracts and controller path a production adapter would use; it explicitly marks its
backend as ``sandbox`` and requires a test to simulate external completion (it never runs a job).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .models import EvidenceReference, NonEmptyStr

BACKEND_NOT_CONFIGURED = "BACKEND_NOT_CONFIGURED"


class SchedulerSubmissionProposal(BaseModel):
    model_config = {"extra": "forbid"}
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    stage: NonEmptyStr
    protocol_ref: NonEmptyStr
    protocol_hash: NonEmptyStr
    config_ref: NonEmptyStr
    config_hash: NonEmptyStr
    input_artifacts: list[EvidenceReference] = Field(default_factory=list)
    input_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    requested_resources: dict[str, str] = Field(default_factory=dict)
    approval_boundary: Literal["scheduler_submission"] = "scheduler_submission"
    idempotency_key: NonEmptyStr


class SchedulerJobIdentity(BaseModel):
    model_config = {"extra": "forbid"}
    backend: NonEmptyStr                 # e.g. "sandbox"; a real adapter would be "slurm"/"pbs"
    external_job_id: NonEmptyStr
    run_id: NonEmptyStr
    stage: NonEmptyStr
    idempotency_key: NonEmptyStr
    submitted_at: str = ""


class SchedulerStatusRecord(BaseModel):
    model_config = {"extra": "forbid"}
    external_job_id: NonEmptyStr
    state: Literal["PENDING", "RUNNING", "COLLECTED", "FAILED", "BACKEND_NOT_CONFIGURED"]
    backend: NonEmptyStr
    detail: str = ""


class SchedulerCollectionRequest(BaseModel):
    model_config = {"extra": "forbid"}
    run_id: NonEmptyStr
    external_job_id: NonEmptyStr


class SchedulerError(Exception):
    pass


class SandboxSchedulerAdapter:
    """Sandbox adapter: drives the typed lifecycle without a real backend. It NEVER runs a job
    and NEVER fabricates completion — a test must call ``simulate_external_completion`` to model
    the external scheduler finishing, mirroring how a real adapter would observe job state."""
    backend = "sandbox"

    def __init__(self, controller, approvals):
        self._c = controller
        self._approvals = approvals
        self._completions: dict = {}  # external_job_id -> (artifact_ref, sha256)

    def submit(self, proposal: SchedulerSubmissionProposal) -> SchedulerJobIdentity:
        # (1) approval, (2) idempotency — same checks the dispatcher enforces.
        if not self._approvals.has_approval(proposal.run_id, proposal.approval_boundary,
                                            proposal.idempotency_key):
            raise SchedulerError("APPROVAL_REQUIRED: scheduler_submission not approved")
        if self._c.action_seen(proposal.idempotency_key):
            raise SchedulerError("DUPLICATE: idempotency_key already submitted")
        external_job_id = f"sandbox-{proposal.idempotency_key}"
        identity = SchedulerJobIdentity(
            backend=self.backend, external_job_id=external_job_id, run_id=proposal.run_id,
            stage=proposal.stage, idempotency_key=proposal.idempotency_key)
        self._c.record_scheduler_submission({
            "external_job_id": external_job_id, "backend": self.backend,
            "idempotency_key": proposal.idempotency_key, "run_id": proposal.run_id,
            "stage": proposal.stage, "protocol_hash": proposal.protocol_hash,
            "config_hash": proposal.config_hash})
        self._c.record_action(proposal.idempotency_key, action_type="submit_scheduler_job",
                              status="PENDING")
        return identity

    def query(self, run_id: str, external_job_id: str) -> SchedulerStatusRecord:
        job = self._c.get_scheduler_job(external_job_id)
        if job is None or job.get("run_id") != run_id:
            raise SchedulerError("unknown or mismatched run/job identity")
        # No real backend: a job is PENDING until a test simulates external completion.
        state = job.get("state", "PENDING")
        if state == "PENDING" and external_job_id in self._completions:
            state = "PENDING"  # completion is observable only via collect, keeping lifecycle honest
        return SchedulerStatusRecord(external_job_id=external_job_id, state=state,
                                     backend=self.backend,
                                     detail="no HPC backend configured; sandbox lifecycle")

    def simulate_external_completion(self, external_job_id: str, artifact_ref: str, sha256: str):
        """TEST-ONLY: model the external scheduler finishing. A production adapter learns this
        from the real scheduler; there is no such backend here."""
        self._completions[external_job_id] = (artifact_ref, sha256)

    def collect(self, request: SchedulerCollectionRequest):
        job = self._c.get_scheduler_job(request.external_job_id)
        if job is None or job.get("run_id") != request.run_id:
            raise SchedulerError("unknown or mismatched run/job identity")
        if request.external_job_id not in self._completions:
            # collect cannot fabricate a completion artifact before the job is done.
            raise SchedulerError("job not complete: no collectable artifact yet")
        artifact_ref, sha256 = self._completions[request.external_job_id]
        return self._c.record_scheduler_collection(request.external_job_id,
                                                   artifact_ref=artifact_ref,
                                                   artifact_sha256=sha256)
