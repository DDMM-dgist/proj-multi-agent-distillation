"""Root-cause-driven recovery-plan proposal (Section 8 of the closure directive).

Bridges ``framework_v2.scientific_adequacy.RootCauseDiagnosis`` /
``route_by_root_cause`` into a recovery-plan proposal by consulting the
diagnosed root cause rather than the failing stage's number. Cannot
authorize execution on its own; it only produces a typed proposal payload.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from framework_v2.scientific_adequacy import (
    DEFAULT_ROOT_CAUSE_ROUTING, RootCauseClass, RootCauseDiagnosis,
    route_by_root_cause,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def propose_recovery_from_diagnosis(
    diagnosis: RootCauseDiagnosis,
    *,
    failing_stage: str,
    supporting_evidence_refs: Optional[list[str]] = None,
    forbidden_actions: Optional[list[str]] = None,
) -> dict:
    """Return a recovery-plan proposal payload whose return-stage set is
    determined by the diagnosed root cause. The framework's default routing
    map is used unless the diagnosis explicitly overrides
    ``admissible_return_stages``.

    Guarantees:
      * FIDELITY_INADEQUACY may route to training.
      * DEPLOYMENT_STATE_MISMATCH and PHYSICAL_OBSERVABLE_IMPLEMENTATION_DEFECT
        may NOT route to training (enforced here).
      * FRAMEWORK_EVIDENCE_READABILITY_DEFECT may NOT route to scientific
        compute stages.
    """
    return_stages = route_by_root_cause(diagnosis)
    if diagnosis.root_cause in (
        RootCauseClass.DEPLOYMENT_STATE_MISMATCH,
        RootCauseClass.PHYSICAL_OBSERVABLE_IMPLEMENTATION_DEFECT,
    ):
        if "training" in return_stages:
            raise ValueError(
                f"routing defect: {diagnosis.root_cause} must not route to training")
    if diagnosis.root_cause == RootCauseClass.FRAMEWORK_EVIDENCE_READABILITY_DEFECT:
        forbidden = {"training", "acquisition", "teacher_labeling",
                     "deployment_md", "physical_validation"}
        overlap = forbidden.intersection(return_stages)
        if overlap:
            raise ValueError(
                f"routing defect: framework-only recovery cannot route to {overlap}")
    payload = {
        "kind": "recovery_plan_proposal",
        "proposed_at": _now(),
        "failing_stage": failing_stage,
        "diagnosis_id": diagnosis.diagnosis_id,
        "root_cause": diagnosis.root_cause.value,
        "admissible_return_stages": return_stages,
        "supporting_evidence_refs": list(supporting_evidence_refs
                                         or diagnosis.supporting_evidence_refs),
        "forbidden_actions": list(forbidden_actions
                                  or diagnosis.forbidden_recovery_actions),
    }
    payload["proposal_sha256"] = _hash(payload)
    return payload


__all__ = ["propose_recovery_from_diagnosis"]
