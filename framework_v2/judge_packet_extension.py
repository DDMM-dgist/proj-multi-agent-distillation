"""Judge-packet scientific-content helper (Blocker 2 partial close).

This module exposes a helper function that any Judge task-packet builder can
call to attach the scientific-adequacy layer to a Judge task:

  build_scientific_extension_block(state, stage_name, observed_evidence_values)

The returned dict is intended to be MERGED into a Judge task packet just before
dispatch. Guarantees at merge time:

  * generic ScientificQuestion (no material-specific numbers in question text);
  * policy type + policy_id + content_sha256 (frozen);
  * preregistration_witness_ref (if the policy is EvaluationAdequacyPolicyV2);
  * in-scope domains / representative-point ref (looked up on the stage's
    bound policies);
  * deterministic scientific adequacy verdict (computed via scientific_gate);
  * criterion-by-criterion observed values (from the caller-supplied
    ``observed_evidence_values`` dict);
  * PASS/FAIL/NOT_EVALUABLE state per policy;
  * NOT_EVALUABLE reasons;
  * evidence artifact references;
  * an explicit PROCEDURAL_VALIDITY_vs_SCIENTIFIC_ADEQUACY divider so Judges
    cannot conflate the two axes.

The three-mutually-blind-Judges architecture is preserved by having the
packet builder call this helper independently per Judge lens; identical policy
content yields identical hash content across Judges. This is a HELPER, not a
rewriter — existing packet builders retain their identity until they opt in.
"""
from __future__ import annotations

from typing import Any, Optional


def build_scientific_extension_block(
    state: dict, stage_name: str,
    *,
    observed_evidence_values: Optional[dict[str, Any]] = None,
    adequacy_verdict_summary: Optional[dict] = None,
) -> dict:
    """Return the scientific-adequacy extension block to merge into a Judge
    packet.

    Empty (i.e. `{"scientific_layer_active": False}`) if no policy is bound
    for the given stage — this makes the helper safe to call unconditionally
    from any packet builder.
    """
    from framework_v2.scientific_adequacy import DEFAULT_SCIENTIFIC_QUESTIONS
    policies = state.get("scientific_policies") or {}
    stage_policies = {k: v for k, v in policies.items()
                      if k.startswith(f"{stage_name}::")}
    if not stage_policies:
        return {"scientific_layer_active": False, "stage": stage_name}
    question = DEFAULT_SCIENTIFIC_QUESTIONS.get(stage_name)
    block = {
        "scientific_layer_active": True,
        "stage": stage_name,
        "layer_separation_note": (
            "PROCEDURAL_VALIDITY (did the correct computation on the right "
            "population with valid provenance?) is DISTINCT from "
            "SCIENTIFIC_ADEQUACY (does the achieved metric satisfy the "
            "pre-registered adequacy policy?). A Judge must not treat a "
            "procedural PASS as evidence of scientific adequacy."),
        "scientific_question": (question.model_dump() if question is not None
                                else {"question_id": f"scientific::{stage_name}",
                                       "stage": stage_name,
                                       "question_text": "Does the evidence "
                                       "satisfy the pre-registered scientific policy?"}),
        "bound_policies": [
            {"key": key,
             "kind": rec["kind"],
             "policy_id": rec["content"].get("policy_id"),
             "content_sha256": rec["content_sha256"],
             "required": rec.get("required", True),
             "bound_at": rec["bound_at"],
             "source_ref": rec["source_ref"],
             "preregistration_witness_ref": rec["content"].get("preregistration_witness_ref"),
             "primary_domains_or_representative_point": (
                 rec["content"].get("primary_domains")
                 or rec["content"].get("representative_point_ref")
                 or rec["content"].get("state_role")),
             "criteria_summary": _summarize_criteria(rec["content"]),
             }
            for key, rec in stage_policies.items()
        ],
        "observed_evidence_values": observed_evidence_values or {},
        "adequacy_verdict_summary": adequacy_verdict_summary,
        "evidence_references": [],   # populated by caller when known
    }
    return block


def _summarize_criteria(policy_content: dict) -> list[dict]:
    """Extract a compact criteria list from the bound policy dict; empty
    when the policy does not carry per-criterion numerical rules
    (e.g. UncertaintyPolicyV2 uses required_status instead)."""
    out = []
    for key in ("per_domain_criteria", "worst_domain_criteria",
                "aggregate_criteria", "relative_to_reference_criteria",
                "outlier_tail_criteria"):
        for c in policy_content.get(key) or []:
            out.append({
                "criterion_id": c.get("criterion_id"),
                "observable": c.get("observable"),
                "operator": c.get("operator"),
                "value": c.get("value"),
                "unit": c.get("unit"),
                "source_class": c.get("source_class"),
                "source_reference": c.get("source_reference"),
                "frozen_before_evaluation": c.get("frozen_before_evaluation"),
            })
    # Uncertainty policy required_status
    if policy_content.get("required_status"):
        out.append({
            "criterion_id": "required_calibration_status",
            "operator": "equals_or_stronger",
            "value": policy_content["required_status"],
        })
    return out


__all__ = ["build_scientific_extension_block"]
