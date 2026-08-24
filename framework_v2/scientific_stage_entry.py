"""Stage-entry policy-requirement assertion (Blocker 3 close).

For scientific stages, the framework blocks the *start* of execution when the
required scientific policy is absent, in addition to the previously-added
`_enforce_scientific_adequacy` at gate-PASS time. The assertion is
opt-in per-run via ``state["scientific_stage_entry_enforcement"] = True``;
this preserves backward compatibility with C12F (whose manifest does not
carry this flag).

Required policies per stage:
  evaluation           => EvaluationAdequacyPolicyV2
  uncertainty          => UncertaintyPolicyV2
  deployment_md        => DeploymentScopeContractV2 + StatePreparationPolicy
  physical_validation  => PhysicalValidationPolicyV2
"""
from __future__ import annotations


REQUIRED_POLICIES_BY_STAGE = {
    "evaluation": ("EvaluationAdequacyPolicyV2",),
    "uncertainty": ("UncertaintyPolicyV2",),
    "deployment_md": ("DeploymentScopeContractV2", "StatePreparationPolicy"),
    "physical_validation": ("PhysicalValidationPolicyV2",),
}


class ScientificPolicyMissingAtStageEntry(RuntimeError):
    """Raised when a stage that requires a bound scientific policy is
    entered without one."""


def assert_stage_entry_policies_bound(state: dict, stage_name: str) -> None:
    """Check that every required policy kind for ``stage_name`` is bound.

    Opt-in: only enforces when ``state["scientific_stage_entry_enforcement"]``
    is truthy. This preserves C12F backward compatibility while giving new
    runs a strict pre-execution assertion.
    """
    if not state.get("scientific_stage_entry_enforcement"):
        return
    required = REQUIRED_POLICIES_BY_STAGE.get(stage_name)
    if not required:
        return
    policies = state.get("scientific_policies") or {}
    missing = []
    for kind in required:
        key = f"{stage_name}::{kind}"
        if key not in policies:
            missing.append(kind)
        elif not policies[key].get("required", True):
            missing.append(f"{kind} (bound with required=False; upgrade to required=True)")
    if missing:
        raise ScientificPolicyMissingAtStageEntry(
            f"stage {stage_name!r} requires bound scientific policies before start: "
            + ", ".join(missing))
