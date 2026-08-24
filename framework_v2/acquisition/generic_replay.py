"""Framework V2 -- ReplayDataMixingPlan (FE-027 P3, §7).

Whether (and how much) prior-round or otherwise-existing data is mixed into a new acquisition/
training round is a genuine scientific decision -- so it is an Agent PROPOSAL (control plane),
never a value the core inherits silently from a past run. The deterministic gate here (evidence
plane) checks only the mechanical invariants; the 3-Judge committee reviews the scientific merit.

Deterministic invariants enforced:
  * NO LEAKAGE -- replayed frame ids are disjoint from the protected-reference set and the blind
    test set (a replay plan can never smuggle a held-out structure into training);
  * PROVENANCE -- every replay source is drawn from an explicitly allowed source set (no dangling
    or unaccounted provenance);
  * COUNT INTEGRITY -- each source's declared ``n_frames`` equals its enumerated frame ids, and no
    frame id is replayed twice;
  * EXPLICIT DECLARATION -- the plan must be declared for THIS run (``inherited`` False); a plan
    silently carried over from a previous run is rejected, per the "don't inherit past-run replay"
    requirement.

The validator returns an issue list (empty iff admissible), matching the bounded-retry contract of
the other acquisition validators.
"""
from __future__ import annotations

from framework_v2.contracts import ContractBase
from pydantic import Field


class ReplaySourceRef(ContractBase):
    """One source of replay/existing data the plan proposes to mix in.

    ``frame_ids`` enumerates exactly which frames are replayed (so leakage/provenance are
    checkable); ``n_frames`` must equal ``len(frame_ids)``. ``provenance_class`` is a freeform
    provenance label (e.g. 'prior_round_labeled', 'sanitized_pool')."""
    source_ref: str
    n_frames: int
    provenance_class: str
    frame_ids: list[str] = Field(default_factory=list)


class ReplayDataMixingPlan(ContractBase):
    """The control-plane artifact: an Agent's proposal for mixing existing data into a round.

    ``inherited`` False asserts the plan was authored FOR this run; a plan carried over from a
    previous run (``inherited`` True) is rejected by the validator. Bound by
    ``target_regime_model_sha256`` to the regime model the mix is designed against."""
    plan_id: str
    target_regime_model_sha256: str
    replay_sources: list[ReplaySourceRef] = Field(default_factory=list)
    inherited: bool = False
    rationale: str = ""

    def all_frame_ids(self) -> list[str]:
        return [fid for s in self.replay_sources for fid in s.frame_ids]


def validate_replay_mixing_plan(
    plan: ReplayDataMixingPlan, *,
    protected_ids: set[str],
    blind_test_ids: set[str],
    allowed_source_refs: set[str],
) -> list[str]:
    """Deterministic leakage/provenance/integrity gate. Returns issues; empty iff admissible."""
    issues: list[str] = []

    if plan.inherited:
        issues.append(
            "replay mixing plan must be explicitly declared for this run, not inherited from a "
            "previous run")

    seen: set[str] = set()
    for src in plan.replay_sources:
        if src.source_ref not in allowed_source_refs:
            issues.append(f"replay source has unaccounted provenance: {src.source_ref}")
        if src.n_frames != len(src.frame_ids):
            issues.append(
                f"replay source {src.source_ref}: n_frames ({src.n_frames}) != enumerated frame "
                f"ids ({len(src.frame_ids)})")
        for fid in src.frame_ids:
            if fid in seen:
                issues.append(f"frame replayed more than once: {fid}")
            seen.add(fid)

    leaked_protected = seen & protected_ids
    if leaked_protected:
        issues.append(
            f"replay plan leaks protected-reference frames into training: "
            f"{sorted(leaked_protected)}")
    leaked_blind = seen & blind_test_ids
    if leaked_blind:
        issues.append(
            f"replay plan leaks blind-test frames into training: {sorted(leaked_blind)}")

    return issues


__all__ = [
    "ReplaySourceRef",
    "ReplayDataMixingPlan",
    "validate_replay_mixing_plan",
]
