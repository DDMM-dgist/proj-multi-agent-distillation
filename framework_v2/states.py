"""Framework V2 — canonical semantic states (generic, material-agnostic).

The closure directive (Section AB) requires that scientifically-different
failures are represented as *distinct* typed states rather than being collapsed
into a generic ``REVISE``. Several of these states already exist, enforced in
their natural home module; this module gives them ONE canonical, importable
vocabulary so producers, judges, the Gate, and recovery routing cannot drift
into different spellings of the same outcome.

Nothing here encodes any material, model family, or campaign. New campaigns and
plugins never add states; the set of scientifically-meaningful outcomes is a
framework invariant. (Failure *routing* codes, by contrast, are extensible via
``workflow.recovery_taxonomy.register_failure_code``.)

Authoritative enforcement homes (unchanged by this module):

  * PASS / REVISE / FAIL      -- ``workflow.controller.RunController.record_gate``
  * NOT_APPLICABLE            -- ``RunController.mark_stage_not_applicable``
  * FRAMEWORK_CAPABILITY_BLOCKER -- ``framework_v2.capability``
  * NOT_CONVERGED             -- ``framework_v2.convergence``
  * REVISE_SPLIT / LINEAGE_LEAKAGE -- ``framework_v2.partition_validator``
  * BLIND_TEST_ACCESS_VIOLATION -- ``framework_v2.blind_test``
  * INVALID_JUDGE_OUTPUT / JUDGE_INVALID_BLOCKER -- ``orchestration.exchange`` /
    the Gate (a judge output that fails deterministic validation is never a vote)
  * REPRESENTATION_INSUFFICIENT -- representation-adequacy decision
    (``framework_v2.representation_adequacy``)
  * EVIDENCE_INSUFFICIENT     -- canonical-review-packet compiler / Gate
  * CAMPAIGN_BLOCKED          -- Controller, when bounded recovery is exhausted
"""
from __future__ import annotations

from enum import Enum


class GateVerdict(str, Enum):
    """The three verdicts a canonical Gate may record for a stage decision.

    Every scientifically consequential stage advances only by recording one of
    these through the single authoritative gate path. A Gate PASSes only when
    all three mutually-blind review lenses return PASS *and* every deterministic
    precondition holds (see :class:`SemanticState` for the non-PASS reasons a
    precondition can raise).
    """
    PASS = "PASS"
    REVISE = "REVISE"
    FAIL = "FAIL"


class SemanticState(str, Enum):
    """The full set of scientifically-distinct typed outcomes (Section AB).

    These are deliberately NOT all gate verdicts. A gate records only
    :class:`GateVerdict`; the remaining members are precondition/quality states
    that either block a gate from being reached (e.g. ``EVIDENCE_INSUFFICIENT``,
    ``INVALID_JUDGE_OUTPUT``) or are specialised recovery-bearing verdicts a
    validator raises before the gate aggregates (e.g. ``REVISE_SPLIT``,
    ``NOT_CONVERGED``, ``REPRESENTATION_INSUFFICIENT``).
    """
    # gate verdicts
    PASS = "PASS"
    REVISE = "REVISE"
    FAIL = "FAIL"
    # applicability
    NOT_APPLICABLE = "NOT_APPLICABLE"
    # evidence / review quality (block the gate; never silently a REVISE)
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    INVALID_JUDGE_OUTPUT = "INVALID_JUDGE_OUTPUT"
    JUDGE_INVALID_BLOCKER = "JUDGE_INVALID_BLOCKER"
    # specialised scientific verdicts (carry a specific recovery route)
    REPRESENTATION_INSUFFICIENT = "REPRESENTATION_INSUFFICIENT"
    REVISE_SPLIT = "REVISE_SPLIT"
    NOT_CONVERGED = "NOT_CONVERGED"
    # hard structural blockers
    FRAMEWORK_CAPABILITY_BLOCKER = "FRAMEWORK_CAPABILITY_BLOCKER"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    BLIND_TEST_ACCESS_VIOLATION = "BLIND_TEST_ACCESS_VIOLATION"
    LINEAGE_LEAKAGE = "LINEAGE_LEAKAGE"
    # terminal
    CAMPAIGN_BLOCKED = "CAMPAIGN_BLOCKED"


# States that must NEVER be silently downgraded to a plain REVISE. If a gate or
# recovery layer encounters one of these it must preserve the specific semantics
# (Section AB: "Do not collapse scientifically different failures into generic
# REVISE").
NON_COLLAPSIBLE_STATES: frozenset[SemanticState] = frozenset({
    SemanticState.EVIDENCE_INSUFFICIENT,
    SemanticState.INVALID_JUDGE_OUTPUT,
    SemanticState.JUDGE_INVALID_BLOCKER,
    SemanticState.REPRESENTATION_INSUFFICIENT,
    SemanticState.REVISE_SPLIT,
    SemanticState.NOT_CONVERGED,
    SemanticState.FRAMEWORK_CAPABILITY_BLOCKER,
    SemanticState.CONTRACT_VIOLATION,
    SemanticState.BLIND_TEST_ACCESS_VIOLATION,
    SemanticState.LINEAGE_LEAKAGE,
    SemanticState.CAMPAIGN_BLOCKED,
})

# States that block a gate from being *reached* (evidence/review quality). They
# are resolved by producing/repairing evidence or re-running the review, not by
# scientific redesign of the decision.
GATE_BLOCKING_STATES: frozenset[SemanticState] = frozenset({
    SemanticState.EVIDENCE_INSUFFICIENT,
    SemanticState.INVALID_JUDGE_OUTPUT,
    SemanticState.JUDGE_INVALID_BLOCKER,
})

# Specialised REVISE-class verdicts that DO map to a Controller-native recovery
# (each carries a specific responsible stage/producer via the recovery mapping
# in the relevant StageReviewSpec).
RECOVERY_BEARING_STATES: frozenset[SemanticState] = frozenset({
    SemanticState.REVISE,
    SemanticState.FAIL,
    SemanticState.REPRESENTATION_INSUFFICIENT,
    SemanticState.REVISE_SPLIT,
    SemanticState.NOT_CONVERGED,
})


def is_gate_verdict(state: str) -> bool:
    return state in (GateVerdict.PASS, GateVerdict.REVISE, GateVerdict.FAIL)


# --- Recovery-routing codes for the new semantic states --------------------
# These register the closure directive's new outcomes into the SINGLE shared
# failure-code registry so recovery routing has an explicit, fail-closed home
# for each. Registration is idempotent (safe to import more than once).
def _register_states_into_taxonomy() -> None:
    from workflow.recovery_taxonomy import register_failure_code

    for code, domain, desc in (
        ("representation_insufficient", "data_coverage",
         "the selected configurational representation is not shown adequate for "
         "the deployment claim (good internal coverage of an unjustified "
         "representation does not prove adequacy)"),
        ("split_unrepresentative", "data_coverage",
         "the dataset split is lineage-safe but does not represent the "
         "deployment manifold (REVISE_SPLIT)"),
        ("evidence_insufficient", "insufficient_evidence",
         "required evidence for the decision/review is absent; recover by "
         "generating or gathering the missing evidence, not by re-judging"),
        ("teacher_distribution_coverage", "teacher_support",
         "Student distillation data coverage of the Teacher training "
         "distribution is unassessed or inadequate"),
        ("replay_strategy_unjustified", "data_coverage",
         "the replay / data-mixing strategy was chosen without comparative "
         "evidence over meaningful alternatives"),
    ):
        register_failure_code(code, domain, desc)


_register_states_into_taxonomy()
