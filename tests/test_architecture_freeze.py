"""ARCHITECTURE FREEZE guard — v2 (deterministic-verdict-ownership refactor).

The v1 freeze (development-pass revision 87c51d3) was consumed by the Stage D-1 holdout, which
exposed an LLM_VERDICT_REGENERATION_CONSISTENCY_FAILURE. The architecture was then deliberately
revised (the deterministic policy now OWNS the authoritative verdict; the LLM owns only
interpretation) and RE-FROZEN here at the post-refactor revision. This test pins the SHA-256 of
every frozen runtime/semantic file so any later edit to one — the deterministic criterion evaluator
+ result schema + authoritative/advisory + verdict-ownership binding, the Judge prompt, canonical
validation/acceptance, the duplicate-read guard, request_limit, authorization policy, controller
logic, and role schemas — fails the suite. The NEXT holdout package may only ADD
fixtures/specs/runner/evaluator-glue/tests; it may not change anything below. Frozen model is a
constant, not a file.

If a frozen file legitimately must change, that is (by definition) a NEW architecture revision and a
NEW freeze — it must not be done silently while a holdout is in flight.

v10 (validation-target-lock) deliberately re-froze workflow/controller.py: it is a real, disclosed
architecture change (see FREEZE_REVISION below for exactly what changed and, just as importantly,
what did NOT), not a workaround to avoid touching frozen code.

v11 (validation-target-lock, automatic establishment) closed a real wiring gap a runtime-route
audit found in v10: contract establishment was still an optional, manual CLI step, so nothing
actually stopped a real run from executing Teacher-side scientific stages before the target
contract existed. v11 deliberately re-froze workflow/controller.py again for the same reason as
v10 — see FREEZE_REVISION below for exactly what changed and what deliberately did not.

v12 (recovery-taxonomy-and-capability-unification) COMPLETES the existing recovery state
machine (Gate failure -> pending recovery -> recovery proposal/approval -> iteration ->
invalidation/re-entry -> verified execution -> revalidation -> same-stage Gate -> recovery
resolution) rather than replacing it — see FREEZE_REVISION below for exactly what changed and
what deliberately did not.

v13 (post-init lifecycle completion), v14 (generic typed reasoning-output acceptance path), and
v15 (recovery-plan reasoning output) deliberately re-froze pydantic_ai_runtime.py/role_outputs.py/
production_router.py (v14), role_outputs.py again (v15), and workflow/controller.py (v13) for the
same reason as v10-v12 — see FREEZE_REVISION below for exactly what changed and what deliberately
did not.

v16 (campaign observability progress_cb threading) deliberately re-froze production_router.py and
controller_bridge.py again for the same reason as v10-v15 — see FREEZE_REVISION below for exactly
what changed and what deliberately did not.

v17 (R16 forensic-defect corrections) deliberately re-froze models.py and pydantic_ai_runtime.py
again for the same reason as v10-v16 — see FREEZE_REVISION below for exactly what changed and what
deliberately did not.

v18 (R17 forensic-defect corrections: hidden-contract closure) deliberately re-froze models.py
again: ``AgentResultModel.status`` changed from a bare ``str`` -- while
``orchestration.exchange.validate_agent_response`` secretly required membership in
``orchestration.specs.RESULT_STATUSES`` -- to a schema-visible ``Literal`` single-sourced from that
same ``RESULT_STATUSES`` set (added as ``AgentResultStatus``), so the value set is now machine
-visible in the generated JSON Schema instead of only surfacing as an opaque "unknown agent result
status" ValueError after the fact. This is the same hidden-constraint defect class as, and was
found during the repository-wide audit prompted by, the R17 ``RecoveryPlanProposal.
corrective_action`` fix (in ``runtimes/pydantic_ai/recovery_bridge.py`` and
``runtimes/pydantic_ai/cli.py``, neither of which is a frozen file). No stage, gate,
recovery-taxonomy, capability-routing, protected-reference, schema_version, role/action-set, or
Judge change; every other field of every frozen file is untouched.

v19 (data_coverage/uncertainty/physical_validation/analysis executor closure) deliberately
re-froze actions.py to add four new backed action_type entries — see FREEZE_REVISION below for
exactly what changed and what deliberately did not.

v20 (R19 forensic-defect corrections: idempotent dispatch + partial-Judge-resume + atomic UTF-8
persistence) deliberately re-froze orchestration/exchange.py, runtimes/pydantic_ai/driver.py, and
runtimes/pydantic_ai/production_router.py again for the same reason as v10-v19 — see
FREEZE_REVISION below for exactly what changed and what deliberately did not.

v21 (autonomous, evidence-driven Teacher-validation planning) deliberately re-froze
workflow/controller.py (schema_version bumped 9->10), runtimes/pydantic_ai/actions.py (one new
backed data-curator action_type, build_split_membership_population), and
runtimes/pydantic_ai/role_outputs.py (registers the new TeacherValidationPlanProposal reasoning
output model) — see FREEZE_REVISION below for exactly what changed and what deliberately did not.

v22 (provenance-derived held-out-role resolution; no target_split autonomy defect) closes a
scientific-planning autonomy defect the v21 mechanism left open: ORIGINAL_HELDOUT_FIDELITY's
evidence was only computable if a caller pre-supplied the literal split name (``target_split=
"test"``), which meant an autonomous campaign author had to hardcode a scientific decision (which
split, and its name) before planning could even begin. workflow/controller.py (schema_version
UNCHANGED at 10 — no new state key, no new field shape; only which value populates the existing
``target_split`` field on an ORIGINAL_HELDOUT_FIDELITY plan) is re-frozen because
commit_teacher_validation_plan now resolves that field from this run's own independently
re-derived TeacherEvidenceProfile.resolved_heldout_split rather than trusting draft.get
("target_split") whenever ORIGINAL_HELDOUT_FIDELITY is selected — see FREEZE_REVISION below for
exactly what changed and what deliberately did not. runtimes/pydantic_ai/cli.py and validation/
teacher_evidence_profile.py (neither frozen) gained the companion split_roles provenance model and
the static-capability/agent-decision stage-routing split; no role/action-set, schema_version,
recovery-taxonomy, protected-reference, or Judge change.

v23 (parameter-dependent validation-action approval; R25 forensic-defect correction) closes an
authorization-routing defect confirmed in production run R25: ``build_teacher_baseline`` and
``validate_teacher_reference`` -- actions that only run existing-Teacher inference for
reporting/validation over already-existing structures/labels, never to create new DFT or
protected-reference labels -- were unconditionally routed through the same ``costly_teacher_
labeling`` boundary as genuinely corpus-growing actions (``acquire_structures``,
``label_with_teacher``). actions.py re-freezes to add ``resolve_action_approval_boundary`` (and
its ``_CONDITIONALLY_GATED_VALIDATION_ACTIONS``/``_declared_label_provenance_flags`` helpers),
which dispatch.py (not frozen) now consults instead of reading ``APPROVAL_GATED_ACTIONS`` directly:
for these two actions only, and only when their own proposal parameters explicitly and
affirmatively declare both ``dft_labels_used`` and ``protected_reference_labels_used`` False (never
inferred from absence), the ``costly_teacher_labeling`` boundary is waived; every other action, and
either of these two with any missing/True/non-boolean declaration, is unaffected and keeps the
unconditional default exactly as before. ``APPROVAL_GATED_ACTIONS``'s dict values are themselves
unchanged. No role/action-set membership, schema_version, recovery-taxonomy, protected-reference,
or Judge change; the separate ``_teacher_validation_downstream_reliance_gap`` policy (cli.py, not
frozen) is untouched.

v24 (R28 forensic-defect correction: bounded/representable external-executor lifecycle) closes the
defect R28 exposed: a Controller-dispatched external executor (the acquisition subprocess) that
never returns could not be recorded as attempted (``attempts`` stayed 0) or judged (``record_gate``
required ``status == "completed"``), so it could only ever be left unrepresentable at
pending/attempts=0 -- an operator had to kill the process from outside the Controller lifecycle to
close the run. workflow/controller.py re-freezes: ``begin_stage_execution`` now increments
``attempts`` and records a ``stage_execution_started`` event AT DISPATCH TIME (before the external
process can hang) instead of only on return; ``heartbeat_stage`` gained an optional ``progress``
payload and now emits an ``executor_heartbeat`` event; three new terminal-state methods --
``fail_stage_execution``, ``timeout_stage_execution``, ``cancel_stage_execution`` -- give a running
external-executor stage a representable failed/timed_out/cancelled status (new module constant
``EXECUTION_TERMINAL_FOR_GATE``); ``record_gate`` now accepts REVISE/FAIL against any of those
terminal statuses (PASS still requires ``status == "completed"``, unchanged) and tolerates zero
registered artifacts when building ``pending_recovery`` for a stage that never completed;
``run_stage`` now launches its command via the new ``workflow.subprocess_runner.run_bounded``
(process-group-scoped, via an optional per-stage ``timeout_s`` from ``initialize()``) instead of a
bare ``subprocess.run``, so a command-list stage can now time out, get ONLY its own process group
killed, and land on ``timed_out`` instead of hanging the Controller forever; ``run_stage`` also now
records its own ``runner``/heartbeat metadata, reusing the same generic mechanism the external
(pydantic_ai-dispatched) path uses. schema_version UNCHANGED at 10 -- every new key
(``timeout_s``, ``runner``, the new event types) is optional/additive with safe ``.get(...,
default)`` reads, so no on-disk migration is required. No role/action-set, recovery-taxonomy,
protected-reference, or Judge change; scientific defaults (e.g. acquisition
``similarity_threshold``) are untouched by this revision.

v25 (R28 forensic-defect correction, part 2: wiring the pydantic_ai dispatch pipeline itself to the
v24 lifecycle hooks) closes the remaining gap: v24 gave the Controller a representable
running/failed/timed_out vocabulary, but nothing in the actual PydanticAI producer-dispatch path
called it yet. dispatch.py (not frozen) gains an optional ``on_dispatch_start`` callback on
``authorize_and_execute``, fired exactly once, immediately before the trusted executor is actually
invoked -- i.e. only after role/capability/approval-boundary/param-validation/idempotency all pass
-- and never for a dry-run or any pre-executor rejection (DENIED, BLOCKED_CAPABILITY,
APPROVAL_REQUIRED, param-INVALID, DUPLICATE), all of which are resolved before this point and can
never hang. dispatch.py stays ignorant of the Controller; it only invokes a generic callback.
controller_bridge.py re-freezes: ``dispatch_via_controller`` gained the same additive
``on_dispatch_start`` passthrough. production_router.py re-freezes: ``run_role`` and
``_accept_via_dispatch`` gained the identical additive passthrough, used only by the
``producer_dispatch`` strategy; every other strategy (judge_gate, typed_result,
typed_reasoning_output, agent_result) ignores it entirely, unaffected. cli.py (not frozen) wires a
closure into this hook that calls ``workflow.controller.begin_stage_execution`` at the one point
that precedes the R28-class hang; every rejection branch that follows first checks whether the
stage actually reached "running" (i.e. the executor was genuinely invoked, as opposed to a
pre-executor rejection which leaves the stage "pending" exactly as before, unaffected). The
PENDING (``ExternalActionPending``) branch in ``run_production_stage`` calls the new
``defer_stage_execution`` back to "pending", preserving the pre-existing resumable-pause contract
byte-for-byte -- see ``tests/test_run_campaign_external_pending.py``, unchanged and still passing.
The ``status not in {"EXECUTED", "DUPLICATE"}`` rejection branch distinguishes a genuine TIMEOUT
(a forward-compatible, generic check for a reason string whose leading exception name ends in
"TimeoutError" -- no such exception exists yet; this is inert until one is introduced), which alone
calls ``timeout_stage_execution`` followed by ``record_gate(..., "FAIL", ...)``, returning the same
``GATE_{decision}`` shape the Judge-scored gate path already returns so run-campaign's existing
recovery loop picks it up for free -- from every OTHER (ordinary, fast, synchronous) executor
rejection, which instead calls ``defer_stage_execution`` back to "pending" and returns the same
``DISPATCH_REJECTED`` result as before, deliberately preserving the established ad-hoc direct
run-stage retry-after-fixing-input contract (``tests/test_base_plus_augmentation_dataset_route.py``,
``tests/test_production_readiness.py``) rather than opening a ``pending_recovery`` for a
non-hanging failure. The same running-check-then-``defer_stage_execution`` pattern is applied to a
declared-outputs-missing rejection and to any exception ``complete_external_stage`` itself raises,
in both cases then preserving the original return value/exception unchanged -- the executor already
succeeded in these two cases, so only the Controller-side "running" mark is reverted, not the
outcome. Either way -- terminal or reverted-to-pending -- ``attempts`` and the
``stage_execution_started``/``stage_execution_deferred`` events remain a permanent, durable record
that the attempt occurred; only whether ``status`` itself lands terminal or resumable differs.
workflow/controller.py re-freezes for one additive method, ``defer_stage_execution`` (return a
running stage to pending, recording a ``stage_execution_deferred`` event, without touching
``attempts``) -- no other controller.py behavior changed in this revision. schema_version UNCHANGED
at 10. No role/action-set, recovery-taxonomy, protected-reference, or Judge change; pre-executor
rejection outcomes (DENIED, BLOCKED_CAPABILITY, APPROVAL_REQUIRED, param-INVALID, DUPLICATE) are
byte-for-byte unaffected.

v26 (C12F Stage-7 forensic-defect correction: recovery-execution verification ordering) closes a
generic circular deadlock. When an approved recovery's ``return_stage`` equals the failed stage
(so that stage's gate is ``pending``) and its ``revalidation.targets`` name DOWNSTREAM stages,
``verify_recovery_execution`` previously required those downstream targets to already be
completed-with-changed evidence before it would verify -- but the downstream stages cannot run
until the return-stage gate passes, which cannot happen until recovery is verified. workflow/
controller.py re-freezes for exactly one relaxation in ``verify_recovery_execution``: the
``revalidation.stages`` list may now be EMPTY at verification time (downstream revalidation is
deferred to normal campaign progression AFTER the return stage re-earns PASS); every stage that IS
present is still validated exactly as before, and the corrective action's own changed evidence at
the return stage (via the required proposed_changes/labeling/student_training evidence) remains
mandatory -- so a recovery with no real corrective change still fails closed. The matching
assembler change lives in runtimes/pydantic_ai/cli.py (not frozen). schema_version UNCHANGED at 10.
No role/action-set, recovery-taxonomy, protected-reference, or Judge change; the approved
RecoveryPlan's stored semantics are untouched (this is purely a verification-ORDERING fix).

v27 (ffv4m recovery-materialization contract: pre-acceptance no-op refusal) closes the generic
RECOVERY_EXECUTION_UNVERIFIED dead-loop (docs/postmortems/ffv4m_recovery_execution_unverified.md). A
data_repair RecoveryPlan whose corrective ``action_type`` equalled its return stage's OWN
deterministic route action, re-run on unchanged inputs, re-emitted a byte-identical artifact
(DUPLICATE) that ``verify_recovery_execution`` correctly rejects -- an unbreakable loop, with no
pre-acceptance check that the corrective could ever MATERIALIZE a changed artifact. workflow/
controller.py re-freezes for additive read-only helpers (``_workflow_cfg``, ``_stage_route_action``,
``_stage_route_parameters``, ``_return_stage_replans_on_recovery``) and one new acceptance guard
``_validate_recovery_materialization`` wired into ``propose_recovery`` BEFORE binding: it classifies
(via the pure, generic ``workflow.recovery_taxonomy.classify_recovery_materialization``) the
materializing transition a corrective will produce, and if the effect is a provable deterministic
no-op AND the return stage's route action is KNOWN for this run, refuses the plan before approval
(never dispatching it into the loop). A same-action corrective still materializes when it authorizes
a scientific recompute, returns to a stage whose bound plan is superseded, or overrides one of the
return stage's DECLARED typed route input parameters with a different value; a corrective dispatching
a DISTINCT action_type materializes a distinct evidence artifact. When the route is unknown (a stage
with no ``pydantic_ai.action`` metadata) nothing is rejected and ``verify_recovery_execution`` stays
the backstop -- the guard is SOUND to skip, never a false rejection. The hash-change verification
invariant is UNCHANGED and un-weakened. schema_version UNCHANGED at 10. No role/action-set,
protected-reference, or Judge change. recovery_taxonomy.py (not frozen) gained the pure classifier
and MATERIALIZING_TRANSITIONS vocabulary; the static compatibility invariant and the ffv4m acceptance
gate are locked by tests/test_recovery_materialization_contract.py.
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREEZE_REVISION = (
    "validation-target-lock (v10; schema_version bumped 7->8 and workflow/controller.py "
    "gained a write-once validation_contract: a new top-level state key, None until "
    "establish_validation_contract() freezes it from the Teacher applicability domain, "
    "validation scope, and dataset split policy, each hash-bound at establishment; "
    "identical re-establishment is an idempotent no-op, differing content hard-fails and "
    "requires a new run; stages may declare produces_student_results, and BOTH run_stage() "
    "and complete_external_stage() (via the shared "
    "_require_validation_contract_for_student_stage helper) refuse to run/complete one "
    "until the contract exists — the guard was applied to both execution paths because "
    "real workflows complete Student-producing stages externally (agent/executor "
    "dispatch), not via run_stage's subprocess path; completing one via either path sets "
    "a permanent student_stage_ever_completed provenance marker on the contract record "
    "(via the shared _mark_student_stage_completed helper). "
    "propose_recovery/start_iteration/verify_recovery_execution were deliberately left "
    "UNCHANGED — no recovery method writes validation_contract, so recovery may re-run any "
    "stage, including a contract-consuming one like teacher_baseline or "
    "reference_validation under the unchanged frozen contract, but can never mutate, "
    "re-establish, or replace it; no role/action-set/Judge change); "
    "v11 (automatic establishment) closes the wiring gap a runtime-route audit found in v10: "
    "RunController.initialize() now accepts an optional top-level validation_contract_sources "
    "mapping ({distillation_scope, validation_profile, dataset_policy} paths); when a workflow "
    "declares it, initialize() snapshots the exact content of those three files into the new "
    "run's own inputs/contract_sources/ area (via shutil.copy2, inside the same temporary "
    "init directory used for every other run-bound artifact) BEFORE building anything from "
    "them, then builds the contract's components from those run-local snapshots — never from "
    "the still-mutable external paths — via the single shared "
    "workflow.contracts.build_validation_contract_components, and establishes the record via "
    "the single shared _build_validation_contract_record (module-level in workflow/controller.py) "
    "that both RunController.establish_validation_contract and initialize() now call — there is "
    "exactly one construction path, never two independently-mutable representations; the "
    "resulting record is written to state['validation_contract'] AND to a validation_contract.json "
    "file inside the same temporary directory, so both are byte-identical from the same write. "
    "Initialization remains atomic: source resolution/snapshotting, domain-equality verification "
    "(distillation_scope's deployment_domain is authoritative; validation_profile's copy must "
    "match exactly or this hard-fails), and contract establishment all happen inside initialize()'s "
    "existing temporary-directory transaction, so any failure (a missing source file, a domain "
    "mismatch) is caught by the existing except-Exception/rmtree(temporary)/raise handler and "
    "leaves no run directory behind at all — no new rollback mechanism was introduced. Existing/"
    "historical workflows that do not declare validation_contract_sources are entirely unaffected: "
    "validation_contract stays None exactly as before, so v10 behavior is preserved byte-for-byte "
    "for them (this is why R11, which has no validation_contract_sources key, was never touched). "
    "The manual establish-validation-contract CLI helper "
    "(workflow.steps.establish_validation_contract_from_configs) still exists as a compatibility/"
    "manual tool and now itself calls the same shared "
    "workflow.contracts.build_validation_contract_components instead of duplicating the "
    "domain-equality/component-building logic inline — a pure refactor with no behavior change, "
    "verified by the pre-existing test suite. No role/action-set/Judge/schema_version change; "
    "schema_version remains 8); "
    "v12 (recovery-taxonomy-and-capability-unification; schema_version bumped 8->9) COMPLETES "
    "the existing recovery state machine rather than replacing it -- the deterministic path "
    "(record_gate REVISE/FAIL -> pending_recovery -> propose_recovery -> approve_recovery -> "
    "start_iteration -> verify_recovery_execution -> record_gate PASS on the same stage -> "
    "recovery resolved) is UNCHANGED end to end. What changed: (1) RECOVERY_CATEGORIES is now "
    "DERIVED from the new workflow.recovery_taxonomy module's registered failure_code registry "
    "(a superset of the original 8 hardcoded values) instead of being its own independent fixed "
    "set, and propose_recovery resolves a submitted failure_category against the LIVE registry "
    "(recovery_taxonomy.resolve_failure_code), not a frozen snapshot, so this is the single "
    "shared vocabulary runtimes.pydantic_ai.root_cause.RootCauseClassification.failure_category "
    "also resolves against -- every one of the original 8 values remains registered, so every "
    "historical plan's failure_category still validates identically; an optional failure_domain "
    "plan field is cross-checked against the registry if supplied, never required. (2) "
    "responsible_agent routing gained an OPTIONAL, additive responsible_capability path: "
    "RunController.initialize() accepts an optional top-level recovery_capability_roster mapping "
    "(capability -> role) for this run only; DEFAULT_RECOVERY_CAPABILITY_ROSTER's value-set is "
    "exactly the original fixed RECOVERY_AGENTS tuple, so a plan supplying only "
    "responsible_agent (every historical plan) validates exactly as before via the same "
    "membership check against the roster's value-set; a plan may instead supply "
    "responsible_capability, resolved through the roster, with responsible_agent (if also "
    "present) cross-checked for consistency. (3) propose_recovery gained OPTIONAL plan fields -- "
    "diagnosis_binding (hash-verified diagnosis artifact + triggering-evidence provenance, fails "
    "closed on a missing/mismatched hash if present), required_input_artifact_roles / "
    "expected_output_artifact_roles (cross-checked against a new optional run-declared "
    "protected_reference_roles list; a plan may not route a protected-reference role into a "
    "training/acquisition input or output without a separate, explicit, human-approved "
    "protected_reference_reuse_authorization {authorized_by, rationale} on that same plan), and "
    "escalation_acknowledged/escalation_rationale -- every one is a no-op for a plan that omits "
    "it. (4) every proposed recovery now always gets a deterministic recovery_signature (hash of "
    "trigger evidence + failure_category + return_stage + corrective-action types) recorded on "
    "it (additive record field, never a required input); an OPTIONAL run-level recovery_policy "
    "mapping (max_recovery_attempts, allowed_action_types, cumulative_budget, "
    "max_repeated_signature) is enforced by propose_recovery ONLY when present -- no default "
    "retry count or budget is invented when it is absent, matching every existing "
    "run/workflow.yaml exactly. (5) two new methods, authorize_recovery_capabilities and "
    "verify_recovery_authorization, add an explicit, human-approval-created, hash-bound "
    "RecoveryAuthorizationEnvelope scoped to the CURRENT activated recovery iteration; "
    "approve_recovery (unchanged) still only approves the recovery PLAN and never authorizes a "
    "costly child action by itself -- authorize_recovery_capabilities is a separate, explicit "
    "call, and verify_recovery_authorization returning None means \"no envelope covers this "
    "action\", not \"this action is forbidden\": a dispatcher must still fall back to the run's "
    "unweakened normal per-action APPROVAL_GATED_ACTIONS requirement in that case. (6) that "
    "dispatcher contract is now actually wired: runtimes.pydantic_ai.dispatch.authorize_and_"
    "execute gained an OPTIONAL recovery_authorization parameter consulted ONLY at the point an "
    "approval-gated action already lacks a normal per-action approval record, as one additional "
    "narrower alternative -- a run/call that never passes recovery_authorization (every existing "
    "caller before this revision) behaves identically to before; runtimes.pydantic_ai."
    "controller_bridge (frozen) gained a ControllerRecoveryAuthorizationStore delegating straight "
    "to verify_recovery_authorization, wired into dispatch_via_controller, so a costly child "
    "action proposed through the controller bridge during an activated recovery iteration gets "
    "exactly one extra, narrower chance to be authorized via that iteration's "
    "RecoveryAuthorizationEnvelope before APPROVAL_REQUIRED is returned; the resulting "
    "ActionOutcome gained an additive recovery_authorization_envelope_sha256 field (audit-only, "
    "None unless an envelope was actually used). dispatch.py is not itself a frozen file; only "
    "controller_bridge.py's hash changes for this sub-revision. No stage, "
    "gate, adjudication, artifact-integrity, or Judge-vote semantics changed; no role/action-set/"
    "Judge change) (7) RunController._validate_protected_reference_roles (called from "
    "propose_recovery) gained one additive check: beyond the existing top-level "
    "required_input_artifact_roles/expected_output_artifact_roles scan, it now also unions in "
    "any `artifact_roles` list found on each of the plan's own proposed_changes entries (e.g. "
    "runtimes.pydantic_ai.acquisition_targeting.AcquisitionTargetProposal/DataRepairProposal's "
    "own artifact-role declarations, new this revision) before checking against the run's "
    "protected_reference_roles -- closing the gap where a protected-reference role named only "
    "inside one typed proposed_changes entry, never lifted into the plan's own top-level role "
    "lists, would otherwise bypass the v12 check entirely. This assumes no particular "
    "proposed_changes shape/kind (any dict with an `artifact_roles` key is scanned) and requires "
    "the exact same protected_reference_reuse_authorization override as before; a plan whose "
    "proposed_changes never declare artifact_roles (every existing plan before this revision) "
    "behaves identically to v12. No other method changed) (8) "
    "recovery-proposal-approval-authority-separation: propose_recovery now requires a NEW plan "
    "field, proposed_by (a bare human display-name string, or a structured "
    "{actor_kind: human|agent|system, canonical_id, display_name?} mapping -- see the new, "
    "unfrozen workflow/actor_identity.py module), and records the resolved ActorIdentity onto "
    "the recovery record; a plan omitting it is rejected exactly like any other required field "
    "(root_cause, return_stage) -- this closes a real gap, since propose_recovery accepts a raw "
    "plan dict directly from disk and a hand-authored plan.json was never required to route "
    "through any agent-facing bridge, so the requirement is enforced here, not only in "
    "runtimes.pydantic_ai.recovery_bridge.RecoveryPlanDraft (which also gained the same "
    "required proposed_by field, passed through build_recovery_plan_draft, purely additively). "
    "approve_recovery now enforces two invariants, both fail-closed and neither previously "
    "implemented: the resolved approved_by actor_kind must be \"human\" (an automated Agent/"
    "System actor can never satisfy human approval, regardless of what identity it supplies), "
    "and -- only if the recovery has a recorded proposed_by -- the approver's canonical_id "
    "(whitespace/case-normalized via ActorIdentity, never a raw string ==) must differ from the "
    "proposer's; a recovery record with no recorded proposed_by (a historical manifest that "
    "predates this revision) skips only the second check, never the first, so it can never "
    "become silently self-approved merely by lacking a proposer. authorize_recovery_capabilities "
    "gained the identical pair of checks against the same recorded proposed_by, independently of "
    "approve_recovery's own check, so a RecoveryAuthorizationEnvelope can never be self-issued by "
    "the original proposing actor even after a different human has validly approved the "
    "recovery. Both human_approval.approved_by and each authorization envelope's authorized_by "
    "now store the resolved {actor_kind, canonical_id, display_name} dict (previously a bare "
    "string) -- an additive record-shape change only; every existing call site (approve_recovery/"
    "authorize_recovery_capabilities' own positional argument, the CLI's --approved-by/"
    "--authorized-by) still accepts a plain string unchanged. No permissive same-human "
    "propose+approve mode exists anywhere in this framework, and none was introduced. "
    "start_iteration, verify_recovery_execution, the recovery state machine's stage sequencing, "
    "and every other method are UNCHANGED; the normal per-action approval mechanism (grant_"
    "action_approval/has_action_approval) is entirely separate machinery and was not touched.) (9) "
    "trust-boundary hardening for the recorded proposer identity: propose_recovery gained one "
    "new OPTIONAL keyword-only argument, proposer (default None), used exclusively by trusted, "
    "non-payload callers -- an agent-facing plan.json's own proposed_by field was never itself a "
    "trustworthy authority claim (an LLM authoring that file could write proposed_by: "
    "\"researcher\" or {actor_kind: \"human\", ...} and, before this revision, have it accepted "
    "outright by normalize_actor_identity). When proposer is supplied, it -- not "
    "plan[\"proposed_by\"] -- is the identity actually recorded on the recovery record; if the "
    "plan payload also declares a proposed_by, it may only AGREE with the trusted proposer "
    "(matching both actor_kind and canonical_id) and is never itself authoritative -- any "
    "disagreement in either field fails closed with a ValueError rather than silently preferring "
    "one side. When proposer is omitted (every call site before this revision: the human-operated "
    "CLI's propose-recovery subcommand, direct/manual calls, and every existing test), behavior is "
    "byte-for-byte unchanged from (8) -- plan[\"proposed_by\"] is trusted outright, which remains "
    "correct only because that omitted-proposer call shape is reserved for genuinely "
    "human-operated entry points (confirmed: no agent-driven/programmatic code anywhere in this "
    "repository shells out to the CLI). The one real wired agent-facing path to propose_recovery, "
    "runtimes.pydantic_ai.orchestrator_bridge._exec_propose_recovery, now derives its trusted "
    "proposer from proposal.requested_by_role -- a Pydantic Literal[\"orchestrator\"]-typed field "
    "on OrchestratorActionProposal that an LLM cannot forge to any other value -- rather than "
    "leaving the plan payload's own proposed_by as the sole source of provenance; this closes the "
    "one confirmed agent-impersonates-human gap. approve_recovery/authorize_recovery_capabilities "
    "and their human-actor-kind/no-self-approval checks from (8) are entirely UNCHANGED by this "
    "revision; no agent-callable path exists anywhere (runtimes.pydantic_ai.dispatch.py, "
    "controller_bridge.py) that could supply an approved_by/authorized_by value to either method, "
    "and the orchestrator bridge's only two declared human-approval-adjacent actions, "
    "request_human_approval and read_human_decision, remain unwired and fail closed as "
    "BLOCKED_CAPABILITY, exactly as before this revision. No stage, gate, recovery-taxonomy, "
    "capability-routing, protected-reference, or schema_version change; no role/action-set/Judge "
    "change.) "
    "v13 (post-init lifecycle completion) closes a real gap a production run (R12) hit directly: "
    "initialize() is create-only (FileExistsError if run_dir exists) and rebind_inputs() only "
    "re-verifies input records ALREADY present in state[\"inputs\"] at their already-declared "
    "source paths -- it cannot append a new one, and there was no sanctioned way to set/unset "
    "recovery_policy after initialization either, so a run that skipped declaring these at init "
    "time (e.g. because a human policy or an additional provenance input was only decided/"
    "discovered afterward) had no path back to a compliant state short of abandoning the run. Two "
    "new methods close this without introducing any new record shape or loosening any existing "
    "check: (1) set_recovery_policy(policy) applies the IDENTICAL validation initialize() already "
    "applies to cfg.get(\"recovery_policy\") (must be None or a mapping, never invented) to the "
    "live state[\"recovery_policy\"] key post-init, recording a recovery_policy_updated event; "
    "recovery_policy None continues to mean _enforce_recovery_policy enforces zero loop-safety "
    "limits, exactly as before this revision -- calling this method cannot make propose_recovery "
    "either stricter or more permissive than a human explicitly requests. (2) bind_new_input"
    "(source, *, copy=True) applies the IDENTICAL existence-check / artifact_digest / sequential-"
    "index snapshot-copy logic initialize() already applies to each declared workflow input, and "
    "appends one new record of the EXACT SAME shape (source, snapshot, copy, source_integrity, "
    "size, sha256, source_sha256) to state[\"inputs\"] -- no new field, no new record kind. Neither "
    "method touches validation_contract, stages, gates, artifacts, or recovery state, and neither "
    "is reachable from any stage-execution path (run_stage/complete_external_stage/rebind_inputs "
    "still call verify_inputs(), whose project-code-revision/source-input-hash checks are "
    "entirely unchanged); a run that never calls either new method behaves byte-for-byte as before "
    "this revision. No stage, gate, contract, recovery-taxonomy, capability-routing, protected-"
    "reference, or schema_version change; no role/action-set/Judge change. "
    "v14 (generic typed reasoning-output acceptance path) closes a real production gap: before "
    "this revision, RootCauseClassification (runtimes.pydantic_ai.root_cause) had NO production "
    "dispatch path at all -- every non-test reference either hand-constructed it directly in "
    "Python (bypassing the agent runtime) or consumed an already-constructed instance "
    "(recovery_bridge.build_recovery_plan_draft); the Analyst's only wired production output "
    "remained AnalystActionProposal, and classify_root_cause's executor was, and remains, an "
    "unimplemented reasoning stub (executors._reasoning, fn=None, always DRY_RUN -- UNCHANGED by "
    "this revision). This revision adds a FOURTH, generic acceptance strategy, typed_reasoning_"
    "output, alongside the existing judge_gate/producer_dispatch/typed_result -- for advisory, "
    "evidence-bound scientific reasoning results that must be genuinely produced by a live "
    "PydanticAI role, Pydantic-validated, and hash-bound as their own persisted artifact, but must "
    "never dispatch an executor or mutate the controller merely because they were accepted. "
    "role_outputs.select_output_model gained an optional second parameter, task (every existing "
    "call site passed none before this revision and gets None here too, so a task-blind caller is "
    "unaffected): if task['context']['expected_output_model'] names a model registered via the new "
    "register_reasoning_output_model(name, model) registry, THAT model -- not the role's fixed "
    "default -- is what pydantic_ai_runtime.PydanticAIRuntime.run enforces as output_type and what "
    "production_router validates against, for that one invocation only; an unregistered name fails "
    "closed (raises) rather than silently falling back to the role default. A task that omits "
    "expected_output_model (every task shape that existed before this revision) resolves exactly "
    "as before: select_output_model(spec) and select_output_model(spec, None) are identical. "
    "RootCauseClassification is registered as the first (not the only ever) member of this "
    "registry; this lets a future scientific role reuse the identical mechanism for a different "
    "registered reasoning model without touching production_router again. production_router."
    "acceptance_strategy gained the matching optional task parameter and now checks role_outputs."
    "is_reasoning_output_model before the existing ActionProposal/typed_result checks (ordering is "
    "immaterial today since no reasoning-output model name collides with either, but reasoning-"
    "output resolution intentionally take priority as the most specific, explicitly-registered "
    "match). run_role gained an optional reasoning_validator keyword (default None, so every "
    "existing caller is unaffected): an instance -> instance callable performing CONTEXTUAL "
    "fail-closed validation the router itself stays ignorant of (e.g. root_cause.validate_root_"
    "cause_classification bound to a specific run's available artifacts/valid recovery targets); "
    "it runs only on the typed_reasoning_output path, only after Pydantic shape validation already "
    "passed, and any exception it raises is a rejection like any other. Acceptance (mode=primary, "
    "no validator rejection) persists the instance as its own attempt-scoped, sha256-hashed JSON "
    "file under <exchange_dir>/reasoning_outputs/ (mirroring driver._write_record's per-attempt-"
    "filename convention so a retry can never overwrite a prior accepted artifact) IN ADDITION TO "
    "the unconditional provenance record every path already wrote; RouteResult.detail for this "
    "strategy is a new ReasoningOutputAcceptance(instance, artifact_path, artifact_sha256) rather "
    "than reusing typed_result's bare-candidate detail shape, so a caller can bind downstream "
    "provenance (e.g. a future RecoveryPlanDraft.diagnosis_binding) to a real hash without "
    "re-deriving one. _validate_typed now returns (instance_or_None, error) instead of (ok, error) "
    "-- an internal signature change consumed only by run_role itself, needed because the "
    "reasoning-output path (unlike producer_dispatch/typed_result, which only needed a bool) must "
    "carry the parsed instance forward to the contextual validator and the persistence step. "
    "judge_gate/producer_dispatch/typed_result/agent_result acceptance behavior, RouteResult's "
    "existing fields, and every existing call site that never passes task or reasoning_validator "
    "are BYTE-FOR-BYTE unchanged by this revision. No stage, gate, recovery-taxonomy, capability-"
    "routing, protected-reference, schema_version, role/action-set, or Judge change; classify_root_"
    "cause's executor binding, dispatch.py, and controller_bridge.py are untouched. "
    "v15 (recovery-plan reasoning output) registers a SECOND member of the v14 typed-reasoning-"
    "output registry, RecoveryPlanProposal (runtimes.pydantic_ai.recovery_bridge, itself NOT a "
    "frozen file), closing the remaining gap v14 deliberately left open: RootCauseClassification "
    "alone cannot supply recovery_bridge.build_recovery_plan_draft's scientific-choice fields "
    "(capability, proposed_changes, labeling, student_training, revalidation, return_stage -- the "
    "Analyst's recommended_recovery_target is advisory, not binding on the actual plan). Those "
    "choices remain genuinely scientific, so nothing deterministic fills them in: RecoveryPlan"
    "Proposal is produced by a live PydanticAI Orchestrator role through the IDENTICAL typed_"
    "reasoning_output path RootCauseClassification already uses -- role_outputs.py's only change is "
    "importing RecoveryPlanProposal and calling the existing register_reasoning_output_model a "
    "second time; acceptance_strategy/run_role/_accept_typed_reasoning_output/_validate_typed are "
    "untouched (is_reasoning_output_model already generalized over the whole registry in v14). The "
    "proposal is evidence-bound back to the diagnosis it answers (diagnosis_artifact_sha256), and "
    "recovery_bridge.validate_recovery_plan_proposal (a new, non-frozen-file function, used only as "
    "a caller-supplied reasoning_validator exactly like root_cause.validate_root_cause_classifica"
    "tion) fails closed on a wrong failed_stage, a stale/mismatched diagnosis binding, an "
    "unregistered capability, or an invalid return_stage. recovery_bridge.build_recovery_plan_draft_"
    "from_proposal re-projects an already-validated proposal into the EXISTING, byte-for-byte-"
    "unchanged build_recovery_plan_draft -- there is still only one function that ever constructs a "
    "RecoveryPlanDraft; this revision only adds a second, agent-driven source for its scientific "
    "fields. No stage, gate, recovery-taxonomy, capability-routing, protected-reference, schema_"
    "version, role/action-set, or Judge change; propose_recovery and every other frozen file are "
    "untouched. "
    "v16 (campaign observability progress_cb threading) re-freezes production_router.py and "
    "controller_bridge.py for a single additive change: an optional progress_cb: Optional[Callable"
    "[[dict], None]] = None parameter, threaded run_role -> _accept_via_dispatch -> "
    "dispatch_via_controller -> dispatch.authorize_and_execute (dispatch.py itself is, as before, "
    "not a frozen file). When given AND the resolved ActionDescriptor.executor's signature "
    "declares a progress_cb parameter (checked via inspect.signature; no existing or hypothetical "
    "executor is required to accept it), authorize_and_execute passes it straight through to that "
    "executor call; otherwise the executor is invoked exactly as before. No executor in this "
    "revision declares progress_cb, so no behavior changes for any existing dispatch path -- this "
    "is a dormant hook for a future long-running executor to report genuine, non-fabricated "
    "progress to the new runtimes/pydantic_ai/events.py CampaignEventEmitter (itself NOT a frozen "
    "file). No change to acceptance strategy selection, typed-output validation, approval/"
    "idempotency/recovery-authorization enforcement order, ActionOutcome fields, or any gate/"
    "recovery semantic; every other frozen file is untouched. "
    "v17 (R16 forensic-defect corrections) re-freezes models.py and pydantic_ai_runtime.py for two "
    "narrow, additive fixes found by forensic analysis of a real production run (R16) that hit a "
    "PydanticAI internal structured-output retry exhaustion during Analyst dispatch. (1) "
    "RuntimeContext.structured_output_retries' default changed 0 -> 1 to match pydantic-ai's own "
    "implicit Agent default (retries=1 used as output_retries when output_retries is not given) -- "
    "no caller in this repository passes this field explicitly, so every existing production run's "
    "actual retry behavior is unchanged; only the field's on-paper default now honestly reflects "
    "what already ran. (2) pydantic_ai_runtime.PydanticAIRuntime._build_agent now actually passes "
    "output_retries=getattr(context, \"structured_output_retries\", 1) into the constructed "
    "Agent(...) -- before this revision the field was threaded onto RuntimeContext but never wired "
    "into Agent construction at all, so it was silently inert; wiring it is the only behavior "
    "change, and it is strictly additive (a context that never set the field still gets the same "
    "default of 1 as before). The separate, root-cause taxonomy machine-visibility fix (Root"
    "CauseClassification.failure_category now typed directly as recovery_taxonomy.failure_category_"
    "enum() instead of a plain str plus a hidden field_validator) lives entirely in root_cause.py, "
    "which is NOT a frozen file, so it required no re-freeze; likewise the classify_failure "
    "structured-output-exhaustion branch (failures.py) and the run-campaign REVISE/FAIL single-"
    "invocation handoff plus campaign_started/campaign_resumed lifecycle-event semantics (cli.py, "
    "events.py) are all in non-frozen files. No stage, gate, recovery-taxonomy, capability-routing, "
    "protected-reference, schema_version, role/action-set, or Judge change; every other frozen file "
    "is untouched. "
    "v18 (R17 forensic-defect corrections: hidden-contract closure) re-freezes models.py for one "
    "narrow, additive fix found by a repository-wide contract-parity audit prompted by the R17 "
    "corrective_action forensic (that fix itself lives entirely in recovery_bridge.py and cli.py, "
    "neither a frozen file, so it required no re-freeze). AgentResultModel.status changed from a "
    "bare str -- while orchestration.exchange.validate_agent_response secretly required membership "
    "in orchestration.specs.RESULT_STATUSES ({'completed','needs_input','blocked','failed'}) -- to "
    "a schema-visible Literal (AgentResultStatus) built directly from that same RESULT_STATUSES "
    "set, so the value set now appears in the generated JSON Schema instead of only surfacing as an "
    "opaque ValueError after the fact. No caller passes a status outside RESULT_STATUSES today, so "
    "no existing production behavior changes; only the field's on-paper type now honestly reflects "
    "what was already enforced. No stage, gate, recovery-taxonomy, capability-routing, protected-"
    "reference, schema_version, role/action-set, or Judge change; every other frozen file is "
    "untouched. "
    "v19 (data_coverage/uncertainty/physical_validation/analysis executor closure) re-freezes "
    "actions.py: it deliberately grows the role/action-set, ADDING four new backed action_type "
    "entries -- build_data_coverage_report (data-curator), build_uncertainty_report (ml-trainer), "
    "build_physical_validation_report (simulation), and generate_run_summary (analyst) -- to "
    "DATA_CURATOR_ACTIONS/ML_TRAINER_ACTIONS/SIMULATION_ACTIONS/ANALYST_ACTIONS respectively. Each "
    "is now backed by a real, deterministic, self-validating executor in executors.py (itself NOT "
    "a frozen file) plus a new structural-contract validator (validation/uncertainty.py, "
    "validation/data_coverage.py's existing validator, validation/run_summary.py -- none frozen), "
    "closing the previously-undispatchable gap for the data_coverage/uncertainty/"
    "physical_validation/analysis stages: a workflow config MAY now opt one of these stages into "
    "run-campaign automation via an explicit pydantic_ai block, exactly like any other stage. This "
    "is strictly additive to the existing action tuples/CAPABILITY_REGISTRY entries -- no existing "
    "action_type, role, or Literal membership is removed or renamed. cli.py (not frozen) gained the "
    "matching _assemble_run_summary_state snapshot assembler and a generate_run_summary pre-dispatch "
    "hook; no default_stage_route entry was added for any of the four stage names, so a workflow "
    "config with no explicit pydantic_ai block for one of them (e.g. a frozen production config "
    "like R17's) is completely unaffected and continues to fail closed exactly as before. No stage, "
    "gate, recovery-taxonomy, capability-routing, protected-reference, schema_version, or Judge "
    "change; every other frozen file, and every other field of actions.py, is untouched. "
    "v20 (R19 forensic-defect corrections: idempotent dispatch + partial-Judge-resume + atomic "
    "UTF-8 persistence) re-freezes orchestration/exchange.py, runtimes/pydantic_ai/driver.py, and "
    "runtimes/pydantic_ai/production_router.py to close one bounded durability defect family "
    "found by forensic analysis of a real production run (R19) that hit an interrupted "
    "three-Judge gate: Judge 1's accepted result already existed, Judge 2's raw-response write "
    "was cut short by a UnicodeEncodeError (leaving a zero-byte raw file) under a non-UTF-8 "
    "default locale, and resuming crashed again with a bare FileExistsError from "
    "FileExchangeRuntime.dispatch before even re-invoking Judge 1. Three changes, deliberately "
    "combined into one revision because they are one defect family: (1) FileExchangeRuntime."
    "dispatch is now idempotent and immutable rather than fail-if-exists: task identity "
    "(task_id) stays deterministic; if no packet exists one is written; if one exists, its "
    "content is canonically compared (json.dumps(..., sort_keys=True)) against the newly "
    "derived task -- identical content is reused as a no-op, differing content fails closed "
    "with the new TaskPacketConflictError (a FileExistsError subclass, so any existing bare "
    "`except FileExistsError` handler -- e.g. runtimes/pydantic_ai/cli.py's single-role-invoke "
    "command, not itself a frozen file -- still catches it); the existing conflicting packet is "
    "never deleted, renamed, or overwritten. (2) every textual JSON/text write in the exchange "
    "persistence path -- FileExchangeRuntime.dispatch/_preserve_raw/accept (exchange.py), "
    "driver._write_record (the shared provenance-write primitive used by both driver.run_task "
    "and, via import, production_router.run_role), and production_router."
    "_persist_reasoning_artifact -- now goes through one new shared helper, orchestration."
    "exchange.atomic_write_text(path, text, encoding=\"utf-8\"): write to a fresh temp file in "
    "the same directory, flush+fsync, then os.replace onto the target only after encoding/"
    "writing fully succeeds; any exception (including a UnicodeEncodeError) removes the temp "
    "file and re-raises without ever truncating or corrupting the existing durable file at the "
    "target path. Every read of a file written this way (dispatch's own existing-packet "
    "comparison, collect, accept's task-packet read) is likewise made an explicit "
    "encoding=\"utf-8\" read instead of the platform/locale default, so a file written as UTF-8 "
    "is not later unreadable under an ASCII/C/POSIX locale. Raw-response suffix-on-"
    "resubmission preservation (_preserve_raw) and the reasoning-output/provenance attempt-"
    "scoped filename conventions are entirely UNCHANGED -- only the underlying write primitive "
    "is atomic+UTF-8 now. No hidden chain-of-thought retention policy changed; this is a "
    "persistence-integrity fix only. (3) runtimes/pydantic_ai/cli.py (NOT a frozen file) gained "
    "resume-aware per-Judge-index logic in run_three_judge_gate: for each judge index the "
    "deterministic task is always re-derived and (re-)dispatched (idempotent per (1)); a new "
    "helper, _resume_judge_vote, checks whether an already-accepted result exists at "
    "exchange/results/{task_id}.json, and if so revalidates it against the CURRENT criteria/"
    "review_lens (validate_judge_vote) and requires a corresponding accepted=true provenance "
    "record (via the new _accepted_judge_provenance_exists) -- if both hold, that vote is "
    "reused and the Judge is NOT invoked again; if no result exists yet (including when only a "
    "malformed/zero-byte/incomplete raw response is present, which is never treated as accepted "
    "state, since accept() only ever writes results/ after full contract validation), the Judge "
    "is invoked exactly once for that index; any existing result that no longer cleanly binds "
    "to the currently derived task fails closed via the new JudgeResumeConflictError. Gate "
    "aggregation itself is UNCHANGED: votes (reused or freshly invoked) are collected exactly "
    "as before, the decision is computed the same way, and controller.record_gate is still "
    "called exactly once at the end -- no new Gate-checkpointing mechanism was introduced "
    "because the existing all-or-nothing record_gate call was already atomic (confirmed during "
    "the R19 forensic audit: the interrupted run's gates/ directory was empty, proving Gate "
    "aggregation was never partially recorded). No stage, contract, recovery-taxonomy, "
    "capability-routing, protected-reference, schema_version, role/action-set, or Judge-vote "
    "SEMANTIC change (the Judge lens/criteria/verdict contract is identical); every other "
    "frozen file, and every other field of the three re-frozen files, is untouched. "
    "v21 (autonomous, evidence-driven Teacher-validation planning) replaces the old SiO2/"
    "Allegro-specific Teacher-validation branch with a generic, additive, evidence-driven "
    "component model (validation/teacher_evidence_profile.py, NOT a frozen file) plus an "
    "autonomous PydanticAI planning pipeline that decides WHICH admissible component(s) a "
    "campaign actually uses. workflow/controller.py (schema_version bumped 9->10) gains three "
    "new additive, all-optional state keys -- teacher_evidence_sources (a run's OPTIONAL frozen "
    "evidence-source paths, resolved/validated at initialize() but not copy-snapshotted), "
    "teacher_validation_plan (None until commit_teacher_validation_plan(), the sole "
    "authoritative validator, independently RE-RUNS inspect_teacher_evidence against this run's "
    "own frozen sources and re-derives the admissible decision space rather than trusting a "
    "submitted draft's embedded profile; write-once/idempotent-on-identical-content exactly "
    "like establish_validation_contract), and stage_applicability (populated only by the new "
    "mark_stage_not_applicable method) -- and one new gate value, NOT_APPLICABLE, for a stage "
    "whose own run evidence establishes it does not apply at all; NOT_APPLICABLE deliberately "
    "bypasses record_gate entirely (a new, separate method) because record_gate's non-PASS "
    "branch invalidates downstream stages and opens a pending recovery, semantics that are "
    "correct for a stage that ran and failed but wrong for one that never applied; "
    "_previous_passed/verify_stage_artifacts treat NOT_APPLICABLE identically to PASS for "
    "upstream gating while skipping artifact verification (a NOT_APPLICABLE stage registers "
    "zero artifacts by construction). A new authorize_downstream_teacher_reliance method gates "
    "-- separately from commit_teacher_validation_plan, which never accepts an evidence-"
    "unsupported claim regardless of any approval -- ONLY costly downstream reliance (Teacher "
    "labeling / Student training) on a plan that is itself valid but lacks predictive-fidelity "
    "evidence (ORIGINAL_HELDOUT_FIDELITY/INDEPENDENT_REFERENCE_FIDELITY), requiring an "
    "authorized human actor exactly like approve_recovery; it is a no-op for a plan that already "
    "includes fidelity evidence. runtimes/pydantic_ai/actions.py re-freezes to add ONE new "
    "backed data-curator action_type, build_split_membership_population (DATA_CURATOR_ACTIONS), "
    "additive exactly like v19's four executor-closure entries. runtimes/pydantic_ai/"
    "role_outputs.py re-freezes to register ONE new typed reasoning-output model, "
    "TeacherValidationPlanProposal (runtimes/pydantic_ai/teacher_validation_plan.py, NOT a "
    "frozen file), via the existing register_reasoning_output_model path -- the same "
    "propose/validate/commit pattern recovery_bridge.py already established for RecoveryPlan, "
    "not a new mechanism. cli.py/orchestrator_bridge.py/tool_manifests.py/workflow/contracts.py "
    "(none frozen) wire automatic pre-Stage-1 planning into run-campaign, a manual "
    "plan-teacher-validation subcommand, an authorize-downstream-teacher-reliance subcommand, "
    "and an OPTIONAL teacher_validation_objectives key in validation_profile.yaml -- the whole "
    "pipeline is opt-in (a workflow that declares no teacher_evidence_sources is entirely "
    "unaffected; teacher_validation_plan stays None exactly as before). No existing stage, "
    "gate verdict other than the new additive NOT_APPLICABLE, recovery-taxonomy, capability-"
    "routing, protected-reference, role/action-set entry, or Judge change; every other field of "
    "every re-frozen file is untouched. "
    "v22 (provenance-derived held-out-role resolution) closes the target_split autonomy defect: "
    "validation/teacher_evidence_profile.py (not frozen) adds a generic split_roles vocabulary "
    "(SPLIT_ROLE_TRAINING/VALIDATION/HELDOUT_EVALUATION) a split manifest MAY declare -- a "
    "{<split name>: role} mapping, merged across every split_source_manifest_paths entry with "
    "fail-closed behavior on any cross-manifest role conflict for the same split name -- and "
    "inspect_teacher_evidence now resolves genuine_holdout_test_available/"
    "genuine_holdout_test_frame_count from EITHER a caller-supplied target_split override, OR "
    "provenance alone (exactly one split name uniquely carrying the heldout_evaluation role "
    "among this training DB's actually-resolved splits), and fails closed (treats no held-out "
    "split as resolved) if zero or multiple such roles are found, or if an explicit override and "
    "the provenance-resolved role disagree. The resolved name (whichever source produced it) is "
    "exposed on the profile as the new resolved_heldout_split field. workflow/controller.py's "
    "commit_teacher_validation_plan is where this changes observable behavior and is why it is "
    "re-frozen: when a committed plan's selected_components includes ORIGINAL_HELDOUT_FIDELITY, "
    "the persisted target_split field is now ALWAYS resolved_target_split -- this run's own "
    "freshly-recomputed profile.resolved_heldout_split -- never draft.get('target_split'), so no "
    "proposer (human or agent) can steer which literal split gets bound merely by writing a "
    "different value into a draft/proposal; a plan selecting ORIGINAL_HELDOUT_FIDELITY with no "
    "resolvable held-out split is unreachable by construction (admissibility itself already "
    "requires genuine_holdout_test_available, which only becomes true alongside a resolved "
    "split name) and raises RuntimeError defensively if it were ever hit. For every OTHER "
    "selected-components combination (i.e. without ORIGINAL_HELDOUT_FIDELITY), target_split "
    "resolution is completely unchanged (still draft.get('target_split'), typically None). "
    "runtimes/pydantic_ai/cli.py (not frozen) separately generalizes "
    "_teacher_validation_not_applicable_reason so a stage's teacher_validation_component MAY "
    "declare a LIST of statically-capable components (STATIC CAPABILITY, a workflow-authoring-"
    "time fact) instead of only a single one; applicability is now the intersection of that "
    "declared capable set with the committed plan's selected_components (the separate, later "
    "AGENT DECISION) -- a single-string declaration continues to work identically to before "
    "(treated as a one-element set). This lets a stage declare it can execute under more than "
    "one admissible component without the workflow author having to guess in advance which one "
    "a given campaign's plan will select; the plan (never the workflow config, never a Judge/LLM "
    "self-routing at dispatch) is still what narrows applicability, so no self-skip/self-route "
    "capability was introduced. configs/templates/workflow.yaml (not frozen, not code) separately "
    "gains a pydantic_ai block for the teacher_baseline stage (role/action + the frozen, bounded "
    "teacher_md_sanity catastrophic-sanity protocol values already used unchanged since R21) so "
    "future new-run authoring does not silently lose it to executor defaults that differ from "
    "the established protocol (timestep_fs, seed) -- template/config only, no executor semantics "
    "changed. No new state key, no schema_version change, no new stage, no gate-verdict change, "
    "no recovery-taxonomy change, no protected-reference change, no role/action-set entry "
    "change, no Judge change; every other field of every re-frozen file is untouched. "
    "v23 (parameter-dependent validation-action approval; R25 forensic-defect correction) "
    "re-froze actions.py to add resolve_action_approval_boundary + "
    "_CONDITIONALLY_GATED_VALIDATION_ACTIONS + _declared_label_provenance_flags: "
    "build_teacher_baseline/validate_teacher_reference's default costly_teacher_labeling "
    "boundary is waived only when the proposal's own parameters explicitly declare both "
    "dft_labels_used and protected_reference_labels_used False (never inferred from absence); "
    "every other action, and either of these two absent an explicit False/False declaration, "
    "keeps the unconditional default. APPROVAL_GATED_ACTIONS's dict values are unchanged; "
    "dispatch.py and cli.py (neither frozen) wire the new resolver in and supply the generic "
    "reference_validation default parameters respectively. No role/action-set membership, "
    "schema_version, recovery-taxonomy, protected-reference, or Judge change; "
    "_teacher_validation_downstream_reliance_gap is untouched. "
    "v24 (R28 forensic-defect correction: bounded/representable external-executor lifecycle) "
    "re-froze workflow/controller.py: begin_stage_execution now records attempts+1 and a "
    "stage_execution_started event at dispatch time (not only on return); heartbeat_stage gained "
    "an optional progress payload and emits executor_heartbeat; new fail_stage_execution/"
    "timeout_stage_execution/cancel_stage_execution methods and the new "
    "EXECUTION_TERMINAL_FOR_GATE constant let record_gate accept REVISE/FAIL against a stage "
    "that ran and definitively did not complete (PASS still requires status == completed, "
    "unchanged), tolerating zero registered artifacts in that case; run_stage now dispatches via "
    "workflow.subprocess_runner.run_bounded with an optional per-stage timeout_s (new optional "
    "stage field from initialize()) instead of a bare subprocess.run, so a command-list stage can "
    "time out with only its own process group killed instead of hanging forever, and now records "
    "its own runner/heartbeat metadata via the same generic mechanism the external-dispatch path "
    "uses. schema_version UNCHANGED at 10; every new key is optional/additive. No role/action-set, "
    "recovery-taxonomy, protected-reference, or Judge change; scientific defaults untouched. "
    "heartbeat_stage also gained an optional pid kwarg (backfills the real OS pid once known, "
    "for a caller whose executor spawns its subprocess after begin_stage_execution already ran), "
    "and complete_external_stage now skips its own attempts increment when the stage already "
    "shows status=='running' (begin_stage_execution already counted that same attempt at "
    "dispatch time) -- a stage that arrives at complete_external_stage still 'pending' (the "
    "historical/direct-call shape) is counted exactly as before. "
    "v25 (R28 forensic-defect correction, part 2: wiring the pydantic_ai dispatch pipeline itself "
    "to the v24 lifecycle hooks) re-froze production_router.py and controller_bridge.py: both "
    "gained an additive on_dispatch_start passthrough (run_role/_accept_via_dispatch, "
    "dispatch_via_controller) mirroring the existing progress_cb passthrough exactly, used only by "
    "the producer_dispatch strategy; every other strategy ignores it. dispatch.py (not frozen) "
    "gained the same optional on_dispatch_start on authorize_and_execute, fired exactly once, "
    "immediately before the trusted executor is actually invoked -- i.e. only after role/"
    "capability/approval-boundary/param-validation/idempotency all pass -- never for a dry-run or "
    "any pre-executor rejection (DENIED, BLOCKED_CAPABILITY, APPROVAL_REQUIRED, param-INVALID, "
    "DUPLICATE); dispatch.py stays ignorant of the Controller, only invoking a generic callback. "
    "cli.py (not frozen) wires begin_stage_execution into that hook; every rejection branch that "
    "follows first checks whether the stage actually reached running (executor genuinely invoked, "
    "vs. a pre-executor rejection which leaves it pending exactly as before, unaffected). The "
    "PENDING (ExternalActionPending) branch calls the new defer_stage_execution (returns the stage "
    "to pending), preserving the pre-existing resumable-pause contract byte-for-byte. The "
    "status-not-in-{EXECUTED,DUPLICATE} rejection branch distinguishes a genuine TIMEOUT (a "
    "forward-compatible, generic check for a reason string whose leading exception name ends in "
    "TimeoutError -- inert until such an exception exists), which alone calls "
    "timeout_stage_execution followed by record_gate(..., FAIL, ...), returning the same "
    "GATE_{decision} shape the Judge-scored gate path already returns so run-campaign's existing "
    "recovery loop picks it up for free -- from every OTHER (ordinary, fast, synchronous) executor "
    "rejection, which instead calls defer_stage_execution back to pending and returns the same "
    "DISPATCH_REJECTED result as before, preserving the established ad-hoc direct run-stage "
    "retry-after-fixing-input contract rather than opening a pending_recovery for a non-hanging "
    "failure. The same running-check-then-defer_stage_execution pattern applies to a "
    "declared-outputs-missing rejection and to any exception complete_external_stage itself "
    "raises, in both cases preserving the original return value/exception unchanged. Either way, "
    "attempts and the stage_execution_started/stage_execution_deferred events remain a permanent "
    "record the attempt occurred; only whether status lands terminal or resumable differs. "
    "workflow/controller.py re-freezes for one additive method, defer_stage_execution (return a "
    "running stage to pending, recording a stage_execution_deferred event, without touching "
    "attempts) -- no other controller.py behavior changed in this revision. schema_version "
    "UNCHANGED at 10. No role/action-set, recovery-taxonomy, protected-reference, or Judge change; "
    "pre-executor rejection outcomes are byte-for-byte unaffected."
    "\n\n"
    "framework-v2-gate-integration re-freezes workflow/controller.py for the Framework V2 "
    "scientific-contract binding + gate enforcement. initialize() gains one additive state key, "
    "framework_v2 = {enabled: False, contracts: {}, stage_bindings: {}, scope_contract_sha256: "
    "None}; schema_version UNCHANGED at 10 (the key defaults via _v2_state() when absent, matching "
    "the established additive-field convention, so every pre-V2 manifest round-trips unchanged). "
    "record_gate gains a single pre-PASS call, _enforce_v2_gate_preconditions(name, bundle), whose "
    "result (when non-None) is attached to the gate event under a framework_v2 key; when V2 is "
    "disabled the method returns None immediately and the gate event is byte-for-byte identical to "
    "before. New additive methods only: _v2_state, v2_enabled, bind_v2_scope_contract, "
    "bind_v2_contract, _register_v2_contract, v2_contract, v2_stage_binding, "
    "_enforce_v2_gate_preconditions, _v2_judge_contradictions, _write_v2_gate_fact. When V2 is "
    "enabled and a convergence_policy is bound to a stage, record_gate refuses a PASS unless the "
    "committee convergence report is convergence_gate_ok (the R31 max-epoch-as-converged guard), "
    "and refuses a PASS when a vote bundle carrying v2_judgments + v2_facts contains a "
    "JUDGE_CONTRADICTION. All framework_v2 imports are lazy (inside the methods, guarded by "
    "ModuleNotFoundError) so the core-only install (no pydantic) is unaffected. No role/action-set, "
    "recovery-taxonomy, protected-reference, schema_version, or pre-V2 gate-outcome change."
    "\n\n"
    "effect-based-approval-boundary (C12 forensic-defect correction: the human-approval boundary "
    "was positioned one stage too early) re-freezes runtimes/pydantic_ai/actions.py to generalize "
    "resolve_action_approval_boundary from the v23 validation-only relaxation to a typed costly-EFFECT "
    "taxonomy. The boundary an action requires is now derived from the materially costly, "
    "non-trivially-reversible effects the proposal ACTUALLY performs -- never its action name or "
    "pipeline position. New additive module members only: COSTLY_EFFECT_BOUNDARY (teacher_inference-> "
    "costly_teacher_labeling, student_training->costly_training, production_md->production_md, "
    "scheduler_submission->scheduler_submission), _INHERENT_COSTLY_ACTIONS (label_with_teacher, "
    "train_committee, evaluate_heldout_fidelity, run_teacher_md, run_student_md, submit_scheduler_job "
    "-- their guarded effect is their defining purpose and can NEVER be relaxed by any declared "
    "parameter), and _incurs_teacher_inference_effect. costly_teacher_labeling is relaxed to None iff "
    "the proposal affirmatively proves it performs no Teacher inference / creates no new labels: for "
    "acquire_structures via the framework-injected typed performs_teacher_inference flag "
    "(cli._acquisition_incurs_teacher_inference deterministically classifies the ACTUAL bound recipe -- "
    "both built-in recipes drive the REAL Teacher calculator during structure generation so both -> True "
    "(augment-atoms's executor unconditionally binds the Teacher for perturbation/relaxation; teacher-md "
    "runs Langevin MD under the Teacher); arbitrary-adapter / unknown / missing / unreadable / non-dict "
    "-> True, fail-closed; only a recipe affirmatively proving a Teacher-free generator -> False), and "
    "for the v23 validation actions via the unchanged explicit "
    "dft_labels_used/protected_reference_labels_used False/False declaration. No boundary other than "
    "costly_teacher_labeling is ever relaxed; a missing/non-boolean declaration keeps the gate. "
    "APPROVAL_GATED_ACTIONS's dict values are unchanged; dispatch.py and cli.py (neither frozen) inject "
    "the trusted recipe-classified flag and wire the resolver in. No role/action-set membership, "
    "schema_version, recovery-taxonomy, protected-reference, or Judge change."
    "\n\n"
    "judge-reproducibility-provenance (C9) re-freezes runtimes/pydantic_ai/models.py to add four "
    "additive Optional fields to RuntimeInvocationRecord (temperature, seed, packet_sha256, "
    "decision_sha256), all defaulting to None on pre-closure and non-judge attempts. They record "
    "the sampling settings actually used for a judge attempt plus the canonical SHAs of the "
    "CanonicalReviewPacket reviewed and the ScientificDecisionRecord under review, making a judge "
    "attempt reproducible and letting an auditor confirm all three lenses reasoned over identical "
    "packet bytes. Purely additive; no existing field, role/action-set, or Judge-vote schema change."
    "\n\n"
    "judge-output-invalid-retry (C12F-blocker) re-freezes runtimes/pydantic_ai/production_router.py "
    "to add ONE additive Optional field, RouteResult.error_category (defaulting None). For the "
    "judge_gate strategy only, a hard schema/structural validation failure of a Judge vote (e.g. a "
    "REVISE/FAIL verdict with an empty required_fix, rejected by orchestration/exchange.py's "
    "validator) is tagged error_category='judge_output_invalid' so the gate can route it through the "
    "SAME canonical INVALID_JUDGE_OUTPUT bounded per-lens retry as a closure-invalid vote, instead "
    "of crashing the campaign. A provider/infrastructure failure keeps error_category=None and its "
    "existing fail-loud path. The retry/blocker loop itself lives in runtimes/pydantic_ai/cli.py "
    "(not a frozen file). No validator is weakened, no verdict semantics change, and every other "
    "field of production_router.py is untouched; no role/action-set or Judge-vote schema change."
    "\n\n"
    "evidence-gap-recovery-domain-guard (recovery-004 correction) re-freezes workflow/controller.py "
    "to add ONE additive fail-closed guard inside propose_recovery: after the existing training "
    "retrain/mode consistency check, RunController._validate_evidence_gap_recovery(resolved_code, "
    "labeling, training) is invoked. When -- and only when -- the plan's failure_category resolves "
    "(via the unchanged workflow.recovery_taxonomy registry) to the insufficient_evidence domain "
    "(evidence_gap / missing_evidence / unknown), the plan may not request student_training.retrain, "
    "labeling.new_dft, or labeling.teacher_relabel; any such request raises ValueError. Rationale: an "
    "insufficient-evidence diagnosis is an EVIDENCE-SURFACING gap, not a grounded model/label/data "
    "deficiency, so costly scientific corrective compute cannot correct the diagnosed cause (this is "
    "the exact recovery-004 defect: diagnosed evidence_gap / insufficient_evidence, routed to "
    "model_retrain / full retrain). The escape hatch for a genuinely grounded fidelity deficiency is "
    "to diagnose it under the model_fitting / student_fidelity domain instead, which the guard leaves "
    "untouched -- it is domain-scoped, never a blanket ban on retraining. Evidence-surfacing recoveries "
    "route to the evidence_repair capability (added to DEFAULT_RECOVERY_CAPABILITY_ROSTER -> "
    "orchestrator) for review-packet regeneration from already-computed artifacts. No existing "
    "validator is weakened, no taxonomy/roster entry is removed, no role/action-set/schema_version/"
    "Judge-vote change; every other propose_recovery path is byte-for-byte unchanged."
    "\n\n"
    "correction-regate-iteration (recovery-004 correction, evidence-surfacing re-gate) re-freezes "
    "workflow/controller.py to add ONE additive method, RunController.open_correction_iteration("
    "reason, authorized_by, regate_stage=None). It exists solely to RE-GATE a stage that already "
    "recorded a gate, after an AUDITED evidence-surfacing/framework correction rather than a "
    "scientific recovery -- the case the recovery path cannot serve, because start_iteration only "
    "bumps the iteration by activating an approved recovery (i.e. running corrective compute, which "
    "the recovery-004 directive forbids). A prior gate attempt leaves immutable, iteration-scoped "
    "Judge task packets (see _judge_task_id); re-deriving the same identity would collide and fail "
    "closed. open_correction_iteration supersedes the current iteration and appends a fresh one with "
    "a non-recovery trigger (kind='evidence_surfacing_correction', failed_stage=None) so neither the "
    "run-stage recovery-execution guard nor record_gate's recovery-resolution branch treats it as a "
    "recovery iteration, giving the corrected re-gate a DISTINCT iterNNN Judge identity while "
    "preserving the prior iteration's packets. It fails closed via _ensure_no_pending_recovery (a "
    "pending REVISE/FAIL must be resolved or superseded first, so it can never paper over an "
    "unaddressed gate) and, unlike start_iteration, invalidates NO artifacts and activates NO "
    "recovery: the already-accepted, lineage-identical artifacts are reused as-is and re-judged, not "
    "recomputed. It never records a verdict and never forces PASS; the three blind Judges still vote "
    "independently. No existing method is changed, no role/action-set/schema_version/Judge-vote "
    "change; every other RunController path is byte-for-byte unchanged."
    "\n\n"
    "r32-freeze-completion (R32 architecture-freeze completion; no code change, disclosure + hash "
    "sync only) formally re-freezes workflow/controller.py at its actual final "
    "framework_v2/scientific-runtime content. It records nothing new in the code; it CLOSES a "
    "disclosure gap discovered during the pre-fresh-run freeze audit, in which controller.py had "
    "been edited past the hash last recorded in this table while three real, test-covered, "
    "runtime-wired methods were never enumerated in any freeze or ledger record. (1) The "
    "framework-v2-gate-integration entry above claimed an exhaustive 'New additive methods only:' "
    "list that was INCOMPLETE; it is corrected here. The framework_v2 CLOSURE-DIRECTIVE REVIEW path "
    "adds three further additive controller methods, all part of the same V2 gate family and all "
    "inert until a V2 StageReviewSpec is bound and a v2_review bundle is supplied (so pre-closure "
    "gate behavior is byte-for-byte unchanged): "
    "bind_v2_stage_review_spec(stage, spec) -- a thin wrapper over bind_v2_contract that binds a "
    "frozen framework_v2.review_spec.StageReviewSpec (dict) to a stage under the 'stage_review_spec' "
    "role; _bound_stage_review_spec(stage) -- returns the bound StageReviewSpec as a pydantic object "
    "(or None when V2 closure review is not bound), used to validate each Judge vote against its own "
    "lens's predeclared criteria; and _enforce_v2_review(name, review_spec_sha, v2_review) -- called "
    "from _enforce_v2_gate_preconditions when a stage_review_spec is bound AND the vote bundle "
    "carries a v2_review, it deterministically validates the CanonicalReviewPacket + one JudgeReview "
    "per declared lens against the frozen StageReviewSpec via "
    "framework_v2.review_packet.validate_judge_review, refuses PASS on any INVALID_JUDGE_OUTPUT "
    "(a structurally invalid review is never a scientific vote) and requires unanimous-PASS lenses, "
    "and returns an audit dict recorded on the gate event. (2) For completeness of THIS frozen "
    "file's record, the other controller.py additions in this same working state are already "
    "verified and disclosed elsewhere and are only cross-referenced here, not re-introduced: the "
    "scientific-policy/adequacy methods bind_scientific_policy + _enforce_scientific_adequacy "
    "(wired into record_gate at PASS time) are FRAMEWORK_EVOLUTION_LEDGER FE-006/FE-007; the "
    "framework_v2 gate methods _v2_state, v2_enabled, bind_v2_scope_contract, bind_v2_contract, "
    "_register_v2_contract, v2_contract, v2_stage_binding, _enforce_v2_gate_preconditions, "
    "_v2_judge_contradictions, _write_v2_gate_fact are the framework-v2-gate-integration set above. "
    "(3) ONLY AFTER the three methods above are recorded, the FROZEN['workflow/controller.py'] hash "
    "is synced from the stale intermediate 3b817b67... to the actual final content hash "
    "75cd60974315b12c131b69033095226926d9a51acc8f18f01d71d322ddd0173b -- this is a re-record of an "
    "already-verified runtime state (full suite green, dedicated framework_v2_regression coverage of "
    "the closure-review path), not a change made to force the freeze test to pass. schema_version "
    "UNCHANGED at 10. No role/action-set, recovery-taxonomy, protected-reference, or Judge-vote "
    "SEMANTIC change; no frozen scientific policy changed; every other frozen file is untouched."
    "\n\n"
    "recovered-holdout-transitive-evidence-bind (FE-031) re-freezes workflow/controller.py "
    "75cd60974315b12c131b69033095226926d9a51acc8f18f01d71d322ddd0173b -> "
    "0f8c5b59f8051e2e2a0bb2a910cab9f4b6c4e3e3be8fc0d90ed67574b821487a for three ADDITIVE edits that "
    "make a recovered-original-holdout reference transitively bind its declared structures "
    "population as its own hash-verified Controller input: module-level "
    "EVIDENCE_STRUCTURE_REFERENCE_KINDS = frozenset({'recovered-original-holdout'}) + pure helper "
    "referenced_evidence_structure(source, *, project_dir) (returns None for anything that is not a "
    "YAML reference of that kind declaring a structures.path; else resolves + hash-verifies the "
    "structures file and returns (path, integrity); fail-closed FileNotFoundError on a missing file "
    "and ValueError on a declared-sha mismatch), wired into initialize() (auto-appends the "
    "structures file as a copy=False input, deduped) and bind_new_input() (same, post-init, with an "
    "input_bound event carrying auto_bound_structures_for). This does NOT weaken validate_evidence "
    "and does NOT make the evidence allowlist recursively trust arbitrary YAML paths -- it binds "
    "exactly the one declared, hash-verified structures population as an explicit input, identically "
    "to a human-authored copy:false structures input; references of any other kind are untouched. "
    "The pre-fix controller.py was reconstructed by removing exactly these three blocks and hashes "
    "to 75cd6097... (byte-identical to the prior pin), so this re-sync blesses the auto-bind fix as "
    "its SOLE delta -- no undocumented drift. schema_version UNCHANGED at 10. No role/action-set, "
    "recovery-taxonomy, protected-reference, gate, approval-boundary, or Judge change; no frozen "
    "scientific policy changed; every other frozen file is untouched."
    "\n\n"
    "recovered-holdout-split-manifest-transitive-bind (FE-032) re-freezes workflow/controller.py "
    "0f8c5b59f8051e2e2a0bb2a910cab9f4b6c4e3e3be8fc0d90ed67574b821487a -> "
    "7685fa4eec8c80dee6ee8db0c571776eb4e4e6890f3b2465d1be7ad02a0f4e69 for three ADDITIVE edits that "
    "make an evidence-bearing (recovered-original-holdout) reference transitively bind its declared "
    "source->split crosswalk manifest as its own hash-verified Controller input -- the exact same "
    "transitive-evidence invariant FE-031 applied to structures.path, now applied to the companion "
    "split_source_manifest the frames' per-frame lineage keys (source_category + source_local_index) "
    "join against. The generic Stage-2 (reference_validation) defect this closes was reproduced "
    "independently in runs ffv4g and ffv4h: the authoritative split manifest was declared in both "
    "reference.yaml (split_source_manifest + split_source_manifest_sha256) AND workflow.yaml "
    "(teacher_evidence_sources.split_source_manifest_paths) yet was never surfaced into "
    "build_split_crosswalk, so every recovered-holdout frame reported source_split_unjoined / domain "
    "'unknown' and the gate could not verify the split-membership/lineage criteria, forcing a "
    "unanimous 3x REVISE despite healthy Teacher-vs-DFT accuracy. The three edits: module-level pure "
    "helper referenced_split_manifest(source, *, project_dir) (returns None for anything that is not "
    "a YAML reference of an EVIDENCE_STRUCTURE_REFERENCE_KINDS kind declaring a split_source_manifest; "
    "else resolves + hash-verifies the manifest and returns (path, integrity); fail-closed "
    "FileNotFoundError on a missing manifest and ValueError on a declared-sha mismatch), wired into "
    "initialize() (auto-appends the manifest as a copy=False input, deduped) and bind_new_input() "
    "(same, post-init, with an input_bound event carrying auto_bound_split_manifest_for). It binds "
    "exactly the one declared, hash-verified manifest as an explicit input, identically to a "
    "human-authored copy:false input; references of any other kind are untouched; validate_evidence "
    "is NOT weakened and the evidence allowlist is NOT made recursively trusting. The pre-fix "
    "controller.py was reconstructed by removing exactly these three blocks and hashes to 0f8c5b59... "
    "(byte-identical to the prior pin), so this re-sync blesses the split-manifest auto-bind fix as "
    "its SOLE delta -- no undocumented drift. The companion Stage-2 evidence-readiness preflight + "
    "gate criterion-evidence surfacing (runtimes/pydantic_ai/reference_validation_readiness.py, NEW) "
    "and its wiring (runtimes/pydantic_ai/cli.py) and the RootCauseValidationError negated-DFT-prose "
    "false-positive fix (runtimes/pydantic_ai/root_cause.py) live entirely in NON-frozen files. "
    "schema_version UNCHANGED at 10. No role/action-set, recovery-taxonomy, protected-reference, "
    "gate, approval-boundary, or Judge change; no frozen scientific policy changed; every other "
    "frozen file is untouched."
    "\n\n"
    "stage10-11-deployment-producers (UNIT 3) re-freezes runtimes/pydantic_ai/actions.py "
    "5dadc1450d9f19fd99c7cd9c2884960ac8c786eb9e09a12564f519aaaf0e8e31 -> "
    "0e4b35cf3621baaabb8a972b16f694196834777fd20b615b93838d26ca3f5539 for a SINGLE ADDITIVE "
    "role/action-set delta: two deterministic READY simulation producers are appended to "
    "SIMULATION_ACTIONS -- resolve_deployment_checkpoint (derives the single canonical deployed "
    "Student checkpoint path+sha256 from the committee manifest by a governed seed selection, "
    "eliminating any hand-typed downstream checkpoint path) and build_deployment_context "
    "(derives the LAMMPS deployment context.yaml -- NVT production OR the dedicated microcanonical "
    "NVE energy-conservation segment -- purely from the frozen validation_profile shared_md_"
    "protocol, eliminating hand-tuned step counts). Both are backed by the new pure module "
    "validation/deployment_resolution.py (NOT frozen) and registered as light READY_EXECUTORs "
    "(READY_EXECUTOR count 33 -> 36 originally, now 37 including FE-066 build_stage10_deployment_plan). The companion Stage-11 NVE-log auto-consumption "
    "(_exec_build_physical_validation_report gains an optional nve_md_manifest param) and the "
    "physical_validation_report Judge adapter live entirely in NON-frozen files "
    "(executors.py, workflow/steps.py, bounded_evidence.py). No stage set, gate, recovery-taxonomy, "
    "capability-routing, protected-reference, schema_version, approval-boundary, or Judge-vote "
    "change; the two new actions are additive and inherit the default (non-gated) simulation "
    "boundary; every other frozen file is untouched."
    "\n\n"
    "teacher-physical-validation-target (FE-067) re-freezes runtimes/pydantic_ai/actions.py "
    "5646a67530d1d5fe44f6e163715448b7c677147240f11496adc7dc9032cb5ec3 -> "
    "f862236b1f1e3ec0ea300cd49a3bfe1c694799b17791b9f5dad7fd70b23cba38 for one ADDITIVE "
    "HPC/approval-gated simulation action, build_teacher_physical_validation_target. The action "
    "freezes a hash-bound TeacherValidationTarget for later Stage-11 Student reproduction and does "
    "not alter any existing action semantics, approval boundary, stage order, or Student training "
    "path. READY_HPC_APPROVAL_GATED count 8 -> 9; every other frozen file is untouched."
    "\n\n"
    "coverage-reacquisition-input-supersession (UNIT 4) re-freezes workflow/controller.py "
    "7685fa4eec8c80dee6ee8db0c571776eb4e4e6890f3b2465d1be7ad02a0f4e69 -> "
    "47ee90dd6138894cf14109a233c8405556b73f7c0015a75ede348526ac84a657 for ADDITIVE edits that make "
    "a coverage-deficit re-acquisition recovery EXECUTABLE (previously a confirmed dead-loop: a "
    "recovery returning to the acquisition stage reused the stale bound acquisition_plan.json, "
    "regenerating byte-identical candidates, which verify_recovery_execution rejects as 'did not "
    "change artifacts'). Three additive controller methods -- active_inputs() (the non-superseded "
    "bound inputs), supersede_input(matcher, *, reason, superseded_by=None) (marks a still-active "
    "bound input superseded via an input_superseded event; NEVER deletes the record or its "
    "content-addressed snapshot -- immutable lineage is preserved, the input is only excluded from "
    "active-input selection; fail-closed when no active input matches), and "
    "supersede_bound_acquisition_plan(*, reason) (retires the run's active *acquisition_plan.json "
    "input, [] when none) -- plus ONE additive block in start_iteration(): when a recovery's "
    "return_stage == 'acquisition' the stale plan is auto-superseded so the re-run re-plans with "
    "gap-driven sizing, and the superseded plan shas are recorded on the new iteration's trigger "
    "(superseded_acquisition_plans). A run with no autonomous planner and no fresh human plan then "
    "fails closed at the acquisition stage (PLAN_INPUT_REQUIRED) -- the correct fail-closed outcome, "
    "not a silent reuse. The two plan-identity resolvers that must honor supersession live in "
    "NON-frozen files (runtimes/pydantic_ai/cli.py _resolve_bound_acquisition_plan and "
    "runtimes/pydantic_ai/default_acquisition_provider.py _acquisition_plan_already_bound, both now "
    "skip superseded inputs), as do the Stage-4 deterministic coverage-adequacy determination and "
    "the data_coverage Judge adapter (executors.py, bounded_evidence.py). No existing controller "
    "method is changed except the single additive start_iteration block; no stage set, gate, "
    "recovery-taxonomy, capability-routing, protected-reference, schema_version, approval-boundary, "
    "or Judge-vote change; every other frozen file is untouched. "
    "fe042-stage4-coverage-adequacy-gate-control (FE-042) re-freezes workflow/controller.py "
    "5f510e590caa1f957a09a18c3260c9b7283b22a37f71b6c6b9ff9aac96415a42 -> "
    "11aa89390906353c83120908886482e593e9cca7e42f210be9ee528be13e7579 for a single additive "
    "deterministic Stage-4 gate control. The three data_coverage Judges review report HONESTY "
    "(access mode, per-config_type counts, lineage, protected-test exclusion), not "
    "configuration-space ADEQUACY, so after FE-041 made the report honest a truthful "
    "COVERAGE_INSUFFICIENT report can earn a unanimous 3/3 PASS while a declared deployment "
    "structure class still has ZERO acquired representatives -- the live ffv4r defect (a PASS gate "
    "would advance to Stage 5 teacher_labeling over structurally unsupported regions). record_gate "
    "gains ONE additive branch: a would-be PASS on a stage whose registered coverage report carries "
    "coverage_assessment.assessment_status == 'COVERAGE_INSUFFICIENT' is downgraded to a scientific "
    "REVISE that flows through the UNCHANGED invalidate_from + pending_recovery path, additionally "
    "recording a coverage_adequacy block (the unsupported declared classes read via the existing "
    "validation.coverage_gap_assessment.unsupported_structure_classes, and recommended_return_stage="
    "'acquisition') on both the gate event and pending_recovery. The determination is READ-ONLY over "
    "the status the FE-038/FE-039 gap gate already computed and wrote into the report -- it invents "
    "no threshold, quota, size, or coverage/acquisition science, and the new additive read-only "
    "helper _coverage_adequacy_control returns None (inert) for COVERAGE_SUFFICIENT / NOT_ASSESSABLE "
    "/ any stage with no coverage report, so every non-coverage gate is byte-for-byte unaffected. The "
    "gate-verdict-from-votes recomputation in _validate_vote_bundle, the vote bundle, and all PASS-only "
    "preconditions are UNCHANGED (the downgrade happens before them, exactly as any other non-PASS "
    "verdict). runtimes/pydantic_ai/cli.py (NOT a frozen file) gained only the additive surfacing of "
    "the pending coverage_adequacy block into the recovery Analyst's recovery_evidence context so the "
    "diagnosis routes to targeted reacquisition. schema_version UNCHANGED; no stage set, gate-vote, "
    "recovery-taxonomy, capability-routing, protected-reference, approval-boundary, or Judge change; "
    "every other frozen file is untouched. "
    "fe045-verify-inputs-superseded-source (FE-045) re-freezes workflow/controller.py "
    "11aa89390906353c83120908886482e593e9cca7e42f210be9ee528be13e7579 -> "
    "ca5dc016801b619991c3a0bd50f335ad0c5e21d33c924353f8d1d3d91218c339 for a single scoping change to "
    "verify_inputs(): the per-input SOURCE-byte re-check (verify_artifact(source, source_integrity), "
    "which raises 'declared workflow input changed after initialization' on mismatch) is now gated "
    "behind `if not record.get('superseded')`, mirroring the existing active/superseded semantics of "
    "active_inputs(). A superseded acquisition_plan input and its superseding plan legitimately share "
    "ONE mutable source path (acquisition/plans/<run_id>.acquisition_plan.json -- the canonical "
    "Stage-3 planner writes a fixed filename), so binding the superseding plan overwrites that source "
    "with the new bytes; re-verifying the shared mutable source against a superseded record's frozen "
    "historical bytes turned a legitimate supersession into a false invariant failure that blocked the "
    "acquisition re-gate immediately after a corrective reacquisition executed (the live ffv4t-eng1 "
    "blocker). The snapshot-integrity check above the gate is UNCHANGED and still fail-closes EVERY "
    "input (superseded or active) against its immutable content-addressed snapshot, so frozen-provenance "
    "integrity is fully preserved; source-byte equality remains mandatory and fail-closed for every "
    "ACTIVE input. No stage set, gate-vote, recovery-taxonomy, capability-routing, protected-reference, "
    "schema_version, approval-boundary, threshold, or Judge change; no science, acquisition, coverage, "
    "or recovery logic touched; every other frozen file is untouched. "
    "FE-064 recovery-proposal-supersession re-freezes workflow/controller.py "
    "2cb69859af792d5d7007c9808abd1ad7a8da53c0883055f7718fd9d0ebb5fac0 -> "
    "ca5dc016801b619991c3a0bd50f335ad0c5e21d33c924353f8d1d3d91218c339 to add the "
    "minimal proposed->superseded recovery lifecycle transition, plus a CLI wrapper. The transition "
    "requires structured provenance, keeps the proposal history immutable apart from terminal "
    "lifecycle metadata, emits recovery_superseded, clears pending_recovery, and does not approve, "
    "activate, execute, or mark the recovery resolved. No stage set, gate-vote, scientific threshold, "
    "protected-reference, or Judge behavior changed; every other frozen file is untouched. "
    "FE-049 (recovery corrective-action parameter-contract validation + species-mapping exposure "
    "action) re-freezes runtimes/pydantic_ai/actions.py: it deliberately grows the role/action-set, "
    "ADDING ONE new backed data-curator action_type, 'validate_species_mapping_consistency' (a READY "
    "deterministic executor that exposes a Teacher labeling manifest's concrete element->type-index "
    "species mapping and fails closed unless every independently-sourced mapping agrees and is "
    "attested), so an evidence-exposure REVISE on teacher_labeling has a corrective action that can "
    "converge. Same additive change class as v19/v21/v23 and stage10-11-deployment-producers (UNIT 3). "
    "The rest of FE-049 -- acceptance-time parameter-contract validation that rejects a "
    "corrective_action whose parameters would KeyError at dispatch -- is in the NON-frozen "
    "runtimes/pydantic_ai/{recovery_bridge,cli,executors,deterministic_executors}.py. No stage set, "
    "gate-vote, recovery-taxonomy, capability-routing, protected-reference, schema_version, "
    "approval-boundary, threshold, or Judge change; every other frozen file, and every other field of "
    "actions.py, is untouched."
)
FROZEN_MODEL = "qwen2.5-7b-instruct"

FROZEN = {
    "runtimes/pydantic_ai/criterion_eval.py":
        "65a1c4fd5560660eef4825b4f4aa0687868cda7e4e6e863c19ec2b8923d12b96",
    "agents/judge.md":
        "cc32f81efdbf825067f2688eb78a2f41982f9183ef78a33badb31906dabc8aa8",
    "orchestration/exchange.py":
        "be100a2525cc8b64c680462aa7fde2ef74d547223e84cb1ece2f081484a23669",
    "runtimes/pydantic_ai/tool_registry.py":
        "3d398a718da1c9e89d03585acdc9fafcfeb2d4767569ffac6027edcd13c1e467",
    "runtimes/pydantic_ai/models.py":
        "9ea323cdcafd552aaa78d25945da64cba0c1ad7e76c144c2e37abb4b46d2d884",
    "runtimes/pydantic_ai/pydantic_ai_runtime.py":
        "81679cccd610f56cb07fb1f70b388286bd456cf96da2c1a2cdb48d4eed25aa62",
    "runtimes/pydantic_ai/role_outputs.py":
        "a1d5ba42801907e621efcba70d5dba4b93b659aa87e0dd4224acfcd69e00d6db",
    "runtimes/pydantic_ai/production_router.py":
        "7f23de2594fa9533f5c1445367bb6accac890e10e502358ce162431474ee8525",
    "runtimes/pydantic_ai/driver.py":
        "db480c20d126b7511e8bbaa4fc2018adb56aa789fabe496ba4f08313379f5939",
    "runtimes/pydantic_ai/actions.py":
        "f862236b1f1e3ec0ea300cd49a3bfe1c694799b17791b9f5dad7fd70b23cba38",
    "runtimes/pydantic_ai/controller_bridge.py":
        "91432692ae394da3b526b9d61b3c6743f34ec74c81f874b04494c96425f7f500",
    "orchestration/specs.py":
        "4b6dc829fe2b6b594cc87e8a62bd944ea9df181cd7f420ae3732c861ce8e43cb",
    "workflow/controller.py":
        "ca5dc016801b619991c3a0bd50f335ad0c5e21d33c924353f8d1d3d91218c339",
    "orchestration/schema/agent_result.schema.json":
        "a38afea9c06c21e647376efd835dec32a16b2f247583a090560cb1843e0eda31",
    "orchestration/schema/agent_spec.schema.json":
        "8b59189f55f72a3c5853093ec88c3284d353475a322c50ca55403bbf5282151b",
    "orchestration/schema/agent_task.schema.json":
        "60d3d49c33c85107830c237cbcc6db23b9c30225990cad7c6f152337f57ce0a5",
    "orchestration/schema/judge_vote.schema.json":
        "682ade03213da8483d2089ed21f34081be612b01c4df615f1ae6facbc4ea18df",
    "agent_specs/analyst.yaml":
        "7fac8bc650f2b06d689327a206cf0802bc0e1cf9351dbd4f44e5f315d2820306",
    "agent_specs/data-curator.yaml":
        "7500edebd058b82be6bb6ca048c5f9cb7136f3440cf12e3161f7b450342ff774",
    "agent_specs/judge.yaml":
        "94727231c06c51daf5f400867454a273c4600fc96f7ab10b7b5df11e52f8fd5d",
    "agent_specs/literature.yaml":
        "7709db297c330083abe4a396a7374e5f9fd4bfdaba62950f72094f24993f9368",
    "agent_specs/ml-trainer.yaml":
        "17ecf5a3244f8e0e798d4f27aa6995d55fe31d5a14741c6f06e6f033e3e9b8ad",
    "agent_specs/orchestrator.yaml":
        "37bb4da36e5075f7f94cb0008a44d07ef7c600db8da48a3b247aaf8b815c3950",
    "agent_specs/simulation.yaml":
        "6b9fd7de3a5248b50622a3c33950411e3864c6b9ef1d28b17f7a27b19d7544e9",
}


class ArchitectureFreezeTests(unittest.TestCase):
    def test_frozen_files_unchanged(self):
        drift = []
        for rel, want in FROZEN.items():
            p = ROOT / rel
            self.assertTrue(p.is_file(), f"frozen file missing: {rel}")
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want:
                drift.append(f"{rel}: {got} != {want}")
        self.assertEqual(drift, [], "frozen architecture files changed:\n" + "\n".join(drift))

    def test_request_limit_default_is_six(self):
        # request_limit=6 is part of the freeze; assert the default constant is intact.
        # pydantic is an optional runtime dep; on the core-only install skip (like every other
        # pydantic-ai test) instead of erroring — the assertion still runs in the pydantic-ai jobs.
        try:
            from runtimes.pydantic_ai.models import RuntimeContext
        except ModuleNotFoundError:
            self.skipTest("pydantic (optional runtime dep) not installed")
        self.assertEqual(RuntimeContext.model_fields["request_limit"].default, 6)


    def test_production_wiring_keeps_frozen_architecture_dimensions(self):
        from runtimes.pydantic_ai.actions import ROLE_ALLOWED_ACTIONS
        from workflow.controller import RunController
        roles = {"orchestrator", "literature", "data-curator", "ml-trainer", "simulation", "analyst", "judge"}
        import json
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "freeze-dimensions", "stages": [{
                "name": "s", "command": None,
                "contract": {"kind": "validation_manifest", "manifest": "m.json", "validator": "validation.report.validate_validation_report"},
            }]}))
            c = RunController.initialize(cfg, root / "run")
        self.assertEqual(set(ROLE_ALLOWED_ACTIONS), {"data-curator", "ml-trainer", "simulation", "analyst"})
        self.assertIn("build_teacher_baseline", ROLE_ALLOWED_ACTIONS["simulation"])
        self.assertIn("validate_teacher_reference", ROLE_ALLOWED_ACTIONS["simulation"])
        self.assertIn("acquire_structures", ROLE_ALLOWED_ACTIONS["data-curator"])
        self.assertNotIn("acquire_structures", ROLE_ALLOWED_ACTIONS["simulation"])
        self.assertNotIn("validate_teacher_reference", ROLE_ALLOWED_ACTIONS["data-curator"])
        self.assertNotIn("validate_teacher_reference", ROLE_ALLOWED_ACTIONS["ml-trainer"])
        self.assertEqual(len(roles), 7)
        self.assertEqual(c.stage("s")["contract"]["kind"], "validation_manifest")

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
