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
    "is untouched."
)
FROZEN_MODEL = "qwen2.5-7b-instruct"

FROZEN = {
    "runtimes/pydantic_ai/criterion_eval.py":
        "65a1c4fd5560660eef4825b4f4aa0687868cda7e4e6e863c19ec2b8923d12b96",
    "agents/judge.md":
        "cc32f81efdbf825067f2688eb78a2f41982f9183ef78a33badb31906dabc8aa8",
    "orchestration/exchange.py":
        "f29362891c18c728dc9de0a8c3ee51590c9be6be53f5b811bdfd861d0e82a8ae",
    "runtimes/pydantic_ai/tool_registry.py":
        "3d398a718da1c9e89d03585acdc9fafcfeb2d4767569ffac6027edcd13c1e467",
    "runtimes/pydantic_ai/models.py":
        "0854b16133dc7b42cf1618a076ec4b30793f499cc212d1f897f282e4bb74b8dc",
    "runtimes/pydantic_ai/pydantic_ai_runtime.py":
        "81679cccd610f56cb07fb1f70b388286bd456cf96da2c1a2cdb48d4eed25aa62",
    "runtimes/pydantic_ai/role_outputs.py":
        "2ca372aea8a105504fa8ed08c9e2fe47b04733564046b788b92d586139e4708a",
    "runtimes/pydantic_ai/production_router.py":
        "85cd6e73eabeb3f03ea29cc809f5d00e0f494b3bf6e37f5b205bc9875b91e65a",
    "runtimes/pydantic_ai/driver.py":
        "571636918a2827ceded12e9ee3b0cad7f23ab73887d61ed0cc2b6d5727986719",
    "runtimes/pydantic_ai/actions.py":
        "c63f8d42bf208f87c2b7d220264e27556fd3ce79ac5d48c13becb0557c66c141",
    "runtimes/pydantic_ai/controller_bridge.py":
        "b046922cd32620cd5dbe1c2dc7b4390a2eb67b4201e473d9b6062c68f8bd869d",
    "orchestration/specs.py":
        "4b6dc829fe2b6b594cc87e8a62bd944ea9df181cd7f420ae3732c861ce8e43cb",
    "workflow/controller.py":
        "ecfa32fcf1743a65dc3c35f77fff9aa7f3880da183d843efa9cfd90b211e194c",
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
