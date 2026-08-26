"""Production CLI for the PydanticAI runtime (Phase 2/D6).

Runs ONE task through a runtime and the existing validation pipeline, with explicit modes and
meaningful exit codes. A real provider call happens only in ``--runtime pydantic-ai`` mode AND
only when credentials preflight READY; with no credential the CLI exits PROVIDER_UNAVAILABLE
without contacting any provider (a missing key is never a silent success).

Usage:
    python -m runtimes.pydantic_ai.cli run-task \
        --runtime pydantic-ai --agent judge --agent-specs-dir agent_specs \
        --task task.json --exchange-dir runs/x/exchange --mode shadow

    python -m runtimes.pydantic_ai.cli run-task \
        --runtime mock --agent judge --agent-specs-dir agent_specs \
        --task task.json --exchange-dir /tmp/ex --mock-response resp.json --mode validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .events import (CampaignEventEmitter, campaign_previously_executed, stage_progress_fields,
                     terminal_class)

# Meaningful, distinct exit codes.
EXIT_SUCCESS = 0
EXIT_VALIDATION_REJECTED = 2
EXIT_PROVIDER_UNAVAILABLE = 3
EXIT_PROVIDER_FAILURE = 4
EXIT_APPROVAL_REQUIRED = 5
EXIT_BLOCKED_POLICY = 6
EXIT_DUPLICATE = 7
EXIT_INTERNAL = 8
EXIT_RECOVERY_REQUIRED = 9
EXIT_RECOVERY_EXECUTION_UNVERIFIED = 10
EXIT_RECOVERY_ACTION_PENDING = 11
EXIT_EXTERNAL_ACTION_PENDING = 12

_PROVIDER_UNAVAILABLE_FAILURES = {"authentication_failure"}
MAX_PRODUCER_GENERATION_ATTEMPTS = 3
MAX_JUDGE_EVIDENCE_PACKET_BYTES = 128 * 1024
MAX_PRODUCER_EVIDENCE_PACKET_BYTES = 128 * 1024
DEFAULT_CONTEXT_WINDOW_TOKENS = 8192
DEFAULT_OUTPUT_TOKEN_RESERVE = 1024
DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS = 512
PRODUCER_CONTEXT_BUDGET_EXCEEDED = "PRODUCER_CONTEXT_BUDGET_EXCEEDED"
PRODUCER_CONTEXT_WINDOW_UNDECLARED = "PRODUCER_CONTEXT_WINDOW_UNDECLARED"
_AUTH_BINDING_MISMATCH = "authoritative action binding mismatch"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pydantic-ai-runtime", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="verify run inputs and build bounded evidence")
    preflight.add_argument("--run-dir", required=True)
    approve = sub.add_parser("approve", help="record explicit human approval for an action boundary")
    approve.add_argument("--run-dir", required=True)
    approve.add_argument("--boundary", required=True)
    approve.add_argument("--note", required=True)
    approve.add_argument("--plan-sha256", default=None,
                         help="optional exact AcquisitionPlan SHA256 binding for acquire_structures")
    approve.add_argument("--action-type", default=None,
                         help="exact action this approval authorizes (e.g. label_with_teacher); "
                              "binds the grant so it cannot authorize any other action")
    stage = sub.add_parser("run-stage", help="run one production-routed stage")
    stage.add_argument("--run-dir", required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--runtime", choices=("mock", "pydantic-ai"), required=True,
                       help="required; use pydantic-ai for production, mock only in tests")
    stage.add_argument("--mock-response", default=None)
    stage.add_argument("--agent-specs-dir", default="agent_specs")
    stage.add_argument("--exchange-dir", default=None)
    stage.add_argument("--repo-root", default=".")
    stage.add_argument("--auto-mock-judges", action="store_true",
                       help="test-only: create three PASS judge votes from frozen evidence")
    stage.add_argument("--mock-judge-response", action="append", default=[],
                       help="test-only: one canned JudgeVote JSON per judge context")
    stage.add_argument("--quiet", action="store_true",
                       help="suppress console progress output (durable event log is unaffected)")
    stage.add_argument("--json-events", action="store_true",
                       help="stream progress events as JSON lines instead of human-readable text")
    campaign = sub.add_parser(
        "run-campaign",
        help="drive a run forward across every eligible stage until completion or a pause state")
    campaign.add_argument("--run-dir", required=True)
    campaign.add_argument("--runtime", choices=("mock", "pydantic-ai"), required=True,
                          help="required; use pydantic-ai for production, mock only in tests")
    campaign.add_argument("--mock-response", default=None,
                          help="test-only: shared canned producer response (usually omit and let "
                               "each stage self-generate its deterministic mock proposal)")
    campaign.add_argument("--agent-specs-dir", default="agent_specs")
    campaign.add_argument("--exchange-dir", default=None)
    campaign.add_argument("--repo-root", default=".")
    campaign.add_argument("--auto-mock-judges", action="store_true",
                          help="test-only: create three PASS judge votes from frozen evidence")
    campaign.add_argument("--mock-judge-response", action="append", default=[],
                          help="test-only: one canned JudgeVote JSON per judge context")
    campaign.add_argument("--mock-analyst-response", default=None,
                          help="test-only: canned RootCauseClassification JSON for a pending "
                               "recovery diagnosis (required with --runtime mock if one is "
                               "pending)")
    campaign.add_argument("--mock-orchestrator-response", default=None,
                          help="test-only: canned RecoveryPlanProposal JSON for a pending "
                               "recovery plan (required with --runtime mock if one is pending)")
    campaign.add_argument("--quiet", action="store_true",
                          help="suppress console progress output (durable event log is unaffected)")
    campaign.add_argument("--json-events", action="store_true",
                          help="stream progress events as JSON lines instead of human-readable text")
    bind_closure = sub.add_parser(
        "bind-closure",
        help="bind the Framework-V2 closure review contracts (DeploymentScope + one frozen "
             "StageReviewSpec per stage) onto a run so record_gate enforces the CanonicalReviewPacket "
             "/ JudgeReview path")
    bind_closure.add_argument("--run-dir", required=True)
    bind_closure.add_argument("--scope-contract", required=True,
                              help="path to the DeploymentScopeContract JSON to bind as the run's "
                                   "single scope source of truth")
    bind_closure.add_argument("--stage", action="append", default=[],
                              help="canonical stage value to bind a default StageReviewSpec to; "
                                   "repeatable (omit to bind all stages that have a gate)")
    bind_closure.add_argument("--validation-profile-version", type=int, default=1)
    bind_closure.add_argument("--convergence-policy", default=None,
                              help="path to a ConvergencePolicy JSON to bind on stages whose "
                                   "StageReviewSpec requires convergence_report evidence; when "
                                   "omitted the framework default policy is bound so the "
                                   "convergence gate is never silently skipped")
    bind_sci = sub.add_parser(
        "bind-scientific-policies",
        help="attach typed scientific-adequacy policies to a run BEFORE any relevant Student "
             "evidence exists; the framework refuses to overwrite an existing binding with a "
             "different content-hash (idempotent on identical content).")
    bind_sci.add_argument("--run-dir", required=True)
    bind_sci.add_argument("--policy-file", action="append", default=[], required=True,
                          metavar="STAGE:KIND:PATH",
                          help="repeatable; e.g. 'evaluation:EvaluationAdequacyPolicyV2:candidates/1.json'")
    bind_sci.add_argument("--source-ref", default="operator-bound-at-init",
                          help="provenance reference recorded with every binding")
    bind_sci.add_argument("--allow-not-required", action="store_true",
                          help="bind with required=False (advisory only, does not block gate PASS); "
                               "default is required=True for canonical campaign use")
    approve_recovery = sub.add_parser(
        "approve-recovery", help="record explicit human approval for a proposed recovery")
    approve_recovery.add_argument("--run-dir", required=True)
    approve_recovery.add_argument("--approved-by", required=True)
    approve_recovery.add_argument("--note", default=None)
    plan_teacher_validation = sub.add_parser(
        "plan-teacher-validation",
        help="manual/debug: run the autonomous Teacher-validation planning step outside "
             "run-campaign's automatic pre-Stage-1 invocation (write-once; a no-op if a plan is "
             "already committed)")
    plan_teacher_validation.add_argument("--run-dir", required=True)
    plan_teacher_validation.add_argument("--runtime", choices=("mock", "pydantic-ai"),
                                        required=True)
    plan_teacher_validation.add_argument("--agent-specs-dir", default="agent_specs")
    plan_teacher_validation.add_argument("--exchange-dir", default=None)
    plan_teacher_validation.add_argument("--repo-root", default=".")
    plan_teacher_validation.add_argument(
        "--mock-orchestrator-response", default=None,
        help="test-only: canned TeacherValidationPlanProposal JSON (required with --runtime "
             "mock); a comma-separated list simulates the bounded semantic-correction retry, one "
             "file per attempt, holding on the last file once exhausted")
    authorize_downstream_reliance = sub.add_parser(
        "authorize-downstream-teacher-reliance",
        help="record explicit human approval for costly downstream reliance (Teacher labeling / "
             "Student training) on a committed Teacher validation plan that lacks predictive-"
             "fidelity evidence")
    authorize_downstream_reliance.add_argument("--run-dir", required=True)
    authorize_downstream_reliance.add_argument("--authorized-by", required=True)
    authorize_downstream_reliance.add_argument("--note", default=None)
    augment_train = sub.add_parser(
        "augment-train",
        help="FE-054: out-of-band post-split TRAIN-only augmentation action (run AFTER Stage-6 "
             "dataset_split PASS and BEFORE Stage-7 training). Plans an autonomous AugmentationPlan "
             "over the frozen TRAIN parents (protected augmentation_parents excluded) and, with "
             "--execute, runs the costly_teacher_labeling generation+labeling+merge into "
             "final_train.extxyz")
    augment_train.add_argument("--run-dir", required=True)
    augment_train.add_argument("--runtime", choices=("mock", "pydantic-ai"), required=True)
    augment_train.add_argument("--agent-specs-dir", default="agent_specs")
    augment_train.add_argument("--exchange-dir", default=None)
    augment_train.add_argument("--repo-root", default=".")
    augment_train.add_argument("--train-dataset", default=None,
                               help="frozen Stage-6 labeled TRAIN parents (default: "
                                    "{run_dir}/artifacts/dataset/train.extxyz)")
    augment_train.add_argument("--base-label-manifest", default=None,
                               help="Stage-5 teacher_labeling manifest for the TRAIN parents "
                                    "(required with --execute)")
    augment_train.add_argument("--teacher-config", default=None,
                               help="Teacher calculator config (required with --execute); default "
                                    "resolved from the run's bound workflow teacher_config")
    augment_train.add_argument("--reference-yaml", default=None,
                               help="protected-reference YAML for merge protection (required with "
                                    "--execute); default resolved from the run's bound reference")
    augment_train.add_argument(
        "--execute", action="store_true",
        help="cross the costly_teacher_labeling boundary: run augment_atoms generation + Teacher "
             "labeling + merge into final_train.extxyz (requires PYDANTIC_AI_SMOKE_CONFIRM=yes for "
             "a warranted, Teacher-driving plan)")
    augment_train.add_argument(
        "--mock-acquisition-response", default=None,
        help="test-only: canned AcquisitionPlanProposal JSON (required with --runtime mock); a "
             "comma-separated list simulates the bounded semantic-correction retry")
    r = sub.add_parser("run-task", help="run one task through the runtime")
    r.add_argument("--runtime", choices=("mock", "pydantic-ai"), required=True)
    r.add_argument("--agent", required=True, help="agent/role name (spec basename)")
    r.add_argument("--agent-specs-dir", default="agent_specs")
    r.add_argument("--task", required=True, help="path to the task JSON")
    r.add_argument("--exchange-dir", required=True)
    r.add_argument("--run-dir", default=None,
                   help="controller run dir (required for producer roles' dispatch)")
    r.add_argument("--repo-root", default=".")
    r.add_argument("--read-allow", action="append", default=[],
                   help="read-only allow-list prefix (repeatable)")
    r.add_argument("--provider", default=None,
                   help="provider kind: local-openai | ollama | anthropic | openai; "
                        "else $PYDANTIC_AI_PROVIDER")
    r.add_argument("--model", default=None, help="provider model id, else $PYDANTIC_AI_MODEL")
    r.add_argument("--base-url", default=None,
                   help="[local] OpenAI-compatible base URL, else $PYDANTIC_AI_BASE_URL")
    r.add_argument("--probe-server", action="store_true",
                   help="[local] TCP-probe the server during preflight (no inference)")
    r.add_argument("--mode", choices=("primary", "shadow", "dry-run", "validate-only"),
                   default="shadow")
    r.add_argument("--mock-response", default=None,
                   help="[--runtime mock] file with the canned raw response JSON")
    r.add_argument("--correlation-id", default="")
    return p


def _print_kv(out, **kw):
    for k, v in kw.items():
        print(f"{k}: {v}", file=out)



def _stage_config(controller, stage_name):
    import yaml
    cfg = yaml.safe_load(Path(controller.state["workflow_config"]).read_text()) or {}
    for stage in cfg.get("stages", []):
        if stage.get("name") == stage_name:
            return stage
    return {}


def _split_membership_manifest_sources(controller):
    """Return the resolved paths of the run's authoritative source->split crosswalk manifest(s):
    every ``teacher_evidence_sources.split_source_manifest_paths`` entry the run declared at init,
    plus every bound Controller input that a ``recovered-original-holdout`` (or other evidence-
    bearing) reference auto-bound as its split manifest (input event ``auto_bound_split_manifest_for``).

    These are surfaced (lineage-only) in a stage's bounded evidence so ``build_split_crosswalk`` can
    join every recovered held-out frame's per-frame lineage keys (``source_category`` +
    ``source_local_index``) against authoritative split membership -- the missing link that made
    reference_validation's frames all report source-split-unjoined / domain unknown despite carrying
    the join keys (see FE-032). Generic: driven by the ``teacher_evidence_sources`` /
    evidence-bearing-reference provenance contract, never a material/filename or a hardcoded stage.
    A frame that carries no lineage keys simply does not join -- surfacing the crosswalk is inert
    for such a stage. Deduplicated by resolved path; only paths that exist on disk are returned."""
    paths = []
    seen = set()

    def _add(raw):
        if not raw:
            return
        resolved = str(Path(raw).resolve())
        if resolved in seen or not Path(resolved).exists():
            return
        seen.add(resolved)
        paths.append(resolved)

    sources = controller.state.get("teacher_evidence_sources") or {}
    for declared in sources.get("split_source_manifest_paths", []) or []:
        _add(declared)
    auto_bound = {
        e.get("source") for e in controller.state.get("events", [])
        if e.get("type") == "input_bound" and e.get("auto_bound_split_manifest_for")}
    for record in controller.state.get("inputs", []):
        if record.get("source") in auto_bound:
            _add(record.get("snapshot") or record.get("source"))
    return paths


def _reference_validation_readiness(controller, proposal, *, report_path=None):
    """Deterministic criterion-evidence record for a Teacher-vs-DFT reference_validation proposal, or
    None when the proposal is not such an action (see reference_validation_readiness). Passes the
    run's authoritative source->split crosswalk manifest(s) so the lineage-join criterion is
    computed against real split membership. Used pre-execution (fail-closed preflight, report_path
    absent) and post-execution (gate-packet criterion surfacing, report_path present)."""
    from .reference_validation_readiness import compute_reference_validation_evidence
    return compute_reference_validation_evidence(
        controller, proposal, split_manifest_paths=_split_membership_manifest_sources(controller),
        report_path=report_path)


def _acquisition_readiness(controller, proposal, *, report_path=None):
    """Deterministic criterion-evidence record for an ``acquire_structures`` proposal, or None when
    the proposal is not that action (see acquisition_readiness). Surfaces the parent->pool join,
    per-parent deployment-domain mapping (resolved from the run's OWN bound frozen
    scope-classification evidence), and selection-control attestations into the acquisition gate
    packet so a Judge can VERIFY each criterion against deterministic evidence rather than infer it
    from the raw manifest. Evidence surfacing only: never re-selects structures or changes any
    AcquisitionPlan field."""
    from .acquisition_readiness import compute_acquisition_evidence
    return compute_acquisition_evidence(controller, proposal, report_path=report_path)


def _species_mapping_gate_evidence(controller, stage_name, declared):
    """FE-053: deterministic element/species -> 0-based model-type-index mapping criterion-evidence
    for a Teacher-labeling gate, or ``None`` when no declared stage output records a
    ``species_mapping_evidence`` block (a no-op for every non-labeling stage, and for any labeling
    manifest predating that field). Mirrors ``_reference_validation_readiness`` /
    ``_acquisition_readiness``: it surfaces the EXACT ordered mapping, its attestation, and the
    cross-source agreement -- hash-bound to the manifest sha256 -- into the gate packet's
    ``validation_outcomes`` so a Judge can VERIFY the mapping criterion against an authoritative
    deterministic result instead of only observing that a ``species_mapping_evidence`` field
    exists. Evidence surfacing only: it never relabels, re-selects, or mutates any artifact, and it
    reuses the FE-049 deterministic ``validate_species_mapping_consistency`` cross-check verbatim
    (called WITHOUT ``out_path`` so nothing is written). A genuine non-attestation or cross-source
    conflict is surfaced as a failed criterion (``ready``/``agree`` False + the reason) so the gate
    REVISEs on explicit deterministic evidence rather than crashing during packet assembly --
    fail-closed semantics preserved.

    The labeling manifest is discovered GENERICALLY as whichever declared JSON output records a
    ``species_mapping_evidence`` dict -- never a hardcoded filename or stage concept.
    """
    from .deterministic_executors import (validate_species_mapping_consistency,
                                          _ValidationFailure)
    from adapters.teacher import SpeciesMappingConflictError
    from workflow.integrity import sha256_file
    manifest_path = None
    for art in declared:
        path = Path(art)
        if path.suffix != ".json" or not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("species_mapping_evidence"), dict):
            manifest_path = path
            break
    if manifest_path is None:
        return None
    proposal = {"parameters": {"manifest_path": str(manifest_path)}}
    try:
        metrics = validate_species_mapping_consistency(proposal).get("metrics", {})
    except (_ValidationFailure, SpeciesMappingConflictError) as exc:
        try:
            detail = json.loads(str(exc))
        except ValueError:
            detail = {"reason": str(exc)}
        return {"stage": stage_name, "kind": "species_mapping_criterion_evidence",
                "ready": False, "attested": bool(detail.get("attested", False)), "agree": False,
                "blocking_gaps": [detail.get("reason", "species_mapping_invalid")],
                "manifest_path": str(manifest_path),
                "manifest_sha256": detail.get("manifest_sha256") or sha256_file(manifest_path),
                "species_to_type_index_map": detail.get("species_to_type_index_map"),
                "sources_cross_checked": detail.get("sources_cross_checked", [])}
    return {"stage": stage_name, "kind": "species_mapping_criterion_evidence",
            "ready": bool(metrics.get("ok")), "attested": bool(metrics.get("attested")),
            "agree": True, "blocking_gaps": [],
            "manifest_path": metrics.get("manifest_path"),
            "manifest_sha256": metrics.get("manifest_sha256"),
            "species_to_type_index_map": metrics.get("species_to_type_index_map"),
            "declared_config_map": metrics.get("declared_config_map"),
            "runtime_map": metrics.get("runtime_map"),
            "compiled_model_map": metrics.get("compiled_model_map"),
            "sources_cross_checked": metrics.get("sources_cross_checked", [])}


def _selective_provenance_inputs(controller, stage_name):
    """Return the small, explicitly-declared set of provenance-only run INPUT paths that
    ``stage_name``'s bounded-evidence assembly should see, beyond its own registered stage output
    artifacts (``c.state["artifacts"]``, which is all ``run_production_stage`` bound before this
    existed).

    Generic over the declaring block's shape, never a hardcoded manifest/stage name: any top-level
    workflow-config key ending in ``_provenance`` whose value is a dict declares its own scope --
    ``applies_to_stage`` (matched against ``stage_name``) and ``bound_evidence_input_indices``
    (the ONLY indices from that block actually added; a block may reference other, purely
    documentary input indices -- e.g. a human-readable provenance record -- that it deliberately
    leaves out of this list). Each index is looked up in ``controller.state["inputs"]`` (the full
    run input roster -- see ``RunController.initialize``). This is deliberately NOT
    ``_cmd_preflight``'s "every input, unfiltered" approach: only inputs a block explicitly opts
    in, by index, are ever added. A block whose declared ``role`` names a run-declared
    ``protected_reference_roles`` entry is skipped entirely, so protected-reference data can never
    reach a stage's Judge through this path.
    """
    import yaml
    cfg = yaml.safe_load(Path(controller.state["workflow_config"]).read_text()) or {}
    protected_roles = set(controller.state.get("protected_reference_roles", []))
    inputs = controller.state.get("inputs", [])
    paths = []
    for key, block in cfg.items():
        if not key.endswith("_provenance") or not isinstance(block, dict):
            continue
        if block.get("applies_to_stage") != stage_name:
            continue
        if block.get("role") in protected_roles:
            continue
        indices = block.get("bound_evidence_input_indices")
        if not isinstance(indices, list):
            continue
        for value in indices:
            if isinstance(value, int) and 0 <= value < len(inputs):
                path = inputs[value].get("snapshot") or inputs[value].get("source")
                if path:
                    paths.append(path)
    return paths


def _committee_training_gate_evidence(controller, declared, provenance_inputs):
    """Surface a compact, verified training-evidence summary in place of the raw multi-seed
    committee OUTPUT DIRECTORY for the training gate's bounded evidence.

    Detected generically off the declared outputs' own shape -- a ``student_committee.manifest.json``
    (the committee manifest contract) alongside a committee output DIRECTORY -- never a hardcoded
    stage name. When that shape is present, the committee directory (whose ``artifact_digest`` is a
    ~1,500-entry per-file listing of intermediate feature-cache files -- filesystem noise that told
    earlier Judges nothing about what was trained, and even misled one into reading a single seed's
    cache files as "multiple runs") is dropped from the EVIDENCE packet and replaced by a small
    semantic summary (dataset provenance, committee/checkpoint identity + hashes, per-seed training
    dynamics from the real LOGs, and deterministic verification of the checkpoint/provenance claims;
    see runtimes.pydantic_ai.training_evidence). This changes ONLY the LLM-facing evidence: the
    committee directory's canonical ``artifact_digest`` and its Controller-registered artifact
    record (with the full tree ``sha256`` the summary re-surfaces) are untouched, as are the
    committee manifest and all provenance inputs, which remain in the packet unchanged.
    """
    declared_paths = [Path(p) for p in declared]
    manifest = next((p for p in declared_paths
                     if p.name == "student_committee.manifest.json" and p.is_file()), None)
    committee_dir = next((p for p in declared_paths if p.is_dir()), None)
    if manifest is None or committee_dir is None:
        return [str(p) for p in declared_paths] + list(provenance_inputs)
    from .training_evidence import write_training_evidence_summary
    summary_path = write_training_evidence_summary(controller.run_dir)
    kept = [p for p in declared_paths if p.resolve() != committee_dir.resolve()]
    return [str(p) for p in kept] + [str(summary_path)] + list(provenance_inputs)


def _gate_lineage_only_artifacts(gate_evidence_artifacts):
    """Which gate-evidence artifacts should be surfaced lineage/integrity-only (see
    bounded_evidence.lineage_reference_summary) rather than fully summarized.

    Shape-detected, never a stage name: when the packet already carries a four-channel
    accuracy_report.json (which surfaces the exact evaluation population and every fidelity metric,
    aggregate + domain/configuration-family resolved), the co-declared raw per-frame predictions
    ``.extxyz`` is redundant SCIENTIFIC content for the gate. Surfacing it lineage-only keeps its
    deterministic provenance/hash binding while dropping its bulky per-frame distribution, so the
    Judge is not required to infer fidelity from a large extxyz artifact and the packet stays within
    the Judge context budget. Returns [] (no change) for any packet without a four-channel report.
    """
    from .bounded_evidence import _is_four_channel_accuracy_report

    paths = [Path(p) for p in gate_evidence_artifacts]
    has_four_channel = False
    for p in paths:
        if p.suffix.lower() == ".json" and p.is_file():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if _is_four_channel_accuracy_report(payload):
                has_four_channel = True
                break
    if not has_four_channel:
        return []
    return [str(p) for p in paths if p.suffix.lower() in {".xyz", ".extxyz"}]


def _is_gate_only_correction_regate(controller, stage_name):
    """True when the current iteration is an audited evidence-surfacing correction re-gate (see
    RunController.open_correction_iteration) of a stage that is ALREADY completed with all of its
    declared outputs present on disk.

    In that case the stage's production action must NOT be re-executed: the correction re-gate only
    re-surfaces corrected evidence and re-judges the SAME, already-accepted authoritative outputs
    (directive: "re-run ONLY the evidence/gate path"; do not rerun a deterministic executor merely
    to obtain a fresh Judge identity). Generic -- keyed off the correction-iteration trigger marker
    and the stage's own completion/outputs, never a hardcoded stage name.
    """
    iteration = controller.state["iterations"][-1]
    trigger = iteration.get("trigger") or {}
    if trigger.get("kind") != "evidence_surfacing_correction":
        return False
    if trigger.get("regate_stage") not in (None, stage_name):
        return False
    stage = controller.stage(stage_name)
    if stage.get("status") != "completed":
        return False
    outputs = stage.get("outputs", [])
    if not outputs:
        return False
    return all((controller.run_dir / rel).exists() for rel in outputs)


def _stage_input_artifact_paths(proposal, artifacts, own_outputs=()):
    """Scope the producer evidence packet to the artifacts a stage actually consumes.

    The producer only ECHOES the Controller's authoritative proposal, so its evidence packet
    should show the stage's declared inputs -- the artifact paths named in the proposal's
    parameters -- not the whole accumulated artifact registry, which grows every stage and past a
    point drives a small local producer model to emit an empty proposal skeleton. Falls back to the
    full registry when no parameter names a registered artifact, preserving prior behavior for
    stages that declare no artifact-valued parameters.

    A stage's OWN declared outputs (``own_outputs``) are always excluded, even when the proposal's
    parameters name them (evaluation, for instance, names its ``report_path``/``labeled_output``
    outputs). The producer proposes to PRODUCE those outputs; it must never receive its own prior
    outputs as input evidence. On a first run this is a no-op (the outputs aren't registered
    artifacts yet); on a re-gate re-run it prevents a stale prior output -- e.g. a fully expanded
    accuracy_report.json -- from ballooning the producer prompt. The GATE evidence packet is built
    separately from the declared outputs and is unaffected.
    """
    referenced = set()

    def _collect(value):
        if isinstance(value, str):
            referenced.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                _collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)

    _collect((proposal or {}).get("parameters") or {})
    excluded = {str(Path(p).resolve()) for p in own_outputs}
    all_paths = [a["path"] for a in artifacts if str(Path(a["path"]).resolve()) not in excluded]
    scoped = [p for p in all_paths if p in referenced]
    return scoped or all_paths


def _teacher_validation_not_applicable_reason(controller, stage_name, stage_cfg):
    """Return a non-empty reason string iff ``stage_name`` declares an OPTIONAL
    ``teacher_validation_component`` -- this stage/executor's STATIC CAPABILITY: which generic
    ``validation.teacher_evidence_profile.VALIDATION_COMPONENTS`` name(s) it is able to execute,
    a single string or a list of them -- and NONE of those capable component(s) intersect this
    run's own COMMITTED ``teacher_validation_plan``'s ``selected_components`` (the separate AGENT
    DECISION of which admissible component(s) this campaign actually uses). Returns None (stage
    IS applicable / dispatch proceeds normally) whenever the stage declares no capability, or the
    intersection of capability and selection is non-empty.

    This is a static-capability/agent-decision split, not a preselection mechanism: a stage may
    declare it is capable of MULTIPLE components (e.g. a reference-validation-style stage capable
    of either ``ORIGINAL_HELDOUT_FIDELITY`` or ``INDEPENDENT_REFERENCE_FIDELITY``) without the
    workflow author having to guess, in advance, which one the plan will actually select -- the
    plan (never the workflow config, never a Judge/LLM self-routing at dispatch time) is what
    narrows the intersection down to what actually applies.

    A stage that declares a capability but has no committed plan yet is left to whatever normal
    dispatch failure follows (never silently marked not-applicable for a merely-not-yet-planned
    run) -- the automatic pre-campaign planning step is expected to have already committed a plan
    before any stage reaches this check when a workflow declares ``teacher_evidence_sources``."""
    declared = stage_cfg.get("teacher_validation_component")
    if not declared:
        return None
    capable = {declared} if isinstance(declared, str) else set(declared)
    plan = controller.state.get("teacher_validation_plan")
    if plan is None:
        return None
    selected = set(plan.get("selected_components") or [])
    if capable & selected:
        return None
    return (f"stage {stage_name!r} is capable of Teacher-validation component(s) "
           f"{sorted(capable)!r}, none of which this run's committed Teacher validation plan "
           f"(selected_components={sorted(selected)!r}) selects")


def _teacher_validation_plan_coverage_gap(controller):
    """Return a non-empty reason string iff this run has a COMMITTED ``teacher_validation_plan``
    whose ``selected_components`` includes at least one component that NO declared stage's
    ``teacher_validation_component`` capability (see ``_teacher_validation_not_applicable_reason``)
    can execute anywhere in this workflow. Returns None when there is no committed plan yet, or
    every selected component is covered by at least one stage's declared capability.

    This is a WHOLE-WORKFLOW coverage invariant, distinct from (and checked before)
    ``_teacher_validation_not_applicable_reason``'s per-stage check: that function only tells one
    stage whether IT should run; it has no way to notice that a selected component is simply never
    covered by ANY stage, in which case every stage capable of *some* component would be marked
    NOT_APPLICABLE one at a time and the campaign would proceed toward acquisition having silently
    never executed a component the committed plan actually selected. Fail-closed here catches that
    fail-open planning/execution mismatch before any stage is marked NOT_APPLICABLE or acquisition
    is dispatched.

    Generic over ``stage_name``/component identity: never a hardcoded stage or material name,
    only the workflow config's own declared ``teacher_validation_component`` values re-derived
    fresh each call, exactly like ``_teacher_validation_not_applicable_reason`` and
    ``_stage_config``."""
    plan = controller.state.get("teacher_validation_plan")
    if plan is None:
        return None
    import yaml
    cfg = yaml.safe_load(Path(controller.state["workflow_config"]).read_text()) or {}
    covered = set()
    for stage in cfg.get("stages", []):
        declared = stage.get("teacher_validation_component")
        if not declared:
            continue
        covered |= {declared} if isinstance(declared, str) else set(declared)
    selected = set(plan.get("selected_components") or [])
    uncovered = sorted(selected - covered)
    if not uncovered:
        return None
    return (f"committed Teacher validation plan selects component(s) {uncovered!r} that no "
           f"declared stage's teacher_validation_component capability can execute in this "
           f"workflow (covered: {sorted(covered)!r}) -- refusing to proceed with a committed "
           f"validation component that would never actually run")


def _teacher_validation_downstream_reliance_gap(controller, stage_name, stage_cfg):
    """Return a non-empty reason string iff ``stage_name`` is a COSTLY downstream-reliance stage
    (``approval_boundary`` in ``{"costly_teacher_labeling", "costly_training"}``) that would rely
    on this run's committed Teacher validation plan while that plan lacks
    ``ORIGINAL_HELDOUT_FIDELITY``/``INDEPENDENT_REFERENCE_FIDELITY``, and no distinct
    ``authorize_downstream_teacher_reliance`` has been recorded for it yet.

    This is a SEPARATE gate from the stage's own generic ``approval_boundary`` action-approval
    mechanism (``grant_action_approval``/``has_action_approval``): that mechanism authorizes the
    action in general, independent of any Teacher-validation evidence; this one specifically
    binds a human's knowing acceptance of relying on an evidence-limited Teacher to the EXACT
    committed plan (see ``RunController.authorize_downstream_teacher_reliance`` -- the approval
    recorded there is bound to this run's ``evidence_profile_sha256``/``validation_objectives``/
    plan ``content_sha256`` and is invalidated by any change to them).

    Returns None (no gap -- dispatch proceeds to the normal approval_boundary/producer path) when
    the stage isn't costly, no plan is committed yet (nothing to check against -- a stage
    reaching this point with no plan either declares no ``teacher_evidence_sources`` at all, or
    planning has not yet run; either way this is not this gate's concern), the plan already
    includes a fidelity component, or downstream reliance has already been authorized.
    """
    route = stage_cfg.get("pydantic_ai") or {}
    if route:
        boundary = route.get("approval_boundary")
    else:
        default = _default_stage_route(stage_name)
        boundary = default[2] if default else None
    if boundary not in ("costly_teacher_labeling", "costly_training"):
        return None
    plan = controller.state.get("teacher_validation_plan")
    if plan is None:
        return None
    fidelity = {"ORIGINAL_HELDOUT_FIDELITY", "INDEPENDENT_REFERENCE_FIDELITY"}
    if fidelity & set(plan.get("selected_components") or []):
        return None
    if plan.get("downstream_reliance_approval") is not None:
        return None
    return (f"stage {stage_name!r} (approval_boundary={boundary!r}) relies on this run's "
           f"committed Teacher validation plan, which selected only "
           f"{plan.get('selected_components')!r} (no predictive-fidelity component) -- run "
           "`authorize-downstream-teacher-reliance` before this stage may dispatch")


def _default_stage_route(stage_name):
    return {
        "teacher_baseline": ("simulation", "build_teacher_baseline", "costly_teacher_labeling"),
        "reference_validation": ("simulation", "validate_teacher_reference", "costly_teacher_labeling"),
        "acquisition": ("data-curator", "acquire_structures", "costly_teacher_labeling"),
        "teacher_labeling": ("data-curator", "label_with_teacher", "costly_teacher_labeling"),
        "dataset_split": ("data-curator", "generate_group_split", None),
        "training": ("ml-trainer", "train_committee", "costly_training"),
        "evaluation": ("ml-trainer", "evaluate_heldout_fidelity", None),
        "teacher_md": ("simulation", "run_teacher_md", "production_md"),
        "deployment_md": ("simulation", "run_student_md", "production_md"),
    }.get(stage_name)


def _stage_route_action(controller, stage_name):
    """Resolve a stage's action_type the SAME way ``_proposal_from_stage`` does -- from the
    stage's own ``pydantic_ai.action`` route metadata, falling back to ``_default_stage_route`` --
    WITHOUT building a proposal, binding a plan, or touching any executor. Used only to decide,
    cheaply and side-effect-free, whether the next eligible stage is the acquisition stage so its
    (expensive) autonomous planning can be deferred until that stage is genuinely entered."""
    route = _stage_config(controller, stage_name).get("pydantic_ai") or {}
    if route:
        return route.get("action")
    default = _default_stage_route(stage_name)
    return default[1] if default else None


def _input_source(controller, contains=None, suffix=None, exclude_contains=None):
    for record in controller.state.get("inputs", []):
        path = record.get("snapshot") or record.get("source")
        if not path:
            continue
        name = Path(path).name
        full = str(path)
        if contains and contains not in name and contains not in full:
            continue
        if exclude_contains and (exclude_contains in name or exclude_contains in full):
            continue
        if suffix and not name.endswith(suffix):
            continue
        return path
    return None


def _fill_default_parameters(controller, stage_name, params, route=None):
    """Fill in the generic defaults this framework infers on the caller's behalf:
    a `reference_validation` stage with no bound parameters resolves its Teacher config
    and report/prediction output paths from the controller's bound inputs/declared
    outputs. Every other stage (including `teacher_baseline`, which requires an explicit
    `structures_path` -- there is no safe generic guess for which bound structures file
    represents "the" deployment-domain baseline) must be given explicit parameters.

    A stage whose ``pydantic_ai`` route declares ``parameters_from_teacher_validation_plan:
    true`` additionally gets ``target_split``/``reference_kind``/``source_dataset_role`` filled
    from this run's own COMMITTED ``teacher_validation_plan`` (see
    ``RunController.commit_teacher_validation_plan``) -- only for keys the stage config does not
    already set explicitly (an explicit stage parameter always wins). This is opt-in per stage
    (never automatic for a stage that does not declare the flag) and fails closed if no plan has
    been committed yet, since there is nothing to fill from."""
    if (route or {}).get("parameters_from_teacher_validation_plan"):
        plan = controller.state.get("teacher_validation_plan")
        if plan is None:
            raise ValueError(
                f"stage {stage_name!r} declares parameters_from_teacher_validation_plan but no "
                "Teacher validation plan has been committed for this run yet"
            )
        for key in ("target_split", "reference_kind", "source_dataset_role"):
            if key not in params and plan.get(key) is not None:
                params[key] = plan[key]
    if not params and stage_name == "reference_validation":
        stage = controller.stage(stage_name)
        outputs = stage.get("outputs") or []
        if len(outputs) != 2:
            raise ValueError("reference_validation requires exactly two declared outputs")
        teacher = _input_source(controller, "teacher", ".yaml")
        if not teacher:
            raise ValueError("reference_validation requires a bound Teacher configuration input")
        return {
            "teacher_config": teacher,
            "report_path": str((controller.run_dir / outputs[0]).resolve()),
            "predictions_path": str((controller.run_dir / outputs[1]).resolve()),
            "domain_fields": ["structural_domain"],
            # validate_teacher_reference's reference_yaml is always resolved (below, via
            # _protected_reference_from_inputs) to a controller-bound protected reference whose
            # kind validate_reference_config only ever accepts as an already-frozen, immutable
            # DFT-labeled population (validation.protected_reference._REFERENCE_KIND_VALIDATORS) --
            # this action structurally never creates new DFT or protected-reference labels, only
            # fresh Teacher predictions for comparison against them. These two generic flags (the
            # same ones build_teacher_baseline's deployment_domain already declares) let
            # runtimes.pydantic_ai.actions.resolve_action_approval_boundary recognize that.
            "dft_labels_used": False,
            "protected_reference_labels_used": False,
        }
    if stage_name == "teacher_baseline" and "structures_path" not in params:
        raise ValueError("teacher_baseline requires explicit pydantic_ai.parameters.structures_path")
    return params


def _resolve_physical_validation_species_mapping(controller, params):
    """Resolve physical_validation's LAMMPS type->species ordering from the
    Controller-bound ``student_config``'s ``deploy.elements`` -- the same ordering
    already used to build that config's LAMMPS ``pair_coeff`` line -- rather than
    trusting any model-supplied mapping. A stage config MAY also carry a literal
    ``specorder`` override, but only as a redundant check against the authoritative
    source: it must match exactly, or dispatch fails closed. No ``student_config`` ->
    no injected mapping at all (self-describing frame formats like extxyz never
    needed one; a raw LAMMPS dump that needs one will fail closed inside the
    executor itself, via validation.species_mapping.requires_specorder)."""
    import yaml
    from validation.species_mapping import validate_specorder
    student_config = params.get("student_config")
    explicit_specorder = params.get("specorder")
    if not student_config:
        if explicit_specorder:
            raise ValueError(
                "physical_validation was given an explicit specorder override with no bound "
                "student_config to prove its provenance against; bind "
                "pydantic_ai.parameters.student_config or remove the override")
        return params
    resolved_path = str(Path(student_config).resolve())
    bound_hash = None
    for record in controller.state.get("inputs", []):
        candidates = {record.get("source"), record.get("snapshot")} - {None}
        if any(str(Path(raw).resolve()) == resolved_path for raw in candidates):
            bound_hash = record.get("sha256")
            break
    if not bound_hash:
        raise ValueError(
            "physical_validation student_config is not a controller-bound input; it must be "
            "declared under the workflow's top-level inputs so its hash is Controller-verified")
    cfg = yaml.safe_load(Path(resolved_path).read_text()) or {}
    specorder = validate_specorder((cfg.get("deploy") or {}).get("elements"))
    if explicit_specorder and list(explicit_specorder) != specorder:
        raise ValueError(
            "physical_validation explicit specorder override contradicts the authoritative "
            f"student_config deploy.elements ordering ({explicit_specorder!r} != {specorder!r})")
    params = dict(params)
    params.pop("specorder", None)
    params["species_mapping"] = {
        "source": "student_config.deploy.elements",
        "specorder": specorder,
        "student_config_sha256": bound_hash,
    }
    return params


def _protected_reference_from_inputs(controller):
    """All controller-bound Teacher-vs-DFT reference configs, keyed by their declared ``kind``
    (``validation.protected_reference._REFERENCE_KIND_VALIDATORS``) -- e.g.
    ``protected-existing-dft`` (the physically-recovered historical artifact, unresolved original-
    selection provenance, permanently protected from Student use) and ``recovered-original-
    holdout`` (an algorithmically-reconstructed partition of the Teacher's own original train/
    validation/test split membership). A run may bind at most one reference config PER kind -- the
    two kinds serve distinct, non-substitutable scientific roles and a run may legitimately bind
    both at once (see ``_proposal_from_stage``, which selects between them by the stage's own
    action and, for ``validate_teacher_reference``, the run's committed ``teacher_validation_plan``
    -- never by which one happens to be bound)."""
    import yaml
    from validation.protected_reference import _REFERENCE_KIND_VALIDATORS, validate_reference_config
    found: dict[str, str] = {}
    for record in controller.state.get("inputs", []):
        raw = record.get("snapshot") or record.get("source")
        if not raw or Path(raw).suffix.lower() not in {".yaml", ".yml"}:
            continue
        try:
            payload = yaml.safe_load(Path(raw).read_text()) or {}
        except Exception:
            continue
        kind = payload.get("kind") if isinstance(payload, dict) else None
        if kind not in _REFERENCE_KIND_VALIDATORS:
            continue
        validate_reference_config(raw)
        resolved = str(Path(raw).resolve())
        if kind in found and found[kind] != resolved:
            raise ValueError(f"multiple {kind!r} reference configs are bound to this run")
        found[kind] = resolved
    return found


def _protection_consuming_action(action):
    return action in {"acquire_structures", "label_with_teacher", "train_committee"}


# Protection reference kinds a protection-consuming action (acquire_structures / label_with_teacher /
# train_committee) enforces disjointness against, in precedence order. Both kinds materialize a
# permanently-protected population the Student may never consume; the identity-only kind (FE-023)
# carries no DFT labels but is an equally-binding protection boundary. EVALUATION-only kinds (e.g.
# recovered-original-holdout) are deliberately excluded -- they are selected by teacher-validation
# plan, not by generic protection-consuming actions (see _resolve_teacher_reference_binding).
_CANONICAL_PROTECTION_REFERENCE_KINDS = ("protected-existing-dft", "protected-structure-identity")


def _canonical_protected_reference_yaml(controller, protected_references=None):
    """The single canonical protected reference_yaml every protection-consuming action resolves
    against, so acquisition and teacher-labeling (and committee training) enforce disjointness
    against the SAME protected population regardless of which protection kind the run bound.

    Returns the resolved absolute path of the highest-precedence bound protection reference
    (``protected-existing-dft`` first, else ``protected-structure-identity``), or ``None`` when the
    run binds no protection reference at all (an explicitly unprotected campaign). Never silently
    substitutes an EVALUATION-only reference for the protected population."""
    if protected_references is None:
        protected_references = _protected_reference_from_inputs(controller)
    for kind in _CANONICAL_PROTECTION_REFERENCE_KINDS:
        ref = protected_references.get(kind)
        if ref:
            return ref
    return None


def _stage_declares_protection_audit(stage_cfg):
    """True iff the stage declares a ``*_protection_audit.json`` output -- i.e. its executor is
    contractually required to emit a protected-reference exclusion audit, which it can only do when a
    canonical protected reference_yaml is resolvable."""
    for out in (stage_cfg.get("outputs") or []):
        if str(out).endswith("_protection_audit.json"):
            return True
    return False


def _acquisition_protection_reference_yaml(controller):
    """The EXACT ``reference_yaml`` the ``acquire_structures`` executor enforces protected-reference
    disjointness against -- resolved controller-native so the autonomous acquisition PLANNER can
    EXCLUDE the SAME protected population BEFORE selection that the executor re-checks AFTER.

    This mirrors ``_proposal_from_stage``'s ``acquire_structures`` resolution EXACTLY (keep the two in
    lock-step): a bound ``protected-existing-dft`` reference takes precedence (there it overrides the
    stage param); otherwise the acquisition stage's own declared ``reference_yaml`` param, resolved
    through the same ``{run_dir}`` / ``{artifacts_dir}`` / ``{project_dir}`` substitution. Returns
    ``None`` only when the run declares no acquisition protection reference at all (an explicitly
    unprotected campaign) -- never a silent empty protected set for a misconfigured one."""
    canonical = _canonical_protected_reference_yaml(controller)
    if canonical:
        return canonical
    import yaml
    cfg = yaml.safe_load(Path(controller.state["workflow_config"]).read_text()) or {}
    subs = {"run_dir": str(controller.run_dir),
            "artifacts_dir": str(controller.run_dir / "artifacts"),
            "project_dir": controller.state["project_dir"]}
    for stage in cfg.get("stages", []):
        route = stage.get("pydantic_ai") or {}
        if route.get("action") != "acquire_structures":
            continue
        ref = (route.get("parameters") or {}).get("reference_yaml")
        if ref:
            return str(Path(str(ref).format(**subs)).resolve())
    return None


def _resolve_teacher_reference_binding(controller, protected_references):
    """Which controller-bound reference config ``validate_teacher_reference`` must execute
    against, resolved from the run's own committed ``teacher_validation_plan`` -- never from
    whichever reference kind simply happens to be bound (R26 forensic finding: with only a
    ``protected-existing-dft`` reference bound, that historical artifact was silently substituted
    for a plan that had selected ``ORIGINAL_HELDOUT_FIDELITY``, whose required evidence is the
    recovered original Teacher held-out split, not the historical population).

    When the committed plan selects ``ORIGINAL_HELDOUT_FIDELITY``, a ``recovered-original-
    holdout`` reference MUST be bound and its own declared ``target_split`` must match the plan's
    ``target_split`` exactly -- fail closed on either gap; the historical ``protected-existing-
    dft`` reference is never substituted, even if it is the only one bound. Outside that
    component selection (no plan committed yet, or a plan that selects it), the existing
    ``protected-existing-dft`` behavior is unchanged.
    """
    import yaml

    plan = controller.state.get("teacher_validation_plan")
    selected = set((plan or {}).get("selected_components") or [])
    if "ORIGINAL_HELDOUT_FIDELITY" in selected:
        recovered = protected_references.get("recovered-original-holdout")
        if not recovered:
            raise ValueError(
                "teacher_validation_plan selects ORIGINAL_HELDOUT_FIDELITY but no "
                "recovered-original-holdout reference is bound to this run -- the historical "
                "protected-existing-dft reference (if bound) can never substitute for it")
        plan_target_split = plan.get("target_split")
        cfg = yaml.safe_load(Path(recovered).read_text()) or {}
        if plan_target_split and cfg.get("target_split") != plan_target_split:
            raise ValueError(
                "bound recovered-original-holdout reference target_split "
                f"{cfg.get('target_split')!r} does not match the committed "
                f"teacher_validation_plan's target_split {plan_target_split!r}")
        return recovered
    protected_reference = protected_references.get("protected-existing-dft")
    if not protected_reference:
        raise ValueError("reference_validation requires a controller-bound protected reference")
    return protected_reference


def _proposal_from_stage(controller, stage_name, stage_cfg):
    route = stage_cfg.get("pydantic_ai") or {}
    if not route:
        default = _default_stage_route(stage_name)
        if not default:
            raise ValueError("stage requires pydantic_ai role/action metadata")
        role, action, boundary = default
        params = {}
    else:
        role = route["role"]
        action = route["action"]
        boundary = route.get("approval_boundary")
        params = dict(route.get("parameters") or {})
    context = {"run_dir": str(controller.run_dir),
               "artifacts_dir": str(controller.run_dir / "artifacts"),
               "project_dir": controller.state["project_dir"]}
    def fmt(value):
        if isinstance(value, str):
            return value.format(**context)
        if isinstance(value, list):
            return [fmt(v) for v in value]
        if isinstance(value, dict):
            return {k: fmt(v) for k, v in value.items()}
        return value
    params = fmt(_fill_default_parameters(controller, stage_name, params, route=route))
    protected_references = _protected_reference_from_inputs(controller)
    if action == "validate_teacher_reference":
        reference_path = _resolve_teacher_reference_binding(controller, protected_references)
        existing = params.get("reference_yaml")
        if existing is not None and str(Path(existing).resolve()) != reference_path:
            raise ValueError("stage proposal reference_yaml does not match the required reference config")
        params["reference_yaml"] = reference_path
    elif _protection_consuming_action(action):
        protected_reference = _canonical_protected_reference_yaml(controller, protected_references)
        existing = params.get("reference_yaml")
        if protected_reference is not None:
            if existing is not None and str(Path(existing).resolve()) != protected_reference:
                raise ValueError("stage proposal reference_yaml does not match the controller-bound protected reference")
            params["reference_yaml"] = protected_reference
        elif existing is None and _stage_declares_protection_audit(stage_cfg):
            raise ValueError(
                f"protection-consuming action {action!r} declares a protection audit output but no "
                "canonical protected reference (protected-existing-dft or protected-structure-identity) "
                "is bound to this run -- cannot emit the required protection audit; fail closed")
    if action == "acquire_structures" and not (
            params.get("acquisition_plan_path") or params.get("acquisition_plan")):
        # An AcquisitionPlan bound to the run (autonomously by the run-campaign planner, or supplied
        # as a human input at init) is a first-class controller input; resolve it into the stage
        # proposal here so the acquisition executor consumes it WITHOUT the workflow.yaml or a human
        # having to hard-code a per-run plan path. Only the auto-fill is generic -- the downstream
        # bound-input identity/hash check (``_bind_acquisition_plan_for_stage``) stays fail-closed.
        bound_plan_path = _resolve_bound_acquisition_plan(controller)
        if bound_plan_path is not None:
            params["acquisition_plan_path"] = bound_plan_path
    if action == "generate_run_summary":
        # The state snapshot is ALWAYS freshly assembled from the current Controller state right
        # before dispatch -- never taken from a config-supplied path -- so the Analyst can never
        # be handed a stale or hand-authored substitute for what actually happened in this run.
        snapshot_dir = controller.run_dir / "artifacts" / "analysis"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / "run_state.snapshot.json"
        snapshot_path.write_text(json.dumps(_assemble_run_summary_state(controller), indent=2) + "\n")
        params["run_state_path"] = str(snapshot_path.resolve())
    if action == "build_physical_validation_report":
        params = _resolve_physical_validation_species_mapping(controller, params)
    # Idempotency is scoped to the CURRENT recovery iteration, not just the stage: quarantining a
    # stage's outputs at start_iteration() is meaningless if the very next dispatch of that same
    # stage/action is silently treated as a DUPLICATE of the pre-recovery attempt and never
    # actually re-executes. Same action + same iteration -> still a duplicate (blocked); same
    # approved corrective action + a NEW iteration -> a fresh, distinct idempotency identity, so
    # it genuinely re-executes.
    iteration_id = controller.state["iterations"][-1]["id"]
    base_idempotency_key = route.get("idempotency_key",
                                     f"{controller.state['run_id']}:{stage_name}:001")
    return {
        "schema_version": 1,
        "run_id": controller.state["run_id"],
        "stage": stage_name,
        "requested_at": "controller-stage-runner",
        "rationale": f"execute run stage {stage_name} through trusted production dispatch",
        "parameters": params,
        "expected_outputs": list(controller.stage(stage_name).get("outputs", [])),
        "approval_boundary": boundary,
        "idempotency_key": f"{base_idempotency_key}:iter{iteration_id}",
        "dry_run": False,
        "requested_by_role": role,
        "action_type": action,
    }, role



def _run_bound_input_paths(controller):
    paths = set()
    for record in controller.state.get("inputs", []):
        for key in ("source", "snapshot"):
            raw = record.get(key)
            if raw:
                paths.add(str(Path(raw).resolve()))
    return paths


def _resolve_bound_acquisition_plan(controller):
    """Return the resolved path of the run's single canonically-bound AcquisitionPlan input, or
    ``None`` if none is bound. Identifies a plan input by its ``*acquisition_plan.json`` source
    (the same identity ``default_acquisition_provider._acquisition_plan_already_bound`` uses) and
    returns the content-addressed ``snapshot`` (falling back to ``source``) so the value is always a
    member of ``_run_bound_input_paths`` -- the downstream ``_bind_acquisition_plan_for_stage`` gate
    stays fail-closed on that membership + the plan's own field/hash validation. Fails closed if more
    than one acquisition-plan input is bound (an ambiguous run whose plan identity is not unique)."""
    matches = []
    for record in controller.state.get("inputs", []):
        if record.get("superseded"):
            continue
        source = record.get("source") or ""
        if str(source).endswith("acquisition_plan.json"):
            raw = record.get("snapshot") or record.get("source")
            matches.append(str(Path(raw).resolve()))
    if not matches:
        return None
    if len(set(matches)) > 1:
        raise ValueError(
            "PLAN_INPUT_REQUIRED: multiple acquisition_plan inputs are bound to this run; "
            "the acquisition plan identity is ambiguous")
    return matches[0]


def _acquisition_incurs_teacher_inference(acquisition_config_path) -> bool:
    """Deterministically classify whether an acquisition recipe performs Teacher inference, from the
    ACTUAL bound acquisition config -- never a self-asserted flag. This is the typed
    capability/effect signal ``actions.resolve_action_approval_boundary`` consumes to decide whether
    ``acquire_structures`` genuinely needs the ``costly_teacher_labeling`` approval.

    Both built-in recipes drive the REAL Teacher ASE calculator during structure generation, so both
    perform materially costly Teacher inference (``True``): ``teacher-md`` runs Langevin MD under the
    Teacher, and ``augment-atoms`` is executed by ``executors._write_executable_augment_config``,
    which UNCONDITIONALLY binds ``augment_atoms_bridge.teacher_calculator`` as the perturbation/
    relaxation calculator (Teacher forward passes over every candidate x relax step) -- structure
    generation here is NOT Teacher-free. A configured ``adapter.acquire`` callable is opaque, and any
    unknown kind / missing / unreadable / non-dict config, all fail closed to ``True``. The gate is
    only ever relaxed for a recipe that AFFIRMATIVELY proves, through an explicit typed
    ``performs_teacher_inference: false`` declaration, that its structure generation uses no Teacher
    calculator -- and only when that recipe kind is not one the Teacher-binding executor overrides.
    (No current recipe kind qualifies; the escape stays for a genuinely geometry-only future recipe,
    e.g. random/classical-potential perturbation with no Teacher calculator.)"""
    from adapters import load_config
    try:
        cfg = load_config(acquisition_config_path)
    except Exception:  # noqa: BLE001 - unreadable/absent config must fail closed (keep the gate)
        return True
    if not isinstance(cfg, dict):
        return True
    if (cfg.get("adapter") or {}).get("acquire"):
        return True
    kind = cfg.get("kind")
    # Recipe kinds whose executor drives the Teacher calculator: their structure generation IS
    # Teacher inference and can never be relaxed by a self-asserted recipe flag (the executor would
    # bind the Teacher regardless of what the config claims).
    if kind in {"augment-atoms", "teacher-md"}:
        return True
    if cfg.get("performs_teacher_inference") is False:
        return False
    return True


def _bind_acquisition_plan_for_stage(controller, proposal):
    if proposal.get("action_type") != "acquire_structures":
        return proposal
    from .executors import (
        _acquisition_plan_payload,
        _is_existing_pool_plan,
        _validate_acquisition_plan,
        _validate_existing_pool_plan,
    )
    params = dict(proposal.get("parameters") or {})
    raw_plan = params.get("acquisition_plan_path") or params.get("acquisition_plan")
    if not raw_plan:
        raise ValueError("PLAN_INPUT_REQUIRED: acquisition requires acquisition_plan_path or acquisition_plan")
    if params.get("acquisition_plan_path"):
        plan_path = str(Path(params["acquisition_plan_path"]).resolve())
        if plan_path not in _run_bound_input_paths(controller):
            raise ValueError("PLAN_INPUT_REQUIRED: acquisition_plan_path must be a controller-bound run input")
    raw_payload = _acquisition_plan_payload(raw_plan)

    # EXISTING_POOL_SELECTION: SELECT an existing subset for canonical labeling -- selection itself
    # runs NO Teacher inference, so the acquire_structures step is Teacher-free and is not gated
    # behind costly_teacher_labeling (the run still STOPs at the downstream teacher_labeling boundary).
    if _is_existing_pool_plan(raw_payload):
        params["performs_teacher_inference"] = False
        plan = _validate_existing_pool_plan(
            raw_payload, reference_yaml=params.get("reference_yaml"),
            proposal_selected_source_indices=(params.get("selected_source_indices") or None))
        params["acquisition_plan_sha256"] = plan["_plan_sha256"]
        params["selected_source_indices"] = list(plan["selected_source_global_indices"])
        outputs = list(controller.stage(proposal["stage"]).get("outputs") or [])
        if len(outputs) >= 3:
            params.setdefault("protection_audit_path", str((controller.run_dir / outputs[2]).resolve()))
        proposal["parameters"] = params
        return proposal

    # Framework-authoritative typed effect classification: derive performs_teacher_inference from the
    # ACTUAL bound acquisition recipe, overriding any value the stage config or producer supplied, so
    # a geometry-only acquisition is not gated behind costly_teacher_labeling while an acquisition
    # that would run Teacher inference still is (fail-closed).
    params["performs_teacher_inference"] = _acquisition_incurs_teacher_inference(
        params.get("acquisition_config"))
    plan = _validate_acquisition_plan(
        raw_payload, reference_yaml=params.get("reference_yaml"),
        seed_structures=params.get("seed_structures"),
        proposal_selected_source_indices=(params.get("selected_source_indices") or None))
    params["acquisition_plan_sha256"] = plan["_plan_sha256"]
    params["selected_source_indices"] = list(plan["selected_source_global_indices"])
    outputs = list(controller.stage(proposal["stage"]).get("outputs") or [])
    if len(outputs) >= 3:
        params.setdefault("protection_audit_path", str((controller.run_dir / outputs[2]).resolve()))
    params.setdefault("executable_config_path", str(
        (controller.run_dir / "artifacts" / "acquisition_augment_atoms.native.yaml").resolve()))
    proposal["parameters"] = params
    return proposal


def _scope_classification_label_map_artifacts(controller, project_dir):
    """Enumerate the DISTINCT frozen config_type->structure-class ``label_map`` artifacts the run's
    closure-bound ``DeploymentScopeContractV2`` regions reference (via ``membership_evidence``), keyed
    by ``label_map`` content SHA -> ``{path, sha256}``. This is the SAME artifact set
    ``acquisition_readiness._load_scope_classification_evidence`` selects the first of; enumerating all
    lets the data-coverage binder fail closed on conflicting frozen mappings rather than silently
    binding the first. Reads only frozen artifacts on disk; invents nothing."""
    import hashlib as _hashlib
    from .acquisition_readiness import _resolve as _resolve_ref

    v2_state = controller._v2_state() if hasattr(controller, "_v2_state") else {}
    scope_sha = (v2_state or {}).get("scope_contract_sha256")
    if not scope_sha:
        return {}
    scope_dict = controller.v2_contract(scope_sha)
    if not isinstance(scope_dict, dict):
        return {}
    seen_refs: set[str] = set()
    artifacts: dict[str, dict] = {}
    for region in scope_dict.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        for ref in region.get("membership_evidence", []) or []:
            if not isinstance(ref, str) or ref in seen_refs:
                continue
            seen_refs.add(ref)
            path = _resolve_ref(project_dir, ref)
            if not path.is_file():
                continue
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            label_map = obj.get("label_map") if isinstance(obj, dict) else None
            if not (isinstance(label_map, list) and label_map):
                continue
            content_sha = _hashlib.sha256(
                json.dumps(label_map, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")).hexdigest()
            artifacts.setdefault(content_sha, {"path": str(path),
                                               "file_sha256": _hashlib.sha256(
                                                   path.read_bytes()).hexdigest()})
    return artifacts


def _bind_scope_classification_for_data_coverage(controller, proposal):
    """Automatically resolve and propagate the run's FROZEN, human-authored config_type->canonical
    structure-class ``label_map`` to the Stage-4 data-coverage adequacy gate (FE-039), from
    authoritative run-bound inputs, so a fresh successor workflow reaches the gate WITHOUT a human
    manually passing ``scope_classification_evidence_path``. The label_map artifact is the SAME one
    ``acquisition_readiness`` (FE-033) resolves from the closure-bound ``DeploymentScopeContractV2``
    regions' ``membership_evidence`` -- no new source, no hard-coded path, no material conditional.

    Fail-closed contract (requirement 5):
      * no authoritative mapping deterministically resolvable -> inject nothing; Stage 4 then reports
        per-class support NOT_ASSESSABLE (the honest FE-038/FE-039 behavior -- never a fabricated PASS
        nor a false insufficiency);
      * MULTIPLE CONFLICTING frozen label_map artifacts -> raise ``SCOPE_CLASSIFICATION_CONFLICT``;
      * a resolved artifact that does not validate as a ``DeploymentScopeContractV2`` label_map is not
        usable and is simply not bound (covered by the canonical resolver).

    Never overrides an explicit inline ``deployment_domain.structure_class_label_map`` nor a
    pre-supplied ``scope_classification_evidence_path`` (a fresh contract that embeds its own scope
    classification, or an explicit human path, wins). Preserves provenance (artifact identity, sha256,
    source contract identity + sha, resolution path) as an audited run event (requirement 4)."""
    if proposal.get("action_type") != "build_data_coverage_report":
        return proposal
    params = dict(proposal.get("parameters") or {})
    if params.get("scope_classification_evidence_path"):
        return proposal
    deployment_domain = params.get("deployment_domain")
    if isinstance(deployment_domain, dict) and deployment_domain.get("structure_class_label_map"):
        return proposal

    from .acquisition_readiness import _load_scope_classification_evidence
    from workflow.controller import now as _now

    project_dir = str(controller.state.get("project_dir") or Path.cwd())
    scope_v2, binding = _load_scope_classification_evidence(controller, project_dir)
    if scope_v2 is None:
        # No authoritative mapping resolvable -> bind nothing; Stage 4 reports NOT_ASSESSABLE.
        return proposal

    artifacts = _scope_classification_label_map_artifacts(controller, project_dir)
    if len(artifacts) > 1:
        raise ValueError(
            "SCOPE_CLASSIFICATION_CONFLICT: the closure-bound DeploymentScopeContractV2 regions "
            f"reference {len(artifacts)} conflicting frozen config_type->structure-class label_map "
            f"artifacts ({sorted(a['path'] for a in artifacts.values())}); the frozen scope "
            "classification is ambiguous and cannot be auto-bound")

    evidence_path = binding["classification_evidence_path"]
    file_sha = next((a["file_sha256"] for a in artifacts.values()
                     if a["path"] == evidence_path), None)
    params["scope_classification_evidence_path"] = evidence_path
    proposal["parameters"] = params

    controller.state.setdefault("events", []).append({
        "at": _now(),
        "type": "scope_classification_auto_bound_for_data_coverage",
        "stage": proposal.get("stage"),
        "scope_contract_sha256": binding.get("scope_contract_sha256"),
        "scope_contract_id": binding.get("scope_contract_id"),
        "classification_evidence_path": evidence_path,
        "classification_evidence_sha256": file_sha,
    })
    controller.save()
    return proposal


_BOUND_PROPOSAL_FIELDS = {
    "run_id", "stage", "requested_by_role", "action_type", "approval_boundary",
    "idempotency_key", "expected_outputs", "parameters",
}


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def _resolve_run_output(run_dir, raw):
    path = Path(raw)
    if not path.is_absolute():
        if ".." in path.parts:
            raise ValueError("expected_outputs may not contain path traversal")
        path = run_dir / path
    resolved = path.resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError("expected_outputs must resolve inside the controller run_dir")
    return resolved


def _expected_outputs_match(run_dir, authoritative, candidate):
    if not isinstance(authoritative, list) or not isinstance(candidate, list):
        return False, "expected_outputs must be lists"
    if len(authoritative) != len(candidate):
        return False, "expected_outputs count differs"
    try:
        auth_paths = [_resolve_run_output(run_dir, value) for value in authoritative]
        cand_paths = [_resolve_run_output(run_dir, value) for value in candidate]
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    for index, (left, right) in enumerate(zip(auth_paths, cand_paths)):
        if left != right:
            return False, f"expected_outputs target differs at index {index}"
    return True, ""


def _proposal_binding_validator(authoritative, controller=None):
    run_dir = Path(controller.run_dir).resolve() if controller is not None else None

    def validate(candidate):
        for field in sorted(_BOUND_PROPOSAL_FIELDS):
            left = candidate.get(field) if isinstance(candidate, dict) else getattr(candidate, field, None)
            right = authoritative.get(field)
            if field == "expected_outputs" and run_dir is not None:
                ok, message = _expected_outputs_match(run_dir, right, left)
                if not ok:
                    return False, f"authoritative action binding mismatch for {field}: {message}"
                continue
            if _canonical(left) != _canonical(right):
                return False, f"authoritative action binding mismatch for {field}"
        return True, ""
    return validate



def _sha256_file(path):
    from workflow.integrity import sha256_file
    return sha256_file(path)


def build_producer_evidence_packet(controller, stage_name, evidence_path):
    evidence_file = Path(evidence_path).resolve()
    bounded = json.loads(evidence_file.read_text(encoding="utf-8"))
    packet = {
        "schema_version": 1,
        "packet_kind": "ProducerEvidencePacket",
        "run_id": controller.state["run_id"],
        "stage": stage_name,
        "bounded_evidence_path": str(evidence_file),
        "bounded_evidence_sha256": _sha256_file(evidence_file),
        "bounded_evidence_bytes": evidence_file.stat().st_size,
        "max_evidence_bytes": bounded.get("max_evidence_bytes"),
        "artifacts": list(bounded.get("artifacts") or []),
        "protocol_refs": list(bounded.get("protocol_refs") or []),
        "validation_outcomes": list(bounded.get("validation_outcomes") or []),
        "controller_stage": {
            "status": controller.stage(stage_name).get("status"),
            "gate": controller.stage(stage_name).get("gate"),
            "attempts": controller.stage(stage_name).get("attempts"),
            "declared_outputs": list(controller.stage(stage_name).get("outputs") or []),
        },
        "primary_evidence_policy": {
            "primary_evidence_inline": True,
            "tool_discovery_required": False,
            "evidence_is_deterministic_summary": True,
        },
    }
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PRODUCER_EVIDENCE_PACKET_BYTES:
        raise ValueError("ProducerEvidencePacket exceeds 128 KB")
    packet["packet_bytes"] = len(encoded)
    return packet


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def producer_context_policy(provider_name, model_id):
    # mock/local-openai/ollama/anthropic keep their existing behavior unchanged (a known model
    # or the DEFAULT_CONTEXT_WINDOW_TOKENS fallback). The "openai" hosted kind is new: an
    # unrecognized hosted-OpenAI model has no verified window here, and reusing another model's
    # number (e.g. the Qwen-validated default) for it would be a silent guess, so it requires an
    # explicit PYDANTIC_AI_CONTEXT_WINDOW_TOKENS declaration instead of assuming one.
    model = (model_id or "").lower()
    known_context = {
        "qwen2.5-7b-instruct": 8192,
    }
    env_override = os.environ.get("PYDANTIC_AI_CONTEXT_WINDOW_TOKENS")
    if model in known_context:
        default_window = known_context[model]
        source = "env" if env_override else "model-default"
    elif env_override:
        default_window = DEFAULT_CONTEXT_WINDOW_TOKENS
        source = "env"
    elif provider_name == "openai":
        default_window = DEFAULT_CONTEXT_WINDOW_TOKENS
        source = "undeclared"
    else:
        default_window = DEFAULT_CONTEXT_WINDOW_TOKENS
        source = "model-default"
    return {
        "context_window_tokens": _env_int("PYDANTIC_AI_CONTEXT_WINDOW_TOKENS", default_window),
        "output_token_reserve": _env_int("PYDANTIC_AI_OUTPUT_TOKEN_RESERVE", DEFAULT_OUTPUT_TOKEN_RESERVE),
        "prompt_safety_margin_tokens": _env_int("PYDANTIC_AI_PROMPT_SAFETY_MARGIN_TOKENS", DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS),
        "source": source,
    }


def _estimate_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        # Conservative fallback for JSON-heavy prompts when an exact provider tokenizer is absent.
        return (len(text.encode("utf-8")) + 2) // 3


def _producer_context_budget(task, spec, context, budget_policy=None):
    from .role_outputs import select_output_model
    from .tool_registry import ReadOnlyToolset
    prompt = getattr(spec, "prompt", "")
    if getattr(context, "tools_enabled", True):
        prompt += "\n\n" + ReadOnlyToolset(context.read_allow_prefixes).context_note()
    task_text = json.dumps({"task": task}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    schema_text = json.dumps(select_output_model(spec).model_json_schema(), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False)
    components = {
        "system_prompt_tokens": _estimate_tokens(prompt),
        "task_tokens": _estimate_tokens(task_text),
        "output_schema_tokens": _estimate_tokens(schema_text),
    }
    budget_policy = budget_policy or {
        "context_window_tokens": DEFAULT_CONTEXT_WINDOW_TOKENS,
        "output_token_reserve": DEFAULT_OUTPUT_TOKEN_RESERVE,
        "prompt_safety_margin_tokens": DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS,
    }
    input_tokens = sum(components.values())
    output_reserve = int(budget_policy.get("output_token_reserve", DEFAULT_OUTPUT_TOKEN_RESERVE))
    safety_margin = int(budget_policy.get("prompt_safety_margin_tokens", DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS))
    total = input_tokens + output_reserve + safety_margin
    window = int(budget_policy.get("context_window_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS))
    diagnostics = {
        **components,
        "estimated_input_tokens": input_tokens,
        "context_window_tokens": window,
        "output_token_reserve": output_reserve,
        "prompt_safety_margin_tokens": safety_margin,
        "estimated_total_tokens": total,
        "task_bytes": len(task_text.encode("utf-8")),
        "producer_evidence_packet_bytes": (task.get("context") or {}).get("producer_evidence_packet", {}).get("packet_bytes"),
    }
    return total <= window, diagnostics


def _producer_task(stage_name, role, evidence_path, controller, authoritative_proposal,
                   retry_feedback=None):
    packet = build_producer_evidence_packet(controller, stage_name, evidence_path)
    instruction = (f"Inspect and reason about the authoritative execution proposal for stage {stage_name}. "
                   "context.producer_evidence_packet is the complete deterministic primary "
                   "evidence packet; do not discover primary evidence with tools. "
                   "authoritative_action_proposal is immutable execution state. Return exactly the "
                   "same execution-critical binding: copy every execution-critical value exactly, "
                   "preserve relative path strings exactly, preserve ALL parameter keys, preserve "
                   "false values, null values, and empty lists/dicts if present, do not normalize "
                   "paths, do not resolve paths, do not drop keys because they appear optional or "
                   "default, do not stringify lists/dicts/booleans/nulls, do not move path parameters "
                   "into metadata or input_artifacts, and do not change action_type, approval boundary, "
                   "idempotency key, stage, expected outputs, or protected-reference binding. Only "
                   "rationale text may differ.")
    if retry_feedback:
        instruction += (" Your previous ActionProposal failed authoritative binding. "
                        "Use the corrective feedback in context.producer_retry_feedback and return "
                        "the COMPLETE authoritative ActionProposal exactly.")
    context = {"stage": stage_name,
               "primary_evidence_inline": True,
               "producer_evidence_packet": packet,
               "authoritative_action_proposal": authoritative_proposal,
               "authoritative_parameter_keys": sorted(
                   (authoritative_proposal.get("parameters") or {}).keys()),
               "authoritative_expected_outputs": list(
                   authoritative_proposal.get("expected_outputs") or [])}
    if retry_feedback:
        context["producer_retry_feedback"] = retry_feedback
    return {
        "schema_version": 1,
        "task_id": f"{stage_name}-producer",
        "agent": role,
        "run_id": controller.state["run_id"],
        "created_at": "controller-stage-runner",
        "instruction": instruction,
        "inputs": [],
        "criteria": list(controller.stage(stage_name).get("gate_criteria") or ["stage action is valid"]),
        "constraints": [
            "Primary evidence is already supplied in context.producer_evidence_packet; do not discover it with tools.",
            "Return exactly one typed ActionProposal; do not run compute directly.",
        ],
        "context": context,
    }



def _load_candidate_from_provenance(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = payload.get("raw_response")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _parameter_diff_summary(authoritative, candidate):
    auth_params = authoritative.get("parameters") or {}
    cand_params = (candidate or {}).get("parameters") or {}
    missing = sorted(set(auth_params) - set(cand_params))
    extra = sorted(set(cand_params) - set(auth_params))
    changed = []

    def walk(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) & set(right)):
                walk(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if left != right:
                changed.append(path)
        elif left != right or type(left) is not type(right):
            changed.append(path)

    for key in sorted(set(auth_params) & set(cand_params)):
        walk(auth_params[key], cand_params[key], f"parameters.{key}")
    bound_changed = []
    for field in sorted(_BOUND_PROPOSAL_FIELDS - {"parameters"}):
        if _canonical((candidate or {}).get(field)) != _canonical(authoritative.get(field)):
            bound_changed.append(field)
    return {"missing_parameter_keys": missing,
            "extra_parameter_keys": extra,
            "changed_fields": sorted(bound_changed + changed)}


def _binding_retry_feedback(authoritative, candidate, message, attempt_index):
    summary = _parameter_diff_summary(authoritative, candidate)
    return {
        "previous_attempt": attempt_index,
        "binding_error": message,
        **summary,
        "instruction": (
            "Return the COMPLETE authoritative ActionProposal. Preserve every execution-critical "
            "field exactly, including all parameter keys, false/null/empty values, lists, dicts, "
            "and path strings. Do not move path parameters into metadata or input_artifacts."
        ),
    }


def _is_authoritative_binding_rejection(result):
    status = getattr(result.detail, "status", "")
    reason = getattr(result.detail, "reason", "") or result.error or ""
    return status == "INVALID" and reason.startswith(_AUTH_BINDING_MISMATCH), reason


def _run_producer_with_binding_retries(runtime, task, spec, context, *, controller, registry,
                                       authoritative_proposal, task_factory, budget_policy=None,
                                       emitter=None, stage_name=None):
    from types import SimpleNamespace
    from .production_router import RouteResult, run_role
    role = authoritative_proposal.get("requested_by_role")
    action = authoritative_proposal.get("action_type")
    result = None
    feedback = None
    for attempt in range(1, MAX_PRODUCER_GENERATION_ATTEMPTS + 1):
        current_task = task if feedback is None else task_factory(feedback)
        budget_ok, diagnostics = _producer_context_budget(current_task, spec, context, budget_policy)
        if not budget_ok:
            reason = (f"{PRODUCER_CONTEXT_BUDGET_EXCEEDED}: estimated_total_tokens="
                      f"{diagnostics['estimated_total_tokens']} context_window_tokens="
                      f"{diagnostics['context_window_tokens']} output_token_reserve="
                      f"{diagnostics['output_token_reserve']} prompt_safety_margin_tokens="
                      f"{diagnostics['prompt_safety_margin_tokens']} estimated_input_tokens="
                      f"{diagnostics['estimated_input_tokens']} producer_evidence_packet_bytes="
                      f"{diagnostics.get('producer_evidence_packet_bytes')}")
            return RouteResult(
                "producer_dispatch", False, False, reason, Path(""),
                SimpleNamespace(status="INVALID", reason=reason, diagnostics=diagnostics))
        if emitter is not None:
            emitter.emit("role_invocation_started", stage=stage_name, role=role, action=action,
                        detail={"attempt": attempt})

        def _progress_cb(progress: dict, _emitter=emitter, _stage=stage_name, _role=role,
                         _action=action, _controller=controller) -> None:
            _emitter.emit("executor_progress", stage=_stage, role=_role, action=_action,
                         detail=progress)
            # Additive Controller-durable heartbeat (see workflow.controller.heartbeat_stage):
            # a no-op unless begin_stage_execution already marked this stage running. A pid
            # discovered only once the trusted executor's own subprocess starts (after dispatch)
            # is backfilled here rather than requiring it up front.
            _controller.heartbeat_stage(_stage, progress=progress, pid=progress.get("pid")
                                        if isinstance(progress, dict) else None)

        def _on_dispatch_start(_controller=controller, _stage=stage_name, _role=role,
                               _action=action, _proposal=authoritative_proposal) -> None:
            # Fires only once dispatch.authorize_and_execute is about to invoke the real trusted
            # executor (past every pre-executor rejection check) -- exactly the point that
            # precedes R28's class of defect (an executor invoked here can hang for hours with no
            # durable record an attempt occurred). See workflow.controller.begin_stage_execution.
            try:
                from .executors import acquisition_plan_sha256_from_proposal
                plan_sha256 = acquisition_plan_sha256_from_proposal(_proposal)
            except Exception:
                plan_sha256 = None
            _controller.begin_stage_execution(_stage, runner_id="pydantic_ai",
                                              executor=f"{_role}:{_action}",
                                              plan_sha256=plan_sha256)

        result = run_role(runtime, current_task, spec, context, controller=controller,
                          registry=registry, mode="primary", on_dispatch_start=_on_dispatch_start,
                          progress_cb=_progress_cb if emitter is not None else None)
        if emitter is not None:
            emitter.emit("role_invocation_completed", stage=stage_name, role=role, action=action,
                        detail={"attempt": attempt, "accepted": result.accepted,
                                "status": getattr(result.detail, "status", "")})
        retryable, reason = _is_authoritative_binding_rejection(result)
        if not retryable or attempt == MAX_PRODUCER_GENERATION_ATTEMPTS:
            return result
        candidate = _load_candidate_from_provenance(result.provenance_path)
        feedback = _binding_retry_feedback(authoritative_proposal, candidate, reason, attempt)
    return result

def _write_mock_response(path, proposal):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposal, indent=2) + "\n")
    return path


def _stage_outputs(controller, stage_name):
    paths = []
    for rel in controller.stage(stage_name).get("outputs", []):
        path = (controller.run_dir / rel).resolve()
        if path.exists():
            paths.append(path)
    return paths


def _stage_validation_outcomes(controller, stage_name):
    outcomes = []
    stage = controller.stage(stage_name)
    contract = stage.get("contract") or {}
    for event in controller.state.get("events", []):
        if event.get("type") != "external_stage_completed" or event.get("stage") != stage_name:
            continue
        outcomes.append({
            "stage": stage_name,
            "validator": contract.get("validator"),
            "contract_kind": contract.get("kind"),
            "contract_manifest": contract.get("manifest"),
            "contract_validated": bool(event.get("contract_validated")),
            "result": "PASS" if event.get("contract_validated") else "FAIL",
        })
    if not outcomes and contract:
        outcomes.append({
            "stage": stage_name,
            "validator": contract.get("validator"),
            "contract_kind": contract.get("kind"),
            "contract_manifest": contract.get("manifest"),
            "contract_validated": False,
            "result": "NOT_RUN",
        })
    return outcomes


def _direct_judge_artifact_paths(gate_context):
    from .bounded_evidence import DIRECT_JUDGE_ARTIFACT_BYTES
    paths = []
    for raw in gate_context.get("artifact_sha256", {}):
        path = Path(raw).resolve()
        if path.exists() and path.stat().st_size <= DIRECT_JUDGE_ARTIFACT_BYTES:
            paths.append(str(path))
    return paths


def _large_artifact_records(gate_context):
    from .bounded_evidence import DIRECT_JUDGE_ARTIFACT_BYTES
    records = []
    for raw, sha in (gate_context.get("artifact_sha256") or {}).items():
        path = Path(raw).resolve()
        size = path.stat().st_size if path.exists() else None
        if size is None or size > DIRECT_JUDGE_ARTIFACT_BYTES:
            records.append({"path": str(path), "sha256": sha, "size": size,
                            "direct_read_exposed": False,
                            "reason": "represented by bounded evidence summary"})
    return records


def build_judge_evidence_packet(controller, stage_name, judge_index, lens, gate_context, evidence_path):
    evidence_file = Path(evidence_path).resolve()
    bounded = json.loads(evidence_file.read_text(encoding="utf-8"))
    packet = {
        "schema_version": 1,
        "packet_kind": "JudgeEvidencePacket",
        "run_id": controller.state["run_id"],
        "stage": stage_name,
        "judge_id": f"judge-{judge_index}",
        "review_lens": lens["id"],
        "review_focus": lens["focus"],
        "criteria": list(gate_context["criteria"]),
        "bounded_evidence_path": str(evidence_file),
        "bounded_evidence_sha256": __import__("workflow.integrity", fromlist=["sha256_file"]).sha256_file(evidence_file),
        "artifact_sha256": dict(gate_context.get("artifact_sha256") or {}),
        "large_artifacts": _large_artifact_records(gate_context),
        "deterministic_validation": list(bounded.get("validation_outcomes") or []),
        "protocol_refs": list(bounded.get("protocol_refs") or []),
        "artifacts": list(bounded.get("artifacts") or []),
        "large_artifact_policy": {
            "direct_read_limit_bytes": 1_000_000,
            "read_limit_is_scientific_failure": False,
            "policy": ("Large Controller-registered scientific artifacts are authoritative "
                       "provenance. Evaluate them through deterministic bounded summaries, "
                       "hashes, manifests, and validation outcomes when supplied."),
        },
        "provenance_summary": {
            "controller_stage_status": controller.stage(stage_name).get("status"),
            "controller_stage_attempts": controller.stage(stage_name).get("attempts"),
            "controller_stage_gate": controller.stage(stage_name).get("gate"),
        },
    }
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_JUDGE_EVIDENCE_PACKET_BYTES:
        raise ValueError("JudgeEvidencePacket exceeds 128 KB")
    packet["packet_bytes"] = len(encoded)
    return packet


def _write_three_pass_votes(controller, stage_name):
    ctx = controller.gate_context(stage_name)
    votes = []
    for index, lens in enumerate(ctx["review_lenses"], 1):
        votes.append({
            "judge_id": f"judge-{index}",
            "review_lens": lens["id"],
            "verdict": "PASS",
            "criteria_checked": [
                {"criterion": c, "value_read": "bounded evidence and deterministic artifacts", "ok": True}
                for c in ctx["criteria"]
            ],
            "rationale": "synthetic integration-test judge context accepted the frozen evidence",
            "required_fix": "",
        })
    bundle = {**ctx, "decision": "PASS", "votes": votes}
    path = controller.run_dir / "gates" / f"{stage_name}.stage-runner.votes.json"
    path.write_text(json.dumps(bundle, indent=2) + "\n")
    controller.record_gate(stage_name, votes_path=path)
    return path



def _judge_task_id(controller, stage_name, judge_index):
    """Iteration-scope the immutable Judge task identity.

    A re-gate of the same stage after a recovery runs under a NEW recovery iteration (the Controller
    opens a fresh iteration whenever a REVISE/FAIL gate binds pending recovery). Its evidence differs
    from the prior iteration's, so re-deriving the historical ``{stage}-judge-{n}`` task_id would
    collide with the earlier iteration's immutable packet and fail closed with TaskPacketConflictError
    (the exchange never overwrites a conflicting packet). Scoping the task_id by the canonical
    ``_current_iteration()["id"]`` -- the same identity the Controller already uses to name saved
    vote bundles (``record_gate`` -> ``{stage}.iteration-{id:03d}.votes.json``) -- makes each
    re-gate produce DISTINCT immutable Judge task/result/raw/provenance artifacts while leaving the
    prior iteration's audit trail untouched. Iteration 1 keeps the historical unsuffixed identity so
    existing runs' on-disk packets are preserved byte-for-byte; INVALID_JUDGE_OUTPUT retries stay
    within one iteration's task_id (same gate attempt), exactly as before."""
    iteration_id = controller._current_iteration()["id"]
    base = f"{stage_name}-judge-{judge_index}"
    if iteration_id <= 1:
        return base
    return f"{base}-iter{iteration_id:03d}"


def _judge_task(stage_name, judge_index, lens, gate_context, evidence_path, controller,
                criteria=None):
    packet = build_judge_evidence_packet(
        controller, stage_name, judge_index, lens, gate_context, evidence_path)
    # When the closure StageReviewSpec is bound for this stage, ``criteria`` is the
    # lens's own predeclared, frozen criterion questions (Section E/G) rather than the
    # shared free-text gate criteria; the live judge-vote validator then requires the
    # answered ``criteria_checked`` to equal them in order, which is what lets each
    # answer map back to its criterion id when re-encoded as a typed JudgeReview.
    task_criteria = list(gate_context["criteria"] if criteria is None else criteria)
    packet["criteria"] = task_criteria
    # Framework-V2 scientific-adequacy layer (Session 2026-08-21 R1 close):
    # additive; inert when no scientific policy is bound for this stage.
    # Every one of the three mutually-blind Judges receives the SAME frozen
    # scientific block for a given stage (identical policy content_sha256);
    # they still each see only their own review_lens.
    scientific_block = _build_scientific_layer_for_stage(controller, stage_name)
    task = {
        "schema_version": 1,
        "task_id": _judge_task_id(controller, stage_name, judge_index),
        "agent": "judge",
        "run_id": controller.state["run_id"],
        "created_at": "controller-stage-runner",
        "instruction": (f"Judge stage {stage_name} against the frozen criteria using "
                        "context.judge_evidence_packet as the complete primary evidence. "
                        "Distinguish PROCEDURAL VALIDITY (correct computation on the "
                        "correct population) from SCIENTIFIC ADEQUACY (evidence meets "
                        "the bound scientific policy). See context.scientific_adequacy_layer "
                        "for the frozen policy content when active."),
        "inputs": [],
        "criteria": task_criteria,
        "constraints": [
            "Primary evidence is already supplied in context.judge_evidence_packet; do not discover it with tools.",
            "Use only the supplied frozen evidence packet, compact summaries, hashes, validation outcomes, and criteria.",
            "Do not assume or inspect another Judge context or vote.",
            "Do not mark REVISE/FAIL solely because a large registered artifact is not exposed for full direct reading.",
            "Do not request compression or filtering merely for LLM readability when deterministic bounded evidence resolves the criterion.",
            "Procedural PASS is NOT scientific adequacy: a stage may satisfy every procedural criterion while its scientific adequacy layer scores FAIL or NOT_EVALUABLE.",
        ],
        "context": {"review_lens": lens["id"], "review_focus": lens["focus"],
                    "stage": stage_name, "primary_evidence_inline": True,
                    "judge_evidence_packet": packet,
                    "scientific_adequacy_layer": scientific_block},
    }
    return task


def _build_scientific_layer_for_stage(controller, stage_name):
    """Build the scientific-adequacy layer that gets attached to every Judge
    task packet for a scientific stage. Returns a dict that is either
    ``{"scientific_layer_active": False, ...}`` (inert) or the full frozen
    policy block per ``framework_v2.judge_packet_extension``.

    Also runs the deterministic adjudicator so the Judge sees the deterministic
    verdict summary (PASS / FAIL / NOT_EVALUABLE + reasons) alongside the
    frozen policy content.
    """
    from framework_v2.judge_packet_extension import build_scientific_extension_block
    from framework_v2.scientific_gate import (
        POLICIES_KEY, assert_stage_scientific_adequacy, ScientificAdequacyBlocked,
    )
    state = controller.state
    observed_values = {}
    verdict = None
    def _load(rel):
        # run_dir is only needed when the scientific layer is active (policies
        # bound); fetched lazily so an inert layer works with a run_dir-less
        # controller.
        p = controller.run_dir / rel
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    if state.get(POLICIES_KEY):
        try:
            verdict = assert_stage_scientific_adequacy(
                state, stage_name,
                accuracy_report_loader=lambda: _load("artifacts/accuracy_report.json"),
                uncertainty_report_loader=lambda: _load("artifacts/uncertainty_report.json"),
                md_manifest_loader=lambda: _load("artifacts/md.manifest.json"),
                validation_report_loader=lambda: _load("artifacts/validation_report.json"),
            )
        except ScientificAdequacyBlocked as e:
            # In the Judge packet we surface the FAIL / NOT_EVALUABLE state as
            # data; we do NOT raise here because the packet is being READ by
            # the Judge, not gated. The Controller's record_gate PASS branch
            # is where the raise happens.
            verdict = {"stage": stage_name,
                       "adjudications": [{"kind": "aggregated",
                                          "verdict": {"status": "BLOCKED",
                                                       "reason": str(e)}}]}
    return build_scientific_extension_block(state, stage_name,
                                             observed_evidence_values=observed_values,
                                             adequacy_verdict_summary=verdict)


def judge_read_allowlist(gate_context, evidence_path):
    return []


class JudgeResumeConflictError(RuntimeError):
    """Raised when a persisted judge-gate result/provenance binding conflicts with the currently
    derived task for the same task_id (e.g. a stale result no longer matching the run's current
    criteria/review_lens, or a result with no corresponding accepted=true provenance record).
    Fails closed rather than silently reusing or overwriting an inconsistent record."""


class JudgeInvalidOutputBlocker(RuntimeError):
    """Raised when a single lens's Judge keeps emitting an INVALID_JUDGE_OUTPUT review (Section J)
    until the bounded invalid-output retry limit is exhausted. An INVALID_JUDGE_OUTPUT is never a
    Gate vote, so the Gate cannot be aggregated; this is a terminal ``JUDGE_INVALID_BLOCKER`` that
    stops the campaign for human attention -- it is NOT a scientific REVISE/FAIL and never triggers
    the Controller recovery path."""


def _accepted_judge_provenance_exists(exchange, task_id: str) -> bool:
    prov_dir = exchange.exchange_dir / "provenance"
    if not prov_dir.is_dir():
        return False
    for path in prov_dir.glob(f"{task_id}.*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # a malformed/partial provenance record is never treated as accepted
        if rec.get("task_id") == task_id and rec.get("accepted") is True:
            return True
    return False


def _resume_judge_vote(exchange, task):
    """Resume-aware Judge-vote resolution for one already-dispatched judge task (Part B of the
    R19 durability revision).

    Returns an already-accepted, revalidated vote dict to reuse (the Judge is NOT invoked again),
    or ``None`` if no accepted result exists yet and the Judge must actually be invoked this run.
    A malformed/zero-byte/incomplete raw response never counts as accepted state, because
    ``FileExchangeRuntime.accept`` only ever writes to ``results/`` after full contract validation
    succeeds -- so the mere existence of a raw response is not checked here at all. Any existing
    result that no longer binds cleanly to the currently derived task (wrong lens/criteria, or no
    matching accepted=true provenance record) fails closed via ``JudgeResumeConflictError`` rather
    than being silently reused or replaced."""
    from orchestration.exchange import validate_judge_vote
    task_id = task["task_id"]
    result_path = exchange.inbox / f"{task_id}.json"
    if not result_path.is_file():
        return None
    try:
        raw_result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JudgeResumeConflictError(
            f"existing judge result at {result_path} is not valid JSON: {exc}") from exc
    try:
        validated = validate_judge_vote(raw_result, list(task["criteria"]),
                                        task["context"]["review_lens"])
    except ValueError as exc:
        raise JudgeResumeConflictError(
            f"existing judge result at {result_path} does not match the currently derived "
            f"criteria/review_lens for task_id={task_id!r}: {exc}") from exc
    if not _accepted_judge_provenance_exists(exchange, task_id):
        raise JudgeResumeConflictError(
            f"judge result at {result_path} exists with no corresponding accepted=true "
            f"provenance record for task_id={task_id!r}")
    return validated


def run_three_judge_gate(controller, stage_name, specs, runtime_factory, runtime_context_factory,
                         evidence_path, *, mode="primary", emitter=None):
    """Run (or resume) the three-Judge gate for ``stage_name``.

    Resume-aware per Judge index: the expected task packet is always deterministically re-derived
    and (re-)dispatched (idempotent no-op if identical content already exists; fails closed via
    ``TaskPacketConflictError`` if it conflicts -- see ``FileExchangeRuntime.dispatch``). If an
    already-accepted, still-valid result exists for that exact task, it is reused and the Judge is
    NOT invoked again; otherwise the Judge is invoked exactly once for that index. Gate aggregation
    is unchanged: all votes (reused or freshly invoked) are collected, the decision is computed,
    and ``controller.record_gate`` is called exactly once at the end, atomically, as before."""
    from orchestration.exchange import FileExchangeRuntime, TaskPacketConflictError
    from framework_v2.review_packet import validate_judge_review as _validate_judge_review
    from .production_router import run_role
    from . import closure_review as _closure
    gate_context = controller.gate_context(stage_name)
    exchange = FileExchangeRuntime(runtime_context_factory(1).exchange_dir)

    # Framework-V2 closure review path (Sections H & J): when a frozen
    # StageReviewSpec is bound to this stage, each lens is asked its OWN
    # predeclared criteria, a single CanonicalReviewPacket is compiled from the
    # stage's real deterministic facts, and each real Judge vote is re-encoded as
    # a typed JudgeReview so ``RunController._enforce_v2_review`` deterministically
    # validates it (INVALID_JUDGE_OUTPUT refuses PASS) and requires unanimity.
    review_spec = _closure.bound_stage_review_spec(controller, stage_name)
    packet = None
    reviews = []
    if review_spec is not None:
        facts = _closure.deterministic_facts_for_stage(
            stage_name, dict(gate_context.get("artifact_sha256") or {}),
            _stage_validation_outcomes(controller, stage_name))
        decision_sha256 = __import__(
            "workflow.integrity", fromlist=["sha256_file"]).sha256_file(
                Path(evidence_path).resolve())
        packet = _closure.compile_review_packet(
            controller=controller, stage_name=stage_name, spec=review_spec,
            facts=facts, decision_sha256=decision_sha256,
            producer_rationale=f"{stage_name} decision rests on registered artifacts and "
            "deterministic validation outcomes")

    # Bounded invalid-output retry (Section J): when the closure StageReviewSpec is bound, each
    # real Judge vote is re-encoded as a typed JudgeReview and validated against the ONE
    # CanonicalReviewPacket BEFORE it can enter Gate aggregation. An INVALID_JUDGE_OUTPUT (e.g. a
    # verdict that deterministically contradicts the packet's authoritative facts) is NOT a vote:
    # that single lens's Judge is retried -- SAME spec/decision/packet SHA/lens/evidence, no
    # prompt/rubric/threshold change -- for up to two corrective attempts. A still-invalid output
    # after the bound is a terminal JUDGE_INVALID_BLOCKER; it never enters the Gate as a REVISE.
    max_invalid_attempts = 3  # initial + at most 2 invalid-output retries

    def _validate_encoded_review(vote):
        """Encode a vote as a typed JudgeReview and deterministically validate it against the
        bound packet/spec. Returns (review, validation) or (None, None) when V2 closure is not
        bound (legacy advisory path -- no per-review validation)."""
        if review_spec is None or packet is None:
            return None, None
        review = _closure.judge_vote_to_review(
            vote, vote["review_lens"], review_spec, packet,
            run_id=controller.state["run_id"], stage=stage_name, judge_index=1)
        return review, _validate_judge_review(review, review_spec, packet)

    votes = []
    for index, lens in enumerate(gate_context["review_lenses"], 1):
        task = _judge_task(stage_name, index, lens, gate_context, evidence_path, controller)
        try:
            exchange.dispatch(specs["judge"], task)
        except TaskPacketConflictError:
            raise  # never overwrite a conflicting existing task packet; fail closed
        reused = _resume_judge_vote(exchange, task)
        vote = None
        valid_review = None
        for attempt in range(1, max_invalid_attempts + 1):
            if attempt == 1 and reused is not None:
                candidate = dict(reused)
                resumed = True
            else:
                ctx = runtime_context_factory(index)
                if emitter is not None:
                    emitter.emit("role_invocation_started", stage=stage_name, role="judge",
                                action="judge_gate",
                                detail={"judge_id": f"judge-{index}", "attempt": attempt})
                res = run_role(runtime_factory(index), task, specs["judge"], ctx, mode=mode)
                if emitter is not None:
                    emitter.emit("role_invocation_completed", stage=stage_name, role="judge",
                                action="judge_gate",
                                detail={"judge_id": f"judge-{index}", "attempt": attempt})
                if res.error or res.detail is None:
                    if getattr(res, "error_category", None) == "judge_output_invalid":
                        # Hard Judge schema/structural validation failure (e.g. a REVISE/FAIL
                        # vote with an empty required_fix). A malformed output is NOT a vote: it
                        # follows the SAME canonical INVALID_JUDGE_OUTPUT semantics as a
                        # closure-invalid vote below -- classify it, keep it out of Gate
                        # aggregation, and retry ONLY this lens's Judge (same
                        # ScientificDecision/StageReviewSpec/CanonicalReviewPacket/lens/rubric,
                        # via the immutably re-derived task) under the same bounded limit. It
                        # never crashes the campaign; exhaustion becomes a JUDGE_INVALID_BLOCKER
                        # below.
                        if emitter is not None:
                            emitter.emit("judge_invalid_output", stage=stage_name, role="judge",
                                        detail={"judge_id": f"judge-{index}",
                                                "review_lens": lens.get("id"), "attempt": attempt,
                                                "state": "INVALID_JUDGE_OUTPUT",
                                                "errors": [res.error] if res.error else []})
                        continue
                    raise RuntimeError(f"Judge {index} failed validation: {res.error}")
                candidate = dict(res.detail)
                resumed = False
            candidate["judge_id"] = f"judge-{index}"
            review, validation = _validate_encoded_review(candidate)
            if validation is not None and not validation.valid:
                # INVALID_JUDGE_OUTPUT: not a vote. Record and retry this lens's Judge only.
                if emitter is not None:
                    emitter.emit("judge_invalid_output", stage=stage_name, role="judge",
                                detail={"judge_id": f"judge-{index}",
                                        "review_lens": lens.get("id"), "attempt": attempt,
                                        "state": validation.state.value,
                                        "errors": list(validation.errors)})
                continue
            vote = candidate
            valid_review = review
            if emitter is not None:
                emitter.emit("judge_result", stage=stage_name, role="judge",
                            detail={"judge_id": f"judge-{index}", "review_lens": lens.get("id"),
                                    "verdict": vote.get("verdict"), "resumed": resumed})
            break
        if vote is None:
            raise JudgeInvalidOutputBlocker(
                f"JUDGE_INVALID_BLOCKER: lens {lens.get('id')!r} Judge produced an "
                f"INVALID_JUDGE_OUTPUT on all {max_invalid_attempts} attempt(s) "
                f"(initial + up to {max_invalid_attempts - 1} invalid-output retries); an "
                "invalid output is never a Gate vote and never a scientific REVISE")
        votes.append(vote)
        if valid_review is not None:
            reviews.append(valid_review)
    decision = "FAIL" if any(v["verdict"] == "FAIL" for v in votes) else (
        "PASS" if all(v["verdict"] == "PASS" for v in votes) else "REVISE")
    bundle = {**gate_context, "decision": decision, "votes": votes}
    if review_spec is not None and packet is not None:
        # Attach the v2_review bundle (all reviews already validated above) so the Controller
        # deterministically re-validates them (Section J) before any PASS.
        bundle["v2_review"] = _closure.assemble_v2_review(packet, reviews)
    path = controller.run_dir / "gates" / f"{stage_name}.production.votes.json"
    path.write_text(json.dumps(bundle, indent=2) + "\n")
    controller.record_gate(stage_name, votes_path=path)
    return decision, path


def _cmd_preflight(args) -> int:
    from workflow.controller import RunController
    from .bounded_evidence import build_bounded_evidence
    c = RunController(args.run_dir)
    c.verify_inputs()
    evidence = c.run_dir / "exchange" / "bounded_evidence" / "preflight.json"
    artifacts = [r.get("snapshot") or r.get("source") for r in c.state.get("inputs", [])]
    build_bounded_evidence([a for a in artifacts if a], evidence,
                           protocol_refs=[c.state.get("workflow_config")])
    print(f"preflight: OK\nbounded_evidence: {evidence}")
    return EXIT_SUCCESS


def _cmd_approve(args) -> int:
    from workflow.controller import RunController
    c = RunController(args.run_dir)
    c.grant_action_approval(args.boundary, note=args.note, plan_sha256=args.plan_sha256,
                            action_type=args.action_type)
    suffix = f" plan_sha256={args.plan_sha256}" if args.plan_sha256 else ""
    if args.action_type:
        suffix += f" action_type={args.action_type}"
    print(f"approval: {args.boundary}{suffix}")
    emitter = CampaignEventEmitter(c.run_dir, run_id=c.state.get("run_id"), quiet=True)
    emitter.emit("approval_granted",
                 detail={"approval_boundary": args.boundary, "action_type": args.action_type})
    return EXIT_SUCCESS


def _cmd_bind_scientific_policies(args) -> int:
    """Attach typed scientific-adequacy policies to a run.

    Each --policy-file argument is STAGE:KIND:PATH. Bindings are content-hash-
    immutable: identical rebinds are idempotent; different-content rebinds
    refuse. Every binding emits a `scientific_policy_bound` event.
    """
    import json as _json
    from workflow.controller import RunController
    c = RunController(args.run_dir)
    bound = []
    for spec in args.policy_file:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            print(f"invalid --policy-file spec {spec!r}; expected STAGE:KIND:PATH",
                  file=sys.stderr)
            return EXIT_INTERNAL
        stage_name, kind, path = parts
        policy_dict = _json.loads(Path(path).read_text())
        # Strip leading '_' documentation keys (candidate files may carry them)
        policy_dict = {k: v for k, v in policy_dict.items()
                       if not k.startswith("_") and k != "kind"}
        record = c.bind_scientific_policy(
            stage_name, kind, policy_dict,
            source_ref=args.source_ref,
            note=f"cli-bound from {path}",
            required=not args.allow_not_required,
        )
        bound.append((stage_name, kind, record["content_sha256"]))
        print(f"bound {stage_name}::{kind} content_sha256={record['content_sha256']}")
    print(f"total_bindings={len(bound)}")
    return EXIT_SUCCESS


def _cmd_bind_closure(args) -> int:
    """Bind the Framework-V2 closure review contracts onto a run.

    Turns on V2 gate enforcement (``bind_v2_scope_contract``) and binds one frozen,
    generic :class:`~framework_v2.review_spec.StageReviewSpec` per requested stage
    (``bind_v2_stage_review_spec``). After this, every ``record_gate`` PASS at a
    bound stage additionally requires the vote bundle to carry a valid
    ``v2_review`` (CanonicalReviewPacket + one deterministic-validated JudgeReview
    per lens) — which ``run_three_judge_gate`` now assembles from the real Judges.
    """
    from workflow.controller import RunController
    from framework_v2.review_spec import default_stage_review_specs
    from framework_v2.contracts import ConvergencePolicy
    from framework_v2.convergence import default_training_convergence_policy

    c = RunController(args.run_dir)
    scope = json.loads(Path(args.scope_contract).read_text(encoding="utf-8"))
    scope_sha = c.bind_v2_scope_contract(scope)
    specs = default_stage_review_specs(
        validation_profile_version=args.validation_profile_version)
    # Optional caller-supplied ConvergencePolicy override (honors the
    # "caller supplies the numbers" contract). When absent, a stage that
    # requires convergence_report evidence falls back to the framework
    # default so the R31 max-epoch gate is NEVER silently skipped.
    conv_override = None
    if getattr(args, "convergence_policy", None):
        conv_override = ConvergencePolicy(
            **json.loads(Path(args.convergence_policy).read_text(encoding="utf-8")))
    stages = list(args.stage)
    if not stages:
        stages = [s["name"] for s in c.state["stages"] if s.get("gate_criteria")]
    bound = []
    conv_bound = []
    for stage_name in stages:
        spec = specs.get(stage_name)
        if spec is None:
            print(f"no default StageReviewSpec for stage {stage_name!r}", file=sys.stderr)
            return EXIT_VALIDATION_REJECTED
        sha = c.bind_v2_stage_review_spec(stage_name, spec.model_dump(mode="json"))
        bound.append((stage_name, sha))
        # A stage whose predeclared criteria consume ``convergence_report``
        # evidence MUST have a bound ConvergencePolicy, else the gate's
        # convergence precondition is inert (the demonstrated Stage 7 defect).
        # Detected generically from the spec's evidence classes -- no stage
        # name is hardcoded.
        needs_convergence = any(
            "convergence_report" in crit.required_evidence_classes
            for crit in spec.criteria)
        if needs_convergence:
            policy = conv_override or default_training_convergence_policy()
            conv_sha = c.bind_v2_contract(
                "convergence_policy", policy.model_dump(mode="json"), stage=stage_name)
            conv_bound.append((stage_name, conv_sha, policy.policy_id))
    print(f"closure bound: scope_contract_sha256={scope_sha}")
    for stage_name, sha in bound:
        print(f"  stage_review_spec[{stage_name}] = {sha}")
    for stage_name, conv_sha, policy_id in conv_bound:
        print(f"  convergence_policy[{stage_name}] = {conv_sha} ({policy_id})")
    return EXIT_SUCCESS


@dataclass
class StageRunResult:
    """Structured outcome of running one production stage — same information ``_cmd_run_stage``
    used to print + turn into an exit code, now returned so a non-CLI caller (``run-campaign``)
    can make its OWN control decisions without parsing printed text or re-deriving exit codes.
    ``reason`` is a stable, generic code (never a stage name or domain concept)."""
    reason: str
    exit_code: int
    message: str
    approval_boundary: Optional[str] = None
    gate_decision: Optional[str] = None
    evidence_path: Optional[Path] = None
    action_status: Optional[str] = None


def run_production_stage(controller, stage_name, *, runtime, agent_specs_dir="agent_specs",
                         exchange_dir=None, repo_root=".", auto_mock_judges=False,
                         mock_response=None, mock_judge_response=None,
                         emitter: Optional[CampaignEventEmitter] = None) -> StageRunResult:
    """Run ONE stage through the real production dispatch + gate path. The single production
    entry point for a stage: ``_cmd_run_stage`` (the ``run-stage`` CLI command) and
    ``run-campaign``'s forward-progression loop both call this and only this -- there is no
    second, parallel implementation of stage execution.

    Generic over ``stage_name``: every stage-specific value it uses (role/action/approval
    boundary, parameters, outputs, gate criteria/lenses) comes from the controller/workflow
    config, never a hardcoded name in this function.
    """
    from workflow.controller import RunController
    from orchestration.specs import load_agent_specs
    from .bounded_evidence import build_bounded_evidence
    from .executors import build_executor_registry
    from .mock_runtime import MockAgentRuntime
    from .models import RuntimeContext
    from .production_router import run_role  # noqa: F401  (kept for parity with prior imports)
    from . import provider as _prov

    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    c.verify_inputs()
    emitter.emit("stage_selected", **stage_progress_fields(c, stage_name))
    stage_cfg = _stage_config(c, stage_name)
    # FE-054: fail closed if this run DECLARES post-split TRAIN augmentation but its finalized
    # merged dataset has not been produced and routed as the training dataset. This is the one
    # explicit precondition that ties Stage-7 training to the out-of-band augmentation action
    # without adding a lifecycle stage. A no-op for any run that does not declare augmentation.
    from .train_augmentation import stage7_augmentation_guard
    augmentation_gap = stage7_augmentation_guard(c, stage_name)
    if augmentation_gap is not None:
        emitter.emit("campaign_blocked", stage=stage_name,
                     detail={"reason": "POST_SPLIT_AUGMENTATION_NOT_FINALIZED"})
        return StageRunResult("POST_SPLIT_AUGMENTATION_REQUIRED", EXIT_VALIDATION_REJECTED,
                              f"FAILED: {augmentation_gap}")
    coverage_gap = _teacher_validation_plan_coverage_gap(c)
    if coverage_gap is not None:
        emitter.emit("campaign_blocked", stage=stage_name,
                    detail={"reason": "TEACHER_VALIDATION_PLAN_COVERAGE_GAP"})
        return StageRunResult("TEACHER_VALIDATION_PLAN_COVERAGE_GAP", EXIT_VALIDATION_REJECTED,
                              f"FAILED: {coverage_gap}")
    not_applicable_reason = _teacher_validation_not_applicable_reason(c, stage_name, stage_cfg)
    if not_applicable_reason is not None:
        c.mark_stage_not_applicable(stage_name, reason=not_applicable_reason)
        emitter.emit("stage_marked_not_applicable", stage=stage_name,
                    detail={"reason": not_applicable_reason})
        return StageRunResult("NOT_APPLICABLE", EXIT_SUCCESS, not_applicable_reason)
    reliance_gap = _teacher_validation_downstream_reliance_gap(c, stage_name, stage_cfg)
    if reliance_gap is not None:
        emitter.emit("approval_required", stage=stage_name,
                    detail={"status": "APPROVAL_REQUIRED",
                            "approval_boundary": "teacher_validation_downstream_reliance"})
        return StageRunResult(
            "APPROVAL_REQUIRED", EXIT_APPROVAL_REQUIRED,
            f"APPROVAL_REQUIRED: teacher_validation_downstream_reliance: {reliance_gap}",
            approval_boundary="teacher_validation_downstream_reliance")
    proposal, role = _proposal_from_stage(c, stage_name, stage_cfg)
    try:
        proposal = _bind_acquisition_plan_for_stage(c, proposal)
        proposal = _bind_scope_classification_for_data_coverage(c, proposal)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("PLAN_INPUT_REQUIRED"):
            return StageRunResult("PLAN_INPUT_REQUIRED", EXIT_VALIDATION_REJECTED, message)
        if message.startswith("SCOPE_CLASSIFICATION_CONFLICT"):
            return StageRunResult("VALIDATION_REJECTED", EXIT_VALIDATION_REJECTED, message)
        raise
    evidence_path = c.run_dir / "exchange" / "bounded_evidence" / f"{stage_name}.json"
    own_outputs = [c.run_dir / rel for rel in c.stage(stage_name).get("outputs", [])]
    upstream = _stage_input_artifact_paths(proposal, c.state.get("artifacts", []),
                                           own_outputs=own_outputs)
    upstream += _selective_provenance_inputs(c, stage_name)
    split_manifests = _split_membership_manifest_sources(c)
    upstream += split_manifests
    build_bounded_evidence(upstream, evidence_path, protocol_refs=[c.state.get("workflow_config")],
                           lineage_only=split_manifests)
    specs = load_agent_specs(agent_specs_dir)
    task = _producer_task(stage_name, role, evidence_path, c, proposal)
    exchange = Path(exchange_dir) if exchange_dir else c.run_dir / "exchange"

    def ctx_factory(_index, provider_name="mock", model_id="mock"):
        return RuntimeContext(exchange_dir=str(exchange), repo_root=repo_root,
                              provider=provider_name, model_id=model_id,
                              read_allow_prefixes=[], tools_enabled=False)

    if runtime == "mock":
        response_path = Path(mock_response) if mock_response else _write_mock_response(
            exchange / "stage_runner" / f"{stage_name}.proposal.json", proposal)
        raw = response_path.read_text()
        producer_runtime = MockAgentRuntime(lambda t, s, ts: (raw, (0, 0)))
        if mock_judge_response:
            if len(mock_judge_response) != 3:
                raise ValueError("--mock-judge-response must be supplied exactly three times")
            judge_raw = [Path(path).read_text() for path in mock_judge_response]

            def judge_runtime_factory(index):
                return MockAgentRuntime(lambda t, s, ts, i=index: (judge_raw[i - 1], (0, 0)))
        else:
            judge_runtime_factory = None
        runtime_provider = "mock"
        runtime_model = "mock"
    else:
        kind = _prov.select_provider_kind()
        if kind in _prov.LOCAL_KINDS:
            pf = _prov.preflight_local(probe=False)
            if pf.status != _prov.LOCAL_READY:
                emitter.emit("resource_pause", stage=stage_name,
                            detail={"status": pf.status, "reason": pf.reason})
                return StageRunResult("PROVIDER_UNAVAILABLE", EXIT_PROVIDER_UNAVAILABLE,
                                      f"provider unavailable: {pf.status}: {pf.reason}")
            if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                emitter.emit("approval_required", stage=stage_name,
                            detail={"status": "APPROVAL_REQUIRED",
                                    "approval_boundary": "PYDANTIC_AI_SMOKE_CONFIRM"})
                return StageRunResult(
                    "APPROVAL_REQUIRED", EXIT_APPROVAL_REQUIRED,
                    "APPROVAL_REQUIRED: set PYDANTIC_AI_SMOKE_CONFIRM=yes for one live local "
                    "inference call", approval_boundary="PYDANTIC_AI_SMOKE_CONFIRM")
            from .pydantic_ai_runtime import PydanticAIRuntime

            def live_runtime_factory(_index):
                return PydanticAIRuntime(
                    model=_prov.build_local_model(kind, pf.model_id, pf.base_url),
                    usage_source="provider")
            producer_runtime = live_runtime_factory(0)
            judge_runtime_factory = live_runtime_factory
            runtime_provider = kind
            runtime_model = pf.model_id
        elif kind in _prov.HOSTED_KINDS:
            pf = _prov.preflight_credentials(provider=kind)
            if pf.status != "READY":
                emitter.emit("resource_pause", stage=stage_name,
                            detail={"status": pf.status, "reason": pf.reason})
                return StageRunResult("PROVIDER_UNAVAILABLE", EXIT_PROVIDER_UNAVAILABLE,
                                      f"provider unavailable: {pf.status}: {pf.reason}")
            if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                emitter.emit("approval_required", stage=stage_name,
                            detail={"status": "APPROVAL_REQUIRED",
                                    "approval_boundary": "PYDANTIC_AI_SMOKE_CONFIRM"})
                return StageRunResult(
                    "APPROVAL_REQUIRED", EXIT_APPROVAL_REQUIRED,
                    "APPROVAL_REQUIRED: set PYDANTIC_AI_SMOKE_CONFIRM=yes for one live provider "
                    "call", approval_boundary="PYDANTIC_AI_SMOKE_CONFIRM")
            from .pydantic_ai_runtime import PydanticAIRuntime

            def live_runtime_factory(_index):
                return PydanticAIRuntime(model=_prov.build_provider_model(pf.model_id),
                                         usage_source="provider")
            producer_runtime = live_runtime_factory(0)
            judge_runtime_factory = live_runtime_factory
            runtime_provider = pf.provider
            runtime_model = pf.model_id
        else:
            emitter.emit("resource_pause", stage=stage_name,
                        detail={"status": "PROVIDER_UNAVAILABLE",
                                "reason": "PYDANTIC_AI_PROVIDER not set to a known provider"})
            return StageRunResult(
                "PROVIDER_UNAVAILABLE", EXIT_PROVIDER_UNAVAILABLE,
                "provider unavailable: set PYDANTIC_AI_PROVIDER to "
                "local-openai|ollama|anthropic|openai")

    ctx = ctx_factory(0, runtime_provider, runtime_model)
    producer_budget_policy = producer_context_policy(runtime_provider, runtime_model)
    if producer_budget_policy["source"] == "undeclared":
        return StageRunResult(
            PRODUCER_CONTEXT_WINDOW_UNDECLARED, EXIT_VALIDATION_REJECTED,
            f"{PRODUCER_CONTEXT_WINDOW_UNDECLARED}: no known/declared context window for model "
            f"'{runtime_model}'; set PYDANTIC_AI_CONTEXT_WINDOW_TOKENS explicitly before running "
            "this model in production")
    registry = build_executor_registry()
    binding_validator = _proposal_binding_validator(proposal, c)
    for descriptor in registry.values():
        descriptor.param_validator = binding_validator
    def producer_task_factory(feedback):
        return _producer_task(stage_name, role, evidence_path, c, proposal, retry_feedback=feedback)

    regate_only = _is_gate_only_correction_regate(c, stage_name)
    if regate_only:
        # Audited evidence-surfacing correction re-gate of an already-completed stage: re-surface
        # corrected evidence and re-judge the SAME accepted authoritative outputs WITHOUT
        # re-executing the (deterministic) production action -- directive "re-run ONLY the
        # evidence/gate path"; never rerun an executor merely to obtain a fresh Judge identity.
        emitter.emit("executor_skipped_regate_only", stage=stage_name, role=role,
                    detail={"iteration": c.state["iterations"][-1]["id"],
                            "reason": "evidence_surfacing_correction re-gate: stage already "
                                      "completed with declared outputs present; NOT re-executing "
                                      "the production action"})
        status = "DUPLICATE"
    else:
        readiness = _reference_validation_readiness(c, proposal)
        if readiness is not None and not readiness["ready"]:
            # Pre-costly evidence-readiness preflight (FE-032): the NON-Teacher gate evidence for a
            # Teacher-vs-DFT reference_validation (per-frame source->split lineage join, TEST-split
            # membership, structure/DFT-label/checkpoint identity + hashes, protected-reference
            # policy) must be establishable BEFORE any Teacher/GPU inference is dispatched. Fail
            # closed here so a run never spends expensive Teacher compute only to discover its gate
            # can never be satisfied (the ffv4g/ffv4h defect). Teacher-dependent numeric metrics
            # are the ONLY criteria left to post-execution.
            emitter.emit("campaign_blocked", stage=stage_name,
                        detail={"reason": "REFERENCE_VALIDATION_EVIDENCE_READINESS_INCOMPLETE",
                                "blocking_gaps": readiness["blocking_gaps"]})
            return StageRunResult(
                "REFERENCE_VALIDATION_EVIDENCE_READINESS_INCOMPLETE", EXIT_VALIDATION_REJECTED,
                "FAILED: reference_validation evidence readiness incomplete before costly Teacher "
                "dispatch: " + "; ".join(readiness["blocking_gaps"]))
        emitter.emit("executor_started", stage=stage_name, role=role,
                    action=proposal.get("action_type"))
        res = _run_producer_with_binding_retries(
            producer_runtime, task, specs[role], ctx, controller=c, registry=registry,
            authoritative_proposal=proposal, task_factory=producer_task_factory,
            budget_policy=producer_budget_policy, emitter=emitter, stage_name=stage_name)
        status = getattr(res.detail, "status", "")
    if status == "APPROVAL_REQUIRED":
        boundary = getattr(res.detail, "reason", "") or proposal.get("approval_boundary")
        emitter.emit("approval_required", stage=stage_name,
                    detail={"status": status, "approval_boundary": boundary})
        return StageRunResult("APPROVAL_REQUIRED", EXIT_APPROVAL_REQUIRED,
                              f"APPROVAL_REQUIRED: {boundary}", approval_boundary=boundary,
                              action_status=status)
    if status == "PENDING":
        # Same dispatch.py contract the recovery corrective-action path already relies on
        # (ExternalActionPending: "not a failure... the SAME idempotency key can be dispatched
        # again later to re-check") -- a forward stage's own primary action can hit this too (e.g.
        # a scheduler-bridge action still queued), and deserves the identical resumable-pause
        # treatment, not the terminal DISPATCH_REJECTED/FAILED outcome below.
        reason = getattr(res.detail, "reason", res.error)
        # The trusted executor WAS invoked (on_dispatch_start already marked the stage running),
        # so undo that mark back to pending -- ExternalActionPending is the one non-terminal
        # outcome, and this pre-R28 resumable-pause contract must not regress.
        if c.stage(stage_name)["status"] == "running":
            c.defer_stage_execution(stage_name, reason=reason)
        emitter.emit("resource_pause", stage=stage_name, detail={"status": status})
        return StageRunResult("EXTERNAL_ACTION_PENDING", EXIT_EXTERNAL_ACTION_PENDING,
                              f"EXTERNAL_ACTION_PENDING: {reason}", action_status=status)
    if status not in {"EXECUTED", "DUPLICATE"}:
        reason = getattr(res.detail, "reason", res.error)
        emitter.emit("executor_failed", stage=stage_name, role=role, detail={"status": status})
        if c.stage(stage_name)["status"] == "running":
            # The trusted executor was genuinely invoked (past every pre-executor rejection check)
            # and either raised or its own completion validator rejected the result. A definitive
            # TIMEOUT (forward-compatible, generic: the reason's leading exception name ends in
            # "TimeoutError" -- inert until such an exception exists) is exactly R28's regression
            # class -- land it on a terminal, representable status and force the campaign through
            # the same Controller-owned FAIL recovery path a Judge-scored gate would (the
            # GATE_{decision} shape below), rather than let an automated retry loop silently
            # re-dispatch a pathological/hanging plan forever. Every OTHER (ordinary, fast,
            # synchronous) EXECUTOR_ERROR/INVALID is deliberately reverted back to pending instead
            # -- an established, tested contract (direct ad-hoc run-stage retries after fixing the
            # underlying input, e.g. tests/test_base_plus_augmentation_dataset_route.py, must keep
            # working without going through propose_recovery/approve_recovery) -- attempts and the
            # stage_execution_started/deferred events remain a permanent, durable record that the
            # attempt occurred either way; only the resumability of `status` differs.
            exc_name = reason.split(":", 1)[0].strip() if isinstance(reason, str) else ""
            if exc_name.endswith("TimeoutError"):
                c.timeout_stage_execution(stage_name)
                c.record_gate(stage_name, "FAIL", evidence=f"executor failed: {status}: {reason}")
                return StageRunResult(
                    "GATE_FAIL", EXIT_VALIDATION_REJECTED,
                    "GATE_FAIL: recovery path is now controlled by the Controller",
                    gate_decision="FAIL", evidence_path=evidence_path, action_status=status)
            c.defer_stage_execution(stage_name, reason=reason)
        return StageRunResult("DISPATCH_REJECTED", EXIT_VALIDATION_REJECTED,
                              f"stage dispatch failed: {status}: {reason}", action_status=status)
    emitter.emit("executor_completed", stage=stage_name, role=role, detail={"status": status})

    declared = [(c.run_dir / rel).resolve() for rel in c.stage(stage_name).get("outputs", [])]
    missing = [str(path) for path in declared if not path.exists()]
    if missing:
        # The executor itself succeeded (EXECUTED), but the Controller's own declared-outputs
        # contract rejects registering it complete -- orthogonal to R28 (no hang, no unresolved
        # attempt), and an established, tested contract expects this to leave the stage retryable
        # exactly as before, not landed on a new terminal status.
        if c.stage(stage_name)["status"] == "running":
            c.defer_stage_execution(stage_name, reason="missing declared outputs")
        return StageRunResult("MISSING_OUTPUTS", EXIT_VALIDATION_REJECTED,
                              "stage missing declared outputs: " + ", ".join(missing),
                              action_status=status)
    if c.stage(stage_name)["status"] != "completed":
        try:
            c.complete_external_stage(stage_name, declared)
        except Exception:
            # Same reasoning as the missing-outputs branch immediately above: revert the running
            # mark so the stage stays retryable exactly as it was before this fix, then propagate
            # the original exception unchanged (existing callers, e.g. _cmd_run_stage, already
            # catch and report it identically).
            if c.stage(stage_name)["status"] == "running":
                c.defer_stage_execution(stage_name, reason="stage could not complete")
            raise
    c = RunController(c.run_dir)
    if c.stage(stage_name)["status"] != "completed":
        return StageRunResult("STAGE_NOT_COMPLETED", EXIT_VALIDATION_REJECTED,
                              "controller stage did not complete", action_status=status)
    emitter.emit("artifact_registered", stage=stage_name,
                detail={"artifacts": [str(p) for p in declared]})
    gate_split_manifests = _split_membership_manifest_sources(c)
    gate_evidence_artifacts = _committee_training_gate_evidence(
        c, declared, _selective_provenance_inputs(c, stage_name)) + gate_split_manifests
    gate_outcomes = _stage_validation_outcomes(c, stage_name)
    # Surface the deterministic per-criterion evidence record (source->split lineage join, TEST
    # membership, structure/DFT-label/checkpoint/prediction identity + hashes, no-historical-reuse,
    # protected-reference policy, and the post-execution global + grouped fidelity metrics with
    # units/denominators) so a Judge can VERIFY each reference_validation gate criterion against
    # deterministic evidence rather than infer it (FE-032). No-op for non-reference-validation stages.
    contract_manifest = (c.stage(stage_name).get("contract") or {}).get("manifest")
    report_path = (c.run_dir / contract_manifest) if contract_manifest else None
    criterion_evidence = _reference_validation_readiness(c, proposal, report_path=report_path)
    if criterion_evidence is not None:
        gate_outcomes = gate_outcomes + [{
            "stage": stage_name,
            "kind": "reference_validation_criterion_evidence",
            "ready": criterion_evidence["ready"],
            "teacher_metrics_pending": criterion_evidence["teacher_metrics_pending"],
            "blocking_gaps": criterion_evidence["blocking_gaps"],
            "criteria": criterion_evidence["criteria"],
        }]
    # Symmetric to the reference_validation surfacer above (FE-033): surface the deterministic
    # acquisition criterion-evidence (parent->pool join, per-parent deployment-domain mapping
    # resolved from the run's own bound frozen scope-classification evidence, and selection-control
    # attestations) so a Judge can VERIFY each acquisition gate criterion against deterministic
    # evidence rather than infer it from the raw manifest. No-op for non-acquisition stages.
    acquisition_evidence = _acquisition_readiness(c, proposal, report_path=report_path)
    if acquisition_evidence is not None:
        gate_outcomes = gate_outcomes + [{
            "stage": stage_name,
            "kind": "acquisition_criterion_evidence",
            "ready": acquisition_evidence["ready"],
            "pending_execution": acquisition_evidence["pending_execution"],
            "blocking_gaps": acquisition_evidence["blocking_gaps"],
            "criteria": acquisition_evidence["criteria"],
        }]
    # FE-053: surface the deterministic element->0-based-type-index mapping criterion evidence for a
    # Teacher-labeling gate (mirrors the reference_validation/acquisition surfacers above) so a Judge
    # can VERIFY the exact mapping against an authoritative deterministic result rather than only
    # observing that the manifest declares a species_mapping_evidence field. No-op for every stage
    # whose declared outputs record no species_mapping_evidence.
    species_mapping_evidence = _species_mapping_gate_evidence(c, stage_name, declared)
    if species_mapping_evidence is not None:
        gate_outcomes = gate_outcomes + [species_mapping_evidence]
    build_bounded_evidence(gate_evidence_artifacts, evidence_path,
                           protocol_refs=[c.state.get("workflow_config")],
                           validation_outcomes=gate_outcomes,
                           lineage_only=_gate_lineage_only_artifacts(gate_evidence_artifacts)
                           + gate_split_manifests)

    decision = "NO_GATE"
    vote_path = None
    if c.stage(stage_name).get("gate_criteria"):
        iteration = c.state["iterations"][-1]
        trigger = iteration.get("trigger")
        if (trigger and trigger.get("failed_stage") == stage_name and
                iteration.get("recovery_execution", {}).get("status") != "verified"):
            # The stage produced fresh output (needed so its own artifacts can later be checked
            # for real change), but its approved recovery's corrective actions have not yet been
            # verified from actual run state -- do not even attempt the judge gate, since
            # Controller.record_gate would refuse to record a PASS here anyway. run-campaign is
            # responsible for assembling and submitting the verification report next.
            return StageRunResult(
                "RECOVERY_EXECUTION_UNVERIFIED", EXIT_RECOVERY_EXECUTION_UNVERIFIED,
                f"RECOVERY_EXECUTION_UNVERIFIED: stage {stage_name!r} cannot be re-gated until "
                "its approved recovery's corrective actions are verified against actual run "
                "state", action_status=status)
        emitter.emit("judging_started", stage=stage_name)
        if auto_mock_judges:
            vote_path = _write_three_pass_votes(c, stage_name)
            decision = "PASS"
        else:
            if runtime == "mock" and judge_runtime_factory is None:
                raise ValueError(
                    "mock run-stage with a gate requires either --auto-mock-judges or three "
                    "--mock-judge-response files")
            gate_ctx = c.gate_context(stage_name)
            judge_allow = judge_read_allowlist(gate_ctx, evidence_path)

            def judge_ctx_factory(i):
                return RuntimeContext(exchange_dir=str(exchange), repo_root=repo_root,
                                      provider=runtime_provider, model_id=runtime_model,
                                      read_allow_prefixes=list(judge_allow), tools_enabled=False)
            try:
                decision, vote_path = run_three_judge_gate(
                    c, stage_name, specs, judge_runtime_factory, judge_ctx_factory, evidence_path,
                    emitter=emitter)
            except JudgeInvalidOutputBlocker as exc:
                # A lens's Judge kept emitting INVALID_JUDGE_OUTPUT until the bounded retry was
                # exhausted. No gate is recorded (an invalid output is never a vote), so no
                # recovery is triggered; this is a terminal blocker for human attention.
                return StageRunResult(
                    "JUDGE_INVALID_BLOCKER", EXIT_VALIDATION_REJECTED, str(exc),
                    evidence_path=evidence_path, action_status=status)
        c = RunController(c.run_dir)
        if c.stage(stage_name)["gate"] != decision:
            return StageRunResult("GATE_RECORD_MISMATCH", EXIT_VALIDATION_REJECTED,
                                  "controller gate did not record the aggregate decision",
                                  gate_decision=decision, evidence_path=evidence_path,
                                  action_status=status)
        emitter.emit("gate_recorded", stage=stage_name, detail={"decision": decision})
        if decision != "PASS":
            return StageRunResult(
                f"GATE_{decision}", EXIT_VALIDATION_REJECTED,
                f"GATE_{decision}: recovery path is now controlled by the Controller",
                gate_decision=decision, evidence_path=evidence_path, action_status=status)

    message = (f"stage: {stage_name}\naction_status: {status}\ngate: {decision}\n"
              f"bounded_evidence: {evidence_path}")
    if vote_path is not None:
        message = f"judge_votes: {vote_path}\n" + message
    return StageRunResult("SUCCESS", EXIT_SUCCESS, message, gate_decision=decision,
                          evidence_path=evidence_path, action_status=status)


def _cmd_run_stage(args) -> int:
    from workflow.controller import RunController

    c = RunController(args.run_dir)
    emitter = CampaignEventEmitter(c.run_dir, run_id=c.state.get("run_id"),
                                   quiet=getattr(args, "quiet", False),
                                   json_events=getattr(args, "json_events", False))
    try:
        result = run_production_stage(
            c, args.stage, runtime=args.runtime, agent_specs_dir=args.agent_specs_dir,
            exchange_dir=args.exchange_dir, repo_root=args.repo_root,
            auto_mock_judges=args.auto_mock_judges, mock_response=args.mock_response,
            mock_judge_response=args.mock_judge_response, emitter=emitter)
    except Exception as exc:
        print(f"run-stage failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_REJECTED
    stream = sys.stdout if result.exit_code == EXIT_SUCCESS else sys.stderr
    print(result.message, file=stream)
    return result.exit_code


# --- run-campaign: the outer, generic production loop -----------------------------------------
#
# COMPLETED, WAITING_FOR_HUMAN_APPROVAL, RECOVERY_REQUIRED, RESOURCE_BLOCKED,
# RECOVERY_EXECUTION_UNVERIFIED, WAITING_FOR_RECOVERY_EVIDENCE, WAITING_FOR_EXTERNAL_ACTION, and
# FAILED are the eight first-class outcomes a campaign run can stop at. None of them is a
# busy-loop: every one is derived from durable Controller state, so re-running the identical
# `run-campaign` command after the blocking condition resolves (an approval is granted, a provider
# comes back up, a recovery is approved, its corrective action finishes, a pending external/
# forward-stage action completes) resumes correctly without replaying already-passed stages or
# re-executing an already-completed action.
#
# An approved recovery's own corrective action -- when its plan names one -- is dispatched by THIS
# loop automatically (see _dispatch_recovery_corrective_action): a human approves scope/budget via
# `approve-recovery`, never the scientific action itself out of band. WAITING_FOR_RECOVERY_EVIDENCE
# is the clean pause for when that corrective action was legitimately dispatched and is still
# external/pending -- never a way to paper over a corrective action that reported done but left
# required outputs missing (that remains FAILED). WAITING_FOR_EXTERNAL_ACTION is the same pause,
# for a FORWARD stage's own primary action (e.g. a scheduler-bridge action still queued) rather
# than an approved recovery's corrective action -- see run_production_stage's `status == "PENDING"`
# handling, which mirrors dispatch.py's ExternalActionPending contract instead of collapsing it
# into the terminal DISPATCH_REJECTED/FAILED outcome.
CAMPAIGN_COMPLETED = "COMPLETED"
CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL = "WAITING_FOR_HUMAN_APPROVAL"
CAMPAIGN_RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
CAMPAIGN_RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
CAMPAIGN_RECOVERY_EXECUTION_UNVERIFIED = "RECOVERY_EXECUTION_UNVERIFIED"
CAMPAIGN_WAITING_FOR_RECOVERY_EVIDENCE = "WAITING_FOR_RECOVERY_EVIDENCE"
CAMPAIGN_WAITING_FOR_EXTERNAL_ACTION = "WAITING_FOR_EXTERNAL_ACTION"
CAMPAIGN_FAILED = "FAILED"


@dataclass
class CampaignRunResult:
    """Structured outcome of one ``run-campaign`` invocation: which of the five first-class
    outcomes it stopped at, why, and (if a stage actually ran) that stage's own StageRunResult."""
    outcome: str
    exit_code: int
    message: str
    stage: Optional[str] = None
    last_stage_result: Optional[StageRunResult] = None


def _next_eligible_stage(controller):
    """The ONE workflow-invariant "what's next" decision a campaign makes: the first declared
    stage (in workflow-config order) whose gate has not resolved as PASS or NOT_APPLICABLE, or
    None once every stage has. Contains no stage name, domain concept, or count -- entirely
    derived from the Controller's own stage list, the same order/gate fields ``_previous_passed``
    already enforces. NOT_APPLICABLE (see ``RunController.mark_stage_not_applicable``) is treated
    identically to PASS here: a stage the run's own evidence resolved as genuinely inapplicable is
    just as "done, may proceed downstream" as one that actually ran and passed."""
    for stage in controller.state["stages"]:
        if stage["gate"] not in ("PASS", "NOT_APPLICABLE"):
            return stage
    return None


def _stage_order_index(controller, stage_name):
    for index, stage in enumerate(controller.state["stages"]):
        if stage["name"] == stage_name:
            return index
    raise ValueError(f"unknown stage: {stage_name}")


def _assemble_recovery_execution_report(controller):
    """Mechanically assemble the ``verify_recovery_execution`` report from CURRENT controller/
    artifact state -- never from what the approved recovery plan merely says should happen. A
    change only counts once a registered, completed stage at or downstream of ``return_stage``
    has produced an artifact whose hash actually differs from (or is absent from) the iteration's
    frozen pre-recovery baseline. Returns ``(report, [])`` when every requirement the plan
    declares has real, current evidence, or ``(None, missing)`` naming exactly what real
    corrective evidence is not yet available -- callers must treat the latter as "not yet done",
    never as failure.
    """
    c = controller
    iteration = c.state["iterations"][-1]
    trigger = iteration.get("trigger")
    if not trigger or iteration.get("recovery_execution", {}).get("status") != "required":
        return None, ["no recovery execution is currently awaiting verification"]
    recovery = next(r for r in c.state["recoveries"] if r["id"] == trigger["recovery_id"])
    plan = recovery["plan"]
    return_stage = trigger["return_stage"]
    return_index = _stage_order_index(c, return_stage)
    baseline_hashes_by_path = {}
    for item in iteration.get("baseline_artifacts", []):
        baseline_hashes_by_path.setdefault(item["path"], set()).add(item["sha256"])

    changed_paths_by_stage = {}
    for stage in c.state["stages"]:
        if _stage_order_index(c, stage["name"]) < return_index or stage["status"] != "completed":
            continue
        for record in c.verify_stage_artifacts(stage["name"]):
            old_hashes = baseline_hashes_by_path.get(record["path"])
            if old_hashes is not None and record["sha256"] in old_hashes:
                continue  # unchanged relative to the frozen pre-recovery baseline
            changed_paths_by_stage.setdefault(stage["name"], []).append(record["path"])

    changed_paths = [p for paths in changed_paths_by_stage.values() for p in paths]
    if not changed_paths:
        return None, [
            f"no artifact at or downstream of return_stage {return_stage!r} has changed since "
            "the recovery baseline yet -- perform the approved corrective action(s), then rerun "
            "run-campaign"]

    missing = []

    def evidence_stage_for(flag_label):
        if return_stage not in changed_paths_by_stage:
            missing.append(f"{flag_label} is required by the plan but return_stage "
                           f"{return_stage!r} has no changed, completed evidence yet")
            return None
        return return_stage

    changes = [{"type": item["type"], "status": "APPLIED", "evidence_artifacts": list(changed_paths)}
              for item in plan["proposed_changes"]]

    labeling_plan = plan["labeling"]
    labeling_report = {"teacher_relabel": labeling_plan["teacher_relabel"],
                       "teacher_relabel_stage": None,
                       "new_dft": labeling_plan["new_dft"], "new_dft_stage": None}
    if labeling_plan["teacher_relabel"]:
        labeling_report["teacher_relabel_stage"] = evidence_stage_for("labeling.teacher_relabel")
    if labeling_plan["new_dft"]:
        labeling_report["new_dft_stage"] = evidence_stage_for("labeling.new_dft")

    training_plan = plan["student_training"]
    training_report = {"retrain": training_plan["retrain"], "mode": training_plan["mode"],
                       "stage": None}
    if training_plan["retrain"]:
        training_report["stage"] = evidence_stage_for("student_training.retrain")

    revalidation_plan = plan["revalidation"]
    # revalidation.targets are POST-return-stage revalidation: they only re-run AFTER the
    # return stage re-earns PASS, so they must NOT be required as pre-gate evidence here.
    # Requiring them would deadlock any recovery whose return_stage == failed_stage while that
    # stage's gate is still pending (the return-stage gate can't run until recovery is verified,
    # but recovery can't verify until the downstream targets -- which can't run until the gate
    # passes -- have changed). Record any target that already has changed evidence; defer the
    # rest to normal campaign progression once the return stage passes.
    revalidation_stages = [target for target in revalidation_plan["targets"]
                           if target in changed_paths_by_stage]
    revalidation_report = {"targets": list(revalidation_plan["targets"]), "stages": revalidation_stages}

    if missing:
        return None, missing

    report = {
        "schema_version": 1, "recovery_id": recovery["id"],
        "previous_iteration": iteration["parent_iteration"], "current_iteration": iteration["id"],
        "changes": changes, "labeling": labeling_report, "student_training": training_report,
        "revalidation": revalidation_report,
    }
    return report, []


def _assemble_run_summary_state(controller):
    """Mechanically assemble a bounded snapshot of CURRENT ``RunController`` state for the
    ``generate_run_summary`` executor -- an Analyst is never handed the raw run_dir; only this
    fixed, hash-bound projection of stages/gates/artifacts/recoveries, so nothing in the resulting
    report can be a narrative substitute for what the Controller actually recorded.
    """
    c = controller
    stages = [
        {"name": stage["name"], "status": stage["status"], "gate": stage["gate"],
         "artifacts": [{"path": record["path"], "sha256": record["sha256"]}
                      for record in c.stage_artifacts(stage["name"])]}
        for stage in c.state["stages"]
    ]
    gate_history = [
        {"stage": event["stage"], "verdict": event["verdict"], "at": event["at"]}
        for event in c.state.get("events", []) if event.get("type") == "gate"
    ]
    recoveries = [
        {"id": recovery["id"], "status": recovery["status"],
         "failed_stage": recovery["failed_stage"]}
        for recovery in c.state.get("recoveries", [])
    ]
    return {"run_id": c.state["run_id"], "stages": stages, "gate_history": gate_history,
            "recoveries": recoveries}


class _ProviderBlocked(Exception):
    """Raised by ``_select_reasoning_provider_runtime`` to unwind to a clean pause outcome instead
    of letting a provider-selection detail leak into the recovery-dispatch control flow."""
    def __init__(self, reason, message, approval_boundary=None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.approval_boundary = approval_boundary


def _select_reasoning_provider_runtime():
    """The exact provider selection/preflight/human-confirm gate ``run_production_stage``'s
    non-mock branch uses, factored out so a reasoning-role dispatch (Analyst/Orchestrator -- still
    no compute, no training, just typed reasoning output) shares the same operational rules
    instead of a second copy. Returns ``(runtime, provider, model_id)`` or raises
    ``_ProviderBlocked``."""
    from . import provider as _prov
    kind = _prov.select_provider_kind()
    if kind in _prov.LOCAL_KINDS:
        pf = _prov.preflight_local(probe=False)
        if pf.status != _prov.LOCAL_READY:
            raise _ProviderBlocked("PROVIDER_UNAVAILABLE",
                                   f"provider unavailable: {pf.status}: {pf.reason}")
        if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
            raise _ProviderBlocked(
                "APPROVAL_REQUIRED", "APPROVAL_REQUIRED: set PYDANTIC_AI_SMOKE_CONFIRM=yes for "
                "one live local inference call", "PYDANTIC_AI_SMOKE_CONFIRM")
        from .pydantic_ai_runtime import PydanticAIRuntime
        return (PydanticAIRuntime(model=_prov.build_local_model(kind, pf.model_id, pf.base_url),
                                  usage_source="provider"), kind, pf.model_id)
    if kind in _prov.HOSTED_KINDS:
        pf = _prov.preflight_credentials(provider=kind)
        if pf.status != "READY":
            raise _ProviderBlocked("PROVIDER_UNAVAILABLE",
                                   f"provider unavailable: {pf.status}: {pf.reason}")
        if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
            raise _ProviderBlocked(
                "APPROVAL_REQUIRED", "APPROVAL_REQUIRED: set PYDANTIC_AI_SMOKE_CONFIRM=yes for "
                "one live provider call", "PYDANTIC_AI_SMOKE_CONFIRM")
        from .pydantic_ai_runtime import PydanticAIRuntime
        return (PydanticAIRuntime(model=_prov.build_provider_model(pf.model_id),
                                  usage_source="provider"), pf.provider, pf.model_id)
    raise _ProviderBlocked(
        "PROVIDER_UNAVAILABLE",
        "provider unavailable: set PYDANTIC_AI_PROVIDER to local-openai|ollama|anthropic|openai")


def admissible_return_stages(stages, failed_stage) -> set:
    """The exact recovery return-stage subset ``RunController.propose_recovery`` can ever accept
    for a gate failure at ``failed_stage``: stages at or before it in declared workflow order
    (``propose_recovery`` rejects ``return_index > self._stage_index(failed_stage)`` -- see
    ``workflow/controller.py``, a frozen file this function deliberately never needs to touch).

    Both the Analyst's recovery-target context and the Orchestrator's return-stage context, and
    ``validate_recovery_plan_proposal``'s own contextual check, must be given exactly this subset
    -- never the full stage-name set. Passing the full set (the pre-fix defect: R20's Orchestrator
    proposed ``return_stage="reference_validation"``, downstream of the failed ``teacher_baseline``
    stage, and it was accepted here only to be rejected much later, deep inside
    ``propose_recovery``) lets a model choose a stage that can never actually be bound, silently
    deferring an avoidable rejection instead of preventing it.
    """
    ordered = [stage["name"] for stage in stages]
    failed_index = ordered.index(failed_stage)
    return set(ordered[:failed_index + 1])


def _stage_evidence_reveals_dft_comparison(artifact_paths) -> bool:
    """Deterministic, evidence-shape-based check: does the failed stage's OWN evidence contain an
    actual Teacher-vs-DFT comparison? Computed once per gate failure from
    ``pending_recovery["artifact_sha256"]``'s keys (the failed stage's own registered output
    artifacts -- never the Analyst's/Orchestrator's own claims) and threaded into both
    ``root_cause.validate_root_cause_classification`` and
    ``recovery_bridge.validate_recovery_plan_proposal`` as ``dft_comparison_evidence_present``.

    Reuses ``bounded_evidence.summarize_artifact``'s existing, generic per-artifact summaries
    rather than adding a second evidence reader: a teacher_baseline-profile JSON report's own
    ``dft_labels_used``/``protected_reference_labels_used`` fields (already computed from
    ``deployment_domain`` -- see ``bounded_evidence._teacher_baseline_report_summary``), or any
    ``.extxyz`` frame evidence whose ``label_keys`` name a dft-labeled channel, are the only two
    ways real Teacher-vs-DFT evidence can currently appear in a stage's own registered artifacts.
    A stage whose own evidence genuinely does recompute a Teacher-vs-DFT channel (e.g.
    reference_validation) trivially satisfies this check via its own artifacts -- this helper adds
    no stage-name special case, it only reads what is actually in the evidence.
    """
    from .bounded_evidence import summarize_artifact
    for path in artifact_paths:
        try:
            summary = summarize_artifact(path)
        except Exception:
            continue
        teacher_baseline = summary.get("teacher_baseline")
        if isinstance(teacher_baseline, dict) and (
                teacher_baseline.get("dft_labels_used") or
                teacher_baseline.get("protected_reference_labels_used")):
            return True
        label_keys = summary.get("label_keys") or []
        if any("dft" in str(key).lower() for key in label_keys):
            return True
    return False


def _pending_gate_vote_bundle(controller, pending):
    """The full validated Judge vote bundle (each vote's own ``rationale``/``required_fix``) for
    the CURRENT ``pending_recovery`` gate failure -- matched by stage name AND
    ``gate_recorded_at`` (not just stage name) so a stage that has failed more than once across
    iterations never picks up a stale vote bundle from an earlier gate failure. Returns None if
    the gate was recorded with no ``--votes`` bundle (a bare PASS-less verdict has no per-judge
    rationale to ground a recovery diagnosis in)."""
    failed_stage = pending["failed_stage"]
    gate_recorded_at = pending["gate_recorded_at"]
    for event in reversed(controller.state.get("events", [])):
        if (event.get("type") == "gate" and event.get("stage") == failed_stage
                and event.get("at") == gate_recorded_at):
            return event.get("vote_bundle")
    return None


# Free-text markers loosely indicating a judge's required_fix/rationale actually alleges a
# Teacher-vs-DFT accuracy/disagreement problem (as opposed to an evidence-exposure, provenance,
# or lineage/mapping-completeness gap) -- same "matched loosely" precedent as root_cause.py's
# _DFT_CHANNEL_MARKERS, and used for exactly the sibling defect class: R26 forensic finding, a
# REVISE whose every judge asked only for exposing counts/manifests/policy text (never for the
# underlying comparison, which was itself reported ok/passing) was still misclassified as a
# teacher-vs-DFT disagreement merely because DFT-comparison evidence was structurally present.
_ACCURACY_DISAGREEMENT_MARKERS = (
    "disagreement", "diverg", "inaccura", "systematic bias", "exceeds the gate threshold",
    "exceeds threshold", "exceeded the threshold", "exceeds the threshold",
)


def _gate_alleges_accuracy_disagreement(vote_bundle) -> bool:
    """Deterministic, text-shape-based check over the ACTUAL validated Judge vote bundle for the
    pending gate failure (see ``_pending_gate_vote_bundle`` -- never trusted from the Analyst's/
    Orchestrator's own claims): does any non-PASS judge's own ``rationale``/``required_fix``
    actually allege a Teacher-vs-DFT accuracy/disagreement problem? Threaded into both
    ``root_cause.validate_root_cause_classification`` and
    ``recovery_bridge.validate_recovery_plan_proposal`` as ``gate_alleges_accuracy_disagreement``,
    alongside (never instead of) ``_stage_evidence_reveals_dft_comparison``: that check answers
    "does the stage's evidence contain a DFT comparison at all", this one answers "did a judge
    actually say that comparison disagreed" -- a REVISE can satisfy the former and still fail the
    latter, exactly as demonstrated in R26 (a reference_validation REVISE whose own accuracy-
    computation criterion every judge marked ``ok: true``).

    No vote bundle at all (bare verdict, no ``--votes``) conservatively returns False: there is no
    judge rationale to ground a disagreement claim in.
    """
    for vote in (vote_bundle or {}).get("votes", []):
        if vote.get("verdict") == "PASS":
            continue
        text = f"{vote.get('rationale', '')} {vote.get('required_fix', '')}".lower()
        if any(marker in text for marker in _ACCURACY_DISAGREEMENT_MARKERS):
            return True
    return False


def _fe046_recovery_progress_check(controller, failed_stage, current_cov):
    """FE-046 invariant 2 (strict-reduction-or-fail-closed).

    A coverage-gap recovery (an FE-042 Stage-4 ``coverage_adequacy`` REVISE that routes to targeted
    reacquisition) must STRICTLY reduce the unsupported declared-class set every successful cycle.
    Compare the current cycle's unsupported set against the most recent DISTINCT prior coverage
    report's (both read from the persisted ``coverage_adequacy`` gate events -- never re-derived, so
    the ceiling survives stage invalidation). Return a fail-closed ``CampaignRunResult`` when the set
    did not shrink (no progress, or oscillation), else ``None`` to let the recovery proceed. Failing
    HERE -- before the Analyst/Orchestrator are dispatched -- is what stops an uncapped non-convergent
    reacquisition from burning hosted-provider credits indefinitely."""
    cur = set(current_cov.get("unsupported_structure_classes") or [])
    cur_sha = current_cov.get("report_sha256")
    prior = None
    for ev in reversed(controller.state.get("events", []) or []):
        if ev.get("type") != "gate" or ev.get("stage") != failed_stage:
            continue
        ca = ev.get("coverage_adequacy")
        if not isinstance(ca, dict) or ca.get("unsupported_structure_classes") is None:
            continue
        if ca.get("report_sha256") == cur_sha:
            continue  # same coverage report as the current cycle -- not a prior cycle
        prior = ca
        break
    if prior is None:
        return None  # first coverage-gap cycle: nothing to compare against
    prev = set(prior.get("unsupported_structure_classes") or [])
    if cur < prev:  # proper subset => the cycle genuinely reduced the unsupported set
        return None
    return CampaignRunResult(
        CAMPAIGN_FAILED, EXIT_BLOCKED_POLICY,
        "RECOVERY_NO_PROGRESS: FE-046 requires each coverage-gap recovery to STRICTLY reduce the "
        f"unsupported declared-class set, but it did not shrink (previous={sorted(prev)}, "
        f"current={sorted(cur)}). Failing closed instead of looping a non-convergent reacquisition.",
        stage=failed_stage)


def _propose_recovery_via_reasoning_roles(controller, *, runtime, agent_specs_dir, exchange_dir,
                                          repo_root, mock_analyst_response,
                                          mock_orchestrator_response,
                                          emitter=None) -> CampaignRunResult:
    """Turn a `pending_recovery` status=="required" gate into a proposed, human-approvable
    recovery: dispatch a real Analyst for a RootCauseClassification, then a real Orchestrator for
    a RecoveryPlanProposal bound to that exact diagnosis, then bind the resulting draft through
    the unchanged ``propose_recovery``/``dispatch_orchestrator_action`` bridge. This function
    authors no scientific judgment itself -- every diagnosis and every recovery choice comes from
    the dispatched agent roles; it only wires the existing, already-validated bridges together and
    always stops at human approval afterward, never granting it.
    """
    from orchestration.specs import load_agent_specs
    from .production_router import run_role
    from .models import RuntimeContext
    from .mock_runtime import MockAgentRuntime
    from .root_cause import validate_root_cause_classification
    from .recovery_bridge import (validate_recovery_plan_proposal,
                                  build_recovery_plan_draft_from_proposal,
                                  valid_corrective_actions_by_capability)
    from .executors import required_parameters_for_action
    from .orchestrator_bridge import OrchestratorActionProposal, dispatch_orchestrator_action
    from workflow.controller import DEFAULT_RECOVERY_CAPABILITY_ROSTER

    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    pending = c.state["pending_recovery"]
    failed_stage = pending["failed_stage"]
    # FE-046 invariant 2: for a coverage-gap recovery, fail closed BEFORE any (paid) LLM diagnosis/
    # proposal if the latest reacquisition did not strictly shrink the unsupported declared-class set.
    cov = pending.get("coverage_adequacy")
    if isinstance(cov, dict) and cov.get("unsupported_structure_classes") is not None:
        no_progress = _fe046_recovery_progress_check(c, failed_stage, cov)
        if no_progress is not None:
            emitter.emit("recovery_no_progress", stage=failed_stage,
                        detail={"unsupported_structure_classes":
                                sorted(cov.get("unsupported_structure_classes") or [])})
            return no_progress
    emitter.emit("recovery_started", stage=failed_stage)
    # Controller-admissible return-stage subset, surfaced to both the Analyst's recovery-target
    # context and the Orchestrator's return-stage context, and validated against below -- see
    # admissible_return_stages's docstring for why this must be a strict subset of all stage
    # names.
    stage_names = admissible_return_stages(c.state["stages"], failed_stage)
    # Deterministically computed from the failed stage's OWN evidence artifacts, never trusted
    # from the Analyst's/Orchestrator's own claims -- see the helper's docstring. Surfaced to both
    # roles' context (so a well-behaved model self-corrects) and enforced by both contextual
    # validators below (so a model that ignores the context still fails closed).
    dft_comparison_evidence_present = _stage_evidence_reveals_dft_comparison(
        pending["artifact_sha256"])
    # Deterministically computed from the ACTUAL Judge vote bundle for this gate failure -- never
    # trusted from the Analyst's/Orchestrator's own claims. See _gate_alleges_accuracy_disagreement's
    # docstring for why this is a distinct, narrower signal than dft_comparison_evidence_present.
    gate_vote_bundle = _pending_gate_vote_bundle(c, pending)
    gate_alleges_accuracy_disagreement = _gate_alleges_accuracy_disagreement(gate_vote_bundle)
    available_artifacts = {a["path"] for a in c.state["artifacts"]}
    roster = c.state.get("recovery_capability_roster") or DEFAULT_RECOVERY_CAPABILITY_ROSTER
    # FE-049: the required top-level parameters each dispatchable corrective action's deterministic
    # executor consumes, single-sourced from the executor registry's input_contract. Surfaced to the
    # Orchestrator's context (so a well-behaved model supplies them) AND enforced at acceptance by
    # validate_recovery_plan_proposal (so a model that ignores the context still fails closed before
    # a human approves an undispatchable plan). None entries are actions with no parseable parameter
    # contract (HPC/interface/reasoning) -- surfaced/enforced only where a real requirement exists.
    action_required_parameters = {
        action: required_parameters_for_action(action)
        for actions in valid_corrective_actions_by_capability(roster).values()
        for action in actions}
    action_required_parameters_context = {
        action: sorted(required)
        for action, required in action_required_parameters.items() if required}
    exchange = Path(exchange_dir) if exchange_dir else c.run_dir / "exchange"
    specs = load_agent_specs(agent_specs_dir)

    def ctx_factory(provider_name, model_id):
        return RuntimeContext(exchange_dir=str(exchange), repo_root=repo_root,
                              provider=provider_name, model_id=model_id,
                              read_allow_prefixes=[], tools_enabled=False)

    analyst_task = {
        "schema_version": 1, "task_id": f"{failed_stage}-recovery-diagnosis",
        "agent": "analyst", "run_id": c.state["run_id"], "created_at": "run-campaign",
        "instruction": (f"Diagnose the root cause of the REVISE/FAIL gate verdict recorded on "
                       f"stage {failed_stage!r} using context.recovery_evidence, the complete "
                       "primary evidence for this gate failure."),
        "inputs": [], "criteria": ["diagnosis is evidence-bound to registered artifacts",
                                   "diagnosis names an actionable recovery target"],
        "constraints": ["Cite only evidence_refs/affected_artifact_refs that are controller-"
                       "registered artifact paths already listed in context.recovery_evidence.",
                       "recommended_recovery_target must be one of "
                       "context.recovery_evidence.valid_recovery_targets -- already restricted "
                       "to the failed stage and stages at or before it in workflow order; a "
                       "downstream stage is never a valid recovery target.",
                       "if context.recovery_evidence.dft_comparison_evidence_present is false, "
                       "the failed stage's own evidence contains no Teacher-vs-DFT comparison -- "
                       "do not name a 'dft' affected_channel or choose failure_category "
                       "reference_disagreement; classify as an evidence/provenance gap instead "
                       "(e.g. evidence_gap or lineage_or_leakage)",
                       "context.recovery_evidence.gate_votes lists every judge's ACTUAL rationale "
                       "and required_fix for this gate failure -- ground the diagnosis in what "
                       "they actually wrote, never in an inference from evidence shape alone. If "
                       "context.recovery_evidence.gate_alleges_accuracy_disagreement is false, no "
                       "judge's required_fix/rationale alleges a Teacher-vs-DFT accuracy/"
                       "disagreement problem (their concerns are evidence-exposure, provenance, "
                       "or lineage/mapping-completeness gaps only) -- do not name a 'dft' "
                       "affected_channel or choose failure_category reference_disagreement merely "
                       "because DFT-comparison evidence exists in the stage's artifacts; classify "
                       "as the evidence/provenance gap the judges actually described instead"],
        "context": {"expected_output_model": "RootCauseClassification", "stage": failed_stage,
                   "recovery_evidence": {
                       "failed_stage": failed_stage, "verdict": pending["verdict"],
                       "gate_recorded_at": pending["gate_recorded_at"],
                       "artifact_sha256": pending["artifact_sha256"],
                       "available_artifacts": sorted(available_artifacts),
                       "valid_recovery_targets": sorted(stage_names),
                       "dft_comparison_evidence_present": dft_comparison_evidence_present,
                       "gate_alleges_accuracy_disagreement": gate_alleges_accuracy_disagreement,
                       # FE-042: present iff a deterministic Stage-4 coverage-adequacy control
                       # (not the Judges) forced this REVISE. It names the declared deployment
                       # structure classes with ZERO acquired representatives and the recommended
                       # return_stage, so the diagnosis routes to targeted reacquisition rather than
                       # a byte-identical data_coverage re-run.
                       "coverage_adequacy": pending.get("coverage_adequacy"),
                       "gate_votes": [
                           {"review_lens": vote.get("review_lens"), "verdict": vote.get("verdict"),
                            "rationale": vote.get("rationale"),
                            "required_fix": vote.get("required_fix")}
                           for vote in (gate_vote_bundle or {}).get("votes", [])
                       ],
                   }},
    }
    if runtime == "mock":
        if not mock_analyst_response:
            raise ValueError(
                "--mock-analyst-response is required: a recovery diagnosis is pending and "
                "--runtime mock cannot self-generate a RootCauseClassification")
        analyst_runtime = MockAgentRuntime(
            lambda t, s, ts: (Path(mock_analyst_response).read_text(), (0, 0)))
        analyst_provider, analyst_model = "mock", "mock"
    else:
        try:
            analyst_runtime, analyst_provider, analyst_model = _select_reasoning_provider_runtime()
        except _ProviderBlocked as exc:
            if exc.reason == "APPROVAL_REQUIRED":
                return CampaignRunResult(CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, EXIT_APPROVAL_REQUIRED,
                                         exc.message, stage=failed_stage)
            return CampaignRunResult(CAMPAIGN_RESOURCE_BLOCKED, EXIT_PROVIDER_UNAVAILABLE,
                                     exc.message, stage=failed_stage)

    def root_cause_validator(classification):
        return validate_root_cause_classification(
            classification, available_artifacts=available_artifacts,
            valid_recovery_targets=stage_names,
            dft_comparison_evidence_present=dft_comparison_evidence_present,
            gate_alleges_accuracy_disagreement=gate_alleges_accuracy_disagreement)

    emitter.emit("role_invocation_started", stage=failed_stage, role="analyst",
                action="recovery_diagnosis")
    analyst_res = run_role(analyst_runtime, analyst_task, specs["analyst"],
                           ctx_factory(analyst_provider, analyst_model), mode="primary",
                           reasoning_validator=root_cause_validator)
    emitter.emit("role_invocation_completed", stage=failed_stage, role="analyst",
                action="recovery_diagnosis", detail={"accepted": analyst_res.accepted})
    if not analyst_res.accepted:
        return CampaignRunResult(CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
                                 f"recovery diagnosis rejected: {analyst_res.error}",
                                 stage=failed_stage)
    classification = analyst_res.detail.instance
    diagnosis_path = analyst_res.detail.artifact_path
    diagnosis_sha256 = analyst_res.detail.artifact_sha256

    orchestrator_task = {
        "schema_version": 1, "task_id": f"{failed_stage}-recovery-plan",
        "agent": "orchestrator", "run_id": c.state["run_id"], "created_at": "run-campaign",
        "instruction": (f"Propose HOW to recover stage {failed_stage!r} from the evidence-bound "
                       "diagnosis in context.diagnosis."),
        "inputs": [], "criteria": ["proposal is bound to the supplied diagnosis artifact",
                                   "capability and return_stage are both registered"],
        "constraints": ["capability must be one of context.valid_capabilities",
                       "return_stage must be one of context.valid_stage_names -- this is already "
                       "the exact set of stages at or before the failed stage in workflow order; "
                       "a downstream stage can never be a valid return_stage and is rejected "
                       "before any diagnosis/proposal is even considered",
                       "diagnosis_artifact_sha256 must equal context.diagnosis_artifact_sha256",
                       "corrective_action, if set, must be {\"action_type\": ..., \"parameters\": "
                       "{...}} with action_type one of "
                       "context.valid_actions_by_capability[capability]; omit corrective_action "
                       "entirely if no registered action applies",
                       "corrective_action.parameters MUST include every parameter listed in "
                       "context.action_required_parameters[action_type] (if present there) -- a "
                       "corrective action whose action_type differs from the return stage's own "
                       "route action does not inherit that stage's parameters, so name every input "
                       "path it needs (e.g. manifest_path); a missing required parameter is rejected",
                       "if context.dft_comparison_evidence_present is false, do not set "
                       "labeling.new_dft, labeling.teacher_relabel, or student_training.retrain -- "
                       "the failed stage's own evidence contains no Teacher-vs-DFT comparison, so "
                       "none of those costly actions is evidence-justified; propose an "
                       "evidence-gathering corrective_action instead",
                       "if context.gate_alleges_accuracy_disagreement is false, do not set "
                       "labeling.new_dft, labeling.teacher_relabel, or student_training.retrain -- "
                       "no judge's required_fix/rationale (context.gate_votes) alleges a "
                       "Teacher-vs-DFT accuracy/disagreement problem, so none of those costly "
                       "actions is justified merely because DFT-comparison evidence exists; "
                       "propose an evidence-gathering corrective_action instead"],
        "context": {"expected_output_model": "RecoveryPlanProposal", "stage": failed_stage,
                   "diagnosis": json.loads(classification.model_dump_json()),
                   "diagnosis_artifact_sha256": diagnosis_sha256,
                   "valid_capabilities": sorted(roster), "valid_stage_names": sorted(stage_names),
                   "valid_actions_by_capability": valid_corrective_actions_by_capability(roster),
                   "action_required_parameters": action_required_parameters_context,
                   "dft_comparison_evidence_present": dft_comparison_evidence_present,
                   "gate_alleges_accuracy_disagreement": gate_alleges_accuracy_disagreement,
                   "gate_votes": [
                       {"review_lens": vote.get("review_lens"), "verdict": vote.get("verdict"),
                        "rationale": vote.get("rationale"), "required_fix": vote.get("required_fix")}
                       for vote in (gate_vote_bundle or {}).get("votes", [])
                   ]},
    }
    if runtime == "mock":
        if not mock_orchestrator_response:
            raise ValueError(
                "--mock-orchestrator-response is required: a recovery diagnosis is pending and "
                "--runtime mock cannot self-generate a RecoveryPlanProposal")
        orchestrator_runtime = MockAgentRuntime(
            lambda t, s, ts: (Path(mock_orchestrator_response).read_text(), (0, 0)))
        orchestrator_provider, orchestrator_model = "mock", "mock"
    else:
        try:
            (orchestrator_runtime, orchestrator_provider,
             orchestrator_model) = _select_reasoning_provider_runtime()
        except _ProviderBlocked as exc:
            if exc.reason == "APPROVAL_REQUIRED":
                return CampaignRunResult(CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, EXIT_APPROVAL_REQUIRED,
                                         exc.message, stage=failed_stage)
            return CampaignRunResult(CAMPAIGN_RESOURCE_BLOCKED, EXIT_PROVIDER_UNAVAILABLE,
                                     exc.message, stage=failed_stage)

    def plan_proposal_validator(proposal):
        # The return stage is chosen by the proposal itself, so its route facts (the same three the
        # controller's no-op materialization guard reads) can only be resolved once we have the
        # proposal in hand -- resolved here from the controller so acceptance rejects an
        # unmaterializable plan BEFORE binding, not at propose_recovery's exit-2 backstop (FE-038).
        return_stage = proposal.return_stage
        return validate_recovery_plan_proposal(
            proposal, expected_failed_stage=failed_stage, expected_diagnosis_sha256=diagnosis_sha256,
            capability_roster=roster, valid_stage_names=stage_names,
            dft_comparison_evidence_present=dft_comparison_evidence_present,
            gate_alleges_accuracy_disagreement=gate_alleges_accuracy_disagreement,
            return_stage_route_action=c._stage_route_action(return_stage),
            return_stage_route_parameters=c._stage_route_parameters(return_stage),
            return_stage_replans=c._return_stage_replans_on_recovery(return_stage),
            action_required_parameters=action_required_parameters)

    emitter.emit("role_invocation_started", stage=failed_stage, role="orchestrator",
                action="recovery_plan_proposal")
    orchestrator_res = run_role(orchestrator_runtime, orchestrator_task, specs["orchestrator"],
                                ctx_factory(orchestrator_provider, orchestrator_model),
                                mode="primary", reasoning_validator=plan_proposal_validator)
    emitter.emit("role_invocation_completed", stage=failed_stage, role="orchestrator",
                action="recovery_plan_proposal", detail={"accepted": orchestrator_res.accepted})
    if not orchestrator_res.accepted:
        return CampaignRunResult(CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
                                 f"recovery plan proposal rejected: {orchestrator_res.error}",
                                 stage=failed_stage)
    proposal = orchestrator_res.detail.instance

    draft = build_recovery_plan_draft_from_proposal(
        classification, proposal,
        proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"},
        diagnosis_artifact_path=str(diagnosis_path), diagnosis_artifact_sha256=diagnosis_sha256,
        artifact_sha256_lookup={a["path"]: a["sha256"] for a in c.state["artifacts"]})
    plan_dir = c.run_dir / "recovery" / "drafts"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{failed_stage}.recovery_plan.draft.json"
    plan_path.write_text(json.dumps(draft.to_plan_json(), indent=2) + "\n")

    action_proposal = OrchestratorActionProposal(
        run_id=c.state["run_id"], stage=failed_stage, requested_at="run-campaign",
        rationale=f"propose recovery for stage {failed_stage!r} from a validated diagnosis",
        idempotency_key=f"{c.state['run_id']}:{failed_stage}:recovery-proposal:001",
        action_type="propose_recovery",
        parameters={"run_dir": str(c.run_dir), "plan_path": str(plan_path)})
    outcome = dispatch_orchestrator_action(action_proposal, controller=c, mode="primary")
    if outcome.status != "EXECUTED":
        return CampaignRunResult(CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
                                 f"propose_recovery dispatch failed: {outcome.status}: "
                                 f"{outcome.reason}", stage=failed_stage)

    recovery_id = outcome.artifact["recovery_id"]
    emitter.emit("recovery_proposed", stage=failed_stage, detail={"recovery_id": recovery_id})
    return CampaignRunResult(
        CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, EXIT_APPROVAL_REQUIRED,
        f"WAITING_FOR_HUMAN_APPROVAL: recovery {recovery_id} for stage {failed_stage!r} has been "
        "proposed and is awaiting human approval (see `approve-recovery`)", stage=failed_stage)


class _RecoveryBaselineRestoreError(Exception):
    """A distinct_evidence_artifact recovery could not restore the return stage's declared outputs
    byte-identically from the frozen recovery baseline."""


def _find_quarantined_baseline(stale_root, stage, name, want_sha256):
    """Locate the byte-identical quarantined copy of a return-stage declared output that
    ``invalidate_from``/``quarantine_artifacts`` moved into ``run_dir/stale/<stamp>/<stage>/<name>``
    when the recovery iteration started. Matched on BOTH (stage subdir, file name) AND sha256, so a
    like-named file from another stage or a content-different quarantine is never mistaken for it.
    Returns the ``Path`` or ``None``."""
    from workflow.integrity import sha256_file
    if not stale_root.is_dir():
        return None
    for stamp_dir in sorted(stale_root.iterdir()):
        candidate = stamp_dir / stage / name
        if candidate.is_file() and sha256_file(candidate) == want_sha256:
            return candidate
    return None


def _restore_return_stage_baseline_outputs(controller, return_stage, iteration):
    """FE-050: restore ``return_stage``'s declared outputs byte-identically from the iteration's
    frozen pre-recovery ``baseline_artifacts`` (the physical bytes were quarantined into
    ``run_dir/stale/`` by ``start_iteration``'s ``invalidate_from``).

    A ``distinct_evidence_artifact`` recovery's corrective action dispatches an executor DISTINCT
    from the return stage's own route action -- it produces NEW evidence and never re-emits the
    stage's declared route outputs. Those declared outputs were quarantined when the recovery
    iteration started, so without restoration the corrective dispatch's declared-outputs check
    fails MISSING_OUTPUTS even though the recovery is proceeding exactly as approved. Restoration is
    strictly byte-identical: each restored file's sha256 must equal its frozen baseline sha256, and
    the source must be the quarantined baseline copy -- never a re-derivation. Fails closed
    (``_RecoveryBaselineRestoreError``) if any declared output has no frozen baseline or no
    byte-identical quarantined copy, so nothing can silently fabricate a stage output."""
    from workflow.integrity import sha256_file
    c = controller
    baseline_by_path = {
        Path(record["path"]).resolve(): record["sha256"]
        for record in iteration.get("baseline_artifacts", [])
        if record["stage"] == return_stage
    }
    declared = [(c.run_dir / rel).resolve() for rel in c.stage(return_stage).get("outputs", [])]
    stale_root = c.run_dir / "stale"
    restored = []
    for dest in declared:
        if dest.exists():
            restored.append(dest)
            continue
        want_sha256 = baseline_by_path.get(dest)
        if want_sha256 is None:
            raise _RecoveryBaselineRestoreError(
                f"declared output {dest} has no frozen recovery baseline to restore from")
        source = _find_quarantined_baseline(stale_root, return_stage, dest.name, want_sha256)
        if source is None:
            raise _RecoveryBaselineRestoreError(
                f"no byte-identical quarantined baseline (sha256 {want_sha256}) found for {dest}")
        import shutil
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        got_sha256 = sha256_file(dest)
        if got_sha256 != want_sha256:
            raise _RecoveryBaselineRestoreError(
                f"restored baseline for {dest} has sha256 {got_sha256}, expected {want_sha256}")
        restored.append(dest)
    return restored


def _corrective_evidence_artifact_path(controller, outcome, params):
    """Resolve the run-local artifact a distinct-evidence corrective action produced (from the
    dispatch ``outcome.artifact`` when freshly EXECUTED, else the corrective's declared
    ``out_path`` on a resumed DUPLICATE). Returns a resolved ``Path`` that exists, or ``None``."""
    candidate = None
    artifact = getattr(outcome, "artifact", None)
    if isinstance(artifact, dict):
        candidate = artifact.get("path")
    if not candidate:
        candidate = params.get("out_path")
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = controller.run_dir / path
    path = path.resolve()
    return path if path.exists() else None


def _dispatch_recovery_corrective_action(controller, trigger, recovery, corrective_action,
                                         *, registry=None,
                                         emitter=None) -> Optional["CampaignRunResult"]:
    """After an approved recovery's ``start_iteration()`` has quarantined ``return_stage`` (and
    everything after it) back to pending, actually perform the ONE corrective action the approved
    plan itself named -- through the SAME dispatch/controller-bridge path any other action goes
    through -- so a human never has to run it out of band after approving scope/budget. Returns
    ``None`` to mean "corrective work for this iteration is done, keep going", or a terminal/paused
    ``CampaignRunResult`` to stop at.

    ``registry`` defaults to the real production ``executors.build_executor_registry()`` (the same
    one ``run_production_stage`` uses for every other action) -- callers may pass a substituted
    registry only to exercise deterministic/pending/failure fixture executors in tests.

    Generic over ``corrective_action`` (an ``{"action_type", "role"?, "parameters"?}`` dict read
    straight off the approved plan's ``recovery_context`` -- see ``recovery_bridge.
    validate_recovery_plan_proposal``): this function contains no stage name, capability, or
    domain concept of its own.

    Deliberately reuses ``_proposal_from_stage``'s OWN per-iteration idempotency key for
    ``return_stage`` (not a separately-suffixed one): once this dispatch executes, the campaign's
    normal forward loop will re-select ``return_stage`` again (its gate is not yet PASS) and call
    ``run_production_stage`` on it as usual -- that call derives the IDENTICAL key, so it sees an
    already-seen action and gets DUPLICATE (a no-op) instead of a second, real re-execution of the
    stage's own route action, and proceeds straight to checking declared outputs and gating.
    """
    from .controller_bridge import dispatch_via_controller
    from .executors import build_executor_registry
    from workflow.controller import DEFAULT_RECOVERY_CAPABILITY_ROSTER

    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    return_stage = trigger["return_stage"]
    base_proposal, base_role = _proposal_from_stage(c, return_stage, _stage_config(c, return_stage))
    roster = c.state.get("recovery_capability_roster") or DEFAULT_RECOVERY_CAPABILITY_ROSTER
    role = (corrective_action.get("role") or
           roster.get(recovery["plan"]["responsible_capability"]) or base_role)
    # Parameter assembly: when the approved corrective action simply RE-RUNS the return
    # stage's OWN route action (same action_type), start from that stage's canonical,
    # controller-resolved parameters (student_config/dataset/output_dir/manifest_path,
    # protected reference_yaml, ...) and let the plan's corrective_action.parameters
    # OVERRIDE/EXTEND them (e.g. a continuation's continue_from map + total_epoch_override).
    # This keeps the plan from having to re-derive protection-resolved paths itself. When
    # the corrective action is a DIFFERENT action_type (e.g. an evidence-gathering action),
    # the base stage's parameters do not apply, so only the plan-supplied parameters are used.
    corrective_params = dict(corrective_action.get("parameters") or {})
    if corrective_action["action_type"] == base_proposal.get("action_type"):
        params = dict(base_proposal.get("parameters") or {})
        params.update(corrective_params)
    else:
        params = corrective_params
    # FE-052: a distinct_evidence_artifact corrective MUST materialize a NEW run-local artifact so
    # verify_recovery_execution detects a change at the return stage (see FE-050). The evidence
    # executors that satisfy this recovery class (e.g. validate_species_mapping_consistency) write
    # their report only to an OPTIONAL ``out_path``; because it is optional it is absent from the
    # executor's REQUIRED-parameter contract, so FE-049's acceptance-time param check does not force
    # it and an approved plan legitimately may omit it (the LLM proposed only ``manifest_path``).
    # When it is missing, inject a deterministic, collision-free, run-local ``out_path`` BEFORE
    # dispatch so the executor materializes its distinct evidence deterministically -- no LLM path
    # authoring, no stage/action/material-specific knowledge. A no-op when the plan already supplied
    # ``out_path`` or the recovery is not distinct-evidence.
    if (recovery.get("materialization_transition") == "distinct_evidence_artifact"
            and not params.get("out_path")):
        params["out_path"] = str(
            c.run_dir / "artifacts"
            / f"{corrective_action['action_type']}.recovery-{recovery['id']:03d}.evidence.json")
    proposal = {
        "run_id": c.state["run_id"], "stage": return_stage,
        "requested_by_role": role, "action_type": corrective_action["action_type"],
        "requested_at": "run-campaign",
        "rationale": f"execute recovery {recovery['id']}'s approved corrective action",
        "idempotency_key": base_proposal["idempotency_key"],
        "parameters": params,
    }
    # FE-044: resolve this corrective proposal's approval boundary through the SAME canonical typed-
    # effect binding the forward acquisition path uses (run_production_stage ->
    # _bind_acquisition_plan_for_stage). Without it a corrective ``acquire_structures`` carries no
    # ``performs_teacher_inference`` effect signal, so actions.resolve_action_approval_boundary fail-
    # closes and a geometry-only / existing-pool reacquisition spuriously trips the
    # ``costly_teacher_labeling`` gate even though forward dispatch of the IDENTICAL bound plan does
    # not. The bound AcquisitionPlan stays authoritative: a Teacher-driven recipe still classifies
    # True and keeps the costly boundary; only an effect the framework proves Teacher-free is relaxed.
    # A no-op for every non-``acquire_structures`` corrective action (the binder returns unchanged).
    try:
        proposal = _bind_acquisition_plan_for_stage(c, proposal)
    except ValueError as exc:
        # The stage has not begun execution yet (begin_stage_execution runs inside
        # dispatch_via_controller below), so a binding rejection is a clean pre-dispatch failure.
        emitter.emit("executor_failed", stage=return_stage, role=role,
                    detail={"status": "VALIDATION_REJECTED"})
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            f"recovery corrective action dispatch failed: {exc}", stage=return_stage)
    # FE-050 (reordered per FE-051): a distinct_evidence_artifact recovery's corrective action
    # dispatches an executor DISTINCT from the return stage's own route action, and that executor may
    # itself READ the return stage's declared outputs (e.g. validate_species_mapping_consistency reads
    # the teacher labeling manifest to expose its species->type-index mapping). start_iteration
    # quarantined those declared outputs into run_dir/stale/ when the recovery iteration began, so
    # they must be restored BYTE-IDENTICALLY from the frozen recovery baseline BEFORE the corrective
    # executor runs -- otherwise the executor fails reading a now-quarantined input. Restoring a
    # byte-identical baseline output leaves its sha256 in the iteration baseline (unchanged), so it is
    # NOT what verify_recovery_execution keys on; the corrective's OWN distinct artifact, registered
    # ADDITIVELY after dispatch (below), is the sole detected change. No Teacher inference / route
    # re-run happens here. A no-op for every non-distinct recovery.
    if recovery.get("materialization_transition") == "distinct_evidence_artifact":
        try:
            _restore_return_stage_baseline_outputs(c, return_stage, c.state["iterations"][-1])
        except _RecoveryBaselineRestoreError as exc:
            if c.stage(return_stage)["status"] == "running":
                c.defer_stage_execution(return_stage, reason="baseline restore failed")
            return CampaignRunResult(
                CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
                f"recovery corrective action for stage {return_stage!r} could not restore its "
                "declared outputs byte-identically from the frozen recovery baseline before "
                f"dispatch: {exc}", stage=return_stage)
    registry = registry if registry is not None else build_executor_registry()
    emitter.emit("executor_started", stage=return_stage, role=role,
                action=corrective_action["action_type"])

    def _progress_cb(progress: dict, _emitter=emitter, _stage=return_stage, _role=role,
                     _action=corrective_action["action_type"], _controller=c) -> None:
        _emitter.emit("executor_progress", stage=_stage, role=_role, action=_action,
                     detail=progress)
        # Same additive Controller-durable heartbeat as run_production_stage's own dispatch path
        # (see workflow.controller.heartbeat_stage): a no-op unless begin_stage_execution already
        # marked this stage running.
        _controller.heartbeat_stage(_stage, progress=progress, pid=progress.get("pid")
                                    if isinstance(progress, dict) else None)

    def _on_dispatch_start(_controller=c, _stage=return_stage, _role=role,
                           _action=corrective_action["action_type"], _proposal=proposal) -> None:
        # Same R28-defect fix as the forward-dispatch path (_run_producer_with_binding_retries):
        # fires only once dispatch.authorize_and_execute is about to invoke the real trusted
        # executor, so the attempt is durably recorded before it can hang. See
        # workflow.controller.begin_stage_execution.
        try:
            from .executors import acquisition_plan_sha256_from_proposal
            plan_sha256 = acquisition_plan_sha256_from_proposal(_proposal)
        except Exception:
            plan_sha256 = None
        _controller.begin_stage_execution(_stage, runner_id="pydantic_ai",
                                          executor=f"{_role}:{_action}",
                                          plan_sha256=plan_sha256)

    outcome = dispatch_via_controller(proposal, controller=c, registry=registry, mode="primary",
                                      progress_cb=_progress_cb, on_dispatch_start=_on_dispatch_start)
    if outcome.status == "PENDING":
        # Same resumable-pause contract as run_production_stage: the executor was genuinely
        # invoked (on_dispatch_start already marked the stage running), so undo that mark back to
        # pending -- ExternalActionPending is the one non-terminal outcome.
        if c.stage(return_stage)["status"] == "running":
            c.defer_stage_execution(return_stage, reason=outcome.reason)
        emitter.emit("executor_pending", stage=return_stage, role=role,
                    detail={"status": outcome.status})
        return CampaignRunResult(
            CAMPAIGN_WAITING_FOR_RECOVERY_EVIDENCE, EXIT_RECOVERY_ACTION_PENDING,
            f"WAITING_FOR_RECOVERY_EVIDENCE: recovery {recovery['id']}'s corrective action for "
            f"stage {return_stage!r} has been dispatched and is still pending: {outcome.reason}",
            stage=return_stage)
    if outcome.status not in {"EXECUTED", "DUPLICATE"}:
        emitter.emit("executor_failed", stage=return_stage, role=role,
                    detail={"status": outcome.status})
        if c.stage(return_stage)["status"] == "running":
            # Same TIMEOUT-vs-ordinary distinction as run_production_stage: a genuine timeout is
            # R28's regression class and must land on a terminal, Controller-recorded state; any
            # other (ordinary, fast, synchronous) rejection reverts to pending instead, so a
            # subsequent recovery iteration or ad-hoc retry is not blocked by a stray "running"
            # stage left over from this dispatch attempt.
            exc_name = outcome.reason.split(":", 1)[0].strip() if isinstance(outcome.reason, str) else ""
            if exc_name.endswith("TimeoutError"):
                c.timeout_stage_execution(return_stage)
            else:
                c.defer_stage_execution(return_stage, reason=outcome.reason)
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            f"recovery corrective action dispatch failed: {outcome.status}: {outcome.reason}",
            stage=return_stage)
    emitter.emit("executor_completed", stage=return_stage, role=role,
                detail={"status": outcome.status})
    # FE-050: the distinct_evidence_artifact corrective produced NEW evidence and never re-emitted the
    # return stage's declared route outputs. Those outputs were already restored BYTE-IDENTICALLY from
    # the frozen baseline BEFORE dispatch (see the FE-051 reorder above), so the declared-outputs
    # check below passes. Here we carry the corrective's own distinct evidence artifact as ADDITIVE
    # return-stage evidence so the stage's registered artifact set differs from the baseline -- exactly
    # what verify_recovery_execution requires to accept the corrective as materialized.
    additive_evidence = []
    if recovery.get("materialization_transition") == "distinct_evidence_artifact":
        corrective_artifact = _corrective_evidence_artifact_path(c, outcome, params)
        if corrective_artifact is None:
            if c.stage(return_stage)["status"] == "running":
                c.defer_stage_execution(return_stage, reason="no corrective evidence artifact")
            return CampaignRunResult(
                CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
                f"recovery corrective action for stage {return_stage!r} reported {outcome.status} "
                "but produced no registrable distinct evidence artifact (a distinct_evidence "
                "recovery must emit an out_path artifact to materialize a change at the return "
                "stage)", stage=return_stage)
        declared_set = {(c.run_dir / rel).resolve()
                        for rel in c.stage(return_stage).get("outputs", [])}
        if corrective_artifact not in declared_set:
            additive_evidence.append(corrective_artifact)
    declared = [(c.run_dir / rel).resolve() for rel in c.stage(return_stage).get("outputs", [])]
    missing = [str(path) for path in declared if not path.exists()]
    if missing:
        # The corrective action itself reports done (EXECUTED) or was already recorded as done
        # (DUPLICATE) for this exact recovery iteration -- either way that is a completed dispatch
        # by definition, so required outputs still being absent is a genuine failure, never a
        # reason to sit in a pause waiting for evidence that will never arrive.
        if c.stage(return_stage)["status"] == "running":
            c.defer_stage_execution(return_stage, reason="missing declared outputs")
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            f"recovery corrective action for stage {return_stage!r} reported {outcome.status} but "
            "declared outputs are still missing: " + ", ".join(missing), stage=return_stage)
    if c.stage(return_stage)["status"] != "completed":
        try:
            c.complete_external_stage(return_stage, declared + additive_evidence)
        except Exception:
            if c.stage(return_stage)["status"] == "running":
                c.defer_stage_execution(return_stage, reason="stage could not complete")
            raise
    emitter.emit("artifact_registered", stage=return_stage,
                detail={"artifacts": [str(p) for p in declared + additive_evidence]})
    return None


def _commit_teacher_validation_plan_via_reasoning_roles(
    controller, *, runtime, agent_specs_dir, exchange_dir, repo_root,
    mock_orchestrator_response, emitter=None,
) -> Optional["CampaignRunResult"]:
    """Automatic pre-campaign Teacher-validation planning: deterministically inspect this run's
    own frozen ``teacher_evidence_sources``, dispatch a real Orchestrator for a
    ``TeacherValidationPlanProposal`` (which admissible component(s) this campaign will actually
    USE -- the one genuinely scientific choice
    ``inspect_teacher_evidence``/``derive_admissible_decision_space`` cannot make
    deterministically), then bind the resulting draft through
    ``commit_teacher_validation_plan``. This function authors no scientific judgment itself --
    the component selection comes entirely from the dispatched Orchestrator; it only wires the
    existing, already-validated bridge together.

    A no-op (returns None immediately) whenever this run declares no ``teacher_evidence_sources``
    (the whole pipeline is opt-in) or a plan is already committed (write-once, idempotent).
    Returns None on success (a plan is now committed, campaign dispatch may proceed); returns a
    terminal/pausing ``CampaignRunResult`` otherwise.
    """
    import dataclasses

    from orchestration.specs import load_agent_specs
    from validation.teacher_evidence_profile import (
        derive_admissible_decision_space, inspect_teacher_evidence,
    )

    from .mock_runtime import MockAgentRuntime
    from .models import RuntimeContext
    from .orchestrator_bridge import OrchestratorActionProposal, dispatch_orchestrator_action
    from .production_router import run_role
    from .teacher_validation_plan import (
        TEACHER_VALIDATION_OBJECTIVE_SEMANTICS,
        build_teacher_validation_plan_draft_from_proposal,
        validate_teacher_validation_plan_proposal,
    )

    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    sources = c.state.get("teacher_evidence_sources")
    if sources is None or c.state.get("teacher_validation_plan") is not None:
        return None

    profile, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
    decision_space = derive_admissible_decision_space(profile)
    if decision_space["insufficient_evidence"]:
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            "FAILED: this run's teacher_evidence_sources admit NO Teacher-validation component "
            "at all (CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING) -- there is nothing for an "
            "autonomous Teacher validation plan to select")

    objectives = c._teacher_validation_objectives()
    exchange = Path(exchange_dir) if exchange_dir else c.run_dir / "exchange"
    specs = load_agent_specs(agent_specs_dir)

    def ctx_factory(provider_name, model_id):
        return RuntimeContext(exchange_dir=str(exchange), repo_root=repo_root,
                              provider=provider_name, model_id=model_id,
                              read_allow_prefixes=[], tools_enabled=False)

    # Generic (never SiO2/run-specific) per-objective rule text -- the SAME dict
    # ``validate_teacher_validation_plan_proposal`` is implemented against, so the task a planner
    # is given can never omit a contract dimension the authoritative validator actually enforces.
    objective_semantics = {obj: TEACHER_VALIDATION_OBJECTIVE_SEMANTICS[obj]
                           for obj in objectives if obj in TEACHER_VALIDATION_OBJECTIVE_SEMANTICS}

    def build_task(prior_rejection=None):
        context = {"expected_output_model": "TeacherValidationPlanProposal",
                  "evidence_profile_sha256": evidence_profile_sha256,
                  "admissible_decision_space": decision_space,
                  "validation_objectives": objectives,
                  "validation_objective_semantics": objective_semantics}
        instruction = ("Decide WHICH admissible Teacher-validation component(s) this campaign "
                      "will actually use, from context.admissible_decision_space -- the evidence "
                      "only establishes what is POSSIBLE, never which of it to use.")
        if prior_rejection is not None:
            context["prior_attempt_rejection"] = prior_rejection
            instruction += (" This is a correction of a prior proposal the authoritative "
                            "validator rejected for the exact reason recorded in "
                            "context.prior_attempt_rejection -- do not repeat it.")
        return {
            "schema_version": 1, "task_id": f"{c.state['run_id']}-teacher-validation-plan",
            "agent": "orchestrator", "run_id": c.state["run_id"], "created_at": "run-campaign",
            "instruction": instruction,
            "inputs": [],
            "criteria": ["selected_components is a non-empty subset of admissible_components",
                        "selected_components jointly satisfies every objective in "
                        "context.validation_objectives that this evidence profile triggers -- "
                        "see context.validation_objective_semantics for each declared "
                        "objective's exact, evidence-conditional rule",
                        "rationale is evidence-bound"],
            "constraints": [
                "selected_components must be drawn only from context.admissible_decision_space."
                "admissible_components -- never a component this evidence does not support",
                "evidence_profile_sha256 must equal context.evidence_profile_sha256",
                "every objective in context.validation_objectives that context.validation_"
                "objective_semantics marks as triggered by this evidence profile must be "
                "satisfied by selected_components -- an admissible-but-objective-violating "
                "selection is rejected exactly like an inadmissible one, never overridable",
            ],
            "context": context,
        }

    task = build_task()
    if runtime == "mock":
        if not mock_orchestrator_response:
            raise ValueError(
                "--mock-orchestrator-response is required: teacher_evidence_sources are "
                "declared for this run and --runtime mock cannot self-generate a "
                "TeacherValidationPlanProposal")
        # Comma-separated list of response files is accepted so a test can simulate the bounded
        # semantic-correction retry below (attempt N reads the Nth path, holding on the last path
        # once exhausted); a single path (the common case) behaves exactly as before.
        mock_response_paths = [Path(p) for p in str(mock_orchestrator_response).split(",")]
        mock_attempt_counter = {"n": 0}

        def _next_mock_response(_t, _s, _ts):
            idx = min(mock_attempt_counter["n"], len(mock_response_paths) - 1)
            mock_attempt_counter["n"] += 1
            return mock_response_paths[idx].read_text(), (0, 0)

        orchestrator_runtime = MockAgentRuntime(_next_mock_response)
        orchestrator_provider, orchestrator_model = "mock", "mock"
    else:
        try:
            (orchestrator_runtime, orchestrator_provider,
             orchestrator_model) = _select_reasoning_provider_runtime()
        except _ProviderBlocked as exc:
            if exc.reason == "APPROVAL_REQUIRED":
                return CampaignRunResult(CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL,
                                         EXIT_APPROVAL_REQUIRED, exc.message)
            return CampaignRunResult(CAMPAIGN_RESOURCE_BLOCKED, EXIT_PROVIDER_UNAVAILABLE,
                                     exc.message)

    def proposal_validator(proposal):
        return validate_teacher_validation_plan_proposal(
            proposal, expected_run_id=c.state["run_id"],
            expected_evidence_profile_sha256=evidence_profile_sha256,
            admissible_components=decision_space["admissible_components"],
            validation_objectives=objectives)

    # Bounded semantic-correction retry: an admissible-but-objective-violating (or otherwise
    # contextually rejected) proposal is not an immediate campaign failure -- the planner gets the
    # exact deterministic rejection reason fed back, under the SAME frozen evidence_profile_sha256
    # / admissible_decision_space / validation_objectives, for up to two corrective attempts.
    # Every attempt is separately provenance-recorded (run_role/_write_provenance already gives
    # each dispatch its own attempt id); only FAILED after the bound is exhausted.
    max_attempts = 3  # initial attempt + at most 2 semantic-correction retries
    res = None
    for attempt_number in range(1, max_attempts + 1):
        attempt_task = task if attempt_number == 1 else build_task(
            prior_rejection={"attempt": attempt_number - 1, "validation_error": res.error})
        emitter.emit("role_invocation_started", role="orchestrator",
                    action="teacher_validation_plan_proposal", detail={"attempt": attempt_number})
        res = run_role(orchestrator_runtime, attempt_task, specs["orchestrator"],
                       ctx_factory(orchestrator_provider, orchestrator_model), mode="primary",
                       reasoning_validator=proposal_validator)
        emitter.emit("role_invocation_completed", role="orchestrator",
                    action="teacher_validation_plan_proposal",
                    detail={"accepted": res.accepted, "attempt": attempt_number})
        if res.accepted:
            break
    if not res.accepted:
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            f"teacher validation plan proposal rejected after {attempt_number} attempt(s) "
            f"(initial + up to {max_attempts - 1} semantic-correction retries): {res.error}")
    proposal = res.detail.instance

    draft = build_teacher_validation_plan_draft_from_proposal(
        proposal, decision_space=decision_space,
        evidence_profile=dataclasses.asdict(profile),
        proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"},
        validation_objectives=objectives)
    plan_dir = c.run_dir / "teacher_validation" / "drafts"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{c.state['run_id']}.teacher_validation_plan.draft.json"
    plan_path.write_text(json.dumps(draft.to_plan_json(), indent=2) + "\n")

    action_proposal = OrchestratorActionProposal(
        run_id=c.state["run_id"], stage="__pre_campaign__", requested_at="run-campaign",
        rationale="commit the autonomously-proposed Teacher validation plan before Stage 1",
        idempotency_key=(
            f"{c.state['run_id']}:teacher_validation_planning:{evidence_profile_sha256}"),
        action_type="commit_teacher_validation_plan",
        parameters={"run_dir": str(c.run_dir), "plan_path": str(plan_path)})
    outcome = dispatch_orchestrator_action(action_proposal, controller=c, mode="primary")
    if outcome.status != "EXECUTED":
        return CampaignRunResult(
            CAMPAIGN_FAILED, EXIT_VALIDATION_REJECTED,
            f"commit_teacher_validation_plan dispatch failed: {outcome.status}: "
            f"{outcome.reason}")
    emitter.emit("teacher_validation_plan_committed",
                detail={"selected_components": outcome.artifact.get("selected_components")})
    return None


def _run_campaign_loop(controller, *, runtime, agent_specs_dir="agent_specs", exchange_dir=None,
                 repo_root=".", auto_mock_judges=False, mock_response=None,
                 mock_judge_response=None, mock_analyst_response=None,
                 mock_orchestrator_response=None, mock_acquisition_response=None,
                 max_iterations=None,
                 recovery_action_registry=None, emitter=None) -> CampaignRunResult:
    """Drive ``controller``'s run forward through ``run_production_stage`` -- the SAME production
    dispatch+gate path ``run-stage`` uses -- for as many stages as current state allows, stopping
    at the first non-forward-progress outcome instead of guessing past it.

    This function owns no scientific judgment: it never authors a diagnosis, a recovery plan, or
    an approval decision. It only reads the declared stage graph and each stage's recorded gate off
    the Controller to pick the next eligible stage, and reads the StageRunResult that stage's real
    production dispatch returns to decide whether to keep going or stop.

    ``max_iterations`` defaults to one more than the declared stage count: a well-formed workflow
    needs at most one iteration per stage to run it, plus one final iteration to observe that no
    stage remains and report COMPLETED, so that bound is both safe and tight. A workflow that
    declares a non-terminal stage with no ``gate.criteria`` can never
    record that stage's gate as PASS (``complete_external_stage`` always leaves a completed gate-
    less stage's gate at "pending", and nothing without criteria ever calls ``record_gate``) -- this
    default bound turns that pre-existing Controller-level authoring mistake into a deterministic
    FAILED outcome instead of an unbounded loop.

    ``recovery_action_registry``, if given, overrides the executor registry used ONLY for an
    approved recovery's automatic corrective-action dispatch (see
    ``_dispatch_recovery_corrective_action``); forward-stage dispatch is unaffected and always uses
    the real production registry. Defaults to that same real registry when omitted.
    """
    from workflow.controller import RunController

    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    # Automatic pre-campaign Teacher-validation planning: happens once, before Stage 1, and is a
    # no-op for any run that never declared teacher_evidence_sources or already has a committed
    # plan (see the function's own docstring for the write-once / opt-in guarantees).
    planning_result = _commit_teacher_validation_plan_via_reasoning_roles(
        c, runtime=runtime, agent_specs_dir=agent_specs_dir, exchange_dir=exchange_dir,
        repo_root=repo_root, mock_orchestrator_response=mock_orchestrator_response,
        emitter=emitter)
    if planning_result is not None:
        return planning_result
    # Autonomous acquisition planning is deliberately LAZY: it is NOT run here, before the stage
    # loop. Its provider's build_context runs expensive geometry work (FPS / coverage / population
    # sizing over the whole candidate pool) that must not be spent before the campaign has actually
    # reached -- and cleared the costly-action authorization boundary of -- the acquisition stage.
    # It is instead triggered inside the loop, only when the next eligible stage is genuinely the
    # acquisition stage (see the ``_stage_route_action(...) == "acquire_structures"`` gate below).
    c = RunController(c.run_dir)
    if max_iterations is None:
        max_iterations = len(c.state["stages"]) + 1
    iterations = 0
    while True:
        if iterations >= max_iterations:
            return CampaignRunResult(
                CAMPAIGN_FAILED, EXIT_INTERNAL,
                f"run-campaign exceeded max_iterations={max_iterations} without reaching a "
                "terminal or pause state")
        iterations += 1
        pending = c.state.get("pending_recovery")
        if pending:
            status = pending.get("status")
            if status == "required":
                return _propose_recovery_via_reasoning_roles(
                    c, runtime=runtime, agent_specs_dir=agent_specs_dir,
                    exchange_dir=exchange_dir, repo_root=repo_root,
                    mock_analyst_response=mock_analyst_response,
                    mock_orchestrator_response=mock_orchestrator_response, emitter=emitter)
            if status == "proposed":
                recovery = next(r for r in c.state["recoveries"]
                                if r["id"] == pending["recovery_id"])
                return CampaignRunResult(
                    CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, EXIT_APPROVAL_REQUIRED,
                    f"WAITING_FOR_HUMAN_APPROVAL: recovery {pending['recovery_id']} for stage "
                    f"{recovery['failed_stage']!r} is awaiting human approval (see "
                    "`approve-recovery`)", stage=recovery["failed_stage"])
            if status == "approved":
                c.start_iteration()
                c = RunController(c.run_dir)
                continue
            return CampaignRunResult(
                CAMPAIGN_RECOVERY_REQUIRED, EXIT_RECOVERY_REQUIRED,
                f"RECOVERY_REQUIRED: unrecognized pending_recovery status {status!r}")
        iteration = c.state["iterations"][-1]
        trigger = iteration.get("trigger")
        if trigger and iteration.get("recovery_execution", {}).get("status") == "required":
            recovery = next(r for r in c.state["recoveries"] if r["id"] == trigger["recovery_id"])
            corrective_action = (recovery["plan"].get("recovery_context") or {}).get(
                "corrective_action")
            return_stage = trigger["return_stage"]
            if corrective_action and c.stage(return_stage)["status"] != "completed":
                # FE-043: a recovery whose return_stage re-runs the ``acquire_structures`` route
                # action fail-closes with PLAN_INPUT_REQUIRED unless a fresh AcquisitionPlan is bound
                # first. ``start_iteration`` has already SUPERSEDED the stale plan (it retires the
                # bound plan precisely so a coverage-deficit re-acquisition re-plans with gap-driven
                # sizing instead of regenerating byte-identical candidates), so re-enter the SAME
                # canonical lazy-planner path the forward loop uses below: the production Stage-3
                # planner reads this run's already-bound FE-042/FE-039 recovery evidence + cumulative
                # acquired population and binds a fresh, coverage-gap-targeted superseding plan, which
                # the corrective dispatch then executes ``acquire_structures`` against. This keeps the
                # Stage-3 planner the single canonical AcquisitionPlan source (no recovery-specific
                # planner). Idempotent: once a plan is bound the provider's ``applies`` is False, so a
                # resumed/re-looped recovery iteration is a no-op and never re-plans.
                if _stage_route_action(c, return_stage) == "acquire_structures":
                    from .default_acquisition_provider import (
                        maybe_install_default_acquisition_provider)
                    maybe_install_default_acquisition_provider(c)
                    from .acquisition_planner import plan_acquisition_via_reasoning_roles
                    plan_result = plan_acquisition_via_reasoning_roles(
                        c, runtime=runtime, agent_specs_dir=agent_specs_dir,
                        exchange_dir=exchange_dir, repo_root=repo_root,
                        mock_producer_response=mock_acquisition_response, emitter=emitter)
                    if plan_result is not None:
                        return plan_result
                    c = RunController(c.run_dir)
                    recovery = next(r for r in c.state["recoveries"]
                                    if r["id"] == trigger["recovery_id"])
                result = _dispatch_recovery_corrective_action(
                    c, trigger, recovery, corrective_action, registry=recovery_action_registry,
                    emitter=emitter)
                if result is not None:
                    return result
                c = RunController(c.run_dir)
                continue
        next_stage = _next_eligible_stage(c)
        if next_stage is None:
            return CampaignRunResult(CAMPAIGN_COMPLETED, EXIT_SUCCESS,
                                     "COMPLETED: every declared stage has passed its gate")
        # Lazy autonomous acquisition planning: only when the next eligible stage is genuinely the
        # acquisition stage do we install the default provider and run the (expensive) planner,
        # binding its deterministically-validated plan as a run input BEFORE this same iteration
        # dispatches the acquisition stage. This defers all FPS/coverage/sizing geometry work past
        # every earlier costly-action authorization boundary (e.g. teacher_baseline) so it is never
        # spent on a campaign that pauses for human authorization before reaching acquisition. It is
        # safe to re-enter every loop turn: the provider's applies() returns False (a cheap no-op)
        # once a plan is already bound, so a resumed/re-looped campaign never re-plans.
        if _stage_route_action(c, next_stage["name"]) == "acquire_structures":
            from .default_acquisition_provider import maybe_install_default_acquisition_provider
            maybe_install_default_acquisition_provider(c)
            from .acquisition_planner import plan_acquisition_via_reasoning_roles
            acquisition_result = plan_acquisition_via_reasoning_roles(
                c, runtime=runtime, agent_specs_dir=agent_specs_dir, exchange_dir=exchange_dir,
                repo_root=repo_root, mock_producer_response=mock_acquisition_response,
                emitter=emitter)
            if acquisition_result is not None:
                return acquisition_result
            c = RunController(c.run_dir)
        result = run_production_stage(
            c, next_stage["name"], runtime=runtime, agent_specs_dir=agent_specs_dir,
            exchange_dir=exchange_dir, repo_root=repo_root, auto_mock_judges=auto_mock_judges,
            mock_response=mock_response, mock_judge_response=mock_judge_response, emitter=emitter)
        if result.exit_code == EXIT_SUCCESS:
            c = RunController(c.run_dir)
            continue
        if result.reason == "APPROVAL_REQUIRED":
            return CampaignRunResult(CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, EXIT_APPROVAL_REQUIRED,
                                     result.message, stage=next_stage["name"],
                                     last_stage_result=result)
        if result.reason == "PROVIDER_UNAVAILABLE":
            return CampaignRunResult(CAMPAIGN_RESOURCE_BLOCKED, EXIT_PROVIDER_UNAVAILABLE,
                                     result.message, stage=next_stage["name"],
                                     last_stage_result=result)
        if result.reason == "EXTERNAL_ACTION_PENDING":
            return CampaignRunResult(CAMPAIGN_WAITING_FOR_EXTERNAL_ACTION,
                                     EXIT_EXTERNAL_ACTION_PENDING, result.message,
                                     stage=next_stage["name"], last_stage_result=result)
        if result.reason == "RECOVERY_EXECUTION_UNVERIFIED":
            c = RunController(c.run_dir)
            report, missing = _assemble_recovery_execution_report(c)
            if report is None:
                return CampaignRunResult(
                    CAMPAIGN_RECOVERY_EXECUTION_UNVERIFIED, EXIT_RECOVERY_EXECUTION_UNVERIFIED,
                    "RECOVERY_EXECUTION_UNVERIFIED: " + "; ".join(missing),
                    stage=next_stage["name"], last_stage_result=result)
            report_dir = c.run_dir / "recovery"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"recovery-{report['recovery_id']:03d}.execution.report.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            c.verify_recovery_execution(report_path)
            emitter.emit("recovery_verified", stage=next_stage["name"],
                         detail={"recovery_id": report["recovery_id"]})
            c = RunController(c.run_dir)
            continue
        if result.reason in ("GATE_REVISE", "GATE_FAIL"):
            # workflow.controller.record_gate has ALREADY durably recorded
            # pending_recovery={"status": "required", ...} for any non-PASS verdict (this ran
            # synchronously inside run_production_stage, before it returned this result) -- so
            # refreshing and looping back to the top-of-loop pending_recovery check above drives
            # this SAME run-campaign invocation straight through Analyst diagnosis and Orchestrator
            # recovery-plan proposal, stopping only at WAITING_FOR_HUMAN_APPROVAL (or a genuine
            # diagnosis/proposal failure) -- never returning RECOVERY_REQUIRED to the shell merely
            # because a gate recorded non-PASS, when the Controller can continue deterministically.
            c = RunController(c.run_dir)
            continue
        if result.reason.startswith("GATE_"):
            # A genuine internal invariant violation (e.g. GATE_RECORD_MISMATCH) rather than a
            # legitimate non-PASS verdict -- pending_recovery is NOT guaranteed set here, so this
            # stays a hard stop rather than looping.
            return CampaignRunResult(CAMPAIGN_RECOVERY_REQUIRED, EXIT_RECOVERY_REQUIRED,
                                     result.message, stage=next_stage["name"],
                                     last_stage_result=result)
        return CampaignRunResult(CAMPAIGN_FAILED, result.exit_code, result.message,
                                 stage=next_stage["name"], last_stage_result=result)


def run_campaign(controller, *, runtime, agent_specs_dir="agent_specs", exchange_dir=None,
                 repo_root=".", auto_mock_judges=False, mock_response=None,
                 mock_judge_response=None, mock_analyst_response=None,
                 mock_orchestrator_response=None, mock_acquisition_response=None,
                 max_iterations=None,
                 recovery_action_registry=None, emitter=None) -> CampaignRunResult:
    """Thin wrapper around ``_run_campaign_loop`` that owns the campaign-level start/resume and
    terminal outcome events -- the loop itself only ever returns once per invocation, so these are
    the single point in the whole call graph where "campaign started/resumed" and "campaign
    paused/completed/failed" can be emitted exactly once without threading them through every one
    of the loop's internal return statements."""
    c = controller
    emitter = emitter or CampaignEventEmitter(c.run_dir, quiet=True)
    # A prior CAMPAIGN EXECUTION lifecycle event (campaign_started/campaign_resumed) in the
    # durable log -- NOT mere file existence -- is what actually evidences a prior run-campaign
    # invocation: approve/approve-recovery write their own events (e.g. approval_granted) to the
    # same log via their own CampaignEventEmitter, so the file can already exist before
    # run-campaign has ever executed once. See events.campaign_previously_executed.
    emitter.emit("campaign_resumed" if campaign_previously_executed(c.run_dir)
                else "campaign_started")
    result = _run_campaign_loop(
        c, runtime=runtime, agent_specs_dir=agent_specs_dir, exchange_dir=exchange_dir,
        repo_root=repo_root, auto_mock_judges=auto_mock_judges, mock_response=mock_response,
        mock_judge_response=mock_judge_response, mock_analyst_response=mock_analyst_response,
        mock_orchestrator_response=mock_orchestrator_response,
        mock_acquisition_response=mock_acquisition_response, max_iterations=max_iterations,
        recovery_action_registry=recovery_action_registry, emitter=emitter)
    emitter.emit(f"campaign_{terminal_class(result.outcome).lower()}", stage=result.stage,
                detail={"outcome": result.outcome, "exit_code": result.exit_code})
    return result


def _cmd_run_campaign(args) -> int:
    from workflow.controller import RunController

    c = RunController(args.run_dir)
    emitter = CampaignEventEmitter(c.run_dir, run_id=c.state.get("run_id"),
                                   quiet=getattr(args, "quiet", False),
                                   json_events=getattr(args, "json_events", False))
    try:
        result = run_campaign(
            c, runtime=args.runtime, agent_specs_dir=args.agent_specs_dir,
            exchange_dir=args.exchange_dir, repo_root=args.repo_root,
            auto_mock_judges=args.auto_mock_judges, mock_response=args.mock_response,
            mock_judge_response=args.mock_judge_response,
            mock_analyst_response=args.mock_analyst_response,
            mock_orchestrator_response=args.mock_orchestrator_response, emitter=emitter)
    except Exception as exc:
        print(f"run-campaign failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_REJECTED
    stream = sys.stdout if result.exit_code == EXIT_SUCCESS else sys.stderr
    print(f"outcome: {result.outcome}\n{result.message}", file=stream)
    return result.exit_code


def _cmd_approve_recovery(args) -> int:
    from workflow.controller import RunController
    c = RunController(args.run_dir)
    recovery = c.approve_recovery(args.approved_by, note=args.note)
    print(f"recovery approved: id={recovery['id']} failed_stage={recovery['failed_stage']}")
    emitter = CampaignEventEmitter(c.run_dir, run_id=c.state.get("run_id"), quiet=True)
    emitter.emit("approval_granted", stage=recovery["failed_stage"],
                detail={"recovery_id": recovery["id"]})
    return EXIT_SUCCESS


def _cmd_plan_teacher_validation(args) -> int:
    from workflow.controller import RunController
    c = RunController(args.run_dir)
    emitter = CampaignEventEmitter(c.run_dir, run_id=c.state.get("run_id"), quiet=True)
    if c.state.get("teacher_evidence_sources") is None:
        print("this run did not declare teacher_evidence_sources; nothing to plan",
             file=sys.stderr)
        return EXIT_BLOCKED_POLICY
    if c.state.get("teacher_validation_plan") is not None:
        print("a Teacher validation plan is already committed for this run (write-once)")
        return EXIT_SUCCESS
    result = _commit_teacher_validation_plan_via_reasoning_roles(
        c, runtime=args.runtime, agent_specs_dir=args.agent_specs_dir,
        exchange_dir=args.exchange_dir, repo_root=args.repo_root,
        mock_orchestrator_response=args.mock_orchestrator_response, emitter=emitter)
    if result is not None:
        stream = sys.stdout if result.exit_code == EXIT_SUCCESS else sys.stderr
        print(f"outcome: {result.outcome}\n{result.message}", file=stream)
        return result.exit_code
    c = RunController(c.run_dir)
    plan = c.state["teacher_validation_plan"]
    print(f"teacher validation plan committed: selected_components={plan['selected_components']}")
    return EXIT_SUCCESS


def _cmd_authorize_downstream_teacher_reliance(args) -> int:
    from workflow.controller import RunController
    c = RunController(args.run_dir)
    plan = c.authorize_downstream_teacher_reliance(args.authorized_by, note=args.note)
    print(f"downstream Teacher reliance: selected_components={plan.get('selected_components')} "
         f"status={plan.get('downstream_reliance_approval') is not None}")
    emitter = CampaignEventEmitter(c.run_dir, run_id=c.state.get("run_id"), quiet=True)
    emitter.emit("approval_granted", detail={"scope": "downstream_teacher_reliance"})
    return EXIT_SUCCESS


def _cmd_augment_train(args) -> int:
    import os
    from workflow.controller import RunController
    from . import train_augmentation as ta

    c = RunController(args.run_dir)
    emitter = CampaignEventEmitter(c.run_dir, run_id=c.state.get("run_id"), quiet=True)

    # Precondition: Stage-6 dataset_split must have PASSED -- augmentation operates on the FROZEN
    # TRAIN/validation/test parent membership. Fail closed rather than augmenting a pre-split pool.
    try:
        split_status = c.stage("dataset_split")["status"]
    except Exception:
        split_status = None
    if split_status != "completed":
        print("augment-train requires a PASSED dataset_split stage (frozen TRAIN parents); "
              f"dataset_split status={split_status!r}", file=sys.stderr)
        return EXIT_BLOCKED_POLICY

    train_dataset = args.train_dataset or str(
        c.run_dir / "artifacts" / "dataset" / "train.extxyz")
    if not Path(train_dataset).is_file():
        print(f"TRAIN parents dataset not found: {train_dataset}", file=sys.stderr)
        return EXIT_VALIDATION_REJECTED

    # 1. Build the TRAIN-parent pool manifest, EXCLUDING protected augmentation_parents.
    reference_id, protected_globals = ta.resolve_protected_source_indices(c)
    train_pool = ta.build_train_pool_manifest(
        train_dataset, ta.augmentation_dir(c.run_dir) / "train_pool",
        protected_source_indices=protected_globals)
    emitter.emit("augmentation_pool_built",
                 detail={"n_frames": train_pool["n_frames"],
                         "n_protected_excluded": train_pool["n_protected_excluded"]})

    # 2. Autonomously plan the augmentation recipe (reused acquisition producer, rebound to TRAIN).
    produced = ta.plan_train_augmentation(
        c, runtime=args.runtime, agent_specs_dir=args.agent_specs_dir,
        exchange_dir=args.exchange_dir, repo_root=args.repo_root,
        train_manifest_path=train_pool["manifest_path"],
        mock_producer_response=args.mock_acquisition_response, emitter=emitter)
    if produced.failure is not None:
        r = produced.failure
        print(f"outcome: {r.outcome}\n{r.message}", file=sys.stderr)
        return r.exit_code

    # 3. Freeze the autonomously-realized AugmentationPlan with full provenance.
    frozen_path = ta.freeze_augmentation_plan(c, produced, train_pool=train_pool, emitter=emitter)
    strategy_kind = produced.ctx.strategy.kind.value
    warranted = strategy_kind != "EXISTING_POOL_SELECTION"
    print(f"AugmentationPlan frozen: {frozen_path}")
    print(f"strategy_kind={strategy_kind} augmentation_warranted={warranted} "
          f"train_parents={train_pool['n_frames']} "
          f"protected_excluded={train_pool['n_protected_excluded']}")

    if not args.execute:
        print("plan phase complete (no --execute); the costly_teacher_labeling generation/"
              "labeling/merge was NOT run")
        return EXIT_SUCCESS

    # 4. Execute -- the costly_teacher_labeling boundary. A warranted (Teacher-driving) plan needs
    #    the explicit smoke-confirm; an unwarranted EXISTING_POOL_SELECTION plan makes NO Teacher
    #    call and simply routes the labeled TRAIN parents through as final_train.
    if warranted and os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
        print("APPROVAL_REQUIRED: a warranted augmentation plan drives the Teacher (augment_atoms "
              "PES + canonical labeling); set PYDANTIC_AI_SMOKE_CONFIRM=yes to authorize the "
              "costly_teacher_labeling execution", file=sys.stderr)
        return EXIT_APPROVAL_REQUIRED

    teacher_config = args.teacher_config
    if teacher_config is None:
        from .default_acquisition_provider import _resolve_bound_teacher_calculator_config
        teacher_config, _ = _resolve_bound_teacher_calculator_config(c)
    reference_yaml = args.reference_yaml or _acquisition_protection_reference_yaml(c)
    base_label_manifest = args.base_label_manifest
    if warranted and not base_label_manifest:
        # No per-split base label manifest is emitted by Stage-5 (full pool) or Stage-6 (split);
        # derive one HONESTLY by projecting the authoritative Stage-5 Teacher binding onto the
        # frozen TRAIN split (train.extxyz is a byte-preserved subset, proven by split_manifest).
        base_label_manifest = str(ta.derive_train_base_label_manifest(c, train_dataset=train_dataset))
        emitter.emit("augmentation_base_label_manifest_derived",
                     detail={"path": base_label_manifest})
    if warranted and (not teacher_config or not base_label_manifest or not reference_yaml):
        print("APPROVAL_REQUIRED inputs missing for execution: a warranted plan needs "
              "--teacher-config, --base-label-manifest, and --reference-yaml (base_label_manifest "
              "is the Stage-5 teacher_labeling manifest for the TRAIN parents)", file=sys.stderr)
        return EXIT_VALIDATION_REJECTED

    try:
        fm = ta.execute_train_augmentation(
            c, base_dataset=train_dataset, base_label_manifest=base_label_manifest,
            teacher_config=teacher_config, reference_yaml=reference_yaml, emitter=emitter)
    except Exception as exc:
        print(f"augment-train execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_REJECTED

    # Register the finalized final_train as a canonical, hash-bound, SUPERSEDING Stage-7 training
    # input through the existing bound-input mechanism (never a parallel registry). The Stage-6
    # dataset_split TRAIN artifact is preserved untouched -- final_train is a superseding TRAINING
    # POPULATION, not a replacement for the historical split provenance. This is the authority the
    # deterministic training-evidence builder resolves the authoritative training population from.
    _register_final_train_as_training_input(c, fm, emitter=emitter)

    print(f"final_train: {fm['final_train_path']}\nfinal_train_sha256={fm['final_train_sha256']} "
          f"n_augmented_children={fm['n_augmented_children']}")
    return EXIT_SUCCESS


def _register_final_train_as_training_input(controller, fm, *, emitter=None) -> None:
    """Bind the finalized ``final_train`` as the canonical superseding Stage-7 training input.

    Idempotent: if a bound input/artifact already carries final_train's resolved path AND sha256,
    nothing is re-bound. Uses ``copy=False`` (hash-bind in place) because final_train already lives
    under the run's ``artifacts/dataset/`` -- no scientific bytes are copied or mutated."""
    from workflow.controller import now as _now
    final_path = fm.get("final_train_path")
    final_sha = fm.get("final_train_sha256")
    if not final_path or not final_sha:
        return
    target = str(Path(final_path).resolve())
    for rec in controller.state.get("inputs", []) or []:
        for key in ("snapshot", "source"):
            p = rec.get(key)
            if p and rec.get("sha256") == final_sha and str(Path(p).resolve()) == target:
                return
    for rec in controller.state.get("artifacts", []) or []:
        p = rec.get("path")
        if p and rec.get("sha256") == final_sha and str(Path(p).resolve()) == target:
            return
    controller.bind_new_input(final_path, copy=False)
    controller.state["events"].append({
        "at": _now(), "type": "post_split_final_train_bound",
        "final_train_path": target, "final_train_sha256": final_sha,
        "base_train_dataset_sha256": fm.get("base_train_dataset_sha256"),
        "n_augmented_children": fm.get("n_augmented_children"),
        "merge_output_sha256": fm.get("merge_output_sha256"),
        "detail": ("finalized post-split augmented TRAIN population bound as the canonical "
                   "superseding Stage-7 training input; Stage-6 dataset_split TRAIN artifact "
                   "preserved as historical provenance"),
    })
    controller.save()
    if emitter is not None:
        emitter.emit("post_split_final_train_bound",
                     detail={"final_train_sha256": final_sha,
                             "n_augmented_children": fm.get("n_augmented_children")})


def main(argv=None) -> int:
    import os
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        return _cmd_preflight(args)
    if args.command == "approve":
        return _cmd_approve(args)
    if args.command == "run-stage":
        return _cmd_run_stage(args)
    if args.command == "run-campaign":
        return _cmd_run_campaign(args)
    if args.command == "bind-closure":
        return _cmd_bind_closure(args)
    if args.command == "bind-scientific-policies":
        return _cmd_bind_scientific_policies(args)
    if args.command == "approve-recovery":
        return _cmd_approve_recovery(args)
    if args.command == "plan-teacher-validation":
        return _cmd_plan_teacher_validation(args)
    if args.command == "authorize-downstream-teacher-reliance":
        return _cmd_authorize_downstream_teacher_reliance(args)
    if args.command == "augment-train":
        return _cmd_augment_train(args)
    if args.command != "run-task":  # pragma: no cover
        return EXIT_INTERNAL
    out = sys.stdout

    # Deferred imports so `--help` and non-pydantic environments don't require the extra.
    try:
        from orchestration.specs import load_agent_specs
        from .models import RuntimeContext
        from .driver import run_task
    except Exception as exc:  # pragma: no cover
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL

    try:
        specs = load_agent_specs(args.agent_specs_dir)
    except Exception as exc:
        print(f"could not load agent specs from {args.agent_specs_dir}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    if args.agent not in specs:
        print(f"unknown agent '{args.agent}'; known: {sorted(specs)}", file=sys.stderr)
        return EXIT_BLOCKED_POLICY
    spec = specs[args.agent]

    task_path = Path(args.task)
    try:
        task = json.loads(task_path.read_text())
    except Exception as exc:
        print(f"could not read task {task_path}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL

    model_id = args.model or os.environ.get("PYDANTIC_AI_MODEL")
    provider = args.provider or (model_id.split(":", 1)[0] if model_id and ":" in model_id else "mock")

    # Build the runtime.
    if args.runtime == "mock":
        from .mock_runtime import MockAgentRuntime
        if not args.mock_response:
            print("--runtime mock requires --mock-response", file=sys.stderr)
            return EXIT_INTERNAL
        raw = Path(args.mock_response).read_text()
        runtime = MockAgentRuntime(lambda t, s, ts: (raw, (0, 0)))
        provider = provider if provider != "mock" else "mock"
    else:  # pydantic-ai
        # Route to the selected backend. A LOCAL (OpenAI-compatible/Ollama) backend needs NO
        # Anthropic credential; the hosted Anthropic path is kept but optional.
        from . import provider as _prov
        if args.provider:
            os.environ.setdefault("PYDANTIC_AI_PROVIDER", args.provider)
        if args.base_url:
            os.environ.setdefault("PYDANTIC_AI_BASE_URL", args.base_url)
        kind = _prov.select_provider_kind()

        def _confirm_missing(provider_name, model_name, note_extra=""):
            _print_kv(out, runtime="pydantic-ai", provider=provider_name, model=model_name,
                      confirmation="MISSING",
                      note=("set PYDANTIC_AI_SMOKE_CONFIRM=yes to authorize ONE real provider "
                            "call; no provider was called" + note_extra))
            return EXIT_APPROVAL_REQUIRED

        if kind in _prov.LOCAL_KINDS:
            pf = _prov.preflight_local(probe=args.probe_server)
            if pf.status != _prov.LOCAL_READY:
                # A not-running server is an OPERATIONAL status, not a runtime/scientific failure.
                _print_kv(out, runtime="pydantic-ai", provider=kind, preflight=pf.status,
                          reason=pf.reason, base_url=pf.base_url, model=pf.model_id,
                          anthropic_key_required=pf.anthropic_key_required)
                return EXIT_PROVIDER_UNAVAILABLE
            if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                return _confirm_missing(kind, pf.model_id, " (local inference)")
            from .pydantic_ai_runtime import PydanticAIRuntime
            model_id = pf.model_id
            provider = kind
            runtime = PydanticAIRuntime(
                model=_prov.build_local_model(kind, pf.model_id, pf.base_url),
                usage_source="provider")
        elif kind in _prov.HOSTED_KINDS:
            pf = _prov.preflight_credentials(provider=kind)
            if pf.status != "READY":
                _print_kv(out, runtime="pydantic-ai", preflight=pf.status, reason=pf.reason,
                          provider=pf.provider, model=pf.model_id)
                return EXIT_PROVIDER_UNAVAILABLE
            # Explicit human confirmation gate for a REAL (billable) provider call. Preflight being
            # READY (credential present) is not sufficient; a live call also requires an intentional
            # PYDANTIC_AI_SMOKE_CONFIRM=yes, so merely having a key in the shell never bills.
            if os.environ.get("PYDANTIC_AI_SMOKE_CONFIRM") != "yes":
                return _confirm_missing(pf.provider, pf.model_id)
            from .pydantic_ai_runtime import PydanticAIRuntime
            model_id = pf.model_id
            provider = pf.provider
            runtime = PydanticAIRuntime(model=_prov.build_provider_model(model_id),
                                        usage_source="provider")
        else:
            _print_kv(out, runtime="pydantic-ai", preflight="NOT_CONFIGURED",
                      reason=("set PYDANTIC_AI_PROVIDER to local-openai|ollama|anthropic|openai "
                              "(local needs PYDANTIC_AI_BASE_URL; hosted needs its API key)"))
            return EXIT_PROVIDER_UNAVAILABLE

    ctx = RuntimeContext(
        exchange_dir=args.exchange_dir, repo_root=args.repo_root,
        provider=provider, model_id=model_id or "mock",
        read_allow_prefixes=args.read_allow, correlation_id=args.correlation_id)

    cli_mode = {"primary": "primary", "shadow": "shadow", "dry-run": "dry_run",
                "validate-only": "validate_only"}[args.mode]

    # The production router selects the acceptance strategy from the role/typed output; producer
    # dispatch needs a controller + executor registry. No manual per-role function selection.
    from .production_router import run_role, acceptance_strategy
    from .executors import build_executor_registry
    controller = None
    strategy = acceptance_strategy(spec)
    if strategy == "producer_dispatch":
        if not args.run_dir:
            print("producer roles require --run-dir (controller manifest)", file=sys.stderr)
            return EXIT_INTERNAL
        from workflow.controller import RunController
        controller = RunController(args.run_dir)
    registry = build_executor_registry()

    try:
        res = run_role(runtime, task, spec, ctx, controller=controller, registry=registry,
                       mode=cli_mode)
    except FileExistsError as exc:
        # An identical re-dispatch is now a silent no-op (see FileExchangeRuntime.dispatch); this
        # only fires for a genuine task-identity conflict (TaskPacketConflictError) or another
        # component's own pre-existing FileExistsError guard.
        print(f"task identity conflict: {exc}", file=sys.stderr)
        return EXIT_DUPLICATE
    except Exception as exc:  # pragma: no cover
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL

    # canonical_validation is explicit so a shadow run (which never "accepts") still reports
    # whether validate_agent_response passed: for judge_gate/agent_result an empty error + a
    # non-None validated payload means the contract validator passed.
    canonical_validation = "n/a"
    if res.strategy in ("judge_gate", "agent_result"):
        canonical_validation = "passed" if (res.detail is not None and not res.error) else "failed"
    _print_kv(
        out,
        task_path=str(task_path), task_id=task.get("task_id", ""), role=args.agent,
        runtime=args.runtime, provider=provider, model=model_id or "mock", mode=args.mode,
        strategy=res.strategy, canonical_validation=canonical_validation,
        accepted=res.accepted, controller_mutation=res.controller_mutated,
        provenance_path=str(res.provenance_path), error=(res.error or ""))

    # Map the routed result to an exit code.
    outcome_status = getattr(res.detail, "status", None)  # ActionOutcome for producer_dispatch
    if outcome_status is not None:
        return {
            "EXECUTED": EXIT_SUCCESS, "DRY_RUN": EXIT_SUCCESS,
            "DENIED": EXIT_BLOCKED_POLICY, "BLOCKED_CAPABILITY": EXIT_BLOCKED_POLICY,
            "APPROVAL_REQUIRED": EXIT_APPROVAL_REQUIRED, "DUPLICATE": EXIT_DUPLICATE,
            "INVALID": EXIT_VALIDATION_REJECTED, "EXECUTOR_ERROR": EXIT_VALIDATION_REJECTED,
            "PENDING": EXIT_EXTERNAL_ACTION_PENDING,
        }.get(outcome_status, EXIT_INTERNAL)
    if res.error and res.strategy in ("judge_gate", "agent_result"):
        return EXIT_VALIDATION_REJECTED
    if res.error:
        return EXIT_VALIDATION_REJECTED
    return EXIT_SUCCESS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
