"""Run commands stage-by-stage and require a recorded PASS before advancing."""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from orchestration.exchange import validate_judge_vote
from workflow import recovery_taxonomy
from workflow.actor_identity import normalize_actor_identity, same_actor
from workflow.integrity import artifact_digest, sha256_file, verify_artifact
from workflow.contracts import (
    build_validation_contract_components, parse_teacher_validation_objectives,
    validate_md_manifest, validate_validation_manifest,
)
from workflow.review_lenses import normalize_review_lenses

# Path-valued vs. pass-through-valued keys a workflow's OPTIONAL `teacher_evidence_sources` block
# may declare -- see RunController.initialize's handling and
# validation.teacher_evidence_profile.inspect_teacher_evidence, whose keyword arguments these
# mirror exactly (this dict is intentionally NOT a copy of that function's signature via
# introspection: an unrelated future kwarg added there should never silently become acceptable
# here without this list being reviewed too).
TEACHER_EVIDENCE_SOURCE_PATH_KEYS = (
    "teacher_model_path", "operational_evaluation_population_path", "original_training_db_path",
    "independent_external_reference_path", "deployment_domain_population_path",
)
TEACHER_EVIDENCE_SOURCE_PASSTHROUGH_KEYS = (
    "target_split", "deployment_domain_matches_original_test_distribution",
    "original_split_confidence", "label_energy_key", "label_forces_key",
)


# RECOVERY_CATEGORIES is DERIVED from the shared workflow.recovery_taxonomy registry (see that
# module) rather than declared as its own independent fixed set: it is exactly the registry's
# full set of registered failure_code values at import time. That registry also reconciles
# runtimes.pydantic_ai.root_cause.RootCauseClassification.failure_category, so diagnosis and
# recovery share one vocabulary rather than two independently-maintained category sets.
# propose_recovery() below resolves a submitted failure_category directly against the live
# registry (not this frozen snapshot) so a code registered by a campaign at runtime is accepted
# immediately; this constant remains for introspection/back-compat only. Every one of the
# original 8 controller-only values is still registered (see recovery_taxonomy's legacy block),
# so historical recovery plans validate identically.
RECOVERY_CATEGORIES = frozenset(recovery_taxonomy.registered_codes())

# DEFAULT_RECOVERY_CAPABILITY_ROSTER maps a registered recovery CAPABILITY to the concrete role
# that fills it when a run does not declare its own recovery_capability_roster (see
# RunController.initialize). Its value-set is exactly the original fixed RECOVERY_AGENTS tuple,
# so a historical plan that supplies only responsible_agent (never responsible_capability)
# validates identically to before: capability-based routing is additive, not a replacement of
# the direct-role path. A run declaring its own roster is not limited to these five roles/names;
# they are a default binding, not a framework constant.
DEFAULT_RECOVERY_CAPABILITY_ROSTER = {
    "data_repair": "data-curator",
    "model_retrain": "ml-trainer",
    "simulation_rerun": "simulation",
    "root_cause_analysis": "analyst",
    "orchestration": "orchestrator",
}
RECOVERY_AGENTS = frozenset(DEFAULT_RECOVERY_CAPABILITY_ROSTER.values())
ADJUDICATION_DECISIONS = {"ACCEPT_DECLARED_LIMITATION", "REQUIRE_SCIENTIFIC_RECOVERY"}
ADJUDICATION_SCOPE_EFFECT = "restrict_scope_to_declared_limitations"

VALIDATION_CONTRACT_COMPONENTS = (
    "teacher_applicability_domain", "validation_scope", "dataset_split_policy",
)


def _build_validation_contract_record(components, source_files=None):
    """Canonicalize validation-contract components into the write-once record shape.

    The single construction path for a ``validation_contract`` record: used both by
    ``RunController.establish_validation_contract`` (post-init, explicit call) and by
    ``RunController.initialize`` (automatic, from run-bound ``validation_contract_sources``
    snapshots) so there is exactly one authoritative way a contract record is ever built.
    """
    if not isinstance(components, dict) or set(components) != set(VALIDATION_CONTRACT_COMPONENTS):
        raise ValueError(
            "validation contract requires exactly the components: "
            + ", ".join(VALIDATION_CONTRACT_COMPONENTS)
        )
    canonical = {}
    for key, value in components.items():
        payload = json.dumps(value, indent=2, sort_keys=True).encode()
        canonical[key] = {"value": value, "sha256": hashlib.sha256(payload).hexdigest()}
    contract_sha256 = hashlib.sha256(
        json.dumps({key: entry["sha256"] for key, entry in canonical.items()},
                   sort_keys=True).encode()
    ).hexdigest()
    return {"established_at": now(), "components": canonical,
           "source_files": dict(source_files or {}), "contract_sha256": contract_sha256,
           "student_stage_ever_completed": False}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def format_context(value, context):
    """Format controller placeholders recursively in contract options."""
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [format_context(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_context(item, context) for key, item in value.items()}
    return value


def git_revision(project_dir):
    """Return the Git commit and a content hash for any tracked/untracked changes."""
    project_dir = Path(project_dir).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(project_dir), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", str(project_dir), "diff", "--binary", "HEAD"], check=True,
            capture_output=True,
        ).stdout
        untracked_raw = subprocess.run(
            ["git", "-C", str(project_dir), "ls-files", "--others", "--exclude-standard", "-z"],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "git_commit": None, "dirty": None, "diff_sha256": None}
    untracked = []
    for raw in untracked_raw.split(b"\0"):
        if not raw:
            continue
        path = project_dir / os.fsdecode(raw)
        if path.exists():
            untracked.append({"path": os.fsdecode(raw), **artifact_digest(path)})
    dirty = bool(status.strip())
    payload = diff + status + json.dumps(untracked, sort_keys=True, default=str).encode()
    return {"available": True, "git_commit": commit, "dirty": dirty,
            "diff_sha256": hashlib.sha256(payload).hexdigest() if dirty else None}


# Durable-state schema version. v7 adds ADDITIVE operational metadata only (runtime attempt
# references, action idempotency, stale-running runner metadata). It does NOT change any
# stage/gate/retry/recovery scientific semantics. v6 manifests remain readable as-is; a v6 run
# is never version-bumped in place — migration happens only on a copy (workflow.manifest_migration)
# or when a fresh run is initialized. See MIGRATION.md / the schema-bump report.
#
# v8 adds the write-once validation-target contract (validation_contract, plus the per-stage
# produces_student_results flag). It IS a scientific-semantics change: a stage flagged
# produces_student_results cannot run until a contract is established. It is still additive for
# every existing run/workflow, because the new stage flag defaults to False (no existing
# workflow.yaml sets it) and validation_contract defaults to None — so migrated v7 manifests and
# their workflows behave exactly as before unless a workflow config opts in explicitly.
#
# v9 completes (does not replace) the existing recovery state machine with genuinely-unified
# diagnosis/recovery vocabulary, capability-based responsible-agent routing, an explicit
# human-approval-created RecoveryAuthorizationEnvelope for costly child actions, optional
# cumulative loop-safety policy, and fail-closed protected-reference-role isolation. Every new
# top-level state key (recovery_capability_roster, recovery_policy, protected_reference_roles)
# defaults to None/[] exactly like v8's validation_contract_sources pattern: a workflow that
# declares none of them gets byte-for-byte v8 behavior. propose_recovery() gained OPTIONAL plan
# fields (responsible_capability, failure_domain, diagnosis_binding,
# required_input_artifact_roles, expected_output_artifact_roles,
# protected_reference_reuse_authorization, escalation_acknowledged/escalation_rationale) that a
# historical plan supplying only the original required fields never has to set; every existing
# recovery-plan fixture (responsible_agent + failure_category from the original 8-value set,
# nothing else new) continues to validate identically because every new check is gated on the
# corresponding optional field being present, or on an explicitly-opted-in recovery_policy /
# protected_reference_roles / recovery_capability_roster. A new recovery_signature is now always
# computed and stored (an additive record field, not a new required input) so repeated proposals
# under materially unchanged evidence/diagnosis/return-stage/corrective-action can be detected
# for the (opt-in) stagnation-escalation policy. No stage/gate/adjudication semantics changed;
# schema_version bumped 8->9 because the recovery-record shape gained new fields.
#
# v10 replaces the old SiO2/Allegro-specific validation branch with a generic, evidence-driven,
# additive component-based Teacher-validation decision model (validation.teacher_evidence_profile)
# plus an autonomous PydanticAI-driven Teacher-validation planning pipeline
# (runtimes.pydantic_ai.teacher_validation_plan). Three new additive top-level state keys, every
# one defaulting to None/{} exactly like v8/v9's pattern -- a workflow that declares none of them
# gets byte-for-byte v9 behavior: teacher_evidence_sources (this run's OPTIONAL frozen evidence
# input paths, validated/resolved at initialize() time; None unless a workflow declares
# teacher_evidence_sources), teacher_validation_plan (the write-once committed
# TeacherValidationPlan record; None until commit_teacher_validation_plan succeeds -- see that
# method's docstring for its independent-re-derivation, fail-closed-on-unsupported-claim
# semantics), stage_applicability ({} unless mark_stage_not_applicable is ever called for this
# run). record_gate's verdict whitelist gained "NOT_APPLICABLE" as a fourth resolved-but-distinct
# gate state (a stage whose evidence makes it genuinely inapplicable, e.g. no admissible
# Teacher-validation component under this run's evidence -- never a substitute for PASS or a way
# to silently skip a stage the workflow otherwise requires); every existing PASS-only consumer
# (_previous_passed, run summary "all stages passed" check, stage_progress_fields) was audited and
# updated to treat PASS and NOT_APPLICABLE as equally "resolved, may proceed downstream". No
# existing stage/gate/recovery semantics changed for a workflow that never declares
# teacher_evidence_sources or calls mark_stage_not_applicable; schema_version bumped 9->10 because
# the manifest shape gained new top-level keys.
SCHEMA_VERSION = 10


class RunController:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.state_path = self.run_dir / "manifest.json"
        if not self.state_path.exists():
            raise FileNotFoundError(f"run is not initialized: {self.run_dir}")
        # Read the manifest EXACTLY as written (no auto-migration): a v6 manifest stays v6 on
        # disk. v7 accessor helpers default the additive fields when absent, so v7 code operates
        # on a v6 manifest in memory without modifying its on-disk schema_version.
        self.state = json.loads(self.state_path.read_text())

    @classmethod
    def initialize(cls, workflow_config, run_dir):
        run_dir = Path(run_dir).resolve()
        workflow_config = Path(workflow_config).resolve()
        cfg = yaml.safe_load(workflow_config.read_text())
        if (not isinstance(cfg, dict) or not isinstance(cfg.get("run_id"), str) or
                not cfg["run_id"].strip()):
            raise ValueError("workflow config requires a non-empty run_id")
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        project_dir = Path.cwd().resolve()
        prepared_inputs = []
        for raw in cfg.get("inputs", []):
            spec = raw if isinstance(raw, dict) else {"path": raw, "copy": True}
            if not isinstance(spec.get("path"), (str, os.PathLike)):
                raise ValueError("every workflow input requires a path")
            source = Path(str(spec["path"]).format(project_dir=str(project_dir)))
            if not source.is_absolute():
                source = (workflow_config.parent / source).resolve()
            if not source.exists():
                raise FileNotFoundError(f"declared workflow input is missing: {source}")
            source_integrity = artifact_digest(source)
            if spec.get("copy", True):
                if not source.is_file():
                    raise ValueError("directory inputs must use copy: false and are hash-bound in place")
            prepared_inputs.append((source, bool(spec.get("copy", True)), source_integrity))
        contract_sources_spec = cfg.get("validation_contract_sources")
        prepared_contract_sources = None
        if contract_sources_spec is not None:
            if not isinstance(contract_sources_spec, dict):
                raise ValueError("validation_contract_sources must be a mapping")
            required_source_keys = ("distillation_scope", "validation_profile", "dataset_policy")
            missing_source_keys = [key for key in required_source_keys
                                   if not isinstance(contract_sources_spec.get(key), (str, os.PathLike))]
            if missing_source_keys:
                raise ValueError(
                    "validation_contract_sources requires non-empty paths for: "
                    + ", ".join(required_source_keys)
                )
            prepared_contract_sources = {}
            for key in required_source_keys:
                source = Path(str(contract_sources_spec[key]).format(project_dir=str(project_dir)))
                if not source.is_absolute():
                    source = (workflow_config.parent / source).resolve()
                if not source.exists():
                    raise FileNotFoundError(f"validation_contract_sources.{key} is missing: {source}")
                prepared_contract_sources[key] = source
        recovery_capability_roster_spec = cfg.get("recovery_capability_roster")
        if recovery_capability_roster_spec is not None:
            if (not isinstance(recovery_capability_roster_spec, dict) or not recovery_capability_roster_spec or
                    any(not isinstance(k, str) or not k.strip() or
                        not isinstance(v, str) or not v.strip()
                        for k, v in recovery_capability_roster_spec.items())):
                raise ValueError(
                    "recovery_capability_roster must be a non-empty mapping of "
                    "non-empty capability -> non-empty role strings"
                )
        recovery_policy_spec = cfg.get("recovery_policy")
        if recovery_policy_spec is not None and not isinstance(recovery_policy_spec, dict):
            raise ValueError("recovery_policy must be a mapping")
        protected_reference_roles_spec = cfg.get("protected_reference_roles")
        if protected_reference_roles_spec is not None:
            if (not isinstance(protected_reference_roles_spec, list) or
                    any(not isinstance(role, str) or not role.strip()
                        for role in protected_reference_roles_spec)):
                raise ValueError("protected_reference_roles must be a list of non-empty strings")
        teacher_evidence_sources_spec = cfg.get("teacher_evidence_sources")
        prepared_teacher_evidence_sources = None
        if teacher_evidence_sources_spec is not None:
            if not isinstance(teacher_evidence_sources_spec, dict):
                raise ValueError("teacher_evidence_sources must be a mapping")
            known_keys = (set(TEACHER_EVIDENCE_SOURCE_PATH_KEYS) |
                         set(TEACHER_EVIDENCE_SOURCE_PASSTHROUGH_KEYS) |
                         {"split_source_manifest_paths"})
            unknown_keys = set(teacher_evidence_sources_spec) - known_keys
            if unknown_keys:
                raise ValueError(
                    "teacher_evidence_sources has unknown key(s): " + ", ".join(sorted(unknown_keys))
                )

            def _resolve_declared_path(raw_value, *, label):
                candidate = Path(str(raw_value).format(project_dir=str(project_dir)))
                if not candidate.is_absolute():
                    candidate = (workflow_config.parent / candidate).resolve()
                if not candidate.exists():
                    raise FileNotFoundError(f"teacher_evidence_sources.{label} is missing: {candidate}")
                return str(candidate)

            prepared_teacher_evidence_sources = {}
            for key in TEACHER_EVIDENCE_SOURCE_PATH_KEYS:
                raw_value = teacher_evidence_sources_spec.get(key)
                prepared_teacher_evidence_sources[key] = (
                    _resolve_declared_path(raw_value, label=key) if raw_value is not None else None
                )
            manifests_spec = teacher_evidence_sources_spec.get("split_source_manifest_paths", [])
            if not isinstance(manifests_spec, list):
                raise ValueError(
                    "teacher_evidence_sources.split_source_manifest_paths must be a list"
                )
            prepared_teacher_evidence_sources["split_source_manifest_paths"] = [
                _resolve_declared_path(value, label="split_source_manifest_paths")
                for value in manifests_spec
            ]
            for key in TEACHER_EVIDENCE_SOURCE_PASSTHROUGH_KEYS:
                if key in teacher_evidence_sources_spec:
                    prepared_teacher_evidence_sources[key] = teacher_evidence_sources_spec[key]
        stages = []
        raw_stages = cfg.get("stages", [])
        if not isinstance(raw_stages, list) or any(not isinstance(item, dict)
                                                   for item in raw_stages):
            raise ValueError("workflow stages must be a list of mappings")
        names = [item.get("name") for item in raw_stages]
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("workflow stages must have unique non-empty names")
        for item in raw_stages:
            command = item.get("command")
            if command is not None and (not isinstance(command, list) or not command):
                raise ValueError(f"stage {item['name']!r} command must be a non-empty list or null")
            outputs = item.get("outputs", [])
            if (not isinstance(outputs, list) or
                    any(not isinstance(value, str) or not value.strip() for value in outputs) or
                    len(outputs) != len(set(outputs))):
                raise ValueError(f"stage {item['name']!r} outputs must be unique non-empty paths")
            for value in outputs:
                output = Path(value)
                if output.is_absolute() or ".." in output.parts:
                    raise ValueError(f"stage {item['name']!r} output must stay inside the run: {value}")
            env = item.get("env")
            if env is not None and (not isinstance(env, str) or not env.strip()):
                raise ValueError(f"stage {item['name']!r} env must be a non-empty string")
            produces_student_results = item.get("produces_student_results", False)
            if not isinstance(produces_student_results, bool):
                raise ValueError(
                    f"stage {item['name']!r} produces_student_results must be a bool"
                )
            gate_config = item.get("gate")
            if gate_config is not None and not isinstance(gate_config, dict):
                raise ValueError(f"stage {item['name']!r} gate must be a mapping")
            gate_criteria = (gate_config or {}).get("criteria")
            if (gate_criteria is not None and
                    (not isinstance(gate_criteria, list) or not gate_criteria or
                     any(not isinstance(value, str) or not value.strip()
                         for value in gate_criteria) or
                     len(gate_criteria) != len(set(gate_criteria)))):
                raise ValueError(
                    f"stage {item['name']!r} gate criteria must be unique non-empty strings"
                )
            gate_review_lenses = None
            if gate_criteria is not None:
                gate_review_lenses = normalize_review_lenses(
                    (gate_config or {}).get("review_lenses")
                )
            contract = item.get("contract")
            if contract is not None and not isinstance(contract, dict):
                raise ValueError(f"stage {item['name']!r} contract must be a mapping")
            if contract is not None:
                contract_kind = contract.get("kind")
                required_fields = {
                    "md_manifest": ("manifest", "committee_manifest"),
                    "validation_manifest": ("manifest", "validator"),
                }
                if contract_kind not in required_fields:
                    raise ValueError(
                        f"stage {item['name']!r} has unknown contract kind: {contract_kind!r}"
                    )
                missing = [field for field in required_fields[contract_kind]
                           if not isinstance(contract.get(field), str) or
                           not contract[field].strip()]
                if missing:
                    raise ValueError(
                        f"stage {item['name']!r} contract is missing: " + ", ".join(missing)
                    )
                if (contract_kind == "validation_manifest" and
                        "." not in contract["validator"]):
                    raise ValueError("validation contract validator must be a dotted callable path")
                if "options" in contract and not isinstance(contract["options"], dict):
                    raise ValueError("validation contract options must be a mapping")
                required_evidence = contract.get("required_evidence")
                if (required_evidence is not None and
                        (not isinstance(required_evidence, list) or
                         any(not isinstance(role, str) or not role.strip()
                             for role in required_evidence) or
                         len(required_evidence) != len(set(required_evidence)))):
                    raise ValueError("contract required_evidence must list unique non-empty roles")
            stages.append({"name": item["name"], "status": "pending", "gate": "pending",
                           "command": command, "outputs": outputs,
                           "env": env, "contract": contract,
                           "gate_criteria": gate_criteria,
                           "gate_review_lenses": gate_review_lenses,
                           "produces_student_results": produces_student_results,
                           "started_at": None, "completed_at": None, "attempts": 0})
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.init-", dir=run_dir.parent))
        try:
            for name in ("logs", "artifacts", "gates", "inputs"):
                (temporary / name).mkdir()
            (temporary / "workflow.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            input_records = []
            for index, (source, copy_input, source_integrity) in enumerate(prepared_inputs):
                destination = None
                if copy_input:
                    temporary_destination = temporary / "inputs" / f"{index:03d}-{source.name}"
                    shutil.copy2(source, temporary_destination)
                    destination = run_dir / "inputs" / temporary_destination.name
                input_records.append({"source": str(source),
                                      "snapshot": str(destination) if destination else None,
                                      "copy": copy_input, "source_integrity": source_integrity,
                                      "size": source_integrity["size"],
                                      "sha256": source_integrity["sha256"],
                                      "source_sha256": source_integrity["sha256"]})
            validation_contract_record = None
            if prepared_contract_sources is not None:
                # Run-bind the three contract source files: snapshot their exact content into
                # the run's own inputs area FIRST, then build the contract from those
                # snapshots (not the external, still-mutable declared paths) — so later edits
                # to the original distillation_scope/validation_profile/dataset_policy files
                # can never change what this run's frozen contract says.
                contract_source_dir = temporary / "inputs" / "contract_sources"
                contract_source_dir.mkdir()
                snapshots = {}
                contract_source_files = {}
                for key, source in prepared_contract_sources.items():
                    destination = contract_source_dir / f"{key}.yaml"
                    shutil.copy2(source, destination)
                    snapshots[key] = destination
                    contract_source_files[key] = {
                        "source": str(source),
                        "snapshot": str(run_dir / "inputs" / "contract_sources" / destination.name),
                        "sha256": artifact_digest(destination)["sha256"],
                    }
                components = build_validation_contract_components(
                    yaml.safe_load(snapshots["distillation_scope"].read_text()),
                    yaml.safe_load(snapshots["validation_profile"].read_text()),
                    yaml.safe_load(snapshots["dataset_policy"].read_text()),
                )
                validation_contract_record = _build_validation_contract_record(
                    components, source_files=contract_source_files
                )
            created_at = now()
            state = {"schema_version": SCHEMA_VERSION, "run_id": cfg["run_id"],
                     "created_at": created_at,
                     "updated_at": created_at, "workflow_config": str(run_dir / "workflow.yaml"),
                     "artifacts": [], "project_dir": str(project_dir), "inputs": input_records,
                     "code_revision": git_revision(project_dir), "events": [], "stages": stages,
                     "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                                     "started_at": created_at, "trigger": None}],
                     "recoveries": [], "adjudications": [], "pending_recovery": None,
                     # v7 additive operational metadata (safe empty defaults):
                     "runtime_attempts": [], "idempotency": {}, "action_approvals": {},
                     "scheduler_jobs": {},
                     # v8: write-once validation-target contract. None unless the workflow
                     # declares validation_contract_sources (established automatically, right
                     # here, from run-bound snapshots) or establish_validation_contract() is
                     # called later. A stage flagged produces_student_results cannot run until
                     # it is set. No recovery method
                     # (propose_recovery/start_iteration/verify_recovery_execution) ever writes
                     # this key, so recovery can re-run any stage — including one that
                     # cross-checks against the contract — but can never mutate it.
                     "validation_contract": validation_contract_record,
                     # v9 additive (all None/[] unless the workflow opts in explicitly):
                     # recovery_capability_roster overrides DEFAULT_RECOVERY_CAPABILITY_ROSTER
                     # for this run only; recovery_policy carries optional, explicitly-authored
                     # loop-safety limits (max_recovery_attempts, max_repeated_signature,
                     # allowed_action_types, cumulative_budget) that propose_recovery enforces
                     # only when present — no default retry count or budget is ever invented;
                     # protected_reference_roles names which artifact roles this run considers
                     # protected-reference so propose_recovery can fail closed if a recovery plan
                     # tries to route one into a training/acquisition input or output role.
                     "recovery_capability_roster": recovery_capability_roster_spec,
                     "recovery_policy": recovery_policy_spec,
                     "protected_reference_roles": protected_reference_roles_spec or [],
                     # v10 additive (None/{} unless the workflow opts in / a method is called):
                     # teacher_evidence_sources freezes this run's OPTIONAL Teacher-evidence input
                     # paths (resolved/validated above); teacher_validation_plan is set only by
                     # commit_teacher_validation_plan; stage_applicability is populated only by
                     # mark_stage_not_applicable.
                     "teacher_evidence_sources": prepared_teacher_evidence_sources,
                     "teacher_validation_plan": None,
                     "stage_applicability": {}}
            if validation_contract_record is not None:
                state["events"].append({
                    "at": validation_contract_record["established_at"],
                    "type": "validation_contract_established",
                    "contract_sha256": validation_contract_record["contract_sha256"],
                })
            (temporary / "manifest.json").write_text(json.dumps(state, indent=2) + "\n")
            if validation_contract_record is not None:
                (temporary / "validation_contract.json").write_text(
                    json.dumps(validation_contract_record, indent=2) + "\n"
                )
            temporary.rename(run_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return cls(run_dir)

    def save(self):
        self.state["updated_at"] = now()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.state_path)

    def set_recovery_policy(self, policy):
        """Explicitly set (or unset, with policy=None) this run's recovery_policy after
        initialization. Uses the identical validation initialize() applies to
        cfg.get("recovery_policy"): must be None or a dict. Never invents a value -- the
        caller supplies exactly what a human has decided, including None to represent
        "no campaign policy has been established yet" (propose_recovery() then enforces
        zero loop-safety limits, per _enforce_recovery_policy)."""
        if policy is not None and not isinstance(policy, dict):
            raise ValueError("recovery_policy must be a mapping or None")
        previous = self.state.get("recovery_policy")
        self.state["recovery_policy"] = policy
        self.state["events"].append({"at": now(), "type": "recovery_policy_updated",
                                     "previous": previous, "policy": policy})
        self.save()
        return policy

    def bind_new_input(self, source, *, copy=True):
        """Add a brand-new run-bound input after initialization. Mirrors, field-for-field,
        the validation and snapshot logic initialize() applies to each declared workflow
        input (existence check, artifact_digest, sequential-index snapshot copy) -- this is
        not a new binding policy, just the same mechanism made available post-init, since
        rebind_inputs() only re-verifies inputs already present in state["inputs"] and
        cannot append new ones."""
        source = Path(source).resolve()
        if not source.exists():
            raise FileNotFoundError(f"input to bind is missing: {source}")
        if copy and not source.is_file():
            raise ValueError("directory inputs must use copy=False and are hash-bound in place")
        source_integrity = artifact_digest(source)
        index = len(self.state["inputs"])
        destination = None
        if copy:
            temporary_destination = self.run_dir / "inputs" / f"{index:03d}-{source.name}"
            shutil.copy2(source, temporary_destination)
            destination = temporary_destination
        record = {"source": str(source), "snapshot": str(destination) if destination else None,
                  "copy": copy, "source_integrity": source_integrity,
                  "size": source_integrity["size"], "sha256": source_integrity["sha256"],
                  "source_sha256": source_integrity["sha256"]}
        self.state["inputs"].append(record)
        self.state["events"].append({"at": now(), "type": "input_bound", "source": str(source),
                                     "sha256": source_integrity["sha256"], "index": index})
        self.save()
        return record

    # --- v7 additive operational metadata (NO change to gate/retry/recovery semantics) --------
    def record_runtime_attempt(self, *, task_id, attempt_id, provenance_path, role="",
                               stage="", correlation_id="", failure_category=""):
        """Record a REFERENCE to a runtime invocation attempt (the provenance itself lives in the
        exchange). Additive: never touches stage/gate/artifact state."""
        entry = {"task_id": task_id, "attempt_id": attempt_id,
                 "provenance_path": str(provenance_path), "role": role, "stage": stage,
                 "correlation_id": correlation_id, "failure_category": failure_category,
                 "recorded_at": now()}
        self.state.setdefault("runtime_attempts", []).append(entry)
        self.save()
        return entry

    def action_seen(self, idempotency_key):
        """True if an action with this idempotency key was already recorded (duplicate guard)."""
        return idempotency_key in self.state.get("idempotency", {})

    def grant_action_approval(self, boundary, *, scope="run", note="", plan_sha256=None):
        """Record a human approval for an approval boundary (e.g. costly_teacher_labeling). This
        is the durable approval record the action dispatcher checks before a costly/side-effecting
        action may execute. Additive; independent of the recovery human-approval state machine."""
        record = {"granted": True, "scope": scope, "note": note, "at": now()}
        if plan_sha256 is not None:
            record["plan_sha256"] = str(plan_sha256)
            record["scope"] = "exact_acquisition_plan"
        self.state.setdefault("action_approvals", {})[boundary] = record
        self.save()

    def has_action_approval(self, boundary, plan_sha256=None):
        record = self.state.get("action_approvals", {}).get(boundary, {})
        if not record.get("granted"):
            return False
        if plan_sha256 is not None:
            return record.get("plan_sha256") == str(plan_sha256)
        return True

    # --- v7 additive: scheduler job lifecycle (pending -> collect -> resume) ------------------
    def record_scheduler_submission(self, job):
        """Record a submitted (external) scheduler job as PENDING. ``job`` is a dict with at least
        external_job_id, backend, idempotency_key, protocol_hash. Additive; no scientific
        semantics. The controller never waits on the job; a later collect resumes the stage."""
        jid = job["external_job_id"]
        rec = dict(job)
        rec.setdefault("state", "PENDING")
        rec["submitted_at"] = now()
        self.state.setdefault("scheduler_jobs", {})[jid] = rec
        self.save()
        return rec

    def get_scheduler_job(self, external_job_id):
        return self.state.get("scheduler_jobs", {}).get(external_job_id)

    def record_scheduler_collection(self, external_job_id, *, artifact_ref, artifact_sha256=""):
        """Record collected artifacts for a job and mark it COLLECTED (enables stage resume)."""
        jobs = self.state.get("scheduler_jobs", {})
        if external_job_id not in jobs:
            raise KeyError(f"unknown scheduler job: {external_job_id}")
        jobs[external_job_id]["state"] = "COLLECTED"
        jobs[external_job_id]["artifact_ref"] = artifact_ref
        jobs[external_job_id]["artifact_sha256"] = artifact_sha256
        jobs[external_job_id]["collected_at"] = now()
        self.save()
        return jobs[external_job_id]

    def record_action(self, idempotency_key, *, action_type="", status="", artifact_ref=""):
        if not idempotency_key:
            raise ValueError("idempotency_key is required to record an action")
        self.state.setdefault("idempotency", {})[idempotency_key] = {
            "action_type": action_type, "status": status, "artifact_ref": artifact_ref,
            "recorded_at": now()}
        self.save()

    def begin_stage_execution(self, name, *, pid=None, runner_id=""):
        """Mark a (typically external/long) stage running WITH runner metadata so a killed run
        can later be detected as stale. Operational only."""
        stage = self.stage(name)
        ts = now()
        stage["status"] = "running"
        stage["runner"] = {"pid": pid, "runner_id": runner_id, "started_at": ts, "last_update": ts}
        self.save()

    def heartbeat_stage(self, name):
        stage = self.stage(name)
        runner = stage.get("runner")
        if runner is not None:
            runner["last_update"] = now()
            self.save()

    def reconcile_stale_stages(self, *, threshold_s, current_time=None, is_pid_alive=None):
        """Clear stages stuck in 'running' after an external kill. A running stage whose runner
        heartbeat is older than ``threshold_s`` (or whose pid is not alive) is set to
        'interrupted' and a ``stale_running_recovered`` event is recorded. This is OPERATIONAL
        retry, DISTINCT from scientific recovery: it never marks a stage PASS and never touches
        artifacts, gate results, or recovery state. Stages without runner metadata are untouched.
        Returns the list of reconciled stage names."""
        now_dt = current_time or dt.datetime.now(dt.timezone.utc)
        is_pid_alive = is_pid_alive or (lambda pid: False)
        reconciled = []
        for stage in self.state["stages"]:
            if stage.get("status") != "running":
                continue
            runner = stage.get("runner")
            if not runner:
                continue
            stale = False
            last = runner.get("last_update")
            if last:
                try:
                    stale = (now_dt - dt.datetime.fromisoformat(last)).total_seconds() > threshold_s
                except ValueError:
                    stale = False
            pid = runner.get("pid")
            if pid is not None and not is_pid_alive(pid):
                stale = True
            if stale:
                stage["status"] = "interrupted"
                runner["interrupted_at"] = now_dt.isoformat()
                self.state.setdefault("events", []).append(
                    {"type": "stale_running_recovered", "stage": stage["name"],
                     "at": now_dt.isoformat()})
                reconciled.append(stage["name"])
        if reconciled:
            self.save()
        return reconciled

    def stage(self, name):
        for stage in self.state["stages"]:
            if stage["name"] == name:
                return stage
        raise KeyError(f"unknown stage: {name}")

    def _previous_passed(self, name):
        for stage in self.state["stages"]:
            if stage["name"] == name:
                return
            if (stage["gate"] not in {"PASS", "NOT_APPLICABLE"} and
                    not self._stage_has_accepted_adjudication(stage["name"])):
                raise RuntimeError(f"stage {name!r} blocked: {stage['name']!r} gate is {stage['gate']}")
            # A NOT_APPLICABLE stage was never executed and so registers no artifacts (see
            # mark_stage_not_applicable) -- verify_stage_artifacts would otherwise raise on the
            # zero-record case it is designed to reject for every OTHER gate outcome.
            if stage["gate"] != "NOT_APPLICABLE":
                self.verify_stage_artifacts(stage["name"])

    def _stage_has_accepted_adjudication(self, name):
        resolution = self.stage(name).get("effective_resolution")
        if not resolution or resolution.get("decision") != "ACCEPT_DECLARED_LIMITATION":
            return False
        matches = [item for item in self.state.get("adjudications", [])
                   if item.get("id") == resolution.get("adjudication_id")]
        if len(matches) != 1 or matches[0].get("status") != "accepted":
            return False
        try:
            verify_artifact(matches[0]["path"], matches[0]["integrity"])
        except (KeyError, FileNotFoundError, RuntimeError):
            return False
        return True

    def verify_inputs(self):
        expected_revision = self.state.get("code_revision")
        if expected_revision and expected_revision.get("available"):
            current_revision = git_revision(self.state["project_dir"])
            if current_revision != expected_revision:
                raise RuntimeError("project code changed after run initialization; start a new run")
        for record in self.state.get("inputs", []):
            source = Path(record["source"])
            if record.get("snapshot"):
                snapshot = Path(record["snapshot"])
                if not snapshot.is_file() or sha256_file(snapshot) != record["sha256"]:
                    raise RuntimeError(f"run input snapshot integrity check failed: {snapshot}")
            try:
                verify_artifact(source, record.get("source_integrity", {"kind": "file",
                                                                        "size": record["size"],
                                                                        "sha256": record["source_sha256"]}))
            except (FileNotFoundError, RuntimeError):
                raise RuntimeError(f"declared workflow input changed after initialization: {source}")

    def rebind_inputs(self):
        """Explicitly accept changed inputs and invalidate all prior stage results."""
        self._ensure_no_pending_recovery()
        revisions = sum(1 for event in self.state["events"] if event["type"] == "inputs_rebound") + 1
        revision_dir = self.run_dir / "inputs" / f"revision-{revisions:03d}"
        if revision_dir.exists():
            raise FileExistsError(f"input revision already exists: {revision_dir}")
        prepared = []
        for index, record in enumerate(self.state.get("inputs", [])):
            source = Path(record["source"])
            integrity = artifact_digest(source)
            if record.get("copy", True) and not source.is_file():
                raise ValueError("copied input became a directory; declare it with copy: false")
            prepared.append((index, record, source, integrity))

        temporary = Path(tempfile.mkdtemp(prefix=f".revision-{revisions:03d}-",
                                          dir=self.run_dir / "inputs"))
        try:
            new_records, changes = [], []
            for index, record, source, integrity in prepared:
                old_snapshot = record.get("snapshot")
                snapshot = None
                if record.get("copy", True):
                    temporary_snapshot = temporary / f"{index:03d}-{source.name}"
                    shutil.copy2(source, temporary_snapshot)
                    snapshot = revision_dir / temporary_snapshot.name
                updated = dict(record)
                updated.update(snapshot=str(snapshot) if snapshot else None,
                               source_integrity=integrity, size=integrity["size"],
                               sha256=integrity["sha256"], source_sha256=integrity["sha256"])
                new_records.append(updated)
                old_sha = record.get("source_integrity", {}).get("sha256",
                                                                  record["source_sha256"])
                changes.append({"source": str(source), "old_sha256": old_sha,
                                "new_sha256": integrity["sha256"],
                                "old_snapshot": old_snapshot,
                                "new_snapshot": str(snapshot) if snapshot else None})
            temporary.rename(revision_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.state["inputs"] = new_records
        if self.state["stages"]:
            self.invalidate_from(self.state["stages"][0]["name"], include_stage=True)
        self.state["events"].append({"at": now(), "type": "inputs_rebound",
                                     "revision": revisions, "changes": changes})
        self.save()
        return changes

    def _stage_index(self, name):
        return next(i for i, stage in enumerate(self.state["stages"]) if stage["name"] == name)

    def invalidate_from(self, name, include_stage=False):
        """Invalidate stale downstream state and remove its artifact records."""
        start = self._stage_index(name) + (0 if include_stage else 1)
        affected = {s["name"] for s in self.state["stages"][start:]}
        if not affected:
            return
        self.quarantine_artifacts(affected)
        for stage in self.state["stages"][start:]:
            stage.update(status="pending", gate="pending", started_at=None, completed_at=None)
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] not in affected]
        self.state["events"].append({"at": now(), "type": "downstream_invalidated",
                                     "after": name, "stages": sorted(affected)})

    def quarantine_artifacts(self, stage_names, exclude_paths=None):
        """Move invalidated run-local outputs aside so they cannot be re-registered by accident."""
        excluded = {Path(path).resolve() for path in (exclude_paths or [])}
        records = [a for a in self.state["artifacts"] if a["stage"] in set(stage_names)]
        if not records:
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for record in sorted(records, key=lambda a: len(Path(a["path"]).parts)):
            source = Path(record["path"])
            if (source.resolve() in excluded or not source.exists() or
                    not source.is_relative_to(self.run_dir)):
                continue
            destination = self.run_dir / "stale" / stamp / record["stage"] / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.move(str(source), str(destination))

    def quarantine_declared_outputs(self, stage):
        """Move unregistered leftovers from a failed attempt out of the output paths."""
        paths = sorted({(self.run_dir / relative).resolve()
                        for relative in stage.get("outputs", [])},
                       key=lambda path: len(path.parts))
        existing = [path for path in paths
                    if path.exists() and path.is_relative_to(self.run_dir)]
        if not existing:
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for source in existing:
            if not source.exists():
                continue
            relative = source.relative_to(self.run_dir)
            destination = self.run_dir / "stale" / stamp / stage["name"] / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

    def stage_artifacts(self, name):
        return [a for a in self.state["artifacts"] if a["stage"] == name]

    def verify_stage_artifacts(self, name):
        records = self.stage_artifacts(name)
        if not records:
            raise RuntimeError(f"stage {name!r} has no registered artifacts")
        for record in records:
            path = Path(record["path"])
            try:
                verify_artifact(path, record)
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(f"artifact integrity check failed for stage {name!r}: {path}") from exc
        return records

    # --- v8: write-once validation-target contract ------------------------------------------
    VALIDATION_CONTRACT_COMPONENTS = VALIDATION_CONTRACT_COMPONENTS

    def establish_validation_contract(self, components, source_files=None):
        """Freeze the Teacher-applicability/validation-scope/dataset-split-policy contract.

        Write-once for the lifetime of this run: an identical re-establishment (same canonical
        component values) is an idempotent no-op; any differing content is a hard failure. This
        is the ONLY method (besides RunController.initialize's automatic establishment, which
        shares the same record-construction path via ``_build_validation_contract_record``)
        that ever writes ``self.state["validation_contract"]`` — no recovery method touches it,
        so recovery can re-run any stage (including one that cross-checks against the contract)
        but can never mutate, re-establish, or replace it. A genuine change to the Teacher
        applicability domain, validation scope, or dataset split policy requires a new run, not
        a call to this method on an existing one.
        """
        candidate = _build_validation_contract_record(components, source_files)
        existing = self.state.get("validation_contract")
        if existing is not None:
            if existing["contract_sha256"] == candidate["contract_sha256"]:
                return existing
            raise ValueError(
                "validation contract is already established for this run with different "
                "content; changing the Teacher applicability domain, validation scope, or "
                "dataset split policy requires a new run, not a mutation of this one"
            )
        if any(stage["status"] != "pending" for stage in self.state["stages"]):
            raise RuntimeError(
                "validation contract must be established before any stage in this run executes"
            )
        record = candidate
        self.state["validation_contract"] = record
        lock_path = self.run_dir / "validation_contract.json"
        lock_path.write_text(json.dumps(record, indent=2) + "\n")
        self.state["events"].append({"at": now(), "type": "validation_contract_established",
                                     "contract_sha256": record["contract_sha256"]})
        self.save()
        return record

    def _require_validation_contract_for_student_stage(self, name, stage):
        if stage.get("produces_student_results") and self.state.get("validation_contract") is None:
            raise RuntimeError(
                f"stage {name!r} produces Student results and cannot run until "
                "establish_validation_contract has been called for this run"
            )

    def _mark_student_stage_completed(self, stage):
        if stage.get("produces_student_results"):
            contract = self.state.get("validation_contract")
            if contract is not None:
                contract["student_stage_ever_completed"] = True

    # --- v10: NOT_APPLICABLE stage lifecycle -------------------------------------------------

    def mark_stage_not_applicable(self, name, *, reason=""):
        """Resolve ``name`` as genuinely NOT_APPLICABLE instead of running it: the caller's own
        evidence establishes there is nothing for this stage to do at all (e.g. no admissible
        Teacher-validation component this run's committed teacher_validation_plan selected
        requires it) -- categorically different from PASS (something ran and satisfied its
        criteria), REVISE/FAIL (something ran and did not), or leaving the stage pending
        (nothing has been decided yet). Never a substitute for actually running a stage the
        workflow genuinely requires; this controller never decides applicability itself -- it
        only records a caller-supplied decision, fail-closed on the same preconditions
        ``run_stage`` enforces.

        Deliberately bypasses ``record_gate`` entirely (rather than being a new accepted verdict
        there): record_gate's non-PASS branch invalidates downstream stages and opens a pending
        recovery, semantics that make sense for a stage that ran and failed its criteria but
        would be actively wrong for a stage that never ran because it does not apply. Setting
        ``gate`` directly here is the one and only path that ever produces the NOT_APPLICABLE
        gate value.
        """
        self._ensure_no_pending_recovery()
        self._previous_passed(name)
        stage = self.stage(name)
        if stage["status"] != "pending":
            raise RuntimeError(
                f"stage {name!r} cannot be marked NOT_APPLICABLE: status is "
                f"{stage['status']!r}, not pending"
            )
        marked_at = now()
        stage.update(status="not_applicable", gate="NOT_APPLICABLE",
                    started_at=marked_at, completed_at=marked_at)
        self.state["stage_applicability"][name] = {"decided_at": marked_at, "reason": reason or ""}
        self.state["events"].append({"at": marked_at, "type": "stage_marked_not_applicable",
                                     "stage": name, "reason": reason or ""})
        self.save()
        return stage

    def run_stage(self, name):
        self._ensure_no_pending_recovery()
        self.verify_inputs()
        self._previous_passed(name)
        stage = self.stage(name)
        if not stage["command"]:
            raise ValueError(f"stage {name!r} has no command")
        self._require_validation_contract_for_student_stage(name, stage)
        self.invalidate_from(name)
        self.quarantine_artifacts({name})
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] != name]
        self.quarantine_declared_outputs(stage)
        context = {"run_dir": str(self.run_dir), "artifacts_dir": str(self.run_dir / "artifacts"),
                   "project_dir": self.state["project_dir"], "python": sys.executable}
        command = [str(x).format(**context) for x in stage["command"]]
        if stage.get("env"):
            if command and Path(command[0]).resolve() == Path(sys.executable).resolve():
                command[0] = "python"
            command = ["conda", "run", "--no-capture-output", "-n", stage["env"], *command]
        stage.update(status="running", started_at=now(), attempts=stage["attempts"] + 1, gate="pending")
        self.save()
        log_path = self.run_dir / "logs" / f"{name}.attempt-{stage['attempts']}.log"
        environment = os.environ.copy()
        project_dir = self.state["project_dir"]
        environment["PYTHONPATH"] = project_dir + os.pathsep + environment.get("PYTHONPATH", "")
        with log_path.open("w") as log:
            try:
                result = subprocess.run(command, cwd=self.run_dir, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                log.write(f"stage launch failed: {exc}\n")
                stage.update(status="failed", completed_at=now())
                self.state["events"].append({"at": now(), "type": "stage_failed",
                                             "stage": name, "returncode": None,
                                             "error": str(exc), "log": str(log_path)})
                self.save()
                raise RuntimeError(f"stage {name!r} could not be launched; see {log_path}") from exc
        stage["completed_at"] = now()
        if result.returncode != 0:
            stage["status"] = "failed"
            self.state["events"].append({"at": now(), "type": "stage_failed", "stage": name,
                                         "returncode": result.returncode, "log": str(log_path)})
            self.save()
            raise RuntimeError(f"stage {name!r} failed; see {log_path}")
        output_paths = []
        for relative in stage["outputs"]:
            path = (self.run_dir / relative).resolve()
            if not path.exists():
                stage["status"] = "failed"
                self.save()
                raise FileNotFoundError(f"declared output missing: {path}")
            output_paths.append(path)
        try:
            self._validate_external_contract(stage, output_paths)
        except Exception as exc:
            stage["status"] = "failed"
            self.state["events"].append({"at": now(), "type": "stage_contract_failed",
                                         "stage": name, "error": str(exc)})
            self.save()
            raise
        stage["status"] = "completed"
        for path in output_paths:
            self.register_artifact(name, path)
        self._mark_student_stage_completed(stage)
        self.save()

    def register_artifact(self, stage, path):
        path = Path(path).resolve()
        digest = artifact_digest(path)
        record = {"stage": stage, "path": str(path), **digest, "registered_at": now()}
        self.state["artifacts"].append(record)
        return record

    def _registered_artifact(self, path):
        path = str(Path(path).resolve())
        matches = [record for record in self.state["artifacts"] if record["path"] == path]
        if len(matches) != 1:
            raise ValueError(f"required upstream artifact is not uniquely registered: {path}")
        verify_artifact(path, matches[0])
        return matches[0]

    def _validation_evidence_allowlist(self, current_artifacts, current_stage=None):
        """Paths that a validation report may cite as run-bound evidence."""
        paths = {Path(path).resolve() for path in current_artifacts}
        for record in self.state.get("inputs", []):
            paths.add(Path(record["source"]).resolve())
            if record.get("snapshot"):
                paths.add(Path(record["snapshot"]).resolve())
        if current_stage is not None:
            upstream = {stage["name"] for stage in
                        self.state["stages"][:self._stage_index(current_stage)]}
            paths.update(Path(record["path"]).resolve()
                         for record in self.state["artifacts"]
                         if record.get("stage") in upstream)
        return sorted(paths)

    def _validate_external_contract(self, stage, artifacts, enforce_required_pass=False):
        contract = stage.get("contract")
        if not contract:
            return None
        context = {"run_dir": str(self.run_dir), "artifacts_dir": str(self.run_dir / "artifacts"),
                   "project_dir": self.state["project_dir"]}
        manifest = Path(str(contract["manifest"]).format(**context))
        if not manifest.is_absolute():
            manifest = self.run_dir / manifest
        manifest = manifest.resolve()
        if manifest not in {Path(path).resolve() for path in artifacts}:
            raise ValueError("external contract manifest must be included in --artifact")
        kind = contract.get("kind")
        if kind == "md_manifest":
            committee = Path(str(contract["committee_manifest"]).format(**context))
            if not committee.is_absolute():
                committee = self.run_dir / committee
            self._registered_artifact(committee)
            return validate_md_manifest(manifest, committee, artifacts,
                                        contract.get("required_evidence"))
        if kind == "validation_manifest":
            return validate_validation_manifest(manifest, contract.get("validator"),
                                                format_context(contract.get("options"), context), artifacts,
                                                self._validation_evidence_allowlist(
                                                    artifacts, stage.get("name")),
                                                enforce_required_pass)
        raise ValueError(f"unknown external stage contract: {kind!r}")

    def complete_external_stage(self, name, artifacts):
        """Register artifacts produced by an agent, scheduler, or external tool."""
        self._ensure_no_pending_recovery()
        self.verify_inputs()
        self._previous_passed(name)
        stage = self.stage(name)
        self._require_validation_contract_for_student_stage(name, stage)
        if not artifacts:
            raise ValueError("at least one artifact is required")
        resolved = []
        for path in artifacts:
            path = Path(path)
            if not path.is_absolute():
                path = self.run_dir / path
            if not path.exists():
                raise FileNotFoundError(f"external artifact is missing: {path}")
            resolved.append(path.resolve())
        submitted = set(resolved)
        declared = {(self.run_dir / relative).resolve() for relative in stage.get("outputs", [])}
        missing_outputs = declared - submitted
        if missing_outputs:
            raise ValueError("external stage is missing declared outputs: " +
                             ", ".join(map(str, sorted(missing_outputs))))
        contract_result = self._validate_external_contract(stage, resolved)
        self.invalidate_from(name)
        self.quarantine_artifacts({name}, exclude_paths=resolved)
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] != name]
        stage.update(status="completed", started_at=stage.get("started_at") or now(),
                     completed_at=now(), attempts=stage["attempts"] + 1, gate="pending")
        for path in resolved:
            self.register_artifact(name, path)
        self._mark_student_stage_completed(stage)
        self.state["events"].append({"at": now(), "type": "external_stage_completed",
                                     "stage": name, "artifacts": [str(x) for x in resolved],
                                     "contract": stage.get("contract"),
                                     "contract_validated": contract_result is not None})
        self.save()

    def _validate_vote_bundle(self, name, votes_path):
        bundle = json.loads(Path(votes_path).read_text())
        criteria = bundle.get("criteria")
        votes = bundle.get("votes")
        if bundle.get("stage", bundle.get("gate")) != name:
            raise ValueError("vote bundle gate/stage does not match the controller stage")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("vote bundle must contain non-empty criteria")
        bound_criteria = self.stage(name).get("gate_criteria")
        if not bound_criteria:
            raise ValueError(
                "Judge PASS/REVISE bundle requires gate.criteria bound at run initialization"
            )
        if criteria != bound_criteria:
            raise ValueError("vote bundle criteria do not match the run-bound gate criteria")
        bound_lenses = self.stage(name).get("gate_review_lenses")
        if not bound_lenses:
            raise ValueError("Judge bundle requires review lenses bound at run initialization")
        if bundle.get("review_lenses") != bound_lenses:
            raise ValueError("vote bundle review lenses do not match the run-bound review lenses")
        if not isinstance(votes, list) or len(votes) != len(bound_lenses):
            raise ValueError("exactly three judge votes are required")
        verdicts = []
        judge_ids = set()
        vote_lenses = []
        expected_lenses = [lens["id"] for lens in bound_lenses]
        for index, vote in enumerate(votes, 1):
            allowed = {"judge_id", "id", "review_lens", "verdict",
                       "criteria_checked", "rationale", "required_fix"}
            if not isinstance(vote, dict) or set(vote) - allowed:
                raise ValueError("judge vote contains unknown fields")
            vote_payload = {key: vote.get(key) for key in (
                "review_lens", "verdict", "criteria_checked", "rationale", "required_fix"
            )}
            validated = validate_judge_vote(
                vote_payload, criteria, review_lens=expected_lenses[index - 1]
            )
            verdict = validated["verdict"]
            review_lens = validated["review_lens"]
            if "judge_id" in vote and "id" in vote:
                raise ValueError("judge vote must use either judge_id or id, not both")
            raw_judge_id = vote.get("judge_id", vote.get("id", index))
            if (isinstance(raw_judge_id, bool) or
                    not isinstance(raw_judge_id, (str, int)) or
                    not str(raw_judge_id).strip()):
                raise ValueError("judge identifier must be a non-empty string or integer")
            judge_id = str(raw_judge_id)
            if judge_id in judge_ids:
                raise ValueError("judge identifiers must be unique")
            judge_ids.add(judge_id)
            vote_lenses.append(review_lens)
            verdicts.append(verdict)
        if vote_lenses != expected_lenses:
            raise ValueError(
                "judge votes must match the ordered run-bound review lenses exactly once"
            )
        decision = "FAIL" if "FAIL" in verdicts else (
            "PASS" if verdicts == ["PASS"] * len(bound_lenses) else "REVISE"
        )
        if bundle.get("decision") != decision:
            raise ValueError("vote bundle decision does not match the recomputed decision")
        expected = {a["path"]: a["sha256"] for a in self.verify_stage_artifacts(name)}
        if not expected:
            raise ValueError("a Judge gate requires at least one registered artifact")
        if bundle.get("artifact_sha256") != expected:
            raise ValueError("vote bundle artifact hashes do not match current registered artifacts")
        return decision, bundle

    def gate_context(self, name):
        """Return verified hashes, criteria, and run-bound Judge lenses."""
        stage = self.stage(name)
        if stage["status"] != "completed":
            raise RuntimeError("gate context requires a completed stage")
        if not stage.get("gate_criteria"):
            raise ValueError(
                "Judge gate requires gate.criteria bound at run initialization"
            )
        hashes = {record["path"]: record["sha256"]
                  for record in self.verify_stage_artifacts(name)}
        if not hashes:
            raise ValueError("a Judge gate requires at least one registered artifact")
        return {"stage": name, "criteria": list(stage["gate_criteria"]),
                "review_lenses": [dict(lens) for lens in stage["gate_review_lenses"]],
                "artifact_sha256": hashes}

    def record_gate(self, name, verdict=None, evidence=None, votes_path=None):
        self._ensure_no_pending_recovery()
        bundle = None
        if votes_path:
            verdict, bundle = self._validate_vote_bundle(name, votes_path)
        elif verdict == "PASS":
            raise ValueError("PASS requires --votes with three validated judge votes")
        if verdict not in {"PASS", "REVISE", "FAIL"}:
            raise ValueError("verdict must be PASS, REVISE, or FAIL")
        stage = self.stage(name)
        if stage["status"] != "completed":
            raise RuntimeError("a gate can only judge a completed stage")
        if verdict == "PASS":
            self._require_verified_recovery_for_pass(name)
        if verdict == "PASS" and (stage.get("contract") or {}).get("kind") == "validation_manifest":
            self._validate_external_contract(
                stage, [record["path"] for record in self.stage_artifacts(name)],
                enforce_required_pass=True,
            )
        saved_votes = None
        if votes_path:
            iteration_id = self._current_iteration()["id"]
            saved_votes = self.run_dir / "gates" / f"{name}.iteration-{iteration_id:03d}.votes.json"
            saved_votes.write_text(json.dumps(bundle, indent=2) + "\n")
        stage["gate"] = verdict
        if verdict == "PASS":
            iteration = self._current_iteration()
            trigger = iteration.get("trigger")
            if trigger and trigger.get("failed_stage") == name:
                recovery = next(item for item in self.state.get("recoveries", [])
                                if item.get("id") == trigger["recovery_id"])
                recovery.update(status="resolved", resolved_at=now())
                iteration["recovery_execution"]["status"] = "resolved"
        if verdict != "PASS":
            self.invalidate_from(name)
        gate_time = now()
        self.state["events"].append({"at": gate_time, "type": "gate", "stage": name,
                                     "verdict": verdict, "evidence": evidence,
                                     "votes": str(saved_votes) if saved_votes else None,
                                     "vote_bundle": bundle})
        if verdict != "PASS":
            self.state["pending_recovery"] = {
                "status": "required", "failed_stage": name, "verdict": verdict,
                "gate_recorded_at": gate_time,
                "artifact_sha256": {record["path"]: record["sha256"]
                                    for record in self.verify_stage_artifacts(name)},
                "votes_integrity": artifact_digest(saved_votes) if saved_votes else None,
            }
        self.save()

    def _ensure_no_pending_recovery(self):
        pending = self.state.get("pending_recovery")
        if pending:
            raise RuntimeError(
                "a REVISE/FAIL recovery is pending; propose, approve, and start the next iteration"
            )


    def _pending_gate_votes_path(self, pending):
        recorded_at = pending.get("gate_recorded_at")
        stage = pending.get("failed_stage")
        matches = [event for event in self.state.get("events", [])
                   if event.get("type") == "gate" and event.get("stage") == stage and
                   event.get("at") == recorded_at and event.get("votes")]
        if len(matches) != 1:
            raise RuntimeError("pending gate vote record is missing or ambiguous")
        path = Path(matches[0]["votes"])
        if not path.is_absolute():
            path = self.run_dir / path
        path = path.resolve()
        if pending.get("votes_integrity"):
            verify_artifact(path, pending["votes_integrity"])
        return path

    def _stage_contract_passed_for_adjudication(self, stage):
        contract = stage.get("contract") or {}
        if not contract or contract.get("kind") != "validation_manifest":
            return True
        records = [record["path"] for record in self.stage_artifacts(stage["name"])]
        self._validate_external_contract(stage, records, enforce_required_pass=True)
        completions = [event for event in self.state.get("events", [])
                       if event.get("type") == "external_stage_completed" and
                       event.get("stage") == stage["name"] and
                       event.get("contract_validated") is True]
        if not completions:
            raise RuntimeError("adjudication requires a completed deterministic contract validation")
        return True

    def _validate_adjudication_eligibility(self, pending):
        if not pending or pending.get("status") != "required":
            raise RuntimeError("no REVISE gate is waiting for adjudication")
        if pending.get("verdict") != "REVISE":
            raise RuntimeError("human adjudication continuation is only available for REVISE")
        stage = self.stage(pending["failed_stage"])
        if stage["status"] != "completed":
            raise RuntimeError("adjudication requires a completed stage")
        current_hashes = {record["path"]: record["sha256"]
                          for record in self.verify_stage_artifacts(stage["name"])}
        if current_hashes != pending.get("artifact_sha256"):
            raise RuntimeError("adjudication artifact binding does not match current artifacts")
        votes_path = self._pending_gate_votes_path(pending)
        decision, bundle = self._validate_vote_bundle(stage["name"], votes_path)
        if decision != "REVISE":
            raise RuntimeError("adjudication requires a raw REVISE vote bundle")
        verdicts = [vote.get("verdict") for vote in bundle.get("votes", [])]
        if len(verdicts) != 3:
            raise RuntimeError("adjudication requires three completed Judge votes")
        if "FAIL" in verdicts:
            raise RuntimeError("adjudication continuation is forbidden after any FAIL vote")
        for vote in bundle.get("votes", []):
            text = " ".join(str(vote.get(field, ""))
                            for field in ("rationale", "required_fix")).lower()
            if "judge invocation failed" in text or "runtime failed" in text:
                raise RuntimeError("adjudication is forbidden after Judge runtime failure")
        self._stage_contract_passed_for_adjudication(stage)
        return stage, votes_path, bundle

    def propose_adjudication(self, proposal_path):
        """Bind a proposed human scientific adjudication to an eligible REVISE gate."""
        pending = self.state.get("pending_recovery")
        stage, votes_path, bundle = self._validate_adjudication_eligibility(pending)
        source = Path(proposal_path).resolve()
        proposal = json.loads(source.read_text())
        if proposal.get("schema_version") != 1:
            raise ValueError("adjudication proposal requires schema_version=1")
        if proposal.get("stage") != stage["name"]:
            raise ValueError("adjudication proposal stage does not match the pending gate")
        decision = proposal.get("proposed_decision")
        if decision not in ADJUDICATION_DECISIONS:
            raise ValueError(f"invalid adjudication decision: {decision!r}")
        if not isinstance(proposal.get("rationale"), str) or not proposal["rationale"].strip():
            raise ValueError("adjudication proposal requires rationale")
        if decision == "ACCEPT_DECLARED_LIMITATION":
            limitations = proposal.get("accepted_limitations")
            restrictions = proposal.get("downstream_claim_restrictions")
            if (not isinstance(limitations, list) or not limitations or
                    any(not isinstance(item, str) or not item.strip() for item in limitations)):
                raise ValueError("ACCEPT_DECLARED_LIMITATION requires accepted_limitations")
            if proposal.get("scope_effect") != ADJUDICATION_SCOPE_EFFECT:
                raise ValueError("adjudication cannot expand applicability")
            if (not isinstance(restrictions, list) or not restrictions or
                    any(not isinstance(item, str) or not item.strip() for item in restrictions)):
                raise ValueError(
                    "ACCEPT_DECLARED_LIMITATION requires downstream_claim_restrictions"
                )
        if decision == "REQUIRE_SCIENTIFIC_RECOVERY":
            recovery_route = proposal.get("recovery_route")
            if not isinstance(recovery_route, str) or not recovery_route.strip():
                raise ValueError("REQUIRE_SCIENTIFIC_RECOVERY requires recovery_route")

        adjudication_id = len(self.state.setdefault("adjudications", [])) + 1
        adjudication_dir = self.run_dir / "adjudications"
        adjudication_dir.mkdir(exist_ok=True)
        destination = adjudication_dir / f"adjudication-{adjudication_id:03d}.json"
        record = {
            "id": adjudication_id,
            "schema_version": 1,
            "status": "proposed",
            "proposed_at": now(),
            "source": str(source),
            "stage": stage["name"],
            "raw_gate": "REVISE",
            "raw_votes": str(votes_path),
            "raw_votes_integrity": artifact_digest(votes_path),
            "raw_vote_verdicts": [vote["verdict"] for vote in bundle["votes"]],
            "gate_binding": {
                "recorded_at": pending["gate_recorded_at"],
                "artifact_sha256": pending["artifact_sha256"],
                "votes_integrity": pending.get("votes_integrity"),
            },
            "proposal": proposal,
            "human_decision": None,
        }
        destination.write_text(json.dumps(record, indent=2) + "\n")
        record["path"] = str(destination)
        record["integrity"] = artifact_digest(destination)
        self.state["adjudications"].append(record)
        self.state["pending_recovery"] = {
            "status": "adjudication_proposed", "adjudication_id": adjudication_id
        }
        self.state["events"].append({"at": now(), "type": "adjudication_proposed",
                                     "adjudication_id": adjudication_id,
                                     "path": str(destination),
                                     "integrity": record["integrity"]})
        self.save()
        return record

    def _pending_adjudication_record(self):
        pending = self.state.get("pending_recovery")
        if not pending or pending.get("status") != "adjudication_proposed":
            raise RuntimeError("no adjudication proposal is waiting for a human decision")
        matches = [item for item in self.state.get("adjudications", [])
                   if item.get("id") == pending.get("adjudication_id")]
        if len(matches) != 1:
            raise RuntimeError("pending adjudication record is missing or ambiguous")
        try:
            verify_artifact(matches[0]["path"], matches[0]["integrity"])
        except (KeyError, FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError("pending adjudication integrity check failed") from exc
        return matches[0]

    def decide_adjudication(self, decided_by, decision=None, note=None):
        """Record a human adjudication decision without changing the raw Judge gate."""
        record = self._pending_adjudication_record()
        if not isinstance(decided_by, str) or not decided_by.strip():
            raise ValueError("adjudication decision requires decided_by")
        proposal = record["proposal"]
        decision = decision or proposal["proposed_decision"]
        if decision != proposal["proposed_decision"]:
            raise ValueError("human decision must match the proposed adjudication decision")
        if decision not in ADJUDICATION_DECISIONS:
            raise ValueError(f"invalid adjudication decision: {decision!r}")
        pending_gate = {
            "status": "required",
            "failed_stage": record["stage"],
            "verdict": record["raw_gate"],
            "gate_recorded_at": record["gate_binding"]["recorded_at"],
            "artifact_sha256": record["gate_binding"]["artifact_sha256"],
            "votes_integrity": record["gate_binding"].get("votes_integrity"),
        }
        self._validate_adjudication_eligibility(pending_gate)
        decision_record = {
            "decision": decision,
            "decided_at": now(),
            "decided_by": decided_by.strip(),
            "note": note or "",
        }
        record["human_decision"] = decision_record
        if decision == "ACCEPT_DECLARED_LIMITATION":
            record["status"] = "accepted"
            self.stage(record["stage"])["effective_resolution"] = {
                "type": "human_scientific_adjudication",
                "adjudication_id": record["id"],
                "raw_gate": record["raw_gate"],
                "raw_vote_verdicts": list(record["raw_vote_verdicts"]),
                "decision": decision,
                "accepted_limitations": list(proposal["accepted_limitations"]),
                "scope_effect": proposal["scope_effect"],
                "downstream_claim_restrictions": list(
                    proposal["downstream_claim_restrictions"]
                ),
            }
            self.state["pending_recovery"] = None
        else:
            record["status"] = "requires_scientific_recovery"
            self.state["pending_recovery"] = {
                **pending_gate, "adjudication_id": record["id"]
            }
        adjudication_path = Path(record["path"])
        adjudication_path.write_text(json.dumps(record, indent=2) + "\n")
        record["integrity"] = artifact_digest(adjudication_path)
        self.state["events"].append({"at": now(), "type": "adjudication_decided",
                                     "adjudication_id": record["id"],
                                     **decision_record})
        self.save()
        return record

    def _current_iteration(self):
        iterations = self.state.setdefault("iterations", [])
        if not iterations:
            iterations.append({"id": 1, "parent_iteration": None, "status": "active",
                               "started_at": self.state.get("created_at", now()),
                               "trigger": None})
        return iterations[-1]

    def _recovery_capability_roster(self):
        return self.state.get("recovery_capability_roster") or DEFAULT_RECOVERY_CAPABILITY_ROSTER

    def _resolve_recovery_routing(self, plan):
        """Resolve responsible_capability/responsible_agent against this run's roster.

        A plan may supply either responsible_capability (preferred; a registered capability
        resolved at runtime to a role from this run's roster) or responsible_agent directly
        (legacy path). Both may be supplied together only if consistent. Returns the resolved
        agent string and the capability (or None if the legacy path was used).
        """
        roster = self._recovery_capability_roster()
        capability = plan.get("responsible_capability")
        agent = plan.get("responsible_agent")
        if capability is not None:
            if not isinstance(capability, str) or not capability.strip():
                raise ValueError("recovery plan responsible_capability must be a non-empty string")
            if capability not in roster:
                raise ValueError(
                    f"recovery responsible_capability is not registered in this run's roster: "
                    f"{capability!r}"
                )
            resolved_agent = roster[capability]
            if agent is not None and agent != resolved_agent:
                raise ValueError(
                    "recovery plan responsible_agent does not match responsible_capability's "
                    f"roster-resolved role ({resolved_agent!r})"
                )
            return resolved_agent, capability
        if not isinstance(agent, str) or not agent.strip():
            raise ValueError("recovery plan requires non-empty responsible_agent or responsible_capability")
        if agent not in set(roster.values()):
            raise ValueError("recovery responsible_agent is not a registered recovery role")
        return agent, None

    def _validate_diagnosis_binding(self, plan):
        """If present, hash-verify the diagnosis artifact and its triggering evidence.

        Fails closed on a nonexistent path or a stale (mismatched) hash so a recovery proposal
        can never claim provenance from a diagnosis artifact that does not, in fact, still exist
        with that exact content. Optional: absent for a plan proposed without a typed diagnosis
        bridge (e.g. a historical/manual plan).
        """
        binding = plan.get("diagnosis_binding")
        if binding is None:
            return
        if not isinstance(binding, dict):
            raise ValueError("recovery plan diagnosis_binding must be an object")

        def _verify(raw_path, sha256, label):
            if (not isinstance(raw_path, str) or not raw_path.strip() or
                    not isinstance(sha256, str) or not sha256.strip()):
                raise ValueError(f"recovery plan {label} requires a path and sha256")
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.run_dir / path
            if not path.is_file() or sha256_file(path) != sha256:
                raise ValueError(f"recovery plan {label} artifact is missing or hash-mismatched: {raw_path}")

        _verify(binding.get("diagnosis_artifact_path"), binding.get("diagnosis_artifact_sha256"),
               "diagnosis_binding")
        triggering_evidence = binding.get("triggering_evidence", [])
        if not isinstance(triggering_evidence, list):
            raise ValueError("recovery plan diagnosis_binding.triggering_evidence must be a list")
        for item in triggering_evidence:
            if not isinstance(item, dict):
                raise ValueError("diagnosis_binding.triggering_evidence entries must be objects")
            _verify(item.get("path"), item.get("sha256"), "diagnosis_binding.triggering_evidence")

    def _validate_protected_reference_roles(self, plan):
        """Fail closed if a plan routes a protected-reference artifact role into a
        training/acquisition input or output role, unless a separate, explicit,
        human-approved protected_reference_reuse_authorization is attached to THIS plan.

        protected_reference_roles is a run-declared, free-form set of role names (see
        RunController.initialize) -- this method never hardcodes any chemistry- or
        campaign-specific role name.

        Also scans each of the plan's own `proposed_changes` for a generic, optional
        `artifact_roles` list (e.g. runtimes.pydantic_ai.acquisition_targeting.
        AcquisitionTargetProposal/DataRepairProposal's own artifact-role declarations) -- a
        protected-reference role named only inside one typed proposed change, never lifted into
        the plan's own top-level required_input_artifact_roles/expected_output_artifact_roles,
        must fail closed exactly the same way. This never assumes any particular proposed_change
        shape/kind -- any change dict that happens to declare `artifact_roles` is checked.
        """
        for label in ("required_input_artifact_roles", "expected_output_artifact_roles"):
            roles = plan.get(label, [])
            if (not isinstance(roles, list) or
                    any(not isinstance(role, str) or not role.strip() for role in roles)):
                raise ValueError(f"recovery plan {label} must be a list of non-empty strings")
        protected = set(self.state.get("protected_reference_roles", []))
        if not protected:
            return
        proposed_change_roles = set()
        for change in plan.get("proposed_changes", []) or []:
            if isinstance(change, dict):
                proposed_change_roles.update(
                    role for role in change.get("artifact_roles", []) or []
                    if isinstance(role, str))
        touched = protected & (set(plan.get("required_input_artifact_roles", [])) |
                               set(plan.get("expected_output_artifact_roles", [])) |
                               proposed_change_roles)
        if not touched:
            return
        override = plan.get("protected_reference_reuse_authorization")
        valid_override = (
            isinstance(override, dict) and
            isinstance(override.get("authorized_by"), str) and override["authorized_by"].strip() and
            isinstance(override.get("rationale"), str) and override["rationale"].strip()
        )
        if not valid_override:
            raise ValueError(
                "recovery plan declares protected-reference artifact role(s) as a "
                f"training/acquisition input or output ({sorted(touched)}) without a "
                "separate, explicit, human-approved protected_reference_reuse_authorization "
                "{authorized_by, rationale}"
            )

    def _recovery_signature(self, pending, category, plan):
        """Deterministic stagnation fingerprint binding trigger evidence + diagnosis + return
        stage + corrective-action semantics, so repetition under materially unchanged evidence
        is detectable (see recovery_policy.max_repeated_signature)."""
        payload = {
            "failed_stage": pending["failed_stage"],
            "artifact_sha256": pending["artifact_sha256"],
            "failure_category": category,
            "return_stage": plan["return_stage"],
            "corrective_action_types": sorted({item["type"] for item in plan["proposed_changes"]}),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _enforce_recovery_policy(self, plan, category, signature):
        """Enforce OPTIONAL, explicitly-authored loop-safety limits. Absent recovery_policy
        (or an absent individual limit within it) means that limit is unenforced -- no default
        retry count or budget is ever invented here."""
        policy = self.state.get("recovery_policy")
        if not policy:
            return
        max_attempts = policy.get("max_recovery_attempts")
        if max_attempts is not None:
            attempt_number = len(self.state.get("recoveries", [])) + 1
            if attempt_number > max_attempts:
                raise ValueError(
                    f"recovery attempt {attempt_number} would exceed "
                    f"recovery_policy.max_recovery_attempts ({max_attempts})"
                )
        allowed_types = policy.get("allowed_action_types")
        if allowed_types is not None:
            disallowed = sorted({item["type"] for item in plan["proposed_changes"]} - set(allowed_types))
            if disallowed:
                raise ValueError(
                    "recovery proposed_changes use action types not permitted by "
                    f"recovery_policy.allowed_action_types: {disallowed}"
                )
        cumulative_budget = policy.get("cumulative_budget")
        if cumulative_budget is not None:
            totals = {}
            for prior in self.state.get("recoveries", []):
                for key, value in (prior.get("plan", {}).get("estimated_cost") or {}).items():
                    if isinstance(value, (int, float)):
                        totals[key] = totals.get(key, 0) + value
            for key, value in (plan.get("estimated_cost") or {}).items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
            over_budget = {key: (value, cumulative_budget[key]) for key, value in totals.items()
                          if key in cumulative_budget and value > cumulative_budget[key]}
            if over_budget:
                raise ValueError(
                    "recovery cumulative estimated_cost exceeds "
                    f"recovery_policy.cumulative_budget: {over_budget}"
                )
        max_repeats = policy.get("max_repeated_signature")
        if max_repeats is not None:
            prior_repeats = sum(1 for item in self.state.get("recoveries", [])
                                if item.get("recovery_signature") == signature)
            if prior_repeats >= max_repeats:
                override_ok = (
                    plan.get("escalation_acknowledged") is True and
                    isinstance(plan.get("escalation_rationale"), str) and
                    plan["escalation_rationale"].strip()
                )
                if not override_ok:
                    raise ValueError(
                        f"recovery_signature repeats {prior_repeats} prior proposal(s) under "
                        "materially unchanged evidence, diagnosis, return stage, and corrective "
                        "action -- this meets or exceeds recovery_policy.max_repeated_signature "
                        f"({max_repeats}); set plan.escalation_acknowledged=true with a non-empty "
                        "plan.escalation_rationale to proceed anyway"
                    )

    def propose_recovery(self, plan_path, *, proposer=None):
        """Bind a scientific recovery proposal to the failed gate and its evidence.

        This is the sole authoritative deterministic validator for a RecoveryPlan: an
        agent-facing typed bridge (see runtimes.pydantic_ai.recovery_bridge) may PRODUCE a
        candidate plan file, but only this method ever binds it to the pending gate, and doing
        so never implies approval or execution -- see approve_recovery/start_iteration.

        Trust boundary for the recorded proposer identity: ``proposer``, when supplied, must be
        a TRUSTED caller/runtime identity (e.g. an orchestrator bridge's own Pydantic
        ``Literal``-typed ``requested_by_role``, never something an LLM-authored plan payload
        can forge) and is always authoritative over ``plan["proposed_by"]``. A plan payload
        field is never itself trusted as an authority claim once a trusted ``proposer`` is
        given -- it may only agree with it; any mismatch in either actor_kind or canonical_id
        fails closed rather than silently preferring one or the other. This is how an
        agent-facing path is prevented from having its plan payload assert a human (or any
        other) identity it does not actually hold. Only when ``proposer`` is omitted (the
        historical shape, used by the human-operated CLI and by direct/manual calls) does this
        method fall back to trusting ``plan["proposed_by"]`` outright -- acceptable only because
        that call shape is reserved for genuinely human-operated entry points.
        """
        pending = self.state.get("pending_recovery")
        if not pending or pending.get("status") != "required":
            raise RuntimeError("no REVISE/FAIL gate is waiting for a recovery proposal")
        source = Path(plan_path).resolve()
        plan = json.loads(source.read_text())
        if plan.get("schema_version") != 1:
            raise ValueError("recovery plan requires schema_version=1")
        failed_stage = pending["failed_stage"]
        if plan.get("failed_stage") != failed_stage:
            raise ValueError("recovery plan failed_stage does not match the pending gate")
        category = plan.get("failure_category")
        try:
            resolved_code = recovery_taxonomy.resolve_failure_code(category) if isinstance(category, str) else None
        except KeyError:
            resolved_code = None
        if resolved_code is None:
            raise ValueError(f"recovery plan has invalid failure_category: {category!r}")
        declared_domain = plan.get("failure_domain")
        if declared_domain is not None and declared_domain != resolved_code.domain:
            raise ValueError(
                f"recovery plan failure_domain {declared_domain!r} does not match the "
                f"registered domain for failure_category {category!r} ({resolved_code.domain!r})"
            )
        for field in ("root_cause", "return_stage"):
            if not isinstance(plan.get(field), str) or not plan[field].strip():
                raise ValueError(f"recovery plan requires non-empty {field}")
        # proposed_by is a provenance-bound proposer identity, required on every NEW proposal
        # (a manual/hand-authored plan.json is still a proposal and is not exempt) so
        # approve_recovery/authorize_recovery_capabilities can later enforce that the same
        # canonical actor never both proposes and approves/authorizes the same recovery. When a
        # trusted caller identity (`proposer`) is supplied, it -- not the payload -- is
        # authoritative; see the trust-boundary note in this method's docstring.
        payload_proposed_by = plan.get("proposed_by")
        if proposer is not None:
            trusted = normalize_actor_identity(proposer, field_name="proposer")
            if payload_proposed_by is not None:
                payload_identity = normalize_actor_identity(payload_proposed_by, field_name="proposed_by")
                if (payload_identity.actor_kind != trusted.actor_kind or
                        payload_identity.canonical_id != trusted.canonical_id):
                    raise ValueError(
                        "recovery plan payload's proposed_by conflicts with the trusted "
                        "caller-supplied proposer identity -- an untrusted plan payload field "
                        "can never override or impersonate a trusted caller identity"
                    )
            proposer = trusted
        else:
            proposer = normalize_actor_identity(payload_proposed_by, field_name="proposed_by")
        resolved_agent, resolved_capability = self._resolve_recovery_routing(plan)
        try:
            return_index = self._stage_index(plan["return_stage"])
        except StopIteration as exc:
            raise ValueError(f"recovery return_stage is unknown: {plan['return_stage']}") from exc
        if return_index > self._stage_index(failed_stage):
            raise ValueError("recovery return_stage cannot be downstream of the failed stage")
        changes = plan.get("proposed_changes")
        if (not isinstance(changes, list) or not changes or
                any(not isinstance(item, dict) or not isinstance(item.get("type"), str) or
                    not item["type"].strip() for item in changes)):
            raise ValueError("recovery plan requires proposed_changes with non-empty type")
        labeling = plan.get("labeling")
        if (not isinstance(labeling, dict) or
                any(not isinstance(labeling.get(key), bool)
                    for key in ("teacher_relabel", "new_dft"))):
            raise ValueError("recovery labeling requires boolean teacher_relabel and new_dft")
        training = plan.get("student_training")
        if (not isinstance(training, dict) or not isinstance(training.get("retrain"), bool) or
                not isinstance(training.get("mode"), str) or not training["mode"].strip()):
            raise ValueError("recovery student_training requires retrain and mode")
        if training["retrain"] == (training["mode"] == "none"):
            raise ValueError("recovery student_training retrain and mode are inconsistent")
        revalidation = plan.get("revalidation")
        if (not isinstance(revalidation, dict) or
                not isinstance(revalidation.get("reuse_profile"), bool) or
                not isinstance(revalidation.get("targets"), list) or
                not revalidation["targets"] or
                any(not isinstance(item, str) or not item.strip()
                    for item in revalidation["targets"])):
            raise ValueError("recovery revalidation requires reuse_profile and non-empty targets")
        if "estimated_cost" not in plan or not isinstance(plan["estimated_cost"], dict):
            raise ValueError("recovery estimated_cost must be an object")
        self._validate_diagnosis_binding(plan)
        self._validate_protected_reference_roles(plan)
        signature = self._recovery_signature(pending, category, plan)
        self._enforce_recovery_policy(plan, category, signature)

        recovery_id = len(self.state.setdefault("recoveries", [])) + 1
        recovery_dir = self.run_dir / "recovery"
        recovery_dir.mkdir(exist_ok=True)
        destination = recovery_dir / f"recovery-{recovery_id:03d}.json"
        record = {
            "id": recovery_id, "iteration": self._current_iteration()["id"],
            "status": "proposed", "proposed_at": now(), "source": str(source),
            "failed_stage": failed_stage, "verdict": pending["verdict"],
            "gate_binding": {
                "recorded_at": pending["gate_recorded_at"],
                "artifact_sha256": pending["artifact_sha256"],
                "votes_integrity": pending.get("votes_integrity"),
            },
            "plan": plan, "human_approval": None,
            "proposed_by": proposer.as_dict(),
            "failure_domain": resolved_code.domain,
            "resolved_responsible_agent": resolved_agent,
            "resolved_responsible_capability": resolved_capability,
            "recovery_signature": signature,
            "authorization_envelopes": [],
        }
        destination.write_text(json.dumps(record, indent=2) + "\n")
        record["path"] = str(destination)
        record["integrity"] = artifact_digest(destination)
        self.state["recoveries"].append(record)
        self.state["pending_recovery"] = {"status": "proposed", "recovery_id": recovery_id}
        self.state["events"].append({"at": now(), "type": "recovery_proposed",
                                     "recovery_id": recovery_id, "path": str(destination),
                                     "integrity": record["integrity"]})
        self.save()
        return record

    def _pending_recovery_record(self, expected_status):
        pending = self.state.get("pending_recovery")
        if not pending or pending.get("status") != expected_status:
            raise RuntimeError(f"no recovery is waiting in {expected_status!r} state")
        matches = [item for item in self.state.get("recoveries", [])
                   if item.get("id") == pending.get("recovery_id")]
        if len(matches) != 1:
            raise RuntimeError("pending recovery record is missing or ambiguous")
        try:
            verify_artifact(matches[0]["path"], matches[0]["integrity"])
        except (KeyError, FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError("pending recovery proposal integrity check failed") from exc
        return matches[0]

    def approve_recovery(self, approved_by, note=None):
        """Record explicit human approval, enforcing separation of proposal and approval
        authority.

        approved_by accepts a bare human display-name string (the historical/manual shape,
        preserved for backward compatibility -- see workflow.actor_identity) or a structured
        {actor_kind, canonical_id} identity. Two invariants are enforced fail-closed, never
        implicitly satisfied: (1) the resolved actor_kind must be "human" -- an automated
        Agent/System actor can never satisfy this approval regardless of what string it
        supplies; (2) if this recovery's proposal recorded a proposer identity, the approver's
        canonical_id must differ from it -- the same canonical actor may not both propose and
        approve the same recovery. A historical recovery record with no recorded proposer
        identity (pre-dates this feature) skips only check (2); check (1) still applies to
        every approval unconditionally.
        """
        recovery = self._pending_recovery_record("proposed")
        approver = normalize_actor_identity(approved_by, field_name="approved_by")
        if approver.actor_kind != "human":
            raise ValueError(
                "recovery approval requires an authorized human approval actor; got "
                f"actor_kind={approver.actor_kind!r} -- an automated Agent/System actor can "
                "never satisfy the human-approval requirement"
            )
        proposed_by = recovery.get("proposed_by")
        if proposed_by is not None:
            proposer = normalize_actor_identity(proposed_by, field_name="proposed_by")
            if same_actor(approver, proposer):
                raise ValueError(
                    "recovery approval actor matches this recovery's recorded proposer "
                    f"identity (canonical_id={approver.canonical_id!r}) -- the same actor "
                    "cannot both propose and approve a recovery"
                )
        approval = {"approved_at": now(), "approved_by": approver.as_dict(),
                    "note": note or ""}
        recovery.update(status="approved", human_approval=approval)
        self.state["pending_recovery"] = {"status": "approved",
                                          "recovery_id": recovery["id"]}
        self.state["events"].append({"at": now(), "type": "recovery_approved",
                                     "recovery_id": recovery["id"], **approval})
        self.save()
        return recovery

    def start_iteration(self):
        """Activate an approved recovery and invalidate from its declared return stage."""
        recovery = self._pending_recovery_record("approved")
        old_iteration = self._current_iteration()
        old_iteration.update(status="superseded", completed_at=now())
        return_stage = recovery["plan"]["return_stage"]
        return_index = self._stage_index(return_stage)
        baseline_artifacts = [dict(record) for record in self.state["artifacts"]
                              if self._stage_index(record["stage"]) >= return_index]
        self.invalidate_from(return_stage, include_stage=True)
        new_iteration = old_iteration["id"] + 1
        self.state["iterations"].append({
            "id": new_iteration, "parent_iteration": old_iteration["id"],
            "status": "active", "started_at": now(),
            "trigger": {"recovery_id": recovery["id"],
                        "failed_stage": recovery["failed_stage"],
                        "return_stage": return_stage},
            "baseline_artifacts": baseline_artifacts,
            "recovery_execution": {"status": "required"},
        })
        recovery.update(status="activated", activated_at=now(),
                        new_iteration=new_iteration)
        self.state["pending_recovery"] = None
        self.state["events"].append({"at": now(), "type": "iteration_started",
                                     "iteration": new_iteration,
                                     "recovery_id": recovery["id"],
                                     "return_stage": return_stage})
        self.save()
        return recovery

    def verify_recovery_execution(self, report_path):
        """Verify that an approved recovery produced changed, registered artifacts."""
        iteration = self._current_iteration()
        trigger = iteration.get("trigger")
        if not trigger:
            raise RuntimeError("the current iteration was not started by a recovery")
        if "baseline_artifacts" not in iteration:
            raise RuntimeError(
                "this recovery iteration has no artifact baseline; start a new recovery iteration"
            )
        execution = iteration.get("recovery_execution", {})
        if execution.get("status") != "required":
            raise RuntimeError("the current recovery execution is not waiting for verification")
        matches = [item for item in self.state.get("recoveries", [])
                   if item.get("id") == trigger.get("recovery_id")]
        if len(matches) != 1 or matches[0].get("status") != "activated":
            raise RuntimeError("the activated recovery record is missing or ambiguous")
        recovery = matches[0]
        source = Path(report_path).resolve()
        report = json.loads(source.read_text())
        if report.get("schema_version") != 1:
            raise ValueError("recovery execution report requires schema_version=1")
        if report.get("recovery_id") != recovery["id"]:
            raise ValueError("recovery execution report has the wrong recovery_id")
        if (report.get("previous_iteration") != iteration["parent_iteration"] or
                report.get("current_iteration") != iteration["id"]):
            raise ValueError("recovery execution report has the wrong iteration binding")

        planned_changes = recovery["plan"]["proposed_changes"]
        applied_changes = report.get("changes")
        if not isinstance(applied_changes, list) or len(applied_changes) != len(planned_changes):
            raise ValueError("recovery execution must report every proposed change exactly once")
        baseline = iteration.get("baseline_artifacts", [])

        def validate_stage(stage_name):
            if not isinstance(stage_name, str) or not stage_name.strip():
                raise ValueError("recovery execution evidence stage must be non-empty")
            stage = self.stage(stage_name)
            if self._stage_index(stage_name) < self._stage_index(trigger["return_stage"]):
                raise ValueError(
                    f"recovery execution stage precedes the approved return stage: {stage_name}"
                )
            if stage["status"] != "completed":
                raise ValueError(f"recovery execution stage is not completed: {stage_name}")
            current = self.verify_stage_artifacts(stage_name)
            previous = [item for item in baseline if item["stage"] == stage_name]
            if previous:
                old_hashes = {item["sha256"] for item in previous}
                new_hashes = {item["sha256"] for item in current}
                if old_hashes == new_hashes:
                    raise ValueError(
                        f"recovery execution did not change artifacts for stage: {stage_name}"
                    )
            return stage_name

        def validate_changed_artifact(raw_path):
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("recovery execution evidence artifact must be non-empty")
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.run_dir / path
            path = path.resolve()
            current = [item for item in self.state["artifacts"]
                       if Path(item["path"]).resolve() == path]
            if len(current) != 1:
                raise ValueError(f"recovery execution artifact is not registered: {path}")
            validate_stage(current[0]["stage"])
            previous = [item for item in baseline
                        if Path(item["path"]).resolve() == path]
            if previous and previous[0]["sha256"] == current[0]["sha256"]:
                raise ValueError(f"recovery execution artifact did not change: {path}")
            return current[0]["stage"]

        change_types = []
        evidence_stages = set()
        for planned, applied in zip(planned_changes, applied_changes):
            if not isinstance(applied, dict) or applied.get("type") != planned["type"]:
                raise ValueError("recovery execution change order/type differs from the approved plan")
            if applied.get("status") != "APPLIED":
                raise ValueError("every recovery execution change must have status APPLIED")
            artifacts = applied.get("evidence_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("every applied recovery change requires evidence_artifacts")
            evidence_stages.update(validate_changed_artifact(path) for path in artifacts)
            change_types.append(applied["type"])

        labeling = recovery["plan"]["labeling"]
        label_report = report.get("labeling")
        if not isinstance(label_report, dict):
            raise ValueError("recovery execution requires labeling")
        for flag, stage_field in (("teacher_relabel", "teacher_relabel_stage"),
                                  ("new_dft", "new_dft_stage")):
            if label_report.get(flag) != labeling[flag]:
                raise ValueError(f"recovery execution labeling.{flag} differs from the plan")
            stage_name = label_report.get(stage_field)
            if labeling[flag]:
                evidence_stages.add(validate_stage(stage_name))
            elif stage_name is not None:
                raise ValueError(f"recovery execution labeling.{stage_field} must be null")

        training = recovery["plan"]["student_training"]
        training_report = report.get("student_training")
        if not isinstance(training_report, dict):
            raise ValueError("recovery execution requires student_training")
        for field in ("retrain", "mode"):
            if training_report.get(field) != training[field]:
                raise ValueError(f"recovery execution student_training.{field} differs from the plan")
        training_stage = training_report.get("stage")
        if training["retrain"]:
            evidence_stages.add(validate_stage(training_stage))
        elif training_stage is not None:
            raise ValueError("recovery execution student_training.stage must be null")

        revalidation = recovery["plan"]["revalidation"]
        revalidation_report = report.get("revalidation")
        if not isinstance(revalidation_report, dict):
            raise ValueError("recovery execution requires revalidation")
        if revalidation_report.get("targets") != revalidation["targets"]:
            raise ValueError("recovery execution revalidation targets differ from the plan")
        stages = revalidation_report.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("recovery execution revalidation requires evidence stages")
        evidence_stages.update(validate_stage(name) for name in stages)

        destination = self.run_dir / "recovery" / f"recovery-{recovery['id']:03d}.execution.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        record = {"status": "verified", "verified_at": now(), "path": str(destination),
                  "integrity": artifact_digest(destination),
                  "change_types": change_types, "evidence_stages": sorted(evidence_stages)}
        iteration["recovery_execution"] = record
        recovery["execution"] = record
        self.state["events"].append({"at": now(), "type": "recovery_execution_verified",
                                     "recovery_id": recovery["id"], **record})
        self.save()
        return record

    def _require_verified_recovery_for_pass(self, stage_name):
        iteration = self._current_iteration()
        trigger = iteration.get("trigger")
        if trigger and trigger.get("failed_stage") == stage_name:
            if iteration.get("recovery_execution", {}).get("status") != "verified":
                raise RuntimeError(
                    "the recovered stage cannot PASS until recovery execution is verified"
                )

    def _activated_recovery_for_current_iteration(self):
        iteration = self._current_iteration()
        trigger = iteration.get("trigger")
        if not trigger:
            return None, None
        matches = [item for item in self.state.get("recoveries", [])
                  if item.get("id") == trigger.get("recovery_id")]
        if len(matches) != 1 or matches[0].get("status") != "activated":
            return None, iteration
        return matches[0], iteration

    def authorize_recovery_capabilities(self, authorized_by, *, capabilities=None,
                                        action_types=None, resource_limits=None,
                                        parameter_envelope=None, permitted_artifact_roles=None,
                                        note=""):
        """Create a hash-bound RecoveryAuthorizationEnvelope for the CURRENT activated recovery
        iteration's costly child actions.

        This is a human-approval-created record stating exactly which capabilities/action
        types/resource limits/parameter envelope/artifact roles are authorized for that
        iteration. Approving the recovery itself (approve_recovery) never authorizes a costly
        child action by itself -- this is a SEPARATE, explicit call. dispatch.py's
        authorize_and_execute() consults verify_recovery_authorization as an ADDITIONAL,
        narrower pre-check; it never replaces or weakens the run's normal per-action
        APPROVAL_GATED_ACTIONS requirement, which still applies unchanged whenever no envelope
        (or no matching envelope) covers a given child action.
        """
        recovery, iteration = self._activated_recovery_for_current_iteration()
        if recovery is None:
            raise RuntimeError("no activated recovery iteration is available to authorize")
        authorizer = normalize_actor_identity(authorized_by, field_name="authorized_by")
        if authorizer.actor_kind != "human":
            raise ValueError(
                "recovery authorization requires an authorized human actor; got "
                f"actor_kind={authorizer.actor_kind!r} -- an automated Agent/System actor can "
                "never issue a RecoveryAuthorizationEnvelope"
            )
        proposed_by = recovery.get("proposed_by")
        if proposed_by is not None:
            proposer = normalize_actor_identity(proposed_by, field_name="proposed_by")
            if same_actor(authorizer, proposer):
                raise ValueError(
                    "recovery authorization actor matches this recovery's recorded proposer "
                    f"identity (canonical_id={authorizer.canonical_id!r}) -- a "
                    "RecoveryAuthorizationEnvelope can never be self-issued by the proposing "
                    "actor"
                )
        permitted_roles = sorted(set(permitted_artifact_roles or []))
        protected = set(self.state.get("protected_reference_roles", []))
        conflict = protected & set(permitted_roles)
        if conflict:
            raise ValueError(
                "recovery authorization envelope cannot permit protected-reference artifact "
                f"role(s): {sorted(conflict)}"
            )
        envelope = {
            "authorized_at": now(), "authorized_by": authorizer.as_dict(), "note": note or "",
            "capabilities": sorted(set(capabilities or [])),
            "action_types": sorted(set(action_types or [])),
            "resource_limits": dict(resource_limits or {}),
            "parameter_envelope": dict(parameter_envelope or {}),
            "permitted_artifact_roles": permitted_roles,
            "recovery_id": recovery["id"], "iteration": iteration["id"],
        }
        envelope["envelope_sha256"] = hashlib.sha256(
            json.dumps(envelope, sort_keys=True).encode()
        ).hexdigest()
        recovery.setdefault("authorization_envelopes", []).append(envelope)
        self.state["events"].append({
            "at": now(), "type": "recovery_authorization_envelope_granted",
            "recovery_id": recovery["id"], "envelope_sha256": envelope["envelope_sha256"],
        })
        self.save()
        return envelope

    def verify_recovery_authorization(self, *, action_type, capability=None, artifact_roles=None,
                                      resource_usage=None):
        """Deterministically check whether a CHILD action is covered by the current recovery
        iteration's authorization envelope(s).

        Returns the matching envelope's envelope_sha256, or None if no envelope authorizes it.
        Returning None does NOT mean the action is forbidden -- callers (dispatch.py) MUST fall
        back to the run's normal per-action approval requirement in that case; this method never
        grants an approval by itself and never runs outside an active recovery iteration.
        """
        recovery, _ = self._activated_recovery_for_current_iteration()
        if recovery is None:
            return None
        roles = set(artifact_roles or [])
        usage = resource_usage or {}
        for envelope in recovery.get("authorization_envelopes", []):
            if envelope["action_types"] and action_type not in envelope["action_types"]:
                continue
            if (capability is not None and envelope["capabilities"] and
                    capability not in envelope["capabilities"]):
                continue
            if (envelope["permitted_artifact_roles"] and
                    not roles.issubset(set(envelope["permitted_artifact_roles"]))):
                continue
            limits = envelope.get("resource_limits") or {}
            if any(key in limits and isinstance(value, (int, float)) and
                  isinstance(limits[key], (int, float)) and value > limits[key]
                  for key, value in usage.items()):
                continue
            return envelope["envelope_sha256"]
        return None

    # --- v10: autonomous Teacher-validation planning -----------------------------------------

    def _teacher_validation_objectives(self):
        """Best-effort read of this run's OPTIONAL ``teacher_validation_objectives`` declaration
        from its own frozen ``validation_profile`` source (established via
        ``validation_contract_sources`` at initialize, or via ``establish_validation_contract``'s
        ``source_files``). Returns ``[]`` whenever this run has no validation contract, or its
        validation_profile source is unavailable/unparseable -- objectives are always optional
        (see ``workflow.contracts.parse_teacher_validation_objectives``); this never hard-fails a
        plan commit on a merely-missing or legacy (non-run-bound) validation_profile source.
        """
        contract = self.state.get("validation_contract")
        if not contract:
            return []
        entry = (contract.get("source_files") or {}).get("validation_profile")
        if not isinstance(entry, dict):
            return []
        raw_path = entry.get("snapshot") or entry.get("path")
        if not raw_path or not Path(raw_path).is_file():
            return []
        try:
            profile_cfg = yaml.safe_load(Path(raw_path).read_text())
            return parse_teacher_validation_objectives(profile_cfg)
        except (ValueError, yaml.YAMLError, OSError):
            return []

    def commit_teacher_validation_plan(self, plan_path, *, proposer=None):
        """Write-once commit of a TeacherValidationPlanDraft (see
        ``runtimes.pydantic_ai.teacher_validation_plan.TeacherValidationPlanDraft``) to this run.

        Sole authoritative validator: independently RE-RUNS
        ``validation.teacher_evidence_profile.inspect_teacher_evidence`` against this run's own
        frozen ``teacher_evidence_sources`` (never trusts the draft's own embedded
        ``evidence_profile``/``admissible_components``, which could be stale, hand-edited, or
        produced by a differently-configured planner) and re-derives the admissible decision
        space from that FRESH profile before accepting ``selected_components`` as a genuine,
        evidence-backed subset of it -- an unsupported claim is rejected unconditionally, never
        overridable by human approval (see ``authorize_downstream_teacher_reliance`` for the
        distinct, narrower approval this framework DOES support: approving costly downstream
        reliance on a plan that is itself valid but lacks predictive-fidelity evidence).

        Write-once: an identical re-commit (same canonical plan content) is an idempotent no-op;
        any differing content is a hard failure -- a genuine change in evidence or selection
        requires a new run, mirroring ``establish_validation_contract``'s write-once policy.

        ``proposer`` trust boundary is identical to ``propose_recovery``'s: a trusted caller
        identity (e.g. the orchestrator bridge's own ``requested_by_role``) is always
        authoritative over the draft's own ``proposed_by`` field; a mismatch fails closed. When
        ``proposer`` is omitted, the draft's own ``proposed_by`` is trusted outright (the manual/
        human-operated call shape).
        """
        from dataclasses import asdict

        from validation.teacher_evidence_profile import (
            derive_admissible_decision_space, inspect_teacher_evidence,
        )

        sources = self.state.get("teacher_evidence_sources")
        if sources is None:
            raise RuntimeError(
                "this run did not declare teacher_evidence_sources at initialization -- there "
                "is no frozen evidence this controller can independently re-derive a Teacher "
                "validation plan against"
            )
        source = Path(plan_path).resolve()
        draft = json.loads(source.read_text())
        if draft.get("schema_version") != 1:
            raise ValueError("teacher validation plan requires schema_version=1")
        if draft.get("run_id") != self.state["run_id"]:
            raise ValueError("teacher validation plan run_id does not match this run")

        profile, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
        if draft.get("evidence_profile_sha256") != evidence_profile_sha256:
            raise ValueError(
                "teacher validation plan's evidence_profile_sha256 does not match this run's "
                "independently re-derived evidence profile -- refusing to commit a plan bound "
                "to stale or different evidence"
            )
        decision_space = derive_admissible_decision_space(profile)
        admissible = set(decision_space["admissible_components"])
        selected = draft.get("selected_components")
        if not isinstance(selected, list) or not selected:
            raise ValueError("teacher validation plan requires a non-empty selected_components list")
        unsupported = sorted(set(selected) - admissible)
        if unsupported:
            raise ValueError(
                "teacher validation plan selects component(s) not admissible under this run's "
                f"independently re-derived evidence: {unsupported} -- admissible: "
                f"{sorted(admissible)}"
            )
        objectives = self._teacher_validation_objectives()
        if "require_predictive_fidelity_when_evidence_supports_it" in objectives:
            fidelity = {"ORIGINAL_HELDOUT_FIDELITY", "INDEPENDENT_REFERENCE_FIDELITY"}
            if (fidelity & admissible) and not (fidelity & set(selected)):
                raise ValueError(
                    "validation_profile objective 'require_predictive_fidelity_when_evidence_"
                    "supports_it' requires selecting ORIGINAL_HELDOUT_FIDELITY or "
                    "INDEPENDENT_REFERENCE_FIDELITY -- the independently re-derived evidence "
                    "admits at least one of them but the plan selected neither"
                )
        if "assess_deployment_applicability_when_domain_evidence_exists" in objectives:
            if "DEPLOYMENT_APPLICABILITY" in admissible and "DEPLOYMENT_APPLICABILITY" not in selected:
                raise ValueError(
                    "validation_profile objective 'assess_deployment_applicability_when_domain_"
                    "evidence_exists' requires selecting DEPLOYMENT_APPLICABILITY -- the "
                    "independently re-derived evidence admits it but the plan did not select it"
                )

        payload_proposed_by = draft.get("proposed_by")
        if proposer is not None:
            trusted = normalize_actor_identity(proposer, field_name="proposer")
            if payload_proposed_by is not None:
                payload_identity = normalize_actor_identity(payload_proposed_by,
                                                             field_name="proposed_by")
                if (payload_identity.actor_kind != trusted.actor_kind or
                        payload_identity.canonical_id != trusted.canonical_id):
                    raise ValueError(
                        "teacher validation plan payload's proposed_by conflicts with the "
                        "trusted caller-supplied proposer identity -- an untrusted plan payload "
                        "field can never override or impersonate a trusted caller identity"
                    )
            proposer = trusted
        else:
            proposer = normalize_actor_identity(payload_proposed_by, field_name="proposed_by")

        # target_split for ORIGINAL_HELDOUT_FIDELITY is resolved HERE, from this run's own
        # independently re-derived evidence profile, never trusted from the draft/proposal: the
        # planner selects the COMPONENT (an evidence-driven decision); which literal split NAME
        # backs it is a provenance fact this Controller looks up only after that selection is
        # validated (see TeacherEvidenceProfile.resolved_heldout_split / SPLIT_ROLES). A draft-
        # supplied target_split is never used for a committed ORIGINAL_HELDOUT_FIDELITY plan, even
        # if present, so no proposer can steer which split gets bound.
        resolved_target_split = draft.get("target_split")
        if "ORIGINAL_HELDOUT_FIDELITY" in selected:
            resolved_target_split = profile.resolved_heldout_split
            if not resolved_target_split:
                raise RuntimeError(
                    "ORIGINAL_HELDOUT_FIDELITY is admissible and selected but this run's "
                    "independently re-derived evidence profile has no resolved_heldout_split -- "
                    "this should be unreachable, since admissibility itself requires "
                    "genuine_holdout_test_available, which only becomes true alongside a "
                    "resolved split name"
                )

        canonical_content = {
            "run_id": self.state["run_id"],
            "evidence_profile_sha256": evidence_profile_sha256,
            "selected_components": sorted(set(selected)),
            "reference_kind": draft.get("reference_kind"),
            "target_split": resolved_target_split,
            "source_dataset_role": draft.get("source_dataset_role"),
            "rationale": draft.get("rationale"),
            "validation_objectives": sorted(objectives),
        }
        content_sha256 = hashlib.sha256(
            json.dumps(canonical_content, sort_keys=True, default=str).encode()
        ).hexdigest()
        existing = self.state.get("teacher_validation_plan")
        if existing is not None:
            if existing.get("content_sha256") == content_sha256:
                return existing
            raise RuntimeError(
                "a Teacher validation plan is already committed for this run with different "
                "content; a genuine change requires a new run, not a mutation of this one"
            )

        record = {
            "schema_version": 1, "run_id": self.state["run_id"], "committed_at": now(),
            "source": str(source), "content_sha256": content_sha256,
            "evidence_profile_sha256": evidence_profile_sha256,
            "evidence_profile": asdict(profile),
            "admissible_components": decision_space["admissible_components"],
            "selected_components": list(selected),
            "components": decision_space["components"],
            "protected_data_restrictions": decision_space["protected_data_restrictions"],
            "approval_conditions": decision_space["approval_conditions"],
            "reference_kind": draft.get("reference_kind"),
            "target_split": resolved_target_split,
            "source_dataset_role": draft.get("source_dataset_role"),
            "rationale": draft.get("rationale"),
            "proposed_by": proposer.as_dict(),
            "validation_objectives": objectives,
            "status": "committed",
            "downstream_reliance_approval": None,
        }
        plan_dir = self.run_dir / "teacher_validation"
        plan_dir.mkdir(exist_ok=True)
        destination = plan_dir / "plan.json"
        destination.write_text(json.dumps(record, indent=2) + "\n")
        record["path"] = str(destination)
        record["integrity"] = artifact_digest(destination)
        self.state["teacher_validation_plan"] = record
        self.state["events"].append({
            "at": now(), "type": "teacher_validation_plan_committed",
            "evidence_profile_sha256": evidence_profile_sha256,
            "selected_components": list(selected), "path": str(destination),
            "integrity": record["integrity"],
        })
        self.save()
        return record

    def authorize_downstream_teacher_reliance(self, authorized_by, *, note=None):
        """Record explicit human approval for COSTLY downstream reliance (Teacher labeling /
        Student training) on a committed Teacher validation plan that does NOT include
        ``ORIGINAL_HELDOUT_FIDELITY`` or ``INDEPENDENT_REFERENCE_FIDELITY`` -- see
        ``validation.teacher_evidence_profile.APPROVAL_CONDITIONS``. This is a DISTINCT approval
        from committing the plan itself: ``commit_teacher_validation_plan`` never accepts an
        evidence-unsupported claim regardless of any human approval, but a plan that is itself
        entirely valid (only weaker components, e.g. just OPERATIONAL_ROBUSTNESS, were admissible
        and selected) may still be knowingly relied upon downstream -- that reliance decision, and
        only that decision, is what this method gates.

        A plan that already includes ORIGINAL_HELDOUT_FIDELITY or INDEPENDENT_REFERENCE_FIDELITY
        needs no such approval (there is no missing-predictive-fidelity condition to authorize);
        calling this method for such a plan is a no-op returning the existing plan record
        unchanged. Requires an authorized human actor exactly like ``approve_recovery`` --
        ``actor_kind`` must resolve to ``"human"``.
        """
        plan = self.state.get("teacher_validation_plan")
        if plan is None:
            raise RuntimeError("no Teacher validation plan is committed for this run yet")
        fidelity = {"ORIGINAL_HELDOUT_FIDELITY", "INDEPENDENT_REFERENCE_FIDELITY"}
        if fidelity & set(plan.get("selected_components", [])):
            return plan
        approver = normalize_actor_identity(authorized_by, field_name="authorized_by")
        if approver.actor_kind != "human":
            raise ValueError(
                "downstream Teacher-reliance authorization requires an authorized human "
                f"actor; got actor_kind={approver.actor_kind!r} -- an automated Agent/System "
                "actor can never satisfy this human-approval requirement"
            )
        approval = {"authorized_at": now(), "authorized_by": approver.as_dict(),
                    "note": note or ""}
        plan["downstream_reliance_approval"] = approval
        destination = Path(plan["path"])
        on_disk = {k: v for k, v in plan.items() if k not in ("path", "integrity")}
        destination.write_text(json.dumps(on_disk, indent=2) + "\n")
        plan["integrity"] = artifact_digest(destination)
        self.state["events"].append({"at": now(), "type": "downstream_teacher_reliance_authorized",
                                     **approval})
        self.save()
        return plan

    def summary(self):
        return [(s["name"], s["status"], s["gate"], s["attempts"]) for s in self.state["stages"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("workflow_config")
    init.add_argument("run_dir")
    run = sub.add_parser("run-stage")
    run.add_argument("run_dir")
    run.add_argument("stage")
    complete = sub.add_parser("complete-stage")
    complete.add_argument("run_dir")
    complete.add_argument("stage")
    complete.add_argument("--artifact", action="append", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("run_dir")
    gate.add_argument("stage")
    gate.add_argument("verdict", nargs="?", choices=["REVISE", "FAIL"])
    gate.add_argument("--evidence")
    gate.add_argument("--votes")
    rebind = sub.add_parser("rebind-inputs")
    rebind.add_argument("run_dir")
    propose_adjudication = sub.add_parser("propose-adjudication")
    propose_adjudication.add_argument("run_dir")
    propose_adjudication.add_argument("proposal")
    decide_adjudication = sub.add_parser("decide-adjudication")
    decide_adjudication.add_argument("run_dir")
    decide_adjudication.add_argument("--decided-by", required=True)
    decide_adjudication.add_argument("--decision", choices=sorted(ADJUDICATION_DECISIONS))
    decide_adjudication.add_argument("--note")
    propose = sub.add_parser("propose-recovery")
    propose.add_argument("run_dir")
    propose.add_argument("plan")
    approve = sub.add_parser("approve-recovery")
    approve.add_argument("run_dir")
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--note")
    iteration = sub.add_parser("start-iteration")
    iteration.add_argument("run_dir")
    authorize = sub.add_parser("authorize-recovery")
    authorize.add_argument("run_dir")
    authorize.add_argument("envelope")
    authorize.add_argument("--authorized-by", required=True)
    authorize.add_argument("--note")
    verify_recovery = sub.add_parser("verify-recovery")
    verify_recovery.add_argument("run_dir")
    verify_recovery.add_argument("report")
    context = sub.add_parser("gate-context")
    context.add_argument("run_dir")
    context.add_argument("stage")
    status = sub.add_parser("status")
    status.add_argument("run_dir")
    args = parser.parse_args()
    if args.action == "init":
        controller = RunController.initialize(args.workflow_config, args.run_dir)
    else:
        controller = RunController(args.run_dir)
    if args.action == "run-stage":
        controller.run_stage(args.stage)
    elif args.action == "complete-stage":
        controller.complete_external_stage(args.stage, args.artifact)
    elif args.action == "gate":
        controller.record_gate(args.stage, args.verdict, args.evidence, args.votes)
    elif args.action == "rebind-inputs":
        controller.rebind_inputs()
    elif args.action == "propose-adjudication":
        controller.propose_adjudication(args.proposal)
    elif args.action == "decide-adjudication":
        controller.decide_adjudication(args.decided_by, args.decision, args.note)
    elif args.action == "propose-recovery":
        controller.propose_recovery(args.plan)
    elif args.action == "approve-recovery":
        controller.approve_recovery(args.approved_by, args.note)
    elif args.action == "start-iteration":
        controller.start_iteration()
    elif args.action == "authorize-recovery":
        envelope_spec = json.loads(Path(args.envelope).read_text())
        controller.authorize_recovery_capabilities(
            args.authorized_by, note=args.note, **envelope_spec)
    elif args.action == "verify-recovery":
        controller.verify_recovery_execution(args.report)
    elif args.action == "gate-context":
        print(json.dumps(controller.gate_context(args.stage), indent=2))
        return
    for row in controller.summary():
        print("\t".join(map(str, row)))
    if args.action == "status" and controller.state.get("pending_recovery"):
        print("RECOVERY\t" + json.dumps(controller.state["pending_recovery"], sort_keys=True))
    if args.action == "status":
        execution = controller._current_iteration().get("recovery_execution")
        if execution:
            print("RECOVERY_EXECUTION\t" + json.dumps(execution, sort_keys=True))


if __name__ == "__main__":
    main()
