"""Autonomous Teacher-validation planning: end-to-end lifecycle tests for the v10 additive,
evidence-driven pipeline (``inspect_teacher_evidence`` -> ``derive_admissible_decision_space`` ->
PydanticAI Orchestrator ``TeacherValidationPlanProposal`` ->
``RunController.commit_teacher_validation_plan`` -> optional
``RunController.authorize_downstream_teacher_reliance`` -> optional stage-level
``RunController.mark_stage_not_applicable``).

Every fixture here uses only generic, boolean evidence facts and generic stage/role names
(``ml-trainer``/``train_committee``/``evaluate_heldout_fidelity``, already-registered production
executors) -- never a material, dataset, or campaign name -- so these tests double as a
demonstration that the mechanism is genuinely evidence-driven, not a SiO2/Allegro-specific branch
(see ``SiO2NoHardcodeProofTests`` below for a source-level proof of the same claim).
"""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------------
# Shared evidence/workflow fixtures
# --------------------------------------------------------------------------------------------

def _frame(local_index, *, category="bulk", split=None):
    a = Atoms("Cu", positions=[[local_index, 0, 0]], cell=[10, 10, 10], pbc=True)
    a.info["structure_id"] = f"s{local_index}"
    a.info["parent_structure_id"] = f"p{local_index}"
    a.info["source_category"] = category
    a.info["source_local_index"] = local_index
    a.info["dft_energy"] = -1.0 * (local_index + 1)
    a.new_array("dft_forces", np.zeros((1, 3)))
    return a


def _write_operational_population(path: Path, n=2) -> None:
    write(str(path), [_frame(i) for i in range(n)])


def _write_training_db_and_manifest(root: Path, *, n_train=3, n_test=1,
                                    heldout_split_name="test", split_roles=None):
    """A fully-labeled training DB whose frames genuinely cross-walk against a split manifest --
    admits OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY, and ORIGINAL_HELDOUT_FIDELITY
    (either via a caller-supplied ``target_split`` override, or -- if ``split_roles`` is given --
    from provenance alone). ``heldout_split_name`` is deliberately overridable so tests can prove
    the mechanism does not assume the literal name "test"; ``split_roles`` is an OPTIONAL generic
    ``{<split name>: training|validation|heldout_evaluation}`` declaration written into the same
    manifest (see ``validation.teacher_evidence_profile.SPLIT_ROLES``)."""
    frames = []
    records = []
    for i in range(n_train):
        frames.append(_frame(i))
        records.append({"source_category": "bulk", "source_local_index": i, "split": "train"})
    for j in range(n_test):
        idx = n_train + j
        frames.append(_frame(idx))
        records.append({"source_category": "bulk", "source_local_index": idx,
                        "split": heldout_split_name})
    db_path = root / "train_db.extxyz"
    write(str(db_path), frames)
    manifest_path = root / "split_manifest.json"
    payload = {"records": records}
    if split_roles is not None:
        payload["split_roles"] = split_roles
    manifest_path.write_text(json.dumps(payload))
    return db_path, manifest_path


def _dummy_teacher_model(root: Path) -> Path:
    path = root / "teacher_model.ckpt"
    path.write_text("not a real checkpoint -- existence is all inspect_teacher_evidence checks")
    return path


def _write_validation_contract_sources(root: Path, *, objectives) -> dict:
    """Minimal, generic three-file ``validation_contract_sources`` bundle (distillation_scope /
    validation_profile / dataset_policy) -- just enough to satisfy
    ``workflow.contracts.build_validation_contract_components`` -- with ``objectives`` written
    into ``validation_profile.yaml``'s OPTIONAL ``teacher_validation_objectives`` key, so a test
    can exercise run-declared objective enforcement without any material-specific content."""
    domain = {"system": "generic-test-domain"}
    scope_path = root / "distillation_scope.yaml"
    scope_path.write_text(yaml.safe_dump({"deployment_domain": domain}, sort_keys=False))
    profile_path = root / "validation_profile.yaml"
    profile_path.write_text(yaml.safe_dump({
        "deployment_domain": domain,
        "checks": [{"name": "generic_check", "category": "structure",
                   "purpose": "student_teacher_fidelity", "reference_source": "teacher",
                   "protocol": "generic test protocol", "required": True, "threshold": None}],
        "teacher_validation_objectives": list(objectives),
    }, sort_keys=False))
    policy_path = root / "dataset_policy.yaml"
    policy_path.write_text(yaml.safe_dump({
        "split_policy": {"train_fraction": 0.8},
    }, sort_keys=False))
    return {"distillation_scope": str(scope_path), "validation_profile": str(profile_path),
           "dataset_policy": str(policy_path)}


def _write_workflow(root: Path, *, teacher_evidence_sources: dict, stage_a_component=None,
                    stage_b_component=None, validation_contract_sources=None) -> Path:
    """Generic two-stage workflow (mirrors tests/test_run_campaign.py's fixture) plus an OPTIONAL
    ``teacher_evidence_sources`` block and an OPTIONAL ``teacher_validation_component`` declared on
    stage_a and/or stage_b -- both new, additive, opt-in workflow-config keys; no stage
    name/count/domain concept is assumed anywhere in the production code that reads them.
    ``validation_contract_sources``, if given (see ``_write_validation_contract_sources``), is
    wired in unchanged so a test can exercise run-declared ``teacher_validation_objectives``."""
    dataset = root / "train.extxyz"
    write(str(dataset), [_frame(i) for i in range(3)])
    student_cfg = root / "student.yaml"
    student_cfg.write_text(
        "kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n"
        "predict:\n  factory: adapters.mock_model.MockCheckpointCalculator\n"
        "  checkpoint_arg: checkpoint\n  kwargs: {}\n")
    stage_a = {
        "name": "stage_a",
        "command": None,
        "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
        "pydantic_ai": {
            "role": "ml-trainer",
            "action": "train_committee",
            "approval_boundary": "costly_training",
            "idempotency_key": "teacher-validation-autonomy-stage-a-001",
            "parameters": {
                "student_config": str(student_cfg),
                "dataset": str(dataset),
                "output_dir": "{artifacts_dir}/committee",
                "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
            },
        },
        "gate": {"criteria": ["committee manifest is complete"]},
    }
    if stage_a_component is not None:
        stage_a["teacher_validation_component"] = stage_a_component
    stage_b = {
        "name": "stage_b",
        "command": None,
        "outputs": ["artifacts/heldout_labeled.extxyz", "artifacts/heldout_report.json"],
        "pydantic_ai": {
            "role": "ml-trainer",
            "action": "evaluate_heldout_fidelity",
            "approval_boundary": "costly_training",
            "idempotency_key": "teacher-validation-autonomy-stage-b-001",
            "parameters": {
                "student_config": str(student_cfg),
                "committee_manifest": "{artifacts_dir}/student_committee.manifest.json",
                "frames_path": str(dataset),
                "labeled_output": "{artifacts_dir}/heldout_labeled.extxyz",
                "report_path": "{artifacts_dir}/heldout_report.json",
            },
        },
        "gate": {"criteria": ["fidelity report is complete"]},
    }
    if stage_b_component is not None:
        stage_b["teacher_validation_component"] = stage_b_component
    cfg = {
        "run_id": "teacher-validation-autonomy",
        "inputs": [str(student_cfg), str(dataset)],
        "teacher_evidence_sources": teacher_evidence_sources,
        **({"validation_contract_sources": validation_contract_sources}
           if validation_contract_sources is not None else {}),
        "stages": [
            stage_a,
            stage_b,
        ],
    }
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return workflow


def _mock_orchestrator_response(root: Path, *, run_id, evidence_profile_sha256,
                                selected_components, rationale="autonomous selection",
                                reference_kind=None, target_split=None,
                                source_dataset_role=None, name="mock_orchestrator_response.json") -> Path:
    path = root / name
    path.write_text(json.dumps({
        "run_id": run_id, "evidence_profile_sha256": evidence_profile_sha256,
        "selected_components": selected_components, "rationale": rationale,
        "reference_kind": reference_kind, "target_split": target_split,
        "source_dataset_role": source_dataset_role,
    }))
    return path


def _hand_crafted_draft(root: Path, *, run_id, evidence_profile_sha256, selected_components,
                        proposed_by=None, name="draft.json") -> Path:
    """A minimal, directly-authored draft JSON matching exactly the fields
    ``RunController.commit_teacher_validation_plan`` actually reads -- used to test the
    controller's OWN independent re-validation without going through the Orchestrator dispatch
    path (i.e. as if a compromised or hand-edited draft file were presented to it)."""
    path = root / name
    path.write_text(json.dumps({
        "schema_version": 1, "run_id": run_id,
        "evidence_profile_sha256": evidence_profile_sha256,
        "selected_components": selected_components,
        "reference_kind": None, "target_split": None, "source_dataset_role": None,
        "rationale": "hand-crafted test draft",
        "proposed_by": proposed_by or {"actor_kind": "human", "canonical_id": "test-operator"},
    }))
    return path


class CaseA_InsufficientEvidencePausesTheCampaign(unittest.TestCase):
    """Case A: a Teacher with a model file but NO operational population, training DB,
    independent reference, or deployment-domain population at all admits NO component --
    the autonomous planner must fail closed (CAMPAIGN_FAILED) rather than fabricate a plan, and
    must never commit one."""

    def test_run_campaign_fails_closed_with_no_plan_committed(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            workflow = _write_workflow(
                root, teacher_evidence_sources={"teacher_model_path": str(teacher_model)})
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
            c = RunController(run_dir)
            self.assertIsNone(c.state["teacher_validation_plan"])
            self.assertEqual(c.stage("stage_a")["status"], "pending",
                             "no stage may dispatch before Teacher-validation planning resolves")


class UMLIPForeignPotentialNoReferencePauseTests(unittest.TestCase):
    """The same generic insufficient-evidence floor, exercised with a distinctly-named "foreign
    universal MLIP" evidence fixture (a model file with no reference data of any kind) -- proving
    the mechanism generalizes to a real-world uMLIP-style Teacher without any material- or
    model-family-specific code path."""

    def test_foreign_universal_potential_with_no_reference_data_pauses(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreign_potential = root / "foreign_universal_potential.ckpt"
            foreign_potential.write_text("stand-in for a foreign/universal MLIP checkpoint")
            workflow = _write_workflow(
                root, teacher_evidence_sources={"teacher_model_path": str(foreign_potential)})
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            code = cli.main(["plan-teacher-validation", "--runtime", "mock",
                             "--run-dir", str(run_dir)])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
            c = RunController(run_dir)
            self.assertIsNone(c.state["teacher_validation_plan"])


class CaseB_SuccessfulAutonomousPlanCommitsAndIsWriteOnce(unittest.TestCase):
    """Case B: evidence admits exactly OPERATIONAL_ROBUSTNESS; an Orchestrator proposal selecting
    it commits successfully, is write-once/idempotent on identical re-submission, and hard-fails
    on a differing re-submission."""

    def _evidence_sources(self, root):
        teacher_model = _dummy_teacher_model(root)
        population = root / "operational_population.extxyz"
        _write_operational_population(population)
        return {"teacher_model_path": str(teacher_model),
                "operational_evaluation_population_path": str(population)}

    def test_plan_committed_with_orchestrator_selected_component(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = self._evidence_sources(root)
            workflow = _write_workflow(root, teacher_evidence_sources=sources)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**c.state["teacher_evidence_sources"])
            mock_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["OPERATIONAL_ROBUSTNESS"])

            code = cli.main(["plan-teacher-validation", "--runtime", "mock",
                             "--run-dir", str(run_dir),
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            c = RunController(run_dir)
            plan = c.state["teacher_validation_plan"]
            self.assertEqual(plan["selected_components"], ["OPERATIONAL_ROBUSTNESS"])
            self.assertEqual(plan["status"], "committed")

            # A second identical plan-teacher-validation invocation is a write-once no-op --
            # commit_teacher_validation_plan itself is never called again because the CLI/state
            # already reports a committed plan.
            code_again = cli.main(["plan-teacher-validation", "--runtime", "mock",
                                   "--run-dir", str(run_dir),
                                   "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code_again, cli.EXIT_SUCCESS)

            # Directly re-committing the identical draft content through the controller is also
            # an idempotent no-op (no duplicate event).
            plan_path = Path(plan["path"])
            before = len([e for e in c.state["events"]
                         if e["type"] == "teacher_validation_plan_committed"])
            c.commit_teacher_validation_plan(plan_path)
            after = len([e for e in c.state["events"]
                        if e["type"] == "teacher_validation_plan_committed"])
            self.assertEqual(before, after)

            # A differing re-submission (same admissible selection, but different canonical
            # content -- here the rationale) is a hard failure, never a silent mutation.
            differing_draft_path = root / "differing_draft.json"
            differing_draft_path.write_text(json.dumps({
                "schema_version": 1, "run_id": c.state["run_id"],
                "evidence_profile_sha256": evidence_profile_sha256,
                "selected_components": ["OPERATIONAL_ROBUSTNESS"],
                "reference_kind": None, "target_split": None, "source_dataset_role": None,
                "rationale": "a genuinely different rationale than the committed plan",
                "proposed_by": {"actor_kind": "human", "canonical_id": "test-operator"},
            }))
            with self.assertRaisesRegex(RuntimeError, "different"):
                c.commit_teacher_validation_plan(differing_draft_path)


class CaseC1_ProposalSelectingUnsupportedComponentIsRejected(unittest.TestCase):
    """Case C1: the Orchestrator's OWN proposal claims a component the evidence does not admit --
    ``validate_teacher_validation_plan_proposal`` rejects it before any draft is ever built or
    committed, and this is unconditional (never overridable)."""

    def test_run_campaign_fails_closed_and_commits_nothing(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            population = root / "operational_population.extxyz"
            _write_operational_population(population)
            sources = {"teacher_model_path": str(teacher_model),
                       "operational_evaluation_population_path": str(population)}
            workflow = _write_workflow(root, teacher_evidence_sources=sources)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            # Only OPERATIONAL_ROBUSTNESS is admissible here -- ORIGINAL_HELDOUT_FIDELITY is not.
            mock_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["ORIGINAL_HELDOUT_FIDELITY"])

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
            c = RunController(run_dir)
            self.assertIsNone(c.state["teacher_validation_plan"])


class CaseC2_CommitIndependentlyRerejectsAHandEditedDraft(unittest.TestCase):
    """Case C2: even bypassing the Orchestrator/proposal-validation path entirely (a hand-crafted
    draft file, as if compromised or manually edited), ``commit_teacher_validation_plan`` itself
    independently re-derives the admissible decision space from this run's OWN frozen evidence and
    rejects an unsupported claim -- proving the fail-closed check does not rely solely on the
    proposal-validation step ever having run."""

    def test_commit_rejects_unsupported_component_regardless_of_draft_content(self):
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            population = root / "operational_population.extxyz"
            _write_operational_population(population)
            sources = {"teacher_model_path": str(teacher_model),
                       "operational_evaluation_population_path": str(population)}
            workflow = _write_workflow(root, teacher_evidence_sources=sources)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)

            malicious_draft = _hand_crafted_draft(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["INDEPENDENT_REFERENCE_FIDELITY"])
            with self.assertRaisesRegex(ValueError, "not admissible"):
                c.commit_teacher_validation_plan(malicious_draft)
            self.assertIsNone(c.state["teacher_validation_plan"])


class CaseD_NotApplicableStageLifecycle(unittest.TestCase):
    """Case D: a stage declaring a ``teacher_validation_component`` the committed plan did NOT
    select is automatically resolved as NOT_APPLICABLE (never dispatched, never blocking the
    campaign) -- categorically distinct from PASS/REVISE/FAIL, and treated identically to PASS by
    upstream gating so the campaign still reaches COMPLETED."""

    def test_stage_marked_not_applicable_and_campaign_completes(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            population = root / "operational_population.extxyz"
            _write_operational_population(population)
            sources = {"teacher_model_path": str(teacher_model),
                       "operational_evaluation_population_path": str(population)}
            # stage_b declares a component the evidence does NOT admit at all (DEPLOYMENT_
            # APPLICABILITY requires a deployment-domain population this run never declares) --
            # so whatever the plan selects, stage_b can never require it. stage_a declares the
            # component the plan actually selects, so the whole-workflow coverage invariant
            # (every selected component must be executable by SOME declared stage capability) is
            # satisfied -- this fixture is about per-stage NOT_APPLICABLE governance, not coverage.
            workflow = _write_workflow(root, teacher_evidence_sources=sources,
                                       stage_a_component="OPERATIONAL_ROBUSTNESS",
                                       stage_b_component="DEPLOYMENT_APPLICABILITY")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            mock_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["OPERATIONAL_ROBUSTNESS"])

            # First launch: Teacher-validation planning commits automatically, then stage_a pauses
            # for its (pre-existing, unrelated) costly_training human-approval boundary.
            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            c = RunController(run_dir)
            self.assertEqual(c.state["teacher_validation_plan"]["selected_components"],
                             ["OPERATIONAL_ROBUSTNESS"])

            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training",
                                       "--note", "test approval"]), cli.EXIT_SUCCESS)

            # The committed plan selected only OPERATIONAL_ROBUSTNESS (no predictive-fidelity
            # component), so the generic costly_training action-approval above is not enough --
            # the distinct, plan-bound downstream-reliance approval is also required before
            # stage_a (a costly_training stage) may dispatch.
            self.assertEqual(cli.main(["authorize-downstream-teacher-reliance",
                                       "--run-dir", str(run_dir),
                                       "--authorized-by", "test-human-approver",
                                       "--note", "accept OPERATIONAL_ROBUSTNESS-only reliance"]),
                             cli.EXIT_SUCCESS)

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            c = RunController(run_dir)
            self.assertEqual(c.stage("stage_a")["gate"], "PASS")
            self.assertEqual(c.stage("stage_b")["status"], "not_applicable")
            self.assertEqual(c.stage("stage_b")["gate"], "NOT_APPLICABLE")
            self.assertIn("stage_b", c.state["stage_applicability"])
            self.assertTrue(any(e["type"] == "stage_marked_not_applicable"
                                for e in c.state["events"]))


class CaseE_DownstreamRelianceApprovalIsASeparateHumanGate(unittest.TestCase):
    """Case E: a committed plan that selected only OPERATIONAL_ROBUSTNESS even though
    ORIGINAL_HELDOUT_FIDELITY was ALSO admissible (no run-declared objective forced it) is itself
    entirely valid -- but costly downstream reliance on it requires a SEPARATE, explicit human
    approval that an automated actor can never satisfy; a plan that DOES include a fidelity
    component needs no such approval at all (no-op)."""

    def _committed_plan_missing_fidelity(self, root):
        from workflow.controller import RunController
        teacher_model = _dummy_teacher_model(root)
        db_path, manifest_path = _write_training_db_and_manifest(root)
        sources = {"teacher_model_path": str(teacher_model),
                   "original_training_db_path": str(db_path),
                   "split_source_manifest_paths": [str(manifest_path)],
                   "target_split": "test"}
        workflow = _write_workflow(root, teacher_evidence_sources=sources)
        run_dir = root / "run"
        RunController.initialize(workflow, run_dir)
        c = RunController(run_dir)
        from validation.teacher_evidence_profile import (
            derive_admissible_decision_space, inspect_teacher_evidence,
        )
        profile, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
        admissible = set(derive_admissible_decision_space(profile)["admissible_components"])
        self.assertIn("ORIGINAL_HELDOUT_FIDELITY", admissible,
                      "fixture must genuinely admit fidelity evidence for this case to be meaningful")
        draft = _hand_crafted_draft(root, run_id=c.state["run_id"],
                                    evidence_profile_sha256=evidence_profile_sha256,
                                    selected_components=["OPERATIONAL_ROBUSTNESS"])
        c.commit_teacher_validation_plan(draft)
        return RunController(run_dir)

    def test_automated_actor_cannot_authorize_downstream_reliance(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._committed_plan_missing_fidelity(Path(tmp))
            with self.assertRaisesRegex(ValueError, "human"):
                c.authorize_downstream_teacher_reliance(
                    {"actor_kind": "agent", "canonical_id": "auto-approver"})
            self.assertIsNone(c.state["teacher_validation_plan"]["downstream_reliance_approval"])

    def test_human_actor_can_authorize_downstream_reliance(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._committed_plan_missing_fidelity(Path(tmp))
            plan = c.authorize_downstream_teacher_reliance("dr-human-approver",
                                                            note="accepted risk")
            self.assertIsNotNone(plan["downstream_reliance_approval"])
            self.assertIsNotNone(
                c.state["teacher_validation_plan"]["downstream_reliance_approval"])

    def test_plan_with_fidelity_component_needs_no_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from workflow.controller import RunController
            from validation.teacher_evidence_profile import inspect_teacher_evidence

            teacher_model = _dummy_teacher_model(root)
            db_path, manifest_path = _write_training_db_and_manifest(root)
            sources = {"teacher_model_path": str(teacher_model),
                       "original_training_db_path": str(db_path),
                       "split_source_manifest_paths": [str(manifest_path)],
                       "target_split": "test"}
            workflow = _write_workflow(root, teacher_evidence_sources=sources)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            draft = _hand_crafted_draft(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["ORIGINAL_HELDOUT_FIDELITY"])
            c.commit_teacher_validation_plan(draft)

            plan = c.authorize_downstream_teacher_reliance(
                {"actor_kind": "agent", "canonical_id": "auto-approver"})
            self.assertIsNone(plan["downstream_reliance_approval"],
                              "a plan that already includes fidelity evidence needs no approval "
                              "at all, so even a non-human caller must get a no-op, not a "
                              "rejection")

    def test_downstream_reliance_gap_is_untouched_by_action_approval_fix(self):
        # R25 forensic fix (parameter-dependent build_teacher_baseline/validate_teacher_reference
        # approval, see runtimes.pydantic_ai.actions.resolve_action_approval_boundary) answers a
        # different question than this gate and must never substitute for it: a plan lacking
        # predictive-fidelity evidence still blocks genuinely costly downstream Teacher labeling
        # and Student training via _teacher_validation_downstream_reliance_gap, independent of
        # whatever action-level approval those actions' own parameters might otherwise satisfy.
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            c = self._committed_plan_missing_fidelity(Path(tmp))
            for stage_name in ("teacher_labeling", "training"):
                gap = cli._teacher_validation_downstream_reliance_gap(c, stage_name, {})
                self.assertIsNotNone(gap, stage_name)
                self.assertIn("committed Teacher validation plan", gap)
            c.authorize_downstream_teacher_reliance("dr-human-approver", note="accepted risk")
            for stage_name in ("teacher_labeling", "training"):
                self.assertIsNone(
                    cli._teacher_validation_downstream_reliance_gap(c, stage_name, {}), stage_name)


class CaseF_ProvenanceOnlyHeldoutRoleNeedsNoTargetSplitInput(unittest.TestCase):
    """Case F (Issue 1): a genuine held-out split is discoverable from provenance
    (``split_roles``) alone -- ``teacher_evidence_sources`` never supplies ``target_split``, the
    Orchestrator's proposal never supplies ``target_split``, and the actual split label is
    deliberately NOT "test" -- yet ORIGINAL_HELDOUT_FIDELITY is admissible, selectable, and
    committable, and the Controller resolves the correct provenance-bound split name into the
    committed plan's ``target_split`` field entirely on its own."""

    def _evidence_sources(self, root):
        teacher_model = _dummy_teacher_model(root)
        db_path, manifest_path = _write_training_db_and_manifest(
            root, heldout_split_name="holdout_eval",
            split_roles={"train": "training", "holdout_eval": "heldout_evaluation"})
        # Deliberately NO "target_split" key anywhere in this run's evidence sources.
        return {"teacher_model_path": str(teacher_model),
                "original_training_db_path": str(db_path),
                "split_source_manifest_paths": [str(manifest_path)]}

    def test_holdout_fidelity_admissible_and_committable_without_target_split(self):
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import (
            derive_admissible_decision_space, inspect_teacher_evidence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = self._evidence_sources(root)
            profile, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            self.assertIsNone(sources.get("target_split"))
            self.assertTrue(profile.genuine_holdout_test_available)
            self.assertEqual(profile.resolved_heldout_split, "holdout_eval")
            admissible = set(derive_admissible_decision_space(profile)["admissible_components"])
            self.assertIn("ORIGINAL_HELDOUT_FIDELITY", admissible)

            workflow = _write_workflow(root, teacher_evidence_sources=sources)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            # The Orchestrator proposal/draft also never supplies target_split -- the planner
            # selects the COMPONENT only.
            draft = _hand_crafted_draft(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["ORIGINAL_HELDOUT_FIDELITY"])
            draft_payload = json.loads(draft.read_text())
            self.assertIsNone(draft_payload["target_split"])

            c.commit_teacher_validation_plan(draft)
            c = RunController(run_dir)
            plan = c.state["teacher_validation_plan"]
            self.assertEqual(plan["selected_components"], ["ORIGINAL_HELDOUT_FIDELITY"])
            # Execution-time resolution: the Controller -- not the draft -- bound the real,
            # provenance-declared, non-"test"-named split.
            self.assertEqual(plan["target_split"], "holdout_eval")

    def test_no_heldout_role_and_no_target_split_leaves_fidelity_inadmissible(self):
        from validation.teacher_evidence_profile import (
            derive_admissible_decision_space, inspect_teacher_evidence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            # split_roles declares only training/validation roles -- no heldout_evaluation
            # anywhere -- and no target_split override either.
            db_path, manifest_path = _write_training_db_and_manifest(
                root, heldout_split_name="val0",
                split_roles={"train": "training", "val0": "validation"})
            sources = {"teacher_model_path": str(teacher_model),
                      "original_training_db_path": str(db_path),
                      "split_source_manifest_paths": [str(manifest_path)]}
            profile, _ = inspect_teacher_evidence(**sources)
            self.assertFalse(profile.genuine_holdout_test_available)
            self.assertIsNone(profile.resolved_heldout_split)
            admissible = set(derive_admissible_decision_space(profile)["admissible_components"])
            self.assertNotIn("ORIGINAL_HELDOUT_FIDELITY", admissible)

    def test_conflicting_heldout_roles_fail_closed_to_inadmissible(self):
        from validation.teacher_evidence_profile import (
            derive_admissible_decision_space, inspect_teacher_evidence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            db_path, m1 = _write_training_db_and_manifest(
                root, heldout_split_name="holdout_eval",
                split_roles={"train": "training", "holdout_eval": "heldout_evaluation"})
            # A second manifest, over the SAME training DB, declares a DIFFERENT split ALSO
            # heldout -- two conflicting held-out candidates must fail closed, never guess.
            m2 = root / "second_manifest.json"
            m2.write_text(json.dumps({
                "records": [{"source_category": "bulk", "source_local_index": i,
                            "split": "train" if i < 3 else "another_holdout"}
                           for i in range(4)],
                "split_roles": {"train": "training", "another_holdout": "heldout_evaluation"},
            }))
            sources = {"teacher_model_path": str(teacher_model),
                      "original_training_db_path": str(db_path),
                      "split_source_manifest_paths": [str(m1), str(m2)]}
            profile, _ = inspect_teacher_evidence(**sources)
            self.assertFalse(profile.genuine_holdout_test_available)
            self.assertIsNone(profile.resolved_heldout_split)
            admissible = set(derive_admissible_decision_space(profile)["admissible_components"])
            self.assertNotIn("ORIGINAL_HELDOUT_FIDELITY", admissible)


class CaseG_StaticStageCapabilityDoesNotPreselectStrategy(unittest.TestCase):
    """Case G (Issue 2): a stage may declare it is STATICALLY CAPABLE of more than one
    validation component without that list preselecting which one a campaign uses -- the
    committed plan's ``selected_components`` (an AGENT DECISION) is what narrows the
    intersection, and a Judge/LLM never self-routes or self-skips at dispatch time."""

    def test_stage_capable_of_either_component_is_applicable_when_plan_selects_either(self):
        from runtimes.pydantic_ai.cli import _teacher_validation_not_applicable_reason

        class _FakeController:
            def __init__(self, selected):
                self.state = {"teacher_validation_plan": {"selected_components": selected}}

        stage_cfg = {"teacher_validation_component":
                     ["ORIGINAL_HELDOUT_FIDELITY", "INDEPENDENT_REFERENCE_FIDELITY"]}

        for selected in (["ORIGINAL_HELDOUT_FIDELITY"], ["INDEPENDENT_REFERENCE_FIDELITY"],
                         ["ORIGINAL_HELDOUT_FIDELITY", "OPERATIONAL_ROBUSTNESS"]):
            reason = _teacher_validation_not_applicable_reason(
                _FakeController(selected), "stage_b", stage_cfg)
            self.assertIsNone(reason, f"expected applicable for selected={selected}")

    def test_stage_capable_of_either_component_is_not_applicable_when_plan_selects_neither(self):
        from runtimes.pydantic_ai.cli import _teacher_validation_not_applicable_reason

        class _FakeController:
            def __init__(self, selected):
                self.state = {"teacher_validation_plan": {"selected_components": selected}}

        stage_cfg = {"teacher_validation_component":
                     ["ORIGINAL_HELDOUT_FIDELITY", "INDEPENDENT_REFERENCE_FIDELITY"]}
        reason = _teacher_validation_not_applicable_reason(
            _FakeController(["OPERATIONAL_ROBUSTNESS"]), "stage_b", stage_cfg)
        self.assertIsNotNone(reason)
        self.assertIn("ORIGINAL_HELDOUT_FIDELITY", reason)
        self.assertIn("INDEPENDENT_REFERENCE_FIDELITY", reason)

    def test_single_string_capability_still_works_unchanged(self):
        from runtimes.pydantic_ai.cli import _teacher_validation_not_applicable_reason

        class _FakeController:
            def __init__(self, selected):
                self.state = {"teacher_validation_plan": {"selected_components": selected}}

        stage_cfg = {"teacher_validation_component": "OPERATIONAL_ROBUSTNESS"}
        self.assertIsNone(_teacher_validation_not_applicable_reason(
            _FakeController(["OPERATIONAL_ROBUSTNESS"]), "stage_b", stage_cfg))
        self.assertIsNotNone(_teacher_validation_not_applicable_reason(
            _FakeController(["ORIGINAL_HELDOUT_FIDELITY"]), "stage_b", stage_cfg))

    def test_list_valued_capability_end_to_end_stage_lifecycle(self):
        # End-to-end: a real committed plan selecting only ONE of stage_b's two capable
        # components leaves stage_b applicable (never not_applicable), proving the workflow
        # author did not have to guess which one the plan would pick.
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            db_path, manifest_path = _write_training_db_and_manifest(root)  # default "test" name
            sources = {"teacher_model_path": str(teacher_model),
                      "original_training_db_path": str(db_path),
                      "split_source_manifest_paths": [str(manifest_path)],
                      "target_split": "test"}
            workflow = _write_workflow(
                root, teacher_evidence_sources=sources,
                stage_b_component=["ORIGINAL_HELDOUT_FIDELITY", "INDEPENDENT_REFERENCE_FIDELITY"])
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            mock_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["ORIGINAL_HELDOUT_FIDELITY"])

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training",
                                       "--note", "test approval"]), cli.EXIT_SUCCESS)

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            c = RunController(run_dir)
            self.assertEqual(c.stage("stage_b")["status"], "completed")
            self.assertNotEqual(c.stage("stage_b")["status"], "not_applicable")


class CaseH_ObjectiveDrivenSemanticCorrectionRetry(unittest.TestCase):
    """Case H: the planner task now explicitly states this run's declared
    ``teacher_validation_objectives`` (not merely admissibility) -- a first proposal that is a
    valid, admissible SUBSET but violates a declared objective is rejected and fed back to the
    planner (not an immediate campaign failure); a corrected second proposal that also satisfies
    the objective is accepted. Mirrors the real regression this fix addresses: OPERATIONAL_
    ROBUSTNESS + TRAINING_CORPUS_CONSISTENCY are both admissible and mutually consistent, but
    ``require_predictive_fidelity_when_evidence_supports_it`` also requires selecting
    ORIGINAL_HELDOUT_FIDELITY (or INDEPENDENT_REFERENCE_FIDELITY) once it is admissible."""

    def _evidence_and_workflow(self, root):
        teacher_model = _dummy_teacher_model(root)
        db_path, manifest_path = _write_training_db_and_manifest(root)
        sources = {"teacher_model_path": str(teacher_model),
                   "original_training_db_path": str(db_path),
                   "split_source_manifest_paths": [str(manifest_path)],
                   "target_split": "test"}
        contract_sources = _write_validation_contract_sources(
            root, objectives=["require_predictive_fidelity_when_evidence_supports_it"])
        # Every component this case's proposals ever select (OPERATIONAL_ROBUSTNESS,
        # ORIGINAL_HELDOUT_FIDELITY) has a declared executable stage, so the whole-workflow
        # coverage invariant is satisfied -- this case is about objective-driven semantic
        # correction, not stage coverage.
        workflow = _write_workflow(root, teacher_evidence_sources=sources,
                                   stage_a_component="OPERATIONAL_ROBUSTNESS",
                                   stage_b_component="ORIGINAL_HELDOUT_FIDELITY",
                                   validation_contract_sources=contract_sources)
        return sources, workflow

    def _events(self, run_dir):
        path = run_dir / "campaign_events.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_objective_violating_first_proposal_is_corrected_on_retry(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import (
            derive_admissible_decision_space, inspect_teacher_evidence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, workflow = self._evidence_and_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            self.assertEqual(c._teacher_validation_objectives(),
                             ["require_predictive_fidelity_when_evidence_supports_it"])
            profile, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            admissible = set(derive_admissible_decision_space(profile)["admissible_components"])
            self.assertIn("ORIGINAL_HELDOUT_FIDELITY", admissible,
                          "fixture must genuinely admit fidelity evidence for this case to be "
                          "meaningful")

            rejected_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["OPERATIONAL_ROBUSTNESS", "TRAINING_CORPUS_CONSISTENCY"],
                rationale="admissible subset that omits fidelity evidence",
                name="attempt_1_rejected.json")
            corrected_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["OPERATIONAL_ROBUSTNESS", "ORIGINAL_HELDOUT_FIDELITY"],
                rationale="corrected to also satisfy the declared fidelity objective",
                name="attempt_2_accepted.json")

            code = cli.main([
                "run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                "--auto-mock-judges", "--mock-orchestrator-response",
                f"{rejected_response},{corrected_response}"])
            # Planning itself succeeds (committed); the campaign then pauses at stage_a's own
            # costly_training approval boundary, exactly like every other case in this file.
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)

            c = RunController(run_dir)
            plan = c.state["teacher_validation_plan"]
            self.assertIsNotNone(plan, "the corrected second proposal must have committed a plan")
            self.assertEqual(set(plan["selected_components"]),
                             {"OPERATIONAL_ROBUSTNESS", "ORIGINAL_HELDOUT_FIDELITY"})

            # Exactly two attempts: the first rejected, the second accepted -- never silently
            # retried past what actually happened, never collapsed into a single event.
            completed = [e for e in self._events(run_dir)
                        if e["event"] == "role_invocation_completed"
                        and e.get("action") == "teacher_validation_plan_proposal"]
            self.assertEqual([e["detail"]["accepted"] for e in completed], [False, True])

            # The rejected first attempt never left behind an accepted reasoning-output artifact
            # that anything could mistake for the committed plan.
            reasoning_dir = run_dir / "exchange" / "reasoning_outputs"
            if reasoning_dir.is_dir():
                for artifact_path in reasoning_dir.glob("*.json"):
                    payload = json.loads(artifact_path.read_text())
                    self.assertEqual(set(payload.get("selected_components", [])),
                                     {"OPERATIONAL_ROBUSTNESS", "ORIGINAL_HELDOUT_FIDELITY"},
                                     "only the accepted proposal may ever be persisted as a "
                                     "reasoning-output artifact")

    def test_objective_violation_that_is_never_corrected_fails_closed_within_the_bound(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, workflow = self._evidence_and_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)

            # This same objective-violating proposal is repeated for every attempt -- the planner
            # never corrects itself, so the bounded retry must still terminate, never loop
            # unboundedly, and never commit a plan.
            always_rejected = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["OPERATIONAL_ROBUSTNESS", "TRAINING_CORPUS_CONSISTENCY"],
                rationale="never corrects the omitted fidelity objective",
                name="always_rejected.json")

            code = cli.main([
                "run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                "--auto-mock-judges", "--mock-orchestrator-response", str(always_rejected)])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
            c = RunController(run_dir)
            self.assertIsNone(c.state["teacher_validation_plan"])

            completed = [e for e in self._events(run_dir)
                        if e["event"] == "role_invocation_completed"
                        and e.get("action") == "teacher_validation_plan_proposal"]
            self.assertEqual(len(completed), 3,
                             "initial attempt + at most 2 semantic-correction retries, then FAILED")
            self.assertEqual([e["detail"]["accepted"] for e in completed], [False, False, False])

    def test_already_valid_first_proposal_incurs_no_additional_llm_call(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, workflow = self._evidence_and_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)

            already_valid = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["OPERATIONAL_ROBUSTNESS", "ORIGINAL_HELDOUT_FIDELITY"],
                rationale="already satisfies the declared fidelity objective on the first try",
                name="already_valid.json")

            code = cli.main([
                "run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                "--auto-mock-judges", "--mock-orchestrator-response", str(already_valid)])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            c = RunController(run_dir)
            self.assertIsNotNone(c.state["teacher_validation_plan"])

            completed = [e for e in self._events(run_dir)
                        if e["event"] == "role_invocation_completed"
                        and e.get("action") == "teacher_validation_plan_proposal"]
            self.assertEqual(len(completed), 1,
                             "a proposal that is valid on the first attempt must incur exactly "
                             "one LLM call, never a speculative retry")
            self.assertTrue(completed[0]["detail"]["accepted"])


class CaseC1b_UnsupportedComponentIsNeverAcceptedAcrossRetries(unittest.TestCase):
    """The bounded semantic-correction retry (Case H) must never blur into overriding Case C1's
    unconditional rule: a proposal claiming a component the evidence does not admit is rejected
    on every attempt (there is nothing an evidence-bound correction could legitimately change it
    to), and the campaign still fails closed within the same bound -- never accepted merely
    because it was retried."""

    def test_inadmissible_component_repeated_across_every_attempt_is_never_accepted(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_model = _dummy_teacher_model(root)
            population = root / "operational_population.extxyz"
            _write_operational_population(population)
            sources = {"teacher_model_path": str(teacher_model),
                       "operational_evaluation_population_path": str(population)}
            workflow = _write_workflow(root, teacher_evidence_sources=sources)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            # ORIGINAL_HELDOUT_FIDELITY is never admissible under this evidence -- unconditional,
            # not merely objective-conditional -- rejection.
            inadmissible = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["ORIGINAL_HELDOUT_FIDELITY"], name="inadmissible.json")

            code = cli.main([
                "run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                "--auto-mock-judges", "--mock-orchestrator-response", str(inadmissible)])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
            c = RunController(run_dir)
            self.assertIsNone(c.state["teacher_validation_plan"])


class CaseI_TeacherValidationPlanCoverageGapFailsClosed(unittest.TestCase):
    """Case I: a committed plan may legitimately select a component the evidence profile
    genuinely admits, yet NO declared stage anywhere in the workflow is capable of executing it
    -- e.g. every stage capable of *some* component would, one at a time, simply be marked
    NOT_APPLICABLE, and the campaign would silently proceed past Teacher validation having never
    executed the component it actually committed to. This is exactly the real
    ``sio2-sox-allegro-simplenn-r24`` production defect (reproduced here with fully generic
    stage/evidence names): the plan committed ``ORIGINAL_HELDOUT_FIDELITY``, but the workflow's
    two Teacher-validation-capable stages only declared ``OPERATIONAL_ROBUSTNESS`` and
    ``TRAINING_CORPUS_CONSISTENCY``. ``_teacher_validation_plan_coverage_gap`` must fail the
    campaign closed BEFORE either stage is dispatched or marked NOT_APPLICABLE -- never merely
    logged, never silently skipped."""

    def _evidence_sources_and_workflow(self, root, *, stage_a_component, stage_b_component):
        teacher_model = _dummy_teacher_model(root)
        population = root / "operational_population.extxyz"
        _write_operational_population(population)
        db_path, manifest_path = _write_training_db_and_manifest(root, heldout_split_name="test")
        sources = {"teacher_model_path": str(teacher_model),
                   "operational_evaluation_population_path": str(population),
                   "original_training_db_path": str(db_path),
                   "split_source_manifest_paths": [str(manifest_path)],
                   "target_split": "test"}
        workflow = _write_workflow(root, teacher_evidence_sources=sources,
                                   stage_a_component=stage_a_component,
                                   stage_b_component=stage_b_component)
        return sources, workflow

    def test_reproduces_r24_defect_fails_closed_and_never_dispatches_a_stage(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Neither stage declares ORIGINAL_HELDOUT_FIDELITY capability -- the exact R24 shape.
            sources, workflow = self._evidence_sources_and_workflow(
                root, stage_a_component="OPERATIONAL_ROBUSTNESS",
                stage_b_component="TRAINING_CORPUS_CONSISTENCY")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            mock_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["ORIGINAL_HELDOUT_FIDELITY"])

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
            c = RunController(run_dir)
            # The plan itself was legitimately committed -- the evidence genuinely admitted the
            # selection; the defect is purely in EXECUTION coverage, not plan validity.
            self.assertEqual(c.state["teacher_validation_plan"]["selected_components"],
                             ["ORIGINAL_HELDOUT_FIDELITY"])
            # Neither stage was ever dispatched OR marked not_applicable -- the campaign never
            # advanced past the coverage-gap check to reach a per-stage decision at all.
            self.assertEqual(c.stage("stage_a")["status"], "pending")
            self.assertEqual(c.stage("stage_b")["status"], "pending")
            self.assertFalse(any(e["type"] == "stage_marked_not_applicable"
                                 for e in c.state["events"]))
            events_log = (run_dir / "campaign_events.jsonl").read_text()
            self.assertIn("TEACHER_VALIDATION_PLAN_COVERAGE_GAP", events_log)

            # Deterministic: re-running without any config change fails closed identically again,
            # rather than e.g. only failing once and then proceeding.
            code_again = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                                   "--auto-mock-judges",
                                   "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code_again, cli.EXIT_VALIDATION_REJECTED)

    def test_corrected_capability_makes_reference_validation_style_stage_applicable(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        from validation.teacher_evidence_profile import inspect_teacher_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The FIX: stage_b (the reference_validation-style stage) now also declares
            # ORIGINAL_HELDOUT_FIDELITY capability alongside TRAINING_CORPUS_CONSISTENCY. stage_a
            # declares no capability at all here (it is the ml-training stage stage_b's own
            # action depends on, not a Teacher-validation execution stage), so it stays
            # unconditionally applicable and produces the committee manifest stage_b consumes --
            # coverage of the selected component is fully satisfied by stage_b alone.
            sources, workflow = self._evidence_sources_and_workflow(
                root, stage_a_component=None,
                stage_b_component=["TRAINING_CORPUS_CONSISTENCY", "ORIGINAL_HELDOUT_FIDELITY"])
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            _, evidence_profile_sha256 = inspect_teacher_evidence(**sources)
            mock_response = _mock_orchestrator_response(
                root, run_id=c.state["run_id"], evidence_profile_sha256=evidence_profile_sha256,
                selected_components=["ORIGINAL_HELDOUT_FIDELITY"])

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            c = RunController(run_dir)
            plan = c.state["teacher_validation_plan"]
            self.assertEqual(plan["selected_components"], ["ORIGINAL_HELDOUT_FIDELITY"])
            # The resolved split comes from the Controller's own evidence-profile resolution, not
            # from any hardcoded value in production logic.
            self.assertEqual(plan["target_split"], "test")

            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training",
                                       "--note", "test approval"]), cli.EXIT_SUCCESS)

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges",
                             "--mock-orchestrator-response", str(mock_response)])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            c = RunController(run_dir)
            # stage_a declares no capability at all, so it is unconditionally applicable
            # regardless of what the plan selects (pre-existing governance, unchanged).
            self.assertEqual(c.stage("stage_a")["status"], "completed")
            # stage_b is now covered and actually executes the committed validation component.
            self.assertEqual(c.stage("stage_b")["status"], "completed")
            self.assertNotEqual(c.stage("stage_b")["status"], "not_applicable")

    def test_applicability_matrix_across_component_selections(self):
        from runtimes.pydantic_ai.cli import _teacher_validation_plan_coverage_gap

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Mirrors the real teacher_baseline/reference_validation capability shape.
            _, workflow = self._evidence_sources_and_workflow(
                root, stage_a_component=["OPERATIONAL_ROBUSTNESS", "DEPLOYMENT_APPLICABILITY"],
                stage_b_component=["TRAINING_CORPUS_CONSISTENCY", "ORIGINAL_HELDOUT_FIDELITY"])

            class _FakeController:
                def __init__(self, selected):
                    self.state = {"workflow_config": str(workflow),
                                  "teacher_validation_plan": {"selected_components": selected}}

            for selected in (["OPERATIONAL_ROBUSTNESS"],
                            ["TRAINING_CORPUS_CONSISTENCY"],
                            ["ORIGINAL_HELDOUT_FIDELITY"],
                            ["OPERATIONAL_ROBUSTNESS", "TRAINING_CORPUS_CONSISTENCY"]):
                gap = _teacher_validation_plan_coverage_gap(_FakeController(selected))
                self.assertIsNone(gap, f"expected full coverage for selected={selected}")

            uncovered = _teacher_validation_plan_coverage_gap(
                _FakeController(["INDEPENDENT_REFERENCE_FIDELITY"]))
            self.assertIsNotNone(uncovered)
            self.assertIn("INDEPENDENT_REFERENCE_FIDELITY", uncovered)

            # No committed plan yet -- never a coverage gap (nothing to check coverage of).
            class _NoPlanController:
                state = {"workflow_config": str(workflow), "teacher_validation_plan": None}
            self.assertIsNone(_teacher_validation_plan_coverage_gap(_NoPlanController()))


# --------------------------------------------------------------------------------------------
# SiO2 no-hardcode proof
# --------------------------------------------------------------------------------------------

_FORBIDDEN_SUBSTRINGS = ("sio2", "allegro", "silica", "silicon dioxide")


def _non_docstring_string_constants(tree: ast.AST):
    """Every string literal in ``tree`` EXCLUDING module/function/class docstrings -- i.e. the
    genuinely executable string content (dict keys, comparisons, messages), which is exactly what
    a hardcoded material/campaign branch would have to use. Comments are never part of the AST at
    all, so they are already excluded without any special-casing."""
    docstring_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstring_ids.add(id(body[0].value))
    values = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstring_ids):
            values.append(node.value)
    return values


def _assert_no_hardcoded_material_name(test_case, values, *, label):
    lowered = [v.lower() for v in values]
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        hits = [v for v in lowered if forbidden in v]
        test_case.assertEqual(
            hits, [], f"{label} contains a hardcoded material/campaign-specific literal "
                      f"(matched {forbidden!r}): {hits!r}")


class SiO2NoHardcodeProofTests(unittest.TestCase):
    """Ordering/generality assertion: the ENTIRE new evidence-model/planner modules, and every
    new function this session added to the shared controller.py/cli.py, contain no material name
    or campaign-specific constant anywhere in their non-docstring string literals -- the only
    mentions of "SiO2"/"Allegro" anywhere in this codebase are disclaiming their absence in
    documentation, never appearing in a comparison, dict key, or branch condition."""

    def test_teacher_evidence_profile_module_is_fully_generic(self):
        source = (ROOT / "validation" / "teacher_evidence_profile.py").read_text()
        tree = ast.parse(source)
        _assert_no_hardcoded_material_name(
            self, _non_docstring_string_constants(tree), label="teacher_evidence_profile.py")

    def test_teacher_validation_plan_module_is_fully_generic(self):
        source = (ROOT / "runtimes" / "pydantic_ai" / "teacher_validation_plan.py").read_text()
        tree = ast.parse(source)
        _assert_no_hardcoded_material_name(
            self, _non_docstring_string_constants(tree), label="teacher_validation_plan.py")

    def test_new_controller_functions_are_fully_generic(self):
        source = (ROOT / "workflow" / "controller.py").read_text()
        tree = ast.parse(source)
        target_names = {"mark_stage_not_applicable", "commit_teacher_validation_plan",
                        "authorize_downstream_teacher_reliance", "_teacher_validation_objectives"}
        values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in target_names:
                values.extend(_non_docstring_string_constants(node))
        self.assertTrue(values, "expected to find the new v10 controller methods by name")
        _assert_no_hardcoded_material_name(self, values, label="controller.py v10 methods")

    def test_new_cli_functions_are_fully_generic(self):
        source = (ROOT / "runtimes" / "pydantic_ai" / "cli.py").read_text()
        tree = ast.parse(source)
        target_names = {"_teacher_validation_not_applicable_reason",
                        "_teacher_validation_plan_coverage_gap",
                        "_commit_teacher_validation_plan_via_reasoning_roles",
                        "_cmd_plan_teacher_validation",
                        "_cmd_authorize_downstream_teacher_reliance"}
        values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in target_names:
                values.extend(_non_docstring_string_constants(node))
        self.assertTrue(values, "expected to find the new v10 cli.py functions by name")
        _assert_no_hardcoded_material_name(self, values, label="cli.py v10 functions")


if __name__ == "__main__":
    unittest.main()
