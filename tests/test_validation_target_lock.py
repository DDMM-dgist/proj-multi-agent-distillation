"""v8 validation-target-lock: the write-once validation_contract and its execution-time
cross-checks in teacher_baseline, data_coverage, and dataset_split.

The scientific invariant under test: the Teacher applicability domain, validation scope, and
dataset split policy must be frozen BEFORE any stage that produces Student results, are
write-once for the lifetime of a run, and can never be mutated or re-established by recovery —
even though recovery may legitimately re-run an earlier, contract-CONSUMING stage (such as
teacher_baseline) after Student results have already existed.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

from validation.data_coverage import validate_data_coverage_report
from validation.report import evidence_record
from validation.teacher_baseline import validate_teacher_baseline_report
from workflow.controller import RunController
from workflow.steps import establish_validation_contract_from_configs, split_dataset


GATE_CRITERION = "artifact is complete and internally consistent"

TEACHER_DOMAIN = {"structure_classes": ["liquid", "crystal"], "temperature_range_K": [300, 1500]}
OTHER_TEACHER_DOMAIN = {"structure_classes": ["liquid"], "temperature_range_K": [300, 1000]}
VALIDATION_SCOPE = {"shared_md_protocol": "nvt-1000K-v1", "checks": ["diffusion", "rdf"]}
SPLIT_POLICY = {"seed": 7, "validation_fraction": 0.2, "test_fraction": 0.2,
                "grouping_key": "parent_structure_id"}


def contract_components(domain=None, scope=None, split_policy=None):
    return {"teacher_applicability_domain": domain or TEACHER_DOMAIN,
            "validation_scope": scope or VALIDATION_SCOPE,
            "dataset_split_policy": split_policy or SPLIT_POLICY}


def write_workflow(root, stages):
    cfg_path = root / "workflow.yaml"
    cfg_path.write_text(yaml.safe_dump({"run_id": "lock-test", "stages": stages},
                                       sort_keys=False))
    return cfg_path


def student_stage(name, output):
    return {"name": name, "command": None, "outputs": [output],
           "produces_student_results": True,
           "gate": {"criteria": [GATE_CRITERION]}}


def plain_stage(name, output):
    return {"name": name, "command": None, "outputs": [output],
           "gate": {"criteria": [GATE_CRITERION]}}


class ContractEstablishmentTests(unittest.TestCase):
    def test_write_once_idempotent_and_hardfails_on_different_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("teacher_baseline", "artifacts/tb.txt")])
            controller = RunController.initialize(cfg, root / "run")
            first = controller.establish_validation_contract(contract_components())
            self.assertEqual(
                len([e for e in controller.state["events"]
                     if e["type"] == "validation_contract_established"]), 1)
            second = controller.establish_validation_contract(contract_components())
            self.assertEqual(first["contract_sha256"], second["contract_sha256"])
            self.assertEqual(
                len([e for e in controller.state["events"]
                     if e["type"] == "validation_contract_established"]), 1,
                "identical re-establishment must not append a new event")
            with self.assertRaisesRegex(ValueError, "different"):
                controller.establish_validation_contract(
                    contract_components(domain=OTHER_TEACHER_DOMAIN))
            self.assertEqual(controller.state["validation_contract"]["contract_sha256"],
                             first["contract_sha256"], "a rejected mutation must not persist")

    def test_must_be_established_before_any_stage_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("teacher_baseline", "artifacts/tb.txt")])
            controller = RunController.initialize(cfg, root / "run")
            artifact = controller.run_dir / "artifacts/tb.txt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("baseline")
            controller.complete_external_stage("teacher_baseline", [artifact])
            with self.assertRaisesRegex(RuntimeError, "before any stage"):
                controller.establish_validation_contract(contract_components())


class ProducesStudentResultsGateTests(unittest.TestCase):
    def test_complete_external_stage_blocks_student_stage_until_contract_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [student_stage("training", "artifacts/model.txt")])
            controller = RunController.initialize(cfg, root / "run")
            model = controller.run_dir / "artifacts/model.txt"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_text("model v1")
            with self.assertRaisesRegex(RuntimeError, "cannot run until"):
                controller.complete_external_stage("training", [model])

            controller.establish_validation_contract(contract_components())
            controller.complete_external_stage("training", [model])
            self.assertTrue(
                controller.state["validation_contract"]["student_stage_ever_completed"])

    def test_run_stage_blocks_student_stage_until_contract_exists(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [
                {"name": "training", "command": [sys.executable, "-c",
                                                 "open('artifacts/model.txt','w').write('m')"],
                 "outputs": ["artifacts/model.txt"], "produces_student_results": True,
                 "gate": {"criteria": [GATE_CRITERION]}},
            ])
            controller = RunController.initialize(cfg, root / "run")
            with self.assertRaisesRegex(RuntimeError, "cannot run until"):
                controller.run_stage("training")
            controller.establish_validation_contract(contract_components())
            controller.run_stage("training")
            self.assertEqual(controller.stage("training")["status"], "completed")
            self.assertTrue(
                controller.state["validation_contract"]["student_stage_ever_completed"])

    def _pass(self, controller, stage):
        artifacts = {a["path"]: a["sha256"] for a in controller.stage_artifacts(stage)}
        vote_path = controller.run_dir / "gates" / f"{stage}.votes.json"
        criteria = controller.stage(stage).get("gate_criteria")
        lenses = controller.stage(stage).get("gate_review_lenses")
        vote_path.write_text(json.dumps({
            "stage": stage, "criteria": criteria, "review_lenses": lenses,
            "artifact_sha256": artifacts, "decision": "PASS",
            "votes": [{"judge_id": f"judge-{i}", "review_lens": lens["id"], "verdict": "PASS",
                      "criteria_checked": [{"criterion": c, "value_read": "verified", "ok": True}
                                          for c in criteria],
                      "rationale": "ok", "required_fix": ""}
                     for i, lens in enumerate(lenses, 1)],
        }))
        controller.record_gate(stage, votes_path=vote_path)


class ConfigDrivenEstablishmentTests(unittest.TestCase):
    def _write_configs(self, root, *, profile_domain=None, split_policy=None):
        scope = root / "distillation_scope.yaml"
        scope.write_text(yaml.safe_dump({"deployment_domain": TEACHER_DOMAIN}))
        profile = root / "validation_profile.yaml"
        profile.write_text(yaml.safe_dump({
            "deployment_domain": profile_domain if profile_domain is not None else TEACHER_DOMAIN,
            "shared_md_protocol": "nvt-1000K-v1", "checks": ["diffusion", "rdf"],
        }))
        policy = root / "dataset_policy.yaml"
        policy.write_text(yaml.safe_dump({
            "split_policy": split_policy if split_policy is not None else SPLIT_POLICY,
        }))
        return scope, profile, policy

    def test_authoritative_domain_mismatch_hardfails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("teacher_baseline", "artifacts/tb.txt")])
            controller = RunController.initialize(cfg, root / "run")
            scope, profile, policy = self._write_configs(
                root, profile_domain=OTHER_TEACHER_DOMAIN)
            with self.assertRaisesRegex(ValueError, "does not match the authoritative"):
                establish_validation_contract_from_configs(
                    controller.run_dir, scope, profile, policy)
            self.assertIsNone(controller.state["validation_contract"])

    def test_success_matches_direct_establishment_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("teacher_baseline", "artifacts/tb.txt")])
            controller = RunController.initialize(cfg, root / "run")
            scope, profile, policy = self._write_configs(root)
            record = establish_validation_contract_from_configs(
                controller.run_dir, scope, profile, policy)
            self.assertEqual(record["components"]["teacher_applicability_domain"]["value"],
                             TEACHER_DOMAIN)
            again = establish_validation_contract_from_configs(
                controller.run_dir, scope, profile, policy)
            self.assertEqual(record["contract_sha256"], again["contract_sha256"])


class SplitPolicyEnforcementTests(unittest.TestCase):
    def _dataset(self, root):
        source = root / "labeled.extxyz"
        frames = []
        for group in range(5):
            for child in range(2):
                atoms = Atoms("H", positions=[[group + child * 0.01, 0, 0]])
                atoms.info.update(structure_id=f"g{group}-c{child}", parent_structure_id=f"g{group}")
                frames.append(atoms)
        write(source, frames)
        return source

    def test_split_dataset_rejects_params_that_differ_from_the_locked_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("teacher_baseline", "artifacts/tb.txt")])
            controller = RunController.initialize(cfg, root / "run")
            controller.establish_validation_contract(contract_components())
            contract_path = controller.run_dir / "validation_contract.json"
            source = self._dataset(root)

            split_dataset(source, root / "splits-ok", root / "split-ok.json",
                          seed=7, validation_fraction=0.2, test_fraction=0.2,
                          validation_contract_path=contract_path)

            with self.assertRaisesRegex(ValueError, "do not match the locked validation contract"):
                split_dataset(source, root / "splits-bad", root / "split-bad.json",
                              seed=99, validation_fraction=0.2, test_fraction=0.2,
                              validation_contract_path=contract_path)


class TeacherBaselineDomainCrossCheckTests(unittest.TestCase):
    def _report(self, root, domain):
        teacher_config = root / "teacher.yaml"
        teacher_config.write_text("kind: mock\n")
        validation_profile = root / "validation.yaml"
        validation_profile.write_text("kind: generic\n")
        distillation_scope = root / "scope.yaml"
        distillation_scope.write_text("deployment_domain: {system: test}\n")
        evidence = root / "teacher-trajectory.xyz"
        evidence.write_text("teacher evidence")
        report = root / "teacher_baseline.json"
        report.write_text(json.dumps({
            "schema_version": 1, "profile": "deployment-v1",
            "teacher": {"config": str(teacher_config)},
            "distillation_scope": str(distillation_scope),
            "validation_profile": str(validation_profile),
            "deployment_domain": domain,
            "applicability": {"status": "CONDITIONAL", "limitations": ["high-T only"]},
            "checks": [{
                "domain": "dynamics", "observable": "diffusion", "status": "RECORDED",
                "value": 1.2, "unit": "A2/ps", "criterion": None,
                "purpose": "student_teacher_fidelity", "reference_source": "teacher",
                "protocol": "nvt-1000K-v1",
            }],
            "evidence": [evidence_record("teacher_config", teacher_config),
                        evidence_record("distillation_scope", distillation_scope),
                        evidence_record("validation_profile", validation_profile),
                        evidence_record("teacher_trajectory", evidence)],
        }))
        return report

    def test_passes_under_matching_domain_and_fails_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("teacher_baseline", "artifacts/tb.txt")])
            controller = RunController.initialize(cfg, root / "run")
            controller.establish_validation_contract(contract_components(domain=TEACHER_DOMAIN))
            contract_path = controller.run_dir / "validation_contract.json"

            matching_report = self._report(root, TEACHER_DOMAIN)
            validate_teacher_baseline_report(
                matching_report, validation_contract_path=contract_path)

            mismatched_report = self._report(root, OTHER_TEACHER_DOMAIN)
            with self.assertRaisesRegex(ValueError, "does not match the run's locked"):
                validate_teacher_baseline_report(
                    mismatched_report, validation_contract_path=contract_path)


class DataCoverageCrossCheckTests(unittest.TestCase):
    def _report(self, root, domain, split_policy):
        evidence = root / "dataset.extxyz"
        frames = []
        for index in range(10):
            atoms = Atoms("Cu", positions=[[index * 0.1, 0, 0]])
            atoms.info.update(parent_structure_id=f"parent-{index}", label_source="teacher")
            frames.extend([atoms.copy() for _ in range(10)])
        write(evidence, frames)
        dataset_policy = root / "dataset_policy.yaml"
        dataset_policy.write_text(yaml.safe_dump({
            "teacher_training_data_access": "unavailable", "split_policy": split_policy,
        }))
        report = root / "coverage.json"
        payload = {
            "schema_version": 1,
            "teacher_training_data_access": "unavailable",
            "dataset_policy": str(dataset_policy),
            "coverage_status": "NOT_ASSESSABLE",
            "deployment_domain": domain,
            "dataset_sources": [{
                "category": "generated_teacher_labeled", "n_parents": 10,
                "n_frames": 100, "fraction": 1.0, "label_sources": ["teacher"],
                "evidence_role": "distillation_dataset",
                "statistics": {"kind": "ase", "grouping_key": "parent_structure_id"},
            }],
            "coverage_dimensions": {},
            "replay_policy": {"enabled": False},
            "identified_gaps": ["teacher training distribution unavailable"],
            "limitations": ["quantitative teacher-set coverage cannot be computed"],
            "evidence": [evidence_record("dataset_policy", dataset_policy),
                        evidence_record("distillation_dataset", evidence)],
        }
        report.write_text(json.dumps(payload))
        return report

    def test_crosschecks_domain_and_split_policy_against_the_locked_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("data_coverage", "artifacts/dc.txt")])
            controller = RunController.initialize(cfg, root / "run")
            controller.establish_validation_contract(
                contract_components(domain=TEACHER_DOMAIN, split_policy=SPLIT_POLICY))
            contract_path = controller.run_dir / "validation_contract.json"

            matching = self._report(root, TEACHER_DOMAIN, SPLIT_POLICY)
            validate_data_coverage_report(matching, validation_contract_path=contract_path)

            bad_domain = self._report(root, OTHER_TEACHER_DOMAIN, SPLIT_POLICY)
            with self.assertRaisesRegex(ValueError, "deployment_domain does not match"):
                validate_data_coverage_report(bad_domain, validation_contract_path=contract_path)

            bad_split = self._report(root, TEACHER_DOMAIN, dict(SPLIT_POLICY, seed=99))
            with self.assertRaisesRegex(ValueError, "split_policy does not match"):
                validate_data_coverage_report(bad_split, validation_contract_path=contract_path)


class RecoveryPreservesContractTests(unittest.TestCase):
    """The explicitly required distinction: after Student results have existed, recovery may
    return to teacher_baseline and re-run it successfully under the SAME frozen contract, but
    changing the contract-relevant configuration requires a new run, not a mutation."""

    def _pass(self, controller, stage):
        artifacts = {a["path"]: a["sha256"] for a in controller.stage_artifacts(stage)}
        vote_path = controller.run_dir / "gates" / f"{stage}.votes.json"
        criteria = controller.stage(stage).get("gate_criteria")
        lenses = controller.stage(stage).get("gate_review_lenses")
        vote_path.write_text(json.dumps({
            "stage": stage, "criteria": criteria, "review_lenses": lenses,
            "artifact_sha256": artifacts, "decision": "PASS",
            "votes": [{"judge_id": f"judge-{i}", "review_lens": lens["id"], "verdict": "PASS",
                      "criteria_checked": [{"criterion": c, "value_read": "verified", "ok": True}
                                          for c in criteria],
                      "rationale": "ok", "required_fix": ""}
                     for i, lens in enumerate(lenses, 1)],
        }))
        controller.record_gate(stage, votes_path=vote_path)

    def test_recovery_rerun_of_teacher_baseline_succeeds_under_unchanged_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [
                plain_stage("teacher_baseline", "artifacts/tb.txt"),
                student_stage("training", "artifacts/model.txt"),
                plain_stage("evaluation", "artifacts/eval.txt"),
            ])
            controller = RunController.initialize(cfg, root / "run")
            controller.establish_validation_contract(contract_components())
            frozen_sha256 = controller.state["validation_contract"]["contract_sha256"]

            tb = controller.run_dir / "artifacts/tb.txt"
            tb.parent.mkdir(parents=True, exist_ok=True)
            tb.write_text("baseline v1")
            controller.complete_external_stage("teacher_baseline", [tb])
            self._pass(controller, "teacher_baseline")

            model = controller.run_dir / "artifacts/model.txt"
            model.write_text("model v1")
            controller.complete_external_stage("training", [model])
            self._pass(controller, "training")
            self.assertTrue(
                controller.state["validation_contract"]["student_stage_ever_completed"])

            evaluation = controller.run_dir / "artifacts/eval.txt"
            evaluation.write_text("failed evidence")
            controller.complete_external_stage("evaluation", [evaluation])
            controller.record_gate("evaluation", "REVISE", evidence="fidelity gap")

            plan = root / "recovery-plan.json"
            plan.write_text(json.dumps({
                "schema_version": 1, "failed_stage": "evaluation",
                "failure_category": "student_fidelity",
                "root_cause": "teacher baseline needs to be regenerated deterministically",
                "responsible_agent": "simulation", "return_stage": "teacher_baseline",
                "proposed_changes": [{"type": "regenerate_teacher_baseline"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": True, "mode": "from_scratch"},
                "revalidation": {"reuse_profile": True, "targets": ["evaluation"]},
                "estimated_cost": {"gpu_hours": 1},
            }))
            recovery = controller.propose_recovery(plan)
            controller.approve_recovery("researcher", "approved rerun")
            controller.start_iteration()
            self.assertEqual(controller.stage("teacher_baseline")["status"], "pending")

            # Re-running teacher_baseline under the SAME (unchanged) contract must succeed.
            self.assertEqual(controller.state["validation_contract"]["contract_sha256"],
                             frozen_sha256)
            tb.write_text("baseline v2 (regenerated)")
            controller.complete_external_stage("teacher_baseline", [tb])
            self._pass(controller, "teacher_baseline")
            self.assertEqual(controller.stage("teacher_baseline")["status"], "completed")
            self.assertEqual(controller.stage("teacher_baseline")["gate"], "PASS")
            self.assertEqual(controller.state["validation_contract"]["contract_sha256"],
                             frozen_sha256, "recovery must never mutate the frozen contract")

            model.write_text("model v2")
            controller.complete_external_stage("training", [model])
            self._pass(controller, "training")
            evaluation.write_text("revalidated evidence")
            controller.complete_external_stage("evaluation", [evaluation])
            with self.assertRaisesRegex(RuntimeError, "recovery execution"):
                self._pass(controller, "evaluation")
            execution = root / "recovery-execution.json"
            execution.write_text(json.dumps({
                "schema_version": 1, "recovery_id": recovery["id"],
                "previous_iteration": 1, "current_iteration": 2,
                "changes": [{"type": "regenerate_teacher_baseline", "status": "APPLIED",
                            "evidence_artifacts": ["artifacts/tb.txt"]}],
                "labeling": {"teacher_relabel": False, "teacher_relabel_stage": None,
                            "new_dft": False, "new_dft_stage": None},
                "student_training": {"retrain": True, "mode": "from_scratch",
                                     "stage": "training"},
                "revalidation": {"targets": ["evaluation"], "stages": ["evaluation"]},
            }))
            controller.verify_recovery_execution(execution)
            self._pass(controller, "evaluation")
            self.assertEqual(controller.stage("evaluation")["gate"], "PASS")
            self.assertEqual(controller.state["recoveries"][-1]["status"], "resolved")

    def test_changing_contract_relevant_config_after_student_results_hardfails_new_run_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [
                plain_stage("teacher_baseline", "artifacts/tb.txt"),
                student_stage("training", "artifacts/model.txt"),
            ])
            controller = RunController.initialize(cfg, root / "run")
            controller.establish_validation_contract(contract_components())

            tb = controller.run_dir / "artifacts/tb.txt"
            tb.parent.mkdir(parents=True, exist_ok=True)
            tb.write_text("baseline v1")
            controller.complete_external_stage("teacher_baseline", [tb])
            self._pass(controller, "teacher_baseline")
            model = controller.run_dir / "artifacts/model.txt"
            model.write_text("model v1")
            controller.complete_external_stage("training", [model])
            self._pass(controller, "training")
            self.assertTrue(
                controller.state["validation_contract"]["student_stage_ever_completed"])

            # A genuine change to the Teacher applicability domain after Student results have
            # existed must hard-fail: it requires a new run, not a mutation of this one.
            with self.assertRaisesRegex(ValueError, "different"):
                controller.establish_validation_contract(
                    contract_components(domain=OTHER_TEACHER_DOMAIN))


class AutomaticContractEstablishmentAtInitializationTests(unittest.TestCase):
    """v11: RunController.initialize() itself establishes the contract for workflows that
    declare validation_contract_sources — no test here calls establish_validation_contract or
    establish_validation_contract_from_configs directly; every assertion is driven purely by
    what a normal RunController.initialize() call produces.
    """

    def _write_configs(self, root, *, profile_domain=None, split_policy=None, domain=None):
        scope = root / "distillation_scope.yaml"
        scope.write_text(yaml.safe_dump({"deployment_domain": domain or TEACHER_DOMAIN}))
        profile = root / "validation_profile.yaml"
        profile.write_text(yaml.safe_dump({
            "deployment_domain": profile_domain if profile_domain is not None
                                 else (domain or TEACHER_DOMAIN),
            "shared_md_protocol": "nvt-1000K-v1", "checks": ["diffusion", "rdf"],
        }))
        policy = root / "dataset_policy.yaml"
        policy.write_text(yaml.safe_dump({
            "split_policy": split_policy if split_policy is not None else SPLIT_POLICY,
        }))
        return scope, profile, policy

    def _write_workflow_with_sources(self, root, stages, scope, profile, policy):
        cfg_path = root / "workflow.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "run_id": "auto-lock-test",
            "validation_contract_sources": {
                "distillation_scope": str(scope),
                "validation_profile": str(profile),
                "dataset_policy": str(policy),
            },
            "stages": stages,
        }, sort_keys=False))
        return cfg_path

    def test_contract_exists_immediately_after_initialize_and_matches_manual_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope, profile, policy = self._write_configs(root)
            cfg = self._write_workflow_with_sources(
                root, [plain_stage("teacher_baseline", "artifacts/tb.txt")], scope, profile, policy)

            controller = RunController.initialize(cfg, root / "run")

            record = controller.state["validation_contract"]
            self.assertIsNotNone(record, "initialize() must establish the contract automatically")
            self.assertEqual(record["components"]["teacher_applicability_domain"]["value"],
                             TEACHER_DOMAIN)
            self.assertEqual(record["components"]["validation_scope"]["value"], VALIDATION_SCOPE)
            self.assertEqual(record["components"]["dataset_split_policy"]["value"], SPLIT_POLICY)
            self.assertFalse(record["student_stage_ever_completed"])
            # exactly one authoritative representation: the on-disk lock file must be a
            # byte-for-byte serialization of the controller-state record.
            lock_path = controller.run_dir / "validation_contract.json"
            self.assertTrue(lock_path.exists())
            self.assertEqual(json.loads(lock_path.read_text()), record)

    def test_contract_is_built_from_run_bound_snapshots_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope, profile, policy = self._write_configs(root)
            original_hashes = {
                "distillation_scope": hashlib.sha256(scope.read_bytes()).hexdigest(),
                "validation_profile": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "dataset_policy": hashlib.sha256(policy.read_bytes()).hexdigest(),
            }
            cfg = self._write_workflow_with_sources(
                root, [plain_stage("teacher_baseline", "artifacts/tb.txt")], scope, profile, policy)

            controller = RunController.initialize(cfg, root / "run")
            record = controller.state["validation_contract"]

            snapshot_dir = controller.run_dir / "inputs" / "contract_sources"
            for key in ("distillation_scope", "validation_profile", "dataset_policy"):
                snapshot_path = snapshot_dir / f"{key}.yaml"
                self.assertTrue(snapshot_path.exists(), f"missing run-bound snapshot: {key}")
                snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                self.assertEqual(snapshot_hash, original_hashes[key])
                self.assertEqual(record["source_files"][key]["sha256"], original_hashes[key])
                # provenance references the run-local snapshot, not only the external source
                self.assertIn(str(controller.run_dir), record["source_files"][key]["snapshot"])

    def test_teacher_baseline_consumes_the_automatically_established_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope, profile, policy = self._write_configs(root, domain=TEACHER_DOMAIN)
            cfg = self._write_workflow_with_sources(
                root, [plain_stage("teacher_baseline", "artifacts/tb.txt")], scope, profile, policy)

            controller = RunController.initialize(cfg, root / "run")
            contract_path = controller.run_dir / "validation_contract.json"

            report = TeacherBaselineDomainCrossCheckTests._report(self, root, TEACHER_DOMAIN)
            validate_teacher_baseline_report(report, validation_contract_path=contract_path)

            mismatched_report = TeacherBaselineDomainCrossCheckTests._report(
                self, root, OTHER_TEACHER_DOMAIN)
            with self.assertRaisesRegex(ValueError, "does not match the run's locked"):
                validate_teacher_baseline_report(
                    mismatched_report, validation_contract_path=contract_path)

    def test_disagreeing_sources_fail_initialization_atomically_with_no_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope, profile, policy = self._write_configs(root, profile_domain=OTHER_TEACHER_DOMAIN)
            cfg = self._write_workflow_with_sources(
                root, [plain_stage("teacher_baseline", "artifacts/tb.txt")], scope, profile, policy)

            run_dir = root / "run"
            with self.assertRaisesRegex(ValueError, "does not match the authoritative"):
                RunController.initialize(cfg, run_dir)

            self.assertFalse(run_dir.exists(), "a failed initialization must leave no run behind")
            leftovers = [p for p in root.iterdir()
                        if p.name.startswith(".run.init-") or p.name.startswith(".run.")]
            self.assertEqual(leftovers, [], "no temporary init directory may survive a failure")

    def test_mutating_original_source_files_after_init_cannot_change_the_frozen_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope, profile, policy = self._write_configs(root, domain=TEACHER_DOMAIN)
            cfg = self._write_workflow_with_sources(
                root, [plain_stage("teacher_baseline", "artifacts/tb.txt")], scope, profile, policy)

            controller = RunController.initialize(cfg, root / "run")
            original_sha256 = controller.state["validation_contract"]["contract_sha256"]

            # Mutate the ORIGINAL external distillation_scope.yaml after initialization.
            scope.write_text(yaml.safe_dump({"deployment_domain": OTHER_TEACHER_DOMAIN}))

            reloaded = RunController(controller.run_dir)
            self.assertEqual(reloaded.state["validation_contract"]["contract_sha256"],
                             original_sha256,
                             "mutating the external source file must not change the run's "
                             "frozen contract")
            self.assertEqual(
                reloaded.state["validation_contract"]["components"]
                ["teacher_applicability_domain"]["value"],
                TEACHER_DOMAIN)

    def test_workflow_without_validation_contract_sources_is_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = write_workflow(root, [plain_stage("teacher_baseline", "artifacts/tb.txt")])
            controller = RunController.initialize(cfg, root / "run")
            self.assertIsNone(controller.state["validation_contract"])
            self.assertFalse((controller.run_dir / "validation_contract.json").exists())


if __name__ == "__main__":
    unittest.main()
