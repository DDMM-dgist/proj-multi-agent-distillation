"""FE-050 -- a ``distinct_evidence_artifact`` recovery restores the return stage's declared outputs
BYTE-IDENTICALLY from the frozen iteration baseline before the corrective dispatch's missing-outputs
check, and registers the corrective's own distinct evidence artifact as ADDITIVE return-stage
evidence so ``verify_recovery_execution`` accepts the recovery as materialized -- with NO re-run of
the return stage's route action (no Teacher inference).

Background (the live eng5 blocker, second of two): after an approved recovery,
``RunController.start_iteration`` calls ``invalidate_from(return_stage, include_stage=True)``, which
QUARANTINES the return stage's declared outputs into ``run_dir/stale/``. A distinct-evidence
corrective action dispatches an executor DISTINCT from the stage's route action -- it produces NEW
evidence and never re-emits those declared outputs -- so
``_dispatch_recovery_corrective_action``'s declared-outputs check fail-closed with MISSING_OUTPUTS
(campaign exit 2) even though the recovery was proceeding exactly as approved. FE-050 restores the
quarantined declared outputs byte-identically (verified sha256-for-sha256, sourced only from the
quarantined baseline copy -- never re-derived) and carries the corrective artifact additively.

Part 1 drives the REAL production path (propose_recovery -> approve_recovery -> start_iteration ->
_dispatch_recovery_corrective_action -> dispatch.authorize_and_execute -> complete_external_stage ->
verify_recovery_execution); only the corrective executor is a fixture. Part 2 unit-tests the two
byte-identical/fail-closed restore helpers directly.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtimes.pydantic_ai import cli
from runtimes.pydantic_ai.cli import (
    _RecoveryBaselineRestoreError, _corrective_evidence_artifact_path,
    _dispatch_recovery_corrective_action, _find_quarantined_baseline,
    _restore_return_stage_baseline_outputs,
)
from runtimes.pydantic_ai.dispatch import ActionDescriptor
from runtimes.pydantic_ai.executors import build_executor_registry
from runtimes.pydantic_ai.models import EvidenceReference
from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
from runtimes.pydantic_ai.root_cause import (
    RootCauseClassification, validate_root_cause_classification,
)
from workflow.controller import RunController
from workflow.integrity import sha256_file

from test_full_lifecycle_integration import FixtureHelpers, GATE_CRITERION

ROOT = Path(__file__).resolve().parent.parent

ROUTE_ACTION = "label_dataset"          # the return stage's own deterministic route action
CORRECTIVE_ACTION = "validate_species_mapping_consistency"  # DISTINCT evidence-exposure action


# --- Part 1: end-to-end distinct-evidence recovery through the real dispatch path ---------------

class _Fe050Fixture(FixtureHelpers):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _staged_controller(self):
        stages = [
            {"name": "prepare", "command": None, "outputs": ["artifacts/baseline.json"],
             "gate": {"criteria": [GATE_CRITERION]}},
            {"name": "labeling", "command": None, "outputs": ["artifacts/labeled.json"],
             "pydantic_ai": {"role": "data-curator", "action": ROUTE_ACTION},
             "gate": {"criteria": [GATE_CRITERION]}},
        ]
        return self._init_controller(
            self.root, stages=stages,
            recovery_capability_roster={"data_repair": "data-curator",
                                        "orchestration": "orchestrator"})

    def _drive_to_started_recovery(self, controller, *, corrective_parameters):
        baseline = controller.run_dir / "artifacts/baseline.json"
        self._write_json(baseline, {"role": "baseline"})
        controller.complete_external_stage("prepare", [baseline])
        self._gate(controller, "prepare", "PASS")

        labeled = controller.run_dir / "artifacts/labeled.json"
        self._write_json(labeled, {"role": "labeled", "n_frames": 11, "revision": 1})
        controller.complete_external_stage("labeling", [labeled])
        self.labeled_baseline_sha = sha256_file(labeled)
        self._gate(controller, "labeling", "REVISE")

        classification = RootCauseClassification(
            run_id="fixture-run", stage="labeling", failure_category="dataset_coverage",
            evidence_refs=[EvidenceReference(role="labeled", path=str(labeled),
                                             integrity={"sha256": sha256_file(labeled)})],
            evidence_summary="labeling gate needs species-mapping consistency evidence exposed",
            confidence=0.9, recommended_recovery_target="labeling",
            recommended_next_action="expose and cross-check the recorded species mapping")
        validate_root_cause_classification(
            classification, available_artifacts=[str(labeled)],
            valid_recovery_targets=[s["name"] for s in controller.state["stages"]])
        diagnosis_path = self._write_json(controller.run_dir / "diagnosis.json",
                                          json.loads(classification.model_dump_json()))
        draft = build_recovery_plan_draft(
            classification,
            proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"},
            failed_stage="labeling", capability="data_repair", return_stage="labeling",
            proposed_changes=[{"type": "evidence_exposure"}],
            labeling={"teacher_relabel": False, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["labeling"]},
            diagnosis_artifact_path=str(diagnosis_path),
            diagnosis_artifact_sha256=sha256_file(diagnosis_path),
            extra_recovery_context={
                "corrective_action": {"action_type": CORRECTIVE_ACTION,
                                      "parameters": corrective_parameters}})
        plan_path = self._write_json(controller.run_dir / "plan.json", draft.to_plan_json())
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})
        controller.start_iteration()

    def _corrective_registry(self, executor):
        registry = build_executor_registry()
        registry[CORRECTIVE_ACTION] = ActionDescriptor(
            action_type=CORRECTIVE_ACTION, role="data-curator",
            approval_boundary=None, executor=executor)
        return registry

    def _dispatch(self, controller, registry):
        c = RunController(controller.run_dir)
        iteration = c.state["iterations"][-1]
        trigger = iteration["trigger"]
        recovery = next(r for r in c.state["recoveries"] if r["id"] == trigger["recovery_id"])
        corrective = recovery["plan"]["recovery_context"]["corrective_action"]
        return c, _dispatch_recovery_corrective_action(
            c, trigger, recovery, corrective, registry=registry)


class Fe050DistinctEvidenceRestoreTests(_Fe050Fixture):
    def _species_report_executor(self):
        run_dir = self.root / "run"

        def executor(proposal):
            out = run_dir / "artifacts/species_mapping.report.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"ok": True, "attested": True,
                                       "species_to_type_index_map": {"O": 0, "Si": 1}}))
            return {"path": str(out), "sha256": sha256_file(out), "metrics": {"ok": True}}

        return executor

    def test_1_recovery_classified_distinct_evidence_artifact(self):
        c = self._staged_controller()
        self._drive_to_started_recovery(
            c, corrective_parameters={"manifest_path": "m.json",
                                      "out_path": "artifacts/species_mapping.report.json"})
        recovery = c.state["recoveries"][-1]
        self.assertEqual(recovery["materialization_transition"], "distinct_evidence_artifact")

    def test_2_quarantine_makes_declared_output_missing_then_restore_recovers_it(self):
        c = self._staged_controller()
        self._drive_to_started_recovery(
            c, corrective_parameters={"out_path": "artifacts/species_mapping.report.json"})
        labeled = c.run_dir / "artifacts/labeled.json"
        # start_iteration quarantined the declared output -> it is genuinely gone (the eng5 blocker).
        self.assertFalse(labeled.exists())
        # A byte-identical copy is preserved under stale/ (the frozen baseline restore source).
        self.assertIsNotNone(_find_quarantined_baseline(
            c.run_dir / "stale", "labeling", "labeled.json", self.labeled_baseline_sha))

        c2, result = self._dispatch(c, self._corrective_registry(self._species_report_executor()))
        self.assertIsNone(result, getattr(result, "message", None))
        # The declared output is restored byte-identically (no route re-run).
        self.assertTrue(labeled.exists())
        self.assertEqual(sha256_file(labeled), self.labeled_baseline_sha)
        self.assertEqual(c2.stage("labeling")["status"], "completed")

    def test_3_corrective_registered_additively_and_recovery_verifies(self):
        c = self._staged_controller()
        self._drive_to_started_recovery(
            c, corrective_parameters={"out_path": "artifacts/species_mapping.report.json"})
        c2, result = self._dispatch(c, self._corrective_registry(self._species_report_executor()))
        self.assertIsNone(result, getattr(result, "message", None))

        c3 = RunController(c.run_dir)
        labeling_paths = {Path(a["path"]).name for a in c3.stage_artifacts("labeling")}
        # BOTH the restored declared output AND the distinct corrective evidence are registered.
        self.assertEqual(labeling_paths, {"labeled.json", "species_mapping.report.json"})

        # verify_recovery_execution accepts: the return-stage artifact set differs from the frozen
        # baseline (the additive species report), even though labeled.json is byte-identical.
        report, missing = cli._assemble_recovery_execution_report(c3)
        self.assertIsNotNone(report, missing)
        report_path = (c3.run_dir / "recovery" /
                       f"recovery-{report['recovery_id']:03d}.execution.report.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        c3.verify_recovery_execution(report_path)  # must not raise
        self.assertEqual(c3.state["iterations"][-1]["recovery_execution"]["status"], "verified")

    def test_4_fail_closed_when_no_quarantined_baseline_to_restore(self):
        c = self._staged_controller()
        self._drive_to_started_recovery(
            c, corrective_parameters={"out_path": "artifacts/species_mapping.report.json"})
        # Destroy the quarantined baseline so no byte-identical restore source exists.
        import shutil
        shutil.rmtree(c.run_dir / "stale")
        c2, result = self._dispatch(c, self._corrective_registry(self._species_report_executor()))
        self.assertIsNotNone(result)
        self.assertEqual(result.outcome, cli.CAMPAIGN_FAILED)
        self.assertIn("byte-identical", result.message)
        self.assertNotEqual(c2.stage("labeling")["status"], "completed")

    def test_5_fail_closed_when_corrective_produces_no_evidence_artifact(self):
        c = self._staged_controller()
        # No out_path in the corrective params AND the executor returns no artifact -> nothing
        # registrable as the distinct evidence change.
        self._drive_to_started_recovery(c, corrective_parameters={"manifest_path": "m.json"})

        def no_artifact_executor(proposal):
            return {"metrics": {"ok": True}}

        c2, result = self._dispatch(c, self._corrective_registry(no_artifact_executor))
        self.assertIsNotNone(result)
        self.assertEqual(result.outcome, cli.CAMPAIGN_FAILED)
        self.assertIn("distinct evidence artifact", result.message)

    def test_7_corrective_can_read_restored_declared_output_at_dispatch(self):
        # FE-051 regression: a distinct-evidence corrective whose executor READS the return stage's
        # own declared output (as validate_species_mapping_consistency reads the teacher labeling
        # manifest) only works if the byte-identical restore runs BEFORE dispatch. start_iteration
        # quarantined labeled.json; this executor reads it at dispatch time and would raise
        # FileNotFoundError under the old restore-after-dispatch ordering.
        c = self._staged_controller()
        self._drive_to_started_recovery(
            c, corrective_parameters={"read_path": "artifacts/labeled.json",
                                      "out_path": "artifacts/species_mapping.report.json"})
        self.assertFalse((c.run_dir / "artifacts/labeled.json").exists())  # quarantined

        run_dir = c.run_dir
        observed = {}

        def reading_executor(proposal):
            params = proposal["parameters"] if isinstance(proposal, dict) else proposal.parameters
            read_path = run_dir / params["read_path"]
            observed["read_sha"] = sha256_file(read_path)  # raises if not restored first
            out = run_dir / params["out_path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"ok": True, "read_sha256": observed["read_sha"]}))
            return {"path": str(out), "sha256": sha256_file(out), "metrics": {"ok": True}}

        c2, result = self._dispatch(c, self._corrective_registry(reading_executor))
        self.assertIsNone(result, getattr(result, "message", None))
        # The executor observed the restored declared output byte-identically at dispatch time.
        self.assertEqual(observed["read_sha"], self.labeled_baseline_sha)
        self.assertEqual(c2.stage("labeling")["status"], "completed")

    def test_6_non_distinct_recovery_does_not_restore(self):
        # A recovery NOT classified distinct_evidence_artifact must take the original path: no
        # baseline restore, so if its corrective does not itself reproduce the declared output the
        # existing MISSING_OUTPUTS contract still fail-closes (FE-050 narrows to distinct-evidence).
        c = self._staged_controller()
        self._drive_to_started_recovery(
            c, corrective_parameters={"out_path": "artifacts/species_mapping.report.json"})
        c_mut = RunController(c.run_dir)
        c_mut.state["recoveries"][-1]["materialization_transition"] = "input_supersession_replan"
        c_mut.save()

        c2, result = self._dispatch(c_mut, self._corrective_registry(self._species_report_executor()))
        # No restore -> labeled.json stays quarantined -> the declared-outputs contract fail-closes.
        self.assertIsNotNone(result)
        self.assertEqual(result.outcome, cli.CAMPAIGN_FAILED)
        self.assertIn("declared outputs are still missing", result.message)
        self.assertFalse((c.run_dir / "artifacts/labeled.json").exists())


# --- Part 2: byte-identical / fail-closed restore helpers (unit) -------------------------------

class _FakeController:
    def __init__(self, run_dir, outputs):
        self.run_dir = run_dir
        self._outputs = outputs

    def stage(self, name):
        return {"outputs": self._outputs}


class Fe050RestoreHelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name)

    def _quarantine(self, stage, name, content):
        dest = self.run_dir / "stale" / "20260826T000000.000000Z" / stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        return dest, sha256_file(dest)

    def test_restore_recovers_byte_identical_from_stale(self):
        _, sha = self._quarantine("labeling", "labeled.json", '{"role":"labeled"}')
        c = _FakeController(self.run_dir, ["artifacts/labeled.json"])
        iteration = {"baseline_artifacts": [
            {"stage": "labeling",
             "path": str((self.run_dir / "artifacts/labeled.json").resolve()), "sha256": sha}]}
        restored = _restore_return_stage_baseline_outputs(c, "labeling", iteration)
        dest = self.run_dir / "artifacts/labeled.json"
        self.assertEqual(restored, [dest.resolve()])
        self.assertTrue(dest.exists())
        self.assertEqual(sha256_file(dest), sha)

    def test_restore_is_idempotent_when_output_already_present(self):
        dest = self.run_dir / "artifacts/labeled.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"role":"labeled"}')
        sha = sha256_file(dest)
        c = _FakeController(self.run_dir, ["artifacts/labeled.json"])
        iteration = {"baseline_artifacts": [
            {"stage": "labeling", "path": str(dest.resolve()), "sha256": sha}]}
        _restore_return_stage_baseline_outputs(c, "labeling", iteration)
        self.assertEqual(sha256_file(dest), sha)  # untouched

    def test_restore_fails_closed_without_baseline_record(self):
        self._quarantine("labeling", "labeled.json", '{"role":"labeled"}')
        c = _FakeController(self.run_dir, ["artifacts/labeled.json"])
        with self.assertRaises(_RecoveryBaselineRestoreError):
            _restore_return_stage_baseline_outputs(c, "labeling", {"baseline_artifacts": []})

    def test_restore_fails_closed_without_byte_identical_quarantine(self):
        c = _FakeController(self.run_dir, ["artifacts/labeled.json"])
        iteration = {"baseline_artifacts": [
            {"stage": "labeling",
             "path": str((self.run_dir / "artifacts/labeled.json").resolve()),
             "sha256": "a" * 64}]}  # no quarantined file matches this sha256
        with self.assertRaises(_RecoveryBaselineRestoreError):
            _restore_return_stage_baseline_outputs(c, "labeling", iteration)

    def test_find_quarantined_requires_matching_sha(self):
        self._quarantine("labeling", "labeled.json", '{"role":"labeled"}')
        stale = self.run_dir / "stale"
        self.assertIsNone(_find_quarantined_baseline(stale, "labeling", "labeled.json", "b" * 64))
        self.assertIsNone(_find_quarantined_baseline(stale, "other_stage", "labeled.json",
                                                     sha256_file(next(stale.rglob("labeled.json")))))

    def test_corrective_artifact_prefers_outcome_then_out_path(self):
        art = self.run_dir / "artifacts/species.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text("{}")
        c = _FakeController(self.run_dir, [])

        class _Outcome:
            artifact = {"path": str(art), "sha256": sha256_file(art)}

        self.assertEqual(_corrective_evidence_artifact_path(c, _Outcome(), {}), art.resolve())

        class _NoArtifact:
            artifact = None

        self.assertEqual(
            _corrective_evidence_artifact_path(c, _NoArtifact(),
                                               {"out_path": "artifacts/species.json"}),
            art.resolve())
        self.assertIsNone(_corrective_evidence_artifact_path(c, _NoArtifact(), {}))
        self.assertIsNone(_corrective_evidence_artifact_path(
            c, _NoArtifact(), {"out_path": "artifacts/missing.json"}))


if __name__ == "__main__":
    unittest.main()
