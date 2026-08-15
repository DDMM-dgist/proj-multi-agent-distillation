"""Proves the thin campaign observability layer (runtimes.pydantic_ai.events.CampaignEventEmitter,
threaded through cli.run_campaign/run_production_stage/recovery helpers) emits the minimum event
vocabulary at the right points, never fabricates progress, never leaks raw provider content, is
UTF-8-safe under an ASCII locale, and persists durably/in order across separate process-like
invocations -- WITHOUT altering any Controller/Gate/recovery semantics (every assertion about
stage/gate/recovery outcome here matches what the existing run-campaign tests already assert;
this file only adds assertions about the parallel event log).
"""
from __future__ import annotations

import json
import locale
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


def _read_events(run_dir: Path) -> list:
    path = run_dir / "campaign_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dataset(path: Path, n_frames: int, offset: int) -> Path:
    frames = []
    for i in range(n_frames):
        atoms = Atoms("Cu", positions=[[i + offset, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.info["structure_id"] = f"s{offset}-{i}"
        atoms.info["parent_structure_id"] = f"seed-pool:{900 + offset + i}"
        frames.append(atoms)
    write(str(path), frames)
    return path


def _manifest_stage(name: str, dataset_path: Path, manifest_rel: str) -> dict:
    pydantic_ai = {
        "role": "data-curator", "action": "build_dataset_manifest",
        "idempotency_key": f"events-test:{name}:001",
        "parameters": {"dataset": str(dataset_path),
                      "manifest_path": f"{{artifacts_dir}}/{manifest_rel.split('/')[-1]}"},
    }
    return {"name": name, "command": None, "outputs": [manifest_rel],
           "gate": {"criteria": ["dataset manifest is complete"]}, "pydantic_ai": pydantic_ai}


def _two_stage_workflow(root: Path) -> Path:
    dataset_a = _dataset(root / "dataset_a.extxyz", 1, 0)
    dataset_b = _dataset(root / "dataset_b.extxyz", 1, 100)
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "events-test-campaign",
        "inputs": [str(dataset_a), str(dataset_b)],
        "stages": [
            _manifest_stage("stage_a", dataset_a, "artifacts/stage_a_manifest.json"),
            _manifest_stage("stage_b", dataset_b, "artifacts/stage_b_manifest.json"),
        ],
    }))
    return workflow


def _approval_gated_workflow(root: Path) -> Path:
    # train_committee/evaluate_heldout_fidelity are the two actions actually wired to an
    # APPROVAL_GATED_ACTIONS boundary (runtimes.pydantic_ai.actions.APPROVAL_GATED_ACTIONS) --
    # unlike a workflow-declared pydantic_ai.approval_boundary key (which the dispatch enforcement
    # path does not read), this is the real production approval-pause boundary.
    dataset = root / "train.extxyz"
    frames = []
    for i in range(3):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"s{i}"
        a.info["parent_structure_id"] = f"p{i}"
        frames.append(a)
    write(str(dataset), frames)
    student_cfg = root / "student.yaml"
    student_cfg.write_text(
        "kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n"
        "predict:\n  factory: adapters.mock_model.MockCheckpointCalculator\n"
        "  checkpoint_arg: checkpoint\n  kwargs: {}\n")
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "events-test-approval-campaign",
        "inputs": [str(student_cfg), str(dataset)],
        "stages": [
            {
                "name": "stage_a", "command": None,
                "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
                "pydantic_ai": {
                    "role": "ml-trainer", "action": "train_committee",
                    "approval_boundary": "costly_training",
                    "idempotency_key": "events-test-approval-campaign-stage-a-001",
                    "parameters": {
                        "student_config": str(student_cfg), "dataset": str(dataset),
                        "output_dir": "{artifacts_dir}/committee",
                        "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
                    },
                },
                "gate": {"criteria": ["committee manifest is complete"]},
            },
            {
                "name": "stage_b", "command": None,
                "outputs": ["artifacts/heldout_labeled.extxyz", "artifacts/heldout_report.json"],
                "pydantic_ai": {
                    "role": "ml-trainer", "action": "evaluate_heldout_fidelity",
                    "approval_boundary": "costly_training",
                    "idempotency_key": "events-test-approval-campaign-stage-b-001",
                    "parameters": {
                        "student_config": str(student_cfg),
                        "committee_manifest": "{artifacts_dir}/student_committee.manifest.json",
                        "frames_path": str(dataset),
                        "labeled_output": "{artifacts_dir}/heldout_labeled.extxyz",
                        "report_path": "{artifacts_dir}/heldout_report.json",
                    },
                },
                "gate": {"criteria": ["fidelity report is complete"]},
            },
        ],
    }))
    return workflow


def _revise_vote(path: Path, lens: str, criteria: list) -> Path:
    path.write_text(json.dumps({
        "review_lens": lens, "verdict": "REVISE",
        "criteria_checked": [{"criterion": c, "value_read": "coverage gap", "ok": False}
                             for c in criteria],
        "rationale": "dataset does not cover the required composition — see manifest",
        "required_fix": "rebuild the manifest from a corrected dataset",
    }))
    return path


class NormalForwardProgressionTests(unittest.TestCase):
    def test_two_stage_completion_emits_ordered_generic_events(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _two_stage_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_SUCCESS)

            events = _read_events(run_dir)
            names = [e["event"] for e in events]
            self.assertEqual(names[0], "campaign_started")
            self.assertEqual(names[-1], "campaign_completed")
            for required in ("stage_selected", "role_invocation_started",
                             "role_invocation_completed", "executor_started",
                             "executor_completed", "artifact_registered", "judging_started",
                             "gate_recorded"):
                self.assertIn(required, names, f"missing {required} in {names}")
            # stage_a's whole lifecycle happens before stage_b is ever selected -- forward
            # progression, not an interleaved/out-of-order log.
            stage_a_gate = names.index("gate_recorded")
            stage_b_selected = [i for i, e in enumerate(events)
                                if e["event"] == "stage_selected" and e.get("stage") == "stage_b"]
            self.assertTrue(stage_b_selected)
            self.assertLess(stage_a_gate, stage_b_selected[0])
            gate_events = [e for e in events if e["event"] == "gate_recorded"]
            self.assertEqual([e["detail"]["decision"] for e in gate_events], ["PASS", "PASS"])
            terminal = events[-1]
            self.assertEqual(terminal["detail"]["outcome"], "COMPLETED")

    def test_human_readable_progress_printed_by_default(self):
        # Requirement 1: human-readable progress must be emitted BY DEFAULT (no --quiet, no
        # --json-events) from run-campaign -- this drives the CLI exactly like a real operator
        # would, capturing real stdout rather than calling the emitter directly.
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _two_stage_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                                 "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            printed = buf.getvalue()
            # Human-readable rendering, not a JSON blob: no line should parse as a JSON object.
            self.assertIn("campaign_started", printed)
            self.assertIn("stage=stage_a", printed)
            for line in printed.splitlines():
                if line.strip():
                    with self.assertRaises(Exception):
                        json.loads(line)


class ApprovalPauseTests(unittest.TestCase):
    def test_approval_required_pauses_and_emits_approval_events(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _approval_gated_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            events = _read_events(run_dir)
            names = [e["event"] for e in events]
            self.assertIn("approval_required", names)
            self.assertEqual(names[-1], "campaign_paused")
            self.assertEqual(events[-1]["detail"]["outcome"], "WAITING_FOR_HUMAN_APPROVAL")
            # Nothing dispatched before the approval boundary was cleared.
            self.assertNotIn("executor_completed", names)

            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training", "--note", "t"]),
                             cli.EXIT_SUCCESS)
            approval_events = _read_events(run_dir)
            self.assertIn("approval_granted", [e["event"] for e in approval_events])

    def test_resume_after_approval_appends_ordered_events_across_two_invocations(self):
        # Requirement: process resume with ordered durable events across separate invocations.
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _approval_gated_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            self.assertEqual(cli.main(["run-campaign", "--runtime", "mock", "--run-dir",
                                       str(run_dir), "--auto-mock-judges"]),
                             cli.EXIT_APPROVAL_REQUIRED)
            first_pass_events = _read_events(run_dir)
            self.assertEqual(first_pass_events[0]["event"], "campaign_started")

            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training", "--note", "t"]),
                             cli.EXIT_SUCCESS)

            # A second, entirely separate CLI process invocation.
            self.assertEqual(cli.main(["run-campaign", "--runtime", "mock", "--run-dir",
                                       str(run_dir), "--auto-mock-judges"]), cli.EXIT_SUCCESS)
            all_events = _read_events(run_dir)

            # Every event the first invocation wrote is still there, unmodified, at the front --
            # a resume APPENDS, it never rewrites/truncates the durable log. (One "approval_granted"
            # event lands in between, from the intervening `approve` CLI invocation.)
            self.assertEqual(all_events[:len(first_pass_events)], first_pass_events)
            after_first_pass = all_events[len(first_pass_events):]
            self.assertGreater(len(after_first_pass), 0)
            self.assertEqual(after_first_pass[0]["event"], "approval_granted")
            second_pass_events = after_first_pass[1:]
            self.assertGreater(len(second_pass_events), 0)
            self.assertEqual(second_pass_events[0]["event"], "campaign_resumed")
            self.assertEqual(all_events[-1]["event"], "campaign_completed")
            # ts is monotonic non-decreasing across the whole durable log (real ordering, not
            # just append order) -- ISO-8601 timestamps sort lexicographically.
            timestamps = [e["ts"] for e in all_events]
            self.assertEqual(timestamps, sorted(timestamps))


class RecoveryPathTests(unittest.TestCase):
    def _setup_gate_revise(self, root: Path):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        dataset_a = _dataset(root / "dataset_a.extxyz", 1, 0)
        dataset_b = _dataset(root / "dataset_b.extxyz", 1, 100)
        workflow = root / "workflow.yaml"
        workflow.write_text(yaml.safe_dump({
            "run_id": "events-recovery-test",
            "inputs": [str(dataset_a), str(dataset_b)],
            "stages": [
                _manifest_stage("stage_a", dataset_a, "artifacts/stage_a_manifest.json"),
                _manifest_stage("stage_b", dataset_b, "artifacts/stage_b_manifest.json"),
            ],
        }))
        run_dir = root / "run"
        RunController.initialize(workflow, run_dir)
        c = RunController(run_dir)

        result_a = cli.run_production_stage(c, "stage_a", runtime="mock", repo_root=str(ROOT),
                                            auto_mock_judges=True)
        self.assertEqual(result_a.reason, "SUCCESS", result_a.message)
        c = RunController(run_dir)

        lenses = [lens["id"] for lens in c.stage("stage_b")["gate_review_lenses"]]
        criteria = c.stage("stage_b")["gate_criteria"]
        vote_paths = [_revise_vote(root / f"revise-{i}.json", lens, criteria)
                     for i, lens in enumerate(lenses, 1)]
        result_b = cli.run_production_stage(
            c, "stage_b", runtime="mock", repo_root=str(ROOT),
            mock_judge_response=[str(p) for p in vote_paths])
        self.assertEqual(result_b.reason, "GATE_REVISE", result_b.message)
        return run_dir

    def test_gate_revise_records_recovery_required_and_gate_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self._setup_gate_revise(root)
            events = _read_events(run_dir)
            gate_events = [e for e in events if e["event"] == "gate_recorded"
                          and e.get("stage") == "stage_b"]
            self.assertEqual(len(gate_events), 1)
            self.assertEqual(gate_events[0]["detail"]["decision"], "REVISE")

    def test_recovery_started_and_proposed_emitted_before_human_approval(self):
        import hashlib
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from workflow.controller import RunController
        from workflow.integrity import sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self._setup_gate_revise(root)
            c = RunController(run_dir)
            self.assertEqual(c.state["pending_recovery"]["status"], "required")
            stage_b_manifest = str((run_dir / "artifacts" / "stage_b_manifest.json").resolve())

            classification_payload = {
                "run_id": c.state["run_id"], "stage": "stage_b",
                "failure_category": "dataset_coverage",
                "evidence_refs": [{"role": "data-curator", "path": stage_b_manifest,
                                  "integrity": {"sha256": sha256_file(Path(stage_b_manifest))}}],
                "evidence_summary": "stage_b's manifest lacks coverage for the required composition",
                "confidence": 0.75, "recommended_recovery_target": "stage_b",
                "recommended_next_action": "rebuild the stage_b manifest from a corrected dataset",
            }
            classification = RootCauseClassification(**classification_payload)
            diagnosis_sha256 = hashlib.sha256(
                (classification.model_dump_json(indent=2) + "\n").encode()).hexdigest()
            analyst_response = root / "mock_analyst_response.json"
            analyst_response.write_text(json.dumps(classification_payload))

            proposal_payload = {
                "run_id": c.state["run_id"], "failed_stage": "stage_b",
                "diagnosis_artifact_sha256": diagnosis_sha256,
                "capability": "data_repair", "return_stage": "stage_b",
                "proposed_changes": [{"type": "rebuild_manifest_from_corrected_dataset"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["stage_b"]},
                "rationale": "rebuild stage_b's manifest from a corrected dataset",
                "corrective_action": {
                    "action_type": "build_dataset_manifest",
                    "parameters": {"dataset": str((root / "dataset_a.extxyz").resolve()),
                                  "manifest_path": stage_b_manifest},
                },
            }
            orchestrator_response = root / "mock_orchestrator_response.json"
            orchestrator_response.write_text(json.dumps(proposal_payload))

            result = cli.run_campaign(
                c, runtime="mock", repo_root=str(ROOT),
                mock_analyst_response=str(analyst_response),
                mock_orchestrator_response=str(orchestrator_response), max_iterations=20)
            self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL,
                             result.message)

            events = _read_events(run_dir)
            names = [e["event"] for e in events]
            self.assertIn("recovery_started", names)
            self.assertIn("recovery_proposed", names)
            analyst_started = [e for e in events if e["event"] == "role_invocation_started"
                               and e.get("role") == "analyst"]
            orchestrator_started = [e for e in events if e["event"] == "role_invocation_started"
                                    and e.get("role") == "orchestrator"]
            self.assertTrue(analyst_started)
            self.assertTrue(orchestrator_started)
            self.assertEqual(names[-1], "campaign_paused")


class TerminalOutcomeTests(unittest.TestCase):
    def test_campaign_failed_emits_terminal_failed_event(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.dispatch import ActionDescriptor, ExternalActionPending
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController

        # Reuses the recovery lifecycle up through an approved recovery, then substitutes a
        # corrective-action executor whose output never satisfies the stage's declared outputs
        # -- a deterministic FAILED outcome, network-free.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recov = RecoveryPathTests()
            recov.assertEqual = self.assertEqual  # bind unittest assertion to this instance
            run_dir = recov._setup_gate_revise(root)

            import hashlib
            from runtimes.pydantic_ai.root_cause import RootCauseClassification
            from workflow.integrity import sha256_file
            c = RunController(run_dir)
            stage_b_manifest = str((run_dir / "artifacts" / "stage_b_manifest.json").resolve())
            classification_payload = {
                "run_id": c.state["run_id"], "stage": "stage_b",
                "failure_category": "dataset_coverage",
                "evidence_refs": [{"role": "data-curator", "path": stage_b_manifest,
                                  "integrity": {"sha256": sha256_file(Path(stage_b_manifest))}}],
                "evidence_summary": "coverage gap",
                "confidence": 0.75, "recommended_recovery_target": "stage_b",
                "recommended_next_action": "rebuild the manifest",
            }
            classification = RootCauseClassification(**classification_payload)
            diagnosis_sha256 = hashlib.sha256(
                (classification.model_dump_json(indent=2) + "\n").encode()).hexdigest()
            analyst_response = root / "mock_analyst_response.json"
            analyst_response.write_text(json.dumps(classification_payload))
            proposal_payload = {
                "run_id": c.state["run_id"], "failed_stage": "stage_b",
                "diagnosis_artifact_sha256": diagnosis_sha256,
                "capability": "data_repair", "return_stage": "stage_b",
                "proposed_changes": [{"type": "rebuild_manifest_from_corrected_dataset"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["stage_b"]},
                "rationale": "rebuild stage_b's manifest",
                "corrective_action": {
                    "action_type": "build_dataset_manifest",
                    "parameters": {"dataset": str((root / "dataset_a.extxyz").resolve()),
                                  "manifest_path": stage_b_manifest},
                },
            }
            orchestrator_response = root / "mock_orchestrator_response.json"
            orchestrator_response.write_text(json.dumps(proposal_payload))
            result = cli.run_campaign(
                c, runtime="mock", repo_root=str(ROOT),
                mock_analyst_response=str(analyst_response),
                mock_orchestrator_response=str(orchestrator_response), max_iterations=20)
            self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL)
            c = RunController(run_dir)
            c.approve_recovery("Dr. Lee", note="approved")

            def _noop_executor(_proposal):
                return {"path": None, "manifest": {}, "sha256": ""}

            registry = build_executor_registry()
            registry["build_dataset_manifest"] = ActionDescriptor(
                action_type="build_dataset_manifest", role="data-curator",
                executor=_noop_executor)

            c = RunController(run_dir)
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20,
                                      recovery_action_registry=registry)
            self.assertEqual(result.outcome, cli.CAMPAIGN_FAILED, result.message)

            events = _read_events(run_dir)
            self.assertEqual(events[-1]["event"], "campaign_failed")
            self.assertEqual(events[-1]["detail"]["outcome"], "FAILED")
            self.assertIn("declared outputs are still missing", result.message)
            # The dispatch itself nominally succeeded (the noop executor raised nothing) -- the
            # missing-outputs check is a separate, downstream verification, so the event that
            # precedes the terminal failure is executor_completed, not executor_failed.
            stage_b_events = [e["event"] for e in events if e.get("stage") == "stage_b"]
            self.assertIn("executor_completed", stage_b_events)


class ExecutorProgressAndPendingTests(unittest.TestCase):
    def _approved_recovery_run_dir(self, root: Path):
        import hashlib
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from workflow.controller import RunController
        from workflow.integrity import sha256_file

        recov = RecoveryPathTests()
        recov.assertEqual = self.assertEqual
        run_dir = recov._setup_gate_revise(root)
        c = RunController(run_dir)
        stage_b_manifest = str((run_dir / "artifacts" / "stage_b_manifest.json").resolve())
        classification_payload = {
            "run_id": c.state["run_id"], "stage": "stage_b",
            "failure_category": "dataset_coverage",
            "evidence_refs": [{"role": "data-curator", "path": stage_b_manifest,
                              "integrity": {"sha256": sha256_file(Path(stage_b_manifest))}}],
            "evidence_summary": "coverage gap",
            "confidence": 0.75, "recommended_recovery_target": "stage_b",
            "recommended_next_action": "rebuild the manifest",
        }
        classification = RootCauseClassification(**classification_payload)
        diagnosis_sha256 = hashlib.sha256(
            (classification.model_dump_json(indent=2) + "\n").encode()).hexdigest()
        analyst_response = root / "mock_analyst_response.json"
        analyst_response.write_text(json.dumps(classification_payload))
        proposal_payload = {
            "run_id": c.state["run_id"], "failed_stage": "stage_b",
            "diagnosis_artifact_sha256": diagnosis_sha256,
            "capability": "data_repair", "return_stage": "stage_b",
            "proposed_changes": [{"type": "rebuild_manifest_from_corrected_dataset"}],
            "labeling": {"teacher_relabel": False, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["stage_b"]},
            "rationale": "rebuild stage_b's manifest",
            "corrective_action": {
                "action_type": "build_dataset_manifest",
                "parameters": {"dataset": str((root / "dataset_a.extxyz").resolve()),
                              "manifest_path": stage_b_manifest},
            },
        }
        orchestrator_response = root / "mock_orchestrator_response.json"
        orchestrator_response.write_text(json.dumps(proposal_payload))
        result = cli.run_campaign(
            c, runtime="mock", repo_root=str(ROOT),
            mock_analyst_response=str(analyst_response),
            mock_orchestrator_response=str(orchestrator_response), max_iterations=20)
        self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL)
        c = RunController(run_dir)
        c.approve_recovery("Dr. Lee", note="approved")
        return run_dir

    def test_executor_that_reports_progress_emits_executor_progress_events(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai import deterministic_executors as de
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController

        def _progress_reporting_executor(proposal, progress_cb=None):
            if progress_cb is not None:
                progress_cb({"phase": "started"})
            result = de.build_dataset_manifest(proposal)
            if progress_cb is not None:
                progress_cb({"phase": "completed"})
            return result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self._approved_recovery_run_dir(root)

            registry = build_executor_registry()
            registry["build_dataset_manifest"] = ActionDescriptor(
                action_type="build_dataset_manifest", role="data-curator",
                executor=_progress_reporting_executor)

            c = RunController(run_dir)
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20,
                                      recovery_action_registry=registry)
            self.assertEqual(result.outcome, cli.CAMPAIGN_COMPLETED, result.message)

            events = _read_events(run_dir)
            progress_events = [e for e in events if e["event"] == "executor_progress"]
            self.assertEqual([e["detail"] for e in progress_events],
                             [{"phase": "started"}, {"phase": "completed"}])
            # No stage in this run used an executor that declares progress_cb -- forward-stage
            # dispatch (stage_a, the ORIGINAL stage_b attempt) never fabricated a progress event.
            forward_stage_progress = [e for e in progress_events
                                      if e.get("action") != "build_dataset_manifest" or
                                      e.get("stage") not in ("stage_b",)]
            self.assertEqual(forward_stage_progress, [])

    def test_pending_executor_pauses_campaign_without_fabricating_completion(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.dispatch import ActionDescriptor, ExternalActionPending
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController

        def _pending_executor(_proposal):
            raise ExternalActionPending("corrective action queued externally")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self._approved_recovery_run_dir(root)

            registry = build_executor_registry()
            registry["build_dataset_manifest"] = ActionDescriptor(
                action_type="build_dataset_manifest", role="data-curator",
                executor=_pending_executor)

            c = RunController(run_dir)
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20,
                                      recovery_action_registry=registry)
            self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_RECOVERY_EVIDENCE,
                             result.message)

            events = _read_events(run_dir)
            names = [e["event"] for e in events]
            self.assertIn("executor_pending", names)
            self.assertNotIn("executor_completed",
                            [e["event"] for e in events if e.get("stage") == "stage_b" and
                             e is events[-2] if len(events) > 1])
            self.assertEqual(names[-1], "campaign_paused")
            self.assertEqual(events[-1]["detail"]["outcome"], "WAITING_FOR_RECOVERY_EVIDENCE")


class UnicodeAndRedactionTests(unittest.TestCase):
    def test_unicode_detail_survives_ascii_locale_console_write(self):
        # Requirement 4: unicode text must never depend on the shell's ASCII/default locale --
        # exercised directly against CampaignEventEmitter/_write_safely (the unit this concern
        # actually belongs to), not by trying to change the real process locale mid-test-suite.
        import io
        from runtimes.pydantic_ai.events import CampaignEventEmitter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            class _AsciiStream(io.TextIOWrapper):
                pass

            # A stream whose declared encoding is ASCII, mimicking a real terminal under
            # LANG=C/POSIX -- writing an em-dash to it directly would raise UnicodeEncodeError.
            raw_path = run_dir / "console.txt"
            stream = open(raw_path, "w", encoding="ascii", newline="\n")
            try:
                emitter = CampaignEventEmitter(run_dir, stream=stream)
                # "decision"/"verdict" are among the few detail keys _render_human echoes to the
                # console (see events._render_human) -- using one here exercises the real
                # ASCII-console code path an em-dash would otherwise crash, not just persistence.
                record = emitter.emit("gate_recorded", stage="stage_b",
                                      detail={"decision": "REVISE — coverage gap"})
                self.assertEqual(record["detail"]["decision"], "REVISE — coverage gap")
            finally:
                stream.close()

            # The console write must not have crashed the process, and the durable JSONL copy
            # must retain the ORIGINAL unicode text losslessly (only the console rendering may
            # be lossy under a real ASCII stream).
            events = _read_events(run_dir)
            self.assertEqual(events[0]["detail"]["decision"], "REVISE — coverage gap")
            console_text = raw_path.read_text(encoding="ascii", errors="replace")
            self.assertIn("decision=REVISE", console_text)

    def test_events_never_contain_raw_provider_response_text(self):
        # Requirement 7: no raw prompt/completion leakage -- run a real mock-runtime campaign and
        # assert that the distinctive literal text of the mock agent's response payload (which IS
        # written to disk elsewhere, e.g. the exchange dir) never appears anywhere in the durable
        # event log.
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _two_stage_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)

            code = cli.main(["run-campaign", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_SUCCESS)

            raw_log = (run_dir / "campaign_events.jsonl").read_text(encoding="utf-8")
            # The mock producer's raw JSON proposal body is written under exchange/stage_runner/
            # -- its distinctive rationale/parameter keys must never be echoed into the event log,
            # which only ever carries generic status/decision/id fields.
            proposal_files = list((run_dir / "exchange" / "stage_runner").glob("*.proposal.json"))
            self.assertTrue(proposal_files)
            for pf in proposal_files:
                raw_proposal_text = pf.read_text()
                proposal_obj = json.loads(raw_proposal_text)
                for key in ("rationale", "requested_at"):
                    value = proposal_obj.get(key)
                    if isinstance(value, str) and len(value) > 12:
                        self.assertNotIn(value, raw_log)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
