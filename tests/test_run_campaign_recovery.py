"""End-to-end proof that ``run_campaign`` drives the ENTIRE bounded-autonomy recovery lifecycle --
diagnosis (Analyst) -> recovery plan proposal (Orchestrator) -> human approval pause ->
automatic ``start_iteration`` -> automatic corrective-action dispatch (no human/Claude out-of-band
step) -> automatic RECOVERY_EXECUTION_UNVERIFIED handling/verification -> final re-gate PASS ->
COMPLETED -- purely through successive ``run_campaign`` calls interspersed with only the one
legitimate human decision point (``approve_recovery``). Network-free (mock runtime only).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


def _dataset(path: Path, n_frames: int, offset: int) -> Path:
    frames = []
    for i in range(n_frames):
        atoms = Atoms("Cu", positions=[[i + offset, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.info["structure_id"] = f"s{offset}-{i}"
        atoms.info["parent_structure_id"] = f"seed-pool:{900 + offset + i}"
        frames.append(atoms)
    write(str(path), frames)
    return path


def _stage(name: str, dataset_path: Path, manifest_rel: str) -> dict:
    return {
        "name": name, "command": None, "outputs": [manifest_rel],
        "gate": {"criteria": ["dataset manifest is complete"]},
        "pydantic_ai": {
            "role": "data-curator", "action": "build_dataset_manifest",
            "idempotency_key": f"recovery-campaign-test:{name}:001",
            "parameters": {"dataset": str(dataset_path), "manifest_path": f"{{artifacts_dir}}/{manifest_rel.split('/')[-1]}"},
        },
    }


def _workflow(root: Path, dataset_a: Path, dataset_b: Path) -> Path:
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "recovery-campaign-test",
        "inputs": [str(dataset_a), str(dataset_b)],
        "stages": [
            _stage("stage_a", dataset_a, "artifacts/stage_a_manifest.json"),
            _stage("stage_b", dataset_b, "artifacts/stage_b_manifest.json"),
        ],
    }))
    return workflow


def _revise_vote(path: Path, lens: str, criteria: list) -> Path:
    path.write_text(json.dumps({
        "review_lens": lens, "verdict": "REVISE",
        "criteria_checked": [{"criterion": c, "value_read": "coverage gap", "ok": False}
                             for c in criteria],
        "rationale": "dataset does not cover the required composition",
        "required_fix": "rebuild the manifest from a corrected dataset",
    }))
    return path


class RunCampaignRecoveryTests(unittest.TestCase):
    def _setup(self, root: Path):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        dataset_a = _dataset(root / "dataset_a.extxyz", 1, 0)
        dataset_b = _dataset(root / "dataset_b.extxyz", 1, 100)
        workflow = _workflow(root, dataset_a, dataset_b)
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
        c = RunController(run_dir)
        self.assertEqual(c.state["pending_recovery"]["status"], "required")
        return c, run_dir

    def _propose_and_approve(self, root: Path, run_dir: Path, *, corrective_dataset: Path,
                             manifest_target: Path):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from workflow.controller import RunController

        from workflow.integrity import sha256_file

        c = RunController(run_dir)
        stage_b_manifest = str((run_dir / "artifacts" / "stage_b_manifest.json").resolve())
        self.assertIn(stage_b_manifest, {a["path"] for a in c.state["artifacts"]})

        classification_payload = {
            "run_id": c.state["run_id"], "stage": "stage_b",
            "failure_category": "dataset_coverage",
            "evidence_refs": [{"role": "data-curator", "path": stage_b_manifest,
                              "integrity": {"sha256": sha256_file(Path(stage_b_manifest))}}],
            "evidence_summary": "stage_b's dataset manifest lacks coverage for the required composition",
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
            "rationale": "rebuild stage_b's manifest from a dataset that fixes the coverage gap",
            "corrective_action": {
                "action_type": "build_dataset_manifest",
                "parameters": {"dataset": str(corrective_dataset.resolve()),
                              "manifest_path": str(manifest_target.resolve())},
            },
        }
        orchestrator_response = root / "mock_orchestrator_response.json"
        orchestrator_response.write_text(json.dumps(proposal_payload))

        result = cli.run_campaign(
            c, runtime="mock", repo_root=str(ROOT),
            mock_analyst_response=str(analyst_response),
            mock_orchestrator_response=str(orchestrator_response), max_iterations=20)
        self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, result.message)
        self.assertEqual(result.exit_code, cli.EXIT_APPROVAL_REQUIRED)

        c = RunController(run_dir)
        self.assertEqual(c.state["pending_recovery"]["status"], "proposed")
        c.approve_recovery("Dr. Lee", note="approved for recovery-campaign test")
        c = RunController(run_dir)
        self.assertEqual(c.state["pending_recovery"]["status"], "approved")
        return c

    def test_full_recovery_lifecycle_reaches_completed_via_run_campaign_only(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _c, run_dir = self._setup(root)
            manifest_target = run_dir / "artifacts" / "stage_b_manifest.json"
            corrected_dataset = _dataset(root / "dataset_b_corrected.extxyz", 3, 500)
            self._propose_and_approve(root, run_dir, corrective_dataset=corrected_dataset,
                                      manifest_target=manifest_target)

            c = RunController(run_dir)
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20)
            self.assertEqual(result.outcome, cli.CAMPAIGN_COMPLETED, result.message)
            self.assertEqual(result.exit_code, cli.EXIT_SUCCESS)

            c = RunController(run_dir)
            self.assertEqual(c.stage("stage_a")["gate"], "PASS")
            self.assertEqual(c.stage("stage_b")["gate"], "PASS")
            self.assertEqual(c.stage("stage_b")["status"], "completed")
            manifest = json.loads(manifest_target.read_text())
            self.assertEqual(manifest["n_frames"], 3)
            recovery = c.state["recoveries"][0]
            self.assertEqual(recovery["status"], "resolved")
            self.assertEqual(recovery["execution"]["status"], "verified")
            iteration = c.state["iterations"][-1]
            self.assertEqual(iteration["recovery_execution"]["status"], "resolved")
            self.assertEqual(
                c.state["idempotency"]["recovery-campaign-test:stage_b:001:iter2"]["status"],
                "EXECUTED")

    def test_corrective_action_pending_pauses_campaign_without_completing_stage(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.dispatch import ActionDescriptor, ExternalActionPending
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController

        def _pending_executor(_proposal):
            raise ExternalActionPending("corrective action queued externally")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _c, run_dir = self._setup(root)
            manifest_target = run_dir / "artifacts" / "stage_b_manifest.json"
            corrected_dataset = _dataset(root / "dataset_b_corrected.extxyz", 3, 500)
            self._propose_and_approve(root, run_dir, corrective_dataset=corrected_dataset,
                                      manifest_target=manifest_target)

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
            self.assertEqual(result.exit_code, cli.EXIT_RECOVERY_ACTION_PENDING)

            c = RunController(run_dir)
            self.assertNotEqual(c.stage("stage_b")["status"], "completed")
            iteration = c.state["iterations"][-1]
            self.assertEqual(iteration["recovery_execution"]["status"], "required")

    def test_corrective_action_executed_but_missing_outputs_fails_closed(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        from runtimes.pydantic_ai.executors import build_executor_registry
        from workflow.controller import RunController

        def _noop_executor(_proposal):
            return {"path": None, "manifest": {}, "sha256": ""}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _c, run_dir = self._setup(root)
            manifest_target = run_dir / "artifacts" / "stage_b_manifest.json"
            corrected_dataset = _dataset(root / "dataset_b_corrected.extxyz", 3, 500)
            self._propose_and_approve(root, run_dir, corrective_dataset=corrected_dataset,
                                      manifest_target=manifest_target)

            registry = build_executor_registry()
            registry["build_dataset_manifest"] = ActionDescriptor(
                action_type="build_dataset_manifest", role="data-curator",
                executor=_noop_executor)

            c = RunController(run_dir)
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20,
                                      recovery_action_registry=registry)
            self.assertEqual(result.outcome, cli.CAMPAIGN_FAILED, result.message)
            self.assertEqual(result.exit_code, cli.EXIT_VALIDATION_REJECTED)
            self.assertIn("declared outputs are still missing", result.message)

            c = RunController(run_dir)
            self.assertNotEqual(c.stage("stage_b")["status"], "completed")


if __name__ == "__main__":
    unittest.main()
