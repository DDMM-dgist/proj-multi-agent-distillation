"""Continuation TRUE-RESUME wiring: threading a per-seed ``continue_from`` checkpoint
plus a MAXIMUM ``total_epoch_override`` from an approved recovery's corrective action all
the way down to SIMPLE-NN v2's native ``neural_network.continue`` resume.

Covers the exact defect that blocked the C12F continuation recovery: the canonical
executor chain (``_dispatch_recovery_corrective_action`` -> ``_exec_train_committee`` ->
``train_committee`` -> ``train_student`` -> ``_train_simple_nn`` -> ``simple_nn_v2_wrapper``)
had no way to continue from an existing checkpoint, so an approved continuation plan would
have silently retrained from scratch. Generic (no run/stage/seed hardcoded in the code).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


class WrapperContinueWiringTests(unittest.TestCase):
    """The SIMPLE-NN wrapper turns --continue-from into a native resume, and omits it
    entirely (fresh training) when not supplied."""

    def _args(self, continue_from, epochs=2000):
        return argparse.Namespace(
            seed=7, batch_size=4, epochs=epochs, precision="double",
            use_stress=False, stress_loss_weight=0.0, continue_from=continue_from)

    def test_continue_from_sets_native_resume_keys(self):
        from adapters.simple_nn_v2_wrapper import _build_input_yaml
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint_bestmodel.pth.tar"
            ckpt.write_bytes(b"stub")
            payload = _build_input_yaml(
                {"neural_network": {}, "data": {}}, self._args(str(ckpt)),
                {"Si": str(ckpt)}, Path(tmp) / "structure_list")
            nn = payload["neural_network"]
            self.assertEqual(nn["continue"], str(ckpt.resolve()))
            self.assertFalse(nn["clear_prev_status"])
            self.assertFalse(nn["clear_prev_optimizer"])
            self.assertEqual(nn["total_epoch"], 2000)

    def test_absent_continue_from_is_fresh_training(self):
        from adapters.simple_nn_v2_wrapper import _build_input_yaml
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "params_Si"
            ckpt.write_bytes(b"stub")
            payload = _build_input_yaml(
                {"neural_network": {}, "data": {}}, self._args(None, epochs=200),
                {"Si": str(ckpt)}, Path(tmp) / "structure_list")
            nn = payload["neural_network"]
            self.assertNotIn("continue", nn)
            self.assertNotIn("clear_prev_status", nn)
            self.assertEqual(nn["total_epoch"], 200)

    def test_missing_continue_checkpoint_fails_closed(self):
        from adapters.simple_nn_v2_wrapper import _build_input_yaml
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _build_input_yaml(
                    {"neural_network": {}, "data": {}},
                    self._args(str(Path(tmp) / "does_not_exist.pth.tar")),
                    {"Si": str(Path(tmp))}, Path(tmp) / "structure_list")


class TrainCommitteeThreadingTests(unittest.TestCase):
    """``train_committee`` resolves a per-seed continue_from map and forwards the
    MAXIMUM total_epoch_override, both keyed generically by seed."""

    def test_per_seed_continue_from_and_epoch_override_forwarded(self):
        import workflow.steps as steps

        captured = []

        class _FakeArtifact:
            kind = "simple-nn"

            def __init__(self, path):
                self.path = path
                self.metadata = {}

        def _fake_train_student(cfg, dataset, out_dir, seed, *, continue_from=None,
                                total_epoch_override=None):
            captured.append({"seed": seed, "continue_from": continue_from,
                             "total_epoch_override": total_epoch_override})
            p = Path(out_dir) / "potential_saved_bestmodel"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("stub")
            return _FakeArtifact(p)

        orig_train_student = steps.train_student
        orig_load_config = steps.load_config
        orig_digest = steps.artifact_digest
        try:
            steps.train_student = _fake_train_student
            steps.load_config = lambda _p: {"committee": {"seeds": [202631, 202632]}}
            steps.artifact_digest = lambda _p: {"sha256": "stub"}
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "committee"
                manifest = Path(tmp) / "committee.manifest.json"
                cont = {"seed-202631": "/ckpts/a.pth.tar"}  # 202632 intentionally absent
                steps.train_committee(
                    "student.yaml", "train.extxyz", out, manifest,
                    continue_from=cont, total_epoch_override=2000)
        finally:
            steps.train_student = orig_train_student
            steps.load_config = orig_load_config
            steps.artifact_digest = orig_digest

        by_seed = {c["seed"]: c for c in captured}
        self.assertEqual(by_seed[202631]["continue_from"], "/ckpts/a.pth.tar")
        self.assertIsNone(by_seed[202632]["continue_from"])  # absent -> fresh
        self.assertEqual(by_seed[202631]["total_epoch_override"], 2000)
        self.assertEqual(by_seed[202632]["total_epoch_override"], 2000)

    def test_bare_int_seed_key_also_matches(self):
        import workflow.steps as steps
        captured = []

        class _FakeArtifact:
            kind = "simple-nn"

            def __init__(self, path):
                self.path = path
                self.metadata = {}

        def _fake_train_student(cfg, dataset, out_dir, seed, *, continue_from=None,
                                total_epoch_override=None):
            captured.append({"seed": seed, "continue_from": continue_from})
            p = Path(out_dir) / "potential_saved_bestmodel"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("stub")
            return _FakeArtifact(p)

        orig = (steps.train_student, steps.load_config, steps.artifact_digest)
        try:
            steps.train_student = _fake_train_student
            steps.load_config = lambda _p: {"committee": {"seeds": [5]}}
            steps.artifact_digest = lambda _p: {"sha256": "stub"}
            with tempfile.TemporaryDirectory() as tmp:
                steps.train_committee(
                    "s.yaml", "d.extxyz", Path(tmp) / "c", Path(tmp) / "m.json",
                    continue_from={"5": "/ckpts/five.pth.tar"})
        finally:
            steps.train_student, steps.load_config, steps.artifact_digest = orig
        self.assertEqual(captured[0]["continue_from"], "/ckpts/five.pth.tar")


# --- corrective-action param merge (matching action_type re-runs the stage's own action) ---

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
            "idempotency_key": f"resume-merge-test:{name}:001",
            "parameters": {"dataset": str(dataset_path),
                           "manifest_path": f"{{artifacts_dir}}/{manifest_rel.split('/')[-1]}"},
        },
    }


def _revise_vote(path: Path, lens: str, criteria: list) -> Path:
    path.write_text(json.dumps({
        "review_lens": lens, "verdict": "REVISE",
        "criteria_checked": [{"criterion": c, "value_read": "coverage gap", "ok": False}
                             for c in criteria],
        "rationale": "dataset does not cover the required composition",
        "required_fix": "rebuild the manifest from a corrected dataset",
    }))
    return path


class CorrectiveActionParamMergeTests(unittest.TestCase):
    """When the approved corrective action re-runs the return stage's OWN action, the
    controller-resolved base-stage parameters are the merge base and the plan's
    corrective_action.parameters override/extend them. Proven end-to-end: the corrective
    action supplies ONLY the corrected ``dataset`` and OMITS ``manifest_path`` -- if the
    base param did not merge, the executor would fail closed with no output path."""

    def test_corrective_action_inherits_base_stage_params(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from workflow.controller import RunController
        from workflow.integrity import sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_a = _dataset(root / "a.extxyz", 1, 0)
            dataset_b = _dataset(root / "b.extxyz", 1, 100)
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({
                "run_id": "resume-merge-test",
                "inputs": [str(dataset_a), str(dataset_b)],
                "stages": [
                    _stage("stage_a", dataset_a, "artifacts/stage_a_manifest.json"),
                    _stage("stage_b", dataset_b, "artifacts/stage_b_manifest.json"),
                ],
            }))
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            self.assertEqual(cli.run_production_stage(
                c, "stage_a", runtime="mock", repo_root=str(ROOT),
                auto_mock_judges=True).reason, "SUCCESS")
            c = RunController(run_dir)
            lenses = [ll["id"] for ll in c.stage("stage_b")["gate_review_lenses"]]
            criteria = c.stage("stage_b")["gate_criteria"]
            votes = [_revise_vote(root / f"r{i}.json", ll, criteria)
                     for i, ll in enumerate(lenses, 1)]
            self.assertEqual(cli.run_production_stage(
                c, "stage_b", runtime="mock", repo_root=str(ROOT),
                mock_judge_response=[str(p) for p in votes]).reason, "GATE_REVISE")

            c = RunController(run_dir)
            stage_b_manifest = str((run_dir / "artifacts" / "stage_b_manifest.json").resolve())
            classification_payload = {
                "run_id": c.state["run_id"], "stage": "stage_b",
                "failure_category": "dataset_coverage",
                "evidence_refs": [{"role": "data-curator", "path": stage_b_manifest,
                                   "integrity": {"sha256": sha256_file(Path(stage_b_manifest))}}],
                "evidence_summary": "stage_b manifest lacks coverage",
                "confidence": 0.75, "recommended_recovery_target": "stage_b",
                "recommended_next_action": "rebuild the stage_b manifest",
            }
            classification = RootCauseClassification(**classification_payload)
            diagnosis_sha256 = hashlib.sha256(
                (classification.model_dump_json(indent=2) + "\n").encode()).hexdigest()
            analyst_response = root / "analyst.json"
            analyst_response.write_text(json.dumps(classification_payload))

            corrected = _dataset(root / "b_corrected.extxyz", 3, 500)
            # NOTE: corrective_action.parameters carries ONLY the corrected dataset; the
            # manifest_path is intentionally omitted so the test fails unless the base
            # stage param merges underneath.
            proposal_payload = {
                "run_id": c.state["run_id"], "failed_stage": "stage_b",
                "diagnosis_artifact_sha256": diagnosis_sha256,
                "capability": "data_repair", "return_stage": "stage_b",
                "proposed_changes": [{"type": "rebuild_manifest_from_corrected_dataset"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["stage_b"]},
                "rationale": "rebuild stage_b manifest from a corrected dataset",
                "corrective_action": {
                    "action_type": "build_dataset_manifest",
                    "parameters": {"dataset": str(corrected.resolve())},
                },
            }
            orchestrator_response = root / "orchestrator.json"
            orchestrator_response.write_text(json.dumps(proposal_payload))

            result = cli.run_campaign(
                c, runtime="mock", repo_root=str(ROOT),
                mock_analyst_response=str(analyst_response),
                mock_orchestrator_response=str(orchestrator_response), max_iterations=20)
            self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL,
                             result.message)

            c = RunController(run_dir)
            c.approve_recovery("Dr. Lee", note="approved for resume-merge test")
            c = RunController(run_dir)
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20)
            self.assertEqual(result.outcome, cli.CAMPAIGN_COMPLETED, result.message)

            manifest = json.loads(
                (run_dir / "artifacts" / "stage_b_manifest.json").read_text())
            # base manifest_path merged (output written) AND corrective dataset override applied
            self.assertEqual(manifest["n_frames"], 3)


if __name__ == "__main__":
    unittest.main()
