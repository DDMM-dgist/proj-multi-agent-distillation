"""Regression: a recovery whose ``return_stage`` equals the failed stage and whose
``revalidation.targets`` are DOWNSTREAM of that stage must NOT deadlock.

Reproduces the C12F Stage-7 circular deadlock generically (training-shaped):

    * the return stage (``stage_a``) failed its own gate, so its gate is ``pending`` and
      every downstream stage is still ``pending`` (never ran);
    * the approved recovery returns to ``stage_a`` and lists the downstream stages
      (``stage_b``, ``stage_c``) as ``revalidation.targets``.

Before the fix, recovery-execution verification demanded changed+completed evidence for the
downstream revalidation targets before the return-stage gate could re-run -- but those stages
cannot run until the return-stage gate passes, which cannot happen until recovery is verified.
Circular. The fix makes verification key on the corrective action's evidence at the return
stage and DEFERS the downstream revalidation targets to normal campaign progression after the
return stage re-earns PASS. Network-free (mock runtime only).
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
            "idempotency_key": f"recovery-ordering-test:{name}:001",
            "parameters": {"dataset": str(dataset_path),
                           "manifest_path": f"{{artifacts_dir}}/{manifest_rel.split('/')[-1]}"},
        },
    }


def _workflow(root: Path, ds_a: Path, ds_b: Path, ds_c: Path) -> Path:
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "recovery-ordering-test",
        "inputs": [str(ds_a), str(ds_b), str(ds_c)],
        "stages": [
            _stage("stage_a", ds_a, "artifacts/stage_a_manifest.json"),
            _stage("stage_b", ds_b, "artifacts/stage_b_manifest.json"),
            _stage("stage_c", ds_c, "artifacts/stage_c_manifest.json"),
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


class RecoveryOrderingDeadlockTests(unittest.TestCase):
    def _fail_stage_a(self, root: Path):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        ds_a = _dataset(root / "ds_a.extxyz", 1, 0)
        ds_b = _dataset(root / "ds_b.extxyz", 1, 100)
        ds_c = _dataset(root / "ds_c.extxyz", 1, 200)
        run_dir = root / "run"
        RunController.initialize(_workflow(root, ds_a, ds_b, ds_c), run_dir)
        c = RunController(run_dir)

        # stage_a fails its own gate: three REVISE lenses -> pending_recovery required.
        lenses = [lens["id"] for lens in c.stage("stage_a")["gate_review_lenses"]]
        criteria = c.stage("stage_a")["gate_criteria"]
        votes = [_revise_vote(root / f"revise-{i}.json", lens, criteria)
                 for i, lens in enumerate(lenses, 1)]
        result = cli.run_production_stage(c, "stage_a", runtime="mock", repo_root=str(ROOT),
                                          mock_judge_response=[str(p) for p in votes])
        self.assertEqual(result.reason, "GATE_REVISE", result.message)
        c = RunController(run_dir)
        self.assertEqual(c.state["pending_recovery"]["status"], "required")
        # Downstream stages never ran.
        self.assertEqual(c.stage("stage_a")["gate"], "REVISE")
        self.assertEqual(c.stage("stage_b")["status"], "pending")
        self.assertEqual(c.stage("stage_c")["status"], "pending")
        return c, run_dir

    def _propose_and_approve(self, root: Path, run_dir: Path, corrective_dataset: Path,
                             manifest_target: Path):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from workflow.controller import RunController
        from workflow.integrity import sha256_file

        c = RunController(run_dir)
        stage_a_manifest = str((run_dir / "artifacts" / "stage_a_manifest.json").resolve())

        classification_payload = {
            "run_id": c.state["run_id"], "stage": "stage_a",
            "failure_category": "dataset_coverage",
            "evidence_refs": [{"role": "data-curator", "path": stage_a_manifest,
                              "integrity": {"sha256": sha256_file(Path(stage_a_manifest))}}],
            "evidence_summary": "stage_a's dataset manifest lacks required coverage",
            "confidence": 0.75, "recommended_recovery_target": "stage_a",
            "recommended_next_action": "rebuild the stage_a manifest from a corrected dataset",
        }
        classification = RootCauseClassification(**classification_payload)
        diagnosis_sha256 = hashlib.sha256(
            (classification.model_dump_json(indent=2) + "\n").encode()).hexdigest()
        analyst_response = root / "analyst.json"
        analyst_response.write_text(json.dumps(classification_payload))

        # The recovery returns to stage_a but lists the DOWNSTREAM stages as revalidation
        # targets -- the exact shape that used to deadlock.
        proposal_payload = {
            "run_id": c.state["run_id"], "failed_stage": "stage_a",
            "diagnosis_artifact_sha256": diagnosis_sha256,
            "capability": "data_repair", "return_stage": "stage_a",
            "proposed_changes": [{"type": "rebuild_manifest_from_corrected_dataset"}],
            "labeling": {"teacher_relabel": False, "new_dft": False},
            "student_training": {"retrain": False, "mode": "none"},
            "revalidation": {"reuse_profile": True, "targets": ["stage_b", "stage_c"]},
            "rationale": "rebuild stage_a's manifest, then revalidate the downstream stages",
            "corrective_action": {
                "action_type": "build_dataset_manifest",
                "parameters": {"dataset": str(corrective_dataset.resolve()),
                              "manifest_path": str(manifest_target.resolve())},
            },
        }
        orchestrator_response = root / "orchestrator.json"
        orchestrator_response.write_text(json.dumps(proposal_payload))

        result = cli.run_campaign(
            c, runtime="mock", repo_root=str(ROOT),
            mock_analyst_response=str(analyst_response),
            mock_orchestrator_response=str(orchestrator_response), max_iterations=30)
        self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, result.message)
        c = RunController(run_dir)
        self.assertEqual(c.state["pending_recovery"]["status"], "proposed")
        c.approve_recovery("Dr. Lee", note="approved for recovery-ordering test")
        return RunController(run_dir)

    def test_downstream_revalidation_targets_do_not_deadlock_return_stage_regate(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _c, run_dir = self._fail_stage_a(root)
            manifest_target = run_dir / "artifacts" / "stage_a_manifest.json"
            corrected = _dataset(root / "ds_a_corrected.extxyz", 3, 500)
            self._propose_and_approve(root, run_dir, corrective_dataset=corrected,
                                      manifest_target=manifest_target)

            c = RunController(run_dir)
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=30)
            # No deadlock: the campaign completes end to end.
            self.assertEqual(result.outcome, cli.CAMPAIGN_COMPLETED, result.message)
            self.assertEqual(result.exit_code, cli.EXIT_SUCCESS)

            c = RunController(run_dir)
            for stage in ("stage_a", "stage_b", "stage_c"):
                self.assertEqual(c.stage(stage)["gate"], "PASS", stage)
                self.assertEqual(c.stage(stage)["status"], "completed", stage)

            recovery = c.state["recoveries"][0]
            self.assertEqual(recovery["status"], "resolved")
            self.assertEqual(recovery["execution"]["status"], "verified")

            # (1) The corrective action was verified from the RETURN STAGE's changed evidence,
            #     with the downstream revalidation targets DEFERRED (empty stages at verify time).
            report = json.loads(
                (run_dir / "recovery" / "recovery-001.execution.report.json").read_text())
            self.assertEqual(report["revalidation"]["targets"], ["stage_b", "stage_c"])
            self.assertEqual(report["revalidation"]["stages"], [])
            self.assertEqual(report["student_training"]["retrain"], False)
            self.assertTrue(report["changes"])
            self.assertEqual(report["changes"][0]["status"], "APPLIED")

            # (2) Strict ordering from the durable event log:
            #     corrective verified -> return-stage re-gates PASS -> downstream gates only after.
            events = c.state["events"]
            verified_idx = next(i for i, e in enumerate(events)
                                if e.get("type") == "recovery_execution_verified")
            pass_idx = {}
            for i, e in enumerate(events):
                if e.get("type") == "gate" and e.get("verdict") == "PASS":
                    pass_idx.setdefault(e["stage"], i)  # first PASS per stage (post-recovery)
            self.assertIn("stage_a", pass_idx)
            self.assertIn("stage_b", pass_idx)
            self.assertIn("stage_c", pass_idx)
            self.assertLess(verified_idx, pass_idx["stage_a"])
            self.assertLess(pass_idx["stage_a"], pass_idx["stage_b"])
            self.assertLess(pass_idx["stage_a"], pass_idx["stage_c"])

    def test_return_stage_corrective_evidence_still_required(self):
        """The fix defers ONLY downstream revalidation -- it must NOT weaken the requirement
        that the corrective action produced changed evidence at the return stage. Verifying with
        no changed artifact at/downstream of the return stage still fails closed."""
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _c, run_dir = self._fail_stage_a(root)
            manifest_target = run_dir / "artifacts" / "stage_a_manifest.json"
            corrected = _dataset(root / "ds_a_corrected.extxyz", 3, 500)
            c = self._propose_and_approve(root, run_dir, corrective_dataset=corrected,
                                          manifest_target=manifest_target)
            # Enter the recovery iteration but do NOT run the corrective action yet.
            c.start_iteration()
            c = RunController(run_dir)
            report, missing = cli._assemble_recovery_execution_report(c)
            self.assertIsNone(report)
            self.assertTrue(missing)
            self.assertTrue(any("has changed" in m or "return_stage" in m for m in missing))


if __name__ == "__main__":
    unittest.main()
