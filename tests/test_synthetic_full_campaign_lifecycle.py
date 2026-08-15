"""Part C, Path 1: a synthetic campaign matching the REAL 12-stage production topology
(``configs/runs/sio2-sox-allegro-simplenn-r17/workflow.yaml``'s stage order:
teacher_baseline, reference_validation, acquisition, data_coverage, teacher_labeling,
dataset_split, training, evaluation, uncertainty, deployment_md, physical_validation, analysis)
reaches ``COMPLETED`` with every stage's gate at ``PASS``, driven entirely through the real
``runtimes.pydantic_ai.cli.run_campaign``/``run_production_stage``/``RunController`` production
path -- no parallel/bypass implementation.

Four of the twelve stages (``data_coverage``, ``uncertainty``, ``physical_validation``,
``analysis``) carry NO ``pydantic_ai`` role/action in the real config -- confirmed by direct
inspection of that file and of ``cli._default_stage_route`` -- so in real production those are
completed by a human/analyst script calling ``complete_external_stage``+``record_gate`` directly,
never by ``run-campaign``. This fixture reproduces that EXACT automated/manual split, and proves
(rather than assumes) that ``run_campaign`` fails closed with a precise ``ValueError`` -- never a
silent skip or a wrong dispatch -- if it is ever asked to advance past one of those four stages
before a human has completed it. dataset_split/training/evaluation reuse the SAME real action
names production uses (``generate_group_split``, ``train_committee``, ``evaluate_heldout_fidelity``
-- the latter two exactly as ``tests/test_run_campaign.py`` already exercises, mock committee, real
approval boundary ``costly_training``). The other four automated stages stand in for what would be
heavier Teacher/MD adapters in production with the SAME already-proven, deterministic, ungated
``build_dataset_manifest`` executor ``tests/test_run_campaign_recovery.py`` uses -- avoiding
scientific compute and optional heavy dependencies while still exercising the identical real
dispatch/gate/controller machinery for each stage slot. No OpenAI network call anywhere (mock
runtime only).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent

# The real production stage order and its automated/manual split (see module docstring).
MANUAL_STAGES = {"data_coverage", "uncertainty", "physical_validation", "analysis"}
STAGE_ORDER = [
    "teacher_baseline", "reference_validation", "acquisition", "data_coverage",
    "teacher_labeling", "dataset_split", "training", "evaluation", "uncertainty",
    "deployment_md", "physical_validation", "analysis",
]
# Automated stages standing in for a heavier real Teacher/MD action with the same already-proven
# ungated build_dataset_manifest executor (see module docstring).
_MANIFEST_STAGES = ["teacher_baseline", "reference_validation", "acquisition", "teacher_labeling",
                    "deployment_md"]


def _tiny_dataset(path: Path, tag: str) -> Path:
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.info["structure_id"] = tag
    write(str(path), [atoms])
    return path


def _manifest_stage_cfg(root: Path, name: str) -> dict:
    dataset = _tiny_dataset(root / f"{name}_dataset.extxyz", name)
    params = {"dataset": str(dataset), "manifest_path": f"{{artifacts_dir}}/{name}_manifest.json"}
    if name == "teacher_baseline":
        # cli._fill_default_parameters hardcodes this requirement by STAGE NAME regardless of
        # the configured action (see cli.py:210-211) -- harmless extra key for
        # build_dataset_manifest's free-form `parameters` dict (Part B audit: ActionProposalBase.
        # parameters is deliberately unconstrained; per-action shape isn't enforced pre-Phase 4).
        params["structures_path"] = str(dataset)
    return {
        "name": name, "command": None, "outputs": [f"artifacts/{name}_manifest.json"],
        "gate": {"criteria": [f"{name} manifest is complete"]},
        "pydantic_ai": {
            "role": "data-curator", "action": "build_dataset_manifest",
            "idempotency_key": f"synthetic-12stage:{name}:001",
            "parameters": params,
        },
    }


def _dataset_split_stage_cfg(root: Path) -> dict:
    dataset = root / "dataset_split_dataset.extxyz"
    frames = []
    for i in range(6):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"ds{i}"
        a.info["parent_structure_id"] = f"seed:{i}"
        frames.append(a)
    write(str(dataset), frames)
    return {
        "name": "dataset_split", "command": None,
        "outputs": ["artifacts/dataset_split/split_manifest.json"],
        "gate": {"criteria": ["split manifest is complete"]},
        "pydantic_ai": {
            "role": "data-curator", "action": "generate_group_split",
            "idempotency_key": "synthetic-12stage:dataset_split:001",
            "parameters": {"dataset": str(dataset),
                          "output_dir": "{artifacts_dir}/dataset_split",
                          "manifest": "{artifacts_dir}/dataset_split/split_manifest.json"},
        },
    }


def _training_evaluation_stage_cfgs(root: Path) -> tuple:
    dataset = root / "training_dataset.extxyz"
    frames = []
    for i in range(3):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"t{i}"
        a.info["parent_structure_id"] = f"p{i}"
        frames.append(a)
    write(str(dataset), frames)
    student_cfg = root / "student.yaml"
    student_cfg.write_text(
        "kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n"
        "predict:\n  factory: adapters.mock_model.MockCheckpointCalculator\n"
        "  checkpoint_arg: checkpoint\n  kwargs: {}\n")
    training = {
        "name": "training", "command": None,
        "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
        "pydantic_ai": {
            "role": "ml-trainer", "action": "train_committee",
            "approval_boundary": "costly_training",
            "idempotency_key": "synthetic-12stage-training-001",
            "parameters": {
                "student_config": str(student_cfg), "dataset": str(dataset),
                "output_dir": "{artifacts_dir}/committee",
                "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
            },
        },
        "gate": {"criteria": ["committee manifest is complete"]},
    }
    evaluation = {
        "name": "evaluation", "command": None,
        "outputs": ["artifacts/heldout_labeled.extxyz", "artifacts/heldout_report.json"],
        "pydantic_ai": {
            "role": "ml-trainer", "action": "evaluate_heldout_fidelity",
            "approval_boundary": "costly_training",
            "idempotency_key": "synthetic-12stage-evaluation-001",
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
    return training, evaluation


def _twelve_stage_workflow(root: Path) -> Path:
    training, evaluation = _training_evaluation_stage_cfgs(root)
    by_name = {"dataset_split": _dataset_split_stage_cfg(root), "training": training,
              "evaluation": evaluation}
    for name in _MANIFEST_STAGES:
        by_name[name] = _manifest_stage_cfg(root, name)
    for name in MANUAL_STAGES:
        # No "pydantic_ai" key at all -- exactly matches the real production config for these
        # four stages, so run_campaign can NEVER auto-dispatch them (see cli._proposal_from_stage).
        by_name[name] = {"name": name, "command": None, "outputs": [f"artifacts/{name}.json"],
                        "gate": {"criteria": [f"{name} is complete"]}}
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "synthetic-12stage", "stages": [by_name[name] for name in STAGE_ORDER],
    }))
    return workflow


class TwelveStageSyntheticCampaignTests(unittest.TestCase):
    def test_all_twelve_stages_pass_and_campaign_completes(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _twelve_stage_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training",
                                       "--note", "pre-approved for synthetic 12-stage test"]),
                             cli.EXIT_SUCCESS)

            c = RunController(run_dir)
            guard = 0
            while True:
                guard += 1
                self.assertLess(guard, 30, "test driver looped without making progress")
                pending_name = next(
                    (s["name"] for s in c.state["stages"] if s["gate"] != "PASS"), None)
                if pending_name is None:
                    break
                if pending_name in MANUAL_STAGES:
                    artifact = c.run_dir / f"artifacts/{pending_name}.json"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text('{"status": "ok"}')
                    c.complete_external_stage(pending_name, [artifact])
                    votes = c.run_dir / "gates" / f"{pending_name}.votes.json"
                    import json as _json
                    criteria = c.stage(pending_name)["gate_criteria"]
                    lenses = c.stage(pending_name)["gate_review_lenses"]
                    votes.write_text(_json.dumps({
                        "stage": pending_name, "criteria": criteria, "review_lenses": lenses,
                        "artifact_sha256": {a["path"]: a["sha256"]
                                            for a in c.stage_artifacts(pending_name)},
                        "decision": "PASS",
                        "votes": [{"judge_id": f"judge-{i}", "review_lens": lens["id"],
                                  "verdict": "PASS",
                                  "criteria_checked": [{"criterion": cr, "value_read": "checked",
                                                        "ok": True} for cr in criteria],
                                  "rationale": "ok", "required_fix": ""}
                                 for i, lens in enumerate(lenses, 1)]}))
                    c.record_gate(pending_name, votes_path=votes)
                    c = RunController(run_dir)
                    continue
                try:
                    result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                              auto_mock_judges=True, max_iterations=20)
                except ValueError as exc:
                    # A workflow-authoring/operator error, not a hidden contract defect: proves
                    # run_campaign fails closed with a precise message rather than silently
                    # skipping or mis-dispatching a stage that has no pydantic_ai route.
                    self.assertIn("pydantic_ai role/action metadata", str(exc))
                    c = RunController(run_dir)
                    continue
                c = RunController(run_dir)
                if result.outcome == cli.CAMPAIGN_COMPLETED:
                    break
                self.assertNotIn(result.outcome, (cli.CAMPAIGN_FAILED,),
                                 f"unexpected terminal failure: {result.message}")

            c = RunController(run_dir)
            for name in STAGE_ORDER:
                self.assertEqual(c.stage(name)["gate"], "PASS", name)
                self.assertEqual(c.stage(name)["status"], "completed", name)
            self.assertIsNone(c.state.get("pending_recovery"))

            # Idempotent resume: re-running after terminal completion must not error or mutate.
            before = c.state
            try:
                final = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                         auto_mock_judges=True, max_iterations=20)
                self.assertEqual(final.outcome, cli.CAMPAIGN_COMPLETED)
            except ValueError:
                self.fail("a fully-completed campaign must not attempt to dispatch any stage")
            after = RunController(run_dir).state
            self.assertEqual(before["stages"], after["stages"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
