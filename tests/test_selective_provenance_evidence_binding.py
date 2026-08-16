"""Regression tests for the R22 Stage-1 evidence-binding fix: a stage's bounded-evidence
assembly (`runtimes.pydantic_ai.cli.run_production_stage`) previously built its `upstream`
artifact list ONLY from `c.state["artifacts"]` (registered stage OUTPUTS), never from
`c.state["inputs"]` (run INPUTS) -- so a run-bound provenance manifest (e.g. a Teacher
train/validation/test split manifest) could never appear in a stage's evidence crosswalk, no
matter how it was declared in workflow.yaml.

`_selective_provenance_inputs` closes this generically: any top-level workflow-config key
ending in `_provenance` may declare `applies_to_stage` + `bound_evidence_input_indices` to
selectively opt specific, indexed run inputs into a specific stage's evidence assembly --
never all inputs (the `_cmd_preflight` anti-pattern), never protected-reference roles.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write


def _split_manifest(path: Path) -> None:
    path.write_text(json.dumps({
        "records": [
            {"source_category": "bulk", "source_local_index": 0, "split": "train"},
            {"source_category": "bulk", "source_local_index": 1, "split": "test"},
        ],
    }))


def _workflow(root: Path, *, applies_to_stage="training", role=None,
              declare_binding=True) -> Path:
    dataset = root / "train.extxyz"
    frames = []
    for i in range(2):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"s{i}"
        a.info["parent_structure_id"] = f"p{i}"
        frames.append(a)
    write(str(dataset), frames)
    student_cfg = root / "student.yaml"
    student_cfg.write_text("kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n")
    split_manifest = root / "split_manifest.json"
    _split_manifest(split_manifest)

    cfg = {
        "run_id": "synthetic-selective-provenance",
        "inputs": [str(student_cfg), str(dataset), str(split_manifest)],
        "stages": [{
            "name": "training",
            "command": None,
            "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
            "pydantic_ai": {
                "role": "ml-trainer",
                "action": "train_committee",
                "approval_boundary": "costly_training",
                "idempotency_key": "synthetic-selective-provenance:001",
                "parameters": {
                    "student_config": str(student_cfg),
                    "dataset": str(dataset),
                    "output_dir": "{artifacts_dir}/committee",
                    "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
                },
            },
            "gate": {"criteria": ["committee manifest is complete"]},
        }],
    }
    if declare_binding:
        block = {
            "role": role or "TEST_SPLIT_RECONSTRUCTION_PROVENANCE",
            "applies_to_stage": applies_to_stage,
            "bound_evidence_input_indices": [2],
            "split_manifest_input_index": 2,
        }
        cfg["test_split_provenance"] = block
        if role:
            cfg["protected_reference_roles"] = [role]
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return workflow


class SelectiveProvenanceInputBindingTests(unittest.TestCase):
    def test_declared_index_is_resolved_from_state_inputs(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.cli import _selective_provenance_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            paths = _selective_provenance_inputs(c, "training")
            self.assertEqual(len(paths), 1)
            self.assertTrue(Path(paths[0]).name.endswith("split_manifest.json"))

    def test_wrong_stage_name_is_not_bound(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.cli import _selective_provenance_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _workflow(root, applies_to_stage="some_other_stage")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            self.assertEqual(_selective_provenance_inputs(c, "training"), [])

    def test_protected_reference_role_is_never_bound(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.cli import _selective_provenance_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _workflow(root, role="teacher_train_partition")
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            self.assertEqual(_selective_provenance_inputs(c, "training"), [])

    def test_no_declaring_block_binds_nothing(self):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.cli import _selective_provenance_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _workflow(root, declare_binding=False)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)
            self.assertEqual(_selective_provenance_inputs(c, "training"), [])

    def test_stage_evidence_bundle_includes_bound_split_manifest_not_all_inputs(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training", "--note", "test"]),
                             cli.EXIT_SUCCESS)
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--stage", "training", "--auto-mock-judges"])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            evidence = json.loads(
                (run_dir / "exchange" / "bounded_evidence" / "training.json").read_text())
            summarized_paths = {a["artifact_path"] for a in evidence["artifacts"]}
            self.assertTrue(any(p.endswith("split_manifest.json") for p in summarized_paths))
            # The bound manifest also seeds the split crosswalk used for provenance.
            self.assertEqual(len(evidence["split_crosswalk_sources"]), 1)
            self.assertTrue(evidence["split_crosswalk_sources"][0]["path"].endswith(
                "split_manifest.json"))
            # student.yaml (an input, but not declared in bound_evidence_input_indices) must
            # never appear -- proves this is selective, not "every input" (_cmd_preflight's
            # unfiltered approach).
            self.assertFalse(any(p.endswith("student.yaml") for p in summarized_paths))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
