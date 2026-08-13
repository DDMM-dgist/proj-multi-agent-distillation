from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from ase import Atoms
from ase.io import write

from runtimes.pydantic_ai import cli


def _sha(path):
    from workflow.integrity import sha256_file
    return sha256_file(path)


def _atoms(index=900, category="bulk", x=1.0):
    atoms = Atoms("Cu", positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.info["structure_id"] = f"seed-pool:{index}"
    atoms.info["parent_structure_id"] = f"seed-pool:{index}"
    atoms.info["source_global_index"] = index
    atoms.info["source_category"] = category
    return atoms


def _protected_reference(root: Path) -> Path:
    protected = root / "protected.extxyz"
    write(str(protected), [_atoms(760, "protected", 0.0)])
    rows = root / "protected_rows.txt"
    rows.write_text("760\n761\n", encoding="utf-8")
    manifest = root / "protected_manifest.json"
    manifest.write_text(json.dumps({"mapping": {
        "logical_test_frames": 1,
        "matched_logical_frames": 1,
        "unmatched_logical_frames": 0,
        "protected_source_rows": 2,
        "conflicting_label_duplicates": 0,
    }}), encoding="utf-8")
    ref = root / "reference.yaml"
    ref.write_text("\n".join([
        "kind: protected-existing-dft",
        "reference_id: fixture-reference",
        "reference_class: ORIGINAL_TEACHER_TEST",
        "status: AVAILABLE_AND_PROTECTED",
        "logical_test_frames: 1",
        "protected_source_rows: 2",
        f"protection_manifest: {manifest}",
        f"protected_source_rows_file: {rows}",
        "duplicate_equivalent:",
        "  source_global_indices: [760, 761]",
        "  label_conflict: false",
        "prohibited_uses: [student_training, student_validation_tuning, acquisition_seed, augmentation_parent, recovery_training]",
        "structures:",
        f"  path: {protected}",
        "  logical_frames: 1",
        f"  sha256: {_sha(protected)}",
        "",
    ]), encoding="utf-8")
    return ref


class AcquisitionLifecycleRunStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg_dir = self.root / "cfg"
        self.cfg_dir.mkdir()
        self.run_dir = self.root / "run"
        self.seed = self.cfg_dir / "seed.extxyz"
        write(str(self.seed), [_atoms(900, "bulk", 1.0), _atoms(901, "void", 2.0)])
        self.teacher = self.cfg_dir / "teacher.yaml"
        self.teacher.write_text("kind: mock\n", encoding="utf-8")
        self.acq = self.cfg_dir / "acq.yaml"
        self.acq.write_text(yaml.safe_dump({
            "kind": "augment-atoms",
            "cli": {"invocation": ["augment-atoms", "{config_path}", "--input", "{seed_path}", "--output", "{out_path}"]},
        }), encoding="utf-8")
        self.reference = _protected_reference(self.cfg_dir)
        self.plan = self.cfg_dir / "plan.json"
        self._write_plan()

    def _write_plan(self, **updates):
        payload = {
            "schema_version": 1,
            "eligible_source_categories": ["bulk"],
            "selected_parent_structure_ids": ["seed-pool:900"],
            "selected_source_global_indices": [900],
            "n_parents": 1,
            "n_per_structure": 1,
            "T_K": 300.0,
            "beta": 0.1,
            "sigma_range_A": [0.01, 0.02],
            "cell_sigma": None,
            "seed": 123,
            "expected_output_count": 1,
            "duplicate_handling": "drop_exact_duplicates",
            "protected_reference_exclusion_report": {
                "status": "PASS",
                "reference_id": "fixture-reference",
                "dft_labels_used_as_selection_scores": False,
            },
        }
        payload.update(updates)
        self.plan.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _workflow(self, include_plan=True):
        params = {
            "acquisition_config": "{run_dir}/inputs/001-acq.yaml",
            "teacher_config": "{run_dir}/inputs/000-teacher.yaml",
            "seed_structures": str(self.seed),
            "out_path": "{run_dir}/artifacts/acquisition_candidates.extxyz",
            "manifest_path": "{run_dir}/artifacts/acquisition.manifest.json",
            "selected_source_indices": [900],
        }
        if include_plan:
            params["acquisition_plan_path"] = "{run_dir}/inputs/003-plan.json"
        cfg = {
            "run_id": "acq-lifecycle",
            "inputs": [
                {"path": str(self.teacher), "role": "teacher_config"},
                {"path": str(self.acq), "role": "acquisition_config"},
                {"path": str(self.reference), "role": "protected_reference"},
            ] + ([{"path": str(self.plan), "role": "acquisition_plan"}] if include_plan else []),
            "stages": [{
                "name": "acquisition",
                "command": None,
                "outputs": [
                    "artifacts/acquisition_candidates.extxyz",
                    "artifacts/acquisition.manifest.json",
                    "artifacts/acquisition_protection_audit.json",
                ],
                "gate": {"criteria": ["acquisition lineage/protection contract passes"]},
                "contract": {
                    "kind": "validation_manifest",
                    "manifest": "artifacts/acquisition_protection_audit.json",
                    "validator": "validation.protected_reference.validate_protection_audit_report",
                    "options": {"reference_yaml": str(self.reference)},
                },
                "pydantic_ai": {
                    "role": "data-curator",
                    "action": "acquire_structures",
                    "approval_boundary": "costly_teacher_labeling",
                    "idempotency_key": "acq-lifecycle:acquisition:001",
                    "parameters": params,
                },
            }],
        }
        path = self.cfg_dir / "workflow.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path

    def _init(self, include_plan=True):
        from workflow.controller import RunController
        return RunController.initialize(self._workflow(include_plan), self.run_dir)

    def _pass_votes(self):
        from workflow.controller import RunController
        ctx = RunController(self.run_dir).gate_context("acquisition")
        paths = []
        for i, lens in enumerate(ctx["review_lenses"], 1):
            vote = {
                "review_lens": lens["id"],
                "verdict": "PASS",
                "criteria_checked": [{"criterion": "acquisition lineage/protection contract passes", "value_read": "bounded evidence", "ok": True}],
                "rationale": "accepted deterministic acquisition evidence",
                "required_fix": "",
            }
            p = self.root / f"judge-{i}.json"
            p.write_text(json.dumps(vote), encoding="utf-8")
            paths.append(str(p))
        return paths

    def test_run_stage_acquisition_without_plan_fails_before_provider_and_executor(self):
        self._init(include_plan=False)
        with mock.patch("runtimes.pydantic_ai.production_router.run_role") as run_role, \
             mock.patch("adapters.acquisition.acquire") as adapter:
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(self.run_dir), "--stage", "acquisition", "--auto-mock-judges"])
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self.assertEqual(run_role.call_count, 0)
        self.assertEqual(adapter.call_count, 0)

    def test_valid_plan_requires_exact_plan_approval(self):
        self._init(include_plan=True)
        code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(self.run_dir), "--stage", "acquisition", "--auto-mock-judges"])
        self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
        from workflow.controller import RunController
        c = RunController(self.run_dir)
        self.assertFalse(c.state.get("idempotency"))

    def test_approval_for_other_plan_hash_is_rejected(self):
        self._init(include_plan=True)
        cli.main(["approve", "--run-dir", str(self.run_dir), "--boundary", "costly_teacher_labeling", "--note", "wrong", "--plan-sha256", "0" * 64])
        code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(self.run_dir), "--stage", "acquisition", "--auto-mock-judges"])
        self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)

    def test_plan_modification_after_exact_approval_is_rejected(self):
        c = self._init(include_plan=True)
        from runtimes.pydantic_ai.executors import acquisition_plan_sha256_from_proposal
        stage_cfg = cli._stage_config(c, "acquisition")
        proposal, _ = cli._proposal_from_stage(c, "acquisition", stage_cfg)
        proposal = cli._bind_acquisition_plan_for_stage(c, proposal)
        plan_sha = acquisition_plan_sha256_from_proposal(proposal)
        cli.main(["approve", "--run-dir", str(self.run_dir), "--boundary", "costly_teacher_labeling", "--note", "exact", "--plan-sha256", plan_sha])
        self._write_plan(seed=999)
        with mock.patch("adapters.acquisition.acquire") as adapter:
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(self.run_dir), "--stage", "acquisition", "--auto-mock-judges"])
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self.assertEqual(adapter.call_count, 0)
        c = __import__("workflow.controller", fromlist=["RunController"]).RunController(self.run_dir)
        self.assertEqual(c.stage("acquisition")["attempts"], 0)
        self.assertFalse(c.state.get("idempotency"))

    def test_successful_mock_adapter_creates_all_stage_outputs_through_run_stage(self):
        c = self._init(include_plan=True)
        from runtimes.pydantic_ai.executors import acquisition_plan_sha256_from_proposal
        stage_cfg = cli._stage_config(c, "acquisition")
        proposal, _ = cli._proposal_from_stage(c, "acquisition", stage_cfg)
        proposal = cli._bind_acquisition_plan_for_stage(c, proposal)
        plan_sha = acquisition_plan_sha256_from_proposal(proposal)
        cli.main(["approve", "--run-dir", str(self.run_dir), "--boundary", "costly_teacher_labeling", "--note", "exact", "--plan-sha256", plan_sha])
        def adapter(cfg, teacher_cfg, seed_path, out_path):
            atoms = Atoms("Cu", positions=[[3, 0, 0]], cell=[10, 10, 10], pbc=True)
            atoms.info["parent"] = "seed-pool:900"
            write(str(out_path), [atoms])
            return out_path
        with mock.patch("adapters.acquisition.acquire", adapter):
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(self.run_dir), "--stage", "acquisition", "--auto-mock-judges"] )
        self.assertEqual(code, cli.EXIT_SUCCESS)
        for rel in ("artifacts/acquisition_candidates.extxyz", "artifacts/acquisition.manifest.json", "artifacts/acquisition_protection_audit.json"):
            self.assertTrue((self.run_dir / rel).is_file(), rel)
        c = __import__("workflow.controller", fromlist=["RunController"]).RunController(self.run_dir)
        self.assertEqual(c.stage("acquisition")["status"], "completed")
        self.assertEqual(c.stage("acquisition")["gate"], "PASS")


if __name__ == "__main__":
    unittest.main()
