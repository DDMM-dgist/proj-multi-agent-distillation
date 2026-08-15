from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
from ase import Atoms
from ase.io import write


def _write_reference_package(root: Path, frames, *, logical=2, rows=(760, 761), historical=None) -> Path:
    from workflow.integrity import sha256_file
    structures = root / "protected_reference.extxyz"
    write(str(structures), frames)
    indices = root / "protected_indices.txt"
    indices.write_text("\n".join(str(i) for i in rows) + "\n")
    manifest = root / "protected_manifest.json"
    manifest.write_text(json.dumps({"mapping": {
        "logical_test_frames": logical,
        "matched_logical_frames": logical,
        "unmatched_logical_frames": 0,
        "protected_source_rows": len(rows),
        "conflicting_label_duplicates": 0,
    }}))
    payload = {
        "kind": "protected-existing-dft",
        "reference_id": "synthetic-protected-reference",
        "reference_class": "ORIGINAL_TEACHER_TEST",
        "status": "AVAILABLE_AND_PROTECTED",
        "logical_test_frames": logical,
        "protected_source_rows": len(rows),
        "protection_manifest": str(manifest),
        "protected_source_rows_file": str(indices),
        "duplicate_equivalent": {"source_global_indices": [760, 761], "label_conflict": False},
        "prohibited_uses": [
            "student_training", "student_validation_tuning", "acquisition_seed",
            "augmentation_parent", "recovery_training",
        ],
        "structures": {"path": str(structures), "logical_frames": logical, "sha256": sha256_file(structures)},
    }
    if historical:
        payload["historical_teacher_prediction"] = historical
    reference = root / "reference.yaml"
    reference.write_text(yaml.safe_dump(payload))
    return reference


def _frames(n=2, *, teacher_nan=False):
    frames = []
    for i in range(n):
        a = Atoms("Cu2", positions=[[0, 0, 0], [1.8 + i * 0.1, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["config_type"] = "bulk" if i == 0 else "surface"
        a.info["dft_energy"] = -2.0 - i
        a.info["teacher_energy"] = float("nan") if teacher_nan and i == 0 else -1.9 - i
        a.arrays["dft_forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        a.arrays["teacher_forces"] = np.array([[0.12, 0.0, 0.0], [-0.08, 0.0, 0.0]])
        frames.append(a)
    return frames


def _teacher_config(root: Path, *, model_text="teacher") -> Path:
    model = root / "teacher.nequip.pth"
    model.write_text(model_text)
    cfg = root / "teacher.yaml"
    cfg.write_text(yaml.safe_dump({"kind": "mock", "checkpoint": str(model)}))
    return cfg


def _metric_subset(src):
    return {
        "n_frames": int(src["n_frames"]),
        "n_atoms": int(src["n_atoms"]),
        "energy_mae": float(src["e_raw_mae_meV"]),
        "energy_rmse": float(src["e_raw_rmse_meV"]),
        "force_component_mae": float(src["f_mae"]),
        "force_component_rmse": float(src["f_rmse"]),
    }


def _valid_report(root: Path, *, teacher_nan=False, historical=False):
    from adapters import load_config
    from adapters.teacher import teacher_model_reference
    from validation.four_channel_audit import channel
    from workflow.integrity import artifact_digest, sha256_file
    frames = _frames(teacher_nan=teacher_nan)
    pred = root / "teacher_reference_predictions.extxyz"
    write(str(pred), frames)
    historical_payload = None
    if historical:
        historical_payload = {"path": str(pred), "sha256": sha256_file(pred)}
    reference = _write_reference_package(root, [a.copy() for a in frames], historical=historical_payload)
    teacher = _teacher_config(root)
    metrics = channel(frames, "dft", "teacher", per_config_type=True, require_complete=True)
    teacher_cfg = load_config(teacher)
    model_path = Path(teacher_model_reference(teacher_cfg)).resolve()
    report = {
        "schema_version": 1,
        "profile": "teacher_reference_validation",
        "stage": "reference_validation",
        "protected_reference_use": "teacher_vs_dft_reference_validation_only",
        "historical_prediction_policy": "PROVENANCE_ONLY_NOT_USED_AS_FRESH_RESULT",
        "teacher": {
            "config": str(teacher.resolve()),
            "config_integrity": artifact_digest(teacher),
            "model": str(model_path),
            "model_integrity": artifact_digest(model_path),
            "model_sha256": sha256_file(model_path),
        },
        "reference": {
            "reference_id": "synthetic-protected-reference",
            "reference_yaml": str(reference.resolve()),
            "structures_path": str((root / "protected_reference.extxyz").resolve()),
            "logical_frames": 2,
            "protected_source_rows": 2,
            "structures_integrity": artifact_digest(root / "protected_reference.extxyz"),
        },
        "prediction_artifact": {
            "path": str(pred.resolve()),
            "integrity": artifact_digest(pred),
            "n_frames": 2,
            "labels": ["teacher_energy", "teacher_forces", "dft_energy", "dft_forces"],
        },
        "metrics": {
            "energy_normalization": "per_atom",
            "energy_unit": "meV/atom",
            "force_unit": "eV/Angstrom",
            "global": _metric_subset(metrics["all"]),
            "by_config_type": {k: _metric_subset(v) for k, v in metrics.items() if k != "all"},
            "domain_fields": {"config_type": "present", "structural_domain": "absent"},
        },
        "evidence": [
            {"role": "teacher_config", "path": str(teacher.resolve()), "integrity": artifact_digest(teacher)},
            {"role": "protected_reference_config", "path": str(reference.resolve()), "integrity": artifact_digest(reference)},
            {"role": "protected_reference_structures", "path": str((root / "protected_reference.extxyz").resolve()), "integrity": artifact_digest(root / "protected_reference.extxyz")},
            {"role": "teacher_reference_predictions", "path": str(pred.resolve()), "integrity": artifact_digest(pred)},
        ],
    }
    path = root / "reference_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path, reference, teacher, pred, report


class ReferenceValidationActionTests(unittest.TestCase):
    def test_route_and_authoritative_binding_exist(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage, _protection_consuming_action, _stage_config
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = _write_reference_package(root, _frames())
            teacher = _teacher_config(root)
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({
                "run_id": "reference-route",
                "inputs": [str(teacher), str(reference)],
                "stages": [{
                    "name": "reference_validation",
                    "command": None,
                    "outputs": ["artifacts/reference_validation.json", "artifacts/teacher_reference_predictions.extxyz"],
                    "gate": {"criteria": ["reference validation report is valid"]},
                }],
            }))
            c = RunController.initialize(workflow, root / "run")
            proposal, role = _proposal_from_stage(c, "reference_validation", _stage_config(c, "reference_validation"))
            self.assertEqual(role, "simulation")
            self.assertEqual(proposal["action_type"], "validate_teacher_reference")
            self.assertEqual(proposal["approval_boundary"], "costly_teacher_labeling")
            self.assertEqual(proposal["parameters"]["reference_yaml"], str(Path(c.state["inputs"][1]["snapshot"]).resolve()))
            self.assertEqual(proposal["expected_outputs"], ["artifacts/reference_validation.json", "artifacts/teacher_reference_predictions.extxyz"])
            self.assertFalse(_protection_consuming_action("validate_teacher_reference"))

    def test_student_data_protection_consuming_actions_remain_unchanged(self):
        from runtimes.pydantic_ai.cli import _protection_consuming_action
        self.assertTrue(_protection_consuming_action("acquire_structures"))
        self.assertTrue(_protection_consuming_action("label_with_teacher"))
        self.assertTrue(_protection_consuming_action("train_committee"))
        self.assertFalse(_protection_consuming_action("validate_teacher_reference"))

    def test_validator_accepts_metrics_and_hash_bound_prediction(self):
        from validation.reference_validation import validate_reference_validation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, reference, teacher, pred, _payload = _valid_report(root)
            out = validate_reference_validation_report(report, reference_yaml=reference, teacher_config=teacher,
                                                       submitted_artifacts=[report, pred])
            self.assertEqual(out["metrics"]["energy_normalization"], "per_atom")
            self.assertIn("bulk", out["metrics"]["by_config_type"])
            self.assertIn("surface", out["metrics"]["by_config_type"])

    def test_wrong_protected_reference_count_rejects(self):
        from validation.reference_validation import validate_reference_validation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, reference, teacher, pred, payload = _valid_report(root)
            payload["reference"]["logical_frames"] = 1
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "reference block mismatch"):
                validate_reference_validation_report(report, reference_yaml=reference, teacher_config=teacher,
                                                     submitted_artifacts=[report, pred])

    def test_wrong_teacher_sha_rejects(self):
        from validation.reference_validation import validate_reference_validation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, reference, teacher, pred, payload = _valid_report(root)
            payload["teacher"]["model_sha256"] = "0" * 64
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "teacher block mismatch"):
                validate_reference_validation_report(report, reference_yaml=reference, teacher_config=teacher,
                                                     submitted_artifacts=[report, pred])

    def test_historical_prediction_substitution_rejects(self):
        from validation.reference_validation import validate_reference_validation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, reference, teacher, pred, _payload = _valid_report(root, historical=True)
            with self.assertRaisesRegex(ValueError, "historical Teacher prediction"):
                validate_reference_validation_report(report, reference_yaml=reference, teacher_config=teacher,
                                                     submitted_artifacts=[report, pred])

    def test_missing_prediction_artifact_rejects(self):
        from validation.reference_validation import validate_reference_validation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, reference, teacher, pred, _payload = _valid_report(root)
            pred.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_reference_validation_report(report, reference_yaml=reference, teacher_config=teacher,
                                                     submitted_artifacts=[report, pred])

    def test_nan_prediction_rejects(self):
        from validation.reference_validation import validate_reference_validation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, reference, teacher, pred, _payload = _valid_report(root, teacher_nan=True)
            with self.assertRaisesRegex(ValueError, "finite teacher_energy"):
                validate_reference_validation_report(report, reference_yaml=reference, teacher_config=teacher,
                                                     submitted_artifacts=[report, pred])

    def test_student_semantics_claim_rejects(self):
        from validation.reference_validation import validate_reference_validation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, reference, teacher, pred, payload = _valid_report(root)
            payload["student_training"] = {"dataset": str(pred)}
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Student-training semantics"):
                validate_reference_validation_report(report, reference_yaml=reference, teacher_config=teacher,
                                                     submitted_artifacts=[report, pred])

    def test_executor_uses_label_with_teacher_primitive_and_validator(self):
        from runtimes.pydantic_ai.executors import _exec_validate_teacher_reference
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = _write_reference_package(root, _frames())
            teacher = _teacher_config(root)

            def fake_label_with_teacher(teacher_cfg, structures_path, out_path, manifest_path, include_stress=False):
                write(str(out_path), _frames())
                from workflow.integrity import sha256_file
                Path(manifest_path).write_text(json.dumps({"output": str(out_path), "sha256": sha256_file(out_path)}))
                return {"output": str(out_path), "sha256": sha256_file(out_path)}

            proposal = {"parameters": {"reference_yaml": str(reference), "teacher_config": str(teacher),
                                        "predictions_path": str(root / "pred.extxyz"),
                                        "report_path": str(root / "report.json")}}
            with mock.patch("adapters.acquisition.label_with_teacher", fake_label_with_teacher):
                result = _exec_validate_teacher_reference(proposal)
            self.assertTrue(Path(result["path"]).is_file())
            self.assertTrue(Path(result["predictions_path"]).is_file())

    def test_agent_spec_prompt_reads_utf8(self):
        from orchestration.specs import load_agent_spec
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_specs").mkdir()
            (root / "agents").mkdir()
            (root / "agents" / "judge.md").write_text("Judge prompt — with em dash\n", encoding="utf-8")
            spec_path = root / "agent_specs" / "judge.yaml"
            spec_path.write_text(yaml.safe_dump({
                "schema_version": 1,
                "name": "judge",
                "role_type": "reviewer",
                "description": "Read a UTF-8 prompt",
                "prompt": "agents/judge.md",
                "task_contract": "AgentTask",
                "result_contract": "JudgeVote",
                "capabilities": ["read_project_files"],
                "accepts": ["AgentTask"],
                "returns": ["JudgeVote"],
                "approval_boundaries": [],
                "delegates": [],
            }), encoding="utf-8")
            spec = load_agent_spec(spec_path, root=root)
            self.assertIn("—", spec.prompt)


if __name__ == "__main__":
    unittest.main()
