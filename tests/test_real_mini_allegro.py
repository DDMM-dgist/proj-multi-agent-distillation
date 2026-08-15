from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parent.parent


class RealMiniAllegroTests(unittest.TestCase):
    def test_real_mini_allegro_teacher_inference(self):
        try:
            import nequip  # noqa: F401
        except Exception as exc:  # pragma: no cover - base env documents known torchvision issue
            self.skipTest(f"NequIP unavailable in this Python environment: {type(exc).__name__}: {exc}")
        from adapters import load_config
        from adapters.acquisition import label_with_teacher
        from workflow.integrity import artifact_digest

        teacher_cfg_path = ROOT / "configs/runs/sio2-sox-allegro-simplenn-r2/teacher.allegro.yaml"
        source = ROOT / "local_inputs/sio2_fresh/seed_pool_11424/bulk_cryst/bulk_cryst.xyz"
        out_dir = ROOT / "work/mini_allegro"
        out_dir.mkdir(parents=True, exist_ok=True)
        mini = out_dir / "mini_structures.extxyz"
        labeled = out_dir / "mini_teacher_labeled.extxyz"
        manifest_path = out_dir / "mini_teacher_labels.manifest.json"
        validation_path = out_dir / "mini_teacher_validation.json"

        frames = read(str(source), index=slice(0, 3))
        self.assertTrue(1 <= len(frames) <= 3)
        for atoms in frames:
            symbols = set(atoms.get_chemical_symbols())
            self.assertTrue(symbols <= {"O", "Si"})
            self.assertIn("O", symbols)
            self.assertIn("Si", symbols)
            atoms.info.setdefault("parent_structure_id", atoms.info.get("structure_id", "mini-seed"))
        write(str(mini), frames)
        manifest = label_with_teacher(load_config(teacher_cfg_path), mini, labeled, manifest_path)
        labeled_frames = read(str(labeled), index=":")
        energies = [a.info["teacher_energy"] for a in labeled_frames]
        fmax = []
        for atoms in labeled_frames:
            forces = np.asarray(atoms.arrays["teacher_forces"], dtype=float)
            self.assertEqual(forces.shape, (len(atoms), 3))
            self.assertTrue(np.all(np.isfinite(forces)))
            fmax.append(float(np.max(np.linalg.norm(forces, axis=1))))
        self.assertTrue(all(math.isfinite(float(e)) for e in energies))
        self.assertEqual(manifest["teacher_model_sha256"],
                         "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57")
        self.assertEqual(manifest["calculator"]["kwargs"]["chemical_symbols"], ["O", "Si"])
        record = {
            "schema_version": 1,
            "test": "real_mini_allegro_teacher_inference",
            "n_frames": len(labeled_frames),
            "energy_min": float(min(energies)),
            "energy_max": float(max(energies)),
            "force_fmax_max": float(max(fmax)),
            "teacher_manifest": str(manifest_path.resolve()),
            "teacher_manifest_integrity": artifact_digest(manifest_path),
            "labeled_output": str(labeled.resolve()),
            "labeled_output_integrity": artifact_digest(labeled),
            "mini_input": str(mini.resolve()),
            "mini_input_integrity": artifact_digest(mini),
            "environment": manifest["environment"],
            "teacher_model_sha256": manifest["teacher_model_sha256"],
            "teacher_config_sha256": manifest["teacher_config_sha256"],
            "element_mapping": manifest["calculator"]["kwargs"]["chemical_symbols"],
        }
        validation_path.write_text(json.dumps(record, indent=2) + "\n")
        self.assertTrue(validation_path.is_file())


    def test_real_mini_allegro_reference_validation_smoke(self):
        try:
            import nequip  # noqa: F401
        except Exception as exc:  # pragma: no cover - base env documents known torchvision issue
            self.skipTest(f"NequIP unavailable in this Python environment: {type(exc).__name__}: {exc}")
        import yaml
        from workflow.integrity import sha256_file
        from runtimes.pydantic_ai.executors import _exec_validate_teacher_reference

        teacher_cfg_path = ROOT / "configs/runs/sio2-sox-allegro-simplenn-r2/teacher.allegro.yaml"
        source = ROOT / "local_inputs/sio2_fresh/seed_pool_11424/bulk_cryst/bulk_cryst.xyz"
        out_dir = ROOT / "work/mini_allegro_reference_validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        reference_structures = out_dir / "mini_protected_reference.extxyz"
        predictions = out_dir / "teacher_reference_predictions.extxyz"
        report = out_dir / "reference_validation.json"
        protected_indices = out_dir / "protected_indices.txt"
        protection_manifest = out_dir / "protected_manifest.json"
        reference_yaml = out_dir / "reference.yaml"

        frames = read(str(source), index=slice(0, 2))
        self.assertEqual(len(frames), 2)
        for atoms in frames:
            atoms.info["config_type"] = atoms.info.get("config_type", "mini_bulk")
            atoms.info["dft_energy"] = 0.0
            atoms.arrays["dft_forces"] = np.zeros((len(atoms), 3), dtype=float)
        write(str(reference_structures), frames)
        protected_indices.write_text("760\n761\n")
        protection_manifest.write_text(json.dumps({"mapping": {
            "logical_test_frames": 2,
            "matched_logical_frames": 2,
            "unmatched_logical_frames": 0,
            "protected_source_rows": 2,
            "conflicting_label_duplicates": 0,
        }}))
        reference_yaml.write_text(yaml.safe_dump({
            "kind": "protected-existing-dft",
            "reference_id": "mini-real-allegro-reference",
            "reference_class": "ORIGINAL_TEACHER_TEST",
            "status": "AVAILABLE_AND_PROTECTED",
            "logical_test_frames": 2,
            "protected_source_rows": 2,
            "protection_manifest": str(protection_manifest),
            "protected_source_rows_file": str(protected_indices),
            "duplicate_equivalent": {"source_global_indices": [760, 761], "label_conflict": False},
            "prohibited_uses": [
                "student_training", "student_validation_tuning", "acquisition_seed",
                "augmentation_parent", "recovery_training",
            ],
            "structures": {"path": str(reference_structures), "logical_frames": 2,
                           "sha256": sha256_file(reference_structures)},
        }))
        result = _exec_validate_teacher_reference({"parameters": {
            "reference_yaml": str(reference_yaml),
            "teacher_config": str(teacher_cfg_path),
            "predictions_path": str(predictions),
            "report_path": str(report),
            "domain_fields": ["structural_domain"],
        }})
        self.assertTrue(Path(result["path"]).is_file())
        self.assertTrue(Path(result["predictions_path"]).is_file())
        payload = json.loads(report.read_text())
        self.assertEqual(payload["profile"], "teacher_reference_validation")
        self.assertEqual(payload["metrics"]["energy_normalization"], "per_atom")
        self.assertEqual(payload["teacher"]["model_sha256"],
                         "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57")


if __name__ == "__main__":
    unittest.main()
