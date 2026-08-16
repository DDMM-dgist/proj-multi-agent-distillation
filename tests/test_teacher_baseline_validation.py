"""R20 forensic-audit checklist item 5 (validator side): validate_teacher_baseline_report must
itself reject a teacher_baseline manifest whose species_mapping is not attested -- the bounded
evidence summary exposing `species_mapping_attested=False` to a Judge (see
tests/test_bounded_evidence_registry.py::TeacherBaselineEvidenceSummaryTests) is only half of
Scope B/C; the validator that gates the stage must independently refuse to accept such a report."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validation.teacher_baseline import validate_teacher_baseline_report
from workflow.integrity import artifact_digest


def _evidence_entry(role: str, path: Path) -> dict:
    return {"role": role, "path": str(path), "integrity": artifact_digest(path)}


def _write_manifest(tmp: Path, *, species_mapping: dict) -> Path:
    teacher_config = tmp / "teacher.yaml"
    teacher_config.write_text("kind: mock\n", encoding="utf-8")
    distillation_scope = tmp / "distillation_scope.json"
    distillation_scope.write_text(json.dumps({"scope": "bulk"}), encoding="utf-8")
    validation_profile = tmp / "validation_profile.json"
    validation_profile.write_text(json.dumps({"profile": "teacher_baseline"}), encoding="utf-8")

    payload = {
        "schema_version": 1,
        "profile": "teacher_baseline",
        "teacher": {"kind": "mock", "config": str(teacher_config), "model_sha256": "abc"},
        "distillation_scope": str(distillation_scope),
        "validation_profile": str(validation_profile),
        "deployment_domain": {"structure_classes": ["bulk"], "dft_labels_used": False,
                              "protected_reference_labels_used": False},
        "applicability": {"status": "CONDITIONAL", "limitations": ["scope-limited"]},
        "species_mapping": species_mapping,
        "checks": [{
            "domain": "operational_teacher_inference",
            "observable": "fresh_teacher_energy_force_finiteness", "status": "PASS",
            "value": 1.0, "unit": "eV/Angstrom",
            "criterion": {"operator": "max", "threshold": 1.0e12},
            "purpose": "deployment_stability", "reference_source": "teacher",
            "protocol": "fresh Teacher inference on declared operational structures",
        }],
        "evidence": [
            _evidence_entry("teacher_config", teacher_config),
            _evidence_entry("distillation_scope", distillation_scope),
            _evidence_entry("validation_profile", validation_profile),
        ],
    }
    manifest_path = tmp / "teacher_baseline.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


class TeacherBaselineSpeciesMappingValidatorTests(unittest.TestCase):
    def test_unattested_species_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_manifest(Path(tmp), species_mapping={
                "declared_chemical_symbols": ["O", "Si"],
                "declared_chemical_species_to_atom_type_map": None,
                "runtime_chemical_species_to_atom_type_map": None,
                "fallback_applied": True,
                "fallback_reason": "chemical_species_to_atom_type_map required",
            })
            with self.assertRaises(ValueError) as ctx:
                validate_teacher_baseline_report(manifest_path)
        self.assertIn("species_mapping is not attested", str(ctx.exception))

    def test_missing_species_mapping_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_manifest(Path(tmp), species_mapping=None)
            payload = json.loads(manifest_path.read_text())
            del payload["species_mapping"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                validate_teacher_baseline_report(manifest_path)
        self.assertIn("requires species_mapping evidence", str(ctx.exception))

    def test_attested_species_mapping_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_manifest(Path(tmp), species_mapping={
                "declared_chemical_symbols": ["O", "Si"],
                "declared_chemical_species_to_atom_type_map": None,
                "runtime_chemical_species_to_atom_type_map": {"O": "O", "Si": "Si"},
                "fallback_applied": True,
                "fallback_reason": "chemical_species_to_atom_type_map required",
            })
            payload = validate_teacher_baseline_report(manifest_path)
        self.assertEqual(payload["species_mapping"]["runtime_chemical_species_to_atom_type_map"],
                         {"O": "O", "Si": "Si"})

    def test_no_mapping_convention_declared_is_attested_by_default(self):
        # A calculator kind (e.g. EMT, or a foundation model that auto-detects species) that
        # declares no chemical_symbols/chemical_species_to_atom_type_map convention at all must
        # not be forced to fabricate a runtime mapping -- see
        # adapters.teacher.species_mapping_is_attested.
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_manifest(Path(tmp), species_mapping={
                "declared_chemical_symbols": None,
                "declared_chemical_species_to_atom_type_map": None,
                "runtime_chemical_species_to_atom_type_map": None,
                "fallback_applied": False,
                "fallback_reason": None,
            })
            payload = validate_teacher_baseline_report(manifest_path)
        self.assertFalse(payload["species_mapping"]["fallback_applied"])


if __name__ == "__main__":
    unittest.main()
