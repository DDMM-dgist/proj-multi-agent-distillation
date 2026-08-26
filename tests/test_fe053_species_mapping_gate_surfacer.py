"""FE-053: deterministic species->type-index mapping criterion-evidence surfacer for the
Teacher-labeling gate. The mapping already exists in the labeling manifest (and is confirmed by the
FE-049 validate_species_mapping_consistency cross-check that passed verify_recovery_execution), but
before FE-053 it was never surfaced into the gate's bounded evidence, so judges could only observe
that a species_mapping_evidence FIELD existed and REVISEd for the missing exact mapping. These tests
pin the surfacer that folds the exact ordered map + attestation + cross-source agreement into the
gate packet's validation_outcomes.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtimes.pydantic_ai.cli import _species_mapping_gate_evidence


def _manifest(tmp: Path, name: str, evidence: dict) -> Path:
    p = tmp / name
    p.write_text(json.dumps({
        "teacher_checkpoint_sha256": "a" * 64,
        "species_mapping_evidence": evidence,
    }, indent=2))
    return p


_ATTESTED_EVIDENCE = {
    "declared_chemical_symbols": ["O", "Si"],
    "runtime_chemical_species_to_atom_type_map": {"O": 0, "Si": 1},
    "compiled_model_type_names_map": {"O": 0, "Si": 1},
    "fallback_applied": False,
    "fallback_reason": None,
}


class Fe053SpeciesMappingSurfacerTests(unittest.TestCase):
    def test_1_surfaces_exact_mapping_when_attested(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            manifest = _manifest(tmp, "teacher_labels.manifest.json", _ATTESTED_EVIDENCE)
            from workflow.integrity import sha256_file
            out = _species_mapping_gate_evidence(None, "teacher_labeling", [manifest])
            self.assertIsNotNone(out)
            self.assertEqual(out["kind"], "species_mapping_criterion_evidence")
            self.assertEqual(out["stage"], "teacher_labeling")
            self.assertTrue(out["ready"])
            self.assertTrue(out["attested"])
            self.assertTrue(out["agree"])
            self.assertEqual(out["blocking_gaps"], [])
            self.assertEqual(out["species_to_type_index_map"], {"O": 0, "Si": 1})
            self.assertEqual(out["manifest_sha256"], sha256_file(manifest))
            self.assertIn("declared_config", out["sources_cross_checked"])

    def test_2_none_when_no_species_mapping_declared(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            plain = tmp / "some_report.json"
            plain.write_text(json.dumps({"unrelated": True}))
            extxyz = tmp / "teacher_labeled.extxyz"
            extxyz.write_text("0\n")
            out = _species_mapping_gate_evidence(None, "teacher_labeling", [plain, extxyz])
            self.assertIsNone(out)

    def test_3_failed_criterion_on_non_attestation(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            evidence = {
                "declared_chemical_symbols": ["O", "Si"],
                "runtime_chemical_species_to_atom_type_map": {},  # empty -> not attested
                "fallback_applied": False,
            }
            manifest = _manifest(tmp, "teacher_labels.manifest.json", evidence)
            out = _species_mapping_gate_evidence(None, "teacher_labeling", [manifest])
            self.assertIsNotNone(out)
            self.assertFalse(out["ready"])
            self.assertFalse(out["attested"])
            self.assertTrue(out["blocking_gaps"])

    def test_4_generic_manifest_discovery_not_by_filename(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            # Deliberately NOT named teacher_labels.manifest.json.
            manifest = _manifest(tmp, "arbitrary_labeling_record.json", _ATTESTED_EVIDENCE)
            out = _species_mapping_gate_evidence(None, "teacher_labeling", [manifest])
            self.assertIsNotNone(out)
            self.assertEqual(out["species_to_type_index_map"], {"O": 0, "Si": 1})

    def test_5_surfacing_writes_no_file(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            manifest = _manifest(tmp, "teacher_labels.manifest.json", _ATTESTED_EVIDENCE)
            before = sorted(p.name for p in tmp.iterdir())
            _species_mapping_gate_evidence(None, "teacher_labeling", [manifest])
            after = sorted(p.name for p in tmp.iterdir())
            self.assertEqual(before, after)  # no evidence file materialized (surfacing only)

    def test_6_conflict_surfaced_as_failed_criterion(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            evidence = {
                "declared_chemical_symbols": ["O", "Si"],          # {O:0, Si:1}
                "runtime_chemical_species_to_atom_type_map": {"O": 1, "Si": 0},  # disagrees
                "compiled_model_type_names_map": {"O": 0, "Si": 1},
                "fallback_applied": False,
            }
            manifest = _manifest(tmp, "teacher_labels.manifest.json", evidence)
            out = _species_mapping_gate_evidence(None, "teacher_labeling", [manifest])
            self.assertIsNotNone(out)
            self.assertFalse(out["ready"])
            self.assertFalse(out["agree"])
            self.assertTrue(out["blocking_gaps"])


if __name__ == "__main__":
    unittest.main()
