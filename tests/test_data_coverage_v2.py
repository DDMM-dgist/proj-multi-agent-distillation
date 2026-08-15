import json
import tempfile
import unittest
from pathlib import Path

from validation.data_coverage import validate_data_coverage_report
from validation.report import evidence_record
from workflow.integrity import sha256_file


class DataCoverageV2Tests(unittest.TestCase):
    def _write_fixture(self, root, overrides=None, policy_content=None, directed_coverage=None):
        reference_manifest = root / "teacher_training_split_manifest.json"
        reference_manifest.write_text(json.dumps({"train": list(range(10))}))

        protected_report = root / "protected_reference_report.json"
        protected_report.write_text(json.dumps({
            "status": "PASS",
            "reference": {"logical_frames": 3, "protected_source_rows": 5},
        }))
        protected_report_sha256 = sha256_file(protected_report)

        policy_path = root / "dataset_policy.yaml"
        policy_path.write_text(policy_content or (
            "kind: generic\n"
            "provenance:\n"
            "  source_dataset_access: full\n"
            "  split_membership_status: reconstructed_unverified_cross_version_rng\n"
            "  deployed_checkpoint_linkage_status: bit_exact_tensor_match\n"
        ))

        evidence = [
            evidence_record("dataset_policy", policy_path),
        ]

        if directed_coverage is None:
            directed_coverage = [
                {
                    "direction": "teacher_support",
                    "query_population": "candidate_population",
                    "reference_population": "teacher_train_partition",
                    "reference_role": "teacher_train_partition",
                    "n_reference_frames": 9140,
                    "n_reference_atoms": 1009444,
                    "structural_method": None,
                    "excluded_partitions": ["validation", "test"],
                    "reference_manifest_path": "teacher_training_split_manifest.json",
                    "reference_manifest_sha256": sha256_file(reference_manifest),
                    "split_membership_verification": {
                        "status": "reconstructed_unverified_cross_version_rng",
                        "caveat": "torch 2.6.0-vs-2.12.1 cross-version RNG identity is unconfirmed",
                    },
                },
                {
                    "direction": "deployment_coverage",
                    "query_population": "deployment_target_population",
                    "reference_population": "student_training_dataset",
                    "reference_role": "student_training_dataset",
                    "n_reference_frames": 2134,
                    "n_reference_atoms": 321256,
                    "structural_method": None,
                },
            ]

        payload = {
            "schema_version": 2,
            "directed_coverage": directed_coverage,
            "protected_reference_status": {
                "role": "protected_reference_pointer",
                "report_path": "protected_reference_report.json",
                "report_sha256": protected_report_sha256,
                "required_logical_frames": 3,
                "required_protected_source_rows": 5,
            },
            "reference_population_partition_overlap": {
                "total": 2134,
                "partitions": {"teacher_train": 1724, "teacher_validation": 202, "teacher_test": 208},
            },
            "dataset_policy": str(policy_path),
            "identified_gaps": [],
            "limitations": [],
            "evidence": evidence,
        }
        if overrides:
            payload.update(overrides)
        manifest_path = root / "data_coverage.json"
        manifest_path.write_text(json.dumps(payload, default=str))
        return manifest_path

    def test_valid_v2_report_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_fixture(Path(tmp))
            result = validate_data_coverage_report(manifest_path)
            self.assertEqual(result["schema_version"], 2)

    def test_unknown_schema_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_fixture(Path(tmp), overrides={"schema_version": 3})
            with self.assertRaises(ValueError):
                validate_data_coverage_report(manifest_path)

    def test_directed_coverage_requires_non_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_fixture(Path(tmp), overrides={"directed_coverage": []})
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("directed_coverage", str(ctx.exception))

    def test_directed_coverage_entry_requires_direction_and_populations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root, directed_coverage=[
                {
                    "query_population": "candidate_population",
                    "reference_population": "student_training_dataset",
                    "reference_role": "student_training_dataset",
                    "n_reference_frames": 5,
                    "n_reference_atoms": 50,
                    "structural_method": None,
                },
            ])
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("direction", str(ctx.exception))

    def test_required_directions_enforced_when_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_fixture(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path, required_directions=["nonexistent_direction"])
            self.assertIn("nonexistent_direction", str(ctx.exception))

    def test_required_directions_pass_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_fixture(Path(tmp))
            result = validate_data_coverage_report(
                manifest_path, required_directions=["teacher_support", "deployment_coverage"]
            )
            self.assertEqual(result["schema_version"], 2)

    def test_teacher_train_partition_role_requires_validation_and_test_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["directed_coverage"][0]["excluded_partitions"] = ["test"]
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("excluded_partitions", str(ctx.exception))

    def test_teacher_train_partition_role_rejects_tampered_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["directed_coverage"][0]["reference_manifest_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("reference_manifest_sha256", str(ctx.exception))

    def test_teacher_train_partition_role_rejects_missing_reference_manifest_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["directed_coverage"][0]["reference_manifest_path"] = "does_not_exist.json"
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("does not exist", str(ctx.exception))

    def test_other_reference_roles_do_not_require_exclusions_or_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_fixture(Path(tmp))
            result = validate_data_coverage_report(manifest_path)
            # The deployment_coverage entry (reference_role=student_training_dataset) has
            # neither excluded_partitions nor a reference_manifest_path, and still passes.
            deployment_entry = next(
                e for e in result["directed_coverage"] if e["direction"] == "deployment_coverage"
            )
            self.assertNotIn("excluded_partitions", deployment_entry)

    def test_split_membership_verification_cannot_claim_cryptographic_without_caveat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["directed_coverage"][0]["split_membership_verification"] = {
                "status": "cryptographically_verified",
            }
            manifest_path.write_text(json.dumps(payload, default=str))
            # Allowed: cryptographically_verified does not require a caveat.
            result = validate_data_coverage_report(manifest_path)
            self.assertEqual(
                result["directed_coverage"][0]["split_membership_verification"]["status"],
                "cryptographically_verified",
            )

    def test_split_membership_verification_requires_caveat_when_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["directed_coverage"][0]["split_membership_verification"] = {
                "status": "reconstructed_unverified_cross_version_rng",
            }
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("caveat", str(ctx.exception))

    def test_split_membership_verification_rejects_unknown_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["directed_coverage"][0]["split_membership_verification"] = {
                "status": "trust_me",
            }
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError):
                validate_data_coverage_report(manifest_path)

    def test_protected_reference_status_rejects_wrong_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["protected_reference_status"]["role"] = "inline_dft_metrics"
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("role", str(ctx.exception))

    def test_protected_reference_status_rejects_wrong_required_frame_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["protected_reference_status"]["required_logical_frames"] = 9999
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("must match the hash-verified", str(ctx.exception))

    def test_protected_reference_status_rejects_tampered_report_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["protected_reference_status"]["report_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("report_sha256", str(ctx.exception))

    def test_reference_population_partition_overlap_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            del payload["reference_population_partition_overlap"]
            manifest_path.write_text(json.dumps(payload, default=str))
            result = validate_data_coverage_report(manifest_path)
            self.assertEqual(result["schema_version"], 2)

    def test_reference_population_partition_overlap_requires_consistent_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root)
            payload = json.loads(manifest_path.read_text())
            payload["reference_population_partition_overlap"]["total"] = 9999
            manifest_path.write_text(json.dumps(payload, default=str))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("sum to total", str(ctx.exception))

    def test_dataset_policy_rejects_legacy_shape_without_provenance_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(
                root, policy_content="kind: generic\nteacher_training_data_access: representative\n"
            )
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("provenance", str(ctx.exception))

    def test_dataset_policy_requires_all_three_provenance_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_fixture(root, policy_content=(
                "kind: generic\n"
                "provenance:\n"
                "  source_dataset_access: full\n"
            ))
            with self.assertRaises(ValueError) as ctx:
                validate_data_coverage_report(manifest_path)
            self.assertIn("split_membership_status", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
