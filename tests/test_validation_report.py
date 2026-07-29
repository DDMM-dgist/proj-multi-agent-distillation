import json
import tempfile
import unittest
from pathlib import Path

from validation.report import evidence_record, validate_validation_report
from validation.teacher_baseline import validate_teacher_baseline_report
from validation.data_coverage import validate_data_coverage_report


class ValidationReportTests(unittest.TestCase):
    def write_report(self, root, check, evidence=None):
        path = root / "report.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "profile": "generic",
            "checks": [check],
            "evidence": evidence or [],
        }))
        return path

    def test_report_accepts_relative_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "trajectory.xyz"
            evidence.write_text("trajectory")
            record = evidence_record("trajectory", evidence)
            record["path"] = evidence.name
            report = self.write_report(root, {
                "domain": "structure", "observable": "density", "status": "PASS",
                "value": 2.1, "unit": "g/cm3",
                "criterion": {"operator": "target_tolerance", "target": 2.0,
                              "tolerance": 0.2},
            }, [record])
            validate_validation_report(report, ["density"], [evidence], True)

    def test_report_rejects_inconsistent_status_and_nonfinite_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.txt"
            evidence.write_text("x")
            record = evidence_record("input", evidence)
            report = self.write_report(root, {
                "domain": "stability", "observable": "drift", "status": "PASS",
                "value": 2.0, "unit": "x",
                "criterion": {"operator": "max", "threshold": 1.0},
            }, [record])
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                validate_validation_report(report)
            payload = json.loads(report.read_text())
            payload["checks"][0]["criterion"]["threshold"] = float("nan")
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "must be finite"):
                validate_validation_report(report)

    def test_report_rejects_mutated_or_unsubmitted_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.txt"
            evidence.write_text("before")
            report = self.write_report(root, {
                "domain": "structure", "observable": "rdf", "status": "RECORDED",
                "value": 1.5, "unit": "Angstrom", "criterion": None,
            }, [evidence_record("trajectory", evidence)])
            with self.assertRaisesRegex(ValueError, "not submitted"):
                validate_validation_report(report, submitted_artifacts=[report],
                                           require_submitted_evidence=True)
            evidence.write_text("after")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                validate_validation_report(report)

    def test_report_rejects_evidence_not_bound_to_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "declared.txt"
            external = root / "external.txt"
            allowed.write_text("declared")
            external.write_text("external")
            report = self.write_report(root, {
                "domain": "structure", "observable": "density", "status": "RECORDED",
                "value": 2.0, "unit": "g/cm3", "criterion": None,
            }, [evidence_record("trajectory", external)])
            with self.assertRaisesRegex(ValueError, "not bound to this run"):
                validate_validation_report(report, allowed_evidence=[allowed, report])

    def test_teacher_baseline_separates_reference_source_and_purpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_config = root / "teacher.yaml"
            teacher_config.write_text("kind: mock\n")
            validation_profile = root / "validation.yaml"
            validation_profile.write_text("kind: generic\n")
            distillation_scope = root / "scope.yaml"
            distillation_scope.write_text("deployment_domain: {system: test}\n")
            evidence = root / "teacher-trajectory.xyz"
            evidence.write_text("teacher evidence")
            report = root / "teacher_baseline.json"
            report.write_text(json.dumps({
                "schema_version": 1, "profile": "deployment-v1",
                "teacher": {"config": str(teacher_config)},
                "distillation_scope": str(distillation_scope),
                "validation_profile": str(validation_profile),
                "deployment_domain": {"structure_classes": ["liquid"]},
                "applicability": {"status": "CONDITIONAL", "limitations": ["high-T only"]},
                "checks": [{
                    "domain": "dynamics", "observable": "diffusion", "status": "RECORDED",
                    "value": 1.2, "unit": "A2/ps", "criterion": None,
                    "purpose": "student_teacher_fidelity", "reference_source": "teacher",
                    "protocol": "nvt-1000K-v1",
                }],
                "evidence": [evidence_record("teacher_config", teacher_config),
                             evidence_record("distillation_scope", distillation_scope),
                             evidence_record("validation_profile", validation_profile),
                             evidence_record("teacher_trajectory", evidence)],
            }))
            payload = validate_teacher_baseline_report(
                report, required_observables=["diffusion"],
                accepted_applicability=["SUPPORTED", "CONDITIONAL"],
                allowed_evidence=[teacher_config, distillation_scope,
                                  validation_profile, evidence],
                enforce_required_pass=True,
            )
            self.assertEqual(payload["applicability"]["status"], "CONDITIONAL")
            broken = json.loads(report.read_text())
            broken["checks"][0]["reference_source"] = "student"
            report.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "reference_source"):
                validate_teacher_baseline_report(report)

    def test_data_coverage_supports_unavailable_teacher_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "dataset.extxyz"
            evidence.write_text("dataset")
            dataset_policy = root / "dataset_policy.yaml"
            dataset_policy.write_text("teacher_training_data_access: unavailable\n")
            report = root / "coverage.json"
            payload = {
                "schema_version": 1,
                "teacher_training_data_access": "unavailable",
                "dataset_policy": str(dataset_policy),
                "coverage_status": "NOT_ASSESSABLE",
                "deployment_domain": {"structure_classes": ["crystal", "liquid"]},
                "dataset_sources": [{
                    "category": "generated_teacher_labeled", "n_parents": 10,
                    "n_frames": 100, "fraction": 1.0, "label_sources": ["teacher"],
                    "evidence_role": "distillation_dataset",
                }],
                "coverage_dimensions": {},
                "replay_policy": {"enabled": False},
                "identified_gaps": ["teacher training distribution unavailable"],
                "limitations": ["quantitative teacher-set coverage cannot be computed"],
                "evidence": [evidence_record("dataset_policy", dataset_policy),
                             evidence_record("distillation_dataset", evidence)],
            }
            report.write_text(json.dumps(payload))
            validate_data_coverage_report(report,
                                          allowed_evidence=[dataset_policy, evidence])
            payload["dataset_sources"][0]["fraction"] = 0.5
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "sum to 1"):
                validate_data_coverage_report(report)


if __name__ == "__main__":
    unittest.main()
