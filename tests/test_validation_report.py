import json
import tempfile
import unittest
from pathlib import Path

from validation.report import evidence_record, validate_validation_report


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


if __name__ == "__main__":
    unittest.main()
