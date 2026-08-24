"""Regression test for the REFERENCE_VALIDATION_VERIFIED_REUSE execution path.

reference_validation historically had exactly ONE executable path: run FRESH Teacher
inference over the protected reference every time (``label_with_teacher``). That collides
with a no-Teacher-inference campaign whose reference contract nonetheless keeps
ORIGINAL_HELDOUT_FIDELITY applicable. The verified-reuse path
(``_reference_validation_verified_reuse``) closes that gap: it deterministically proves an
ALREADY-EXISTING Teacher-vs-reference artifact is identity-, provenance-, and
scope-compatible with THIS run's reference contract + Teacher identity, then re-derives the
Teacher-vs-DFT metrics from that verified artifact WITHOUT any fresh Teacher inference.

This test drives the REAL executor dispatch (``_exec_validate_teacher_reference``) against
the real, on-disk historical Teacher-vs-reference evidence recovered from the committed
SiO2-x campaign. It proves:

  * all-conditions-pass produces a VERIFIED_HISTORICAL_REUSE report, a byte-identical
    prediction artifact (SHA preserved), and NEVER calls ``label_with_teacher`` (monkeypatched
    to fail-loud);
  * a tampered Teacher-model identity, a tampered protected-reference structure hash, and a
    mismatched declared historical-prediction SHA each fail closed with
    AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED and the exact failing check -- never a
    silent pass and never a fresh-inference fallback.

The test is skipped (not failed) when the committed campaign artifacts are absent, so it
stays green in a stripped checkout while still exercising the real integration where the
evidence exists.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

_ROOT = Path("/home/hyunjin/distill-real-user")
_R31 = _ROOT / "runs" / "sio2-sox-allegro-simplenn-r31"
_HIST_REPORT = _R31 / "artifacts" / "reference_validation.json"
_HIST_PRED = _R31 / "artifacts" / "teacher_reference_predictions.extxyz"
_TEACHER_CONFIG = _R31 / "inputs" / "000-teacher.allegro.yaml"
_BASE_REFERENCE_YAML = _R31 / "inputs" / "006-reference.yaml"

_HAVE_ARTIFACTS = all(p.is_file() for p in (_HIST_REPORT, _HIST_PRED, _TEACHER_CONFIG, _BASE_REFERENCE_YAML))


@unittest.skipUnless(_HAVE_ARTIFACTS, "committed SiO2-x reference-validation artifacts are absent")
class ReferenceValidationVerifiedReuseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="verified-reuse-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.hist_pred_sha = json.loads(_HIST_REPORT.read_text())["prediction_artifact"]["integrity"]["sha256"]

    # --- reference contract with a declared historical_teacher_prediction block ------------
    def _reference_yaml(self, *, declared_sha=None) -> Path:
        cfg = yaml.safe_load(_BASE_REFERENCE_YAML.read_text())
        cfg["historical_teacher_prediction"] = {
            "path": str(_HIST_PRED),
            "sha256": declared_sha if declared_sha is not None else self.hist_pred_sha,
        }
        out = self.tmp / "c12-reference.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return out

    def _proposal(self, *, reference_yaml, historical_report):
        return {"parameters": {
            "reference_yaml": str(reference_yaml),
            "teacher_config": str(_TEACHER_CONFIG),
            "predictions_path": str(self.tmp / "teacher_reference_predictions.extxyz"),
            "report_path": str(self.tmp / "reference_validation.json"),
            "historical_report": str(historical_report),
            "historical_predictions": str(_HIST_PRED),
        }}

    def _mutated_report(self, mutate) -> Path:
        payload = copy.deepcopy(json.loads(_HIST_REPORT.read_text()))
        mutate(payload)
        out = self.tmp / "tampered_historical_report.json"
        out.write_text(json.dumps(payload))
        return out

    # --- positive: verified reuse, no fresh Teacher inference ------------------------------
    def test_all_conditions_pass_verified_reuse_without_teacher_inference(self):
        from runtimes.pydantic_ai import executors

        proposal = self._proposal(reference_yaml=self._reference_yaml(),
                                  historical_report=_HIST_REPORT)
        with mock.patch("adapters.acquisition.label_with_teacher",
                        side_effect=AssertionError("fresh Teacher inference must NOT run")):
            result = executors._exec_validate_teacher_reference(proposal)

        report = result["report"]
        self.assertEqual(result["evidence_source"], "VERIFIED_HISTORICAL_REUSE")
        self.assertEqual(report["historical_prediction_policy"], "VERIFIED_HISTORICAL_REUSE")
        self.assertEqual(report["evidence_source"], "VERIFIED_HISTORICAL_REUSE")
        # byte-identical prediction artifact: SHA preserved
        self.assertEqual(report["prediction_artifact"]["integrity"]["sha256"], self.hist_pred_sha)
        self.assertEqual(result["predictions_integrity"]["sha256"], self.hist_pred_sha)
        # every declared verified-reuse condition recorded as VERIFIED
        conditions = report["reuse_verification"]["conditions"]
        self.assertTrue(conditions)
        self.assertTrue(all(c["status"] == "VERIFIED" for c in conditions))
        checks = {c["check"] for c in conditions}
        for required in {"1a", "1b", "2", "3a", "3b", "3c", "4a", "4b", "5a", "5b", "5c", "6a", "7a", "8"}:
            self.assertIn(required, checks)
        # the report round-trips through the deterministic validator in reuse mode
        from validation.reference_validation import validate_reference_validation_report
        validate_reference_validation_report(
            self.tmp / "reference_validation.json",
            reference_yaml=str(self._reference_yaml()),
            teacher_config=str(_TEACHER_CONFIG),
            submitted_artifacts=[self.tmp / "reference_validation.json",
                                 self.tmp / "teacher_reference_predictions.extxyz"],
            reuse_verified_historical=True)

    # --- contract dispatch: the external validation_manifest contract must carry the
    # reuse option, else the gate-time re-validation rejects the historical SHA. This is the
    # exact gap that aborted the C12 campaign AFTER the executor produced a valid reuse report.
    def test_validation_manifest_contract_requires_reuse_option(self):
        from runtimes.pydantic_ai import executors
        from workflow.contracts import validate_validation_manifest

        proposal = self._proposal(reference_yaml=self._reference_yaml(),
                                  historical_report=_HIST_REPORT)
        with mock.patch("adapters.acquisition.label_with_teacher",
                        side_effect=AssertionError("fresh Teacher inference must NOT run")):
            executors._exec_validate_teacher_reference(proposal)

        manifest = self.tmp / "reference_validation.json"
        submitted = [manifest, self.tmp / "teacher_reference_predictions.extxyz"]
        validator = "validation.reference_validation.validate_reference_validation_report"

        # WITH the reuse option (as the Controller stage contract now declares): accepted.
        validate_validation_manifest(manifest, validator,
                                     options={"reuse_verified_historical": True},
                                     submitted_artifacts=submitted)

        # WITHOUT the reuse option: the deterministic gate re-validation fails closed on the
        # historical-SHA fresh-output guard -- proving the contract option is load-bearing.
        with self.assertRaises(ValueError) as ctx:
            validate_validation_manifest(manifest, validator, options=None,
                                         submitted_artifacts=submitted)
        self.assertIn("historical Teacher prediction SHA", str(ctx.exception))

    # --- teeth: tampered Teacher model identity fails closed -------------------------------
    def test_tampered_teacher_model_sha_blocks(self):
        from runtimes.pydantic_ai import executors

        def flip_model_sha(payload):
            payload["teacher"]["model_sha256"] = "0" * 64

        proposal = self._proposal(reference_yaml=self._reference_yaml(),
                                  historical_report=self._mutated_report(flip_model_sha))
        with self.assertRaises(executors._ReferenceReuseBlocked) as ctx:
            executors._exec_validate_teacher_reference(proposal)
        msg = str(ctx.exception)
        self.assertIn("AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED", msg)
        self.assertIn("check 1a", msg)

    # --- teeth: tampered protected-reference structure hash fails closed -------------------
    def test_tampered_reference_structures_sha_blocks(self):
        from runtimes.pydantic_ai import executors

        def flip_struct_sha(payload):
            payload["reference"]["structures_integrity"]["sha256"] = "1" * 64

        proposal = self._proposal(reference_yaml=self._reference_yaml(),
                                  historical_report=self._mutated_report(flip_struct_sha))
        with self.assertRaises(executors._ReferenceReuseBlocked) as ctx:
            executors._exec_validate_teacher_reference(proposal)
        msg = str(ctx.exception)
        self.assertIn("AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED", msg)
        self.assertIn("check 2", msg)

    # --- teeth: mismatched declared historical-prediction SHA fails closed -----------------
    def test_declared_historical_prediction_sha_mismatch_blocks(self):
        from runtimes.pydantic_ai import executors

        proposal = self._proposal(reference_yaml=self._reference_yaml(declared_sha="2" * 64),
                                  historical_report=_HIST_REPORT)
        with self.assertRaises(executors._ReferenceReuseBlocked) as ctx:
            executors._exec_validate_teacher_reference(proposal)
        msg = str(ctx.exception)
        self.assertIn("AUTONOMOUS_REPRODUCIBLE_12_STAGE_WORKFLOW_BLOCKED", msg)
        self.assertIn("check 4b", msg)


if __name__ == "__main__":
    unittest.main()
