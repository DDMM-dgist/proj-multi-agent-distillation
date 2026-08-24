"""A stage's producer input evidence packet must never contain the stage's OWN declared outputs.

The producer only echoes the Controller's authoritative proposal to PRODUCE a stage's outputs, so
feeding it that stage's prior outputs as input evidence is semantically wrong -- and, concretely,
it was the blast radius behind the Stage-8 re-gate context overflow (recovery-004 correction): on a
re-run the prior accuracy_report.json is a registered artifact AND is named in the evaluation
proposal's ``report_path``/``labeled_output`` parameters, so it was pulled into the producer packet
and expanded by the four-channel accuracy adapter until the producer prompt overflowed. The GATE
packet, built separately from the declared outputs, still surfaces that rich summary to the Judges.
"""
from __future__ import annotations

import unittest

from runtimes.pydantic_ai.cli import _stage_input_artifact_paths


class StageInputExcludesOwnOutputsTests(unittest.TestCase):
    def _artifacts(self, *paths):
        return [{"path": p} for p in paths]

    def test_own_output_named_in_parameters_is_excluded(self):
        # evaluation names both its outputs (report_path/labeled_output) AND a genuine input
        # (frames_path) in parameters; only the genuine input survives into the producer packet.
        proposal = {"parameters": {
            "frames_path": "/run/artifacts/teacher_reference_predictions.extxyz",
            "labeled_output": "/run/artifacts/evaluated.extxyz",
            "report_path": "/run/artifacts/accuracy_report.json",
        }}
        artifacts = self._artifacts(
            "/run/artifacts/teacher_reference_predictions.extxyz",
            "/run/artifacts/evaluated.extxyz",
            "/run/artifacts/accuracy_report.json",
        )
        result = _stage_input_artifact_paths(
            proposal, artifacts,
            own_outputs=["/run/artifacts/evaluated.extxyz",
                         "/run/artifacts/accuracy_report.json"])
        self.assertEqual(result, ["/run/artifacts/teacher_reference_predictions.extxyz"])

    def test_own_outputs_excluded_even_on_registry_fallback(self):
        # No parameter names a registered artifact -> the helper falls back to the registry, but a
        # stage's own outputs are still stripped from that fallback.
        proposal = {"parameters": {"unrelated_flag": "true"}}
        artifacts = self._artifacts(
            "/run/artifacts/upstream_input.extxyz",
            "/run/artifacts/accuracy_report.json",
        )
        result = _stage_input_artifact_paths(
            proposal, artifacts, own_outputs=["/run/artifacts/accuracy_report.json"])
        self.assertEqual(result, ["/run/artifacts/upstream_input.extxyz"])

    def test_no_own_outputs_declared_preserves_prior_behavior(self):
        proposal = {"parameters": {"dataset": "/run/artifacts/a.extxyz"}}
        artifacts = self._artifacts("/run/artifacts/a.extxyz", "/run/artifacts/b.extxyz")
        self.assertEqual(_stage_input_artifact_paths(proposal, artifacts),
                         ["/run/artifacts/a.extxyz"])


if __name__ == "__main__":
    unittest.main()
