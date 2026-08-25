"""FE-041 regression: DATA_COVERAGE_GEOMETRY_ONLY_ACCESS_PROVENANCE.

The frozen scientific inputs of the fresh SiO2-x campaign truthfully declare
``teacher_training_data_access: representative_geometry_only`` -- the Teacher's training data is
available only as representative GEOMETRIES (zero DFT labels), a materially different evidence
state from a labelled ``representative`` subset. Before FE-041 the deterministic schema could only
represent ``full``/``representative``/``unavailable``, so the executor silently defaulted to
``representative`` and the truthful provenance was lost, driving all three Stage-4 Judges to REVISE
on an ambiguity that had nothing to do with the real structural coverage gaps.

FE-041 adds the typed access mode ``representative_geometry_only`` and makes the executor resolve
the value AUTHORITATIVELY from the frozen run-bound input(s) (bound ``dataset_policy`` /
``distillation_scope``) rather than from the LLM proposal, failing closed on conflicting frozen
declarations. It does NOT touch FE-039 coverage semantics: the real structural gaps still surface.

Tests A-G below map one-to-one to the FE-041 directive's required regressions.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_FROZEN_SCOPE = (_PROJECT / "docs"
                 / "FRESH_CAMPAIGN_FROZEN_POLICIES_v2_fresh_frozen_framework_validation"
                 / "02_deployment_scope_v2.json")


def _write_candidate(root: Path, config_types):
    from ase import Atoms
    from ase.io import write
    frames = []
    for i, ct in enumerate(config_types):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["parent_structure_id"] = f"p{i}"
        a.info["config_type"] = ct
        frames.append(a)
    candidate = root / "candidate.extxyz"
    write(str(candidate), frames)
    manifest = root / "acq.manifest.json"
    manifest.write_text(json.dumps({"n_frames": len(frames) + 4, "elements": ["Cu"]}))
    return candidate, manifest


def _write_policy(path: Path, access=None):
    import yaml
    doc = {"kind": "generic", "provenance": {"note": "fe041 regression policy"}}
    if access is not None:
        doc["teacher_training_data_access"] = access
    path.write_text(yaml.safe_dump(doc))
    return path


def _run(root: Path, config_types, deployment_domain, **extra):
    from runtimes.pydantic_ai.executors import _exec_build_data_coverage_report
    candidate, manifest = _write_candidate(root, config_types)
    params = {"candidate_dataset": str(candidate), "acquisition_manifest": str(manifest),
              "report_path": str(root / "report.json"),
              "deployment_domain": deployment_domain, **extra}
    return _exec_build_data_coverage_report({"parameters": params})


class TestA_SchemaAcceptsGeometryOnly(unittest.TestCase):
    """(A) The typed data-coverage schema accepts representative_geometry_only."""

    def test_access_modes_contains_geometry_only(self):
        from validation.data_coverage import ACCESS_MODES
        self.assertIn("representative_geometry_only", ACCESS_MODES)
        # the coarser modes remain valid, and the mode is DISTINCT from representative
        self.assertIn("representative", ACCESS_MODES)
        self.assertNotEqual("representative_geometry_only", "representative")

    def test_typed_coverage_assessment_preserves_geometry_only_mode(self):
        # The typed FE-038 evidence block carries the geometry-only mode verbatim; full end-to-end
        # validation of the assembled block is exercised by the executor tests (B/C/F) which run
        # validate_data_coverage_report -> validate_coverage_assessment on the real report.
        from validation.coverage_assessment import build_coverage_assessment
        a = build_coverage_assessment(
            teacher_training_data_access="representative_geometry_only",
            teacher_access_limitations=["geometry only; no DFT labels"], dimensions=[],
            acquisition_lineage={"equality_result": "PASS"},
            protected_reference_exclusion={"result": "PASS"})
        self.assertEqual(a["teacher_training_data_access"]["mode"],
                         "representative_geometry_only")
        self.assertEqual(a["teacher_training_data_access"]["limitations"],
                         ["geometry only; no DFT labels"])


class TestB_AuthoritativeFrozenValueEmitted(unittest.TestCase):
    """(B) The executor emits the frozen authoritative value unchanged."""

    def test_frozen_policy_value_flows_into_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            policy = _write_policy(root / "dataset_policy.yaml",
                                   access="representative_geometry_only")
            out = _run(root, ["bulk", "bulk"], {"structure_classes": ["bulk"]},
                       dataset_policy=str(policy))
            report = out["report"]
            self.assertEqual(report["teacher_training_data_access"],
                             "representative_geometry_only")
            prov = report["teacher_training_data_access_provenance"]
            self.assertEqual(prov["resolved_from"], "frozen_authoritative_input")
            self.assertEqual(prov["resolved_value"], "representative_geometry_only")
            self.assertEqual(len(prov["declarations"]), 1)
            self.assertEqual(prov["declarations"][0]["source_role"], "dataset_policy")
            self.assertTrue(prov["declarations"][0]["source_sha256"])


class TestC_NeverCollapsedToRepresentative(unittest.TestCase):
    """(C) A geometry-only frozen value is never collapsed to representative, even when the LLM
    proposal injects the coarser 'representative' (the exact pre-FE-041 defect)."""

    def test_proposal_representative_does_not_override_frozen_geometry_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            policy = _write_policy(root / "dataset_policy.yaml",
                                   access="representative_geometry_only")
            out = _run(root, ["bulk"], {"structure_classes": ["bulk"]},
                       dataset_policy=str(policy),
                       teacher_training_data_access="representative")  # LLM proposal (ignored)
            self.assertEqual(out["report"]["teacher_training_data_access"],
                             "representative_geometry_only")

    def test_absent_proposal_still_resolves_frozen_geometry_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            policy = _write_policy(root / "dataset_policy.yaml",
                                   access="representative_geometry_only")
            out = _run(root, ["bulk"], {"structure_classes": ["bulk"]},
                       dataset_policy=str(policy))
            self.assertEqual(out["report"]["teacher_training_data_access"],
                             "representative_geometry_only")


class TestD_ConflictingDeclarationsFailClosed(unittest.TestCase):
    """(D) Conflicting authoritative frozen declarations fail closed -- never silently pick one."""

    def test_conflicting_frozen_sources_raise(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            policy = _write_policy(root / "dataset_policy.yaml",
                                   access="representative_geometry_only")
            scope = _write_policy(root / "distillation_scope.yaml", access="representative")
            with self.assertRaises(ValueError) as ctx:
                _run(root, ["bulk"], {"structure_classes": ["bulk"]},
                     dataset_policy=str(policy), distillation_scope=str(scope))
            self.assertIn("TEACHER_ACCESS_CONFLICT", str(ctx.exception))

    def test_agreeing_frozen_sources_preserve_both_provenances(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            policy = _write_policy(root / "dataset_policy.yaml",
                                   access="representative_geometry_only")
            scope = _write_policy(root / "distillation_scope.yaml",
                                  access="representative_geometry_only")
            out = _run(root, ["bulk"], {"structure_classes": ["bulk"]},
                       dataset_policy=str(policy), distillation_scope=str(scope))
            prov = out["report"]["teacher_training_data_access_provenance"]
            self.assertEqual(prov["resolved_value"], "representative_geometry_only")
            roles = {d["source_role"] for d in prov["declarations"]}
            self.assertEqual(roles, {"dataset_policy", "distillation_scope"})


class TestE_StructuralGapsPreserved(unittest.TestCase):
    """(E) FE-039 semantics untouched: the ffv4q-derived coverage still reports the SAME four
    structurally unsupported deployment classes. Driven from the REAL frozen label_map + the
    immutable ffv4q acquired counts, so the finding is DERIVED, not hard-coded."""

    _FFV4Q_COUNTS = {"SiOx_crystal_amorphous_interfaces": 2, "bulk_cryst": 4, "cluster": 3,
                     "highpressure_int_AL": 3, "quench_int_AL": 1}
    _EXPECTED_UNSUPPORTED = {"liquid_or_melt_SiO2", "surface_SiO2", "oxygen_vacancy_SiO2",
                             "condensed_pure_Si_boundary"}

    def test_four_unsupported_classes_still_derived(self):
        if not _FROZEN_SCOPE.is_file():
            self.skipTest("frozen deployment scope artifact not present")
        from runtimes.pydantic_ai.executors import _resolve_frozen_structure_class_label_map
        from validation.coverage_gap_assessment import (build_structure_class_dimensions,
                                                        unsupported_structure_classes)
        doc = json.loads(_FROZEN_SCOPE.read_text())
        declared = list(doc["primary_domains"])
        label_map = _resolve_frozen_structure_class_label_map(
            {"scope_classification_evidence_path": str(_FROZEN_SCOPE)}, deployment_domain={})
        dims = build_structure_class_dimensions(declared, self._FFV4Q_COUNTS, label_map)
        unsupported = set(unsupported_structure_classes(dims))
        self.assertEqual(unsupported, self._EXPECTED_UNSUPPORTED)


class TestF_RecoveryDrivenByStructuralGaps(unittest.TestCase):
    """(F) With the access-label ambiguity removed, the report simultaneously (a) states the honest
    geometry-only access mode and (b) names the structural coverage gaps -- so a downstream
    REVISE/recovery is driven by the real unsupported regions, not the resolved access label."""

    def test_report_honest_label_plus_structural_gaps(self):
        if not _FROZEN_SCOPE.is_file():
            self.skipTest("frozen deployment scope artifact not present")
        doc = json.loads(_FROZEN_SCOPE.read_text())
        deployment_domain = {"structure_classes": list(doc["primary_domains"])}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            policy = _write_policy(root / "dataset_policy.yaml",
                                   access="representative_geometry_only")
            # a candidate that occupies only one declared class -> the others are unsupported
            occupied = doc["label_map"][0]["raw_label"]
            out = _run(root, [occupied, occupied], deployment_domain,
                       dataset_policy=str(policy),
                       scope_classification_evidence_path=str(_FROZEN_SCOPE))
            report = out["report"]
            # honest access label preserved (not the coarse 'representative')
            self.assertEqual(report["teacher_training_data_access"],
                             "representative_geometry_only")
            # the FE-039 structural gap gate still names unsupported classes as concrete gaps
            structural_gaps = [g for g in report["identified_gaps"]
                               if "structurally UNSUPPORTED" in g]
            self.assertTrue(structural_gaps)
            self.assertEqual(report["coverage_assessment"]["assessment_status"],
                             "COVERAGE_INSUFFICIENT")


class TestG_LegacyModesUnchanged(unittest.TestCase):
    """(G) Existing full/representative/unavailable behavior remains valid and unchanged."""

    def test_representative_default_when_nothing_declares(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # no dataset_policy path, no proposal param -> executor auto-generates a stub policy
            out = _run(root, ["bulk", "bulk"], {"structure_classes": ["bulk"]})
            report = out["report"]
            self.assertEqual(report["teacher_training_data_access"], "representative")
            self.assertEqual(report["teacher_training_data_access_provenance"]["resolved_from"],
                             "historical_default")

    def test_full_frozen_value_flows_through(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            policy = _write_policy(root / "dataset_policy.yaml", access="full")
            out = _run(root, ["bulk"], {"structure_classes": ["bulk"]},
                       dataset_policy=str(policy))
            self.assertEqual(out["report"]["teacher_training_data_access"], "full")

    def test_unavailable_proposal_param_still_not_assessable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # no frozen declaration -> proposal-parameter fallback preserved (pre-FE-041 behavior)
            out = _run(root, ["bulk", "bulk"],
                       {"structure_classes": ["bulk"],
                        "coverage_requirement": {"min_frames_by_config_type": {"bulk": 1}}},
                       teacher_training_data_access="unavailable")
            self.assertEqual(out["report"]["teacher_training_data_access"], "unavailable")
            self.assertEqual(out["report"]["coverage_status"], "NOT_ASSESSABLE")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
