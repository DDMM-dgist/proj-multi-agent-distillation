"""Priority #3 requirement #5: extensible bounded-evidence adapter/registry.

Proves (a) the JSON-evidence dispatch in runtimes.pydantic_ai.bounded_evidence is a genuine
registry -- a newly registered adapter fires without editing `_json_summary` -- and (b) the
built-in structural-coverage-evidence adapter reduces a real
`coverage.report.build_directed_coverage_evidence` payload to exactly the whitelisted semantic
fields (direction; query/reference population identities; slice/domain memberships;
supported/unsupported counts+fractions; descriptive distance distributions; provenance hashes;
limitations) and never leaks representation/search-backend internals (e.g. SOAP descriptor
hyperparameters, cKDTree worker counts).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.io import write

from coverage.nn_distance import EnvironmentDistanceRecord
from coverage.reference_pool import ReferencePool
from coverage.report import build_directed_coverage_evidence
from runtimes.pydantic_ai import bounded_evidence
from runtimes.pydantic_ai.bounded_evidence import (
    _JSON_EVIDENCE_ADAPTERS,
    build_split_crosswalk,
    register_json_evidence_adapter,
    summarize_artifact,
)


def _split_sourced_atoms(category, local_index, deployment_slice=None, x=1.0):
    atoms = Atoms("Cu", positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.info["source_category"] = category
    atoms.info["source_local_index"] = local_index
    if deployment_slice is not None:
        atoms.info["deployment_slice_membership"] = deployment_slice
    return atoms


def _real_shaped_split_manifest(records) -> dict:
    """Matches the actual production split-manifest shape (see
    configs/provenance/teacher_training_split_manifest.json): a dict with a top-level ``records``
    list, each carrying ``source_category``/``source_local_index``/``split`` (plus ``global_index``,
    which the crosswalk join never needs)."""
    return {
        "schema_version": 1,
        "purpose": "teacher_training_split_reconstruction",
        "source_dataset": "test-fixture",
        "records": records,
    }


def _fake_reference_pool(**overrides) -> ReferencePool:
    fields = dict(
        population_role="teacher_train_partition",
        representation_hash="repr-hash-abc123",
        representation_provenance={
            "representation": "soap",
            "descriptor_config": {"r_cut": 5.0, "n_max": 8, "l_max": 6, "species": ["Si", "O"]},
        },
        search_backend_provenance={
            "backend": "exact_kdtree", "library": "scipy.spatial.cKDTree", "workers": 4,
        },
        reference_manifest_sha256="manifest-sha-def456",
        canonical_batch=None,
        global_indices_by_compatibility_key={},
        slices={},
        total_atoms=1000,
        total_frames=50,
    )
    fields.update(overrides)
    return ReferencePool(**fields)


def _fake_records() -> list:
    return [
        EnvironmentDistanceRecord(
            direction="teacher_support", query_population="candidate_population",
            reference_population="teacher_train_partition", query_structure_id="s1",
            query_environment_index=0, query_slice_labels=("bulk",),
            global_distance=0.1, global_matched=True,
            slice_distances={"slice_a": 0.2}, slice_matched={"slice_a": True},
        ),
        EnvironmentDistanceRecord(
            direction="teacher_support", query_population="candidate_population",
            reference_population="teacher_train_partition", query_structure_id="s2",
            query_environment_index=0, query_slice_labels=("surface",),
            global_distance=None, global_matched=False,
            slice_distances={"slice_a": None}, slice_matched={"slice_a": False},
        ),
    ]


class RegistryExtensibilityTests(unittest.TestCase):
    def test_new_adapter_fires_without_editing_dispatch(self):
        saved = list(_JSON_EVIDENCE_ADAPTERS)
        try:
            register_json_evidence_adapter(
                "custom_probe", lambda payload: payload.get("kind") == "custom_probe_v1",
                lambda payload: {"echo": payload.get("value")},
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "probe.json"
                path.write_text(json.dumps({"kind": "custom_probe_v1", "value": 42}))
                summary = summarize_artifact(path)
        finally:
            _JSON_EVIDENCE_ADAPTERS[:] = saved
        self.assertEqual(summary["custom_probe"], {"echo": 42})

    def test_unregistered_shape_gets_no_extra_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.json"
            path.write_text(json.dumps({"schema_version": 1, "status": "PASS"}))
            summary = summarize_artifact(path)
        self.assertNotIn("teacher_baseline", summary)
        self.assertNotIn("label_manifest", summary)
        self.assertNotIn("structural_coverage_evidence", summary)


class TeacherBaselineEvidenceSummaryTests(unittest.TestCase):
    """Scope C: the bounded evidence a Judge actually reads for teacher_baseline must expose the
    runtime species/type-mapping attestation (Scope B) alongside the pre-existing operational
    finiteness/provenance fields, and must make explicit that no DFT/protected-reference labels
    were used -- never silently omit the one field a Judge would need to check the gate criterion
    "Teacher calculator loads with the declared element/type mapping without runtime
    reconciliation or fallback"."""

    def _payload(self, *, fallback_applied, runtime_mapping):
        return {
            "schema_version": 1, "profile": "teacher_baseline",
            "teacher": {"kind": "mock", "config": "/x/teacher.yaml", "model_sha256": "abc"},
            "deployment_domain": {"structure_classes": ["bulk"],
                                  "dft_labels_used": False,
                                  "protected_reference_labels_used": False},
            "applicability": {"status": "CONDITIONAL", "limitations": ["scope-limited"]},
            "species_mapping": {
                "declared_chemical_symbols": ["O", "Si"],
                "declared_chemical_species_to_atom_type_map": None,
                "runtime_chemical_species_to_atom_type_map": runtime_mapping,
                "fallback_applied": fallback_applied,
                "fallback_reason": "chemical_species_to_atom_type_map required" if fallback_applied else None,
            },
            "checks": [{
                "domain": "operational_teacher_inference",
                "observable": "fresh_teacher_energy_force_finiteness", "status": "PASS",
                "value": 1.0, "unit": "eV/Angstrom",
                "criterion": {"operator": "max", "threshold": 1.0e12},
                "purpose": "deployment_stability", "reference_source": "teacher",
                "protocol": "fresh Teacher inference on declared operational structures",
                "details": {"n_frames": 3},
            }, {
                "domain": "operational_teacher_inference",
                "observable": "runtime_species_type_mapping_attested",
                "status": "PASS" if runtime_mapping else "FAIL",
                "value": 1 if runtime_mapping else 0, "unit": "boolean",
                "criterion": {"operator": "equals", "target": 1},
                "purpose": "deployment_stability", "reference_source": "teacher",
                "protocol": "deterministic capture of bound calculator kwargs",
                "details": {"fallback_applied": fallback_applied},
            }],
            "evidence": [],
        }

    def test_species_mapping_attestation_and_dft_isolation_are_exposed_to_the_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher_baseline.json"
            path.write_text(json.dumps(self._payload(
                fallback_applied=True, runtime_mapping={"O": "O", "Si": "Si"})))
            summary = summarize_artifact(path)
        tb = summary["teacher_baseline"]
        self.assertTrue(tb["species_mapping_attested"])
        self.assertEqual(tb["species_mapping_check_status"], "PASS")
        self.assertTrue(tb["species_mapping_fallback_applied"])
        self.assertFalse(tb["dft_labels_used"])
        self.assertFalse(tb["protected_reference_labels_used"])

    def test_unattested_species_mapping_is_visible_as_not_attested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher_baseline.json"
            path.write_text(json.dumps(self._payload(
                fallback_applied=True, runtime_mapping=None)))
            summary = summarize_artifact(path)
        tb = summary["teacher_baseline"]
        self.assertFalse(tb["species_mapping_attested"])
        self.assertEqual(tb["species_mapping_check_status"], "FAIL")


class StructuralCoverageEvidenceAdapterTests(unittest.TestCase):
    def _build_payload(self) -> dict:
        pool = _fake_reference_pool()
        records = _fake_records()
        return build_directed_coverage_evidence(
            direction="teacher_support", query_population="candidate_population",
            reference_pool=pool, records=records, excluded_partitions=("validation", "test"),
        )

    def test_adapter_recognizes_real_coverage_report_shape(self):
        payload = self._build_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage_evidence.json"
            path.write_text(json.dumps(payload))
            summary = summarize_artifact(path)
        self.assertIn("structural_coverage_evidence", summary)

    def test_semantic_summary_exposes_only_whitelisted_fields(self):
        payload = self._build_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage_evidence.json"
            path.write_text(json.dumps(payload))
            summary = summarize_artifact(path)
        sc = summary["structural_coverage_evidence"]

        self.assertEqual(sc["direction"], "teacher_support")
        self.assertEqual(sc["query_population"], "candidate_population")
        self.assertEqual(sc["reference_population"], "teacher_train_partition")
        self.assertEqual(sc["n_query_environments"], 2)
        self.assertEqual(sc["n_query_structures"], 2)

        self.assertEqual(sc["query_slice_memberships"], ["bulk", "surface"])
        self.assertEqual(sc["reference_slice_memberships"], ["slice_a"])

        overall = sc["overall_distance_distribution"]
        self.assertEqual(overall["supported_count"], 1)
        self.assertEqual(overall["unsupported_count"], 1)
        self.assertAlmostEqual(overall["supported_fraction"], 0.5)
        self.assertAlmostEqual(overall["unsupported_fraction"], 0.5)
        self.assertIn("mean", overall["distance_distribution"])
        self.assertIn("p95", overall["distance_distribution"])

        self.assertEqual(sc["provenance_hashes"],
                         {"representation_hash": "repr-hash-abc123",
                          "reference_manifest_sha256": "manifest-sha-def456"})
        self.assertEqual(sorted(sc["limitations"]), ["test", "validation"])

    def test_representation_and_search_backend_internals_never_leak(self):
        payload = self._build_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage_evidence.json"
            path.write_text(json.dumps(payload))
            summary = summarize_artifact(path)
        sc = summary["structural_coverage_evidence"]
        blob = json.dumps(sc)
        # SOAP descriptor hyperparameters and cKDTree backend internals must not appear anywhere
        # in the Analyst-facing summary -- only their hash-level fingerprints may.
        for leaked in ("descriptor_config", "r_cut", "l_max", "cKDTree", "scipy", "workers",
                       "representation_provenance", "search_backend_provenance"):
            self.assertNotIn(leaked, blob)


class SourceSplitLineageRegressionTests(unittest.TestCase):
    """R20 forensic-audit checklist items 1-4: the fix that replaces the incorrect
    `parent_structure_id`-based descendant-lineage interpretation (which falsely reported
    "missing_lineage_frames=2133" out of 2,134 frozen teacher_baseline frames, since those frames
    were never given a `parent_structure_id` in the first place) with a deterministic join of each
    frame's own `source_category`/`source_local_index` against the authoritative Teacher split
    manifest."""

    def test_split_sourced_frames_no_longer_report_stale_2133_style_missing_lineage(self):
        # 5 frames, all carrying real source-split provenance and all resolvable against the
        # manifest below -- the pre-fix code path would have reported 4 (n_frames - 1) of these as
        # "missing lineage" purely because they lack `parent_structure_id`.
        frames = [_split_sourced_atoms("bulk_cryst", i, "bulk", x=float(i)) for i in range(5)]
        manifest = _real_shaped_split_manifest([
            {"global_index": i, "source_category": "bulk_cryst", "source_local_index": i, "split": "train"}
            for i in range(5)
        ])
        with tempfile.TemporaryDirectory() as tmp:
            frames_path = Path(tmp) / "frames.extxyz"
            manifest_path = Path(tmp) / "split_manifest.json"
            write(str(frames_path), frames)
            manifest_path.write_text(json.dumps(manifest))
            crosswalk = build_split_crosswalk([manifest_path])
            summary = summarize_artifact(frames_path, split_crosswalk=crosswalk)
        self.assertEqual(summary["n_frames"], 5)
        self.assertEqual(summary["source_split_joined_frames"], 5)
        self.assertEqual(summary["missing_lineage_frames"], 0)
        self.assertNotEqual(summary["missing_lineage_frames"], summary["n_frames"] - 1)

    def test_category_and_domain_counts_reflect_real_source_fields_not_unknown_placeholder(self):
        frames = [
            _split_sourced_atoms("bulk_cryst", 0, "bulk", x=0.0),
            _split_sourced_atoms("bulk_cryst", 1, "bulk", x=1.0),
            _split_sourced_atoms("surface_slab", 2, "surface", x=2.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.extxyz"
            write(str(path), frames)
            summary = summarize_artifact(path)
        self.assertEqual(summary["category_counts"], {"bulk_cryst": 2, "surface_slab": 1})
        self.assertEqual(summary["domain_counts"], {"bulk": 2, "surface": 1})
        self.assertNotIn("unknown", summary["category_counts"])
        self.assertNotIn("unknown", summary["domain_counts"])

    def test_crosswalk_joins_deterministically_to_authoritative_split_manifest(self):
        manifest = _real_shaped_split_manifest([
            {"global_index": 0, "source_category": "bulk_cryst", "source_local_index": 0, "split": "train"},
            {"global_index": 1, "source_category": "bulk_cryst", "source_local_index": 1, "split": "val"},
            {"global_index": 2, "source_category": "surface_slab", "source_local_index": 0, "split": "test"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "split_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            crosswalk = build_split_crosswalk([manifest_path])
        self.assertEqual(crosswalk["resolved"], {
            ("bulk_cryst", 0): "train", ("bulk_cryst", 1): "val", ("surface_slab", 0): "test",
        })
        self.assertEqual(crosswalk["ambiguous"], set())
        self.assertEqual(len(crosswalk["sources"]), 1)
        self.assertEqual(crosswalk["sources"][0]["path"], str(manifest_path.resolve()))
        self.assertTrue(crosswalk["sources"][0]["sha256"])

    def test_conflicting_split_manifest_entries_fail_closed_to_ambiguous(self):
        # Same (source_category, source_local_index) key resolves to two different splits across
        # manifests -- the crosswalk must refuse to guess and must never silently keep either value.
        manifest_a = _real_shaped_split_manifest([
            {"global_index": 0, "source_category": "bulk_cryst", "source_local_index": 0, "split": "train"},
        ])
        manifest_b = _real_shaped_split_manifest([
            {"global_index": 0, "source_category": "bulk_cryst", "source_local_index": 0, "split": "test"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "split_manifest_a.json"
            path_b = Path(tmp) / "split_manifest_b.json"
            path_a.write_text(json.dumps(manifest_a))
            path_b.write_text(json.dumps(manifest_b))
            crosswalk = build_split_crosswalk([path_a, path_b])
        self.assertNotIn(("bulk_cryst", 0), crosswalk["resolved"])
        self.assertIn(("bulk_cryst", 0), crosswalk["ambiguous"])

    def test_frame_summary_reports_ambiguous_lineage_frames_as_ambiguous_not_joined(self):
        frames = [_split_sourced_atoms("bulk_cryst", 0, "bulk", x=0.0)]
        manifest_a = _real_shaped_split_manifest([
            {"global_index": 0, "source_category": "bulk_cryst", "source_local_index": 0, "split": "train"},
        ])
        manifest_b = _real_shaped_split_manifest([
            {"global_index": 0, "source_category": "bulk_cryst", "source_local_index": 0, "split": "test"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            frames_path = Path(tmp) / "frames.extxyz"
            path_a = Path(tmp) / "split_manifest_a.json"
            path_b = Path(tmp) / "split_manifest_b.json"
            write(str(frames_path), frames)
            path_a.write_text(json.dumps(manifest_a))
            path_b.write_text(json.dumps(manifest_b))
            crosswalk = build_split_crosswalk([path_a, path_b])
            summary = summarize_artifact(frames_path, split_crosswalk=crosswalk)
        self.assertEqual(summary["source_split_ambiguous_frames"], 1)
        self.assertEqual(summary["source_split_joined_frames"], 0)
        self.assertEqual(summary["missing_lineage_frames"], 1)


if __name__ == "__main__":
    unittest.main()
