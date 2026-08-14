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

from coverage.nn_distance import EnvironmentDistanceRecord
from coverage.reference_pool import ReferencePool
from coverage.report import build_directed_coverage_evidence
from runtimes.pydantic_ai import bounded_evidence
from runtimes.pydantic_ai.bounded_evidence import (
    _JSON_EVIDENCE_ADAPTERS,
    register_json_evidence_adapter,
    summarize_artifact,
)


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


if __name__ == "__main__":
    unittest.main()
