import unittest

from ase import Atoms

from coverage.descriptor_config import SoapDescriptorConfig
from coverage.soap_distance_policy import SoapDistancePolicy


def _isolated_atom():
    # A single H atom alone in a large periodic box: with r_cut small relative to
    # the box, its local environment is empty (no neighbors within cutoff, even
    # through periodic images) -- this gives a fixed, position-independent SOAP
    # vector for every "isolated" frame, regardless of where the atom sits.
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)


def _paired_atoms():
    # Two H atoms 1.0 Angstrom apart: each atom has exactly one neighbor within
    # r_cut, at a fixed distance -- this gives a fixed SOAP vector distinct from
    # the isolated case, for every "paired" frame.
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)


def _non_periodic_atom():
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=False)


class CoveragePipelineTests(unittest.TestCase):
    """Exercises the generic CoverageRepresentation / DistancePolicy /
    SearchBackend / ReferencePool / nn_distance / aggregate / report pipeline
    via its current implementations (SOAP, SoapDistancePolicy,
    ExactKDTreeBackend). Nothing here reads `config_type` or any other
    campaign-specific metadata field -- slice membership is supplied
    externally, exactly as a real campaign adapter (see coverage.adapters and
    tests/test_sio2_config_type_adapter.py) would.

    The three interfaces (representation construction, comparison semantics,
    search mechanics) are deliberately exercised as separately-injectable
    collaborators throughout, never as one bundled object, to keep the
    Representation/DistancePolicy/SearchBackend separation from silently
    regressing back into one coupled class.
    """

    def setUp(self):
        try:
            import dscribe  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("dscribe (optional structural-coverage dep) not installed")
        self.descriptor_config = SoapDescriptorConfig(
            r_cut=2.0, n_max=2, l_max=1, sigma=0.3, species=("H",), rbf="gto", periodic=True,
        )
        self.distance_policy = SoapDistancePolicy(
            normalization="none", metric="euclidean",
            central_species_matching=True, periodic_consistency_required=False,
        )

    def _representation(self, descriptor_config=None, distance_policy=None):
        from coverage.soap_representation import SoapCoverageRepresentation

        return SoapCoverageRepresentation(
            descriptor_config or self.descriptor_config, distance_policy or self.distance_policy,
        )

    def _backend(self):
        from coverage.exact_kdtree_backend import ExactKDTreeBackend

        return ExactKDTreeBackend()

    def test_compute_reflects_local_environment_not_position(self):
        rep = self._representation()
        batch = rep.compute([_isolated_atom(), _isolated_atom()], structure_ids=["f0", "f1"])
        self.assertEqual(len(batch), 2)
        # Same empty local environment in both frames -> identical descriptor vector.
        self.assertTrue((batch.vectors[0] == batch.vectors[1]).all())

    def test_compute_rejects_unlisted_species(self):
        rep = self._representation()
        oxygen_frame = Atoms("O", positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)
        with self.assertRaises(ValueError):
            rep.compute([oxygen_frame])

    def test_representation_compute_has_no_cross_call_state(self):
        # A representation instance must be safely reusable across independent
        # batches with different structures under the same structure_id -- no
        # build_index/query_nearest coupling and no periodicity bookkeeping
        # left over from a prior compute() call.
        distance_policy = SoapDistancePolicy(
            normalization="none", metric="euclidean",
            central_species_matching=False, periodic_consistency_required=True,
        )
        rep = self._representation(distance_policy=distance_policy)
        batch1 = rep.compute([_isolated_atom()], structure_ids=["f0"])
        batch2 = rep.compute([_non_periodic_atom()], structure_ids=["f0"])
        self.assertNotEqual(batch1.compatibility_key, batch2.compatibility_key)

    def test_representation_batch_environment_ids_are_canonical_identity(self):
        rep = self._representation()
        batch = rep.compute([_isolated_atom(), _paired_atoms()], structure_ids=["f0", "f1"])
        self.assertEqual(batch.environment_ids(), (("f0", 0), ("f1", 0), ("f1", 1)))

    def test_distance_policy_normalize_l2(self):
        import numpy as np

        policy = SoapDistancePolicy(
            normalization="l2", metric="cosine",
            central_species_matching=True, periodic_consistency_required=False,
        )
        vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
        normalized = policy.normalize(vectors)
        self.assertAlmostEqual(float(np.linalg.norm(normalized[0])), 1.0, places=10)
        self.assertAlmostEqual(float(np.linalg.norm(normalized[1])), 1.0, places=10)

    def test_distance_policy_provenance_reflects_fields(self):
        provenance = self.distance_policy.provenance()
        self.assertEqual(provenance["normalization"], "none")
        self.assertEqual(provenance["metric"], "euclidean")

    def test_exact_kdtree_backend_is_exact_and_has_provenance(self):
        backend = self._backend()
        self.assertTrue(backend.is_exact())
        provenance = backend.provenance()
        self.assertEqual(provenance["backend"], "exact_kdtree")
        self.assertTrue(provenance["is_exact"])

    def test_exact_kdtree_backend_rejects_empty_index(self):
        import numpy as np

        backend = self._backend()
        with self.assertRaises(ValueError):
            backend.build_index(np.zeros((0, 4)), [])

    def test_reference_pool_builds_global_and_per_slice_indices_without_vector_duplication(self):
        from coverage.reference_pool import build_reference_pool

        rep = self._representation()
        backend = self._backend()
        pool = build_reference_pool(
            population_role="teacher_train_partition",
            representation=rep,
            distance_policy=self.distance_policy,
            search_backend=backend,
            structures=[_isolated_atom(), _paired_atoms()],
            reference_manifest_sha256="deadbeef" * 8,
            structure_ids=["f0", "f1"],
            slice_membership=[("isolated",), ("paired",)],
        )
        self.assertEqual(pool.slice_names(), ("isolated", "paired"))
        self.assertEqual(pool.total_atoms, 3)  # 1 isolated atom + 2 paired atoms
        self.assertEqual(pool.total_frames, 2)
        # Canonical storage: the pool holds exactly one vector array (on
        # canonical_batch); a SlicePool never carries its own vectors field --
        # only index positions into the canonical batch.
        self.assertEqual(pool.canonical_batch.vectors.shape[0], 3)
        slice_pool_fields = pool.slices["isolated"].__dataclass_fields__.keys()
        self.assertNotIn("vectors", slice_pool_fields)
        self.assertIn("environment_positions", slice_pool_fields)
        self.assertEqual(pool.slices["isolated"].environment_positions, (0,))
        self.assertEqual(pool.slices["paired"].environment_positions, (1, 2))

    def test_reference_pool_rejects_empty_structures(self):
        from coverage.reference_pool import build_reference_pool

        rep = self._representation()
        with self.assertRaises(ValueError):
            build_reference_pool(
                population_role="teacher_train_partition", representation=rep,
                distance_policy=self.distance_policy, search_backend=self._backend(),
                structures=[], reference_manifest_sha256="hash",
            )

    def test_reference_pool_rejects_blank_population_role(self):
        from coverage.reference_pool import build_reference_pool

        rep = self._representation()
        with self.assertRaises(ValueError):
            build_reference_pool(
                population_role="   ", representation=rep,
                distance_policy=self.distance_policy, search_backend=self._backend(),
                structures=[_isolated_atom()], reference_manifest_sha256="hash",
            )

    def test_nn_distance_matches_own_slice_exactly(self):
        from coverage.nn_distance import compute_environment_distances
        from coverage.reference_pool import build_reference_pool

        rep = self._representation()
        backend = self._backend()
        pool = build_reference_pool(
            population_role="teacher_train_partition", representation=rep,
            distance_policy=self.distance_policy, search_backend=backend,
            structures=[_isolated_atom(), _paired_atoms()],
            reference_manifest_sha256="deadbeef" * 8,
            structure_ids=["f0", "f1"], slice_membership=[("isolated",), ("paired",)],
        )

        query_batch = rep.compute(
            [_isolated_atom(), _paired_atoms()], structure_ids=["q_isolated", "q_paired"]
        )
        records = compute_environment_distances(
            direction="teacher_support", query_population="candidate_population",
            reference_pool=pool, distance_policy=self.distance_policy, search_backend=backend,
            query_batch=query_batch,
            query_slice_labels={"q_isolated": ("isolated",), "q_paired": ("paired",)},
        )

        isolated_record = next(r for r in records if r.query_structure_id == "q_isolated")
        paired_records = [r for r in records if r.query_structure_id == "q_paired"]

        # An isolated candidate atom matches the "isolated" reference slice exactly.
        self.assertAlmostEqual(isolated_record.slice_distances["isolated"], 0.0, places=10)
        # ...but is measurably different from the "paired" reference slice.
        self.assertGreater(isolated_record.slice_distances["paired"], 1e-6)
        # Global distance (from the mandatory global indices) is the exact-match distance.
        self.assertAlmostEqual(isolated_record.global_distance, 0.0, places=10)

        # Both atoms in the paired candidate frame match the "paired" reference slice exactly.
        self.assertEqual(len(paired_records), 2)
        for record in paired_records:
            self.assertAlmostEqual(record.slice_distances["paired"], 0.0, places=10)
            self.assertGreater(record.slice_distances["isolated"], 1e-6)
            self.assertAlmostEqual(record.global_distance, 0.0, places=10)

    def test_nn_distance_surfaces_unmatched_central_species_explicitly(self):
        from coverage.nn_distance import compute_environment_distances
        from coverage.reference_pool import build_reference_pool

        species_config = SoapDescriptorConfig(
            r_cut=2.0, n_max=2, l_max=1, sigma=0.3, species=("H", "F"), rbf="gto", periodic=True,
        )
        rep = self._representation(descriptor_config=species_config)
        backend = self._backend()
        pool = build_reference_pool(
            population_role="teacher_train_partition", representation=rep,
            distance_policy=self.distance_policy, search_backend=backend,
            structures=[_isolated_atom()], reference_manifest_sha256="hash", structure_ids=["f0"],
        )
        fluorine_frame = Atoms("F", positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)
        query_batch = rep.compute([fluorine_frame], structure_ids=["q0"])
        records = compute_environment_distances(
            direction="teacher_support", query_population="candidate_population",
            reference_pool=pool, distance_policy=self.distance_policy, search_backend=backend,
            query_batch=query_batch,
        )
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].global_matched)
        self.assertIsNone(records[0].global_distance)

    def test_nn_distance_surfaces_unmatched_periodicity_symmetrically_not_as_a_crash(self):
        # Prior architecture treated a periodicity mismatch as a hard
        # ValueError while a central-species mismatch was ordinary
        # matched=False evidence -- this refactor unifies both under the same
        # opaque compatibility_key mechanism, so periodicity mismatches are
        # now quantitative "unmatched" evidence too, never a crash.
        from coverage.nn_distance import compute_environment_distances
        from coverage.reference_pool import build_reference_pool

        distance_policy = SoapDistancePolicy(
            normalization="none", metric="euclidean",
            central_species_matching=False, periodic_consistency_required=True,
        )
        rep = self._representation(distance_policy=distance_policy)
        backend = self._backend()
        pool = build_reference_pool(
            population_role="teacher_train_partition", representation=rep,
            distance_policy=distance_policy, search_backend=backend,
            structures=[_isolated_atom()], reference_manifest_sha256="hash", structure_ids=["f0"],
        )
        query_batch = rep.compute([_non_periodic_atom()], structure_ids=["q0"])
        records = compute_environment_distances(
            direction="teacher_support", query_population="candidate_population",
            reference_pool=pool, distance_policy=distance_policy, search_backend=backend,
            query_batch=query_batch,
        )
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].global_matched)
        self.assertIsNone(records[0].global_distance)

    def test_aggregate_summarize_matches_known_values(self):
        from coverage.aggregate import summarize

        stats = summarize([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["p50"], 2.0)
        self.assertAlmostEqual(stats["max"], 4.0)

    def test_aggregate_summarize_rejects_empty_input(self):
        from coverage.aggregate import summarize

        with self.assertRaises(ValueError):
            summarize([])

    def _teacher_support_records(self):
        from coverage.nn_distance import compute_environment_distances
        from coverage.reference_pool import build_reference_pool

        rep = self._representation()
        backend = self._backend()
        pool = build_reference_pool(
            population_role="teacher_train_partition", representation=rep,
            distance_policy=self.distance_policy, search_backend=backend,
            structures=[_isolated_atom(), _paired_atoms()],
            reference_manifest_sha256="deadbeef" * 8,
            structure_ids=["f0", "f1"], slice_membership=[("isolated",), ("paired",)],
        )
        query_batch = rep.compute([_isolated_atom()], structure_ids=["candidate"])
        records = compute_environment_distances(
            direction="teacher_support", query_population="candidate_population",
            reference_pool=pool, distance_policy=self.distance_policy, search_backend=backend,
            query_batch=query_batch,
        )
        return pool, records

    def test_aggregate_overall_global_summary_reports_unmatched_fraction(self):
        from coverage.aggregate import overall_global_summary
        from coverage.nn_distance import compute_environment_distances
        from coverage.reference_pool import build_reference_pool

        species_config = SoapDescriptorConfig(
            r_cut=2.0, n_max=2, l_max=1, sigma=0.3, species=("H", "F"), rbf="gto", periodic=True,
        )
        rep = self._representation(descriptor_config=species_config)
        backend = self._backend()
        pool = build_reference_pool(
            population_role="teacher_train_partition", representation=rep,
            distance_policy=self.distance_policy, search_backend=backend,
            structures=[_isolated_atom()], reference_manifest_sha256="hash", structure_ids=["f0"],
        )
        fluorine_frame = Atoms("F", positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)
        query_batch = rep.compute(
            [_isolated_atom(), fluorine_frame], structure_ids=["q_h", "q_f"]
        )
        records = compute_environment_distances(
            direction="teacher_support", query_population="candidate_population",
            reference_pool=pool, distance_policy=self.distance_policy, search_backend=backend,
            query_batch=query_batch,
        )
        summary = overall_global_summary(records)
        self.assertEqual(summary["n_unmatched"], 1)
        self.assertAlmostEqual(summary["unmatched_fraction"], 0.5)

    def test_report_builds_directed_coverage_evidence_with_exclusions_recorded(self):
        from coverage.report import build_directed_coverage_evidence

        pool, records = self._teacher_support_records()
        evidence = build_directed_coverage_evidence(
            "teacher_support", "candidate_population", pool, records,
            excluded_partitions=("validation", "test"),
        )
        self.assertEqual(evidence["reference_population"], "teacher_train_partition")
        self.assertEqual(evidence["excluded_partitions"], ["validation", "test"])
        self.assertEqual(evidence["n_query_environments"], 1)
        self.assertIn("mean", evidence["overall_global_summary"])
        self.assertEqual(
            evidence["provenance"]["representation_hash"], pool.representation_hash
        )
        self.assertEqual(evidence["provenance"]["reference_manifest_sha256"], "deadbeef" * 8)
        self.assertEqual(
            evidence["provenance"]["search_backend_provenance"]["backend"], "exact_kdtree"
        )

    def test_report_rejects_missing_exclusions_for_teacher_train_partition_role(self):
        from coverage.report import build_directed_coverage_evidence

        pool, records = self._teacher_support_records()
        with self.assertRaises(ValueError):
            build_directed_coverage_evidence(
                "teacher_support", "candidate_population", pool, records,
            )

    def test_report_rejects_records_from_a_different_direction(self):
        from coverage.report import build_directed_coverage_evidence

        pool, records = self._teacher_support_records()
        with self.assertRaises(ValueError):
            build_directed_coverage_evidence(
                "deployment_coverage", "candidate_population", pool, records,
                excluded_partitions=("validation", "test"),
            )

    def test_report_allows_no_exclusions_for_non_teacher_train_partition_role(self):
        from coverage.nn_distance import compute_environment_distances
        from coverage.reference_pool import build_reference_pool
        from coverage.report import build_directed_coverage_evidence

        rep = self._representation()
        backend = self._backend()
        pool = build_reference_pool(
            population_role="student_training_dataset", representation=rep,
            distance_policy=self.distance_policy, search_backend=backend,
            structures=[_isolated_atom()], reference_manifest_sha256="hash", structure_ids=["f0"],
        )
        query_batch = rep.compute([_isolated_atom()], structure_ids=["deployment0"])
        records = compute_environment_distances(
            direction="deployment_coverage", query_population="deployment_target_population",
            reference_pool=pool, distance_policy=self.distance_policy, search_backend=backend,
            query_batch=query_batch,
        )
        evidence = build_directed_coverage_evidence(
            "deployment_coverage", "deployment_target_population", pool, records,
        )
        self.assertEqual(evidence["reference_population"], "student_training_dataset")
        self.assertEqual(evidence["excluded_partitions"], [])

    def test_protected_reference_pointer_cross_checks_real_constants(self):
        from coverage.report import protected_reference_pointer
        from validation.reference_validation import REQUIRED_LOGICAL_FRAMES, REQUIRED_PROTECTED_SOURCE_ROWS

        pointer = protected_reference_pointer("configs/runs/example/protected_reference_report.json")
        self.assertEqual(pointer["required_logical_frames"], REQUIRED_LOGICAL_FRAMES)
        self.assertEqual(pointer["required_protected_source_rows"], REQUIRED_PROTECTED_SOURCE_ROWS)


if __name__ == "__main__":
    unittest.main()
