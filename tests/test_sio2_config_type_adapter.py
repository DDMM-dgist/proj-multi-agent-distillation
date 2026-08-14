import unittest

from ase import Atoms

from coverage.adapters.sio2_config_type import (
    config_type_of,
    config_type_slice_labels_by_id,
    config_type_slice_membership,
)


class Sio2ConfigTypeAdapterTests(unittest.TestCase):
    """This is the ONLY test file in this repository allowed to read/write
    `atoms.info["config_type"]` -- generic coverage.* tests (see
    tests/test_coverage_pipeline.py) never depend on this field or this
    adapter, confirming config_type is fully isolated to the SiO2-x campaign
    adapter rather than baked into the generic coverage engine.
    """

    def test_config_type_of_reads_explicit_field(self):
        atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        atoms.info["config_type"] = "bulk_amorphous"
        self.assertEqual(config_type_of(atoms), "bulk_amorphous")

    def test_config_type_of_falls_back_to_source_then_unknown(self):
        with_source = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        with_source.info["source"] = "dft_anchor"
        self.assertEqual(config_type_of(with_source), "dft_anchor")

        bare = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        self.assertEqual(config_type_of(bare), "unknown")

    def test_config_type_slice_membership_is_single_element_per_structure(self):
        a1 = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        a1.info["config_type"] = "bulk_amorphous"
        a2 = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        a2.info["config_type"] = "defect"

        membership = config_type_slice_membership([a1, a2])
        self.assertEqual(membership, [("bulk_amorphous",), ("defect",)])

    def test_config_type_slice_labels_by_id_keys_on_structure_id(self):
        a1 = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        a1.info["config_type"] = "bulk_amorphous"
        a2 = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        a2.info["config_type"] = "defect"

        labels = config_type_slice_labels_by_id([a1, a2], ["f0", "f1"])
        self.assertEqual(labels, {"f0": ("bulk_amorphous",), "f1": ("defect",)})

    def test_config_type_slice_labels_by_id_rejects_length_mismatch(self):
        a1 = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            config_type_slice_labels_by_id([a1], ["f0", "f1"])


if __name__ == "__main__":
    unittest.main()
