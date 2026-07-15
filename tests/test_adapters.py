import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adapters.contracts import PredictionBatch
from adapters.preflight import check_student_config
from adapters.student import _train_grace_fs, lammps_pair_style_block, load_student
from validation.surface_energy import surface_energy


class AdapterContractTests(unittest.TestCase):
    def test_model_artifact_and_checkpoint_path_are_unambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "potential_saved_bestmodel"
            checkpoint.write_text("mock")
            cfg = {"kind": "simple-nn"}
            artifact = load_student(cfg, checkpoint)
            self.assertEqual(artifact.path, checkpoint)
            self.assertEqual(load_student(cfg, artifact), artifact)

    def test_lammps_element_order_comes_from_config(self):
        cfg = {
            "kind": "simple-nn",
            "deploy": {"lammps_pair_style": "nn", "elements": ["Al", "O", "N"]},
        }
        block = lammps_pair_style_block(cfg, "/models/student")
        self.assertIn("/models/student Al O N", block)
        self.assertNotIn("Si O", block)

    def test_missing_element_order_fails(self):
        with self.assertRaisesRegex(ValueError, "deploy.elements"):
            lammps_pair_style_block({"kind": "simple-nn", "deploy": {}}, "model")

    def test_prediction_batch_shape_contract(self):
        with self.assertRaises(ValueError):
            PredictionBatch(energies=np.zeros(2), forces=[np.zeros((1, 3))])

    def test_preflight_accepts_generic_elements(self):
        cfg = {
            "kind": "simple-nn",
            "train": {
                "config_template": "unused",
                "descriptor_params": {"Al": "unused", "O": "unused"},
            },
            "deploy": {"elements": ["Al", "O"]},
        }
        checks = check_student_config(cfg, check_files=False)
        self.assertIn("element order=Al,O", checks)

    def test_surface_energy_uses_all_exposed_surfaces(self):
        self.assertAlmostEqual(surface_energy(12.0, 10.0, 5.0, 2), 0.2)

    def test_grace_fs_runner_preserves_seed_in_train_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.yaml"
            dataset = root / "labels.pkl.gz"
            template.write_text('data: "{{DATASET_PATH}}"')
            dataset.write_text("mock")
            out = root / "run"
            out.mkdir()
            artifact = out / "seed" / "3" / "FS_model.yaml"

            def fake_run(command, **kwargs):
                if "-sf" in command:
                    artifact.parent.mkdir(parents=True)
                    artifact.write_text("mock model")

            cfg = {"train": {"config_template": str(template), "binary": "gracemaker"}}
            with patch("adapters.student.subprocess.run", side_effect=fake_run) as run:
                self.assertEqual(_train_grace_fs(cfg, dataset, out, 3), artifact)
            self.assertEqual(run.call_args_list[0].args[0][:3], ["gracemaker", "--seed", "3"])
            self.assertIn("-sf", run.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
