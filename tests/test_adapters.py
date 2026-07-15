import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import read, write

from adapters import load_config
from adapters.contracts import PredictionBatch
from adapters.reference_dft import render_incar
from adapters.preflight import check_student_config
from adapters.student import _train_grace_fs, lammps_pair_style_block, load_student
from adapters.teacher import teacher_model_reference
from adapters.uncertainty import spearman
from validation.structure_dynamics import compute_msd, compute_nve_drift
from validation.surface_energy import surface_energy
from workflow.steps import split_dataset


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

    @patch("adapters.preflight.shutil.which", return_value=None)
    def test_schema_only_preflight_does_not_require_conda(self, _which):
        cfg = {"kind": "grace-fs", "train": {"config_template": "missing", "env": "remote-env"},
               "deploy": {"elements": ["Mo", "Nb", "Ta"]}}
        self.assertIn("training environment=remote-env", check_student_config(cfg, check_files=False))

    def test_surface_energy_uses_all_exposed_surfaces(self):
        self.assertAlmostEqual(surface_energy(12.0, 10.0, 5.0, 2), 0.2)

    def test_msd_unwraps_periodic_boundary_crossing(self):
        first = Atoms("H", positions=[[9.9, 0, 0]], cell=[10, 10, 10], pbc=True)
        second = Atoms("H", positions=[[0.1, 0, 0]], cell=[10, 10, 10], pbc=True)
        self.assertAlmostEqual(compute_msd([first, second])["H"][-1], 0.04, places=8)

    def test_nve_drift_uses_sampling_stride(self):
        energies = np.array([0.0, 0.1, 0.2])
        stride_drift, _ = compute_nve_drift(energies, 1.0, 1, sample_interval_steps=100)
        unit_drift, _ = compute_nve_drift(energies, 1.0, 1)
        self.assertAlmostEqual(stride_drift * 100, unit_drift)

    def test_spearman_uses_average_ranks_for_ties(self):
        self.assertAlmostEqual(spearman([1, 1, 2], [1, 2, 3]), 0.8660254037844387)

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

    def test_dataset_split_keeps_parent_groups_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "labeled.extxyz"
            frames = []
            for group in range(5):
                for child in range(2):
                    atoms = Atoms("H", positions=[[group + child * 0.01, 0, 0]])
                    atoms.info.update(structure_id=f"g{group}-c{child}", parent_structure_id=f"g{group}")
                    frames.append(atoms)
            write(source, frames)
            result = split_dataset(source, root / "splits", root / "split.json",
                                   seed=7, validation_fraction=0.2, test_fraction=0.2)
            group_sets = [set(item["group_ids"]) for item in result["splits"].values()]
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])
            self.assertEqual(sum(len(read(item["path"], index=":"))
                                 for item in result["splits"].values()), len(frames))

    def test_dataset_split_requires_explicit_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "untracked.extxyz"
            frames = [Atoms("H", positions=[[i, 0, 0]], info={"structure_id": f"s-{i}"})
                      for i in range(4)]
            write(source, frames)
            with self.assertRaisesRegex(ValueError, "missing required lineage key"):
                split_dataset(source, root / "splits", root / "split.json")

    def test_dft_and_teacher_paths_resolve_from_config_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "INCAR.template"
            template.write_text("ENCUT={ENCUT}\nKSPACING={KSPACING}\nISMEAR={ISMEAR}\nSIGMA={SIGMA}\nNSW={NSW}\nIBRION={IBRION}\n")
            checkpoint = root / "teacher.model"
            checkpoint.write_text("mock")
            cfg_path = root / "dft.yaml"
            cfg_path.write_text("kind: vasp\nincar_template: INCAR.template\nencut_ev: 400\nkspacing_inv_angstrom: 0.2\nsmearing: {ismear: 0, sigma: 0.05}\nrelaxation: {nsw: 1, ibrion: 2}\n")
            render_incar(load_config(cfg_path), root / "INCAR")
            teacher_cfg_path = root / "teacher.yaml"
            teacher_cfg_path.write_text("kind: mace\ncheckpoint: teacher.model\ncalculator: {module: x, class: Y}\n")
            self.assertEqual(Path(teacher_model_reference(load_config(teacher_cfg_path))), checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
