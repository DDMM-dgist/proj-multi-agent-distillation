import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import read, write

from adapters import load_config
from adapters.acquisition import langevin_friction
from adapters.contracts import PredictionBatch
from adapters.reference_dft import render_incar
from adapters.preflight import (check_acquisition_config, check_student_config,
                                check_teacher_config, check_validation_profile)
from adapters.student import (_render_simple_nn_config, _train_grace_fs,
                              lammps_pair_style_block, load_student)
from adapters.teacher import teacher_model_reference
from adapters.uncertainty import committee_force_std, spearman
from validation.structure_dynamics import compute_msd, compute_nve_drift, read_energy_log
from validation.surface_energy import surface_energy, validate_surface_manifest
from workflow.contracts import validate_validation_manifest
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

    def test_pilot_preflight_rejects_placeholders_and_missing_mace_head(self):
        student = {"kind": "grace-fs", "train": {"config_template": "unused"},
                   "deploy": {"elements": ["A", "B", "C"]}}
        with self.assertRaisesRegex(ValueError, "placeholder"):
            check_student_config(student, check_files=False, require_ready=True)
        teacher = {"kind": "mace-mh1", "model": "model", "calculator": {
            "factory": "mace.calculators.mace_mp", "kwargs": {}}}
        with self.assertRaisesRegex(ValueError, "head"):
            check_teacher_config(teacher, check_files=False)

    def test_preflight_requires_teacher_md_friction_units_and_resolved_thresholds(self):
        acquisition = {"kind": "teacher-md", "temperature_K": 1000, "timestep_fs": 1,
                       "n_steps": 10, "snapshot_interval": 1, "friction": 0.01}
        with self.assertRaisesRegex(ValueError, "friction"):
            check_acquisition_config(acquisition)
        profile = {"kind": "x", "checks": ["surface_excess_energy"],
                   "surface_energetics": {"thresholds": {"student_dft": None}}}
        with self.assertRaisesRegex(ValueError, "unresolved"):
            check_validation_profile(profile, require_ready=True)

    def test_surface_energy_uses_all_exposed_surfaces(self):
        self.assertAlmostEqual(surface_energy(12.0, 10.0, 5.0, 2), 0.2)

    def test_surface_manifest_requires_comparable_protocols_and_recomputes_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surface.json"
            base = {
                "orientation": "(1 1 0)", "termination": "t1", "area_a2": 10.0,
                "n_surfaces": 2, "geometry_protocol": "g1",
                "reference_convention": "bulk-v1", "composition_balance_confirmed": True,
                "nonstoichiometric": False, "slab_energy_ev": 12.0,
                "bulk_reference_ev": 10.0, "surface_energy_J_m2": 1.602176634,
            }
            entries = [dict(base, method=method) for method in ("teacher", "student", "dft")]
            path.write_text(json.dumps({"quantity": "static_surface_excess_energy",
                                        "unit": "J/m2", "entries": entries}))
            validate_surface_manifest(path, ["teacher", "student", "dft"])
            validate_validation_manifest(
                path, "validation.surface_energy.validate_surface_manifest",
                {"required_methods": ["teacher", "student", "dft"]})
            entries[1]["orientation"] = "(1 0 0)"
            path.write_text(json.dumps({"quantity": "static_surface_excess_energy",
                                        "unit": "J/m2", "entries": entries}))
            with self.assertRaisesRegex(ValueError, "same geometry"):
                validate_surface_manifest(path, ["teacher", "student", "dft"])

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

    def test_committee_sigma_matches_cartesian_rms_definition(self):
        forces = np.array([[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]])
        per_atom, frame = committee_force_std(forces, aggregate="mean")
        self.assertAlmostEqual(per_atom[0], 1.0 / np.sqrt(3.0))
        self.assertAlmostEqual(frame, per_atom[0])

    def test_teacher_md_friction_is_explicitly_per_fs(self):
        from ase import units
        self.assertAlmostEqual(langevin_friction({"friction_per_fs": 0.01}), 0.01 / units.fs)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            langevin_friction({"friction": 0.01})

    def test_energy_log_reader_accepts_standard_csv_and_legacy_whitespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "energy.csv"
            csv_path.write_text("step,temperature,potential_energy,kinetic_energy,total_energy\n0,300,-2,1,-1\n10,300,-1.9,1,-0.9\n")
            old_path = root / "energy.dat"
            old_path.write_text("step temp pe ke etotal\n0 300 -2 1 -1\n10 300 -1.9 1 -0.9\n")
            for path in (csv_path, old_path):
                steps, energies = read_energy_log(path)
                np.testing.assert_array_equal(steps, [0, 10])
                np.testing.assert_allclose(energies, [-1.0, -0.9])

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

    def test_simple_nn_template_is_rendered_without_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.yaml"
            params = root / "params_X"
            template.write_text("params: {X_PARAMS_PATH}\nepochs: {TOTAL_EPOCH}\nprecision: {DOUBLE_PRECISION}\nrate: {LEARNING_RATE}\n")
            params.write_text("descriptor")
            cfg = {"_project_dir": str(root), "train": {
                "config_template": str(template), "descriptor_params": {"X": str(params)},
                "nodes": "10-10", "batch_size": 4, "total_epoch": 20,
                "learning_rate": 0.001, "double_precision": True, "use_stress": False}}
            rendered = _render_simple_nn_config(cfg, root / "out")
            text = rendered.read_text()
            self.assertIn("epochs: 20", text)
            self.assertIn("precision: true", text)
            self.assertNotRegex(text, r"\{[A-Z_]+\}")

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
