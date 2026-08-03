import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixCom
from ase.io import read, write

from adapters import load_config
from adapters.acquisition import acquire, langevin_friction, run_teacher_md
from adapters.contracts import PredictionBatch
from adapters.reference_dft import render_incar, render_reference_input
from adapters.md_backend import render_input as render_md_input, run as run_md
from adapters.preflight import (check_acquisition_config, check_acquisition_files,
                                check_dataset_policy, check_dft_config, check_md_config,
                                check_scope_config,
                                check_student_config, check_teacher_config,
                                check_uncertainty_config, check_validation_profile)
from adapters.student import (_render_simple_nn_config, _train_grace_fs,
                              lammps_pair_style_block, load_student,
                              predict_student, train_student)
from adapters.simple_nn_v2_predict import (
    _materialize_test_list, predict as predict_simple_nn)
from adapters.simple_nn_v2_wrapper import _dataset_to_extxyz_with_ref_labels

try:  # SIMPLE-NN is a runtime dependency, not a test dependency.
    import simple_nn.utils.features  # noqa: F401
    _SIMPLE_NN_AVAILABLE = True
except Exception:
    _SIMPLE_NN_AVAILABLE = False
from adapters.teacher import load_teacher, teacher_model_reference
from adapters.uncertainty import committee_force_std, spearman
from validation.structure_dynamics import (compute_msd, compute_nve_drift, compute_rdf,
                                           read_energy_log)
from validation.report import evidence_record
from validation.surface_energy import (surface_energy, validate_surface_manifest,
                                       validate_surface_report)
from validation.four_channel_audit import channel
from workflow.contracts import validate_validation_manifest
from workflow.steps import merge_datasets, split_dataset


ROOT = Path(__file__).resolve().parent.parent


class AdapterContractTests(unittest.TestCase):
    def test_config_loader_rejects_non_mapping_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            path.write_text("- not\n- a\n- mapping\n")
            with self.assertRaisesRegex(ValueError, "YAML mapping"):
                load_config(path)

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

    def test_pilot_preflight_rejects_placeholders_and_required_teacher_kwargs(self):
        student = {"kind": "grace-fs", "train": {"config_template": "unused"},
                   "deploy": {"elements": ["A", "B", "C"]}}
        with self.assertRaisesRegex(ValueError, "placeholder"):
            check_student_config(student, check_files=False, require_ready=True)
        teacher = {"kind": "mace-mh1", "model": "model", "calculator": {
            "factory": "mace.calculators.mace_mp", "required_kwargs": ["head"],
            "kwargs": {}}}
        with self.assertRaisesRegex(ValueError, "head"):
            check_teacher_config(teacher, check_files=False)

    def test_unknown_student_kind_uses_configured_adapter_callables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.extxyz"
            write(dataset, Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True))
            cfg = {
                "kind": "brand-new-architecture", "adapter_test_token": "ok",
                "adapter": {
                    "train": "adapters.mock_model.train_external_adapter",
                    "load": "adapters.mock_model.load_external_adapter",
                    "deploy": "adapters.mock_model.deploy_external_adapter",
                    "preflight": "adapters.mock_model.preflight_external_adapter",
                },
                "committee": {"n_seeds": 2},
                "deploy": {"elements": ["Cu"]},
                "predict": {"factory": "adapters.mock_model.MockCheckpointCalculator",
                            "checkpoint_arg": "checkpoint"},
            }
            checks = check_student_config(cfg, check_files=True, require_ready=True)
            self.assertIn("student kind=brand-new-architecture", checks)
            artifact = train_student(cfg, dataset, root / "out", 7)
            self.assertEqual(artifact.kind, "brand-new-architecture")
            self.assertEqual(load_student(cfg, artifact.path).path, artifact.path)
            block = lammps_pair_style_block(cfg, artifact)
            self.assertIn("pair_style external", block)
            self.assertIn("Cu", block)

    def test_unknown_student_kind_can_use_command_and_declarative_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {
                "kind": "command-student",
                "train": {
                    "command": [sys.executable, "-c",
                                "from pathlib import Path; "
                                "Path(r'{out_dir}/model.bin').write_text('ok')"],
                    "artifact": "model.bin",
                },
                "deploy": {"elements": ["Al", "N"],
                           "lammps_pair_style": "custom/style",
                           "pair_coeff_template": "pair_coeff * * {checkpoint} {elements}"},
            }
            checks = check_student_config(cfg, check_files=True)
            self.assertIn("training adapter=command", checks)
            artifact = train_student(cfg, root / "data", root / "out", 1)
            self.assertEqual(artifact.path.read_text(), "ok")
            self.assertIn("pair_style custom/style",
                          lammps_pair_style_block(cfg, artifact))

    def test_student_prediction_uses_configured_callable(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.json"
            checkpoint.write_text('{"seed": 3}')
            structures = [Atoms("Cu", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)]
            cfg = {"kind": "external", "adapter": {
                "predict": "adapters.mock_model.predict_external_adapter"}}
            result = predict_student(cfg, checkpoint, structures)
            self.assertEqual(len(result.energies), 1)
            self.assertEqual(result.forces[0].shape, (1, 3))

    def test_student_prediction_command_uses_validated_extxyz_exchange(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.bin"
            checkpoint.write_text("model")
            structures = [Atoms("Cu", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)]
            cfg = {"kind": "command-student", "predict": {
                "command": ["predict-tool", "{structures}", "{output}"]}}

            def fake_run(command, **kwargs):
                frames = read(command[1], index=":")
                for atoms in frames:
                    atoms.info["student_energy"] = 1.25
                    atoms.arrays["student_forces"] = np.zeros((len(atoms), 3))
                write(command[2], frames, format="extxyz")
                return subprocess.CompletedProcess(command, 0)

            with patch("adapters.student.subprocess.run", side_effect=fake_run):
                result = predict_student(cfg, checkpoint, structures)
            self.assertEqual(result.energies.tolist(), [1.25])
            self.assertEqual(result.forces[0].shape, (1, 3))

    def test_ready_student_requires_a_prediction_interface(self):
        with self.assertRaisesRegex(ValueError, "requires adapter.predict"):
            check_student_config({"kind": "mock"}, check_files=False, require_ready=True)

    def test_simple_nn_label_materialization_fails_closed_for_partial_stress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for index in range(2):
                atoms = Atoms("H", positions=[[0, 0, 0]])
                atoms.info["teacher_energy"] = 0.0
                atoms.arrays["teacher_forces"] = np.zeros((1, 3))
                if index == 0:
                    atoms.info["teacher_stress"] = np.zeros(6).tolist()
                frames.append(atoms)
            source = root / "frames.extxyz"
            write(source, frames)
            with self.assertRaisesRegex(ValueError, "teacher_stress on every frame"):
                _dataset_to_extxyz_with_ref_labels(
                    source, root / "labeled.extxyz", require_stress=True
                )

    def test_simple_nn_prediction_parses_and_binds_test_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root / "seed-1"
            training.mkdir()
            checkpoint = training / "potential_saved_bestmodel"
            checkpoint.write_text("model")
            (training / "simple_nn_input.yaml").write_text(
                "neural_network:\n  use_scale: false\n  use_pca: false\n"
            )
            atoms = Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
            atoms.info["teacher_energy"] = 1.5
            atoms.arrays["teacher_forces"] = np.asarray([[0.1, 0.2, 0.3]])
            structures = root / "heldout.extxyz"
            output = root / "predictions.extxyz"
            write(structures, [atoms])

            result = {
                "N": [1],
                "NN_E": [1.25],
                "NN_F": [np.asarray([[0.01, 0.02, 0.03]])],
                "DFT_E": [1.5],
                "DFT_F": [np.asarray([[0.1, 0.2, 0.3]])],
            }

            def fake_run(input_path, cwd):
                if Path(input_path).name == "generate_input.yaml":
                    (Path(cwd) / "total_list").write_text("1:./feature.pt\n")
                    (Path(cwd) / "feature.pt").write_text("stub")
                else:
                    (Path(cwd) / "test_result").write_text("mock")

            # `_materialize_test_list` reaches into SIMPLE-NN's own tag parser;
            # keep this wiring test SIMPLE-NN-free by faking that call. The
            # tag-stripping contract is exercised separately in
            # SimpleNnTestListMaterializationTests below.
            def fake_materialize(total_list, work_dir, n_frames):
                out = Path(work_dir) / "test_list"
                out.write_text("./feature.pt\n" * n_frames)
                return out

            with patch("adapters.simple_nn_v2_predict._run_simple_nn",
                       side_effect=fake_run), patch(
                           "adapters.simple_nn_v2_predict._materialize_test_list",
                           side_effect=fake_materialize), patch(
                           "adapters.simple_nn_v2_predict._load_test_result",
                           return_value=result):
                predict_simple_nn(checkpoint, structures, output)

            predicted = read(output, index=0)
            self.assertAlmostEqual(predicted.info["student_energy"], 1.25)
            np.testing.assert_allclose(
                predicted.arrays["student_forces"], [[0.01, 0.02, 0.03]]
            )

            result["DFT_E"] = [9.0]
            with patch("adapters.simple_nn_v2_predict._run_simple_nn",
                       side_effect=fake_run), patch(
                           "adapters.simple_nn_v2_predict._materialize_test_list",
                           side_effect=fake_materialize), patch(
                           "adapters.simple_nn_v2_predict._load_test_result",
                           return_value=result), self.assertRaisesRegex(
                               ValueError, "reference labels do not match"):
                predict_simple_nn(checkpoint, structures, output)

    @unittest.skipUnless(_SIMPLE_NN_AVAILABLE,
                         "SIMPLE-NN v2 not importable in this environment")
    def test_simple_nn_materialize_test_list_strips_tags_via_simple_nn(self):
        # Regression: SIMPLE-NN's test-mode FilelistDataset reads each line of
        # the test_list verbatim (data_handler.py:22-28), so the tagged
        # "TAG:PATH" lines produced by feature generation must be stripped
        # before test mode. This test exercises the real SIMPLE-NN parser.
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            data = work / "data"
            data.mkdir()
            for name in ("data1.pt", "data2.pt"):
                (data / name).write_text("stub")
            total = work / "total_list"
            total.write_text("1:./data/data1.pt\n1:./data/data2.pt\n")
            out = _materialize_test_list(total, work, 2)
            self.assertEqual(out, work / "test_list")
            self.assertEqual(
                out.read_text(),
                "./data/data1.pt\n./data/data2.pt\n",
            )

    @unittest.skipUnless(_SIMPLE_NN_AVAILABLE,
                         "SIMPLE-NN v2 not importable in this environment")
    def test_simple_nn_materialize_test_list_rejects_empty_total_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            total = work / "total_list"
            total.write_text("")
            with self.assertRaisesRegex(
                    RuntimeError, "0 entries; expected 2"):
                _materialize_test_list(total, work, 2)

    @unittest.skipUnless(_SIMPLE_NN_AVAILABLE,
                         "SIMPLE-NN v2 not importable in this environment")
    def test_simple_nn_materialize_test_list_rejects_frame_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            data = work / "data"
            data.mkdir()
            (data / "data1.pt").write_text("stub")
            total = work / "total_list"
            total.write_text("1:./data/data1.pt\n")  # one entry
            with self.assertRaisesRegex(
                    RuntimeError, r"1 entries; expected 3"):
                _materialize_test_list(total, work, 3)

    @unittest.skipUnless(_SIMPLE_NN_AVAILABLE,
                         "SIMPLE-NN v2 not importable in this environment")
    def test_simple_nn_materialize_test_list_rejects_missing_feature_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            total = work / "total_list"
            total.write_text("1:./data/does_not_exist.pt\n")
            with self.assertRaisesRegex(
                    RuntimeError, "feature file is missing"):
                _materialize_test_list(total, work, 1)

    def test_unknown_md_and_reference_backends_use_configured_callables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md_cfg = {"kind": "external-md", "adapter": {
                "renderer": "adapters.mock_model.render_external_md",
                "runner": "adapters.mock_model.run_external_md"}}
            dft_cfg = {"kind": "external-reference", "reference_theory": "mock-level-1", "adapter": {
                "renderer": "adapters.mock_model.render_external_reference"}}
            self.assertIn("MD backend adapter=configured", check_md_config(md_cfg))
            self.assertIn("DFT renderer adapter=configured", check_dft_config(dft_cfg))
            output = root / "reference.json"
            render_reference_input(dft_cfg, output, {"cutoff": 1})
            self.assertEqual(json.loads(output.read_text())["overrides"]["cutoff"], 1)
            md_input = root / "external.md"
            render_md_input(md_cfg, {"kind": "x"}, root / "model", "unused", {}, md_input)
            self.assertTrue(md_input.is_file())
            self.assertEqual(run_md(md_cfg, md_input, root, mpi_ranks=3)["mpi_ranks"], 3)

    def test_teacher_factory_may_be_model_free_when_contract_declares_it(self):
        cfg = {"kind": "model-free-calculator", "calculator": {
            "factory": "ase.calculators.emt.EMT", "model_arg": None}}
        checks = check_teacher_config(cfg, check_files=True)
        self.assertIn("teacher calculator=ase.calculators.emt.EMT", checks)

    def test_teacher_factory_accepts_positional_model_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.json"
            checkpoint.write_text('{"seed": 7}')
            cfg = {"kind": "positional-factory", "model": str(checkpoint),
                   "calculator": {
                       "factory": "adapters.mock_model.MockCheckpointCalculator",
                       "model_arg": "__positional__", "kwargs": {}}}
            calculator = load_teacher(cfg)
            self.assertEqual(calculator.seed, 7)

    def test_unknown_acquisition_backend_uses_configured_callable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed, output = root / "seed.extxyz", root / "output.extxyz"
            write(seed, Atoms("H", positions=[[0, 0, 0]]))
            cfg = {"kind": "external-acquisition", "adapter": {
                "acquire": "adapters.mock_model.acquire_external_adapter"}}
            self.assertIn("acquisition adapter=configured", check_acquisition_config(cfg))
            check_acquisition_files(cfg)
            acquire(cfg, {}, seed, output)
            self.assertEqual(read(output).info["parent_structure_id"], "external-0")

    def test_preflight_requires_teacher_md_friction_units_and_resolved_thresholds(self):
        acquisition = {"kind": "teacher-md", "temperature_K": 1000, "timestep_fs": 1,
                       "n_steps": 10, "snapshot_interval": 1, "friction": 0.01}
        with self.assertRaisesRegex(ValueError, "friction"):
            check_acquisition_config(acquisition)
        profile = {"kind": "x", "checks": ["surface_excess_energy"],
                   "surface_energetics": {"thresholds": {"student_dft": None}}}
        with self.assertRaisesRegex(ValueError, "unresolved"):
            check_validation_profile(profile, require_ready=True)

    def test_preflight_checks_scope_structured_validation_and_dataset_policy(self):
        scope = {"deployment_domain": {"system": "generic"},
                 "teacher_training_data_access": "unavailable",
                 "approval_boundaries": {"costly_training": True}}
        self.assertIn("scope teacher-data access=unavailable",
                      check_scope_config(scope, require_ready=True))
        profile = {"kind": "generic", "checks": [{
            "name": "density", "category": "structure",
            "purpose": "student_teacher_fidelity", "reference_source": "teacher",
            "protocol": "density-v1", "required": True,
            "threshold": {"operator": "max", "threshold": 0.1},
        }]}
        self.assertIn("validation checks=density",
                      check_validation_profile(profile, require_ready=True))
        policy = {"teacher_training_data_access": "unavailable", "sources": [{
            "category": "generated_teacher_labeled", "label_sources": ["teacher"],
            "selection": "project seed structures",
        }], "replay_policy": {"enabled": False}, "duplicate_policy": "error",
            "coverage_dimensions": []}
        self.assertIn("dataset sources=generated_teacher_labeled",
                      check_dataset_policy(policy, require_ready=True))

    def test_preflight_rejects_invalid_md_sampling_and_missing_augment_binary(self):
        invalid = {"kind": "teacher-md", "temperature_K": -1, "timestep_fs": 0,
                   "n_steps": -2, "snapshot_interval": 0, "friction_per_fs": -0.1}
        with self.assertRaisesRegex(ValueError, "temperature_K"):
            check_acquisition_config(invalid)
        augment = {"kind": "augment-atoms", "command": ["missing-augment"]}
        check_acquisition_config(augment)
        with patch("adapters.preflight.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "missing-augment"):
                check_acquisition_files(augment)

    def test_preflight_checks_committee_uncertainty_md_and_dft_contracts(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            check_student_config({"kind": "mock", "deploy": {"elements": ["Cu"]},
                                  "committee": {"n_seeds": 0}}, check_files=False)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            check_uncertainty_config({"kind": "committee-force-std", "aggregate": "median"})
        self.assertIn("MD backend kind=lammps", check_md_config(
            {"kind": "lammps", "template_dir": "missing", "binary": "lmp"},
            check_files=False))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "INCAR"
            template.write_text("{XC_BLOCK}\nENCUT={ENCUT}\n")
            cfg = {"kind": "vasp", "reference_theory": "SCAN",
                   "incar_template": str(template), "encut_ev": 400,
                   "kspacing_inv_angstrom": 0.2, "smearing": {"ismear": 0, "sigma": 0.1},
                   "relaxation": {"nsw": 0, "ibrion": -1}, "template_variables": {}}
            with self.assertRaisesRegex(ValueError, "XC_BLOCK"):
                check_dft_config(cfg)

    def test_surface_energy_uses_all_exposed_surfaces(self):
        self.assertAlmostEqual(surface_energy(12.0, 10.0, 5.0, 2), 0.2)
        self.assertAlmostEqual(surface_energy("12.0", "10.0", "5.0", "2"), 0.2)

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

    def test_surface_report_binds_raw_evidence_and_profile_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "validation.yaml"
            profile.write_text(
                "surface_energetics:\n"
                "  thresholds:\n"
                "    student_teacher_max_abs_J_m2: 0.1\n"
                "    student_dft_max_abs_J_m2: 0.05\n"
                "required_pass_observables:\n"
                "  - surface_delta:student_teacher\n"
                "  - surface_delta:student_dft\n"
            )
            evidence = {}
            for method in ("teacher", "student", "dft"):
                for part in ("slab", "bulk"):
                    role = f"{method}_{part}"
                    path = root / f"{role}.out"
                    path.write_text(f"raw {role}\n")
                    evidence[role] = path
            energies = {"teacher": 2.0, "student": 3.0, "dft": 2.5}
            entries = []
            values = {}
            for method, delta_e in energies.items():
                value = surface_energy(delta_e, 0.0, 100.0, 2) * 16.02176634
                values[method] = value
                entries.append({
                    "method": method, "orientation": "x", "termination": "t",
                    "area_a2": 100.0, "n_surfaces": 2,
                    "geometry_protocol": "g", "reference_convention": "r",
                    "composition_balance_confirmed": True, "nonstoichiometric": False,
                    "slab_energy_ev": delta_e, "bulk_reference_ev": 0.0,
                    "surface_energy_J_m2": value,
                    "slab_evidence_role": f"{method}_slab",
                    "bulk_evidence_role": f"{method}_bulk",
                })
            checks = []
            for observable, value, threshold in (
                ("surface_delta:student_teacher",
                 values["student"] - values["teacher"], 0.1),
                ("surface_delta:student_dft",
                 values["student"] - values["dft"], 0.05),
            ):
                checks.append({"domain": "energetics", "observable": observable,
                               "status": "PASS", "value": value, "unit": "J/m2",
                               "criterion": {"operator": "max_abs",
                                             "threshold": threshold}})
            report = root / "surface-report.json"
            records = [evidence_record(role, path) for role, path in evidence.items()]
            report.write_text(json.dumps({
                "schema_version": 1, "profile": str(profile.resolve()),
                "checks": checks, "evidence": records,
                "surface": {"quantity": "static_surface_excess_energy",
                            "unit": "J/m2", "entries": entries},
            }))
            submitted = [report, *evidence.values()]
            validate_surface_report(report, profile,
                                    required_methods=["teacher", "student", "dft"],
                                    submitted_artifacts=submitted,
                                    allowed_evidence=submitted,
                                    enforce_required_pass=True)
            checks[0]["criterion"]["threshold"] = 0.2
            report.write_text(json.dumps({
                "schema_version": 1, "profile": str(profile.resolve()),
                "checks": checks, "evidence": records,
                "surface": {"quantity": "static_surface_excess_energy",
                            "unit": "J/m2", "entries": entries},
            }))
            with self.assertRaisesRegex(ValueError, "does not match profile"):
                validate_surface_report(report, profile,
                                        required_methods=["teacher", "student", "dft"],
                                        submitted_artifacts=submitted,
                                        allowed_evidence=submitted)

    def test_external_validation_callable_is_not_restricted_to_builtin_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external.json"
            path.write_text(json.dumps({"value": 3}))
            payload = validate_validation_manifest(
                path, "adapters.mock_model.validate_external_manifest",
                {"expected_value": 3})
            self.assertEqual(payload["value"], 3)

    def test_surface_manifest_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surface.json"
            entries = []
            for method in ("teacher", "student", "dft"):
                entries.append({"method": method, "orientation": "x", "termination": "t",
                                "area_a2": 1.0, "n_surfaces": 2,
                                "geometry_protocol": "g", "reference_convention": "r",
                                "composition_balance_confirmed": True,
                                "nonstoichiometric": False, "slab_energy_ev": float("nan"),
                                "bulk_reference_ev": 0.0,
                                "surface_energy_J_m2": float("nan")})
            path.write_text(json.dumps({"quantity": "static_surface_excess_energy",
                                        "unit": "J/m2", "entries": entries}))
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_surface_manifest(path, ["teacher", "student", "dft"])

    def test_required_channel_rejects_partial_frame_coverage(self):
        frames = []
        for index in range(3):
            atoms = Atoms("H", positions=[[0, 0, 0]])
            atoms.info["teacher_energy"] = 0.0
            atoms.arrays["teacher_forces"] = np.zeros((1, 3))
            if index == 0:
                atoms.info["student_energy_seed01"] = 0.0
                atoms.arrays["student_forces_seed01"] = np.zeros((1, 3))
            frames.append(atoms)
        with self.assertRaisesRegex(RuntimeError, "1/3 frames"):
            channel(frames, "teacher", "student", require_complete=True)

    def test_msd_unwraps_periodic_boundary_crossing(self):
        first = Atoms("H", positions=[[9.9, 0, 0]], cell=[10, 10, 10], pbc=True)
        second = Atoms("H", positions=[[0.1, 0, 0]], cell=[10, 10, 10], pbc=True)
        self.assertAlmostEqual(compute_msd([first, second])["H"][-1], 0.04, places=8)

    def test_rdf_uses_supported_ase_api(self):
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]],
                      cell=[20, 20, 20], pbc=True)
        distances, partial = compute_rdf([atoms], ["Cu"], r_max=6.0, nbins=50)
        self.assertEqual(len(distances), 50)
        self.assertEqual(len(partial["Cu-Cu"]), 50)
        self.assertTrue(np.isfinite(partial["Cu-Cu"]).all())

    def test_rdf_heteronuclear_pair_is_finite_with_symbol_keys(self):
        # Two species so both homo- and heteronuclear partials are well defined.
        atoms = Atoms("Cu2O2",
                      positions=[[0, 0, 0], [5.0, 0, 0], [2.5, 0, 0], [7.5, 0, 0]],
                      cell=[20, 20, 20], pbc=True)
        distances, partial = compute_rdf([atoms], ["Cu", "O"], r_max=6.0, nbins=40)
        # Keys stay symbol-based (sorted), e.g. "Cu-O", not atomic numbers.
        self.assertEqual(set(partial), {"Cu-Cu", "Cu-O", "O-O"})
        for key, rdf in partial.items():
            self.assertEqual(len(rdf), len(distances))
            self.assertEqual(len(rdf), 40)
            self.assertTrue(np.isfinite(rdf).all(), key)

    def test_rdf_raises_on_non_finite_instead_of_silent_zero(self):
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]],
                      cell=[20, 20, 20], pbc=True)
        nan_rdf = (np.full(50, np.nan), np.linspace(0, 6.0, 50))
        with patch("validation.structure_dynamics.get_rdf", return_value=nan_rdf):
            with self.assertRaisesRegex(ValueError, "non-finite RDF for pair Cu-Cu"):
                compute_rdf([atoms], ["Cu"], r_max=6.0, nbins=50)

    def test_rdf_raises_on_wrong_length_with_pair_name(self):
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]],
                      cell=[20, 20, 20], pbc=True)
        # get_rdf returns fewer bins than requested -> must raise, not truncate.
        short = (np.ones(10), np.linspace(0, 6.0, 10))
        with patch("validation.structure_dynamics.get_rdf", return_value=short):
            with self.assertRaisesRegex(ValueError, "wrong length.*Cu-Cu|Cu-Cu.*wrong length"):
                compute_rdf([atoms], ["Cu"], r_max=6.0, nbins=50)

    def test_nve_drift_uses_sampling_stride(self):
        energies = np.array([0.0, 0.1, 0.2])
        stride_drift, _ = compute_nve_drift(energies, 1.0, 1, sample_interval_steps=100)
        unit_drift, _ = compute_nve_drift(energies, 1.0, 1)
        self.assertAlmostEqual(stride_drift * 100, unit_drift)
        with self.assertRaisesRegex(ValueError, "at least two"):
            compute_nve_drift([0.0], 1.0, 1)

    def test_uncertainty_cli_honors_config_and_records_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_path, config_path, report_path = (root / "frames.extxyz",
                                                      root / "uncertainty.yaml",
                                                      root / "report.json")
            frames = []
            for index in range(1, 11):
                atoms = Atoms("H", positions=[[0, 0, 0]])
                atoms.arrays["teacher_forces"] = np.zeros((1, 3))
                atoms.arrays["student_forces_seed01"] = np.array([[index * 0.01, 0, 0]])
                atoms.arrays["student_forces_seed02"] = np.array([[index * 0.03, 0, 0]])
                frames.append(atoms)
            write(frames_path, frames)
            config_path.write_text("aggregate: mean\ntop_fraction: 0.3\nrequire_complete: true\n")
            subprocess.run([sys.executable, str(ROOT / "validation/committee_uncertainty.py"),
                            str(frames_path), "--config", str(config_path),
                            "--output", str(report_path)], check=True, capture_output=True,
                           text=True)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["top_fraction"], 0.3)
            self.assertEqual(report["n_total_frames"], 10)
            self.assertEqual(report["n_skipped_frames"], 0)
            self.assertEqual(report["n_committee_seeds"], 2)
            self.assertEqual({item["role"] for item in report["evidence"]},
                             {"evaluated_frames", "uncertainty_config"})

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

    def test_teacher_md_seed_reproduces_thermostat_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_path = root / "seed.extxyz"
            atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[6, 6, 6],
                          pbc=True)
            atoms.info["parent_structure_id"] = "seed-1"
            write(seed_path, atoms)
            teacher_cfg = {"kind": "mock", "calculator": {
                "factory": "ase.calculators.emt.EMT", "model_arg": None, "kwargs": {}}}
            acquisition_cfg = {"temperature_K": 300, "timestep_fs": 0.5,
                               "friction_per_fs": 0.01, "n_steps": 3,
                               "snapshot_interval": 1, "seed": 17}
            first_path, second_path = root / "first.extxyz", root / "second.extxyz"
            run_teacher_md(acquisition_cfg, teacher_cfg, seed_path, first_path)
            run_teacher_md(acquisition_cfg, teacher_cfg, seed_path, second_path)
            first, second = read(first_path, index=":"), read(second_path, index=":")
            self.assertEqual(len(first), len(second))
            self.assertGreater(len(first), 0)
            for left, right in zip(first, second):
                # (5) identical positions for the same random seed
                np.testing.assert_allclose(left.positions, right.positions)
                self.assertEqual(left.info["random_seed"], 17)
                # (2) the runtime-added FixCom must not leak into the snapshot
                self.assertEqual([type(c).__name__ for c in left.constraints], [])
                # (A) FixCom provenance for a no-constraint seed with runtime FixCom
                self.assertTrue(left.info["runtime_fixcom_applied"])
                self.assertFalse(left.info["source_had_fixcom"])
                self.assertTrue(left.info["snapshot_fixcom_omitted"])
                # (3) fix_center_of_mass metadata preserved and True
                self.assertTrue(left.info["fix_center_of_mass"])
                # (7) provenance metadata preserved through write/read
                self.assertEqual(left.info["parent_structure_id"], "seed-1")
                self.assertEqual(left.info["timestep_fs"], 0.5)
                self.assertEqual(left.info["snapshot_interval"], 1)
                # (6) momenta/velocities survive the round-trip
                self.assertEqual(left.get_momenta().shape, (2, 3))
                self.assertTrue(np.isfinite(left.get_momenta()).all())
                self.assertTrue(np.any(left.get_momenta() != 0.0))

    def test_teacher_md_preserves_seed_constraints_and_drops_runtime_fixcom(self):
        # A seed with a genuine FixAtoms constraint: the run adds a FixCom for the
        # MD, which must be stripped from the snapshot while FixAtoms is preserved.
        # We inspect the in-memory snapshots (by intercepting the writer) so the test
        # checks our restoration logic, not ASE's version-dependent extxyz constraint
        # serialization (3.23-3.25 do not persist FixAtoms to extxyz at all).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # .traj preserves the seed's FixAtoms on every ASE version (extxyz does
            # not persist it before 3.26), so this isolates our snapshot restoration
            # from ASE's version-dependent extxyz constraint serialization.
            seed_path = root / "seed.traj"
            atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[6, 6, 6],
                          pbc=True)
            atoms.info["parent_structure_id"] = "seed-fix"
            atoms.set_constraint([FixAtoms(indices=[0])])
            write(seed_path, atoms)
            teacher_cfg = {"kind": "mock", "calculator": {
                "factory": "ase.calculators.emt.EMT", "model_arg": None, "kwargs": {}}}
            acquisition_cfg = {"temperature_K": 300, "timestep_fs": 0.5,
                               "friction_per_fs": 0.01, "n_steps": 2,
                               "snapshot_interval": 1, "seed": 3,
                               "fix_center_of_mass": True}
            out_path = root / "out.extxyz"
            captured = {}

            def capture_write(path, images, **kwargs):
                captured["frames"] = [im.copy() for im in images]

            with patch("adapters.acquisition.write", side_effect=capture_write):
                run_teacher_md(acquisition_cfg, teacher_cfg, seed_path, out_path)
            frames = captured["frames"]
            self.assertGreater(len(frames), 0)
            for frame in frames:
                kinds = [type(c).__name__ for c in frame.constraints]
                # (4/C) original FixAtoms preserved on the snapshot; runtime FixCom removed
                self.assertIn("FixAtoms", kinds)
                self.assertNotIn("FixCom", kinds)
                self.assertFalse(frame.info["source_had_fixcom"])
                self.assertTrue(frame.info["runtime_fixcom_applied"])
                self.assertTrue(frame.info["snapshot_fixcom_omitted"])

    def test_teacher_md_seed_with_existing_fixcom_writes_and_records_provenance(self):
        # A seed that ALREADY carries a FixCom: no runtime FixCom is added, but the
        # snapshot must still omit FixCom (ASE 3.26 cannot serialize it) and record
        # that omission, rather than silently dropping it or crashing the writer.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_path = root / "seed.traj"
            atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[6, 6, 6],
                          pbc=True)
            atoms.info["parent_structure_id"] = "seed-fixcom"
            atoms.set_constraint([FixCom()])
            write(seed_path, atoms)
            teacher_cfg = {"kind": "mock", "calculator": {
                "factory": "ase.calculators.emt.EMT", "model_arg": None, "kwargs": {}}}
            acquisition_cfg = {"temperature_K": 300, "timestep_fs": 0.5,
                               "friction_per_fs": 0.01, "n_steps": 2,
                               "snapshot_interval": 1, "seed": 5,
                               "fix_center_of_mass": True}
            out_path = root / "out.extxyz"
            # Real extxyz write + read: must not crash (this is the ASE 3.26 regression).
            run_teacher_md(acquisition_cfg, teacher_cfg, seed_path, out_path)
            frames = read(out_path, index=":")
            self.assertGreater(len(frames), 0)
            for frame in frames:
                self.assertNotIn("FixCom",
                                 [type(c).__name__ for c in frame.constraints])
                self.assertTrue(frame.info["source_had_fixcom"])
                self.assertFalse(frame.info["runtime_fixcom_applied"])
                self.assertTrue(frame.info["snapshot_fixcom_omitted"])

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

    def test_dataset_merge_preserves_lineage_and_controls_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for source_index, offset in enumerate((0.0, 1.0)):
                path = root / f"source-{source_index}.extxyz"
                atoms = [Atoms("H", positions=[[offset, 0, 0]],
                               info={"parent_structure_id": "shared-parent",
                                     "structure_id": "frame-0"})]
                if source_index == 0:
                    atoms.append(Atoms("H", positions=[[0.5, 0, 0]],
                                       info={"parent_structure_id": "shared-parent",
                                             "structure_id": "frame-0"}))
                write(path, atoms); sources.append(path)
            output, manifest = root / "merged.extxyz", root / "merge.json"
            result = merge_datasets(sources, output, manifest)
            frames = read(output, index=":")
            self.assertEqual(result["n_frames"], 3)
            self.assertEqual({a.info["parent_structure_id"] for a in frames},
                             {"shared-parent"})
            self.assertEqual(len({a.info["structure_id"] for a in frames}), 3)
            with self.assertRaisesRegex(ValueError, "exact duplicate"):
                merge_datasets([sources[0], sources[0]], root / "bad.extxyz",
                               root / "bad.json")

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

    def test_dft_template_uses_configured_reference_theory_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "INCAR.template"
            template.write_text("{XC_BLOCK}\nENCUT={ENCUT}\nKSPACING={KSPACING}\n"
                                "ISMEAR={ISMEAR}\nSIGMA={SIGMA}\nNSW={NSW}\nIBRION={IBRION}\n")
            cfg = {"kind": "vasp", "incar_template": str(template),
                   "reference_theory": "PBE", "template_variables": {"XC_BLOCK": "GGA = PE"},
                   "encut_ev": 500, "kspacing_inv_angstrom": 0.25,
                   "smearing": {"ismear": 0, "sigma": 0.05},
                   "relaxation": {"nsw": 0, "ibrion": -1}}
            output = root / "INCAR"
            render_incar(cfg, output)
            self.assertIn("GGA = PE", output.read_text())
            self.assertNotIn("SCAN", output.read_text())


if __name__ == "__main__":
    unittest.main()
