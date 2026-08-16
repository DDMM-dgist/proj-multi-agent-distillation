"""Regression tests for the physical_validation LAMMPS species-mapping fix.

Forensic background: a raw LAMMPS dump's ``ITEM: ATOMS`` record carries only an
integer ``type`` column (see ``templates/lammps/prod_md.in.template``'s
``dump ... id type x y z fx fy fz``, no ``element``/``species`` column), and ASE's
LAMMPS-dump reader treats bare integer types as literal atomic numbers when no
``specorder`` is given (type 1 -> H, type 2 -> He, ...) -- silently wrong for any real
campaign whose LAMMPS atom-type order doesn't coincidentally match the periodic table
by index. These tests prove: (1) ``validation.species_mapping``'s detection/validation
primitives, (2) ``_exec_build_physical_validation_report`` correctly decodes a LAMMPS
dump when given a resolved ``species_mapping`` and fails closed without one, while
leaving self-describing formats (extxyz) untouched, (3) ``cli.py`` resolves that
mapping from the Controller-bound ``student_config``'s ``deploy.elements`` -- never a
model-supplied hash -- and fails closed on missing/contradictory bindings.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent

_LAMMPS_DUMP_TEMPLATE = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
{n_atoms}
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z fx fy fz
{atom_lines}
"""
_POSITIONS = [(0.0, 0.0, 0.0), (1.6, 0.0, 0.0), (0.0, 1.6, 0.0), (0.0, 0.0, 1.6)]


def _write_lammps_dump(path: Path, types) -> Path:
    """A single-frame LAMMPS ``dump custom ... id type x y z fx fy fz`` file (no
    element/species column) with one atom per declared integer ``type``."""
    lines = [f"{i} {t} {pos[0]} {pos[1]} {pos[2]} 0.0 0.0 0.0"
            for i, (t, pos) in enumerate(zip(types, _POSITIONS), start=1)]
    path.write_text(_LAMMPS_DUMP_TEMPLATE.format(n_atoms=len(types), atom_lines="\n".join(lines)))
    return path


def _validation_profile(path: Path) -> Path:
    path.write_text(yaml.safe_dump({
        "kind": "project-validation",
        "checks": [
            {"name": "rdf_Si_Si", "category": "structure", "required": True, "threshold": None},
            {"name": "rdf_Si_O", "category": "structure", "required": True, "threshold": None},
            {"name": "rdf_O_O", "category": "structure", "required": True, "threshold": None},
            {"name": "coordination_Si", "category": "structure", "required": True, "threshold": None},
            {"name": "coordination_O", "category": "structure", "required": True, "threshold": None},
            {"name": "density", "category": "structure", "required": True, "threshold": None},
        ],
    }))
    return path


class SpeciesMappingPrimitivesTests(unittest.TestCase):
    def test_requires_specorder_true_for_lammps_dump_without_element_column(self):
        from validation.species_mapping import requires_specorder
        with tempfile.TemporaryDirectory() as tmp:
            dump = _write_lammps_dump(Path(tmp) / "trajectory.dump", [1, 2, 1, 2])
            self.assertTrue(requires_specorder(dump))

    def test_requires_specorder_false_for_extxyz(self):
        from validation.species_mapping import requires_specorder
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.extxyz"
            write(str(path), [Atoms("OSi", positions=[[0, 0, 0], [1.6, 0, 0]],
                                    cell=[10, 10, 10], pbc=True)])
            self.assertFalse(requires_specorder(path))

    def test_validate_specorder_rejects_empty(self):
        from validation.species_mapping import validate_specorder
        with self.assertRaises(ValueError):
            validate_specorder([])

    def test_validate_specorder_rejects_duplicates(self):
        from validation.species_mapping import validate_specorder
        with self.assertRaises(ValueError):
            validate_specorder(["O", "O"])

    def test_validate_specorder_rejects_unknown_symbols(self):
        from validation.species_mapping import validate_specorder
        with self.assertRaises(ValueError):
            validate_specorder(["O", "Xx"])

    def test_validate_specorder_accepts_valid_ordered_list(self):
        from validation.species_mapping import validate_specorder
        self.assertEqual(validate_specorder(["O", "Si"]), ["O", "Si"])

    def test_reversing_specorder_changes_decoded_species(self):
        from ase.io import read
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 3x type-1, 1x type-2: asymmetric composition so a reversed ordering is
            # not merely a symmetric relabeling of the same multiset.
            dump = _write_lammps_dump(root / "trajectory.dump", [1, 2, 1, 1])
            forward = read(str(dump), index=":", specorder=["O", "Si"])[0].get_chemical_symbols()
            reversed_ = read(str(dump), index=":", specorder=["Si", "O"])[0].get_chemical_symbols()
            self.assertEqual(forward, ["O", "Si", "O", "O"])
            self.assertEqual(reversed_, ["Si", "O", "Si", "Si"])
            self.assertNotEqual(forward, reversed_)


class PhysicalValidationExecutorSpeciesMappingTests(unittest.TestCase):
    def _run(self, root, frames_path, params_extra):
        from runtimes.pydantic_ai.executors import _exec_build_physical_validation_report
        profile = _validation_profile(root / "validation_profile.yaml")
        report_path = root / "report.json"
        params = {
            "validation_profile": str(profile),
            "frames_path": str(frames_path),
            "r_max": 4.0,
            "cutoffs": {"Si-O": 2.2, "default": 3.0},
            "report_path": str(report_path),
        }
        params.update(params_extra)
        result = _exec_build_physical_validation_report({"parameters": params})
        return result, report_path

    def test_lammps_dump_without_species_mapping_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump = _write_lammps_dump(root / "trajectory.dump", [2, 2, 1, 1])
            with self.assertRaises(ValueError) as ctx:
                self._run(root, dump, {})
            self.assertIn("species_mapping", str(ctx.exception))

    def test_extxyz_frames_do_not_require_species_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_path = root / "frames.extxyz"
            atoms = Atoms("SiSiOO", positions=[p for p in _POSITIONS],
                         cell=[10, 10, 10], pbc=True)
            write(str(frames_path), [atoms])
            _, report_path = self._run(root, frames_path, {})
            report = json.loads(report_path.read_text())
            self.assertNotIn("species_mapping", report)
            names = {c["observable"] for c in report["checks"]}
            self.assertIn("rdf_Si_Si", names)

    def test_lammps_dump_with_species_mapping_uses_real_species_for_rdf_coordination_and_density(self):
        from validation.structure_dynamics import compute_density
        from ase.io import read as ase_read
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump = _write_lammps_dump(root / "trajectory.dump", [2, 2, 1, 1])  # Si, Si, O, O
            species_mapping = {"source": "student_config.deploy.elements",
                               "specorder": ["O", "Si"],
                               "student_config_sha256": "deadbeef"}
            _, report_path = self._run(root, dump, {"species_mapping": species_mapping})
            report = json.loads(report_path.read_text())
            self.assertEqual(report["species_mapping"], species_mapping)
            checks = {c["observable"]: c for c in report["checks"]}
            for observable in ("rdf_Si_Si", "rdf_Si_O", "rdf_O_O",
                              "coordination_Si", "coordination_O", "density"):
                self.assertIn(observable, checks)
            expected_density, _ = compute_density(
                ase_read(str(dump), index=":", specorder=["O", "Si"]))
            self.assertAlmostEqual(checks["density"]["value"], expected_density, places=9)
            # The H/He mis-decoding this fix prevents would give a density roughly an
            # order of magnitude lower (H/He masses vs. true O/Si masses).
            wrong_density, _ = compute_density(ase_read(str(dump), index=":"))
            self.assertGreater(checks["density"]["value"], wrong_density * 5)


def _init_controller_with_student_config(root: Path, student_config_path: Path):
    from workflow.controller import RunController
    workflow = {
        "run_id": "species-mapping-cli-test",
        "stages": [{"name": "physical_validation", "command": None, "outputs": [],
                   "gate": {"criteria": ["physical validation report is complete"]}}],
        "inputs": [str(student_config_path)],
    }
    workflow_path = root / "workflow.yaml"
    workflow_path.write_text(yaml.safe_dump(workflow))
    run_dir = root / "run"
    RunController.initialize(workflow_path, run_dir)
    return RunController(run_dir)


def _write_student_config(root: Path, elements) -> Path:
    path = root / "student.simple-nn.yaml"
    path.write_text(yaml.safe_dump(
        {"kind": "mock", "deploy": {"lammps_pair_style": "nn", "elements": elements}}))
    return path


class CliSpeciesMappingResolutionTests(unittest.TestCase):
    def test_resolves_specorder_from_bound_student_config_using_controller_bound_hash(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = _write_student_config(root, ["O", "Si"])
            controller = _init_controller_with_student_config(root, student_config)
            record = controller.state["inputs"][0]
            student_path = record.get("snapshot") or record["source"]
            params = cli._resolve_physical_validation_species_mapping(
                controller, {"student_config": student_path})
            mapping = params["species_mapping"]
            self.assertEqual(mapping["specorder"], ["O", "Si"])
            self.assertEqual(mapping["source"], "student_config.deploy.elements")
            self.assertEqual(mapping["student_config_sha256"], record["sha256"])

    def test_reversed_student_config_ordering_changes_resolved_specorder(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = _write_student_config(root, ["Si", "O"])
            controller = _init_controller_with_student_config(root, student_config)
            record = controller.state["inputs"][0]
            student_path = record.get("snapshot") or record["source"]
            params = cli._resolve_physical_validation_species_mapping(
                controller, {"student_config": student_path})
            self.assertEqual(params["species_mapping"]["specorder"], ["Si", "O"])

    def test_explicit_specorder_matching_authoritative_passes(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = _write_student_config(root, ["O", "Si"])
            controller = _init_controller_with_student_config(root, student_config)
            record = controller.state["inputs"][0]
            student_path = record.get("snapshot") or record["source"]
            params = cli._resolve_physical_validation_species_mapping(
                controller, {"student_config": student_path, "specorder": ["O", "Si"]})
            self.assertEqual(params["species_mapping"]["specorder"], ["O", "Si"])
            self.assertNotIn("specorder", params)

    def test_explicit_specorder_contradicting_authoritative_fails_closed(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = _write_student_config(root, ["O", "Si"])
            controller = _init_controller_with_student_config(root, student_config)
            record = controller.state["inputs"][0]
            student_path = record.get("snapshot") or record["source"]
            with self.assertRaises(ValueError) as ctx:
                cli._resolve_physical_validation_species_mapping(
                    controller, {"student_config": student_path, "specorder": ["Si", "O"]})
            self.assertIn("contradicts", str(ctx.exception))

    def test_explicit_specorder_without_student_config_fails_closed(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = _write_student_config(root, ["O", "Si"])
            controller = _init_controller_with_student_config(root, student_config)
            with self.assertRaises(ValueError):
                cli._resolve_physical_validation_species_mapping(
                    controller, {"specorder": ["O", "Si"]})

    def test_student_config_not_a_controller_bound_input_fails_closed(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = _write_student_config(root, ["O", "Si"])
            controller = _init_controller_with_student_config(root, student_config)
            unbound = root / "unbound_student.yaml"
            unbound.write_text(student_config.read_text())
            with self.assertRaises(ValueError):
                cli._resolve_physical_validation_species_mapping(
                    controller, {"student_config": str(unbound)})

    def test_no_student_config_and_no_specorder_returns_params_unchanged(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = _write_student_config(root, ["O", "Si"])
            controller = _init_controller_with_student_config(root, student_config)
            params = {"frames_path": "x.extxyz"}
            result = cli._resolve_physical_validation_species_mapping(controller, params)
            self.assertEqual(result, params)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
