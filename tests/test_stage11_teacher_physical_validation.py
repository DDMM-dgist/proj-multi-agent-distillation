import copy
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

import validation.teacher_physical_validation as tpv
from validation.report import validate_validation_report
from workflow.integrity import sha256_file


def _frames(n_frames=4):
    base = np.array([[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0], [1.6, 1.6, 0]], dtype=float)
    syms = ["Si", "O", "Si", "O"]
    out = []
    for i in range(n_frames):
        a = Atoms(syms, positions=base + [0.01 * i, 0, 0], cell=[10, 10, 10], pbc=True)
        out.append(a)
    return out


class MockCalc(Calculator):
    implemented_properties = ["energy", "forces"]
    calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        type(self).calls += 1
        self.results["energy"] = -float(len(atoms))
        self.results["forces"] = np.zeros((len(atoms), 3))


class Provider:
    def make_ase_calculator(self):
        return MockCalc()


def _engine(atoms, calc, params, seed, sample_fn):
    for step in range(0, int(params["n_steps"]) + 1, int(params["sample_stride"])):
        moved = atoms.copy()
        moved.positions[:, 0] += 0.01 * step
        moved.calc = calc
        sample_fn(moved, step)


MD = {"ensemble": "NVE", "timestep_fs": 1.0, "n_steps": 8, "sample_stride": 2, "seed": 1}
TEACHER = {"model": "/mock", "model_sha256": "a" * 64}
RDF = {"name": "rdf_Si_O", "kind": "rdf_peak_position", "units": "A",
       "params": {"center_species": "Si", "neighbor_species": "O", "r_max": 4.0, "nbins": 80},
       "comparison_criterion": {"operator": "max_abs_deviation", "threshold": 0.1}}
DENSITY = {"name": "density", "kind": "density", "units": "g/cm3",
           "params": {}}


def _target(tmp_path, specs=None):
    start = tmp_path / "start.extxyz"
    write(str(start), [_frames(1)[0]], format="extxyz")
    return tpv.compute_teacher_validation_target(
        objective_profile_sha256="profile-sha",
        teacher_identity=TEACHER,
        md_protocol=MD,
        start_structures_path=str(start),
        observable_specs=specs or [RDF, DENSITY],
        target_path=str(tmp_path / "target.json"),
        trajectory_out_path=str(tmp_path / "teacher_traj.extxyz"),
        teacher_calculator_provider=Provider(),
        md_engine=_engine,
    )


def _profile(tmp_path):
    p = tmp_path / "validation_profile.yaml"
    p.write_text("kind: physical_validation\nchecks: []\n")
    return p


def test_teacher_target_is_hash_bound_and_tamper_evident(tmp_path):
    target = _target(tmp_path)
    assert target["artifact"] == "teacher_validation_target"
    assert tpv.verify_teacher_validation_target(tmp_path / "target.json")["target_sha256"] == target["target_sha256"]
    changed = copy.deepcopy(target)
    changed["observable_results"][0]["value"] = 999
    with pytest.raises(ValueError, match="integrity mismatch"):
        tpv.verify_teacher_validation_target(changed)



def test_executor_builds_teacher_physical_validation_target_with_mock_provider(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_teacher_physical_validation_target

    start = tmp_path / "start.extxyz"
    write(str(start), [_frames(1)[0]], format="extxyz")
    res = _exec_build_teacher_physical_validation_target({"parameters": {
        "objective_profile_sha256": "profile-sha",
        "teacher_identity": TEACHER,
        "md_protocol": MD,
        "observable_specs": [RDF],
        "target_path": str(tmp_path / "executor_target.json"),
        "mode": "COMPUTE",
        "start_structures_path": str(start),
        "trajectory_out_path": str(tmp_path / "executor_teacher_traj.extxyz"),
        "teacher_calculator_provider": Provider(),
        "md_engine": _engine,
    }})
    assert Path(res["path"]).is_file()
    assert res["target"]["target_sha256"]
    assert tpv.verify_teacher_validation_target(res["path"])["target_sha256"] == res["target"]["target_sha256"]

def test_stage11_reproduction_uses_same_observable_definitions(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_physical_validation_report

    target = _target(tmp_path)
    student_frames = tmp_path / "student.extxyz"
    write(str(student_frames), read(target["trajectory"]["path"], index=":"), format="extxyz")
    report_path = tmp_path / "validation_report.json"
    res = _exec_build_physical_validation_report({"parameters": {
        "validation_profile": str(_profile(tmp_path)),
        "frames_path": str(student_frames),
        "report_path": str(report_path),
        "teacher_validation_target": str(tmp_path / "target.json"),
        "teacher_validation_target_sha256": target["target_sha256"],
    }})
    report = validate_validation_report(report_path)
    assert report["mode"] == "teacher_target_reproduction"
    assert res["teacher_validation_target_sha256"] == target["target_sha256"]
    checks = {c["observable"]: c for c in report["checks"]}
    assert checks["rdf_Si_O"]["status"] == "PASS"
    assert checks["rdf_Si_O"]["details"]["abs_deviation"] == 0.0
    assert checks["density"]["status"] == "RECORDED"
    assert checks["density"]["criterion"] is None
    assert any(e["role"] == "teacher_validation_target" for e in report["evidence"])


def test_stage11_rejects_wrong_or_mutated_teacher_reference(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_physical_validation_report

    target = _target(tmp_path)
    student_frames = tmp_path / "student.extxyz"
    write(str(student_frames), read(target["trajectory"]["path"], index=":"), format="extxyz")
    with pytest.raises(ValueError, match="sha256 does not match"):
        _exec_build_physical_validation_report({"parameters": {
            "validation_profile": str(_profile(tmp_path)),
            "frames_path": str(student_frames),
            "report_path": str(tmp_path / "bad.json"),
            "teacher_validation_target": str(tmp_path / "target.json"),
            "teacher_validation_target_sha256": "b" * 64,
        }})
    payload = json.loads((tmp_path / "target.json").read_text())
    payload["md_protocol"]["n_steps"] = 999
    (tmp_path / "target.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity mismatch"):
        _exec_build_physical_validation_report({"parameters": {
            "validation_profile": str(_profile(tmp_path)),
            "frames_path": str(student_frames),
            "report_path": str(tmp_path / "bad2.json"),
            "teacher_validation_target": str(tmp_path / "target.json"),
        }})


def test_stage11_student_cannot_redefine_frozen_observable_params(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_physical_validation_report

    target = _target(tmp_path)
    student_frames = tmp_path / "student.extxyz"
    write(str(student_frames), read(target["trajectory"]["path"], index=":"), format="extxyz")
    with pytest.raises(ValueError, match="may not redefine"):
        _exec_build_physical_validation_report({"parameters": {
            "validation_profile": str(_profile(tmp_path)),
            "frames_path": str(student_frames),
            "report_path": str(tmp_path / "report.json"),
            "teacher_validation_target": str(tmp_path / "target.json"),
            "r_max": 6.0,
        }})


def test_diffusivity_and_adf_require_explicit_parameters(tmp_path):
    frames = _frames(6)
    with pytest.raises(ValueError, match="fit_start_frame"):
        tpv.evaluate_observable({"name": "D", "kind": "diffusivity", "params": {"species": "Si", "timestep_fs": 1.0}}, frames)
    adf = tpv.evaluate_observable({"name": "angle", "kind": "adf", "params": {
        "center_species": "Si", "neighbor_species": "O", "r_cut_A": 3.0,
    }}, frames)
    assert adf["status"] == tpv.COMPUTED
    assert adf["details"]["n_triplets"] >= 0
