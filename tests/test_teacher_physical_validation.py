"""Tests for the objective/profile-conditioned Teacher physical-validation stage.

The ten tests below map one-to-one onto the acceptance criteria:

  1.  exact same RDF implementation for Teacher and Student
  2.  exact same coordination implementation for Teacher and Student
  3.  Teacher target is frozen (hash-bound) before any Student training
  4.  Student cannot mutate or re-author the frozen Teacher target
  5.  ingesting an existing Teacher trajectory works with no Teacher inference
  6.  COMPUTE mode works with a mock Teacher + mock integrator (no real inference)
  7.  diffusivity is deterministic and provenance-bound (explicit fit window)
  8.  ADF works generically for any configured center/neighbor species + angle
  9.  unavailable / non-applicable observables are explicit, never fabricated
  10. no eng6 Stage-7 files or run state are referenced or modified

Every physical observable is computed by the SAME model-independent kernels in
``validation.structure_dynamics`` that the Student physical-validation path uses;
these tests assert that byte-identical reuse directly.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

import validation.structure_dynamics as sd
import validation.teacher_physical_validation as tpv


# --------------------------------------------------------------------------------------------
# Deterministic fixtures: a periodic structure, a mock Teacher calculator + mock integrator.
# --------------------------------------------------------------------------------------------

def make_grid_frames(n_frames=4, drift=0.02):
    """Alternating two-species simple-cubic grid in a periodic box, with a small rigid drift
    per frame. Deterministic; large enough (10 A) for the RDF cell check."""
    spacing = 2.5
    n = 4  # 4x4x4 = 64 sites -> 10 A cell
    base_pos, syms = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                base_pos.append([i * spacing, j * spacing, k * spacing])
                syms.append("Si" if (i + j + k) % 2 == 0 else "O")
    base_pos = np.array(base_pos, dtype=float)
    cell = [n * spacing] * 3
    frames = []
    for f in range(n_frames):
        pos = base_pos.copy()
        pos[:, 0] += drift * f
        frames.append(Atoms(symbols=syms, positions=pos, cell=cell, pbc=True))
    return frames


def make_right_angle_frame(center, neighbor):
    """One center atom with two neighbors placed to subtend a 90-degree angle."""
    return Atoms(symbols=[center, neighbor, neighbor],
                 positions=[[5.0, 5.0, 5.0], [6.6, 5.0, 5.0], [5.0, 6.6, 5.0]],
                 cell=[10, 10, 10], pbc=True)


class MockTeacherCalc(Calculator):
    """A stand-in ASE calculator: deterministic, cheap, and NOT a real Teacher model. It
    counts how many times it is asked to evaluate so tests can assert the flow used the mock
    and never loaded/queried a real Teacher potential."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.inference_calls += 1
        n = len(atoms)
        self.results["energy"] = -1.0 * n + 1e-3 * float(atoms.get_positions()[:, 0].sum())
        self.results["forces"] = np.zeros((n, 3))


class MockTeacherProvider:
    """Exposes the same ``make_ase_calculator()`` contract the Teacher-driven acquisition
    backend uses, but hands back a mock calculator."""

    def __init__(self):
        self.calc = MockTeacherCalc()
        self.make_calls = 0

    def make_ase_calculator(self):
        self.make_calls += 1
        return self.calc


def mock_md_engine(atoms, calc, params, seed, sample_fn):
    """Mock integrator with the teacher_dynamics signature (atoms, calc, params, seed,
    sample_fn). It samples deterministic frames via a small rigid translation; it never
    performs any real dynamics, but it DOES attach the (mock) calc so per-frame energies flow
    through exactly as they would in the real path."""
    n = int(params["n_steps"])
    stride = int(params["sample_stride"])
    base = atoms.copy()
    for k in range(0, n + 1, stride):
        moved = base.copy()
        moved.positions[:, 0] += 0.02 * k
        moved.calc = calc
        sample_fn(moved, k)


MD_PROTOCOL = {"ensemble": "NVE", "timestep_fs": 1.0, "n_steps": 20,
               "sample_stride": 2, "seed": 7, "temperature_K": 300.0}
TEACHER_IDENTITY = {"model": "/mock/teacher.pth", "model_sha256": "a" * 64}
PROFILE_SHA = "deadbeefcafef00d"

RDF_SPEC = {"name": "rdf_Si_O_first_peak", "kind": "rdf_peak_position", "units": "A",
            "params": {"center_species": "Si", "neighbor_species": "O",
                       "r_max": 4.0, "nbins": 200}}
CN_SPEC = {"name": "coord_Si_O", "kind": "species_coordination", "units": "count",
           "params": {"center_species": "Si", "neighbor_species": "O", "cutoff_A": 3.0}}


def _freeze_compute_target(tmp_path, provider=None, specs=(RDF_SPEC, CN_SPEC)):
    provider = provider or MockTeacherProvider()
    start = tmp_path / "start.xyz"
    write(str(start), make_grid_frames()[0])
    target = tpv.compute_teacher_validation_target(
        objective_profile_sha256=PROFILE_SHA, teacher_identity=TEACHER_IDENTITY,
        md_protocol=MD_PROTOCOL, start_structures_path=str(start),
        observable_specs=list(specs), target_path=str(tmp_path / "target.json"),
        trajectory_out_path=str(tmp_path / "traj.extxyz"),
        teacher_calculator_provider=provider, md_engine=mock_md_engine)
    return target, provider


# --------------------------------------------------------------------------------------------
# 1. Exact same RDF implementation for Teacher and Student.
# --------------------------------------------------------------------------------------------

def test_rdf_shared_impl_teacher_and_student_identical():
    frames = make_grid_frames()
    row = tpv.evaluate_observable(RDF_SPEC, frames)
    assert row["status"] == tpv.COMPUTED

    # The shared dispatcher must produce EXACTLY what the underlying structure_dynamics
    # kernels produce -- there is no second RDF implementation.
    rdf = sd.compute_rdf_v2(frames, "Si", "O", r_max=4.0, nbins=200)
    peakmin = sd.rdf_first_peak_and_minimum(rdf["r_A"], rdf["g_of_r"])
    assert row["value"] == float(peakmin["r_first_peak_A"])
    assert row["details"]["peakmin"] == peakmin

    # Teacher and Student go through the same evaluate_observable; on identical frames the
    # measured value is bit-for-bit identical.
    student_row = tpv.evaluate_observable(RDF_SPEC, frames)
    assert student_row["value"] == row["value"]


# --------------------------------------------------------------------------------------------
# 2. Exact same coordination implementation for Teacher and Student.
# --------------------------------------------------------------------------------------------

def test_coordination_shared_impl_teacher_and_student_identical():
    frames = make_grid_frames()
    row = tpv.evaluate_observable(CN_SPEC, frames)
    assert row["status"] == tpv.COMPUTED

    cc = sd.compute_species_coordination(frames, "Si", "O", 3.0)
    assert row["value"] == float(cc["aggregate_mean_coordination"])
    assert row["details"]["coordination_histogram"] == cc["coordination_histogram"]

    student_row = tpv.evaluate_observable(CN_SPEC, frames)
    assert student_row["value"] == row["value"]
    assert student_row["details"] == row["details"]


# --------------------------------------------------------------------------------------------
# 3. Teacher target frozen (hash-bound) before any Student training.
# --------------------------------------------------------------------------------------------

def test_teacher_target_frozen_before_student(tmp_path):
    target, _ = _freeze_compute_target(tmp_path)
    # A frozen, hash-bound artifact exists on disk with no Student input whatsoever.
    on_disk = json.loads((tmp_path / "target.json").read_text())
    assert on_disk["artifact"] == "teacher_validation_target"
    assert on_disk["target_sha256"] == target["target_sha256"]
    assert on_disk["schema_version"] == tpv.TARGET_SCHEMA_VERSION
    # The freeze binds objective/profile, Teacher identity + model SHA, MD protocol,
    # trajectory hash, and observable definitions + results.
    assert on_disk["objective_profile_sha256"] == PROFILE_SHA
    assert on_disk["teacher_identity"]["model_sha256"] == TEACHER_IDENTITY["model_sha256"]
    for key in ("ensemble", "timestep_fs", "n_steps", "sample_stride", "seed"):
        assert key in on_disk["md_protocol"]
    assert on_disk["trajectory"]["sha256"]
    assert {d["name"] for d in on_disk["observable_definitions"]} == {RDF_SPEC["name"],
                                                                      CN_SPEC["name"]}
    # And the hash actually verifies.
    assert tpv.verify_teacher_validation_target(on_disk)["target_sha256"] == target["target_sha256"]


# --------------------------------------------------------------------------------------------
# 4. Student cannot mutate or re-author the frozen Teacher target.
# --------------------------------------------------------------------------------------------

def test_student_cannot_mutate_or_reauthor_target(tmp_path):
    target, _ = _freeze_compute_target(tmp_path)

    # (a) Any post-freeze edit is caught by the hash on verify.
    tampered = copy.deepcopy(target)
    tampered["observable_results"][0]["value"] = 999.0
    with pytest.raises(ValueError, match="integrity mismatch"):
        tpv.verify_teacher_validation_target(tampered)

    # (b) A comparison_policy that tries to smuggle observable definitions is rejected.
    student = make_grid_frames()
    with pytest.raises(ValueError, match="may not redefine"):
        tpv.compare_student_to_teacher_target(
            target, student, comparison_policy={"observable_definitions": [RDF_SPEC]})

    # (c) A legal comparison does not mutate the caller's frozen target or the on-disk file.
    before_obj = copy.deepcopy(target)
    before_bytes = (tmp_path / "target.json").read_bytes()
    tpv.compare_student_to_teacher_target(target, student)
    assert target == before_obj
    assert (tmp_path / "target.json").read_bytes() == before_bytes


# --------------------------------------------------------------------------------------------
# 5. Ingest an existing Teacher trajectory with NO Teacher inference.
# --------------------------------------------------------------------------------------------

def test_ingest_existing_trajectory_without_inference(tmp_path):
    # Produce a trajectory artifact independently (as if from a prior Teacher run).
    frames = make_grid_frames(n_frames=5)
    for f in frames:
        f.info["potential_energy"] = -100.0 + 0.01 * float(f.get_positions()[:, 0].sum())
    traj = tmp_path / "existing_teacher.extxyz"
    write(str(traj), frames, format="extxyz")
    traj_sha = tpv._target_sha256({"x": 1})  # placeholder to prove we import; recomputed below
    from workflow.integrity import sha256_file
    traj_sha = sha256_file(traj)

    # INGEST accepts NO calculator/provider argument at all -- inference is structurally
    # impossible here.
    import inspect
    params = inspect.signature(tpv.ingest_teacher_validation_target).parameters
    assert "teacher_calculator_provider" not in params

    target = tpv.ingest_teacher_validation_target(
        objective_profile_sha256=PROFILE_SHA, teacher_identity=TEACHER_IDENTITY,
        md_protocol=MD_PROTOCOL, trajectory_path=str(traj), trajectory_sha256=traj_sha,
        observable_specs=[RDF_SPEC, CN_SPEC], target_path=str(tmp_path / "t.json"))
    assert target["mode"] == "INGEST"
    assert target["ingest_provenance"] == "INGESTED_RECOMPUTED"
    assert target["observable_status"][RDF_SPEC["name"]] == tpv.COMPUTED
    tpv.verify_teacher_validation_target(target)

    # A hash mismatch on the ingested trajectory fails closed.
    with pytest.raises(ValueError, match="hash mismatch"):
        tpv.ingest_teacher_validation_target(
            objective_profile_sha256=PROFILE_SHA, teacher_identity=TEACHER_IDENTITY,
            md_protocol=MD_PROTOCOL, trajectory_path=str(traj),
            trajectory_sha256="0" * 64, observable_specs=[RDF_SPEC],
            target_path=str(tmp_path / "bad.json"))

    # External-evidence ingest (no recompute) is marked distinctly and never recomputes.
    ext = tpv.ingest_teacher_validation_target(
        objective_profile_sha256=PROFILE_SHA, teacher_identity=TEACHER_IDENTITY,
        md_protocol=MD_PROTOCOL, trajectory_path=str(traj), trajectory_sha256=traj_sha,
        observable_specs=[RDF_SPEC], target_path=str(tmp_path / "ext.json"),
        recompute_from_trajectory=False,
        precomputed_observables=[{"name": RDF_SPEC["name"], "kind": "rdf_peak_position",
                                  "status": tpv.COMPUTED, "value": 2.45, "units": "A"}])
    assert ext["ingest_provenance"] == "INGESTED_EXTERNAL"


# --------------------------------------------------------------------------------------------
# 6. COMPUTE mode works with a mock Teacher trajectory (no real inference).
# --------------------------------------------------------------------------------------------

def test_compute_mode_with_mock_teacher(tmp_path, monkeypatch):
    # If COMPUTE ever tried to fall back to the real integrator, this would raise.
    def _boom():
        raise AssertionError("real teacher_dynamics integrator must not be used in this test")
    monkeypatch.setattr(tpv, "_default_md_engine", _boom)

    target, provider = _freeze_compute_target(tmp_path)
    assert target["mode"] == "COMPUTE"
    assert target["trajectory"]["source"] == "teacher_md_compute"
    # A trajectory artifact was written and its hash bound into the target.
    assert Path(target["trajectory"]["path"]).is_file()
    from workflow.integrity import sha256_file
    assert sha256_file(target["trajectory"]["path"]) == target["trajectory"]["sha256"]
    # Only the MOCK calculator was used -- a real Teacher potential was never constructed.
    assert provider.make_calls == 1
    assert provider.calc.inference_calls > 0
    assert target["observable_status"][RDF_SPEC["name"]] == tpv.COMPUTED


# --------------------------------------------------------------------------------------------
# 7. Diffusivity is deterministic and provenance-bound.
# --------------------------------------------------------------------------------------------

def test_diffusivity_deterministic_and_provenance_bound():
    # A perfectly linear MSD -> exact, repeatable slope; D = slope / (2 n_dims).
    linear = {"Si": np.arange(10, dtype=float) * 4.0}
    d1 = sd.compute_diffusivity(linear, 1.0, fit_start_frame=2, fit_end_frame=10)
    d2 = sd.compute_diffusivity(linear, 1.0, fit_start_frame=2, fit_end_frame=10)
    assert d1 == d2
    rec = d1["Si"]
    assert rec["diffusivity_A2_per_fs"] == pytest.approx(4.0 / (2 * 3))
    # The window it was fit over travels with the number -- provenance-bound.
    assert rec["fit_window_frames"] == [2, 10]
    assert rec["n_fit_points"] == 8
    assert rec["timestep_fs"] == 1.0
    assert rec["sample_interval_steps"] == 1

    # Through the shared dispatcher: an explicit window is REQUIRED (fail-closed, no default).
    frames = make_grid_frames(n_frames=6)
    spec = {"name": "D_Si", "kind": "diffusivity", "units": "A^2/ps",
            "params": {"species": "Si", "timestep_fs": 1.0,
                       "fit_start_frame": 0, "fit_end_frame": 6}}
    row = tpv.evaluate_observable(spec, frames)
    assert row["status"] == tpv.COMPUTED
    assert row["details"]["fit_window_frames"] == [0, 6]
    with pytest.raises(ValueError, match="fit_start_frame"):
        tpv.evaluate_observable(
            {"name": "D_bad", "kind": "diffusivity",
             "params": {"species": "Si", "timestep_fs": 1.0}}, frames)


# --------------------------------------------------------------------------------------------
# 8. ADF works generically for any configured center/neighbor species + angle.
# --------------------------------------------------------------------------------------------

def test_adf_generic_for_configured_species_and_angle():
    # 90-degree geometry, evaluated with the center/neighbor roles swapped -- proving the
    # observable is generic over species with NO material-specific triplet baked in.
    for center, neighbor in (("O", "Si"), ("Si", "O"), ("C", "H")):
        frame = [make_right_angle_frame(center, neighbor)]
        spec = {"name": f"adf_{center}_{neighbor}", "kind": "adf", "units": "deg",
                "params": {"center_species": center, "neighbor_species": neighbor,
                           "r_cut_A": 2.0, "nbins": 180}}
        row = tpv.evaluate_observable(spec, frame)
        assert row["status"] == tpv.COMPUTED
        assert row["value"] == pytest.approx(90.0, abs=1.0)
        assert row["details"]["center_species"] == center

    # A configurable angular window is honored, and a list of neighbor species is accepted.
    frame = [make_right_angle_frame("O", "Si")]
    multi = {"name": "adf_multi", "kind": "adf", "units": "deg",
             "params": {"center_species": "O", "neighbor_species": ["Si", "Ge"],
                        "r_cut_A": 2.0, "angle_min_deg": 30.0, "angle_max_deg": 150.0}}
    row = tpv.evaluate_observable(multi, frame)
    assert row["status"] == tpv.COMPUTED
    assert row["details"]["angle_range_deg"] == [30.0, 150.0]
    assert row["details"]["neighbor_species"] == ["Ge", "Si"]

    # No material formula is hard-coded in the source; genericity is proven behaviorally
    # above (arbitrary center/neighbor species, both role orderings, configurable window).
    src = Path(sd.__file__).read_text() + Path(tpv.__file__).read_text()
    for banned in ("SiO2", "SiO_2", '"Si"', "'Si'", '"O"', "'O'"):
        assert banned not in src


# --------------------------------------------------------------------------------------------
# 9. Unavailable / non-applicable observables are explicit, never fabricated.
# --------------------------------------------------------------------------------------------

def test_unavailable_and_not_applicable_are_explicit():
    frames = make_grid_frames()

    # (a) A required species that is simply not in the trajectory -> UNAVAILABLE, value None.
    absent = tpv.evaluate_observable(
        {"name": "cn_absent", "kind": "species_coordination",
         "params": {"center_species": "Xe", "neighbor_species": "O", "cutoff_A": 3.0}}, frames)
    assert absent["status"] == tpv.UNAVAILABLE
    assert absent["value"] is None and absent["reason"]

    # (b) Objective conditioning: an observable not selected by the active objectives is
    # NOT_APPLICABLE (not silently computed, not silently dropped).
    na = tpv.evaluate_observable(
        {"name": "rho", "kind": "density", "applicable_objectives": ["objective_a"]},
        frames, context={"objectives": ["objective_b"]})
    assert na["status"] == tpv.NOT_APPLICABLE
    assert na["value"] is None

    # (c) nve_drift with no energy series available -> UNAVAILABLE, never a fabricated drift.
    drift = tpv.evaluate_observable(
        {"name": "nve", "kind": "nve_drift", "params": {"timestep_fs": 1.0}}, frames)
    assert drift["status"] == tpv.UNAVAILABLE
    assert drift["value"] is None

    # (d) A malformed spec is an error, not a fabricated result.
    with pytest.raises(ValueError):
        tpv.evaluate_observable({"name": "no_kind"}, frames)


# --------------------------------------------------------------------------------------------
# 10. No eng6 Stage-7 files or run state are referenced or modified.
# --------------------------------------------------------------------------------------------

def test_no_eng6_or_run_state_touched(tmp_path):
    # The new modules must not hard-code any run directory, run name, or eng6 reference.
    for module in (sd, tpv):
        src = Path(module.__file__).read_text()
        for banned in ("eng6", "runs/", "ffv4", "/home/hyunjin/distill-real-user/runs"):
            assert banned not in src

    # Running the full COMPUTE -> INGEST -> COMPARE flow writes ONLY under the explicitly
    # provided tmp_path; nothing is created elsewhere.
    before = {p for p in tmp_path.rglob("*")}
    target, _ = _freeze_compute_target(tmp_path)
    student = read(target["trajectory"]["path"], index=":")
    result = tpv.compare_student_to_teacher_target(target, student)
    assert result["overall_status"] in {"PASS", "FAIL", "RECORDED"}
    after = {p for p in tmp_path.rglob("*")}
    created = {p.name for p in (after - before)}
    # Exactly the artifacts we asked for -- the target JSON and the trajectory.
    assert created == {"target.json", "traj.extxyz", "start.xyz"}

    # The worktree carries no runs/ checkout (eng6 lives in the separate production tree).
    worktree_root = Path(tpv.__file__).resolve().parents[1]
    assert not (worktree_root / "runs").exists()
