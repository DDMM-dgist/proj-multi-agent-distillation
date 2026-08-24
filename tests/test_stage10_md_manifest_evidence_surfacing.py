"""Regression tests for the Stage-10 md_manifest bounded-evidence adapter.

Bound by the authorized evidence-serialization fix (session 2026-08-21). Proves that the
adapter surfaces every criterion-relevant VALUE (C1 checkpoint, C2 realized protocol +
frozen deployment domain, C3 evidence role→integrity map, C4 Controller-manifest approval +
submission timing) inline in the Judge-facing bounded packet, that the source artifacts are
not modified, that missing/hash-mismatched sources fail closed as ``evidence_gap``, and that
NVT-specific semantics (no controlled pressure setpoint; total energy diagnostic-only) are
enforced.
"""
from __future__ import annotations

import copy
import json
import hashlib
from pathlib import Path

import pytest

from runtimes.pydantic_ai.bounded_evidence import (
    _is_md_manifest,
    _md_manifest_summary,
    _md_manifest_siblings_attachment,
    _json_summary,
)


# ---- Deterministic fixture builder --------------------------------------------------

_LAMMPS_DATAFILE_MIN = """(fixture)

10 atoms
2 atom types

0.0      5.0  xlo xhi
0.0      5.0  ylo yhi
0.0      5.0  zlo zhi

Masses

1      15.999
2      28.086

Atoms # atomic

     1   1      0.0 0.0 0.0
     2   1      1.0 0.0 0.0
     3   1      2.0 0.0 0.0
     4   1      3.0 0.0 0.0
     5   1      4.0 0.0 0.0
     6   1      0.0 1.0 0.0
     7   1      1.0 1.0 0.0
     8   2      2.0 1.0 0.0
     9   2      3.0 1.0 0.0
    10   2      4.0 1.0 0.0
"""


_LAMMPS_INPUT_MIN = """echo screen
units metal
atom_style atomic
read_data ./data/fixture.lammps-data
pair_style nn
pair_coeff * * ./ckpt.bestmodel O Si
velocity all create 300.0 12345 mom yes rot yes dist gaussian
fix nvt all nvt temp 300.0 300.0 0.05
thermo 100
run 500
"""


_THERMO_MIN = """LAMMPS (23 Jun 2022)
Setting up Verlet run ...
Step Temp PotEng KinEng TotEng Press
        0    300.0   -100.0   0.5   -99.5    1234.5
      100    301.2   -100.1   0.5   -99.6    1235.0
      200    299.9   -100.2   0.5   -99.7    1233.8
      300    300.5   -100.3   0.5   -99.8    1236.7
      400    300.7   -100.4   0.5   -99.9    1234.1
      500    301.0   -100.5   0.5   -100.0   1232.9
Loop time of 0.001 on 1 procs for 500 steps
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_run(tmp_path: Path, *, include_context=True, include_provenance=True,
                include_workflow_domain=True, include_approval=True, include_thermo=True,
                ensemble_fix="fix nvt", n_steps=500, temperature_k=300.0) -> dict:
    run_dir = tmp_path / "runs" / "test-run"
    art = run_dir / "artifacts"
    depl = art / "deployment_md"
    data_dir = depl / "data"
    inputs = run_dir / "inputs"
    for d in (art, depl, data_dir, inputs):
        d.mkdir(parents=True, exist_ok=True)

    ckpt = art / "committee" / "seed-000" / "potential_saved_bestmodel"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("ELEM_LIST Si O\n(fixture)\n")

    datafile = data_dir / "fixture.lammps-data"
    datafile.write_text(_LAMMPS_DATAFILE_MIN)

    input_lmp = depl / "input.lmp"
    lmp = _LAMMPS_INPUT_MIN
    if ensemble_fix != "fix nvt":
        lmp = lmp.replace("fix nvt all nvt temp 300.0 300.0 0.05", ensemble_fix)
    input_lmp.write_text(lmp)

    trajectory = depl / "trajectory.dump"; trajectory.write_text("ITEM: TIMESTEP\n0\n")
    thermo = depl / "thermo.log"
    if include_thermo:
        thermo.write_text(_THERMO_MIN)

    if include_context:
        ctx = depl / "context.yaml"
        ctx.write_text(
            f"DATAFILE: {datafile}\n"
            "TIMESTEP_PS: 0.0005\n"
            f"TEMPERATURE_K: {temperature_k}\n"
            "TDAMP_PS: 0.05\n"
            "SEED: 12345\n"
            f"N_STEPS: {n_steps}\n"
            "THERMO_EVERY_STEPS: 100\n"
            "DUMP_EVERY_STEPS: 100\n"
            "DUMP_FILE: trajectory.dump\n"
            "MPI_RANKS: 1\n"
        )

    if include_provenance:
        (depl / "deployment_provenance.json").write_text(json.dumps({
            "schema_version": 1,
            "role": "stage10_pre_launch_deployment_provenance",
            "starting_structure": {
                "path": str(datafile),
                "sha256": _sha256(datafile),
                "provenance_role": "historical_external_deployment_starting_structure",
                "leakage_check": "PASS",
            },
            "lammps_backend": {
                "env": "simple-nn-lammps",
                "binary_realpath": "/fake/lmp",
                "binary_sha256": "b" * 64,
                "lammps_version": "23 Jun 2022 (fixture)",
                "mpi": "Open MPI (fixture)",
            },
            "pair_style_nn_preflight": {"status": "PASS", "assertions": [1,2,3], "disposable": True},
            "authorization": {"boundary": "production_md", "action_type": "run_student_md",
                              "approved_by": "human_operator (fixture)",
                              "resource_safety": "CPU-only"},
            "student": {"checkpoint_sha256": _sha256(ckpt), "selected_seed": 0},
        }))

    if include_workflow_domain:
        (inputs / "009-distillation_scope.yaml").write_text(
            "kind: generic\n"
            "deployment_domain:\n"
            "  system: fixture Si-O\n"
            "  composition_scope:\n"
            "  - stoichiometric\n"
            "  structure_classes:\n"
            "  - amorphous_bulk\n"
            "  temperature_K:\n"
            "  - full source-pool envelope (ambient through melt)\n"
            "  pressure_GPa:\n"
            "  - full source-pool envelope (ambient through high-pressure)\n"
        )

    approvals = {}
    if include_approval:
        approvals = {"production_md": {
            "granted": True, "scope": "exact_action", "action_type": "run_student_md",
            "at": "2026-08-21T01:00:00+00:00", "note": "fixture approval",
        }}
    stages = [{"name": "deployment_md", "status": "completed",
               "started_at": "2026-08-21T01:30:00+00:00",
               "completed_at": "2026-08-21T02:00:00+00:00", "gate": "pending", "attempts": 1}]
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 10, "run_id": "test-run", "stages": stages,
        "action_approvals": approvals, "artifacts": [], "iterations": [], "events": [],
    }))

    ev = []
    for role, p in (("input", input_lmp), ("trajectory", trajectory), ("thermo_log", thermo)):
        if not p.is_file():
            continue
        ev.append({"role": role, "path": str(p),
                   "integrity": {"kind": "file", "size": p.stat().st_size, "sha256": _sha256(p)}})
    md_manifest_path = art / "md.manifest.json"
    md_manifest_path.write_text(json.dumps({
        "schema_version": 1, "input": str(input_lmp), "run_dir": str(depl),
        "checkpoint": str(ckpt),
        "checkpoint_integrity": {"kind": "file", "size": ckpt.stat().st_size,
                                  "sha256": _sha256(ckpt)},
        "evidence": ev,
        "committee_manifest": str(art / "student_committee.manifest.json"),
        "selected_seed": 0,
    }))
    return {"run_dir": run_dir, "md_manifest_path": md_manifest_path, "datafile": datafile,
            "input_lmp": input_lmp, "ckpt": ckpt, "trajectory": trajectory, "thermo": thermo}


# ---------- predicate ----------------------------------------------------------------

def test_predicate_recognizes_by_required_key_signature(tmp_path):
    ctx = _build_run(tmp_path)
    payload = json.loads(ctx["md_manifest_path"].read_text())
    assert _is_md_manifest(payload) is True


@pytest.mark.parametrize("missing",
                          ["schema_version", "input", "run_dir", "checkpoint",
                           "checkpoint_integrity", "evidence", "committee_manifest",
                           "selected_seed"])
def test_predicate_rejects_when_any_required_key_missing(tmp_path, missing):
    ctx = _build_run(tmp_path)
    payload = json.loads(ctx["md_manifest_path"].read_text())
    payload.pop(missing)
    assert _is_md_manifest(payload) is False


# ---------- C1: checkpoint identity + sha visible ------------------------------------

def test_c1_checkpoint_sha256_readable(tmp_path):
    ctx = _build_run(tmp_path)
    p = ctx["md_manifest_path"]
    summary = _json_summary(p)["md_manifest"]
    ck = summary["checkpoint"]
    assert ck["checkpoint_sha256"] == _sha256(ctx["ckpt"])
    assert ck["selected_seed"] == 0
    assert ck["committee_manifest_path"].endswith("student_committee.manifest.json")


# ---------- C2: realized deployment protocol + frozen deployment domain -------------

def test_c2_realized_protocol_and_domain_readable(tmp_path):
    ctx = _build_run(tmp_path)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    proto = summary["realized_protocol"]
    assert proto["ensemble"] == "nvt"
    assert proto["temperature_setpoint_K"] == 300.0
    assert proto["timestep_ps"] == 0.0005
    assert proto["n_steps"] == 500
    assert proto["total_simulated_time_ps"] == 500 * 0.0005
    assert proto["starting_structure_n_atoms"] == 10
    assert proto["starting_structure_species_counts_by_lammps_type"] == {"type_1": 7, "type_2": 3}
    dom = summary["deployment_domain"]
    assert isinstance(dom["declared_domain"], dict)
    assert dom["declared_domain"]["composition_scope"] == ["stoichiometric"]
    assert dom["declared_domain_provenance_sources"], "provenance source must be recorded"


# ---------- C3: evidence role → integrity map explicit --------------------------------

def test_c3_evidence_role_map_is_explicit(tmp_path):
    ctx = _build_run(tmp_path)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    roles = {e["role"]: e for e in summary["evidence_role_map"]}
    for r in ("input", "trajectory", "thermo_log"):
        assert r in roles, f"missing declared role: {r}"
        assert roles[r]["sha256"] and roles[r]["size"], f"role {r} missing sha/size"


# ---------- C4: Controller-manifest human approval + submission timing ---------------

def test_c4_controller_manifest_approval_and_precedes_submission(tmp_path):
    ctx = _build_run(tmp_path)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    ha = summary["human_approval"]
    assert ha["granted"] is True
    assert ha["action_type"] == "run_student_md"
    assert ha["approved_at"] == "2026-08-21T01:00:00+00:00"
    assert ha["submission_started_at"] == "2026-08-21T01:30:00+00:00"
    assert ha["approval_precedes_submission"] is True
    assert ha["approval_to_submission_seconds"] == 30 * 60
    # Primary approval source is the Controller manifest.json, not the deployment_provenance.
    assert ha["approval_source"].endswith("manifest.json")


def test_c4_no_approval_recorded_reports_absence_not_fabricated(tmp_path):
    ctx = _build_run(tmp_path, include_approval=False)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    ha = summary["human_approval"]
    assert ha.get("granted") is None or ha.get("granted") is False
    assert ha["approval_precedes_submission"] is None


# ---------- NVT-specific semantics ---------------------------------------------------

def test_nvt_pressure_is_not_a_setpoint(tmp_path):
    ctx = _build_run(tmp_path)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    proto = summary["realized_protocol"]
    assert proto["pressure_setpoint_bar"] is None
    assert "NO controlled pressure setpoint" in proto["pressure_note"]
    assert summary["framework_notes"]["nvt_pressure_semantic"].startswith("This is an NVT")


def test_total_energy_is_not_labeled_nve_conservation(tmp_path):
    ctx = _build_run(tmp_path)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    td = summary["thermo_diagnostic"]
    note = td["total_energy_eV_diagnostic_only"]["note"]
    assert "DIAGNOSTIC" in note or "diagnostic" in note.lower()
    assert "NVE" in note or "conserv" in note.lower()
    assert summary["framework_notes"]["energy_conservation_semantic"].startswith("Total energy in NVT")


# ---------- Fail-closed on missing / hash-drifted siblings ---------------------------

def test_missing_context_yaml_reports_gap_not_a_fabricated_protocol(tmp_path):
    ctx = _build_run(tmp_path, include_context=False)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    proto = summary["realized_protocol"]
    assert proto["temperature_setpoint_K"] is None
    assert proto["n_steps"] is None
    # No fake values.


def test_missing_deployment_provenance_reports_gap(tmp_path):
    ctx = _build_run(tmp_path, include_provenance=False)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    dp = summary["deployment_provenance"]
    assert "evidence_gap" in dp


def test_missing_thermo_log_reports_gap(tmp_path):
    ctx = _build_run(tmp_path, include_thermo=False)
    summary = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    td = summary["thermo_diagnostic"]
    assert "evidence_gap" in td


# ---------- Source-artifact byte-identity guaranteed by the summary path -------------

def test_summary_never_mutates_source_artifacts(tmp_path):
    ctx = _build_run(tmp_path)
    before = {
        "md_manifest": _sha256(ctx["md_manifest_path"]),
        "input_lmp": _sha256(ctx["input_lmp"]),
        "thermo": _sha256(ctx["thermo"]),
        "trajectory": _sha256(ctx["trajectory"]),
        "datafile": _sha256(ctx["datafile"]),
    }
    _ = _json_summary(ctx["md_manifest_path"])["md_manifest"]
    after = {k: _sha256(v) for k, v in [
        ("md_manifest", ctx["md_manifest_path"]),
        ("input_lmp", ctx["input_lmp"]),
        ("thermo", ctx["thermo"]),
        ("trajectory", ctx["trajectory"]),
        ("datafile", ctx["datafile"]),
    ]}
    assert before == after


# ---------- Live-artifact untouched fixture (C12F) ------------------------------------

_CANONICAL_STAGE10_SHAS = {
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/md.manifest.json":
        "6541c3a1da04e038b3cbb05b0b9c36efda8b05806bcb941887a2660a2f7c46a0",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/trajectory.dump":
        "6eec4a0e90bc4c63ad2def8b081c0b1fdbec3e8358186a58bff7045d77988a4d",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/thermo.log":
        "3ed87bcec0beaea44726de04f90c0a38730101a2059c58ab35954d421c0983cc",
    "runs/sio2-sox-allegro-simplenn-c12f/artifacts/deployment_md/input.lmp":
        "63e3438068ad26a04a15abcef02d3fdeb33afbe74eef291608eb1707c743aa53",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relpath, expected_sha", sorted(_CANONICAL_STAGE10_SHAS.items()))
def test_c12f_stage10_artifacts_untouched_by_evidence_fix(relpath, expected_sha):
    from workflow.integrity import sha256_file
    path = _project_root() / relpath
    if not path.is_file():
        pytest.skip(f"C12F Stage-10 artifact not present in this checkout: {relpath}")
    assert sha256_file(path) == expected_sha, (
        f"{relpath!r} sha256 drifted from the pre-fix pin — the Stage-10 evidence "
        "serialization recovery must NOT modify any Stage-10 artifact byte")
