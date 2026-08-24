"""UNIT 3 tests: Stage-10/11 deployment resolution + dedicated NVE energy-conservation segment.

Covers the deployment-path autonomy added on top of the existing Stage-10/11 MD subsystem:
  * validation.deployment_resolution.resolve_selected_checkpoint (governed, no static path)
  * validation.deployment_resolution.derive_nve_segment_protocol (frozen-protocol derivation)
  * validation.deployment_resolution.build_deployment_context (NVT production vs NVE segment)
  * runtimes.pydantic_ai.executors deployment producers + Stage-11 NVE-log consumption
  * workflow.steps.run_md nve_energy_log evidence auto-recording
  * runtimes.pydantic_ai.bounded_evidence physical_validation_report Judge surfacing
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.deployment_resolution import (
    resolve_selected_checkpoint,
    derive_nve_segment_protocol,
    build_deployment_context,
    load_shared_md_protocol,
)
from workflow.integrity import sha256_file


_FROZEN_SHARED_MD_PROTOCOL = {
    "temperature_K": 300.0,
    "pressure_GPa": 0.0,
    "timestep_fs": 0.5,
    "nvt_equilibration_ps": 20.0,
    "nvt_production_ps": 50.0,
    "sampling_interval_fs": 10.0,
}


def _committee(tmp_path, seeds=(1, 2, 3), losses=None):
    """Write per-seed checkpoints + a committee manifest with real sha256 integrity."""
    models = []
    for s in seeds:
        ckpt = tmp_path / f"seed-{s}" / "mock-model.json"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps({"seed": int(s)}) + "\n")
        meta = {"trainer_kind": "analytic_mock"}
        if losses is not None:
            meta["loss"] = losses[s]
        models.append({"kind": "mock", "seed": int(s), "path": str(ckpt),
                       "integrity": {"kind": "file", "size": ckpt.stat().st_size,
                                     "sha256": sha256_file(ckpt)},
                       "metadata": meta})
    manifest = tmp_path / "committee.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": models}, indent=2))
    return manifest


# ----------------------------------------------------------------- checkpoint resolution


def test_resolve_explicit_seed_derives_path_and_sha(tmp_path):
    manifest = _committee(tmp_path, seeds=(1, 2, 3))
    res = resolve_selected_checkpoint(manifest, selected_seed=2)
    assert res["selected_seed"] == 2
    assert res["checkpoint_path"].endswith("seed-2/mock-model.json")
    assert res["checkpoint_sha256"] == sha256_file(Path(res["checkpoint_path"]))
    assert res["cross_check_match"] is True
    # semantic guard: committee-manifest sha is not the checkpoint sha
    assert res["committee_manifest_sha256"] == sha256_file(manifest)
    assert res["committee_manifest_sha_is_not_the_checkpoint_sha"] is True


def test_resolve_requires_a_governed_selection(tmp_path):
    manifest = _committee(tmp_path, seeds=(1, 2))
    with pytest.raises(ValueError, match="governed decision"):
        resolve_selected_checkpoint(manifest)


def test_resolve_unknown_seed_fails_closed(tmp_path):
    manifest = _committee(tmp_path, seeds=(1, 2))
    with pytest.raises(ValueError, match="not present"):
        resolve_selected_checkpoint(manifest, selected_seed=99)


def test_resolve_select_by_min_validation_loss(tmp_path):
    manifest = _committee(tmp_path, seeds=(1, 2, 3), losses={1: 0.5, 2: 0.2, 3: 0.9})
    res = resolve_selected_checkpoint(manifest, select_by="min_validation_loss")
    assert res["selected_seed"] == 2
    assert "min_validation_loss" in res["selection_derivation"]


def test_resolve_select_by_loss_fails_closed_when_no_numeric_loss(tmp_path):
    manifest = _committee(tmp_path, seeds=(1, 2))  # mock committee: no numeric loss
    with pytest.raises(ValueError, match="numeric metadata.loss"):
        resolve_selected_checkpoint(manifest, select_by="min_validation_loss")


def test_resolve_detects_checkpoint_drift(tmp_path):
    manifest = _committee(tmp_path, seeds=(1, 2))
    payload = json.loads(manifest.read_text())
    # tamper the on-disk checkpoint after the manifest recorded its sha
    Path(payload["models"][0]["path"]).write_text('{"seed": 1, "tampered": true}\n')
    with pytest.raises(ValueError, match="does not match the sha256"):
        resolve_selected_checkpoint(manifest, selected_seed=1)


def test_resolve_rejects_both_selectors(tmp_path):
    manifest = _committee(tmp_path, seeds=(1, 2))
    with pytest.raises(ValueError, match="exactly one"):
        resolve_selected_checkpoint(manifest, selected_seed=1, select_by="min_validation_loss")


# ----------------------------------------------------------------- NVE protocol derivation


def test_nve_protocol_derived_from_frozen_shared_protocol():
    nve = derive_nve_segment_protocol(_FROZEN_SHARED_MD_PROTOCOL)
    # 0.5 fs = 0.0005 ps; 20 ps warmup -> 40000 steps; NVE window mirrors equilibration (20 ps).
    assert nve["timestep_ps"] == pytest.approx(0.0005)
    assert nve["warmup_steps"] == 40000
    assert nve["nve_steps"] == 40000
    # 10 fs sampling / 0.5 fs step = every 20 steps
    assert nve["thermo_every_steps"] == 20
    assert nve["n_expected_energy_samples"] == 40000 // 20 + 1
    assert nve["temperature_K"] == 300.0
    assert nve["ensemble_role"] == "nve_energy_conservation_segment"
    # the autonomous choices carry recorded rationale (not silently baked in)
    assert "nve_segment_ps" in nve["autonomous_choice_rationale"]
    assert "tdamp_ps" in nve["autonomous_choice_rationale"]


def test_nve_protocol_prefers_explicit_window():
    nve = derive_nve_segment_protocol(_FROZEN_SHARED_MD_PROTOCOL, nve_segment_ps=10.0)
    assert nve["nve_segment_ps"] == 10.0
    assert nve["nve_steps"] == 20000
    assert "frozen NVE window" in nve["autonomous_choice_rationale"]["nve_segment_ps"]


def test_nve_protocol_fails_closed_on_missing_field():
    bad = {k: v for k, v in _FROZEN_SHARED_MD_PROTOCOL.items() if k != "timestep_fs"}
    with pytest.raises(ValueError, match="timestep_fs"):
        derive_nve_segment_protocol(bad)


# ----------------------------------------------------------------- deployment context


def test_build_nvt_production_context():
    ctx = build_deployment_context(_FROZEN_SHARED_MD_PROTOCOL, "nvt", "/data/bulk.lammps-data",
                                   velocity_seed=12345)
    assert ctx["DATAFILE"] == "/data/bulk.lammps-data"
    assert ctx["TEMPERATURE_K"] == 300.0
    assert ctx["TIMESTEP_PS"] == pytest.approx(0.0005)
    assert ctx["N_STEPS"] == 100000  # 50 ps production / 0.0005 ps
    assert ctx["DUMP_FILE"] == "trajectory.dump"
    assert "ENERGY_LOG" not in ctx  # production context has no NVE energy log
    assert "_nve_protocol" not in ctx


def test_build_nve_segment_context_is_distinct():
    ctx = build_deployment_context(_FROZEN_SHARED_MD_PROTOCOL, "nve", "/data/bulk.lammps-data",
                                   velocity_seed=777, energy_log="nve_energy.csv")
    assert ctx["ENERGY_LOG"] == "nve_energy.csv"
    assert ctx["WARMUP_STEPS"] == 40000
    assert ctx["N_STEPS"] == 40000
    assert "DUMP_FILE" not in ctx  # NVE segment logs energy, not a production dump
    assert ctx["_nve_protocol"]["ensemble_role"] == "nve_energy_conservation_segment"


def test_build_context_rejects_unknown_ensemble():
    with pytest.raises(ValueError, match="unknown deployment ensemble"):
        build_deployment_context(_FROZEN_SHARED_MD_PROTOCOL, "npt", "/x", velocity_seed=1)


def test_load_shared_md_protocol_from_profile(tmp_path):
    import yaml
    prof = tmp_path / "validation_profile.yaml"
    prof.write_text(yaml.safe_dump({"kind": "project-validation",
                                     "shared_md_protocol": _FROZEN_SHARED_MD_PROTOCOL}))
    loaded = load_shared_md_protocol(prof)
    assert loaded["timestep_fs"] == 0.5


def test_load_shared_md_protocol_fails_closed_when_absent(tmp_path):
    import yaml
    prof = tmp_path / "vp.yaml"
    prof.write_text(yaml.safe_dump({"kind": "generic"}))
    with pytest.raises(ValueError, match="no shared_md_protocol"):
        load_shared_md_protocol(prof)


# ----------------------------------------------------------------- executor producers


def test_exec_resolve_deployment_checkpoint_writes_provenance(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_resolve_deployment_checkpoint
    manifest = _committee(tmp_path, seeds=(1, 2))
    datafile = tmp_path / "bulk.lammps-data"
    datafile.write_text("dummy\n")
    out = tmp_path / "deployment_provenance.json"
    res = _exec_resolve_deployment_checkpoint({"parameters": {
        "committee_manifest": str(manifest),
        "selected_seed": 1,
        "starting_structure": str(datafile),
        "out_path": str(out),
    }})
    prov = json.loads(out.read_text())
    assert prov["student"]["selected_seed"] == 1
    assert prov["starting_structure"]["sha256"] == sha256_file(datafile)
    assert prov["ensemble_role"] == "deployment_md"
    assert res["student"]["checkpoint_sha256"] == prov["student"]["checkpoint_sha256"]


def test_exec_build_deployment_context_nve(tmp_path):
    import yaml
    from runtimes.pydantic_ai.executors import _exec_build_deployment_context
    out = tmp_path / "context.yaml"
    res = _exec_build_deployment_context({"parameters": {
        "shared_md_protocol": _FROZEN_SHARED_MD_PROTOCOL,
        "ensemble": "nve",
        "starting_structure": "/data/bulk.lammps-data",
        "velocity_seed": 42,
        "energy_log": "nve_energy.csv",
        "out_path": str(out),
    }})
    written = yaml.safe_load(out.read_text())
    # the plain template context must NOT carry the nested _nve_protocol dict
    assert "_nve_protocol" not in written
    assert written["ENERGY_LOG"] == "nve_energy.csv"
    assert res["nve_protocol"]["nve_steps"] == 40000


# ----------------------------------------------------------- Stage-11 automatic NVE consumption


def _write_nve_energy_log(path, n=50, slope_eV=0.0):
    """A drift-controlled NVE energy CSV in the nve_drift.in.template format."""
    lines = ["step,temperature,potential_energy,kinetic_energy,total_energy"]
    for i in range(n):
        etot = -1000.0 + slope_eV * i
        lines.append(f"{i * 20},300.0,-1100.0,100.0,{etot}")
    path.write_text("\n".join(lines) + "\n")


def _validation_profile(tmp_path):
    import yaml
    prof = tmp_path / "validation_profile.yaml"
    prof.write_text(yaml.safe_dump({
        "kind": "physical_validation",
        "shared_md_protocol": _FROZEN_SHARED_MD_PROTOCOL,
        "checks": [{
            "name": "nve_drift", "category": "dynamics", "purpose": "deployment_stability",
            "reference_source": "other", "required": True,
            "threshold": {"operator": "max_abs", "threshold": 1.0, "unit": "meV/atom/ps"},
        }],
    }))
    return prof


def test_stage11_consumes_nve_manifest_energy_log(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_physical_validation_report
    from workflow.integrity import artifact_digest
    from ase import Atoms
    from ase.io import write

    frames_path = tmp_path / "frames.extxyz"
    write(str(frames_path), [Atoms("Si2O", positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]],
                                    cell=[16, 16, 16], pbc=True)], format="extxyz")

    energy_log = tmp_path / "run" / "nve_energy.csv"
    energy_log.parent.mkdir(parents=True, exist_ok=True)
    _write_nve_energy_log(energy_log, n=50, slope_eV=0.0)  # ~zero drift -> should pass

    # A dedicated-NVE-segment MD manifest carrying the energy log as first-class evidence.
    nve_manifest = tmp_path / "nve.manifest.json"
    nve_manifest.write_text(json.dumps({
        "schema_version": 1,
        "evidence": [{"role": "nve_energy_log", "path": str(energy_log),
                      "integrity": artifact_digest(energy_log)}],
    }))

    report_path = tmp_path / "physical_validation.json"
    res = _exec_build_physical_validation_report({"parameters": {
        "validation_profile": str(_validation_profile(tmp_path)),
        "frames_path": str(frames_path),
        "nve_md_manifest": str(nve_manifest),
        "report_path": str(report_path),
        "n_atoms": 3,
        "timestep_fs": 0.5,
    }})
    report = json.loads(report_path.read_text())
    nve_check = next(c for c in report["checks"] if c["observable"] == "nve_drift")
    assert nve_check["value"] == pytest.approx(0.0, abs=1e-9)
    assert nve_check["unit"] == "meV/atom/ps"
    # the resolved NVE energy log is surfaced as evidence on the report
    assert any(e.get("role") == "nve_energy_log" for e in res["report"]["evidence"])


def test_stage11_fails_closed_when_nve_manifest_lacks_energy_log(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_physical_validation_report
    from ase import Atoms
    from ase.io import write
    frames_path = tmp_path / "frames.extxyz"
    write(str(frames_path), [Atoms("Si2O", positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]],
                                    cell=[16, 16, 16], pbc=True)], format="extxyz")
    nve_manifest = tmp_path / "nve.manifest.json"
    nve_manifest.write_text(json.dumps({"schema_version": 1, "evidence": []}))
    with pytest.raises(ValueError, match="no evidence entry with role 'nve_energy_log'"):
        _exec_build_physical_validation_report({"parameters": {
            "validation_profile": str(_validation_profile(tmp_path)),
            "frames_path": str(frames_path),
            "nve_md_manifest": str(nve_manifest),
            "n_atoms": 3,
        }})


# ----------------------------------------------------------- Judge-facing surfacing


def test_physical_validation_report_judge_surfacing(tmp_path):
    from runtimes.pydantic_ai.bounded_evidence import _json_summary
    report = {
        "schema_version": 1, "profile": "physical_validation",
        "checks": [
            {"domain": "structure", "observable": "rdf_Si_O", "value": 1.6, "unit": "Angstrom",
             "criterion": None},
            {"domain": "dynamics", "observable": "nve_drift", "value": 0.3,
             "unit": "meV/atom/ps",
             "criterion": {"operator": "max_abs", "threshold": 1.0, "unit": "meV/atom/ps"},
             "details": {"n_samples": 50}},
        ],
        "evidence": [{"role": "nve_energy_log", "path": "/run/nve_energy.csv",
                      "integrity": {"sha256": "a" * 64}}],
    }
    path = tmp_path / "physical_validation.json"
    path.write_text(json.dumps(report))
    summary = _json_summary(path)
    pv = summary["physical_validation_report"]
    assert pv["nve_drift"]["value"] == 0.3
    assert pv["nve_drift"]["passed"] is True
    assert pv["nve_drift"]["energy_log_evidence"]["role"] == "nve_energy_log"
    rdf = next(o for o in pv["observables"] if o["observable"] == "rdf_Si_O")
    assert rdf["role"] == "descriptive"
    assert rdf["passed"] is None


def test_physical_validation_report_marks_failing_nve():
    from runtimes.pydantic_ai.bounded_evidence import _physical_validation_report_summary
    pv = _physical_validation_report_summary({
        "profile": "physical_validation",
        "checks": [{"domain": "dynamics", "observable": "nve_drift", "value": 5.0,
                    "unit": "meV/atom/ps",
                    "criterion": {"operator": "max_abs", "threshold": 1.0,
                                  "unit": "meV/atom/ps"}}],
        "evidence": [],
    })
    assert pv["nve_drift"]["passed"] is False
