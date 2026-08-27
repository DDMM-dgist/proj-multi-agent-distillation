import json
from pathlib import Path

import pytest

from validation.deployment_plan import (
    HUMAN_OPERATIONAL_INPUT_REQUIRED,
    READY,
    build_stage10_deployment_plan,
    validate_stage10_deployment_plan,
)
from workflow.integrity import sha256_file


def _committee(tmp_path):
    ckpt = tmp_path / "seed-1" / "model.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text('{"seed":1}\n')
    manifest = tmp_path / "committee.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": [{
        "kind": "mock", "seed": 1, "path": str(ckpt),
        "integrity": {"kind": "file", "size": ckpt.stat().st_size, "sha256": sha256_file(ckpt)},
        "metadata": {"loss": 0.1},
    }]}))
    return manifest


def _profile(tmp_path):
    p = tmp_path / "validation_profile.yaml"
    p.write_text(
        "shared_md_protocol:\n"
        "  temperature_K: 300.0\n"
        "  pressure_GPa: 0.0\n"
        "  timestep_fs: 0.5\n"
        "  nvt_equilibration_ps: 20.0\n"
        "  nvt_production_ps: 50.0\n"
        "  sampling_interval_fs: 10.0\n"
    )
    return p


def _start_contract(tmp_path):
    start = tmp_path / "start.lammps-data"
    start.write_text("data\n")
    c = tmp_path / "start.json"
    c.write_text(json.dumps({
        "deployment_point_id": "ambient_amorphous_SiO2_point",
        "selection_rule": "explicit frozen fixture for test",
        "path": str(start),
        "sha256": sha256_file(start),
    }))
    return c


def _md_config(tmp_path, status):
    p = tmp_path / "md_backend.yaml"
    p.write_text(
        "kind: lammps\n"
        "binary: lmp\n"
        "env: simple-nn-lammps\n"
        f"provisioning_status: {status}\n"
        "preflight:\n"
        "  require_binary: true\n"
        "  require_pair_style: nn\n"
    )
    return p


def test_stage10_plan_records_pending_backend_as_operational_boundary(tmp_path):
    plan = build_stage10_deployment_plan(
        md_config=_md_config(tmp_path, "PENDING_BEFORE_DEPLOYMENT_MD"),
        validation_profile=_profile(tmp_path),
        committee_manifest=_committee(tmp_path),
        starting_structure_contract=_start_contract(tmp_path),
        selected_seed=1,
        velocity_seed=12345,
    )
    assert plan["scientific_status"] == READY
    assert plan["operational_status"] == HUMAN_OPERATIONAL_INPUT_REQUIRED
    assert plan["executable"] is False
    assert plan["backend"]["provisioning_status"] == "PENDING_BEFORE_DEPLOYMENT_MD"
    validate_stage10_deployment_plan(plan)


def test_stage10_plan_requires_explicit_start_and_seed_policy(tmp_path):
    with pytest.raises(ValueError, match="starting structure"):
        build_stage10_deployment_plan(
            md_config=_md_config(tmp_path, "PROVISIONED"),
            validation_profile=_profile(tmp_path),
            committee_manifest=_committee(tmp_path),
            starting_structure_contract={},
            selected_seed=1,
            velocity_seed=12345,
        )
    with pytest.raises(ValueError, match="velocity_seed"):
        build_stage10_deployment_plan(
            md_config=_md_config(tmp_path, "PROVISIONED"),
            validation_profile=_profile(tmp_path),
            committee_manifest=_committee(tmp_path),
            starting_structure_contract=_start_contract(tmp_path),
            selected_seed=1,
        )


def test_stage10_plan_executable_only_when_backend_provisioned(tmp_path):
    plan = build_stage10_deployment_plan(
        md_config=_md_config(tmp_path, "PROVISIONED"),
        validation_profile=_profile(tmp_path),
        committee_manifest=_committee(tmp_path),
        starting_structure_contract=_start_contract(tmp_path),
        selected_seed=1,
        velocity_seed=12345,
    )
    assert plan["operational_status"] == READY
    assert plan["executable"] is True
    validate_stage10_deployment_plan(plan)


def test_executor_writes_plan_and_run_md_refuses_nonexecutable_plan(tmp_path):
    from runtimes.pydantic_ai.executors import _exec_build_stage10_deployment_plan, _exec_run_student_md

    out = tmp_path / "stage10_plan.json"
    res = _exec_build_stage10_deployment_plan({"parameters": {
        "md_config": str(_md_config(tmp_path, "PENDING_BEFORE_DEPLOYMENT_MD")),
        "validation_profile": str(_profile(tmp_path)),
        "committee_manifest": str(_committee(tmp_path)),
        "starting_structure_contract": str(_start_contract(tmp_path)),
        "selected_seed": 1,
        "velocity_seed": 12345,
        "out_path": str(out),
    }})
    assert Path(res["path"]).is_file()
    assert res["plan"]["executable"] is False
    with pytest.raises(ValueError, match="not executable"):
        _exec_run_student_md({"parameters": {"deployment_plan": str(out)}})
