"""Stage-8 multi-population, role-bound, channel-separated evaluation contract.

Covers the reusable framework contract (framework_v2.evaluation_population) and
the orchestrator (workflow.steps.evaluate_multi_population): role<->channel
firewall, channel uniqueness, training/protected leakage fail-closed,
missing-required-channel fail-closed, population SHA identity binding,
multi-population role separation, and provenance/hash binding.

Real committee inference is monkeypatched out; the real four_channel_audit
``channel`` computation and the real leakage fingerprints run unchanged.
"""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from framework_v2.evaluation_population import (
    EvaluationPopulation, EvaluationPopulationRole as Role,
    MultiPopulationEvaluationPlan, assert_no_training_leakage,
    EvaluationLeakageError, STUDENT_VS_TEACHER, STUDENT_VS_DFT, TEACHER_VS_DFT,
    ROLE_ALLOWED_CHANNELS,
)
from workflow.integrity import sha256_file
import workflow.steps as steps


# --------------------------------------------------------------------------
# contract-level validators (pure)
# --------------------------------------------------------------------------
def _pop(pid, role, sha, channels, **kw):
    return EvaluationPopulation(population_id=pid, role=role, frames_path=f"/x/{pid}",
                                structures_sha256=sha, required_channels=channels, **kw)


def test_role_channel_firewall_distillation_rejects_dft():
    with pytest.raises(ValueError, match="not permitted"):
        _pop("d", Role.DISTILLATION_HOLDOUT, "a" * 64, [STUDENT_VS_DFT])


def test_role_channel_firewall_dft_rejects_student_vs_teacher():
    with pytest.raises(ValueError, match="not permitted"):
        _pop("d", Role.DFT_PROTECTED_HOLDOUT, "a" * 64, [STUDENT_VS_TEACHER])


def test_empty_required_channels_rejected():
    with pytest.raises(ValueError, match="no.*required_channels"):
        _pop("d", Role.DISTILLATION_HOLDOUT, "a" * 64, [])


def test_plan_rejects_duplicate_channel_across_populations():
    a = _pop("a", Role.DFT_PROTECTED_HOLDOUT, "a" * 64, [STUDENT_VS_DFT])
    b = _pop("b", Role.DFT_PROTECTED_HOLDOUT, "b" * 64, [STUDENT_VS_DFT, TEACHER_VS_DFT])
    with pytest.raises(ValueError, match="two populations|duplicate role"):
        MultiPopulationEvaluationPlan(plan_id="p", populations=[a, b])


def test_plan_channel_assignments_unique():
    dist = _pop("dist", Role.DISTILLATION_HOLDOUT, "a" * 64, [STUDENT_VS_TEACHER])
    dft = _pop("dft", Role.DFT_PROTECTED_HOLDOUT, "b" * 64, [STUDENT_VS_DFT, TEACHER_VS_DFT])
    plan = MultiPopulationEvaluationPlan(plan_id="p", populations=[dist, dft])
    assert plan.channel_assignments() == {
        STUDENT_VS_TEACHER: "dist", STUDENT_VS_DFT: "dft", TEACHER_VS_DFT: "dft"}
    # plan identity is a stable content hash
    assert plan.content_sha256() == MultiPopulationEvaluationPlan(
        plan_id="p", populations=[dist, dft]).content_sha256()


def test_role_allowed_channels_are_disjoint_and_cover_all():
    dist = ROLE_ALLOWED_CHANNELS[Role.DISTILLATION_HOLDOUT]
    dft = ROLE_ALLOWED_CHANNELS[Role.DFT_PROTECTED_HOLDOUT]
    assert dist.isdisjoint(dft)
    assert dist | dft == {STUDENT_VS_TEACHER, STUDENT_VS_DFT, TEACHER_VS_DFT}


def test_leakage_guard_fails_closed():
    with pytest.raises(EvaluationLeakageError, match="leaks into the training set"):
        assert_no_training_leakage({"f1", "f2"}, {"f2"}, population_id="p")
    assert_no_training_leakage({"f1"}, {"f2"}, population_id="p")  # disjoint OK


# --------------------------------------------------------------------------
# orchestrator (workflow.steps.evaluate_multi_population)
# --------------------------------------------------------------------------
def _frame(seed, *, teacher=False, dft=False, n=6, cell=10.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, cell, size=(n, 3))
    at = Atoms("Si" + str(n // 2) + "O" + str(n - n // 2), positions=pos, cell=[cell] * 3, pbc=True)
    at.info["config_type"] = "bulk_cryst" if seed % 2 else "liquid"
    if teacher:
        at.info["teacher_energy"] = float(rng.uniform(-100, -50))
        at.arrays["teacher_forces"] = rng.uniform(-1, 1, size=(n, 3))
    if dft:
        at.info["dft_energy"] = float(rng.uniform(-100, -50))
        at.arrays["dft_forces"] = rng.uniform(-1, 1, size=(n, 3))
    return at


def _write_frames(path, frames):
    write(str(path), frames)
    return sha256_file(str(path))


def _fake_predict(cfg, committee, frames):
    # embed synthetic single-seed student predictions (seed 01)
    for i, at in enumerate(frames):
        at.info["student_energy_seed01"] = float(-70 + 0.1 * i)
        at.arrays["student_forces_seed01"] = np.full((len(at), 3), 0.05)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(steps, "load_config", lambda p: {})
    monkeypatch.setattr(steps, "verify_artifact", lambda *a, **k: None)
    monkeypatch.setattr(steps, "_predict_committee_onto", _fake_predict)


def _committee(tmp_path):
    m = tmp_path / "committee.json"
    m.write_text(json.dumps({"models": [{"seed": 1, "path": str(tmp_path / "ck.pth"),
                                         "integrity": {}}]}))
    (tmp_path / "ck.pth").write_text("x")
    (tmp_path / "student.yaml").write_text("deploy: {}\n")
    return str(m), str(tmp_path / "student.yaml")


def _run(tmp_path, plan, training_frames):
    manifest, cfg = _committee(tmp_path)
    tr = tmp_path / "train.extxyz"
    _write_frames(tr, training_frames)
    labeled = tmp_path / "evaluated.extxyz"
    report = tmp_path / "accuracy_report.json"
    return steps.evaluate_multi_population(
        cfg, manifest, plan, str(tr), str(labeled), str(report),
        code_revision="deadbeef")


def test_multi_population_role_separation_and_provenance(tmp_path, patched):
    # distillation holdout: teacher labels only
    dist_frames = [_frame(i, teacher=True) for i in range(1, 5)]
    dist_path = tmp_path / "dist.extxyz"
    dist_sha = _write_frames(dist_path, dist_frames)
    # dft holdout: carries DFT ground truth AND Teacher predictions (a
    # Teacher-labeled DFT holdout) so both student_vs_dft and teacher_vs_dft
    # are computable. Disjoint structures (high seeds).
    dft_frames = [_frame(i, teacher=True, dft=True) for i in range(100, 104)]
    dft_path = tmp_path / "dft.extxyz"
    dft_sha = _write_frames(dft_path, dft_frames)
    # training set: different structures again
    train_frames = [_frame(i, teacher=True) for i in range(500, 503)]

    plan = MultiPopulationEvaluationPlan(plan_id="eng-plan", populations=[
        EvaluationPopulation(population_id="distillation", role=Role.DISTILLATION_HOLDOUT,
                             frames_path=str(dist_path), structures_sha256=dist_sha,
                             required_channels=[STUDENT_VS_TEACHER]),
        EvaluationPopulation(population_id="dft_protected", role=Role.DFT_PROTECTED_HOLDOUT,
                             frames_path=str(dft_path), structures_sha256=dft_sha,
                             required_channels=[STUDENT_VS_DFT, TEACHER_VS_DFT],
                             source_manifest_sha256="c" * 64),
    ])
    rep = _run(tmp_path, plan, train_frames)

    # channel separation: each channel produced by exactly the right population
    assert set(rep["channels"]) == {STUDENT_VS_TEACHER, STUDENT_VS_DFT, TEACHER_VS_DFT}
    assert rep["channels"][STUDENT_VS_TEACHER]["population"]["population_id"] == "distillation"
    assert rep["channels"][STUDENT_VS_TEACHER]["population"]["role"] == "DISTILLATION_HOLDOUT"
    assert rep["channels"][STUDENT_VS_DFT]["population"]["population_id"] == "dft_protected"
    assert rep["channels"][TEACHER_VS_DFT]["population"]["population_id"] == "dft_protected"
    # per-domain metrics present
    assert "all" in rep["channels"][STUDENT_VS_TEACHER]["metrics"]
    # provenance/hash binding
    assert rep["plan_sha256"] == plan.content_sha256()
    assert rep["channels"][STUDENT_VS_DFT]["population"]["structures_sha256"] == dft_sha
    assert rep["channels"][STUDENT_VS_DFT]["population"]["source_manifest_sha256"] == "c" * 64
    assert rep["code_revision"] == "deadbeef"
    assert rep["committee_models"][0]["seed"] == 1
    assert rep["channel_assignments"][STUDENT_VS_DFT] == "dft_protected"


def test_dft_channel_over_teacher_only_population_fails_closed(tmp_path, patched):
    # A DFT_PROTECTED_HOLDOUT whose frames actually LACK dft labels: the
    # require_complete channel guard must fail closed.
    frames = [_frame(i, teacher=True) for i in range(1, 4)]  # no dft labels
    p = tmp_path / "nodft.extxyz"
    sha = _write_frames(p, frames)
    plan = MultiPopulationEvaluationPlan(plan_id="p", populations=[
        EvaluationPopulation(population_id="dft", role=Role.DFT_PROTECTED_HOLDOUT,
                             frames_path=str(p), structures_sha256=sha,
                             required_channels=[STUDENT_VS_DFT])])
    with pytest.raises(RuntimeError, match="incomplete"):
        _run(tmp_path, plan, [_frame(900, teacher=True)])


def test_training_leakage_fails_closed(tmp_path, patched):
    shared = _frame(7, dft=True)
    eval_frames = [shared, _frame(8, dft=True)]
    p = tmp_path / "dft.extxyz"
    sha = _write_frames(p, eval_frames)
    plan = MultiPopulationEvaluationPlan(plan_id="p", populations=[
        EvaluationPopulation(population_id="dft", role=Role.DFT_PROTECTED_HOLDOUT,
                             frames_path=str(p), structures_sha256=sha,
                             required_channels=[STUDENT_VS_DFT])])
    # training set contains the SAME structure -> leakage
    with pytest.raises(EvaluationLeakageError, match="leaks into the training set"):
        _run(tmp_path, plan, [_frame(7, dft=True)])


def test_population_sha_mismatch_fails_closed(tmp_path, patched):
    frames = [_frame(i, dft=True) for i in range(1, 4)]
    p = tmp_path / "dft.extxyz"
    _write_frames(p, frames)
    plan = MultiPopulationEvaluationPlan(plan_id="p", populations=[
        EvaluationPopulation(population_id="dft", role=Role.DFT_PROTECTED_HOLDOUT,
                             frames_path=str(p), structures_sha256="f" * 64,  # wrong
                             required_channels=[STUDENT_VS_DFT])])
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        _run(tmp_path, plan, [_frame(900, dft=True)])


def test_distillation_only_population_succeeds(tmp_path, patched):
    frames = [_frame(i, teacher=True) for i in range(1, 5)]
    p = tmp_path / "dist.extxyz"
    sha = _write_frames(p, frames)
    plan = MultiPopulationEvaluationPlan(plan_id="p", populations=[
        EvaluationPopulation(population_id="dist", role=Role.DISTILLATION_HOLDOUT,
                             frames_path=str(p), structures_sha256=sha,
                             required_channels=[STUDENT_VS_TEACHER])])
    rep = _run(tmp_path, plan, [_frame(900, teacher=True)])
    assert set(rep["channels"]) == {STUDENT_VS_TEACHER}
    assert rep["evaluation_kind"] == "multi_population"
