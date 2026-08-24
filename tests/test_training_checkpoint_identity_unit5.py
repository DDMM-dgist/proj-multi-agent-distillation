"""UNIT 5 tests: canonical run-published Student checkpoint identity (Stage-7 -> Stage-10).

The training stage publishes a committee manifest (per-seed checkpoint identity + integrity). The
deployment stage must resolve the deployed checkpoint ONLY from the manifest the training stage
published in THIS run -- it must not be able to deploy a checkpoint the training stage never
published. These tests cover the run-binding that guarantees that property:

  * validation.deployment_resolution.resolve_published_committee_manifest -- derives the canonical
    published committee-manifest identity (path + sha256) from the run's own artifact registry,
    failing closed if training never published, the producing stage is unfinished, the active set is
    ambiguous, or the on-disk manifest has drifted.
  * validation.deployment_resolution.resolve_selected_checkpoint(expected_manifest_sha256=...) --
    rejects any consumed manifest whose sha256 is not the published one.
  * runtimes.pydantic_ai.executors._exec_resolve_deployment_checkpoint -- passes the expected sha
    through, so a foreign/unpublished manifest fails closed end to end.
  * the Stage-7 costly-training approval boundary + governed seed selection (audit properties).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.deployment_resolution import (
    CANONICAL_COMMITTEE_MANIFEST_NAME,
    resolve_published_committee_manifest,
    resolve_selected_checkpoint,
)
from workflow.integrity import sha256_file


def _published_committee(dir_path, *, seeds=(1, 2, 3), losses=None,
                         name=CANONICAL_COMMITTEE_MANIFEST_NAME):
    """Write per-seed checkpoints + a canonical committee manifest with real sha256 integrity."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    models = []
    for s in seeds:
        ckpt = dir_path / f"seed-{s}" / "mock-model.json"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps({"seed": int(s)}) + "\n")
        meta = {"trainer_kind": "analytic_mock"}
        if losses is not None:
            meta["loss"] = losses[s]
        models.append({"kind": "mock", "seed": int(s), "path": str(ckpt),
                       "integrity": {"kind": "file", "size": ckpt.stat().st_size,
                                     "sha256": sha256_file(ckpt)},
                       "metadata": meta})
    manifest = dir_path / name
    manifest.write_text(json.dumps({"schema_version": 1, "models": models}, indent=2))
    return manifest


def _state_with_published_manifest(manifest, *, stage="training", status="completed"):
    """A minimal controller-state mapping recording the manifest as a completed-stage artifact."""
    return {
        "stages": [{"name": stage, "status": status}],
        "artifacts": [{"stage": stage, "path": str(Path(manifest).resolve()),
                       "sha256": sha256_file(manifest)}],
    }


# --------------------------------------------------- resolve_published_committee_manifest


def test_published_manifest_resolves_from_registry(tmp_path):
    manifest = _published_committee(tmp_path / "committee", seeds=(1, 2, 3))
    state = _state_with_published_manifest(manifest)
    res = resolve_published_committee_manifest(state)
    assert res["path"] == str(manifest.resolve())
    assert res["sha256"] == sha256_file(manifest)
    assert res["stage"] == "training"
    assert res["published_seeds"] == [1, 2, 3]


def test_published_manifest_fails_closed_when_never_published(tmp_path):
    state = {"stages": [{"name": "training", "status": "completed"}], "artifacts": []}
    with pytest.raises(ValueError, match="never published"):
        resolve_published_committee_manifest(state)


def test_published_manifest_fails_closed_when_producing_stage_unfinished(tmp_path):
    manifest = _published_committee(tmp_path / "committee")
    state = _state_with_published_manifest(manifest, status="running")
    with pytest.raises(ValueError, match="has not completed"):
        resolve_published_committee_manifest(state)


def test_published_manifest_fails_closed_when_ambiguous(tmp_path):
    m1 = _published_committee(tmp_path / "c1", seeds=(1, 2))
    m2 = _published_committee(tmp_path / "c2", seeds=(3, 4))
    state = {
        "stages": [{"name": "training", "status": "completed"}],
        "artifacts": [
            {"stage": "training", "path": str(m1.resolve()), "sha256": sha256_file(m1)},
            {"stage": "training", "path": str(m2.resolve()), "sha256": sha256_file(m2)},
        ],
    }
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_published_committee_manifest(state)


def test_published_manifest_fails_closed_on_disk_drift(tmp_path):
    manifest = _published_committee(tmp_path / "committee")
    state = _state_with_published_manifest(manifest)
    # tamper the manifest on disk after it was registered
    manifest.write_text(manifest.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="does not match its registered sha256"):
        resolve_published_committee_manifest(state)


def test_published_manifest_accepts_controller_object(tmp_path):
    manifest = _published_committee(tmp_path / "committee")

    class _FakeController:
        state = _state_with_published_manifest(manifest)

    res = resolve_published_committee_manifest(_FakeController())
    assert res["sha256"] == sha256_file(manifest)


# --------------------------------------------------- expected_manifest_sha256 guard


def test_expected_sha_guard_passes_on_match(tmp_path):
    manifest = _published_committee(tmp_path / "committee", seeds=(1, 2))
    res = resolve_selected_checkpoint(
        manifest, selected_seed=1, expected_manifest_sha256=sha256_file(manifest))
    assert res["published_manifest_binding_verified"] is True
    assert res["expected_manifest_sha256"] == sha256_file(manifest)


def test_expected_sha_guard_rejects_foreign_manifest(tmp_path):
    published = _published_committee(tmp_path / "published", seeds=(1, 2))
    # a DIFFERENT manifest (checkpoints the training stage never published) with a real seed
    foreign = _published_committee(tmp_path / "foreign", seeds=(1, 2))
    assert sha256_file(foreign) != sha256_file(published)
    with pytest.raises(ValueError, match="never published"):
        resolve_selected_checkpoint(
            foreign, selected_seed=1, expected_manifest_sha256=sha256_file(published))


def test_expected_sha_guard_rejects_in_memory_manifest(tmp_path):
    manifest = _published_committee(tmp_path / "committee", seeds=(1, 2))
    payload = json.loads(manifest.read_text())
    with pytest.raises(ValueError, match="in-memory dict"):
        resolve_selected_checkpoint(
            payload, selected_seed=1, expected_manifest_sha256=sha256_file(manifest))


def test_binding_not_verified_when_no_expected_sha(tmp_path):
    manifest = _published_committee(tmp_path / "committee", seeds=(1, 2))
    res = resolve_selected_checkpoint(manifest, selected_seed=1)
    assert res["published_manifest_binding_verified"] is False
    assert res["expected_manifest_sha256"] is None


# --------------------------------------------------- executor end-to-end run-binding


def _resolve_deploy(manifest, tmp_path, *, expected_sha=None, selected_seed=1):
    from runtimes.pydantic_ai.executors import _exec_resolve_deployment_checkpoint
    datafile = tmp_path / "start.lammps-data"
    datafile.write_text("dummy\n")
    out = tmp_path / "deployment_provenance.json"
    params = {"committee_manifest": str(manifest), "selected_seed": selected_seed,
              "starting_structure": str(datafile), "out_path": str(out)}
    if expected_sha is not None:
        params["expected_committee_manifest_sha256"] = expected_sha
    return _exec_resolve_deployment_checkpoint({"parameters": params}), out


def test_executor_binds_to_published_manifest_end_to_end(tmp_path):
    manifest = _published_committee(tmp_path / "committee", seeds=(1, 2))
    state = _state_with_published_manifest(manifest)
    published = resolve_published_committee_manifest(state)
    res, out = _resolve_deploy(manifest, tmp_path, expected_sha=published["sha256"])
    prov = json.loads(out.read_text())
    assert prov["student"]["selected_seed"] == 1
    assert res["student"]["published_manifest_binding_verified"] is True


def test_executor_rejects_unpublished_manifest_end_to_end(tmp_path):
    published = _published_committee(tmp_path / "published", seeds=(1, 2))
    state = _state_with_published_manifest(published)
    published_id = resolve_published_committee_manifest(state)
    # deployment proposal names a DIFFERENT committee (never published by training) but supplies the
    # run-published expected sha -> must fail closed.
    foreign = _published_committee(tmp_path / "foreign", seeds=(1, 2))
    with pytest.raises(ValueError, match="never published"):
        _resolve_deploy(foreign, tmp_path, expected_sha=published_id["sha256"])


# --------------------------------------------------- audit properties


def test_train_committee_is_inherently_costly_training():
    from runtimes.pydantic_ai.actions import (
        _INHERENT_COSTLY_ACTIONS, APPROVAL_GATED_ACTIONS)
    assert "train_committee" in _INHERENT_COSTLY_ACTIONS
    assert APPROVAL_GATED_ACTIONS.get("train_committee") == "costly_training"


def test_deployment_seed_selection_stays_governed_even_when_bound(tmp_path):
    manifest = _published_committee(tmp_path / "committee", seeds=(1, 2))
    # A verified run-binding does NOT license inventing a seed: selection is still governed.
    with pytest.raises(ValueError, match="governed decision"):
        resolve_selected_checkpoint(
            manifest, expected_manifest_sha256=sha256_file(manifest))
