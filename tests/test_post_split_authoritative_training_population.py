"""FE-058: authoritative Stage-7 training-population resolution for post-split augmentation.

When a workflow declares ``training.pydantic_ai.requires_post_split_augmentation: true`` (FE-054),
the completed augment-train lifecycle produces a merged ``final_train`` that Stage 7 actually
consumes. These tests pin the deterministic training-evidence builder's resolution of the
AUTHORITATIVE training population and the identity invariant
``consumed_training_dataset_sha == authoritative_training_population_sha``:

  * with augmentation declared+finalized+REGISTERED, the authoritative population is final_train
    (not the pre-augmentation Stage-6 split), and the Stage-6 TRAIN sha stays preserved in lineage;
  * consumed==final_train passes the identity check; consumed!=final_train still fails it;
  * augmentation declared but final_train NOT registered fails closed (invariant not relaxed);
  * non-augmented workflows still resolve to the Stage-6 TRAIN split;
  * all committee members' metadata is reconstructable from existing input.yaml/LOG (no retrain);
  * no duplicate/parallel authority is created (registration is idempotent-shaped).

Network-free; builds synthetic run_dirs on tmp.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtimes.pydantic_ai.training_evidence import (
    NOT_RECORDED,
    build_training_evidence_summary,
    reconstruct_member_effective_config,
    resolve_authoritative_training_population,
)

_LOG = """\
SEED: {seed}
Total traning epoch : 200
Epoch 001 E RMSE(T V) 9.1 9.2 F RMSE(T V) 8.1 8.2 learning_rate: 0.001
Epoch 200 E RMSE(T V) 3.39 3.62 F RMSE(T V) 2.71 2.38 learning_rate: 0.0001
Best loss lammps potential written at 200 epoch
Elapsed time in training: 1120.5 s.
Total wall time: 1125.3 s.
"""

_INPUT_YAML = """\
generate_features: true
params:
  Si: /x/params_Si
  O: /x/params_O
neural_network:
  method: Adam
  nodes: "30-30"
  batch_size: 32
  total_epoch: 200
  learning_rate: 0.0001
  double_precision: true
  use_force: true
  use_stress: false
"""

_WORKFLOW = """\
stages:
  - name: dataset_split
  - name: training
    pydantic_ai:
      action: train_committee
      requires_post_split_augmentation: {declared}
"""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _make_run(tmp: Path, *, declared=True, register_final=True, consumed_matches_final=True,
              finalize=True):
    run_dir = tmp / "run"
    artifacts = run_dir / "artifacts"
    dataset_dir = artifacts / "dataset"
    committee_dir = artifacts / "committee"
    aug_dir = run_dir / "augmentation"
    for d in (dataset_dir, committee_dir, aug_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Stage-6 TRAIN split (65 "frames"), and the post-split merged final_train (577 "frames").
    train_path = dataset_dir / "train.extxyz"
    train_path.write_text("stage6-train-bytes\n", encoding="utf-8")
    split_train_sha = _sha(train_path)

    final_path = dataset_dir / "final_train.extxyz"
    final_path.write_text("post-split-merged-final-train-bytes\n", encoding="utf-8")
    final_sha = _sha(final_path)

    # Consumed dataset the committee actually trained on.
    consumed_sha = final_sha if consumed_matches_final else "deadbeefdeadbeef"

    seeds = [202601, 202602, 202603, 202604]
    models = []
    for i, seed in enumerate(seeds):
        seed_dir = committee_dir / f"seed-{seed}"
        seed_dir.mkdir()
        ck = seed_dir / "potential_saved_bestmodel"
        ck.write_text(f"checkpoint-{seed}\n", encoding="utf-8")
        (seed_dir / "LOG").write_text(_LOG.format(seed=seed), encoding="utf-8")
        (seed_dir / "input.yaml").write_text(_INPUT_YAML, encoding="utf-8")
        models.append({"seed": seed, "path": str(ck), "metadata": {},
                       "integrity": {"kind": "file", "size": 20 + i, "sha256": _sha(ck)}})

    (artifacts / "student_committee.manifest.json").write_text(json.dumps({
        "run_id": "eng-test", "dataset": str(final_path),
        "dataset_integrity": {"size": 30, "sha256": consumed_sha},
        "student_config": str(artifacts / "student.yaml"),
        "student_config_integrity": {"sha256": "cfg-sha"},
        "models": models,
    }), encoding="utf-8")

    (dataset_dir / "split_manifest.json").write_text(json.dumps({
        "run_id": "eng-test",
        "splits": {"train": {"sha256": split_train_sha, "n_frames": 65}},
        "overlap_checks": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }), encoding="utf-8")
    (artifacts / "dataset_split_protection_audit.json").write_text(json.dumps({
        "checks": {"protected_source_indices": "PASS"},
    }), encoding="utf-8")

    if finalize:
        children = aug_dir / "labeling" / "children_labeled.extxyz"
        children.parent.mkdir(parents=True, exist_ok=True)
        children.write_text("children\n", encoding="utf-8")
        children_sha = _sha(children)
        merge_manifest = aug_dir / "final_train.merge.manifest.json"
        merge_manifest.write_text(json.dumps({
            "n_frames": 577, "output_integrity": final_sha,
            "sources": [
                {"path": str(train_path), "integrity": {"sha256": split_train_sha}, "n_frames": 65},
                {"path": str(children), "integrity": {"sha256": children_sha}, "n_frames": 512},
            ],
        }), encoding="utf-8")
        (aug_dir / "augmentation_finalized.json").write_text(json.dumps({
            "augmentation_warranted": True, "augmentation_complete": True,
            "strategy_kind": "LOCAL_PERTURBATION",
            "plan_content_sha256": "plan-sha",
            "base_train_dataset_sha256": split_train_sha,
            "n_augmented_children": 512,
            "merge_manifest": str(merge_manifest),
            "merge_output_sha256": final_sha,
            "final_train_path": str(final_path),
            "final_train_sha256": final_sha,
        }), encoding="utf-8")

    wf_path = run_dir / "workflow.yaml"
    wf_path.write_text(_WORKFLOW.format(declared=str(bool(declared)).lower()), encoding="utf-8")

    inputs = []
    if register_final:
        inputs.append({"source": str(final_path), "snapshot": None, "copy": False,
                       "sha256": final_sha, "source_sha256": final_sha, "size": 30})
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "eng-test", "workflow_config": str(wf_path), "inputs": inputs,
        "artifacts": [{"stage": "dataset_split", "path": str(train_path),
                       "sha256": split_train_sha},
                      {"path": str(committee_dir), "sha256": "committee-agg-sha"}],
    }), encoding="utf-8")
    return run_dir, split_train_sha, final_sha, seeds


class AuthoritativePopulationResolutionTests(unittest.TestCase):
    def _resolve(self, run_dir):
        state = json.loads((Path(run_dir) / "manifest.json").read_text())
        split = json.loads((Path(run_dir) / "artifacts" / "dataset" / "split_manifest.json"
                            ).read_text())
        return resolve_authoritative_training_population(run_dir, state, split)

    def test_final_train_becomes_authoritative_when_declared_and_registered(self):
        with TemporaryDirectory() as td:
            run_dir, split_sha, final_sha, _ = _make_run(Path(td))
            res = self._resolve(run_dir)
            self.assertTrue(res["resolution_ok"])
            self.assertEqual(res["authoritative_training_population_sha256"], final_sha)
            self.assertEqual(res["authoritative_producing_stage"], "post_split_train_augmentation")
            self.assertEqual(res["authoritative_training_population_n_frames"], 577)
            self.assertNotEqual(final_sha, split_sha)

    def test_stage6_split_train_preserved_in_lineage(self):
        with TemporaryDirectory() as td:
            run_dir, split_sha, final_sha, _ = _make_run(Path(td))
            res = self._resolve(run_dir)
            self.assertEqual(res["stage6_split_train_sha256"], split_sha)
            self.assertEqual(res["stage6_split_train_n_frames"], 65)
            prov = res["post_split_augmentation"]
            self.assertEqual(prov["base_train_dataset_sha256"], split_sha)
            self.assertEqual(prov["merge_output_sha256"], final_sha)
            self.assertEqual(prov["n_augmented_children"], 512)

    def test_non_augmented_run_uses_stage6_train(self):
        with TemporaryDirectory() as td:
            run_dir, split_sha, _, _ = _make_run(Path(td), declared=False)
            res = self._resolve(run_dir)
            self.assertTrue(res["resolution_ok"])
            self.assertEqual(res["authoritative_training_population_sha256"], split_sha)
            self.assertEqual(res["authoritative_producing_stage"], "dataset_split")
            self.assertIsNone(res["post_split_augmentation"])

    def test_declared_but_unregistered_fails_closed(self):
        with TemporaryDirectory() as td:
            run_dir, _, _, _ = _make_run(Path(td), register_final=False)
            res = self._resolve(run_dir)
            self.assertFalse(res["resolution_ok"])
            self.assertIn("not registered", res["resolution_detail"])

    def test_declared_but_not_finalized_fails_closed(self):
        with TemporaryDirectory() as td:
            run_dir, _, _, _ = _make_run(Path(td), finalize=False, register_final=False)
            res = self._resolve(run_dir)
            self.assertFalse(res["resolution_ok"])


class IdentityInvariantTests(unittest.TestCase):
    def _checks(self, run_dir):
        s = build_training_evidence_summary(run_dir)
        return s, {o["check"]: o["ok"] for o in s["verification_outcomes"]}

    def test_consumed_equals_final_train_passes_identity(self):
        with TemporaryDirectory() as td:
            run_dir, _, final_sha, _ = _make_run(Path(td), consumed_matches_final=True)
            s, checks = self._checks(run_dir)
            self.assertTrue(
                checks["training_dataset_hash_matches_authoritative_training_population"])
            self.assertTrue(s["all_verifications_passed"])
            self.assertEqual(
                s["dataset_provenance"]["authoritative_training_population_sha256"], final_sha)
            self.assertEqual(s["dataset_provenance"]["producing_stage"],
                             "post_split_train_augmentation")
            self.assertEqual(s["dataset_provenance"]["n_frames"], 577)

    def test_consumed_not_final_train_still_fails_identity(self):
        with TemporaryDirectory() as td:
            run_dir, _, _, _ = _make_run(Path(td), consumed_matches_final=False)
            s, checks = self._checks(run_dir)
            self.assertFalse(
                checks["training_dataset_hash_matches_authoritative_training_population"])
            self.assertFalse(s["all_verifications_passed"])

    def test_declared_unregistered_final_train_fails_identity(self):
        with TemporaryDirectory() as td:
            run_dir, _, _, _ = _make_run(Path(td), register_final=False)
            s, checks = self._checks(run_dir)
            # invariant not relaxed: even though consumed bytes == final_train bytes, the population
            # is not the canonical authoritative input, so the identity check fails closed.
            self.assertFalse(
                checks["training_dataset_hash_matches_authoritative_training_population"])


class CommitteeMetadataReconstructionTests(unittest.TestCase):
    def test_all_members_effective_config_reconstructed(self):
        with TemporaryDirectory() as td:
            run_dir, _, _, seeds = _make_run(Path(td))
            s = build_training_evidence_summary(run_dir)
            members = s["committee"]["members"]
            self.assertEqual(len(members), 4)
            for m in members:
                cfg = m["effective_config"]
                self.assertTrue(cfg["input_yaml_present"])
                self.assertEqual(cfg["nodes"], "30-30")
                self.assertEqual(cfg["total_epoch"], 200)
                self.assertEqual(cfg["learning_rate"], 0.0001)
                self.assertEqual(cfg["descriptor_elements"], ["O", "Si"])
                # per-member LOG dynamics folded in (no retrain)
                self.assertTrue(m["training_dynamics"]["log_present"])
                self.assertEqual(m["training_dynamics"]["best_epoch"], 200)
                self.assertEqual(m["training_dynamics"]["wall_time_s"], 1125.3)

    def test_missing_input_yaml_marked_unavailable_not_invented(self):
        with TemporaryDirectory() as td:
            run_dir, _, _, _ = _make_run(Path(td))
            (Path(run_dir) / "artifacts" / "committee" / "seed-202601" / "input.yaml").unlink()
            cfg = reconstruct_member_effective_config(
                Path(run_dir) / "artifacts" / "committee", 202601)
            self.assertFalse(cfg["input_yaml_present"])
            self.assertNotIn("nodes", cfg)


class NoParallelAuthorityTests(unittest.TestCase):
    def test_single_authoritative_population_no_duplicate(self):
        with TemporaryDirectory() as td:
            run_dir, split_sha, final_sha, _ = _make_run(Path(td))
            res = self._res(run_dir)
            # exactly one authoritative sha; Stage-6 preserved as distinct lineage, not a rival
            self.assertEqual(res["authoritative_training_population_sha256"], final_sha)
            self.assertNotEqual(
                res["authoritative_training_population_sha256"], res["stage6_split_train_sha256"])
            state = json.loads((Path(run_dir) / "manifest.json").read_text())
            matches = [r for r in state["inputs"] if r.get("sha256") == final_sha]
            self.assertEqual(len(matches), 1)

    def _res(self, run_dir):
        state = json.loads((Path(run_dir) / "manifest.json").read_text())
        split = json.loads((Path(run_dir) / "artifacts" / "dataset" / "split_manifest.json"
                            ).read_text())
        return resolve_authoritative_training_population(run_dir, state, split)


class _StubController:
    """Minimal controller surface for the registration helper: a state dict, a bind_new_input that
    appends a run-bound input record (mirroring the real controller's record shape), and save()."""
    def __init__(self):
        self.state = {"inputs": [], "artifacts": [], "events": []}
        self.bind_calls = 0

    def bind_new_input(self, source, *, copy=True):
        self.bind_calls += 1
        sha = _sha(Path(source))
        self.state["inputs"].append({"source": str(Path(source).resolve()), "snapshot": None,
                                     "copy": copy, "sha256": sha, "source_sha256": sha})

    def save(self):
        pass


class RegisterFinalTrainHelperTests(unittest.TestCase):
    def _fm(self, final_path):
        return {"final_train_path": str(final_path), "final_train_sha256": _sha(Path(final_path)),
                "base_train_dataset_sha256": "base", "n_augmented_children": 512,
                "merge_output_sha256": "merge"}

    def test_binds_final_train_once_and_records_event(self):
        from runtimes.pydantic_ai.cli import _register_final_train_as_training_input
        with TemporaryDirectory() as td:
            ft = Path(td) / "final_train.extxyz"
            ft.write_text("merged\n", encoding="utf-8")
            c = _StubController()
            _register_final_train_as_training_input(c, self._fm(ft))
            self.assertEqual(c.bind_calls, 1)
            self.assertEqual(len(c.state["inputs"]), 1)
            evts = [e for e in c.state["events"] if e["type"] == "post_split_final_train_bound"]
            self.assertEqual(len(evts), 1)

    def test_idempotent_when_already_bound_no_parallel_authority(self):
        from runtimes.pydantic_ai.cli import _register_final_train_as_training_input
        with TemporaryDirectory() as td:
            ft = Path(td) / "final_train.extxyz"
            ft.write_text("merged\n", encoding="utf-8")
            c = _StubController()
            _register_final_train_as_training_input(c, self._fm(ft))
            _register_final_train_as_training_input(c, self._fm(ft))  # second call is a no-op
            self.assertEqual(c.bind_calls, 1)
            self.assertEqual(len(c.state["inputs"]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
