"""R31 training-gate evidence contract: compact, deterministic training-evidence summary.

These tests pin the SIMPLE_NN LOG parser and the evidence builder to *only* surfacing values that
genuinely exist (never inventing), and to computing the eight deterministic verification outcomes
the training gate's criteria ask about. Network-free; builds a synthetic run_dir on tmp.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtimes.pydantic_ai.training_evidence import (
    NOT_RECORDED,
    TRAINING_EVIDENCE_PROFILE,
    build_training_evidence_summary,
    parse_simple_nn_log,
    write_training_evidence_summary,
)

_LOG = """\
SEED: 202631
Total traning epoch : 200
Epoch 001 E RMSE(T V) 9.1000 9.2000 F RMSE(T V) 8.1000 8.2000 learning_rate: 0.001
Epoch 200 E RMSE(T V) 3.3922 3.6290 F RMSE(T V) 2.7129 2.3886 learning_rate: 0.0001
Best loss lammps potential written at 200 epoch
Elapsed time in training: 1120.5 s.
Total wall time: 1125.3 s.
"""


class ParseSimpleNnLogTests(unittest.TestCase):
    def test_extracts_real_values(self):
        p = parse_simple_nn_log(_LOG)
        self.assertEqual(p["seed"], 202631)
        self.assertEqual(p["epochs_requested"], 200)
        self.assertEqual(p["epochs_completed"], 200)
        self.assertEqual(p["best_epoch"], 200)
        self.assertEqual(p["stopping_reason"], "completed_all_requested_epochs")
        self.assertEqual(p["wall_time_s"], 1125.3)
        self.assertEqual(p["training_elapsed_s"], 1120.5)
        # final-epoch metrics come from the LAST epoch line, not the first
        self.assertEqual(p["final_train_energy_rmse"], 3.3922)
        self.assertEqual(p["final_valid_energy_rmse"], 3.6290)
        self.assertEqual(p["final_train_force_rmse"], 2.7129)
        self.assertEqual(p["final_valid_force_rmse"], 2.3886)

    def test_rmse_note_disclaims_evaluation_metric_interpretation(self):
        note = parse_simple_nn_log(_LOG)["rmse_note"]
        self.assertIn("NOT Student-vs-Teacher", note)
        self.assertIn("not asserted", note)

    def test_missing_fields_become_not_recorded_never_invented(self):
        p = parse_simple_nn_log("SEED: 999\n(no epoch lines, no timing)\n")
        self.assertEqual(p["seed"], 999)
        self.assertEqual(p["epochs_requested"], NOT_RECORDED)
        self.assertEqual(p["epochs_completed"], NOT_RECORDED)
        self.assertEqual(p["best_epoch"], NOT_RECORDED)
        self.assertEqual(p["wall_time_s"], NOT_RECORDED)
        self.assertEqual(p["stopping_reason"], NOT_RECORDED)
        self.assertEqual(p["final_valid_force_rmse"], NOT_RECORDED)

    def test_incomplete_run_reports_stopped_at_epoch(self):
        text = ("SEED: 5\nTotal traning epoch : 200\n"
                "Epoch 042 E RMSE(T V) 1 2 F RMSE(T V) 3 4 learning_rate: 0.01\n")
        p = parse_simple_nn_log(text)
        self.assertEqual(p["epochs_completed"], 42)
        self.assertEqual(p["stopping_reason"], "stopped_at_epoch_42_of_200")


def _make_run(tmp: Path, *, n_seeds=4, matching_dataset=True, with_logs=True,
              protection_pass=True, overlaps_zero=True):
    run_dir = tmp / "run"
    artifacts = run_dir / "artifacts"
    dataset_dir = artifacts / "dataset"
    committee_dir = artifacts / "committee"
    dataset_dir.mkdir(parents=True)
    committee_dir.mkdir(parents=True)

    train_path = dataset_dir / "train.extxyz"
    train_path.write_text("frame-data\n", encoding="utf-8")
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()

    seeds = list(range(202631, 202631 + n_seeds))
    models = []
    for i, seed in enumerate(seeds):
        seed_dir = committee_dir / f"seed-{seed}"
        seed_dir.mkdir()
        ck = seed_dir / "potential_saved_bestmodel"
        ck.write_text(f"checkpoint-{seed}\n", encoding="utf-8")
        if with_logs:
            (seed_dir / "LOG").write_text(
                _LOG.replace("SEED: 202631", f"SEED: {seed}"), encoding="utf-8")
        models.append({
            "seed": seed, "path": str(ck),
            "integrity": {"kind": "file", "size": 20 + i,
                          "sha256": hashlib.sha256(ck.read_bytes()).hexdigest()},
        })

    manifest = {
        "run_id": "test-run", "dataset": str(train_path),
        "dataset_integrity": {"size": 11,
                              "sha256": train_sha if matching_dataset else "deadbeef"},
        "student_config": str(artifacts / "student_config.yaml"),
        "student_config_integrity": {"sha256": "cfg-sha"},
        "models": models,
    }
    (artifacts / "student_committee.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")

    (dataset_dir / "split_manifest.json").write_text(json.dumps({
        "run_id": "test-run",
        "splits": {"train": {"sha256": train_sha, "n_frames": 354}},
        "overlap_checks": ({"train_validation": 0, "train_test": 0, "validation_test": 0}
                           if overlaps_zero else {"train_test": 3}),
    }), encoding="utf-8")

    (artifacts / "dataset_split_protection_audit.json").write_text(json.dumps({
        "checks": ({"protected_source_indices": "PASS",
                    "protected_logical_geometries": "PASS",
                    "protected_parent_lineage": "PASS"}
                   if protection_pass else {"protected_source_indices": "FAIL"}),
    }), encoding="utf-8")

    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "test-run",
        "artifacts": [{"path": str(committee_dir), "sha256": "committee-agg-sha"}],
    }), encoding="utf-8")
    return run_dir, train_sha, seeds


class BuildTrainingEvidenceSummaryTests(unittest.TestCase):
    def test_happy_path_all_verifications_pass(self):
        with TemporaryDirectory() as td:
            run_dir, train_sha, seeds = _make_run(Path(td))
            s = build_training_evidence_summary(run_dir)
            self.assertEqual(s["profile"], TRAINING_EVIDENCE_PROFILE)
            self.assertTrue(s["all_verifications_passed"])
            self.assertEqual(s["committee"]["n_models"], 4)
            self.assertEqual(s["committee"]["seeds"], seeds)
            self.assertEqual(s["dataset_provenance"]["sha256"], train_sha)
            self.assertEqual(s["dataset_provenance"]["accepted_split_train_sha256"], train_sha)
            self.assertTrue(s["dataset_provenance"]["belongs_to_this_run"])
            self.assertEqual(s["dataset_provenance"]["n_frames"], 354)
            self.assertEqual(s["committee"]["committee_dir_sha256"], "committee-agg-sha")
            # dynamics recovered per seed
            self.assertEqual(len(s["training_dynamics"]), 4)
            self.assertTrue(all(d["log_present"] for d in s["training_dynamics"]))
            self.assertTrue(all(d["best_epoch"] == 200 for d in s["training_dynamics"]))

    def test_checkpoint_hashes_are_distinct(self):
        with TemporaryDirectory() as td:
            run_dir, _, _ = _make_run(Path(td))
            s = build_training_evidence_summary(run_dir)
            hashes = [m["checkpoint_sha256"] for m in s["committee"]["members"]]
            self.assertEqual(len(set(hashes)), 4)
            check = {o["check"]: o["ok"] for o in s["verification_outcomes"]}
            self.assertTrue(check["checkpoint_hashes_distinct"])
            self.assertTrue(check["all_checkpoint_files_exist"])

    def test_dataset_hash_mismatch_fails_only_that_check(self):
        with TemporaryDirectory() as td:
            run_dir, _, _ = _make_run(Path(td), matching_dataset=False)
            s = build_training_evidence_summary(run_dir)
            check = {o["check"]: o["ok"] for o in s["verification_outcomes"]}
            self.assertFalse(check["training_dataset_hash_matches_accepted_split_train"])
            self.assertFalse(s["all_verifications_passed"])
            # unrelated checks are unaffected
            self.assertTrue(check["committee_size_is_frozen_4"])
            self.assertTrue(check["checkpoint_hashes_distinct"])

    def test_wrong_committee_size_flagged(self):
        with TemporaryDirectory() as td:
            run_dir, _, _ = _make_run(Path(td), n_seeds=3)
            s = build_training_evidence_summary(run_dir)
            check = {o["check"]: o["ok"] for o in s["verification_outcomes"]}
            self.assertFalse(check["committee_size_is_frozen_4"])

    def test_protection_and_overlap_failures_flagged(self):
        with TemporaryDirectory() as td:
            run_dir, _, _ = _make_run(Path(td), protection_pass=False, overlaps_zero=False)
            s = build_training_evidence_summary(run_dir)
            check = {o["check"]: o["ok"] for o in s["verification_outcomes"]}
            self.assertFalse(check["protected_reference_overlap_zero"])
            self.assertFalse(check["train_val_test_parent_family_leakage_zero"])

    def test_missing_logs_degrade_to_not_recorded_without_raising(self):
        with TemporaryDirectory() as td:
            run_dir, _, _ = _make_run(Path(td), with_logs=False)
            s = build_training_evidence_summary(run_dir)
            self.assertTrue(all(d["log_present"] is False for d in s["training_dynamics"]))
            # checkpoint/provenance verifications still pass; only dynamics are absent
            self.assertTrue(s["all_verifications_passed"])

    def test_writer_emits_single_json_artifact(self):
        with TemporaryDirectory() as td:
            run_dir, _, _ = _make_run(Path(td))
            out = write_training_evidence_summary(run_dir)
            self.assertTrue(out.exists())
            self.assertEqual(out.name, "training_evidence_summary.json")
            reloaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["profile"], TRAINING_EVIDENCE_PROFILE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
