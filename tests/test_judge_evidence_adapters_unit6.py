"""UNIT 6 tests: bounded-evidence Judge adapters for previously-uncovered stages.

The Judges read `_json_summary` of a stage's produced report, not the raw file. The default summary
exposes only top-level KEY NAMES + a few generic scalars, so criterion-relevant VALUES that live in
nested dicts/lists (or a non-generic top-level scalar) are invisible. These tests prove the new
adapters surface exactly those values for:

  * Stage-2 reference_validation  -- Teacher-vs-reference disagreement + reference span
  * Stage-3 acquisition           -- parents/plan-envelope/produced diversity/plan lineage
  * Stage-6 dataset_split         -- leakage counts + per-split sizes/grouping
  * Stage-12 analysis run_summary -- campaign_outcome + gate/recovery ledgers + bounding claims

and that the too-loose ``label_manifest`` predicate no longer spuriously fires on the acquisition
or dataset_split manifests.
"""
from __future__ import annotations

import json
from pathlib import Path

from runtimes.pydantic_ai.bounded_evidence import _json_summary


def _summary(tmp_path, name, payload):
    p = Path(tmp_path) / name
    p.write_text(json.dumps(payload, indent=2))
    return _json_summary(p)


# ------------------------------------------------------------------ Stage 2: reference_validation


def _reference_validation_payload():
    return {
        "schema_version": 1,
        "profile": "teacher_reference_validation",
        "stage": "reference_validation",
        "protected_reference_use": "teacher_vs_dft_reference_validation_only",
        "historical_prediction_policy": "PROVENANCE_ONLY_NOT_USED_AS_FRESH_RESULT",
        "teacher": {"model_sha256": "a" * 64},
        "reference": {"reference_id": "ref-001", "structures_path": "/run/ref.xyz",
                      "logical_frames": 120, "protected_source_rows": 240},
        "prediction_artifact": {"path": "/run/pred.xyz", "n_frames": 120,
                                "labels": ["teacher_energy", "teacher_forces",
                                           "dft_energy", "dft_forces"]},
        "metrics": {"energy_normalization": "per_atom", "energy_unit": "meV/atom",
                    "force_unit": "eV/Angstrom",
                    "global": {"energy_mae": 12.3, "energy_rmse": 15.7,
                               "force_component_mae": 0.08, "force_component_rmse": 0.11},
                    "by_config_type": {"bulk": {"energy_mae": 10.0}},
                    "domain_fields": {"in_domain": True}},
        "checks": [{"domain": "teacher_reference", "observable": "energy_mae",
                    "status": "RECORDED", "value": 12.3, "unit": "meV/atom", "criterion": None}],
        "evidence": [],
    }


def test_reference_validation_adapter_surfaces_disagreement_and_scope(tmp_path):
    s = _summary(tmp_path, "reference_validation.json", _reference_validation_payload())
    rv = s["reference_validation_report"]
    assert rv["global_disagreement"]["energy_mae"] == 12.3
    assert rv["disagreement_by_config_type"]["bulk"]["energy_mae"] == 10.0
    assert rv["reference_id"] == "ref-001"
    assert rv["reference_logical_frames"] == 120
    assert rv["prediction_labels"] == ["teacher_energy", "teacher_forces",
                                       "dft_energy", "dft_forces"]
    assert rv["energy_unit"] == "meV/atom"
    assert rv["checks"][0]["observable"] == "energy_mae"


# ------------------------------------------------------------------ Stage 3: acquisition


def _acquisition_payload(operation="acquire_structures", n_parents=3):
    parents = [f"parent-{i}" for i in range(n_parents)]
    return {
        "schema_version": 1,
        "operation": operation,
        "stage": "acquisition",
        "acquisition_plan_sha256": "b" * 64,
        "selected_parent_structure_ids": parents,
        "selected_source_global_indices": list(range(n_parents)),
        "eligible_source_categories": ["bulk", "amorphous"],
        "selected_source_records": [{"id": p} for p in parents],
        "expected_output_count": n_parents * 4,
        "actual_output_count": n_parents * 4,
        "n_frames": n_parents * 4,
        "elements": ["Si", "O"],
        "duplicate_handling": "reject_exact",
        "dft_labels_used_as_selection_scores": False,
        "framework_plan_envelope": {"n_parents": n_parents, "n_per_structure": 4,
                                    "T_K": 300.0, "beta": 1.0, "sigma_range_A": [0.01, 0.1],
                                    "seed": 7, "expected_output_count": n_parents * 4},
        "protection_audit_result": {"ok": True},
    }


def test_acquisition_adapter_surfaces_selection_science(tmp_path):
    s = _summary(tmp_path, "acquisition.manifest.json", _acquisition_payload())
    aq = s["acquisition_manifest"]
    assert aq["operation"] == "acquire_structures"
    assert aq["acquisition_plan_sha256"] == "b" * 64
    assert aq["n_selected_parents"] == 3
    assert aq["selected_parent_sample"] == ["parent-0", "parent-1", "parent-2"]
    assert aq["elements"] == ["Si", "O"]
    assert aq["eligible_source_categories"] == ["bulk", "amorphous"]
    assert aq["framework_plan_envelope"]["n_per_structure"] == 4
    assert aq["actual_output_count"] == 12


def test_acquisition_adapter_matches_existing_pool_path(tmp_path):
    s = _summary(tmp_path, "acq.json", _acquisition_payload(operation="select_existing_pool"))
    assert s["acquisition_manifest"]["operation"] == "select_existing_pool"


def test_acquisition_adapter_bounds_large_parent_list(tmp_path):
    s = _summary(tmp_path, "acq.json", _acquisition_payload(n_parents=500))
    aq = s["acquisition_manifest"]
    assert aq["n_selected_parents"] == 500
    assert len(aq["selected_parent_sample"]) == 32  # bounded sample, not the raw vector


def test_acquisition_manifest_no_longer_matches_label_manifest(tmp_path):
    s = _summary(tmp_path, "acq.json", _acquisition_payload())
    assert "acquisition_manifest" in s
    assert "label_manifest" not in s  # tightened predicate no longer fires on n_frames alone


# ------------------------------------------------------------------ Stage 6: dataset_split


def _dataset_split_payload():
    return {
        "schema_version": 1,
        "source": "/run/dataset.extxyz",
        "source_sha256": "c" * 64,
        "seed": 42,
        "grouping_key": "parent_structure_id",
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "splits": {
            "test": {"path": "/run/test.extxyz", "sha256": "d" * 64, "n_frames": 6,
                     "group_ids": ["g1"]},
            "validation": {"path": "/run/val.extxyz", "sha256": "e" * 64, "n_frames": 6,
                           "group_ids": ["g2"]},
            "train": {"path": "/run/train.extxyz", "sha256": "f" * 64, "n_frames": 48,
                      "group_ids": ["g3", "g4", "g5"]},
        },
        "overlap_checks": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }


def test_dataset_split_adapter_surfaces_leakage_and_sizes(tmp_path):
    s = _summary(tmp_path, "split_manifest.json", _dataset_split_payload())
    ds = s["dataset_split_manifest"]
    assert ds["overlap_checks"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert ds["split_frame_counts"] == {"test": 6, "validation": 6, "train": 48}
    assert ds["split_group_counts"] == {"test": 1, "validation": 1, "train": 3}
    assert ds["grouping_key"] == "parent_structure_id"
    assert ds["seed"] == 42


def test_dataset_split_manifest_no_longer_matches_label_manifest(tmp_path):
    s = _summary(tmp_path, "split_manifest.json", _dataset_split_payload())
    assert "dataset_split_manifest" in s
    assert "label_manifest" not in s  # source_sha256 alone must not trigger label_manifest


# ------------------------------------------------------------------ Stage 12: analysis run_summary


def _run_summary_payload():
    return {
        "schema_version": 1,
        "run_id": "run-xyz",
        "campaign_outcome": "RECOVERY_IN_PROGRESS_OR_REQUIRED",
        "stages": [
            {"name": "training", "status": "completed", "gate": "PASS",
             "artifacts": [{"path": "/a", "sha256": "1"}, {"path": "/b", "sha256": "2"}]},
            {"name": "evaluation", "status": "completed", "gate": "REVISE", "artifacts": []},
        ],
        "gate_history": [{"stage": "training", "verdict": "PASS", "at": "t0"},
                         {"stage": "evaluation", "verdict": "REVISE", "at": "t1"}],
        "recoveries": [{"id": 1, "status": "approved", "failed_stage": "evaluation"}],
        "identified_gaps": ["gap-a"],
        "limitations": ["mechanical snapshot only"],
        "evidence": [],
    }


def test_run_summary_adapter_surfaces_outcome_and_ledgers(tmp_path):
    s = _summary(tmp_path, "analysis.json", _run_summary_payload())
    rs = s["run_summary_report"]
    assert rs["campaign_outcome"] == "RECOVERY_IN_PROGRESS_OR_REQUIRED"
    assert rs["stages"][0] == {"name": "training", "status": "completed", "gate": "PASS",
                               "n_artifacts": 2}
    assert rs["gate_history"] == [{"stage": "training", "verdict": "PASS"},
                                  {"stage": "evaluation", "verdict": "REVISE"}]
    assert rs["recoveries"] == [{"id": 1, "status": "approved", "failed_stage": "evaluation"}]
    assert rs["identified_gaps"] == ["gap-a"]
    assert rs["limitations"] == ["mechanical snapshot only"]


# ------------------------------------------------------------------ label_manifest still works


def test_real_label_manifest_still_matches(tmp_path):
    payload = {
        "schema_version": 1,
        "teacher_model_sha256": "a" * 64,
        "teacher_config_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "sha256": "d" * 64,
        "n_frames": 100,
        "labels": ["energy", "forces"],
        "units": {"energy": "eV", "forces": "eV/Angstrom"},
        "environment": {"python": "3.13"},
    }
    s = _summary(tmp_path, "label.manifest.json", payload)
    lm = s["label_manifest"]
    assert lm["n_frames"] == 100
    assert lm["labels"] == ["energy", "forces"]
    assert lm["teacher_model_sha256"] == "a" * 64
