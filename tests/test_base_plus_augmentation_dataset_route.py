"""Base-plus-augmentation Student distillation dataset route.

This is the production dataset-route correction approved in the prior campaign review, not an
unrelated feature. It replaces the historical single-source dataset_split with a route that
merges the same-run Teacher-baseline operational dataset (2,134 unprotected structures in the
real campaign) with the same-run Option A augmentation dataset (72 structures; 2,206 combined)
before splitting. Each test method below is annotated with the specific original production
requirement it proves; where no test proved a requirement, a new test was added rather than
assuming the six original tests were sufficient.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write

from runtimes.pydantic_ai import cli
from workflow.integrity import sha256_file

TEACHER_MODEL_SHA = "a" * 64
TEACHER_CONFIG_SHA = "b" * 64


def _frame(x, structure_id, category="bulk_amo", local_index=0, parent=None, energy=1.0,
           force_value=0.0):
    atoms = Atoms("SiO2", positions=[[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]], cell=[8, 8, 8], pbc=True)
    atoms.info["structure_id"] = structure_id
    atoms.info["source_category"] = category
    atoms.info["source_local_index"] = local_index
    atoms.info["source_config_type"] = category
    atoms.info["teacher_energy"] = float(energy)
    atoms.info["label_source"] = "teacher"
    if parent is not None:
        atoms.info["parent_structure_id"] = parent
    atoms.arrays["teacher_forces"] = np.full((len(atoms), 3), float(force_value))
    return atoms


class BasePlusAugmentationDatasetRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = self.root / "cfg"
        self.cfg.mkdir()
        self.run_dir = self.root / "run"
        self.base_frames = [
            _frame(0, "teacher-baseline:bulk_amo:000000", local_index=0),
            _frame(4, "teacher-baseline:bulk_amo:000001", local_index=1),
            _frame(8, "teacher-baseline:bulk_amo:000002", local_index=2),
        ]
        self.aug_frames = [
            _frame(0.2, "aug:0", local_index=0, parent="seed-pool:100", energy=2.0, force_value=5.0),
            _frame(4.2, "aug:1", local_index=1, parent="seed-pool:101", energy=3.0, force_value=6.0),
        ]
        self.reference = self._reference()
        self.workflow = self._workflow()

    def _reference(self):
        protected = self.cfg / "protected.extxyz"
        write(protected, [_frame(20, "protected", local_index=99)])
        rows = self.cfg / "protected_rows.txt"
        rows.write_text("760\n761\n", encoding="utf-8")
        rows_csv = self.cfg / "protected_source_rows.csv"
        rows_csv.write_text(
            "global_index,category,source_file_relative,source_local_index,natoms,config_type\n"
            "760,bulk_amo,bulk_amo/bulk_amo.xyz,660,3,bulk_amo\n"
            "761,bulk_amo,bulk_amo/bulk_amo.xyz,661,3,bulk_amo\n",
            encoding="utf-8",
        )
        manifest = self.cfg / "protected_manifest.json"
        manifest.write_text(json.dumps({"mapping": {
            "logical_test_frames": 1,
            "matched_logical_frames": 1,
            "unmatched_logical_frames": 0,
            "protected_source_rows": 2,
            "conflicting_label_duplicates": 0,
        }}), encoding="utf-8")
        ref = self.cfg / "reference.yaml"
        ref.write_text("\n".join([
            "kind: protected-existing-dft",
            "reference_id: fixture-reference",
            "reference_class: ORIGINAL_TEACHER_TEST",
            "status: AVAILABLE_AND_PROTECTED",
            "logical_test_frames: 1",
            "protected_source_rows: 2",
            f"protection_manifest: {manifest}",
            f"protected_source_rows_file: {rows}",
            f"protected_source_rows_csv: {rows_csv}",
            "duplicate_equivalent:",
            "  source_global_indices: [760, 761]",
            "  label_conflict: false",
            "prohibited_uses: [student_training, student_validation_tuning, acquisition_seed, augmentation_parent, recovery_training]",
            "structures:",
            f"  path: {protected}",
            "  logical_frames: 1",
            f"  sha256: {sha256_file(protected)}",
            "",
        ]), encoding="utf-8")
        return ref

    def _workflow(self):
        cfg = {
            "run_id": "dataset-route",
            "inputs": [{"path": str(self.reference), "role": "protected_reference"}],
            "stages": [{
                "name": "dataset_split",
                "command": None,
                "outputs": [
                    "artifacts/dataset/train.extxyz",
                    "artifacts/dataset/validation.extxyz",
                    "artifacts/dataset/test.extxyz",
                    "artifacts/dataset/split_manifest.json",
                    "artifacts/dataset_split_protection_audit.json",
                ],
                "gate": {"criteria": ["dataset split/protection passes"]},
                "contract": {
                    "kind": "validation_manifest",
                    "manifest": "artifacts/dataset_split_protection_audit.json",
                    "validator": "validation.protected_reference.validate_protection_audit_report",
                    "options": {"reference_yaml": str(self.reference)},
                },
                "pydantic_ai": {
                    "role": "data-curator",
                    "action": "generate_group_split",
                    "idempotency_key": "dataset-route:dataset_split:001",
                    "parameters": {
                        "base_dataset": "{run_dir}/artifacts/teacher_baseline_operational.extxyz",
                        "augmentation_dataset": "{run_dir}/artifacts/teacher_labeled.extxyz",
                        "base_label_manifest": "{run_dir}/artifacts/teacher_baseline_labels.manifest.json",
                        "augmentation_label_manifest": "{run_dir}/artifacts/teacher_labels.manifest.json",
                        "run_dir": "{run_dir}",
                        "merged_dataset": "{run_dir}/artifacts/dataset/student_distillation_labeled.extxyz",
                        "merge_manifest": "{run_dir}/artifacts/dataset/merge_manifest.json",
                        "dataset": "{run_dir}/artifacts/dataset/student_distillation_labeled.extxyz",
                        "output_dir": "{run_dir}/artifacts/dataset",
                        "manifest": "{run_dir}/artifacts/dataset/split_manifest.json",
                        "protection_audit_path": "{run_dir}/artifacts/dataset_split_protection_audit.json",
                        "reference_yaml": str(self.reference),
                        "seed": 7,
                        "validation_fraction": 0.2,
                        "test_fraction": 0.2,
                    },
                },
            }],
        }
        path = self.cfg / "workflow.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path

    def _init(self):
        from workflow.controller import RunController
        controller = RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames)
        return controller

    def _place_label_sources(self, base_frames, aug_frames, *, artifacts_dir=None,
                              base_teacher_model_sha=TEACHER_MODEL_SHA,
                              base_teacher_config_sha=TEACHER_CONFIG_SHA,
                              aug_teacher_model_sha=TEACHER_MODEL_SHA,
                              aug_teacher_config_sha=TEACHER_CONFIG_SHA,
                              corrupt_base_after_manifest=False,
                              corrupt_aug_after_manifest=False):
        """Write the base+augmentation label sources as same-run artifacts, with their
        label_with_teacher-shaped provenance manifests (schema matches
        adapters.acquisition.label_with_teacher's real output). ``artifacts_dir`` lets a test
        simulate a foreign-run (e.g. R10) artifact by writing outside self.run_dir."""
        artifacts = Path(artifacts_dir) if artifacts_dir is not None else self.run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        base_path = artifacts / "teacher_baseline_operational.extxyz"
        aug_path = artifacts / "teacher_labeled.extxyz"
        write(base_path, base_frames)
        write(aug_path, aug_frames)
        base_manifest = {
            "schema_version": 1,
            "teacher_model_sha256": base_teacher_model_sha,
            "teacher_config_sha256": base_teacher_config_sha,
            "output": str(base_path),
            "sha256": sha256_file(base_path),
            "n_frames": len(base_frames),
        }
        aug_manifest = {
            "schema_version": 1,
            "teacher_model_sha256": aug_teacher_model_sha,
            "teacher_config_sha256": aug_teacher_config_sha,
            "output": str(aug_path),
            "sha256": sha256_file(aug_path),
            "n_frames": len(aug_frames),
        }
        (artifacts / "teacher_baseline_labels.manifest.json").write_text(
            json.dumps(base_manifest), encoding="utf-8")
        (artifacts / "teacher_labels.manifest.json").write_text(
            json.dumps(aug_manifest), encoding="utf-8")
        if corrupt_base_after_manifest:
            write(base_path, base_frames + [_frame(99, "extra-untracked-frame")])
        if corrupt_aug_after_manifest:
            write(aug_path, aug_frames + [_frame(99, "extra-untracked-frame")])
        return base_path, aug_path

    def _run_stage(self):
        return cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(self.run_dir),
                         "--stage", "dataset_split", "--auto-mock-judges"])

    def _no_partial_outputs(self):
        self.assertFalse((self.run_dir / "artifacts/dataset/student_distillation_labeled.extxyz").exists())
        self.assertFalse((self.run_dir / "artifacts/dataset/merge_manifest.json").exists())
        self.assertFalse((self.run_dir / "artifacts/dataset/split_manifest.json").exists())
        self.assertFalse((self.run_dir / "artifacts/dataset/train.extxyz").exists())

    # -- merge composition, frame counts, teacher-label contract -------------------------------
    def test_run_stage_merges_base_and_augmentation_before_split(self):
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_SUCCESS)
        merged = read(self.run_dir / "artifacts/dataset/student_distillation_labeled.extxyz", ":")
        self.assertEqual(len(merged), 5)
        parents = {a.info["parent_structure_id"] for a in merged}
        self.assertIn("seed-pool:100", parents)
        self.assertIn("seed-pool:101", parents)
        self.assertIn("seed-pool:100", {a.info["parent_structure_id"] for a in merged if a.info["student_distillation_source"] == "base_teacher_operational"})
        merge = json.loads((self.run_dir / "artifacts/dataset/merge_manifest.json").read_text())
        self.assertEqual(merge["n_frames_pre_dedup"], 5)
        self.assertEqual(merge["n_frames"], 5)
        self.assertEqual(merge["teacher_label_contract"]["energy_key"], "teacher_energy")
        # Requirement: exact Teacher model/config/checkpoint binding across both label sources.
        self.assertEqual(merge["teacher_binding"]["teacher_model_sha256"], TEACHER_MODEL_SHA)
        self.assertEqual(merge["teacher_binding"]["teacher_config_sha256"], TEACHER_CONFIG_SHA)
        # Requirement: same-run Teacher-baseline label reuse (proven, not merely assumed).
        self.assertTrue(merge["same_run_verified"])
        self.assertEqual(Path(merge["run_dir"]), self.run_dir.resolve())
        # Numerical labels (energy AND per-atom force arrays) are carried through unaltered by
        # schema normalization — not just re-present, but bit-for-bit the same values.
        by_parent = {a.info["parent_structure_id"]: a for a in merged}
        self.assertEqual(by_parent["seed-pool:100"].info["teacher_energy"], 2.0)
        self.assertEqual(by_parent["seed-pool:101"].info["teacher_energy"], 3.0)
        np.testing.assert_array_equal(
            by_parent["seed-pool:100"].arrays["teacher_forces"], np.full((3, 3), 5.0))
        np.testing.assert_array_equal(
            by_parent["seed-pool:101"].arrays["teacher_forces"], np.full((3, 3), 6.0))
        # Requirement: protected-reference exclusion at the source-index level. The audit's
        # selected_source_indices must be the real resolved global indices of every merged
        # frame (not a stub), since the dataset_split contract's own validator
        # (validate_protection_audit_report) recomputes assert_source_indices_allowed from
        # exactly this field.
        audit = json.loads(
            (self.run_dir / "artifacts/dataset_split_protection_audit.json").read_text())
        self.assertEqual(audit["selected_source_indices"], [100, 101, 102])

    # -- deterministic exact-geometry dedup + hard failure on conflicting labels ---------------
    def test_duplicate_with_same_label_is_deduplicated(self):
        aug_frames = list(self.aug_frames)
        clone = self.base_frames[0].copy()
        clone.info["parent_structure_id"] = "seed-pool:100"
        aug_frames.append(clone)
        self.aug_frames = aug_frames
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_SUCCESS)
        merge = json.loads((self.run_dir / "artifacts/dataset/merge_manifest.json").read_text())
        self.assertEqual(merge["n_frames_pre_dedup"], 6)
        self.assertEqual(merge["n_frames"], 5)
        self.assertEqual(merge["n_exact_duplicates"], 1)

    def test_duplicate_conflicting_label_fails_without_partial_outputs(self):
        aug_frames = list(self.aug_frames)
        clone = self.base_frames[0].copy()
        clone.info["parent_structure_id"] = "seed-pool:100"
        clone.info["teacher_energy"] = 99.0
        aug_frames.append(clone)
        self.aug_frames = aug_frames
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    def test_split_failure_cleans_merged_partial_outputs(self):
        self.base_frames = [_frame(0, "teacher-baseline:bulk_amo:000000", local_index=0)]
        self.aug_frames = [_frame(4, "aug:0", local_index=1, parent="seed-pool:101", energy=2.0)]
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    def test_protected_parent_lineage_is_rejected(self):
        self.aug_frames = [
            _frame(0.2, "aug:0", local_index=0, parent="seed-pool:760", energy=2.0),
            _frame(4.2, "aug:1", local_index=1, parent="seed-pool:101", energy=3.0),
        ]
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    # -- protected-reference exclusion at the logical-geometry level ---------------------------
    # (an exact-geometry duplicate of the protected reference frame, injected via augmentation
    # with an unrelated/unprotected parent lineage and source index of its own — only a direct
    # geometry-fingerprint match, not lineage/category/source-index, can catch this leak.)
    def test_protected_logical_geometry_is_rejected(self):
        self.aug_frames = [
            _frame(20, "aug:0", local_index=0, parent="seed-pool:999", energy=2.0),
            _frame(4.2, "aug:1", local_index=1, parent="seed-pool:101", energy=3.0),
        ]
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    # -- protected-reference exclusion at the category/local -> global source-index level ------
    # (base frames derive parent_structure_id from source_category/source_local_index via the
    # reference.yaml protected_source_rows_csv offset table, not from an explicit override; this
    # is a distinct code path from the explicit-override case above and was previously untested.)
    def test_protected_category_local_offset_lineage_is_rejected(self):
        # bulk_amo local_index=660 maps to global source-index 760 (a protected row) via the
        # offset table in self._reference(); the base frame itself never states this directly.
        self.base_frames = [
            _frame(0, "teacher-baseline:bulk_amo:protected", local_index=660),
            _frame(4, "teacher-baseline:bulk_amo:000001", local_index=1),
            _frame(8, "teacher-baseline:bulk_amo:000002", local_index=2),
        ]
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    # -- protected-reference exclusion at the source-index level must be a genuine, re-verifiable
    #    check, not a stub. The dataset_split contract independently recomputes it from the
    #    audit's selected_source_indices field via validate_protection_audit_report.
    def test_protection_audit_source_indices_reject_tampered_leak(self):
        from validation.protected_reference import validate_protection_audit_report
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_SUCCESS)
        audit_path = self.run_dir / "artifacts/dataset_split_protection_audit.json"
        # The genuine audit content (real, non-empty resolved source indices) must
        # independently re-verify clean against the reference config.
        validate_protection_audit_report(audit_path, self.reference)
        # A tampered copy that claims a protected row (760) was among the selected source
        # indices must be rejected by re-verification — proving the check is a live guard
        # over the field's actual content, not a vacuous pass regardless of what's in it.
        tampered = json.loads(audit_path.read_text())
        tampered["selected_source_indices"] = tampered["selected_source_indices"] + [760]
        tampered_path = self.run_dir / "artifacts/tampered_protection_audit.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_protection_audit_report(tampered_path, self.reference)

    def test_no_r10_artifact_paths_are_used(self):
        self._init()
        wf = yaml.safe_load(self.workflow.read_text())
        params = wf["stages"][0]["pydantic_ai"]["parameters"]
        self.assertNotIn("sio2-sox-allegro-simplenn-r10", json.dumps(params))

    # -- same-run Teacher-baseline label reuse / no cross-run (e.g. R10) artifact reuse ---------
    def test_base_dataset_outside_run_dir_is_rejected(self):
        # Simulates pointing base_dataset at another run's (e.g. R10's) already-computed
        # teacher_baseline_operational.extxyz instead of this run's own artifact.
        # The stage parameters must be overridden in the *source* workflow.yaml before
        # RunController.initialize() copies it into run_dir/workflow.yaml — _stage_config()
        # re-reads only that copy, so an edit made after initialize() would silently be ignored.
        from workflow.controller import RunController
        foreign = self.root / "foreign_run" / "artifacts"
        wf = yaml.safe_load(self.workflow.read_text())
        params = wf["stages"][0]["pydantic_ai"]["parameters"]
        params["base_dataset"] = str(foreign / "teacher_baseline_operational.extxyz")
        params["base_label_manifest"] = str(foreign / "teacher_baseline_labels.manifest.json")
        self.workflow.write_text(yaml.safe_dump(wf), encoding="utf-8")
        RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames)
        self._place_label_sources(self.base_frames, self.aug_frames, artifacts_dir=foreign)
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    def test_augmentation_dataset_outside_run_dir_is_rejected(self):
        from workflow.controller import RunController
        foreign = self.root / "foreign_run" / "artifacts"
        wf = yaml.safe_load(self.workflow.read_text())
        params = wf["stages"][0]["pydantic_ai"]["parameters"]
        params["augmentation_dataset"] = str(foreign / "teacher_labeled.extxyz")
        params["augmentation_label_manifest"] = str(foreign / "teacher_labels.manifest.json")
        self.workflow.write_text(yaml.safe_dump(wf), encoding="utf-8")
        RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames)
        self._place_label_sources(self.base_frames, self.aug_frames, artifacts_dir=foreign)
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    # -- exact Teacher model/config/checkpoint binding across both label sources ----------------
    def test_mismatched_teacher_model_binding_is_rejected(self):
        from workflow.controller import RunController
        RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames,
                                  aug_teacher_model_sha="c" * 64)
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    def test_mismatched_teacher_config_binding_is_rejected(self):
        from workflow.controller import RunController
        RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames,
                                  aug_teacher_config_sha="d" * 64)
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    def test_missing_teacher_binding_field_is_rejected(self):
        from workflow.controller import RunController
        RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames,
                                  aug_teacher_model_sha=None)
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    # -- artifact contract validity: dataset bytes must match their own label manifest ----------
    def test_base_dataset_not_matching_its_own_label_manifest_is_rejected(self):
        from workflow.controller import RunController
        RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames,
                                  corrupt_base_after_manifest=True)
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    def test_augmentation_dataset_not_matching_its_own_label_manifest_is_rejected(self):
        from workflow.controller import RunController
        RunController.initialize(self.workflow, self.run_dir)
        self._place_label_sources(self.base_frames, self.aug_frames,
                                  corrupt_aug_after_manifest=True)
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()

    # -- base seed + all descendants co-located in one split family; deterministic group-disjoint
    #    split with actual reported frame counts (the prior version of this test asserted
    #    groups[x] == groups[x], a tautology that could never fail; replaced with a real check).
    def test_split_is_group_disjoint_with_accurate_reported_counts(self):
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_SUCCESS)
        merged = read(self.run_dir / "artifacts/dataset/student_distillation_labeled.extxyz", ":")
        split = json.loads((self.run_dir / "artifacts/dataset/split_manifest.json").read_text())
        group_to_split = {}
        for split_name, rec in split["splits"].items():
            for group in rec["group_ids"]:
                self.assertNotIn(group, group_to_split, f"group {group} assigned to >1 split")
                group_to_split[group] = split_name
        for split_name, rec in split["splits"].items():
            frames_in_split = read(rec["path"], ":")
            self.assertEqual(len(frames_in_split), rec["n_frames"],
                             f"{split_name} reported frame count does not match its file")
            for atoms in frames_in_split:
                self.assertEqual(group_to_split[atoms.info["parent_structure_id"]], split_name,
                                 "a frame landed in a split other than its own parent family's")
        for atoms in merged:
            self.assertIn(atoms.info["parent_structure_id"], group_to_split)

    # -- merge before split: a rejected merge must never let split run at all -------------------
    def test_merge_failure_prevents_split_from_running(self):
        aug_frames = list(self.aug_frames)
        clone = self.base_frames[0].copy()
        clone.info["parent_structure_id"] = "seed-pool:100"
        clone.info["teacher_energy"] = 99.0
        aug_frames.append(clone)
        self.aug_frames = aug_frames
        self._init()
        code = self._run_stage()
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self.assertFalse((self.run_dir / "artifacts/dataset/split_manifest.json").exists())
        dataset_dir = self.run_dir / "artifacts/dataset"
        if dataset_dir.exists():
            leftover = list(dataset_dir.glob("*.extxyz")) + list(dataset_dir.glob("*.json"))
            self.assertEqual(leftover, [], f"partial merge outputs left behind: {leftover}")

    # -- deterministic rerun: fixing the input after a rejected attempt must let a retry with the
    #    same idempotency key succeed cleanly (no partial-output or idempotency-key residue).
    def test_rerun_after_fixed_input_succeeds_deterministically(self):
        aug_frames = list(self.aug_frames)
        clone = self.base_frames[0].copy()
        clone.info["parent_structure_id"] = "seed-pool:100"
        clone.info["teacher_energy"] = 99.0
        self.aug_frames = aug_frames + [clone]
        self._init()
        first = self._run_stage()
        self.assertEqual(first, cli.EXIT_VALIDATION_REJECTED)
        self._no_partial_outputs()
        # Fix the conflicting duplicate in place and retry the same stage/idempotency key.
        self._place_label_sources(self.base_frames, list(self.aug_frames[:-1]))
        second = self._run_stage()
        self.assertEqual(second, cli.EXIT_SUCCESS)
        merged = read(self.run_dir / "artifacts/dataset/student_distillation_labeled.extxyz", ":")
        self.assertEqual(len(merged), 5)


if __name__ == "__main__":
    unittest.main()
