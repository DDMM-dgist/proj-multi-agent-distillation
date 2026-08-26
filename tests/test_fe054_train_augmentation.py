"""FE-054 -- post-split TRAIN-only augmentation action: focused unit tests.

These prove the DETERMINISTIC seams of ``runtimes.pydantic_ai.train_augmentation`` without any
GPU / Teacher / LLM: the protected-parent exclusion when building the TRAIN-parent pool manifest,
the child-lineage remap the merge requires, the declaration switch + Stage-7 fail-closed guard, the
AugmentationPlan freeze provenance, and the honest EXISTING_POOL_SELECTION ("no augmentation
warranted") finalize path end-to-end. The costly LOCAL_PERTURBATION generation/labeling/merge is
exercised live in the campaign run, not here.
"""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import yaml

from ase import Atoms
from ase.io import write as ase_write

from workflow.controller import RunController
from runtimes.pydantic_ai import train_augmentation as ta


def _make_frame(source_global_index=None, parent_structure_id=None, symbol="Si"):
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=[6.0, 6.0, 6.0], pbc=True)
    if source_global_index is not None:
        atoms.info["source_global_index"] = int(source_global_index)
    if parent_structure_id is not None:
        atoms.info["parent_structure_id"] = parent_structure_id
    return atoms


def _make_controller(root, *, training_route=None, run_id="fe054"):
    """A minimal non-v2 controller with a canonical ``training`` stage.

    ``training_route`` is merged into ``training.pydantic_ai`` so tests can declare
    ``requires_post_split_augmentation`` and the resolvable ``parameters.dataset``."""
    stage = {"name": "training",
             "command": ["true"],
             "outputs": []}
    if training_route is not None:
        # The real training stage routes action=train_committee; the guard keys on that action.
        stage["pydantic_ai"] = {"action": "train_committee", **training_route}
    cfg = {"run_id": run_id, "stages": [stage]}
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump(cfg, sort_keys=False))
    RunController.initialize(workflow, root / "run")
    return RunController(root / "run")


class FrameSourceGlobalIndex(unittest.TestCase):
    def test_reads_explicit_info_key(self):
        self.assertEqual(ta._frame_source_global_index(_make_frame(source_global_index=7), 0), 7)

    def test_parses_seed_pool_lineage(self):
        atoms = _make_frame(parent_structure_id="seed-pool:42")
        self.assertEqual(ta._frame_source_global_index(atoms, 0), 42)

    def test_fails_closed_without_provenance(self):
        with self.assertRaises(ValueError):
            ta._frame_source_global_index(_make_frame(), 0)


class BuildTrainPoolManifest(unittest.TestCase):
    def _train_file(self, root, globals_):
        frames = [_make_frame(source_global_index=g) for g in globals_]
        path = root / "train.extxyz"
        ase_write(str(path), frames, format="extxyz")
        return path

    def test_excludes_protected_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._train_file(root, [10, 11, 12, 13])
            out = ta.build_train_pool_manifest(
                train, root / "train_pool", protected_source_indices={11, 13})
            self.assertEqual(out["n_frames"], 2)
            self.assertEqual(out["n_protected_excluded"], 2)
            self.assertEqual(out["excluded_globals"], [11, 13])
            manifest = json.loads(Path(out["manifest_path"]).read_text())
            self.assertEqual(manifest["total_frames"], 2)
            self.assertEqual(manifest["categories"][0]["n_frames"], 2)
            self.assertIn("sanitized_pool_manifest_sha256", manifest)
            # The filtered parents file physically holds only the two kept frames.
            from ase.io import read as ase_read
            kept = ase_read(out["parents_path"], index=":")
            self.assertEqual([a.info["source_global_index"] for a in kept], [10, 12])

    def test_no_protection_keeps_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._train_file(root, [0, 1, 2])
            out = ta.build_train_pool_manifest(train, root / "train_pool")
            self.assertEqual(out["n_frames"], 3)
            self.assertEqual(out["n_protected_excluded"], 0)

    def test_all_excluded_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._train_file(root, [5, 6])
            with self.assertRaises(ValueError):
                ta.build_train_pool_manifest(
                    train, root / "train_pool", protected_source_indices={5, 6})


class DeriveTrainBaseLabelManifest(unittest.TestCase):
    def _setup(self, root, *, train_globals=(0, 1, 2), tamper_train=False,
               source_manifest_count=1):
        c = _make_controller(root)
        ds = c.run_dir / "artifacts" / "dataset"
        ds.mkdir(parents=True, exist_ok=True)
        train = ds / "train.extxyz"
        ase_write(str(train), [_make_frame(source_global_index=g) for g in train_globals],
                  format="extxyz")
        train_sha = ta._sha256_file(train)
        source_sha = "poolsha_" + "a" * 56
        # Authoritative Stage-5 manifest(s): output sha == the split source sha.
        for i in range(source_manifest_count):
            (c.run_dir / "artifacts" / f"teacher_labels{i}.manifest.json").write_text(json.dumps({
                "schema_version": 1, "sha256": source_sha, "n_frames": 81,
                "teacher_model_sha256": "tm" + "0" * 62,
                "teacher_config_sha256": "tc" + "0" * 62,
                "units": "eV", "output": "pool.extxyz"}))
        split = {"schema_version": 1, "source_sha256": source_sha,
                 "splits": {"train": {"sha256": ("bad" if tamper_train else train_sha),
                                      "n_frames": len(train_globals)}}}
        (ds / "split_manifest.json").write_text(json.dumps(split))
        return c, train

    def test_projects_stage5_binding_onto_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, train = self._setup(Path(tmp))
            out = ta.derive_train_base_label_manifest(c, train_dataset=str(train))
            payload = json.loads(Path(out).read_text())
            # sha256 is recomputed to the TRAIN split bytes (not the full-pool sha).
            self.assertEqual(payload["sha256"], ta._sha256_file(train))
            self.assertEqual(payload["n_frames"], 3)
            # Teacher binding copied verbatim so the merge can prove same-Teacher.
            self.assertEqual(payload["teacher_model_sha256"], "tm" + "0" * 62)
            self.assertEqual(payload["teacher_config_sha256"], "tc" + "0" * 62)
            self.assertEqual(payload["derived_from"]["split"], "train")

    def test_fails_closed_on_train_byte_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, train = self._setup(Path(tmp), tamper_train=True)
            with self.assertRaises(ValueError):
                ta.derive_train_base_label_manifest(c, train_dataset=str(train))

    def test_fails_closed_when_source_manifest_not_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, train = self._setup(Path(tmp), source_manifest_count=2)
            with self.assertRaises(ValueError):
                ta.derive_train_base_label_manifest(c, train_dataset=str(train))


class RemapChildLineage(unittest.TestCase):
    def test_remaps_item_id_to_seed_pool(self):
        children = [_make_frame(parent_structure_id="train_parents#0"),
                    _make_frame(parent_structure_id="train_parents#1")]
        ta.remap_child_lineage(children, {"train_parents#0": "seed-pool:10",
                                          "train_parents#1": "seed-pool:11"})
        self.assertEqual([c.info["parent_structure_id"] for c in children],
                         ["seed-pool:10", "seed-pool:11"])

    def test_unknown_parent_fails_closed(self):
        children = [_make_frame(parent_structure_id="train_parents#9")]
        with self.assertRaises(ValueError):
            ta.remap_child_lineage(children, {"train_parents#0": "seed-pool:10"})


class DeclarationAndGuard(unittest.TestCase):
    def test_augmentation_required_true_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _make_controller(root, training_route={"requires_post_split_augmentation": True})
            self.assertTrue(ta.augmentation_required(c))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _make_controller(root, training_route=None)
            self.assertFalse(ta.augmentation_required(c))

    def test_guard_noop_when_not_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _make_controller(root, training_route=None)
            self.assertIsNone(ta.stage7_augmentation_guard(c, "training"))

    def test_guard_fails_closed_when_declared_but_unfinalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _make_controller(root, training_route={"requires_post_split_augmentation": True})
            reason = ta.stage7_augmentation_guard(c, "training")
            self.assertIsNotNone(reason)
            self.assertIn("no finalized augmentation manifest", reason)

    def test_verify_and_guard_pass_after_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ft = "{run_dir}/artifacts/dataset/final_train.extxyz"
            c = _make_controller(root, training_route={
                "requires_post_split_augmentation": True,
                "parameters": {"dataset": ft}})
            # An honest EXISTING_POOL_SELECTION finalize (no Teacher call) satisfies the guard.
            base = root / "labeled_train.extxyz"
            base.write_text("dummy labeled train parents\n")
            frozen_plan = {"strategy_kind": "EXISTING_POOL_SELECTION",
                           "plan_content_sha256": "abc123",
                           "protected_augmentation_parents_excluded": []}
            ta._finalize_no_augmentation(c, frozen_plan=frozen_plan, base_dataset=str(base))
            ok, reason = ta.verify_finalized_augmentation(c)
            self.assertTrue(ok, reason)
            self.assertIsNone(ta.stage7_augmentation_guard(c, "training"))

    def test_verify_fails_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ft = "{run_dir}/artifacts/dataset/final_train.extxyz"
            c = _make_controller(root, training_route={
                "requires_post_split_augmentation": True,
                "parameters": {"dataset": ft}})
            base = root / "labeled_train.extxyz"
            base.write_text("dummy labeled train parents\n")
            ta._finalize_no_augmentation(
                c, frozen_plan={"strategy_kind": "EXISTING_POOL_SELECTION",
                                "plan_content_sha256": "abc123",
                                "protected_augmentation_parents_excluded": []},
                base_dataset=str(base))
            # Tamper with the finalized dataset after the manifest recorded its hash.
            ta.final_train_path(c.run_dir).write_text("tampered\n")
            ok, reason = ta.verify_finalized_augmentation(c)
            self.assertFalse(ok)
            self.assertIn("hash mismatch", reason)

    def test_verify_fails_when_not_routed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Declared, but training.dataset points at the PRE-augmentation split, not final_train.
            c = _make_controller(root, training_route={
                "requires_post_split_augmentation": True,
                "parameters": {"dataset": "{run_dir}/artifacts/dataset/train.extxyz"}})
            base = root / "labeled_train.extxyz"
            base.write_text("dummy\n")
            ta._finalize_no_augmentation(
                c, frozen_plan={"strategy_kind": "EXISTING_POOL_SELECTION",
                                "plan_content_sha256": "abc123",
                                "protected_augmentation_parents_excluded": []},
                base_dataset=str(base))
            ok, reason = ta.verify_finalized_augmentation(c)
            self.assertFalse(ok)
            self.assertIn("not routed", reason)


class FreezeAugmentationPlan(unittest.TestCase):
    def _stub_produced(self, *, strategy_kind, legacy=None, existing_pool=None):
        plan = types.SimpleNamespace(content_sha256=lambda: "planSHA")
        realized = types.SimpleNamespace(
            plan=plan, legacy_projection=legacy, existing_pool_projection=existing_pool)
        coverage = types.SimpleNamespace(content_sha256=lambda: "coverageSHA")
        strategy = types.SimpleNamespace(kind=types.SimpleNamespace(value=strategy_kind))
        ctx = types.SimpleNamespace(
            strategy=strategy, coverage=coverage, teacher_identity_sha256="teacherSHA",
            required_param_keys=("T_K", "beta"), param_bounds={"T_K": (0.0, 1000.0)})
        return types.SimpleNamespace(realized=realized, ctx=ctx)

    def _train_pool(self):
        return {"manifest_path": "/m", "parents_path": "/p", "n_frames": 4,
                "excluded_globals": [1], "n_protected_excluded": 1}

    def test_freeze_records_warranted_local_perturbation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _make_controller(root)
            legacy = {"schema_version": 1, "n_per_structure": 2, "T_K": 300.0}
            produced = self._stub_produced(strategy_kind="LOCAL_PERTURBATION", legacy=legacy)
            out = ta.freeze_augmentation_plan(c, produced, train_pool=self._train_pool())
            frozen = json.loads(out.read_text())
            self.assertTrue(frozen["augmentation_warranted"])
            self.assertEqual(frozen["strategy_kind"], "LOCAL_PERTURBATION")
            self.assertEqual(frozen["executable_projection"], legacy)
            self.assertEqual(frozen["teacher_identity_sha256"], "teacherSHA")
            self.assertEqual(frozen["protected_augmentation_parents_excluded"], [1])

    def test_freeze_records_unwarranted_existing_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = _make_controller(root)
            existing = {"schema_version": 1, "pool_path": "/pool", "n_selected": 4}
            produced = self._stub_produced(
                strategy_kind="EXISTING_POOL_SELECTION", existing_pool=existing)
            out = ta.freeze_augmentation_plan(c, produced, train_pool=self._train_pool())
            frozen = json.loads(out.read_text())
            self.assertFalse(frozen["augmentation_warranted"])
            self.assertEqual(frozen["executable_projection"], existing)


class ExecuteNoAugmentation(unittest.TestCase):
    def test_existing_pool_selection_copies_parents_without_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "fe054-exec"
            c = _make_controller(root, run_id=run_id, training_route={
                "requires_post_split_augmentation": True,
                "parameters": {"dataset": "{run_dir}/artifacts/dataset/final_train.extxyz"}})
            base = root / "labeled_train.extxyz"
            base.write_text("labeled TRAIN parents payload\n")
            # A frozen unwarranted plan on disk drives execute down the no-Teacher path.
            fp = ta.plan_path(c.run_dir, run_id)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps({
                "strategy_kind": "EXISTING_POOL_SELECTION",
                "augmentation_warranted": False,
                "plan_content_sha256": "planSHA",
                "protected_augmentation_parents_excluded": []}))
            fm = ta.execute_train_augmentation(
                c, base_dataset=str(base), base_label_manifest=None,
                teacher_config=None, reference_yaml=None)
            self.assertFalse(fm["augmentation_warranted"])
            self.assertEqual(fm["n_augmented_children"], 0)
            ft = ta.final_train_path(c.run_dir)
            self.assertEqual(ft.read_text(), base.read_text())
            ok, reason = ta.verify_finalized_augmentation(c)
            self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
