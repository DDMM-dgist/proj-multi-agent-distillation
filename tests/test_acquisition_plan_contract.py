from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ase import Atoms
from ase.io import read, write


def _run(d: Path):
    state = {
        "schema_version": 7, "run_id": "r", "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
        "project_dir": "p", "inputs": [], "code_revision": "x", "events": [],
        "stages": [{"name": "acquisition", "status": "pending", "gate": "pending", "artifacts": []}],
        "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                        "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
        "recoveries": [], "pending_recovery": None,
        "runtime_attempts": [], "idempotency": {}, "action_approvals": {},
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state), encoding="utf-8")
    return d


def _protected_reference_package(root: Path, protected_atoms) -> Path:
    from workflow.integrity import sha256_file
    ref = root / "protected.xyz"
    write(str(ref), [protected_atoms])
    indices = root / "protected_indices.txt"
    indices.write_text("760\n761\n", encoding="utf-8")
    manifest = root / "protected_manifest.json"
    manifest.write_text(json.dumps({"mapping": {
        "logical_test_frames": 1,
        "matched_logical_frames": 1,
        "unmatched_logical_frames": 0,
        "protected_source_rows": 2,
        "conflicting_label_duplicates": 0,
    }}), encoding="utf-8")
    reference = root / "reference.yaml"
    reference.write_text("\n".join([
        "kind: protected-existing-dft",
        "reference_id: test-reference",
        "reference_class: ORIGINAL_TEACHER_TEST",
        "status: AVAILABLE_AND_PROTECTED",
        "logical_test_frames: 1",
        "protected_source_rows: 2",
        f"protection_manifest: {manifest}",
        f"protected_source_rows_file: {indices}",
        "duplicate_equivalent:",
        "  source_global_indices: [760, 761]",
        "  label_conflict: false",
        "prohibited_uses: [student_training, student_validation_tuning, acquisition_seed, augmentation_parent, recovery_training]",
        "structures:",
        f"  path: {ref}",
        "  logical_frames: 1",
        f"  sha256: {sha256_file(ref)}",
        "",
    ]), encoding="utf-8")
    return reference


def _atoms(x=1.0, parent="seed-pool:900", category="bulk"):
    a = Atoms("Cu", positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True)
    a.info["structure_id"] = parent
    a.info["parent_structure_id"] = parent
    if parent.startswith("seed-pool:"):
        a.info["source_global_index"] = int(parent.split(":", 1)[1])
    a.info["source_category"] = category
    return a


class AcquisitionPlanContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        protected = _atoms(0.0, "seed-pool:760")
        self.reference = _protected_reference_package(self.root, protected)
        self.seed = self.root / "seed.extxyz"
        write(str(self.seed), [_atoms(1.0, "seed-pool:900"), _atoms(2.0, "seed-pool:901")])
        self.teacher = self.root / "teacher.yaml"
        self.teacher.write_text("\n".join([
            "kind: mock",
            "calculator:",
            "  factory: adapters.mock_model.MockCheckpointCalculator",
            "  model_arg: null",
            "  kwargs:",
            "    device: cpu",
            "",
        ]), encoding="utf-8")
        self.acq_cfg = self.root / "acq.yaml"
        self.acq_cfg.write_text("\n".join([
            "kind: augment-atoms",
            "env: augment",
            "cli:",
            "  executable: augment-atoms",
            "  invocation: [augment-atoms, '{config_path}']",
            "installed_schema:",
            "  config:",
            "    units: {default: eV}",
            "    max_force: {default: 30.0}",
            "    min_separation: {default: 0.5}",
            "    max_relax_steps: {default: 20}",
            "    similarity_threshold: {default: 0.1}",
            "planning_policy:",
            "  concrete_parameters_frozen_at: approved AcquisitionPlan before acquisition execution",
            "",
        ]), encoding="utf-8")
        self.plan_exec = self.root / "augment-input.yaml"
        self.plan_exec.write_text("n_per_structure: 1\n", encoding="utf-8")
        self.plan_path = self.root / "plan.json"
        self._write_plan()
        from workflow.controller import RunController
        self.controller = RunController(_run(self.root / "run"))

    def _approve_current_plan(self):
        from runtimes.pydantic_ai.executors import acquisition_plan_sha256_from_proposal
        self.controller.grant_action_approval(
            "costly_teacher_labeling",
            plan_sha256=acquisition_plan_sha256_from_proposal(self._proposal()))

    def _write_plan(self, **updates):
        plan = {
            "schema_version": 1,
            "eligible_source_categories": ["bulk"],
            "selected_parent_structure_ids": ["seed-pool:900"],
            "selected_source_global_indices": [900],
            "n_parents": 1,
            "n_per_structure": 1,
            "T_K": 300.0,
            "beta": 0.1,
            "sigma_range_A": [0.01, 0.02],
            "cell_sigma": None,
            "seed": 123,
            "expected_output_count": 1,
            "duplicate_handling": "drop_exact_duplicates",
            "protected_reference_exclusion_report": {
                "status": "PASS",
                "reference_id": "test-reference",
                "dft_labels_used_as_selection_scores": False,
            },
            "executable_config_path": str(self.plan_exec),
        }
        plan.update(updates)
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return plan

    def _proposal(self, **params):
        p = {
            "requested_by_role": "data-curator",
            "action_type": "acquire_structures",
            "idempotency_key": "r:acquisition:001",
            "run_id": "r",
            "stage": "acquisition",
            "requested_at": "t",
            "rationale": "test",
            "parameters": {
                "acquisition_config": str(self.acq_cfg),
                "teacher_config": str(self.teacher),
                "seed_structures": str(self.seed),
                "out_path": str(self.root / "out.extxyz"),
                "manifest_path": str(self.root / "manifest.json"),
                "reference_yaml": str(self.reference),
                "selected_source_indices": [900],
                "acquisition_plan_path": str(self.plan_path),
            },
        }
        p["parameters"].update(params)
        return p

    def _dispatch(self, proposal, adapter=None):
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        if adapter is None:
            return dispatch_via_controller(proposal, controller=self.controller,
                                           registry=build_executor_registry(), mode="primary")
        with mock.patch("adapters.acquisition.acquire", adapter):
            return dispatch_via_controller(proposal, controller=self.controller,
                                           registry=build_executor_registry(), mode="primary")

    def test_missing_plan_fails_before_adapter(self):
        calls = {"n": 0}
        def adapter(*_args):
            calls["n"] += 1
        prop = self._proposal(acquisition_plan_path=None)
        out = self._dispatch(prop, adapter)
        self.assertEqual(out.status, "INVALID")
        self.assertIn("PLAN_INPUT_REQUIRED", out.reason)
        self.assertEqual(calls["n"], 0)
        self.assertNotIn("r:acquisition:001", self.controller.state.get("idempotency", {}))

    def test_incomplete_plan_fails_closed(self):
        self._write_plan(beta=None)
        data = json.loads(self.plan_path.read_text())
        data.pop("beta")
        self.plan_path.write_text(json.dumps(data), encoding="utf-8")
        self._approve_current_plan()
        out = self._dispatch(self._proposal(), lambda *_args: None)
        self.assertEqual(out.status, "EXECUTOR_ERROR")
        self.assertIn("missing required fields", out.reason)

    def test_protected_parent_and_duplicate_equivalent_rows_reject(self):
        for bad in (760, 761):
            self._write_plan(selected_source_global_indices=[bad])
            self._approve_current_plan()
            out = self._dispatch(self._proposal(selected_source_indices=[bad]), lambda *_args: None)
            self.assertEqual(out.status, "EXECUTOR_ERROR")
            self.assertIn("protected reference leakage", out.reason)

    def test_protected_logical_geometry_rejects_before_adapter(self):
        protected_seed = self.root / "protected_seed.extxyz"
        write(str(protected_seed), [_atoms(0.0, "seed-pool:900")])
        calls = {"n": 0}
        def adapter(*_args):
            calls["n"] += 1
        self._approve_current_plan()
        out = self._dispatch(self._proposal(seed_structures=str(protected_seed)), adapter)
        self.assertEqual(out.status, "EXECUTOR_ERROR")
        self.assertIn("protected logical reference geometry", out.reason)
        self.assertEqual(calls["n"], 0)

    def test_empty_parent_selection_rejects(self):
        self._write_plan(selected_parent_structure_ids=[], selected_source_global_indices=[], n_parents=0, expected_output_count=0)
        self._approve_current_plan()
        out = self._dispatch(self._proposal(selected_source_indices=[]), lambda *_args: None)
        self.assertEqual(out.status, "EXECUTOR_ERROR")
        self.assertIn("selected_parent_structure_ids", out.reason)

    def test_plan_output_count_mismatch_rejects_after_single_adapter_call(self):
        calls = {"n": 0}
        def adapter(cfg, teacher_cfg, seed_path, out_path):
            calls["n"] += 1
            write(str(out_path), [_atoms(1.0, "seed-pool:900"), _atoms(2.0, "seed-pool:900")])
            return out_path
        self._approve_current_plan()
        out = self._dispatch(self._proposal(), adapter)
        self.assertEqual(out.status, "EXECUTOR_ERROR")
        self.assertIn("output count mismatch", out.reason)
        self.assertEqual(calls["n"], 1)
        self.assertNotIn("r:acquisition:001", self.controller.state.get("idempotency", {}))


    def test_old_dict_calculator_binding_is_not_native_data2objects_schema(self):
        import importlib.util
        if importlib.util.find_spec("augment_atoms") is None or importlib.util.find_spec("data2objects") is None:
            self.skipTest("augment-atoms/data2objects package is not installed in this environment")
        import dacite
        import yaml
        from augment_atoms import AugmentConfig
        import data2objects
        payload = {
            "data": {"input": str(self.seed), "output": str(self.root / "out.extxyz")},
            "model": {"calculator": {
                "module": "nequip.ase",
                "class": "NequIPCalculator",
                "constructor": "from_compiled_model",
                "model_arg": "__positional__",
                "kwargs": {"device": "cpu"},
            }},
            "config": {
                "n_per_structure": 1,
                "T": 300.0,
                "beta": 0.1,
                "sigma_range": [0.01, 0.02],
                "seed": 123,
                "units": "eV",
                "cell_sigma": None,
                "max_force": 30.0,
                "min_separation": 0.5,
                "max_relax_steps": 1,
                "similarity_threshold": 0.1,
            },
        }
        path = self.root / "old-dict-native.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        object_dict = data2objects.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
        self.assertIsInstance(object_dict["model"]["calculator"], dict)
        with self.assertRaises(dacite.exceptions.WrongTypeError):
            dacite.from_dict(data_class=AugmentConfig, data=object_dict, config=dacite.Config(strict=True))

    def test_real_augment_atoms_shape_native_config_dispatch_and_lineage_mapping(self):
        import yaml
        self._write_plan(
            eligible_source_categories=["bulk", "void"],
            selected_parent_structure_ids=["seed-pool:900", "seed-pool:901"],
            selected_source_global_indices=[900, 901],
            n_parents=2,
            n_per_structure=1,
            expected_output_count=2,
            max_force=25.0,
            min_separation=0.7,
            max_relax_steps=9,
            similarity_threshold=0.12,
        )
        calls = {"n": 0, "command": None}

        def fake_run(command, check, cwd=None, env=None):
            calls["n"] += 1
            calls["command"] = list(command)
            self.assertTrue(check)
            self.assertIsNone(cwd)
            self.assertIn("PYTHONPATH", env or {})
            native_cfg = yaml.safe_load(self.plan_exec.read_text(encoding="utf-8"))
            self.assertEqual(set(native_cfg), {"data", "model", "config"})
            self.assertEqual(native_cfg["data"], {
                "input": str(self.seed.resolve()),
                "output": str((self.root / "out.extxyz").resolve()),
            })
            self.assertEqual(native_cfg["model"]["calculator"], {
                "+runtimes.pydantic_ai.augment_atoms_bridge.teacher_calculator": {
                    "teacher_config": str(self.teacher.resolve()),
                }
            })
            self.assertEqual(native_cfg["config"]["n_per_structure"], 1)
            self.assertEqual(native_cfg["config"]["T"], 300.0)
            self.assertEqual(native_cfg["config"]["sigma_range"], [0.01, 0.02])
            self.assertEqual(native_cfg["config"]["max_force"], 25.0)
            first = Atoms("Cu", positions=[[3, 0, 0]], cell=[10, 10, 10], pbc=True)
            first.info.update({"starting-structure": 0, "id": "native-child-0", "parent": "native-parent-uuid-0", "level": 1, "sigma": 0.01, "relax_steps": 3})
            second = Atoms("Cu", positions=[[4, 0, 0]], cell=[10, 10, 10], pbc=True)
            second.info.update({"starting-structure": 1, "id": "native-child-1", "parent": "native-parent-uuid-1", "level": 1, "sigma": 0.02, "relax_steps": 4})
            write(str(self.root / "out.extxyz"), [first, second])
            return subprocess.CompletedProcess(command, 0)

        self._approve_current_plan()
        prop = self._proposal(selected_source_indices=[900, 901])
        with mock.patch("adapters.acquisition.subprocess.run", fake_run):
            out = self._dispatch(prop)
        self.assertEqual(out.status, "EXECUTED")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(calls["command"][:4], ["conda", "run", "-n", "augment"])
        self.assertEqual(calls["command"][4:], ["augment-atoms", str(self.plan_exec.resolve())])
        frames = read(str(self.root / "out.extxyz"), index=":")
        self.assertEqual([a.info["parent_structure_id"] for a in frames], ["seed-pool:900", "seed-pool:901"])
        self.assertEqual([a.info["parent"] for a in frames], ["native-parent-uuid-0", "native-parent-uuid-1"])
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(manifest["actual_output_count"], 2)
        self.assertEqual(manifest["selected_source_global_indices"], [900, 901])
        self.assertEqual(manifest["translated_command"], calls["command"])
        self.assertIn("framework_plan_envelope", manifest)
        self.assertNotIn("selected_parent_structure_ids", manifest["native_executable_config_payload"])
        self.assertTrue((self.root / "acquisition_protection_audit.json").is_file())

    def test_protection_failure_quarantines_partial_outputs_without_idempotency(self):
        self._approve_current_plan()
        def adapter(cfg, teacher_cfg, seed_path, out_path):
            write(str(out_path), [_atoms(1.0, "seed-pool:760")])
            return out_path
        out = self._dispatch(self._proposal(), adapter)
        self.assertEqual(out.status, "EXECUTOR_ERROR")
        self.assertIn("protected reference descendant", out.reason)
        self.assertFalse((self.root / "out.extxyz").exists())
        self.assertFalse((self.root / "manifest.json").exists())
        self.assertNotIn("r:acquisition:001", self.controller.state.get("idempotency", {}))

    def test_executor_runs_at_most_once_with_idempotency(self):
        calls = {"n": 0}
        def adapter(cfg, teacher_cfg, seed_path, out_path):
            calls["n"] += 1
            write(str(out_path), [_atoms(1.0, "seed-pool:900")])
            return out_path
        self._approve_current_plan()
        prop = self._proposal()
        first = self._dispatch(prop, adapter)
        second = self._dispatch(prop, adapter)
        self.assertEqual(first.status, "EXECUTED")
        self.assertEqual(second.status, "DUPLICATE")
        self.assertEqual(calls["n"], 1)

    def test_active_config_hashes_are_advisory_not_authoritative(self):
        from runtimes.pydantic_ai.cli import _BOUND_PROPOSAL_FIELDS
        self.assertNotIn("active_config_hashes", _BOUND_PROPOSAL_FIELDS)


if __name__ == "__main__":
    unittest.main()
