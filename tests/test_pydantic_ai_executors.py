"""Phase 6: producer trusted-executor integration + honest backing matrix.

Proves AVAILABLE actions reach their EXISTING scientific implementation through the SAME
registry/controller/validator path (sandbox-primary, synthetic zero-cost input), and that
NOT_IMPLEMENTED / AVAILABLE_HPC actions are never faked as executed. Network-free; skips
without the ``pydantic`` extra.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _run(d: Path):
    state = {
        "schema_version": 7, "run_id": "r", "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
        "project_dir": "p", "inputs": [], "code_revision": "x", "events": [],
        "stages": [{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}],
        "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                        "started_at": "2026-08-07T00:00:00+00:00", "trigger": None}],
        "recoveries": [], "pending_recovery": None,
        "runtime_attempts": [], "idempotency": {}, "action_approvals": {},
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(state))
    return d




def _protected_reference_package(root: Path, protected_atoms) -> Path:
    from ase.io import write
    ref = root / "protected.xyz"
    write(str(ref), [protected_atoms])
    indices = root / "protected_indices.txt"
    indices.write_text("760\n761\n")
    manifest = root / "protected_manifest.json"
    manifest.write_text(json.dumps({"mapping": {
        "logical_test_frames": 1,
        "matched_logical_frames": 1,
        "unmatched_logical_frames": 0,
        "protected_source_rows": 2,
        "conflicting_label_duplicates": 0,
    }}))
    from workflow.integrity import sha256_file
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
    ]))
    return reference

def _prop(role, action, key, parameters):
    return {"requested_by_role": role, "action_type": action, "idempotency_key": key,
            "run_id": "r", "stage": "s", "requested_at": "t", "rationale": "sandbox",
            "parameters": parameters}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ExecutorIntegrationTests(unittest.TestCase):
    def _dispatch(self, tmp, prop, mode="primary"):
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        c = RunController(_run(Path(tmp) / "run"))
        return c, dispatch_via_controller(prop, controller=c, registry=build_executor_registry(),
                                          mode=mode)

    def test_compute_nve_drift_reaches_real_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, o = self._dispatch(tmp, _prop("simulation", "compute_nve_drift", "n1",
                                             {"energies": [0.0, 0.1, 0.2], "timestep_fs": 1.0,
                                              "n_atoms": 1}))
            self.assertEqual(o.status, "EXECUTED")
            self.assertIn("nve_drift", o.artifact["metrics"])

    def test_committee_disagreement_reaches_real_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            forces = [[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]]  # 2 seeds, 1 atom
            _, o = self._dispatch(tmp, _prop("ml-trainer", "compute_committee_disagreement", "c1",
                                             {"forces_per_seed": forces, "aggregate": "mean"}))
            self.assertEqual(o.status, "EXECUTED")
            self.assertIn("u_frame", o.artifact["metrics"])

    def test_compute_rdf_reaches_real_executor_with_synthetic_frames(self):
        from ase import Atoms
        from ase.io import write
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "cu.extxyz"
            write(str(frames), Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]],
                                     cell=[20, 20, 20], pbc=True))
            _, o = self._dispatch(tmp, _prop("simulation", "compute_rdf", "r1",
                                             {"frames_path": str(frames), "elements": ["Cu"],
                                              "r_max": 6.0, "nbins": 50}))
            self.assertEqual(o.status, "EXECUTED")
            self.assertIn("Cu-Cu", o.artifact["metrics"]["rdf_peaks"])

    def test_interface_and_reasoning_actions_are_not_executed(self):
        # scheduler interface (no HPC backend) and Analyst reasoning output carry no inline
        # executor -> DRY_RUN, never a fake EXECUTED.
        with tempfile.TemporaryDirectory() as tmp:
            # scheduler submission is approval-gated -> APPROVAL_REQUIRED (still never executed)
            _, o1 = self._dispatch(tmp, _prop("simulation", "submit_scheduler_job", "sc1", {}))
            self.assertEqual(o1.status, "APPROVAL_REQUIRED")
            self.assertFalse(o1.executed)
        with tempfile.TemporaryDirectory() as tmp:
            # reasoning output has no executor -> DRY_RUN, never a fake EXECUTED
            _, o2 = self._dispatch(tmp, _prop("analyst", "classify_root_cause", "rc1", {}))
            self.assertEqual(o2.status, "DRY_RUN")
            self.assertFalse(o2.executed)

    def test_detect_atomic_overlap_reaches_real_executor(self):
        from ase import Atoms
        from ase.io import write
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "ov.extxyz"
            write(str(frames), Atoms("Cu2", positions=[[0, 0, 0], [0.2, 0, 0]],
                                     cell=[20, 20, 20], pbc=True))  # 0.2 A -> overlap
            _, o = self._dispatch(tmp, _prop("data-curator", "detect_atomic_overlap", "ov1",
                                             {"frames_path": str(frames),
                                              "min_distance_threshold": 0.5}))
            self.assertEqual(o.status, "EXECUTED")
            self.assertEqual(o.artifact["metrics"]["n_overlapping"], 1)

    def test_sample_seed_pool_is_deterministic(self):
        from ase import Atoms
        from ase.io import write
        with tempfile.TemporaryDirectory() as tmp:
            frames_path = Path(tmp) / "pool.extxyz"
            frames = []
            for i in range(10):
                a = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
                a.info["structure_id"] = f"s{i}"
                a.info["parent_structure_id"] = f"p{i % 3}"
                frames.append(a)
            write(str(frames_path), frames)
            p = _prop("data-curator", "sample_seed_pool", "sp1",
                      {"frames_path": str(frames_path), "count": 4, "seed": 20260804})
            _, o1 = self._dispatch(tmp, p)
            _, o2 = self._dispatch(tmp, {**p, "idempotency_key": "sp2"})
            self.assertEqual(o1.status, "EXECUTED")
            self.assertEqual(o1.artifact["metrics"]["selected_ids"],
                             o2.artifact["metrics"]["selected_ids"])  # same seed -> same selection
            self.assertEqual(o1.artifact["metrics"]["selected_count"], 4)


    def test_training_with_unprotected_dataset_passes_production_guard(self):
        from ase import Atoms
        from ase.io import write
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
            reference_yaml = _protected_reference_package(root, protected)
            allowed = Atoms("Cu", positions=[[1, 0, 0]], cell=[10, 10, 10], pbc=True)
            allowed.info["structure_id"] = "allowed"
            allowed.info["parent_structure_id"] = "seed-pool:900"
            dataset = root / "allowed.extxyz"
            write(str(dataset), [allowed, allowed.copy(), allowed.copy()])
            student_cfg = root / "student.yaml"
            student_cfg.write_text("kind: mock\ncommittee:\n  seeds: [1]\n")
            c = RunController(_run(root / "run"))
            c.grant_action_approval("costly_training")
            prop = _prop("ml-trainer", "train_committee", "guard-ok", {
                "student_config": str(student_cfg), "dataset": str(dataset),
                "output_dir": str(root / "committee"),
                "manifest_path": str(root / "committee.json"),
                "reference_yaml": str(reference_yaml),
            })
            out = dispatch_via_controller(prop, controller=c, registry=build_executor_registry(),
                                          mode="primary")
            self.assertEqual(out.status, "EXECUTED")

    def test_training_blocks_protected_geometry_before_execution(self):
        from ase import Atoms
        from ase.io import write
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
            reference_yaml = _protected_reference_package(root, protected)
            protected.info["structure_id"] = "protected"
            protected.info["parent_structure_id"] = "seed-pool:760"
            dataset = root / "blocked.extxyz"
            write(str(dataset), [protected, protected.copy(), protected.copy()])
            student_cfg = root / "student.yaml"
            student_cfg.write_text("kind: mock\ncommittee:\n  seeds: [1]\n")
            c = RunController(_run(root / "run"))
            c.grant_action_approval("costly_training")
            manifest = root / "committee.json"
            prop = _prop("ml-trainer", "train_committee", "guard-block", {
                "student_config": str(student_cfg), "dataset": str(dataset),
                "output_dir": str(root / "committee"), "manifest_path": str(manifest),
                "reference_yaml": str(reference_yaml),
            })
            out = dispatch_via_controller(prop, controller=c, registry=build_executor_registry(),
                                          mode="primary")
            self.assertEqual(out.status, "EXECUTOR_ERROR")
            self.assertFalse(manifest.exists())

    def test_completion_validator_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "committee.json"
            manifest.write_text(json.dumps({"models": []}))  # 0 seeds != expected 4
            _, o = self._dispatch(tmp, _prop("ml-trainer", "validate_training_completion", "vt1",
                                             {"committee_manifest": str(manifest),
                                              "expected_seeds": 4}))
            self.assertEqual(o.status, "INVALID")   # fail-closed, not a passing artifact
            self.assertFalse(o.executed)

    def test_hpc_action_requires_approval_then_reaches_real_train_wrapper(self):
        from ase import Atoms
        from ase.io import write
        from workflow.controller import RunController
        from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
        from runtimes.pydantic_ai.executors import build_executor_registry
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = RunController(_run(root / "run"))
            dataset = root / "train.extxyz"
            frames = []
            for i in range(3):
                a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
                a.info["parent_structure_id"] = f"p{i}"
                a.info["structure_id"] = f"s{i}"
                frames.append(a)
            write(str(dataset), frames)
            student_cfg = root / "student.yaml"
            student_cfg.write_text("kind: mock\ncommittee:\n  seeds: [1, 2]\n")
            prop = _prop("ml-trainer", "train_committee", "t1", {
                "student_config": str(student_cfg), "dataset": str(dataset),
                "output_dir": str(root / "committee"),
                "manifest_path": str(root / "committee.json"),
            })
            reg = build_executor_registry()
            o1 = dispatch_via_controller(prop, controller=c, registry=reg, mode="primary")
            self.assertEqual(o1.status, "APPROVAL_REQUIRED")
            c.grant_action_approval("costly_training")
            o2 = dispatch_via_controller(prop, controller=c, registry=reg, mode="primary")
            self.assertEqual(o2.status, "EXECUTED")
            self.assertTrue(Path(o2.artifact["path"]).is_file())
            self.assertEqual(len(o2.artifact["manifest"]["models"]), 2)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class BackingMatrixTests(unittest.TestCase):
    def test_every_role_action_has_a_binding(self):
        from runtimes.pydantic_ai.actions import ROLE_ALLOWED_ACTIONS
        from runtimes.pydantic_ai.executors import BINDINGS
        all_actions = set().union(*ROLE_ALLOWED_ACTIONS.values())
        self.assertEqual(set(BINDINGS), all_actions)

    def test_required_heavy_actions_have_registry_executors(self):
        from runtimes.pydantic_ai.executors import BINDINGS, build_executor_registry
        reg = build_executor_registry()
        required = {"build_teacher_baseline", "validate_teacher_reference",
                    "acquire_structures", "label_with_teacher", "run_teacher_md",
                    "train_committee", "evaluate_heldout_fidelity", "run_student_md"}
        for action, b in BINDINGS.items():
            if b.status == "READY_EXECUTOR" or action in required:
                self.assertIsNotNone(reg[action].executor, action)
            elif b.status in {"READY_INTERFACE_BACKEND_NOT_CONFIGURED", "READY_REASONING_OUTPUT"}:
                self.assertIsNone(reg[action].executor, action)

    def test_no_action_is_left_not_implemented(self):
        from runtimes.pydantic_ai.executors import BINDINGS
        self.assertEqual([a for a, b in BINDINGS.items() if b.status == "NOT_IMPLEMENTED"], [])

    def test_status_counts_are_honest(self):
        from collections import Counter
        from runtimes.pydantic_ai.executors import BINDINGS
        counts = Counter(b.status for b in BINDINGS.values())
        # 36: 33 (through the v21 autonomous Teacher-validation-planning revision, incl.
        # build_split_membership_population) + 3 Stage-10 deployment producers
        # (build_stage10_deployment_plan, resolve_deployment_checkpoint, build_deployment_context) + 1 FE-049 species-mapping
        # exposure producer (validate_species_mapping_consistency).
        self.assertEqual(counts["READY_EXECUTOR"], 37)
        self.assertEqual(counts["READY_HPC_APPROVAL_GATED"], 9)
        self.assertEqual(counts["READY_INTERFACE_BACKEND_NOT_CONFIGURED"], 3)
        self.assertEqual(counts["READY_REASONING_OUTPUT"], 1)
        self.assertEqual(counts.get("NOT_IMPLEMENTED", 0), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
