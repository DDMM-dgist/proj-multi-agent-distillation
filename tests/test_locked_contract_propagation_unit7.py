"""UNIT 7 tests: locked-contract PROPAGATION (not just validation) in production producers.

The write-once validation_contract freezes three scientific-scope components before any Student
result exists: ``teacher_applicability_domain`` (the deployment domain), ``validation_scope``, and
``dataset_split_policy``. Earlier work proved the *validators* reject a report whose value drifts
from the locked contract (``test_validation_target_lock``). These tests prove the complementary
*producer* invariant that closes the re-authoring surface:

  * A contract-CONSUMING executor must SOURCE the locked value VERBATIM from the run's own frozen
    contract -- it must never let a proposal (or a later recovery attempt) author the value and
    merely hope the downstream validator catches the drift.

Concretely:

  * Stage-6 dataset_split (``_exec_generate_group_split``) -- exercised end to end (pure Python,
    no Teacher/torch): a proposal that supplies CONFLICTING split parameters is overridden by the
    locked ``dataset_split_policy``; a proposal that OMITS them still gets the locked values, not
    the code defaults.
  * A static no-dangling-contract integrity check (UNIT 7 / task #552): every contract-consuming
    producer (Stage-1 teacher_baseline, Stage-4 data_coverage, Stage-6 dataset_split) resolves the
    run's frozen contract AND threads it into its self-validator, so a re-authored locked value
    fails closed in-process rather than silently propagating.
"""
from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

from runtimes.pydantic_ai import executors
from runtimes.pydantic_ai.executors import _exec_generate_group_split
from workflow.controller import RunController


TEACHER_DOMAIN = {"structure_classes": ["liquid", "crystal"], "temperature_range_K": [300, 1500]}
VALIDATION_SCOPE = {"shared_md_protocol": "nvt-1000K-v1", "checks": ["diffusion", "rdf"]}
LOCKED_SPLIT_POLICY = {"seed": 7, "validation_fraction": 0.2, "test_fraction": 0.2,
                       "grouping_key": "parent_structure_id"}


def _contract_components(split_policy=None):
    return {"teacher_applicability_domain": TEACHER_DOMAIN,
            "validation_scope": VALIDATION_SCOPE,
            "dataset_split_policy": split_policy or LOCKED_SPLIT_POLICY}


def _controller_with_contract(root, split_policy=None):
    cfg = root / "workflow.yaml"
    cfg.write_text(yaml.safe_dump(
        {"run_id": "unit7", "stages": [
            {"name": "teacher_baseline", "command": None, "outputs": ["artifacts/tb.txt"],
             "gate": {"criteria": ["complete"]}}]},
        sort_keys=False))
    controller = RunController.initialize(cfg, root / "run")
    controller.establish_validation_contract(_contract_components(split_policy))
    return controller, controller.run_dir / "validation_contract.json"


def _grouped_dataset(path):
    """Six frames across three independent parent groups -- the minimum for group splitting."""
    frames = []
    for group in range(3):
        for child in range(2):
            atoms = Atoms("Cu", positions=[[group + child * 0.01, 0, 0]],
                          cell=[10, 10, 10], pbc=True)
            atoms.info.update(structure_id=f"g{group}-c{child}",
                              parent_structure_id=f"g{group}")
            frames.append(atoms)
    write(str(path), frames)
    return path


class Stage6SplitPolicyPropagationTests(unittest.TestCase):
    def test_locked_policy_overrides_a_conflicting_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, contract_path = _controller_with_contract(root)
            dataset = _grouped_dataset(root / "src.extxyz")
            # A proposal that tries to RE-AUTHOR the split parameters (seed/fractions) different
            # from the locked policy. Under the propagation invariant the locked values win and the
            # stage does NOT fail closed (the produced params match the contract by construction).
            result = _exec_generate_group_split({"parameters": {
                "dataset": str(dataset),
                "output_dir": str(root / "splits"),
                "manifest": str(root / "split_manifest.json"),
                "seed": 999, "validation_fraction": 0.4, "test_fraction": 0.3,
                "validation_contract_path": str(contract_path),
            }})
            manifest = result["manifest"]
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["validation_fraction"], 0.2)
            self.assertEqual(manifest["test_fraction"], 0.2)
            self.assertEqual(manifest["grouping_key"], "parent_structure_id")

    def test_locked_policy_supplies_values_a_proposal_omits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, contract_path = _controller_with_contract(root)
            dataset = _grouped_dataset(root / "src.extxyz")
            # Proposal omits split parameters entirely -> propagation supplies the locked values,
            # NOT the code defaults (default seed is 2026, locked seed is 7).
            result = _exec_generate_group_split({"parameters": {
                "dataset": str(dataset),
                "output_dir": str(root / "splits"),
                "manifest": str(root / "split_manifest.json"),
                "validation_contract_path": str(contract_path),
            }})
            self.assertEqual(result["manifest"]["seed"], 7)

    def test_without_a_contract_proposal_values_are_used(self):
        # No frozen contract bound -> legacy behaviour: the proposal's own params (or defaults)
        # apply. This proves propagation is gated on a real locked contract, not always-on.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _grouped_dataset(root / "src.extxyz")
            result = _exec_generate_group_split({"parameters": {
                "dataset": str(dataset),
                "output_dir": str(root / "splits"),
                "manifest": str(root / "split_manifest.json"),
                "seed": 123, "validation_fraction": 0.2, "test_fraction": 0.2,
            }})
            self.assertEqual(result["manifest"]["seed"], 123)


class NoDanglingContractProducerTests(unittest.TestCase):
    """Task #552: every contract-consuming producer resolves the run's frozen contract AND threads
    it into its self-validator, so a re-authored locked value fails closed in-process. This is a
    static (source-level) integrity check so it holds even where a live Teacher run is too costly to
    exercise (Stage-1 teacher_baseline).
    """

    def _assert_binds_contract(self, func, *, locked_component, validator_call):
        src = inspect.getsource(func)
        self.assertIn("_resolve_validation_contract_path", src,
                      f"{func.__name__} must resolve the run's frozen validation contract")
        self.assertIn(locked_component, src,
                      f"{func.__name__} must source the locked {locked_component} component")
        self.assertIn(validator_call, src,
                      f"{func.__name__} must self-validate via {validator_call}")
        # The self-validator call must be threaded with the resolved contract path -- otherwise the
        # locked hash-check is inert and a re-authored value would propagate unchecked.
        self.assertIn("validation_contract_path=", src,
                      f"{func.__name__} must pass validation_contract_path into its self-validator")

    def test_teacher_baseline_producer_binds_locked_domain(self):
        self._assert_binds_contract(
            executors._exec_build_teacher_baseline,
            locked_component="teacher_applicability_domain",
            validator_call="validate_teacher_baseline_report(")

    def test_data_coverage_producer_binds_locked_domain(self):
        self._assert_binds_contract(
            executors._exec_build_data_coverage_report,
            locked_component="teacher_applicability_domain",
            validator_call="validate_data_coverage_report(")

    def test_split_producer_binds_locked_split_policy(self):
        self._assert_binds_contract(
            executors._exec_generate_group_split,
            locked_component="dataset_split_policy",
            validator_call="split_dataset(")

    def test_teacher_baseline_no_longer_takes_domain_only_from_proposal(self):
        # The specific UNIT-7 blocker that was fixed: the Stage-1 producer must NOT source the
        # deployment_domain unconditionally from the proposal params. The locked-contract branch
        # must come FIRST; the proposal assignment may survive only as a guarded fallback for the
        # no-contract (legacy) path.
        src = inspect.getsource(executors._exec_build_teacher_baseline)
        self.assertIn("locked_deployment_domain", src)
        propagation_branch = "if locked_deployment_domain is not None:"
        fallback_branch = 'elif "deployment_domain" in p:'
        self.assertIn(propagation_branch, src)
        self.assertIn(fallback_branch, src)
        # the propagation branch must precede the proposal fallback
        self.assertLess(src.index(propagation_branch), src.index(fallback_branch))


if __name__ == "__main__":
    unittest.main()
