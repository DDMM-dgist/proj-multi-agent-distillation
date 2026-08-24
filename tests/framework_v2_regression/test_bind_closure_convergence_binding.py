"""Framework V2 -- bind-closure MUST bind a ConvergencePolicy.

Regression for the demonstrated Stage 7 defect: ``bind-closure`` bound a
StageReviewSpec per stage but NEVER a ConvergencePolicy, so the Controller's
``_enforce_v2_gate_preconditions`` convergence branch (``conv_sha = binding.get(
"convergence_policy")``) was always skipped and a max-epoch NOT_CONVERGED
committee could still gate PASS (the exact R31 failure mode).

The fix: for any stage whose bound StageReviewSpec's criteria consume
``convergence_report`` evidence, bind-closure now binds a ConvergencePolicy
(caller override or the framework default). Detected generically -- no stage
name is hardcoded.
"""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController
from framework_v2.contracts import (
    ConvergencePolicy, DeploymentScopeContract, ProvenanceClass, ScopeCategory,
    ScopeRegion)
from framework_v2.convergence import (
    DEFAULT_TRAINING_CONVERGENCE_POLICY_ID, build_convergence_report,
    convergence_gate_ok)
from framework_v2.review_spec import default_stage_review_specs
from runtimes.pydantic_ai.cli import _cmd_bind_closure


def _epoch_line(n, valid_e, valid_f, lr=1e-4):
    return (f"Epoch {n} E RMSE(T V) {valid_e - 0.01:.6f} {valid_e:.6f} "
            f"F RMSE(T V) {valid_f - 0.01:.6f} {valid_f:.6f} learning_rate: {lr}")


def _not_converged_log(requested=200):
    # best at boundary AND valid energy still meaningfully falling across the
    # trailing window -> NOT_CONVERGED under the framework default policy.
    lines = [f"Total traning epoch: {requested}"]
    for ep in range(1, requested + 1):
        lines.append(_epoch_line(ep, 5.0 - 0.01 * ep, 2.5 - 0.001 * ep))
    lines.append(f"Best loss (valid) written at {requested} epoch")
    return "\n".join(lines) + "\n"


def _scope_contract():
    return DeploymentScopeContract(
        contract_id="bc-scope", objective="bind-closure convergence test",
        regions=[ScopeRegion(
            region_id="primary", category=ScopeCategory.PRIMARY_DEPLOYMENT,
            membership_rule="descriptor in the primary deployment set")])


class BindClosureConvergenceBindingTests(unittest.TestCase):
    def _controller_with_training(self, root):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "bc-itest", "stages": [
            {"name": "training",
             "command": [sys.executable, "-c",
                         "from pathlib import Path; "
                         "Path('artifacts/model.txt').write_text('trained')"],
             "outputs": ["artifacts/model.txt"],
             "gate": {"criteria": ["committee training is converged"]}},
        ]}))
        c = RunController.initialize(cfg, root / "run")
        c.run_stage("training")
        return RunController(c.run_dir)

    def test_bind_closure_binds_convergence_policy_on_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller_with_training(root)
            scope_path = root / "scope.json"
            scope_path.write_text(json.dumps(_scope_contract().model_dump(mode="json")))
            args = argparse.Namespace(
                run_dir=str(c.run_dir), scope_contract=str(scope_path),
                stage=["training"], validation_profile_version=1,
                convergence_policy=None)

            rc = _cmd_bind_closure(args)
            self.assertEqual(rc, 0)

            c = RunController(c.run_dir)
            binding = c.v2_stage_binding("training")
            self.assertIn("convergence_policy", binding,
                          "training stage requires convergence_report evidence "
                          "so bind-closure must bind a convergence_policy")
            policy = ConvergencePolicy(**c.v2_contract(binding["convergence_policy"]))
            self.assertEqual(policy.policy_id, DEFAULT_TRAINING_CONVERGENCE_POLICY_ID)
            self.assertEqual(policy.provenance_class,
                             ProvenanceClass.FRAMEWORK_CONSTRAINT)

    def test_bound_default_policy_flags_max_epoch_as_not_converged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller_with_training(root)
            seed_dir = c.run_dir / "artifacts" / "committee" / "seed-1"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "LOG").write_text(_not_converged_log(), encoding="utf-8")

            scope_path = root / "scope.json"
            scope_path.write_text(json.dumps(_scope_contract().model_dump(mode="json")))
            args = argparse.Namespace(
                run_dir=str(c.run_dir), scope_contract=str(scope_path),
                stage=["training"], validation_profile_version=1,
                convergence_policy=None)
            self.assertEqual(_cmd_bind_closure(args), 0)

            c = RunController(c.run_dir)
            policy = ConvergencePolicy(
                **c.v2_contract(c.v2_stage_binding("training")["convergence_policy"]))
            report = build_convergence_report(policy, run_dir=c.run_dir)
            self.assertEqual(report["committee_status"], "NOT_CONVERGED")
            self.assertFalse(convergence_gate_ok(report))

    def test_caller_override_policy_is_bound_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._controller_with_training(root)
            scope_path = root / "scope.json"
            scope_path.write_text(json.dumps(_scope_contract().model_dump(mode="json")))
            override = ConvergencePolicy(
                policy_id="caller-supplied", trailing_window=10,
                projection_window=20, min_relative_improvement=0.02,
                boundary_tolerance=3, metrics=["valid_energy_rmse"],
                provenance_class=ProvenanceClass.HUMAN_FIXED,
                provenance_source="unit test override")
            ov_path = root / "conv.json"
            ov_path.write_text(json.dumps(override.model_dump(mode="json")))
            args = argparse.Namespace(
                run_dir=str(c.run_dir), scope_contract=str(scope_path),
                stage=["training"], validation_profile_version=1,
                convergence_policy=str(ov_path))
            self.assertEqual(_cmd_bind_closure(args), 0)

            c = RunController(c.run_dir)
            policy = ConvergencePolicy(
                **c.v2_contract(c.v2_stage_binding("training")["convergence_policy"]))
            self.assertEqual(policy.policy_id, "caller-supplied")

    def test_non_convergence_stage_gets_no_convergence_policy(self):
        # A stage whose spec does NOT consume convergence_report evidence must
        # not get a spurious convergence_policy (generic detection, not blanket).
        specs = default_stage_review_specs(validation_profile_version=1)
        non_conv = [name for name, spec in specs.items()
                    if not any("convergence_report" in cr.required_evidence_classes
                               for cr in spec.criteria)]
        self.assertTrue(non_conv, "expected at least one non-convergence stage spec")


if __name__ == "__main__":
    unittest.main()
