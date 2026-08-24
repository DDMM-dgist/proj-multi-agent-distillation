"""Phase 4: enforcement of role manifests + capability registry in the dispatch path.

Proves default-deny at every step. Network-free; skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import unittest

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _prop(role, action, key="k1", run_id="r1", **kw):
    p = {"requested_by_role": role, "action_type": action, "idempotency_key": key,
         "run_id": run_id, "stage": "s", "requested_at": "t", "rationale": "because"}
    p.update(kw)
    return p


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class DispatchEnforcementTests(unittest.TestCase):
    def setUp(self):
        from runtimes.pydantic_ai.dispatch import (
            default_registry, InMemoryApprovalStore, InMemoryIdempotencyStore)
        self.registry = default_registry()
        self.approvals = InMemoryApprovalStore()
        self.idem = InMemoryIdempotencyStore()

    def _run(self, proposal, mode="dry_run"):
        from runtimes.pydantic_ai.dispatch import authorize_and_execute
        return authorize_and_execute(proposal, registry=self.registry, approvals=self.approvals,
                                     idempotency=self.idem, mode=mode)

    # --- the required cross-role denials -------------------------------------
    def test_judge_requesting_producer_action_is_denied(self):
        o = self._run(_prop("judge", "train_committee"))
        self.assertEqual(o.status, "DENIED")

    def test_analyst_requesting_simulation_action_is_denied(self):
        o = self._run(_prop("analyst", "run_teacher_md"))
        self.assertEqual(o.status, "DENIED")

    def test_analyst_requesting_arbitrary_code_is_denied(self):
        o = self._run(_prop("analyst", "exec_arbitrary_python"))
        self.assertEqual(o.status, "DENIED")

    def test_data_curator_requesting_ml_trainer_action_is_denied(self):
        o = self._run(_prop("data-curator", "train_committee"))
        self.assertEqual(o.status, "DENIED")

    def test_ml_trainer_requesting_scheduler_action_is_denied(self):
        o = self._run(_prop("ml-trainer", "submit_scheduler_job"))
        self.assertEqual(o.status, "DENIED")

    def test_simulation_requesting_scheduler_script_is_blocked_capability(self):
        o = self._run(_prop("simulation", "generate_scheduler_script"))
        self.assertEqual(o.status, "BLOCKED_CAPABILITY")

    def test_simulation_requesting_arbitrary_dft_is_not_executed(self):
        o = self._run(_prop("simulation", "run_dft"), mode="primary")
        self.assertIn(o.status, ("APPROVAL_REQUIRED", "BLOCKED_CAPABILITY"))
        self.assertFalse(o.executed)

    # --- capability fail-closed ---------------------------------------------
    def test_unavailable_capabilities_are_fail_closed(self):
        for action in ("compute_eos", "compute_mechanics", "compute_ring_statistics",
                       "compute_sq_fsdp", "compute_adf", "compute_channel_d", "fine_tune_teacher"):
            o = self._run(_prop("simulation", action), mode="primary")
            self.assertEqual(o.status, "BLOCKED_CAPABILITY", action)
            self.assertFalse(o.executed, action)

    def test_action_not_in_manifest_is_default_denied(self):
        o = self._run(_prop("data-curator", "totally_made_up_action"))
        self.assertEqual(o.status, "DENIED")

    # --- approval enforcement -----------------------------------------------
    def test_approval_gated_action_without_approval_is_blocked(self):
        o = self._run(_prop("data-curator", "label_with_teacher"), mode="primary")
        self.assertEqual(o.status, "APPROVAL_REQUIRED")
        self.assertFalse(o.executed)

    def test_approval_gated_action_with_approval_passes_gate(self):
        self.approvals.grant("r1", "costly_teacher_labeling")
        o = self._run(_prop("data-curator", "label_with_teacher"), mode="primary")
        # no inline executor registered by default -> DRY_RUN (but the approval gate passed)
        self.assertEqual(o.status, "DRY_RUN")

    # --- typed parameter / artifact validation ------------------------------
    def test_invalid_parameters_are_rejected(self):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        self.registry["inspect_dataset"] = ActionDescriptor(
            action_type="inspect_dataset", role="data-curator",
            param_validator=lambda p: (False, "missing dataset path"))
        o = self._run(_prop("data-curator", "inspect_dataset"))
        self.assertEqual(o.status, "INVALID")
        self.assertIn("dataset path", o.reason)

    # --- idempotency ---------------------------------------------------------
    def test_duplicate_idempotency_key_is_not_re_executed(self):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        calls = []
        self.registry["inspect_dataset"] = ActionDescriptor(
            action_type="inspect_dataset", role="data-curator",
            executor=lambda p: (calls.append(1) or {"path": "runs/x/summary.json", "sha256": "z"}))
        first = self._run(_prop("data-curator", "inspect_dataset", key="dup1"), mode="primary")
        second = self._run(_prop("data-curator", "inspect_dataset", key="dup1"), mode="primary")
        self.assertEqual(first.status, "EXECUTED")
        self.assertEqual(second.status, "DUPLICATE")
        self.assertEqual(len(calls), 1)  # executed exactly once

    # --- trusted executor happy path ----------------------------------------
    def test_trusted_executor_runs_only_after_all_checks(self):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        self.registry["detect_duplicates"] = ActionDescriptor(
            action_type="detect_duplicates", role="data-curator",
            executor=lambda p: {"path": "runs/x/dupes.json", "sha256": "abc", "n_duplicates": 0})
        o = self._run(_prop("data-curator", "detect_duplicates", key="dd1"), mode="primary")
        self.assertEqual(o.status, "EXECUTED")
        self.assertEqual(o.artifact["n_duplicates"], 0)
        self.assertEqual(o.executor, "<lambda>")

    def test_dry_run_never_executes(self):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        calls = []
        self.registry["detect_duplicates"] = ActionDescriptor(
            action_type="detect_duplicates", role="data-curator",
            executor=lambda p: calls.append(1))
        o = self._run(_prop("data-curator", "detect_duplicates"), mode="dry_run")
        self.assertEqual(o.status, "DRY_RUN")
        self.assertEqual(calls, [])

    # --- Defect-2 forensic fix: Teacher-evidence COMPUTE gate is never relaxed by label provenance
    # build_teacher_baseline/validate_teacher_reference run REAL Teacher forward passes on GPU to
    # build their report/comparison. That GPU compute IS the effect costly_teacher_labeling guards,
    # independently of whether the run also grows the training corpus. A "creates no new
    # DFT/protected-reference labels" declaration is about corpus growth, NOT compute, so it must
    # NOT relax the boundary -- this is exactly the bypass that let a fresh 9,295-frame Teacher
    # baseline dispatch on GPU with action_approvals={}. The ONLY relaxation for validate_teacher_
    # reference is a bound prior verified historical_report (the executor's verified-reuse path,
    # which recomputes metrics from existing predictions and runs NO fresh Teacher).
    def test_build_teacher_baseline_with_no_new_labels_STILL_requires_approval(self):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        self.registry["build_teacher_baseline"] = ActionDescriptor(
            action_type="build_teacher_baseline", role="simulation",
            approval_boundary="costly_teacher_labeling",
            executor=lambda p: {"path": "runs/x/teacher_baseline.json", "sha256": "z"})
        o = self._run(_prop("simulation", "build_teacher_baseline",
                            parameters={"deployment_domain": {
                                "dft_labels_used": False,
                                "protected_reference_labels_used": False}}),
                      mode="primary")
        self.assertEqual(o.status, "APPROVAL_REQUIRED")
        self.assertFalse(o.executed)

    def test_validate_teacher_reference_fresh_requires_approval(self):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        self.registry["validate_teacher_reference"] = ActionDescriptor(
            action_type="validate_teacher_reference", role="simulation",
            approval_boundary="costly_teacher_labeling",
            executor=lambda p: {"path": "runs/x/reference_report.json", "sha256": "z"})
        o = self._run(_prop("simulation", "validate_teacher_reference",
                            parameters={"dft_labels_used": False,
                                       "protected_reference_labels_used": False}),
                      mode="primary")
        self.assertEqual(o.status, "APPROVAL_REQUIRED")
        self.assertFalse(o.executed)

    def test_validate_teacher_reference_verified_reuse_needs_no_approval(self):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        self.registry["validate_teacher_reference"] = ActionDescriptor(
            action_type="validate_teacher_reference", role="simulation",
            approval_boundary="costly_teacher_labeling",
            executor=lambda p: {"path": "runs/x/reference_report.json", "sha256": "z"})
        o = self._run(_prop("simulation", "validate_teacher_reference",
                            parameters={"historical_report": {
                                "reference_id": "prior", "metrics": {}}}),
                      mode="primary")
        self.assertEqual(o.status, "EXECUTED")
        self.assertTrue(o.executed)

    def test_build_teacher_baseline_without_declared_flags_still_requires_approval(self):
        # Fail-closed: absence of the flags is never treated as proof of "no new labels".
        o = self._run(_prop("simulation", "build_teacher_baseline"), mode="primary")
        self.assertEqual(o.status, "APPROVAL_REQUIRED")
        self.assertFalse(o.executed)

    def test_build_teacher_baseline_with_true_flags_still_requires_approval(self):
        o = self._run(_prop("simulation", "build_teacher_baseline",
                            parameters={"deployment_domain": {
                                "dft_labels_used": True,
                                "protected_reference_labels_used": False}}),
                      mode="primary")
        self.assertEqual(o.status, "APPROVAL_REQUIRED")
        self.assertFalse(o.executed)

    def test_label_with_teacher_still_requires_approval_even_with_safe_looking_params(self):
        # label_with_teacher genuinely creates new labels -- the exemption must never extend to
        # it even if it happens to carry a similarly-shaped parameters dict.
        o = self._run(_prop("data-curator", "label_with_teacher",
                            parameters={"deployment_domain": {
                                "dft_labels_used": False,
                                "protected_reference_labels_used": False}}),
                      mode="primary")
        self.assertEqual(o.status, "APPROVAL_REQUIRED")
        self.assertFalse(o.executed)

    def test_acquire_structures_still_requires_approval_even_with_safe_looking_params(self):
        o = self._run(_prop("data-curator", "acquire_structures",
                            parameters={"dft_labels_used": False,
                                       "protected_reference_labels_used": False}),
                      mode="primary")
        # acquire_structures fails closed on the missing AcquisitionPlan before the approval
        # check is even reached (dispatch.py checks plan_sha256 first for this action).
        self.assertEqual(o.status, "INVALID")
        self.assertFalse(o.executed)

    def test_resolve_action_approval_boundary_unit(self):
        from runtimes.pydantic_ai.actions import resolve_action_approval_boundary
        # build_teacher_baseline always runs the Teacher on GPU -> never relaxed by label provenance.
        self.assertEqual(resolve_action_approval_boundary(
            "build_teacher_baseline", "costly_teacher_labeling",
            {"deployment_domain": {"dft_labels_used": False,
                                   "protected_reference_labels_used": False}}),
            "costly_teacher_labeling")
        self.assertEqual(resolve_action_approval_boundary(
            "build_teacher_baseline", "costly_teacher_labeling", {}),
            "costly_teacher_labeling")
        # validate_teacher_reference: fresh -> gated; verified-reuse (historical_report) -> relaxed.
        self.assertEqual(resolve_action_approval_boundary(
            "validate_teacher_reference", "costly_teacher_labeling",
            {"dft_labels_used": False, "protected_reference_labels_used": False}),
            "costly_teacher_labeling")
        self.assertIsNone(resolve_action_approval_boundary(
            "validate_teacher_reference", "costly_teacher_labeling",
            {"historical_report": {"reference_id": "prior", "metrics": {}}}))
        self.assertEqual(resolve_action_approval_boundary(
            "label_with_teacher", "costly_teacher_labeling",
            {"dft_labels_used": False, "protected_reference_labels_used": False}),
            "costly_teacher_labeling")
        self.assertEqual(resolve_action_approval_boundary(
            "acquire_structures", "costly_teacher_labeling",
            {"dft_labels_used": False, "protected_reference_labels_used": False}),
            "costly_teacher_labeling")
        # geometry-only acquisition (framework-classified) is the one genuine relaxation.
        self.assertIsNone(resolve_action_approval_boundary(
            "acquire_structures", "costly_teacher_labeling",
            {"performs_teacher_inference": False}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
