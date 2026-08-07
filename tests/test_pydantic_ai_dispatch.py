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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
