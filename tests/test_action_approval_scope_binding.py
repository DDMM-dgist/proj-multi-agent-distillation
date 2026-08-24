"""Regression tests for action-scoped approval binding (cross-action reuse impossible).

The demonstrated defect: an approval recorded for one expensive action (Stage 3 Teacher-driven
``acquire_structures``) could be silently reused to authorize a DIFFERENT expensive action (Stage 5
``label_with_teacher``), because the durable approval was keyed only by its coarse boundary
(``costly_teacher_labeling``) and ``has_action_approval`` treated ``plan_sha256=None`` as a wildcard
that ignored the plan binding entirely.

The fix binds every grant to the exact action it authorizes (and, where applicable, its plan/decision
identity) and fails closed on every bound dimension:

  * a grant bound to a specific ``action_type`` authorizes ONLY that action;
  * a plan-scoped grant requires the exact ``plan_sha256`` -- ``None`` (or a mismatch) can never
    broaden or consume it.

These tests pin, at the controller/durable-record level:

  1. a Stage 3 acquisition approval cannot authorize Stage 5 labeling;
  2. a Stage 5 labeling approval cannot authorize another expensive action;
  3. an exact-matching approval authorizes only its intended action;
  4. a missing/None plan identity cannot broaden approval scope.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflow.controller import RunController


def _controller(root: Path) -> RunController:
    # A minimal initialized run is enough: action_approvals is additive durable state independent
    # of stage/gate semantics.
    state = {
        "schema_version": 7, "run_id": "r", "created_at": "2026-08-20T00:00:00+00:00",
        "updated_at": "2026-08-20T00:00:00+00:00", "workflow_config": "w", "artifacts": [],
        "project_dir": "p", "inputs": [], "code_revision": "x", "events": [],
        "stages": [{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}],
        "iterations": [], "recoveries": [], "pending_recovery": None,
        "runtime_attempts": [], "idempotency": {}, "action_approvals": {},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(state))
    return RunController(root)


class ActionApprovalScopeBindingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.c = _controller(Path(self._tmp.name))
        self.PLAN = "7d53aff2154c322a32ea2c3f20d83ed1cfa2cb3d8e183bc77264d32855c7e5ae"

    # (1) Stage 3 acquisition approval cannot authorize Stage 5 labeling ------------------------
    def test_acquisition_approval_cannot_authorize_labeling(self):
        # Exactly the C12 grant: costly_teacher_labeling bound to acquire_structures + plan hash.
        self.c.grant_action_approval(
            "costly_teacher_labeling", action_type="acquire_structures", plan_sha256=self.PLAN)
        # Stage 5 label_with_teacher shares the SAME boundary but is a different action and (as the
        # dispatcher does today) supplies no plan hash. It must NOT be authorized.
        self.assertFalse(
            self.c.has_action_approval("costly_teacher_labeling",
                                       action_type="label_with_teacher", plan_sha256=None))
        # Even if a caller forged the acquisition plan hash, the action mismatch alone fails closed.
        self.assertFalse(
            self.c.has_action_approval("costly_teacher_labeling",
                                       action_type="label_with_teacher", plan_sha256=self.PLAN))

    # (2) Stage 5 labeling approval cannot authorize another expensive action -------------------
    def test_labeling_approval_cannot_authorize_other_actions(self):
        self.c.grant_action_approval("costly_teacher_labeling", action_type="label_with_teacher")
        # Same boundary, different action -> denied.
        self.assertFalse(
            self.c.has_action_approval("costly_teacher_labeling", action_type="acquire_structures"))
        # Different expensive boundaries were never granted at all -> denied.
        self.assertFalse(
            self.c.has_action_approval("costly_training", action_type="train_committee"))
        self.assertFalse(
            self.c.has_action_approval("production_md", action_type="run_student_md"))

    # (3) Exact matching approval authorizes only its intended action ---------------------------
    def test_exact_match_authorizes_only_intended_action(self):
        self.c.grant_action_approval(
            "costly_teacher_labeling", action_type="acquire_structures", plan_sha256=self.PLAN)
        # Exact action + exact plan -> authorized.
        self.assertTrue(
            self.c.has_action_approval("costly_teacher_labeling",
                                       action_type="acquire_structures", plan_sha256=self.PLAN))
        # Same action, different plan -> denied.
        self.assertFalse(
            self.c.has_action_approval("costly_teacher_labeling",
                                       action_type="acquire_structures", plan_sha256="deadbeef"))
        # Different action, exact plan -> denied.
        self.assertFalse(
            self.c.has_action_approval("costly_teacher_labeling",
                                       action_type="label_with_teacher", plan_sha256=self.PLAN))

    # (4) Missing/None plan identity cannot broaden approval scope ------------------------------
    def test_none_plan_identity_cannot_broaden_scope(self):
        self.c.grant_action_approval(
            "costly_teacher_labeling", action_type="acquire_structures", plan_sha256=self.PLAN)
        # A plan-scoped grant is never consumed by a None plan hash, even with the right action.
        self.assertFalse(
            self.c.has_action_approval("costly_teacher_labeling",
                                       action_type="acquire_structures", plan_sha256=None))

    # --- provenance is preserved on the durable record ----------------------------------------
    def test_grant_records_action_and_plan_provenance(self):
        self.c.grant_action_approval(
            "costly_teacher_labeling", note="human ok", action_type="label_with_teacher")
        rec = self.c.state["action_approvals"]["costly_teacher_labeling"]
        self.assertTrue(rec["granted"])
        self.assertEqual(rec["action_type"], "label_with_teacher")
        self.assertEqual(rec["scope"], "exact_action")
        self.assertEqual(rec["note"], "human ok")
        self.assertIn("at", rec)

    # --- an unbound legacy grant still authorizes its single boundary (backward compatible) ----
    def test_unbound_grant_is_backward_compatible(self):
        self.c.grant_action_approval("costly_training")
        # No action binding recorded -> the boundary grant still authorizes (pre-existing behavior
        # relied on by tests that grant a boundary for the one action they then run).
        self.assertTrue(self.c.has_action_approval("costly_training", action_type="train_committee"))
        self.assertTrue(self.c.has_action_approval("costly_training"))


if __name__ == "__main__":
    unittest.main()
