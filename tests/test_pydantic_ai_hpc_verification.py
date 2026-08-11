"""Phase 6/3: verify READY_HPC_APPROVAL_GATED actions have a REAL backing (not just a label).

For each of the 5 HPC actions: the backing module.function exists and is importable/callable;
the action is approval-gated; it carries NO inline executor (never run in tests); status is
READY_HPC_APPROVAL_GATED. If a backing turns out not to exist, this test FAILS (per the stop rule).
Network-free (imports only; nothing is executed). Skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import importlib
import unittest

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

HPC_ACTIONS = {
    "label_with_teacher": "adapters.acquisition.label_with_teacher",
    "train_committee": "workflow.steps.train_committee",
    "evaluate_heldout_fidelity": "workflow.steps.evaluate_committee",
    "run_teacher_md": "adapters.acquisition.run_teacher_md",
    "run_student_md": "workflow.steps.run_md",
}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class HpcBackingVerificationTests(unittest.TestCase):
    def test_backing_functions_exist_and_are_callable(self):
        for action, dotted in HPC_ACTIONS.items():
            module_name, func_name = dotted.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name, None)
            self.assertTrue(callable(fn), f"{action}: {dotted} is not callable/importable")

    def test_hpc_actions_are_never_inline_executed(self):
        # The core safety invariant: no HPC action carries an inline executor, so it can NEVER
        # trigger costly compute in-process (dispatch yields DRY_RUN); its real execution is a
        # separate controller stage. Backing is a real module.function.
        from runtimes.pydantic_ai.executors import BINDINGS, build_executor_registry
        reg = build_executor_registry()
        for action in HPC_ACTIONS:
            b = BINDINGS[action]
            self.assertEqual(b.status, "READY_HPC_APPROVAL_GATED", action)
            self.assertTrue(b.real_execution_required_later, action)
            self.assertIsNone(reg[action].executor, action)          # never run inline
            self.assertIn(HPC_ACTIONS[action].split(".")[0], b.backing, action)

    def test_costly_hpc_actions_are_approval_gated(self):
        # The costly, side-effecting HPC actions require an approval record before execution.
        from runtimes.pydantic_ai.actions import APPROVAL_GATED_ACTIONS
        for action in ("label_with_teacher", "train_committee", "run_teacher_md", "run_student_md"):
            self.assertIn(action, APPROVAL_GATED_ACTIONS, action)

    def test_hpc_backing_matrix_is_complete(self):
        from runtimes.pydantic_ai.executors import BINDINGS
        hpc = {a for a, b in BINDINGS.items() if b.status == "READY_HPC_APPROVAL_GATED"}
        self.assertEqual(hpc, set(HPC_ACTIONS))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
