"""Forensic regressions for the two demonstrated ffv4f defects plus the Defect-1 binding lesson.

These pin, hermetically (mock runtime only; no real Teacher / GPU / network), three properties the
ffv4f run violated:

  * DEFECT 2 -- COSTLY_ACTION_APPROVAL_BYPASS. A stage whose action carries the
    ``costly_teacher_labeling`` boundary (real Teacher forward passes on GPU) must NOT dispatch its
    executor while ``action_approvals`` is empty: ``run_campaign`` must pause at
    WAITING_FOR_HUMAN_APPROVAL with the costly executor invoked ZERO times, and must dispatch it
    ONLY after an explicit human action approval is recorded. Crucially, the SEPARATE
    ``_teacher_validation_downstream_reliance_gap`` being satisfied (a plan that already selects a
    predictive-fidelity component, so that gate returns "proceed") must NOT by itself authorize the
    costly compute -- scientific-adequacy / downstream-reliance is orthogonal to permission to spend
    GPU Teacher inference. (ffv4f dispatched a fresh 9,295-frame Teacher baseline with
    ``action_approvals={}`` precisely because the compute boundary had been wrongly relaxed.)

  * DEFECT 3 -- LAZY ACQUISITION PLANNING (execution order). The autonomous acquisition planner
    (whose ``build_context`` runs the expensive FPS / coverage / population-sizing geometry) must
    NOT run before the campaign has reached -- and cleared the costly-action boundary of -- the
    acquisition stage. It must be consulted ONLY on the loop turn whose next-eligible stage is
    genuinely the acquisition stage.

  * DEFECT 1 -- RECOVERED_HOLDOUT must be a CONTROLLER-BOUND INPUT. When the committed
    ``teacher_validation_plan`` selects ``ORIGINAL_HELDOUT_FIDELITY``, the recovered-original-holdout
    reference MUST be resolvable from the controller-input-derived reference map; a reference that is
    NOT present there (e.g. supplied only via a stage's ``reference_yaml`` parameter) fails closed.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


# =====================================================================================
# DEFECT 2 -- costly Teacher-inference action approval, at the real run_campaign level
# =====================================================================================
class _SpyExecutor:
    """A trusted-executor stand-in that records every invocation and, when it IS allowed to run,
    writes the stage's single declared output so the stage can complete + gate normally."""

    def __init__(self, run_dir: Path, output_rel: str):
        self.calls = 0
        self._run_dir = run_dir
        self._output_rel = output_rel
        self.__name__ = "spy_build_teacher_baseline"

    def __call__(self, proposal):
        self.calls += 1
        out = self._run_dir / self._output_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"teacher_baseline": "spy-materialized"}\n')
        return {"path": str(out), "sha256": "spy"}


def _costly_teacher_baseline_workflow(root: Path) -> Path:
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "costly-boundary-regression",
        "stages": [{
            "name": "teacher_baseline",
            "command": None,
            "outputs": ["artifacts/teacher_baseline.json"],
            "gate": {"criteria": ["the teacher baseline report is complete"]},
            "pydantic_ai": {
                "role": "simulation", "action": "build_teacher_baseline",
                "approval_boundary": "costly_teacher_labeling",
                "idempotency_key": "costly-boundary-regression:teacher_baseline:001",
                "parameters": {"structures_path": "local_inputs/structures.xyz"},
            },
        }],
    }, sort_keys=False))
    return workflow


class CostlyTeacherApprovalBoundaryCampaignTests(unittest.TestCase):
    def _patched_registry_factory(self, spy):
        from runtimes.pydantic_ai.dispatch import ActionDescriptor
        from runtimes.pydantic_ai.executors import build_executor_registry

        def _factory():
            reg = build_executor_registry()
            reg["build_teacher_baseline"] = ActionDescriptor(
                action_type="build_teacher_baseline", role="simulation",
                approval_boundary="costly_teacher_labeling", executor=spy)
            return reg
        return _factory

    def test_no_approval_pauses_and_never_dispatches_teacher_executor(self):
        import runtimes.pydantic_ai.executors as executors_mod
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        original = executors_mod.build_executor_registry
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "run"
                RunController.initialize(_costly_teacher_baseline_workflow(root), run_dir)
                c = RunController(run_dir)
                self.assertEqual(c.state.get("action_approvals", {}), {})

                spy = _SpyExecutor(run_dir, "artifacts/teacher_baseline.json")
                executors_mod.build_executor_registry = self._patched_registry_factory(spy)

                result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                          auto_mock_judges=True, max_iterations=5)

                self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL,
                                 result.message)
                self.assertEqual(result.exit_code, cli.EXIT_APPROVAL_REQUIRED)
                self.assertEqual(spy.calls, 0,
                                 "the costly Teacher executor must NEVER be dispatched without an "
                                 "explicit human action approval")
                c = RunController(run_dir)
                self.assertEqual(c.stage("teacher_baseline")["status"], "pending")
                self.assertFalse(
                    (run_dir / "artifacts" / "teacher_baseline.json").exists(),
                    "no Teacher baseline artifact may be materialized before approval")
        finally:
            executors_mod.build_executor_registry = original

    def test_dispatch_permitted_only_after_explicit_approval(self):
        import runtimes.pydantic_ai.executors as executors_mod
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        original = executors_mod.build_executor_registry
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "run"
                RunController.initialize(_costly_teacher_baseline_workflow(root), run_dir)
                c = RunController(run_dir)

                spy = _SpyExecutor(run_dir, "artifacts/teacher_baseline.json")
                executors_mod.build_executor_registry = self._patched_registry_factory(spy)

                # First pass: no approval -> pause, no dispatch.
                first = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                         auto_mock_judges=True, max_iterations=5)
                self.assertEqual(first.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL)
                self.assertEqual(spy.calls, 0)

                # Record an explicit human approval for the costly boundary, then resume.
                c = RunController(run_dir)
                c.grant_action_approval("costly_teacher_labeling", note="human authorizes baseline")
                result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                          auto_mock_judges=True, max_iterations=5)

                self.assertEqual(result.outcome, cli.CAMPAIGN_COMPLETED, result.message)
                self.assertGreaterEqual(spy.calls, 1,
                                        "after explicit approval the costly executor may dispatch")
                c = RunController(run_dir)
                self.assertEqual(c.stage("teacher_baseline")["gate"], "PASS")
        finally:
            executors_mod.build_executor_registry = original

    def test_downstream_reliance_adequacy_does_not_authorize_costly_compute(self):
        """The reliance gate and the compute-approval gate are orthogonal: a plan that selects a
        predictive-fidelity component makes ``_teacher_validation_downstream_reliance_gap`` return
        None (proceed), yet ``resolve_action_approval_boundary`` must STILL require
        ``costly_teacher_labeling`` for the Teacher-inference action."""
        from runtimes.pydantic_ai.actions import resolve_action_approval_boundary
        from runtimes.pydantic_ai import cli

        class _StubController:
            def __init__(self, plan):
                self.state = {"teacher_validation_plan": plan}

        plan = {"selected_components": ["ORIGINAL_HELDOUT_FIDELITY", "OPERATIONAL_ROBUSTNESS"]}
        stub = _StubController(plan)
        stage_cfg = {"pydantic_ai": {"role": "simulation", "action": "build_teacher_baseline",
                                     "approval_boundary": "costly_teacher_labeling"}}

        # Reliance gate is satisfied (adequate) -> it would let dispatch proceed...
        self.assertIsNone(
            cli._teacher_validation_downstream_reliance_gap(stub, "teacher_baseline", stage_cfg))
        # ...but the compute-approval boundary is UNCHANGED: costly GPU Teacher work still gated.
        self.assertEqual(
            resolve_action_approval_boundary("build_teacher_baseline",
                                             "costly_teacher_labeling", {}),
            "costly_teacher_labeling")


# =====================================================================================
# DEFECT 3 -- lazy acquisition planning (execution order)
# =====================================================================================
class _PlannerSpy:
    """Records every call to the autonomous acquisition planner and (optionally) returns a terminal
    CampaignRunResult so the loop stops right after the planner is consulted."""

    def __init__(self, terminal=None):
        self.calls = 0
        self._terminal = terminal

    def __call__(self, controller, **kwargs):
        self.calls += 1
        return self._terminal


def _two_stage_costly_then_acquisition_workflow(root: Path) -> Path:
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "lazy-acquisition-regression",
        "stages": [
            {
                "name": "teacher_baseline", "command": None,
                "outputs": ["artifacts/teacher_baseline.json"],
                "gate": {"criteria": ["teacher baseline complete"]},
                "pydantic_ai": {
                    "role": "simulation", "action": "build_teacher_baseline",
                    "approval_boundary": "costly_teacher_labeling",
                    "idempotency_key": "lazy-acquisition-regression:teacher_baseline:001",
                    "parameters": {"structures_path": "local_inputs/structures.xyz"},
                },
            },
            {
                "name": "acquisition", "command": None,
                "outputs": ["artifacts/acquired.json"],
                "gate": {"criteria": ["acquisition batch closes the coverage gap"]},
                "pydantic_ai": {
                    "role": "data-curator", "action": "acquire_structures",
                    "approval_boundary": "costly_teacher_labeling",
                    "idempotency_key": "lazy-acquisition-regression:acquisition:001",
                    "parameters": {},
                },
            },
        ],
    }, sort_keys=False))
    return workflow


def _acquisition_only_workflow(root: Path) -> Path:
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "lazy-acquisition-entered-regression",
        "stages": [{
            "name": "acquisition", "command": None,
            "outputs": ["artifacts/acquired.json"],
            "gate": {"criteria": ["acquisition batch closes the coverage gap"]},
            "pydantic_ai": {
                "role": "data-curator", "action": "acquire_structures",
                "approval_boundary": "costly_teacher_labeling",
                "idempotency_key": "lazy-acquisition-entered-regression:acquisition:001",
                "parameters": {},
            },
        }],
    }, sort_keys=False))
    return workflow


class LazyAcquisitionPlanningTests(unittest.TestCase):
    def _patch_planner(self, spy):
        """Patch the two late-imported acquisition hooks the loop calls. Returns a restore fn."""
        import runtimes.pydantic_ai.acquisition_planner as planner_mod
        import runtimes.pydantic_ai.default_acquisition_provider as provider_mod
        orig_plan = planner_mod.plan_acquisition_via_reasoning_roles
        orig_install = provider_mod.maybe_install_default_acquisition_provider
        install_calls = []
        planner_mod.plan_acquisition_via_reasoning_roles = spy
        provider_mod.maybe_install_default_acquisition_provider = (
            lambda c: install_calls.append(1))

        def _restore():
            planner_mod.plan_acquisition_via_reasoning_roles = orig_plan
            provider_mod.maybe_install_default_acquisition_provider = orig_install
        return _restore, install_calls

    def test_planner_not_consulted_before_pre_acquisition_costly_boundary(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        spy = _PlannerSpy(terminal=None)
        restore, install_calls = self._patch_planner(spy)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "run"
                RunController.initialize(
                    _two_stage_costly_then_acquisition_workflow(root), run_dir)
                c = RunController(run_dir)

                # No approval -> campaign pauses at Stage-1 (teacher_baseline) costly boundary,
                # BEFORE acquisition is ever the next-eligible stage.
                result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                          auto_mock_judges=True, max_iterations=5)
                self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL,
                                 result.message)
                self.assertEqual(result.stage, "teacher_baseline")
                self.assertEqual(spy.calls, 0,
                                 "the expensive acquisition planner (FPS / population sizing) must "
                                 "NOT run before the pre-acquisition costly boundary is cleared")
                self.assertEqual(install_calls, [],
                                 "the acquisition provider must not even be installed pre-boundary")
                c = RunController(run_dir)
                self.assertFalse(
                    any(str(i.get("source", "")).endswith("acquisition_plan.json")
                        for i in c.state["inputs"]),
                    "no acquisition plan may be bound before the acquisition stage is entered")
        finally:
            restore()

    def test_planner_runs_when_acquisition_is_genuinely_entered(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController

        terminal = cli.CampaignRunResult(
            cli.CAMPAIGN_RESOURCE_BLOCKED, cli.EXIT_PROVIDER_UNAVAILABLE,
            "spy planner consulted for the acquisition stage")
        spy = _PlannerSpy(terminal=terminal)
        restore, install_calls = self._patch_planner(spy)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "run"
                RunController.initialize(_acquisition_only_workflow(root), run_dir)
                c = RunController(run_dir)

                result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                          auto_mock_judges=True, max_iterations=5)
                # The loop reached the acquisition stage, so the planner WAS consulted (once), and
                # its terminal result short-circuited the loop.
                self.assertEqual(spy.calls, 1,
                                 "the planner must run exactly when acquisition is next-eligible")
                self.assertEqual(install_calls, [1],
                                 "the default provider is installed lazily at that same point")
                self.assertEqual(result.outcome, cli.CAMPAIGN_RESOURCE_BLOCKED, result.message)
        finally:
            restore()


# =====================================================================================
# DEFECT 1 -- recovered-original-holdout must be a controller-bound INPUT
# =====================================================================================
class RecoveredHoldoutBindingTests(unittest.TestCase):
    class _StubController:
        def __init__(self, *, plan=None, inputs=None):
            self.state = {}
            if plan is not None:
                self.state["teacher_validation_plan"] = plan
            self.state["inputs"] = list(inputs or [])

    def test_present_recovered_holdout_binds(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference.yaml"
            ref.write_text(yaml.safe_dump(
                {"kind": "recovered-original-holdout", "target_split": "test"}))
            stub = self._StubController(
                plan={"selected_components": ["ORIGINAL_HELDOUT_FIDELITY"],
                      "target_split": "test"})
            resolved = cli._resolve_teacher_reference_binding(
                stub, {"recovered-original-holdout": str(ref)})
            self.assertEqual(resolved, str(ref))

    def test_absent_recovered_holdout_fails_closed(self):
        """The exact ffv4f defect: ORIGINAL_HELDOUT_FIDELITY selected but the recovered-original-
        holdout reference is NOT in the controller-input-derived reference map (e.g. it was only a
        stage ``reference_yaml`` parameter). Binding must fail closed, never silently substitute a
        protected-existing-dft reference."""
        from runtimes.pydantic_ai import cli
        stub = self._StubController(
            plan={"selected_components": ["ORIGINAL_HELDOUT_FIDELITY"], "target_split": "test"})
        with self.assertRaises(ValueError) as ctx:
            cli._resolve_teacher_reference_binding(
                stub, {"protected-existing-dft": "/some/historical/reference.yaml"})
        self.assertIn("recovered-original-holdout", str(ctx.exception))

    def test_target_split_mismatch_fails_closed(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference.yaml"
            ref.write_text(yaml.safe_dump(
                {"kind": "recovered-original-holdout", "target_split": "validation"}))
            stub = self._StubController(
                plan={"selected_components": ["ORIGINAL_HELDOUT_FIDELITY"],
                      "target_split": "test"})
            with self.assertRaises(ValueError) as ctx:
                cli._resolve_teacher_reference_binding(
                    stub, {"recovered-original-holdout": str(ref)})
            self.assertIn("target_split", str(ctx.exception))

    def test_reference_discovery_scans_only_controller_inputs(self):
        """``_protected_reference_from_inputs`` derives the reference map ONLY from
        ``controller.state['inputs']`` -- a reference kind that never became a controller input
        (only a stage parameter) is structurally invisible to binding, and a bound non-reference
        yaml is ignored."""
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "some_config.yaml"
            other.write_text(yaml.safe_dump({"kind": "not-a-reference-kind", "foo": 1}))
            # Empty inputs -> no references discovered.
            self.assertEqual(
                cli._protected_reference_from_inputs(self._StubController(inputs=[])), {})
            # A bound yaml whose kind is not a reference kind is ignored (no validation attempted).
            self.assertEqual(
                cli._protected_reference_from_inputs(
                    self._StubController(inputs=[{"source": str(other)}])), {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
