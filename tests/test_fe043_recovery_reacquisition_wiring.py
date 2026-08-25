"""FE-043 -- recovery->acquisition wiring: an approved recovery whose ``return_stage`` re-runs the
``acquire_structures`` route action must re-enter the SINGLE canonical Stage-3 acquisition planner
(``acquisition_planner.plan_acquisition_via_reasoning_roles``) to bind a fresh, superseding
AcquisitionPlan BEFORE the corrective dispatch runs ``acquire_structures`` -- otherwise the dispatch
fail-closes with ``PLAN_INPUT_REQUIRED`` (the live ffv4s regression).

These tests exercise the real production entry points (``run_campaign``/``_run_campaign_loop`` ->
``_dispatch_recovery_corrective_action`` -> ``dispatch.authorize_and_execute``) with a spy standing
in ONLY for the (expensive, geometry-heavy) canonical planner and a fixture executor standing in for
the (expensive, Teacher-driving) ``acquire_structures`` executor -- the same substitution pattern
``tests/test_run_campaign_recovery.py`` uses. No scientific compute runs here, and no second,
recovery-specific acquisition planner is introduced: the spy simply proves the ONE canonical planner
function is the thing the recovery path re-enters.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtimes.pydantic_ai import cli
from runtimes.pydantic_ai.cli import _resolve_bound_acquisition_plan, _stage_route_action
from runtimes.pydantic_ai.default_acquisition_provider import _acquisition_plan_already_bound
from runtimes.pydantic_ai.dispatch import ActionDescriptor
from runtimes.pydantic_ai.executors import build_executor_registry
from runtimes.pydantic_ai.models import EvidenceReference
from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
from runtimes.pydantic_ai.root_cause import (
    RootCauseClassification, validate_root_cause_classification,
)
from workflow.controller import RunController
from workflow.integrity import sha256_file

from test_full_lifecycle_integration import FixtureHelpers, GATE_CRITERION

ROOT = Path(__file__).resolve().parent.parent

# The five zero-occupancy declared structure classes the live ffv4s Stage-4 coverage report named
# (FE-042 downgraded a 3/3 Judge-PASS-on-COVERAGE_INSUFFICIENT gate to REVISE and bound these as the
# recovery's targeted-reacquisition gap). Reproduced verbatim as the recovery fixture's evidence --
# never consumed as a threshold/quota by the wiring under test, only carried to the planner.
FFV4S_UNSUPPORTED_CLASSES = [
    "amorphous_bulk_SiO2", "liquid_or_melt_SiO2", "surface_SiO2",
    "oxygen_vacancy_SiO2", "condensed_pure_Si_boundary",
]


def _write_plan(path: Path, *, seed: int, n_per: int, n_parents: int = 2) -> Path:
    """Write a VALID EXISTING_POOL_SELECTION AcquisitionPlan (the ffv4t projection: SELECT an existing
    subset, no frame generation, no Teacher inference). It passes ``_validate_existing_pool_plan``
    with no file I/O (``pool_path`` is a truthy marker, never read here) so the recovery corrective
    dispatch's FE-044 ``_bind_acquisition_plan_for_stage`` call classifies it geometry-only
    (``performs_teacher_inference=False``). ``seed`` makes the selected indices/SHA distinct so the
    stale and fresh plans have different identities. ``n_per`` is unused by a selection plan (kept for
    the caller signature); ``n_parents`` -> ``n_selected``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [seed * 100 + i for i in range(n_parents)]
    path.write_text(json.dumps({
        "schema_version": 1,
        "pool_path": "existing_pool.extxyz",
        "eligible_source_categories": ["amorphous_bulk_SiO2"],
        "selected_parent_structure_ids": [f"parent-{seed}-{i}" for i in range(n_parents)],
        "selected_source_global_indices": selected,
        "n_selected": n_parents,
        "expected_output_count": n_parents,
        "duplicate_handling": "reject_duplicates",
        "labeling_population_sizing": {"recommended_population_size": n_parents,
                                       "method": "fixture-deterministic"},
        "protected_reference_exclusion_report": {
            "status": "PASS", "dft_labels_used_as_selection_scores": False},
    }))
    return path


class _Fe043Fixture(FixtureHelpers):
    """Drives a prepare/acquisition/data_coverage controller to an APPROVED recovery whose
    ``return_stage`` is ``acquisition`` and whose approved corrective action is ``acquire_structures``
    -- the exact ffv4s shape -- then re-enters ``run_campaign`` with the canonical planner and the
    ``acquire_structures`` executor both replaced by observable fixtures."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.calls = []  # shared ordered call log across planner + corrective executor
        self.planner_plan_bound_observations = []

    def _staged_controller(self):
        stages = [
            {"name": "prepare", "command": None, "outputs": ["artifacts/baseline.json"],
             "gate": {"criteria": [GATE_CRITERION]}},
            {"name": "acquisition", "command": None, "outputs": ["artifacts/acquired.json"],
             "gate": {"criteria": [GATE_CRITERION]}},
            {"name": "data_coverage", "command": None, "outputs": ["artifacts/coverage.json"],
             "gate": {"criteria": [GATE_CRITERION]}},
        ]
        return self._init_controller(
            self.root, stages=stages,
            recovery_capability_roster={"data_repair": "data-curator",
                                        "orchestration": "orchestrator"})

    def _drive_to_approved_recovery(self, controller):
        # A pre-recovery acquisition plan + a non-plan cumulative-population input, so we can prove
        # start_iteration retires ONLY the stale plan (re-plan) while preserving already-acquired data.
        stale_plan = _write_plan(self.root / "acquisition_plan.json", seed=7, n_per=4)
        self.stale_plan_record = controller.bind_new_input(stale_plan)
        cumulative = self._write_json(self.root / "already_acquired_pool.json",
                                      {"role": "cumulative_acquired", "n_frames": 40})
        self.cumulative_record = controller.bind_new_input(cumulative)

        baseline = controller.run_dir / "artifacts/baseline.json"
        self._write_json(baseline, {"role": "baseline"})
        controller.complete_external_stage("prepare", [baseline])
        self._gate(controller, "prepare", "PASS")

        acquired = controller.run_dir / "artifacts/acquired.json"
        self._write_json(acquired, {"role": "acquired", "revision": 1, "n_frames": 40})
        controller.complete_external_stage("acquisition", [acquired])
        self._gate(controller, "acquisition", "PASS")

        coverage = controller.run_dir / "artifacts/coverage.json"
        self._write_json(coverage, {"role": "coverage", "coverage_status": "PARTIAL"})
        controller.complete_external_stage("data_coverage", [coverage])
        self._gate(controller, "data_coverage", "REVISE")

        classification = RootCauseClassification(
            run_id="fixture-run", stage="data_coverage", failure_category="dataset_coverage",
            evidence_refs=[EvidenceReference(role="coverage", path=str(coverage),
                                             integrity={"sha256": sha256_file(coverage)})],
            evidence_summary="declared structure classes have zero acquired representatives",
            confidence=0.9, recommended_recovery_target="acquisition",
            recommended_next_action="re-acquire a gap-driven batch for the unsupported classes")
        validate_root_cause_classification(
            classification, available_artifacts=[str(coverage)],
            valid_recovery_targets=[s["name"] for s in controller.state["stages"]])
        diagnosis_path = self._write_json(controller.run_dir / "diagnosis.json",
                                          json.loads(classification.model_dump_json()))
        draft = build_recovery_plan_draft(
            classification,
            proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"},
            failed_stage="data_coverage", capability="data_repair", return_stage="acquisition",
            proposed_changes=[{"type": "reacquire_gap_driven_batch"}],
            labeling={"teacher_relabel": False, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["acquisition", "data_coverage"]},
            diagnosis_artifact_path=str(diagnosis_path),
            diagnosis_artifact_sha256=sha256_file(diagnosis_path),
            extra_recovery_context={
                "corrective_action": {"action_type": "acquire_structures",
                                      "parameters": {"note": "reacquire gap-driven batch"}},
                "unsupported_structure_classes": list(FFV4S_UNSUPPORTED_CLASSES),
                "coverage_adequacy": {"control": "fe042_coverage_adequacy",
                                      "assessment_status": "COVERAGE_INSUFFICIENT",
                                      "recommended_return_stage": "acquisition"},
            })
        plan_path = self._write_json(controller.run_dir / "plan.json", draft.to_plan_json())
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})

    # --- observable fixtures for the two expensive substitutions --------------------------------
    def _spy_planner(self, *, bind_fresh: bool):
        """A stand-in for the canonical Stage-3 planner. Records the controller state it is handed
        (so tests can prove the stale plan was already superseded and the cumulative population is
        preserved), optionally binds a fresh superseding plan, and returns None (no pause)."""
        test = self

        def planner(controller, **_kwargs):
            test.calls.append("planner")
            # Observation of the FIRST (corrective re-entry) invocation; a later forward no-op
            # invocation sees the now-bound plan, so only the first observation is the re-plan proof.
            test.planner_plan_bound_observations.append(
                _acquisition_plan_already_bound(controller))
            test.planner_saw_cumulative = any(
                str(r.get("source", "")).endswith("already_acquired_pool.json")
                and not r.get("superseded")
                for r in controller.state["inputs"])
            recovery = controller.state["recoveries"][-1]
            test.planner_saw_classes = (recovery["plan"].get("recovery_context") or {}).get(
                "unsupported_structure_classes")
            # Mirror the real planner's idempotency: once a plan is bound its provider's applies()
            # is False, so a re-looped/forward invocation is a no-op (never binds a second plan).
            if bind_fresh and not _acquisition_plan_already_bound(controller):
                fresh = _write_plan(test.root / "reacq" / "acquisition_plan.json",
                                    seed=99, n_per=9)
                controller.bind_new_input(fresh)
            return None

        return planner

    def _corrective_registry(self):
        """Real production registry with ONLY ``acquire_structures`` replaced by a fixture executor
        that fail-closes if no AcquisitionPlan reached it and otherwise writes the enlarged
        cumulative ``acquired.json``. approval_boundary=None keeps the (orthogonal, already-covered)
        costly-labeling approval gate out of this wiring test."""
        test = self
        run_dir = self.controller.run_dir

        def executor(proposal):
            params = proposal.get("parameters") or {}
            test.dispatch_saw_plan_path = params.get("acquisition_plan_path")
            test.calls.append("dispatch")
            if not (params.get("acquisition_plan_path") or params.get("acquisition_plan")):
                raise ValueError(
                    "PLAN_INPUT_REQUIRED: AcquisitionPlan is required before "
                    "acquire_structures execution")
            out = run_dir / "artifacts/acquired.json"
            out.write_text(json.dumps({"role": "acquired", "revision": 2, "n_frames": 61}))
            return {"path": str(out), "manifest": {"n_frames": 61}, "sha256": sha256_file(out)}

        registry = build_executor_registry()
        registry["acquire_structures"] = ActionDescriptor(
            action_type="acquire_structures", role="data-curator", approval_boundary=None,
            executor=executor)
        return registry

    def _run_after_recovery(self, *, planner, stop_forward=True):
        """Run ``run_campaign`` from the approved-recovery state with the spy planner installed.
        When ``stop_forward`` is set, the first forward ``run_production_stage`` call (which would
        follow the corrective dispatch) is intercepted so the loop stops immediately AFTER the
        FE-043 corrective reacquisition, without invoking any real forward-stage compute."""
        import runtimes.pydantic_ai.acquisition_planner as ap
        import runtimes.pydantic_ai.default_acquisition_provider as dap

        orig_planner = ap.plan_acquisition_via_reasoning_roles
        orig_install = dap.maybe_install_default_acquisition_provider
        orig_run_stage = cli.run_production_stage
        ap.plan_acquisition_via_reasoning_roles = planner
        dap.maybe_install_default_acquisition_provider = lambda c: self.calls.append("install")

        def _stop_forward_stage(controller, stage_name, **_kwargs):
            self.forward_stage_reached = stage_name
            return cli.StageRunResult(
                "APPROVAL_REQUIRED", cli.EXIT_APPROVAL_REQUIRED,
                "FE043-TEST-STOP: forward loop reached after corrective reacquisition")

        if stop_forward:
            cli.run_production_stage = _stop_forward_stage
        self.addCleanup(setattr, ap, "plan_acquisition_via_reasoning_roles", orig_planner)
        self.addCleanup(setattr, dap, "maybe_install_default_acquisition_provider", orig_install)
        self.addCleanup(setattr, cli, "run_production_stage", orig_run_stage)

        c = RunController(self.controller.run_dir)
        return cli.run_campaign(c, runtime="mock", repo_root=str(ROOT), auto_mock_judges=True,
                                max_iterations=8,
                                recovery_action_registry=self._corrective_registry())


class Fe043WiringTests(_Fe043Fixture):
    def setUp(self):
        super().setUp()
        self.controller = self._staged_controller()
        self.assertEqual(_stage_route_action(self.controller, "acquisition"), "acquire_structures")
        self._drive_to_approved_recovery(self.controller)

    # A -- a return_stage=acquisition recovery cannot dispatch acquire_structures before a fresh
    #      AcquisitionPlan is bound: if the re-entered planner binds nothing, the corrective dispatch
    #      fail-closes with PLAN_INPUT_REQUIRED (never a silent plan-less acquisition).
    def test_A_no_plan_bound_fail_closes_before_acquire(self):
        result = self._run_after_recovery(planner=self._spy_planner(bind_fresh=False),
                                          stop_forward=False)
        self.assertEqual(result.outcome, cli.CAMPAIGN_FAILED, result.message)
        self.assertIn("PLAN_INPUT_REQUIRED", result.message)
        # The planner WAS re-entered (control reached the FE-043 branch) but bound no plan.
        self.assertIn("planner", self.calls)
        c = RunController(self.controller.run_dir)
        self.assertFalse(_acquisition_plan_already_bound(c))
        self.assertNotEqual(c.stage("acquisition")["status"], "completed")

    # B -- the SINGLE canonical acquisition-planning path is what the recovery re-enters (no
    #      recovery-specific planner): the provider install + canonical planner both fire, in order,
    #      before the corrective dispatch.
    def test_B_canonical_planner_path_is_invoked_for_recovery(self):
        self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        self.assertEqual(self.calls[:3], ["install", "planner", "dispatch"])
        # planner ran while the stale plan was ALREADY retired (start_iteration re-plan contract).
        self.assertFalse(self.planner_plan_bound_observations[0])

    # C -- the exact ffv4s recovery fixture produces a fresh superseding AcquisitionPlan and the
    #      corrective acquire_structures dispatch executes WITHOUT PLAN_INPUT_REQUIRED.
    def test_C_ffv4s_fixture_binds_superseding_plan_and_avoids_plan_input_required(self):
        result = self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        self.assertNotIn("PLAN_INPUT_REQUIRED", result.message)
        self.assertEqual(self.forward_stage_reached, "acquisition")
        c = RunController(self.controller.run_dir)
        # A fresh plan is uniquely active; the pre-recovery plan is superseded.
        self.assertTrue(_acquisition_plan_already_bound(c))
        fresh = _resolve_bound_acquisition_plan(c)
        self.assertTrue(fresh.endswith("acquisition_plan.json"))
        self.assertNotIn(self.stale_plan_record["sha256"],
                         [r["sha256"] for r in c.active_inputs()])
        # The corrective dispatch saw the freshly-bound plan (not a plan-less proposal).
        self.assertIsNotNone(self.dispatch_saw_plan_path)
        # The ffv4s coverage-gap classes reached the planner as evidence.
        self.assertEqual(self.planner_saw_classes, FFV4S_UNSUPPORTED_CLASSES)

    # D -- already-acquired (cumulative) and other non-plan inputs are preserved through recovery
    #      activation: start_iteration retires ONLY the stale acquisition plan, so the canonical
    #      planner still sees the cumulative population it must exclude-from / build upon.
    def test_D_already_acquired_and_other_inputs_preserved_for_planner(self):
        self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        self.assertTrue(self.planner_saw_cumulative)
        c = RunController(self.controller.run_dir)
        # Exactly the stale plan was superseded by start_iteration; the cumulative input is active.
        iteration = c.state["iterations"][-1]
        self.assertEqual(iteration["trigger"]["superseded_acquisition_plans"],
                         [self.stale_plan_record["sha256"]])
        self.assertTrue(any(str(r.get("source", "")).endswith("already_acquired_pool.json")
                            and not r.get("superseded") for r in c.state["inputs"]))

    # E -- after the corrective reacquisition, the acquisition stage is completed against the fresh
    #      superseding plan with the enlarged cumulative population, and data_coverage (Stage 4) is
    #      the next stage eligible to consume it.
    def test_E_stage4_consumes_superseding_cumulative_population(self):
        self._run_after_recovery(planner=self._spy_planner(bind_fresh=True))
        c = RunController(self.controller.run_dir)
        self.assertEqual(c.stage("acquisition")["status"], "completed")
        acquired = json.loads((c.run_dir / "artifacts/acquired.json").read_text())
        self.assertEqual(acquired["revision"], 2)
        self.assertEqual(acquired["n_frames"], 61)
        # data_coverage was invalidated back to pending by start_iteration and is the next stage the
        # forward loop selects to reassess against the enlarged population.
        self.assertNotEqual(c.stage("data_coverage")["gate"], "PASS")
        self.assertEqual(self.forward_stage_reached, "acquisition")


if __name__ == "__main__":
    unittest.main()
