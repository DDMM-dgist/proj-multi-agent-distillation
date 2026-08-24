"""UNIT 4 -- executable coverage->acquisition re-acquisition recovery + input rebinding.

Covers the confirmed dead-loop fix: a coverage-deficit recovery that returns to the
``acquisition`` stage must RETIRE (supersede) the stale bound ``acquisition_plan.json`` input so
the re-run re-plans instead of regenerating byte-identical candidates (which
``verify_recovery_execution`` rejects as "did not change artifacts"). Also covers the generic
input-supersession primitive and the two plan-identity resolvers that must skip superseded inputs.

Every test drives the real production entry points (RunController, recovery_bridge, the cli /
default_acquisition_provider plan resolvers). No scientific compute runs here.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtimes.pydantic_ai.cli import _resolve_bound_acquisition_plan
from runtimes.pydantic_ai.default_acquisition_provider import _acquisition_plan_already_bound
from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
from runtimes.pydantic_ai.models import EvidenceReference
from runtimes.pydantic_ai.root_cause import (
    RootCauseClassification, validate_root_cause_classification,
)
from workflow.controller import RunController
from workflow.integrity import sha256_file

from test_full_lifecycle_integration import FixtureHelpers, GATE_CRITERION


def _write_plan(path: Path, *, seed: int, n_per: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"kind": "augment-atoms", "seed": seed, "n_parents": 2, "n_per_structure": n_per,
         "expected_output_count": 2 * n_per, "sigma_range_A": [0.02, 0.08], "cell_sigma": 0.0}))
    return path


class InputSupersessionPrimitiveTest(FixtureHelpers):
    """The generic bind/supersede/active-inputs primitive and both plan-identity resolvers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.controller = self._init_controller(self.root)

    def _bind_plan(self, name="acquisition_plan.json", *, seed=7, n_per=4):
        plan = _write_plan(self.root / name, seed=seed, n_per=n_per)
        return self.controller.bind_new_input(plan)

    def test_supersede_input_marks_record_excludes_from_active_and_emits_event(self):
        record = self._bind_plan()
        before = len(self.controller.active_inputs())
        self.controller.supersede_input(record["source"], reason="retired for test")
        active = self.controller.active_inputs()
        self.assertEqual(len(active), before - 1)
        superseded = [r for r in self.controller.state["inputs"] if r.get("superseded")]
        self.assertEqual(len(superseded), 1)
        self.assertEqual(superseded[0]["superseded_reason"], "retired for test")
        self.assertTrue(any(e["type"] == "input_superseded"
                            for e in self.controller.state["events"]))
        # the record and its content-addressed snapshot are NOT deleted (immutable lineage)
        self.assertIn(record, self.controller.state["inputs"])

    def test_supersede_input_fails_closed_when_no_active_input_matches(self):
        with self.assertRaisesRegex(ValueError, "no active bound input matches"):
            self.controller.supersede_input("/nonexistent/acquisition_plan.json",
                                            reason="x")

    def test_double_supersede_is_fail_closed(self):
        record = self._bind_plan()
        self.controller.supersede_input(record["source"], reason="first")
        with self.assertRaisesRegex(ValueError, "no active bound input matches"):
            self.controller.supersede_input(record["source"], reason="second")

    def test_supersede_bound_acquisition_plan_returns_empty_when_none_bound(self):
        self.assertEqual(self.controller.supersede_bound_acquisition_plan(reason="x"), [])

    def test_resolvers_skip_superseded_and_accept_a_fresh_rebound_plan(self):
        first = self._bind_plan(seed=7, n_per=4)
        self.assertTrue(_acquisition_plan_already_bound(self.controller))
        self.assertEqual(_resolve_bound_acquisition_plan(self.controller),
                         str(Path(first["snapshot"]).resolve()))

        # Retire the stale plan, then bind a fresh, larger, differently-seeded plan.
        self.controller.supersede_bound_acquisition_plan(reason="coverage deficit")
        self.assertFalse(_acquisition_plan_already_bound(self.controller))
        self.assertIsNone(_resolve_bound_acquisition_plan(self.controller))

        second = _write_plan(self.root / "reacq" / "acquisition_plan.json", seed=99, n_per=9)
        second_rec = self.controller.bind_new_input(second)
        # No ambiguity: only ONE active acquisition_plan remains, so the resolver returns it.
        self.assertTrue(_acquisition_plan_already_bound(self.controller))
        self.assertEqual(_resolve_bound_acquisition_plan(self.controller),
                         str(Path(second_rec["snapshot"]).resolve()))

    def test_two_active_plans_are_ambiguous_but_superseding_one_disambiguates(self):
        self._bind_plan(name="acquisition_plan.json", seed=7)
        second = _write_plan(self.root / "b" / "acquisition_plan.json", seed=8, n_per=4)
        self.controller.bind_new_input(second)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _resolve_bound_acquisition_plan(self.controller)
        # Retiring the first restores a unique active plan identity.
        self.controller.supersede_input(second, reason="retire duplicate")
        self.assertEqual(_resolve_bound_acquisition_plan(self.controller).endswith(
            "acquisition_plan.json"), True)


class StartIterationReacquisitionSupersessionTest(FixtureHelpers):
    """start_iteration must auto-retire the active acquisition plan iff the recovery returns to
    the acquisition stage -- breaking the dead-loop by forcing a re-plan."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

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

    def _drive_to_approved_recovery(self, controller, *, return_stage):
        baseline = controller.run_dir / "artifacts/baseline.json"
        self._write_json(baseline, {"role": "baseline"})
        controller.complete_external_stage("prepare", [baseline])
        self._gate(controller, "prepare", "PASS")

        acquired = controller.run_dir / "artifacts/acquired.json"
        self._write_json(acquired, {"role": "acquired", "revision": 1})
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
            evidence_summary="candidate coverage is below the frozen requirement",
            confidence=0.9, recommended_recovery_target=return_stage,
            recommended_next_action="re-acquire a larger gap-driven batch")
        validate_root_cause_classification(
            classification, available_artifacts=[str(coverage)],
            valid_recovery_targets=[s["name"] for s in controller.state["stages"]])
        diagnosis_path = self._write_json(controller.run_dir / "diagnosis.json",
                                          json.loads(classification.model_dump_json()))
        draft = build_recovery_plan_draft(
            classification,
            proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"},
            failed_stage="data_coverage", capability="data_repair", return_stage=return_stage,
            proposed_changes=[{"type": "reacquire_larger_gap_driven_batch"}],
            labeling={"teacher_relabel": False, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["acquisition", "data_coverage"]},
            diagnosis_artifact_path=str(diagnosis_path),
            diagnosis_artifact_sha256=sha256_file(diagnosis_path))
        plan_path = self._write_json(controller.run_dir / "plan.json", draft.to_plan_json())
        controller.propose_recovery(plan_path)
        controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"})

    def test_return_to_acquisition_retires_the_stale_plan(self):
        controller = self._staged_controller()
        plan = _write_plan(self.root / "acquisition_plan.json", seed=7, n_per=4)
        plan_record = controller.bind_new_input(plan)
        self.assertTrue(_acquisition_plan_already_bound(controller))

        self._drive_to_approved_recovery(controller, return_stage="acquisition")
        controller.start_iteration()

        # The stale plan is retired so the acquisition re-run re-plans (no reuse -> no dead-loop).
        self.assertFalse(_acquisition_plan_already_bound(controller))
        self.assertIsNone(_resolve_bound_acquisition_plan(controller))
        iteration = controller._current_iteration()
        self.assertEqual(iteration["trigger"]["superseded_acquisition_plans"],
                         [plan_record["sha256"]])
        self.assertTrue(any(e["type"] == "input_superseded"
                            for e in controller.state["events"]))

    def test_return_to_non_acquisition_stage_leaves_the_plan_active(self):
        controller = self._staged_controller()
        plan = _write_plan(self.root / "acquisition_plan.json", seed=7, n_per=4)
        controller.bind_new_input(plan)

        self._drive_to_approved_recovery(controller, return_stage="prepare")
        controller.start_iteration()

        # A recovery that does NOT return to acquisition must not touch the plan binding.
        self.assertTrue(_acquisition_plan_already_bound(controller))
        self.assertIsNotNone(_resolve_bound_acquisition_plan(controller))
        iteration = controller._current_iteration()
        self.assertEqual(iteration["trigger"]["superseded_acquisition_plans"], [])


class CoverageAdequacyDeterminationTest(unittest.TestCase):
    """Stage-4 coverage adequacy: COMPLETE is earned ONLY by a frozen coverage_requirement being
    satisfied by the real per-config_type counts; it is never self-asserted by a proposal."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _candidate(self, config_types):
        from ase import Atoms
        from ase.io import write
        frames = []
        for i, ct in enumerate(config_types):
            a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
            a.info["parent_structure_id"] = f"p{i}"
            a.info["config_type"] = ct
            frames.append(a)
        path = self.root / "candidate.extxyz"
        write(str(path), frames)
        manifest = self.root / "acq.manifest.json"
        manifest.write_text(json.dumps({"n_frames": len(frames) + 4, "elements": ["Cu"]}))
        return path, manifest

    def _run(self, config_types, deployment_domain, **extra):
        from runtimes.pydantic_ai.executors import _exec_build_data_coverage_report
        candidate, manifest = self._candidate(config_types)
        params = {"candidate_dataset": str(candidate), "acquisition_manifest": str(manifest),
                  "report_path": str(self.root / "report.json"),
                  "deployment_domain": deployment_domain, **extra}
        return _exec_build_data_coverage_report({"parameters": params})

    def test_no_frozen_requirement_is_fail_closed_partial(self):
        out = self._run(["bulk", "bulk"], {"structure_classes": ["bulk"]})
        self.assertEqual(out["report"]["coverage_status"], "PARTIAL")

    def test_frozen_requirement_met_earns_complete(self):
        out = self._run(["bulk", "bulk", "bulk", "surface"],
                        {"structure_classes": ["bulk", "surface"],
                         "coverage_requirement": {"min_frames_by_config_type":
                                                  {"bulk": 2, "surface": 1}}})
        self.assertEqual(out["report"]["coverage_status"], "COMPLETE")

    def test_frozen_requirement_unmet_is_partial_and_names_the_shortfall(self):
        out = self._run(["bulk", "bulk", "bulk", "surface"],
                        {"structure_classes": ["bulk", "surface"],
                         "coverage_requirement": {"min_frames_by_config_type":
                                                  {"bulk": 2, "surface": 3}}})
        report = out["report"]
        self.assertEqual(report["coverage_status"], "PARTIAL")
        self.assertTrue(any("surface" in gap and ">= 3" in gap
                            for gap in report["identified_gaps"]))

    def test_proposal_cannot_self_assert_complete(self):
        with self.assertRaisesRegex(ValueError, "may not be self-asserted"):
            self._run(["bulk"], {"structure_classes": ["bulk"]}, coverage_status="COMPLETE")

    def test_proposal_may_declare_conservative_partial(self):
        out = self._run(["bulk"], {"structure_classes": ["bulk"]}, coverage_status="PARTIAL")
        self.assertEqual(out["report"]["coverage_status"], "PARTIAL")

    def test_unavailable_access_is_not_assessable_even_with_a_requirement(self):
        out = self._run(["bulk", "bulk"],
                        {"structure_classes": ["bulk"],
                         "coverage_requirement": {"min_frames_by_config_type": {"bulk": 1}}},
                        teacher_training_data_access="unavailable")
        self.assertEqual(out["report"]["coverage_status"], "NOT_ASSESSABLE")

    def test_judge_adapter_surfaces_status_and_requirement_check(self):
        from runtimes.pydantic_ai import bounded_evidence
        out = self._run(["bulk", "bulk", "bulk", "surface"],
                        {"structure_classes": ["bulk", "surface"],
                         "coverage_requirement": {"min_frames_by_config_type":
                                                  {"bulk": 2, "surface": 3}}})
        summary = bounded_evidence.summarize_artifact(Path(out["path"]))
        cov = summary["data_coverage_report"]
        self.assertEqual(cov["coverage_status"], "PARTIAL")
        self.assertEqual(cov["adequacy_basis"], "frozen_coverage_requirement")
        self.assertEqual(cov["config_type_coverage"]["total_frames"], 4)
        self.assertEqual(cov["coverage_requirement_check"]["surface"]["met"], False)
        self.assertEqual(cov["coverage_requirement_check"]["bulk"]["met"], True)


if __name__ == "__main__":
    unittest.main()
