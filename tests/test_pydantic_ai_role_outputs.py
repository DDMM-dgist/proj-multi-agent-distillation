"""Phase 2/D2: role-specific typed outputs, capability registry, and output-model selection.

Network-free; skips when the optional ``pydantic`` extra is absent.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _common():
    return dict(run_id="r1", stage="data_curation", requested_at="2026-08-07T00:00:00Z",
                rationale="cover a deployment coverage gap", idempotency_key="k1")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ActionProposalTests(unittest.TestCase):
    def test_each_producer_accepts_its_own_action_and_rejects_others(self):
        import pydantic
        from runtimes.pydantic_ai.actions import (
            DataCuratorActionProposal, MLTrainerActionProposal,
            SimulationActionProposal, AnalystActionProposal)
        cases = [
            (DataCuratorActionProposal, "generate_group_split", "run_teacher_md"),
            (MLTrainerActionProposal, "train_committee", "compute_rdf"),
            (SimulationActionProposal, "run_teacher_md", "train_committee"),
            (AnalystActionProposal, "compare_force_errors", "generate_group_split"),
        ]
        for model, own, foreign in cases:
            ok = model(action_type=own, **_common())
            self.assertEqual(ok.action_type, own)
            self.assertTrue(ok.dry_run)  # dry_run defaults True
            with self.assertRaises(pydantic.ValidationError, msg=f"{model.__name__} accepted {foreign}"):
                model(action_type=foreign, **_common())

    def test_role_label_is_fixed_per_model(self):
        from runtimes.pydantic_ai.actions import DataCuratorActionProposal
        self.assertEqual(
            DataCuratorActionProposal(action_type="inspect_dataset", **_common()).requested_by_role,
            "data-curator")

    def test_extra_fields_rejected(self):
        import pydantic
        from runtimes.pydantic_ai.actions import SimulationActionProposal
        with self.assertRaises(pydantic.ValidationError):
            SimulationActionProposal(action_type="compute_rdf", surprise=1, **_common())

    def test_required_fields_enforced(self):
        import pydantic
        from runtimes.pydantic_ai.actions import DataCuratorActionProposal
        for drop in ("run_id", "stage", "requested_at", "rationale", "idempotency_key"):
            payload = {k: v for k, v in _common().items() if k != drop}
            with self.assertRaises(pydantic.ValidationError):
                DataCuratorActionProposal(action_type="inspect_dataset", **payload)

    def test_the_llm_cannot_return_an_executable_string(self):
        # There is no field that carries a shell/command string; parameters is a bounded dict
        # and action_type is a constrained Literal. Assert no 'command'/'cmd'/'script' field.
        from runtimes.pydantic_ai.actions import SimulationActionProposal
        fields = set(SimulationActionProposal.model_fields)
        self.assertFalse({"command", "cmd", "script", "shell", "argv"} & fields)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class CapabilityRegistryTests(unittest.TestCase):
    def test_out_of_scope_actions_registered_with_status(self):
        from runtimes.pydantic_ai.actions import CAPABILITY_REGISTRY, capability_status
        expected = {
            "compute_eos": "NOT_AVAILABLE", "compute_mechanics": "NOT_AVAILABLE",
            "compute_ring_statistics": "NOT_AVAILABLE", "compute_sq_fsdp": "NOT_AVAILABLE",
            "compute_adf": "NOT_AVAILABLE", "compute_channel_d": "NOT_AVAILABLE",
            "fine_tune_teacher": "OUT_OF_CURRENT_SCOPE",
            "generate_dft_inputs": "APPROVAL_REQUIRED", "run_dft": "APPROVAL_REQUIRED",
            "generate_scheduler_script": "NOT_AVAILABLE",
        }
        for action, status in expected.items():
            entry = capability_status(action)
            self.assertIsNotNone(entry, action)
            self.assertEqual(entry.status, status, action)
            self.assertTrue(entry.reason)  # every entry states why

    def test_unavailable_actions_never_appear_in_allowed_sets(self):
        from runtimes.pydantic_ai.actions import CAPABILITY_REGISTRY, ROLE_ALLOWED_ACTIONS
        allowed = set().union(*ROLE_ALLOWED_ACTIONS.values())
        self.assertEqual(set(CAPABILITY_REGISTRY) & allowed, set())

    def test_approval_gated_actions_are_in_allowed_sets(self):
        from runtimes.pydantic_ai.actions import APPROVAL_GATED_ACTIONS, ROLE_ALLOWED_ACTIONS
        allowed = set().union(*ROLE_ALLOWED_ACTIONS.values())
        for action in APPROVAL_GATED_ACTIONS:
            self.assertIn(action, allowed)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class SelectorAndTypedOutputTests(unittest.TestCase):
    def test_select_output_model_for_every_role(self):
        from runtimes.pydantic_ai.role_outputs import select_output_model
        from orchestration.specs import load_agent_specs
        specs = load_agent_specs(str(ROOT / "agent_specs"))
        expected = {
            "judge": "JudgeVoteModel", "data-curator": "DataCuratorActionProposal",
            "ml-trainer": "MLTrainerActionProposal", "simulation": "SimulationActionProposal",
            "analyst": "AnalystActionProposal", "orchestrator": "OrchestratorPlan",
            "literature": "LiteratureEvidence",
        }
        for name, model_name in expected.items():
            self.assertEqual(select_output_model(specs[name]).__name__, model_name, name)

    def test_selector_fallback_is_generic_result(self):
        from runtimes.pydantic_ai.role_outputs import select_output_model

        class Spec:
            name = "some-new-role"
        self.assertEqual(select_output_model(Spec()).__name__, "AgentResultModel")

    def test_literature_evidence_allows_blocked_without_fabrication(self):
        from runtimes.pydantic_ai.role_outputs import LiteratureEvidence
        ev = LiteratureEvidence(status="blocked", summary="no source retrieved", sources=[])
        self.assertEqual(ev.sources, [])

    def test_source_record_requires_title_and_type(self):
        import pydantic
        from runtimes.pydantic_ai.role_outputs import SourceRecord
        with self.assertRaises(pydantic.ValidationError):
            SourceRecord(source_type="journal")  # missing title
        rec = SourceRecord(title="a-SiO2 density", source_type="journal", value=2.2, unit="g/cm3")
        self.assertEqual(rec.access_status, "retrieved")

    def test_orchestrator_plan_typed(self):
        from runtimes.pydantic_ai.role_outputs import OrchestratorPlan, AgentTaskProposal
        plan = OrchestratorPlan(
            run_id="r1", current_stage="teacher_baseline", rationale="proceed", summary="plan",
            proposed_tasks=[AgentTaskProposal(agent="data-curator", instruction="inspect",
                                              rationale="need coverage")])
        self.assertEqual(plan.proposed_tasks[0].agent, "data-curator")


# --- R17 contract-parity audit regression: AgentResultModel.status was exposed as a bare `str`
# while orchestration.exchange.validate_agent_response secretly required membership in
# orchestration.specs.RESULT_STATUSES -- the same hidden-constraint defect class as the R17
# RecoveryPlanProposal.corrective_action bug, found during the repository-wide audit it prompted.
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class AgentResultStatusContractTests(unittest.TestCase):
    def _result(self, **over):
        from runtimes.pydantic_ai.models import AgentResultModel
        base = dict(task_id="t1", agent="data-curator", status="completed",
                    summary="did the thing")
        base.update(over)
        return AgentResultModel(**base)

    def test_registered_status_values_accepted(self):
        from orchestration.specs import RESULT_STATUSES
        for status in RESULT_STATUSES:
            self.assertEqual(self._result(status=status).status, status)

    def test_unregistered_status_rejected_by_schema_with_precise_reason(self):
        import pydantic
        with self.assertRaises(pydantic.ValidationError) as ctx:
            self._result(status="not_a_real_status")
        message = str(ctx.exception)
        self.assertIn("status", message)

    def test_schema_exposes_full_status_enum_single_sourced_from_result_statuses(self):
        """The allowed value set must be MACHINE-VISIBLE in the generated JSON Schema, not only
        enforced after the fact by orchestration.exchange.validate_agent_response."""
        from orchestration.specs import RESULT_STATUSES
        from runtimes.pydantic_ai.models import AgentResultModel
        schema = AgentResultModel.model_json_schema()
        self.assertEqual(set(schema["properties"]["status"]["enum"]), set(RESULT_STATUSES))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
