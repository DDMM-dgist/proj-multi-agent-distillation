"""Phase 6/2 production wiring: the generic ``typed_reasoning_output`` acceptance strategy
(production_router._accept_typed_reasoning_output) actually dispatches a live agent role and
accepts its result as a registered reasoning model (RootCauseClassification), not as the role's
default proposal/plan output. Network-free (mock runtime); skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = str(ROOT / "agent_specs")

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _classification_payload(**over):
    base = dict(
        run_id="r", stage="validation", failure_category="student_fidelity",
        affected_channel="student_vs_teacher",
        affected_artifact_refs=[{"role": "ml-trainer", "path": "runs/r/eval/errorc.json"}],
        evidence_refs=[{"role": "ml-trainer", "path": "runs/r/eval/errorc.json"}],
        evidence_summary="force MAE above threshold on held-out set",
        confidence=0.7, excluded_alternatives=["data_coverage"],
        recommended_recovery_target="student_training",
        recommended_next_action="retrain committee on augmented set")
    base.update(over)
    return base


def _analyst_task(**context_over):
    context = {"expected_output_model": "RootCauseClassification"}
    context.update(context_over)
    return {"schema_version": 1, "task_id": "an-rc-1", "agent": "analyst",
            "created_at": "2026-08-08T00:00:00Z", "instruction": "diagnose", "inputs": [],
            "criteria": [], "constraints": [], "context": context}


def _recovery_plan_proposal_payload(**over):
    base = dict(
        run_id="r", failed_stage="produce_evidence", diagnosis_artifact_sha256="d" * 64,
        capability="data_repair", return_stage="prepare",
        proposed_changes=[{"type": "add_deployment_frames"}],
        labeling={"teacher_relabel": True, "new_dft": False},
        student_training={"retrain": False, "mode": "none"},
        revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
        rationale="coverage gap requires more deployment-representative frames")
    base.update(over)
    return base


def _orchestrator_task(**context_over):
    context = {"expected_output_model": "RecoveryPlanProposal"}
    context.update(context_over)
    return {"schema_version": 1, "task_id": "orc-rp-1", "agent": "orchestrator",
            "created_at": "2026-08-08T00:00:00Z", "instruction": "propose recovery plan",
            "inputs": [], "criteria": [], "constraints": [], "context": context}


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ReasoningOutputRouteTests(unittest.TestCase):
    def _spec(self):
        from orchestration.specs import load_agent_specs
        return load_agent_specs(SPECS)["analyst"]

    def _run(self, tmp, payload, *, mode="primary", reasoning_validator=None):
        from runtimes.pydantic_ai.production_router import run_role
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.models import RuntimeContext

        ex = Path(tmp) / "ex"
        ex.mkdir()
        raw = json.dumps(payload)
        ctx = RuntimeContext(exchange_dir=str(ex), repo_root=str(ROOT))
        task = _analyst_task()
        return run_role(MockAgentRuntime(lambda t, s, ts: (raw, (0, 0))), task, self._spec(),
                        ctx, mode=mode, reasoning_validator=reasoning_validator)

    def test_task_declared_output_model_overrides_role_default(self):
        from runtimes.pydantic_ai.production_router import acceptance_strategy
        strategy = acceptance_strategy(self._spec(), _analyst_task())
        self.assertEqual(strategy, "typed_reasoning_output")
        # the SAME spec, without the task hint, keeps its ordinary producer strategy
        self.assertEqual(acceptance_strategy(self._spec(), None), "producer_dispatch")

    def test_accepted_output_is_root_cause_classification_not_action_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, _classification_payload())
            self.assertEqual(res.strategy, "typed_reasoning_output")
            self.assertTrue(res.accepted)
            self.assertFalse(res.controller_mutated)  # reasoning acceptance never dispatches
            from runtimes.pydantic_ai.root_cause import RootCauseClassification
            self.assertIsInstance(res.detail.instance, RootCauseClassification)

    def test_unregistered_failure_taxonomy_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, _classification_payload(failure_category="totally_made_up"))
            self.assertFalse(res.accepted)
            self.assertIsNotNone(res.error)

    def test_nonexistent_evidence_ref_fails_closed_via_contextual_validator(self):
        from runtimes.pydantic_ai.root_cause import validate_root_cause_classification
        with tempfile.TemporaryDirectory() as tmp:
            validator = lambda c: validate_root_cause_classification(
                c, available_artifacts={"runs/r/eval/errorc.json"},
                valid_recovery_targets={"student_training", "data_curation"})
            bad = _classification_payload(
                evidence_refs=[{"role": "x", "path": "runs/r/does_not_exist.json"}])
            res = self._run(tmp, bad, reasoning_validator=validator)
            self.assertFalse(res.accepted)
            self.assertIn("nonexistent artifact", res.error)

    def test_shadow_mode_never_persists_or_mutates(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, _classification_payload(), mode="shadow")
            self.assertFalse(res.accepted)
            self.assertFalse(res.controller_mutated)
            self.assertIsNone(res.detail.artifact_path)

    def test_accepted_artifact_is_hash_bound_and_persisted(self):
        from workflow.integrity import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, _classification_payload())
            path = res.detail.artifact_path
            self.assertTrue(path.exists())
            self.assertEqual(res.detail.artifact_sha256, sha256_file(path))

    def test_reasoning_output_performs_no_executor_dispatch(self):
        # No controller/registry were even supplied to run_role -- if this path ever tried to
        # dispatch an executor it would raise (producer_dispatch requires them); it must not.
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, _classification_payload())
            self.assertTrue(res.accepted)

    # --- SECOND registered reasoning model, different role: proves the mechanism is generic,
    # not special-cased to the Analyst/RootCauseClassification pairing. ------------------------

    def _run_orchestrator(self, tmp, payload, *, mode="primary", reasoning_validator=None):
        from runtimes.pydantic_ai.production_router import run_role
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.models import RuntimeContext
        from orchestration.specs import load_agent_specs

        ex = Path(tmp) / "ex"
        ex.mkdir()
        raw = json.dumps(payload)
        ctx = RuntimeContext(exchange_dir=str(ex), repo_root=str(ROOT))
        spec = load_agent_specs(SPECS)["orchestrator"]
        task = _orchestrator_task()
        return run_role(MockAgentRuntime(lambda t, s, ts: (raw, (0, 0))), task, spec, ctx,
                        mode=mode, reasoning_validator=reasoning_validator)

    def test_orchestrator_task_declared_output_model_overrides_role_default(self):
        from runtimes.pydantic_ai.production_router import acceptance_strategy
        from orchestration.specs import load_agent_specs
        spec = load_agent_specs(SPECS)["orchestrator"]
        self.assertEqual(acceptance_strategy(spec, _orchestrator_task()), "typed_reasoning_output")
        self.assertEqual(acceptance_strategy(spec, None), "typed_result")

    def test_orchestrator_accepted_output_is_recovery_plan_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run_orchestrator(tmp, _recovery_plan_proposal_payload())
            self.assertEqual(res.strategy, "typed_reasoning_output")
            self.assertTrue(res.accepted)
            self.assertFalse(res.controller_mutated)
            from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
            self.assertIsInstance(res.detail.instance, RecoveryPlanProposal)

    def test_orchestrator_stale_diagnosis_binding_fails_closed(self):
        from runtimes.pydantic_ai.recovery_bridge import validate_recovery_plan_proposal
        with tempfile.TemporaryDirectory() as tmp:
            validator = lambda p: validate_recovery_plan_proposal(
                p, expected_failed_stage="produce_evidence", expected_diagnosis_sha256="e" * 64,
                capability_roster={"data_repair": "data-curator"},
                valid_stage_names={"prepare", "produce_evidence"})
            res = self._run_orchestrator(tmp, _recovery_plan_proposal_payload(),
                                         reasoning_validator=validator)
            self.assertFalse(res.accepted)
            self.assertIn("diagnosis_artifact_sha256", res.error)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
