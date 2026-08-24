"""Framework V2 -- integration through the REAL Controller + PydanticAI runtime.

The other V2 regression tests exercise the contract chain standalone. This
module closes the "no real integration" blocker: it drives an actual
``workflow.controller.RunController`` (init from a workflow.yaml, run a stage,
record a gate) with V2 contracts bound to the run, and it exercises the actual
``runtimes.pydantic_ai.dispatch.authorize_and_execute`` capability-negotiation
path and the actual ``runtimes.pydantic_ai.bounded_evidence.build_bounded_evidence``
fact-consumption path. No expensive scientific compute runs -- committee LOGs
are tiny synthetic SIMPLE-NN logs and the stage command is a one-line writer.
"""
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


GATE_CRITERION = "committee training is converged and internally consistent"


def _epoch_line(n, valid_e, valid_f=0.2, lr=1e-4):
    return (f"Epoch {n} E RMSE(T V) {valid_e - 0.01:.6f} {valid_e:.6f} "
            f"F RMSE(T V) {valid_f - 0.01:.6f} {valid_f:.6f} learning_rate: {lr}")


def _log(*, requested, best, valid_e_series):
    """Build a synthetic SIMPLE-NN LOG. ``valid_e_series`` is a list of
    (epoch, valid_energy_rmse) points."""
    lines = [f"Total traning epoch: {requested}"]
    for ep, ve in valid_e_series:
        lines.append(_epoch_line(ep, ve))
    lines.append(f"Best loss (valid) written at {best} epoch")
    return "\n".join(lines) + "\n"


# Boundary=100, tolerance=5. NOT_CONVERGED: best at boundary AND valid_e still
# meaningfully falling over the trailing window.
_NOT_CONVERGED_LOG = _log(
    requested=100, best=100,
    valid_e_series=[(ep, 0.20 - 0.001 * (ep - 90)) for ep in range(90, 101)])
# CONVERGED_AT_MAX: best at boundary but valid_e flat across the trailing window.
_CONVERGED_LOG = _log(
    requested=100, best=100,
    valid_e_series=[(ep, 0.100) for ep in range(90, 101)])


def _scope_contract():
    return DeploymentScopeContract(
        contract_id="itest-scope",
        objective="integration test deployment scope",
        regions=[ScopeRegion(
            region_id="primary", category=ScopeCategory.PRIMARY_DEPLOYMENT,
            membership_rule="descriptor in the primary deployment set")],
    )


def _convergence_policy():
    return ConvergencePolicy(
        policy_id="itest-conv", trailing_window=10, projection_window=50,
        min_relative_improvement=0.05, boundary_tolerance=5,
        metrics=["valid_energy_rmse"],
        provenance_class=ProvenanceClass.HUMAN_FIXED, provenance_source="itest")


class _Base(unittest.TestCase):
    def _controller(self, root):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "v2-itest", "stages": [
            {"name": "training",
             "command": [sys.executable, "-c",
                         "from pathlib import Path; "
                         "Path('artifacts/model.txt').write_text('trained')"],
             "outputs": ["artifacts/model.txt"],
             "gate": {"criteria": [GATE_CRITERION]}},
        ]}))
        c = RunController.initialize(cfg, root / "run")
        c.run_stage("training")
        return RunController(c.run_dir)

    def _write_committee(self, controller, seed_logs):
        committee = controller.run_dir / "artifacts" / "committee"
        for seed_id, text in seed_logs.items():
            d = committee / f"seed-{seed_id}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "LOG").write_text(text, encoding="utf-8")

    def _pass_bundle(self, controller, stage, **extra):
        artifacts = {a["path"]: a["sha256"]
                     for a in controller.stage_artifacts(stage)}
        criteria = controller.stage(stage)["gate_criteria"]
        lenses = controller.stage(stage)["gate_review_lenses"]

        def vote(i, lens):
            return {"judge_id": f"judge-{i}", "review_lens": lens["id"],
                    "verdict": "PASS",
                    "criteria_checked": [{"criterion": crit,
                                          "value_read": "verified", "ok": True}
                                         for crit in criteria],
                    "rationale": "ok", "required_fix": ""}
        bundle = {"stage": stage, "criteria": criteria, "review_lenses": lenses,
                  "artifact_sha256": artifacts, "decision": "PASS",
                  "votes": [vote(i, lens) for i, lens in enumerate(lenses, 1)]}
        bundle.update(extra)
        path = controller.run_dir / "gates" / f"{stage}.votes.json"
        path.write_text(json.dumps(bundle))
        return path


class ConvergenceGateIntegrationTests(_Base):
    def test_v2_disabled_is_noop_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            self.assertFalse(c.v2_enabled())
            path = self._pass_bundle(c, "training")
            c.record_gate("training", votes_path=path)
            c = RunController(c.run_dir)
            self.assertEqual(c.stage("training")["gate"], "PASS")
            event = [e for e in c.state["events"] if e.get("type") == "gate"][-1]
            self.assertNotIn("framework_v2", event)

    def test_v2_refuses_pass_when_not_converged(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
            c.bind_v2_contract("convergence_policy",
                               _convergence_policy().model_dump(mode="json"),
                               stage="training")
            self._write_committee(c, {"202631": _NOT_CONVERGED_LOG})
            path = self._pass_bundle(c, "training")
            with self.assertRaisesRegex(RuntimeError, "NOT_CONVERGED|refusing PASS"):
                c.record_gate("training", votes_path=path)
            # Fail-closed: the gate must not have recorded a PASS.
            c = RunController(c.run_dir)
            self.assertNotEqual(c.stage("training").get("gate"), "PASS")

    def test_v2_allows_pass_when_converged_and_records_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
            conv_sha = c.bind_v2_contract(
                "convergence_policy",
                _convergence_policy().model_dump(mode="json"), stage="training")
            self._write_committee(c, {"202631": _CONVERGED_LOG,
                                      "202632": _CONVERGED_LOG})
            path = self._pass_bundle(c, "training")
            c.record_gate("training", votes_path=path)
            c = RunController(c.run_dir)
            self.assertEqual(c.stage("training")["gate"], "PASS")
            event = [e for e in c.state["events"] if e.get("type") == "gate"][-1]
            self.assertIn("framework_v2", event)
            self.assertEqual(event["framework_v2"]["convergence_status"],
                             "CONVERGED_AT_MAX")
            self.assertEqual(event["framework_v2"]["convergence_policy_sha256"],
                             conv_sha)
            fact = (c.run_dir / "framework_v2" / "gate_facts"
                    / "training.convergence_report.json")
            self.assertTrue(fact.is_file())

    def test_v2_refuses_pass_on_judge_contradiction(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            # Enable V2 (scope only; no convergence policy on this stage).
            c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
            v2_facts = [{
                "fact_id": "f1", "kind": "registered_checkpoint",
                "observed": True, "expected": True, "verdict": "FAIL",
                "validator": "itest", "rationale": "checkpoint missing"}]
            v2_judgments = [{
                "judgment_id": "j1", "judge_role": "reviewer",
                "cited_fact_ids": ["f1"],
                "claims": [{"claim_id": "c1", "about_kind": "registered_checkpoint",
                            "asserted_verdict": "PASS", "quote": "looks fine"}],
                "interpretation": "the checkpoint is fine",
                "verdict_advice": "PASS"}]
            path = self._pass_bundle(c, "training", v2_facts=v2_facts,
                                     v2_judgments=v2_judgments)
            with self.assertRaisesRegex(RuntimeError, "JUDGE_CONTRADICTION|contradicts"):
                c.record_gate("training", votes_path=path)
            c = RunController(c.run_dir)
            self.assertNotEqual(c.stage("training").get("gate"), "PASS")


class StageReviewSpecGateIntegrationTests(_Base):
    """The closure-directive review objects (StageReviewSpec + CanonicalReviewPacket
    + JudgeReview) enforced additively on a PASS through the real Controller."""

    def _spec_and_review(self, controller):
        from framework_v2.review_spec import default_stage_review_specs
        from framework_v2.review_packet import (
            CanonicalReviewPacketCompiler, JudgeReview, CriterionResult)
        from framework_v2.states import GateVerdict
        spec = default_stage_review_specs()["training"]
        review_spec_sha = controller.bind_v2_stage_review_spec(
            "training", spec.model_dump(mode="json"))
        packet = CanonicalReviewPacketCompiler().compile(
            packet_id="pk", run_id="v2-itest", stage="training",
            decision_id="d1", decision_sha256="dsha",
            validation_profile_id="vp", validation_profile_version=1,
            validation_profile_sha256="vpsha", stage_review_spec=spec,
            producer_rationale="trained committee is converged")
        reviews = []
        for lens in spec.lens_ids:
            crits = spec.criteria_for_lens(lens)
            reviews.append(JudgeReview(
                review_id=f"rev-{lens}", run_id="v2-itest", stage="training",
                lens_id=lens, packet_sha256=packet.packet_sha256(),
                stage_review_spec_sha256=spec.content_sha256(), verdict=GateVerdict.PASS,
                criteria_results=[CriterionResult(
                    criterion_id=c.criterion_id, lens_id=lens, ok=True,
                    value_read="verified") for c in crits],
                rationale="all criteria satisfied"))
        return spec, review_spec_sha, packet, reviews

    def test_v2_review_valid_unanimous_pass_records_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
            spec, review_spec_sha, packet, reviews = self._spec_and_review(c)
            v2_review = {"packet": packet.model_dump(mode="json"),
                         "reviews": [r.model_dump(mode="json") for r in reviews]}
            path = self._pass_bundle(c, "training", v2_review=v2_review)
            c.record_gate("training", votes_path=path)
            c = RunController(c.run_dir)
            self.assertEqual(c.stage("training")["gate"], "PASS")
            event = [e for e in c.state["events"] if e.get("type") == "gate"][-1]
            self.assertEqual(event["framework_v2"]["v2_review_spec_sha256"],
                             review_spec_sha)
            self.assertEqual(event["framework_v2"]["v2_packet_sha256"],
                             packet.packet_sha256())

    def test_v2_review_wrong_packet_sha_refuses_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
            spec, _sha, packet, reviews = self._spec_and_review(c)
            tampered = [r.model_copy(update={"packet_sha256": "deadbeef"}).model_dump(mode="json")
                        for r in reviews]
            v2_review = {"packet": packet.model_dump(mode="json"), "reviews": tampered}
            path = self._pass_bundle(c, "training", v2_review=v2_review)
            with self.assertRaisesRegex(RuntimeError, "INVALID_JUDGE_OUTPUT|refusing PASS"):
                c.record_gate("training", votes_path=path)
            c = RunController(c.run_dir)
            self.assertNotEqual(c.stage("training").get("gate"), "PASS")

    def test_v2_review_non_unanimous_refuses_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
            from framework_v2.states import GateVerdict
            spec, _sha, packet, reviews = self._spec_and_review(c)
            # flip the last review to a valid REVISE (failed criterion + fix)
            last = reviews[-1]
            revised_results = [cr.model_copy(update={"ok": False}) for cr in last.criteria_results]
            reviews[-1] = last.model_copy(update={
                "verdict": GateVerdict.REVISE, "criteria_results": revised_results,
                "required_fix": "address the unmet criterion"})
            v2_review = {"packet": packet.model_dump(mode="json"),
                         "reviews": [r.model_dump(mode="json") for r in reviews]}
            path = self._pass_bundle(c, "training", v2_review=v2_review)
            with self.assertRaisesRegex(RuntimeError, "unanimous PASS|refusing PASS"):
                c.record_gate("training", votes_path=path)


class DispatchCapabilityIntegrationTests(unittest.TestCase):
    def test_unmet_capability_is_blocked_through_real_dispatch(self):
        from runtimes.pydantic_ai.dispatch import (
            authorize_and_execute, default_registry, InMemoryApprovalStore,
            InMemoryIdempotencyStore)
        from runtimes.pydantic_ai.actions import (
            ROLE_ALLOWED_ACTIONS, APPROVAL_GATED_ACTIONS)
        chosen = None
        for role, actions in ROLE_ALLOWED_ACTIONS.items():
            for a in sorted(actions):
                if a == "acquire_structures" or APPROVAL_GATED_ACTIONS.get(a):
                    continue
                chosen = (role, a)
                break
            if chosen:
                break
        role, action = chosen
        reg = default_registry()
        reg[action].supported_capabilities = ("some.other.capability",)
        prop = {"requested_by_role": role, "action_type": action, "run_id": "r1",
                "idempotency_key": "k1",
                "parameters": {"required_capabilities":
                               ["acquisition.per_parent_augmentation_count"]}}
        out = authorize_and_execute(
            prop, registry=reg, approvals=InMemoryApprovalStore(),
            idempotency=InMemoryIdempotencyStore(), mode="dry_run")
        self.assertEqual(out.status, "BLOCKED_CAPABILITY")
        self.assertIn("FRAMEWORK_CAPABILITY_BLOCKER", out.reason)

    def test_met_capability_passes_through_real_dispatch(self):
        from runtimes.pydantic_ai.dispatch import (
            authorize_and_execute, default_registry, InMemoryApprovalStore,
            InMemoryIdempotencyStore)
        from runtimes.pydantic_ai.actions import (
            ROLE_ALLOWED_ACTIONS, APPROVAL_GATED_ACTIONS)
        chosen = None
        for role, actions in ROLE_ALLOWED_ACTIONS.items():
            for a in sorted(actions):
                if a == "acquire_structures" or APPROVAL_GATED_ACTIONS.get(a):
                    continue
                chosen = (role, a)
                break
            if chosen:
                break
        role, action = chosen
        reg = default_registry()
        reg[action].supported_capabilities = (
            "acquisition.per_parent_augmentation_count",)
        prop = {"requested_by_role": role, "action_type": action, "run_id": "r1",
                "idempotency_key": "k1",
                "parameters": {"required_capabilities":
                               ["acquisition.per_parent_augmentation_count"]}}
        out = authorize_and_execute(
            prop, registry=reg, approvals=InMemoryApprovalStore(),
            idempotency=InMemoryIdempotencyStore(), mode="dry_run")
        self.assertEqual(out.status, "DRY_RUN")


class EvidenceCompilerRuntimeConsumptionTests(unittest.TestCase):
    def test_build_bounded_evidence_consumes_deterministic_facts(self):
        from runtimes.pydantic_ai.bounded_evidence import build_bounded_evidence
        from framework_v2.facts import DeterministicFact, FactVerdict
        f = DeterministicFact(
            fact_id="pv-1", kind="physical_observable_within_tolerance",
            observed=1.0, expected=1.0, verdict=FactVerdict.PASS,
            validator="itest", rationale="within tolerance")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ev.json"
            payload = build_bounded_evidence([], out, facts=[f])
            self.assertEqual(len(payload["deterministic_facts"]), 1)
            self.assertEqual(payload["deterministic_facts"][0]["fact_id"], "pv-1")
            checks = [o["check"] for o in payload["validation_outcomes"]]
            self.assertIn("physical_observable_within_tolerance", checks)
            # The written bundle must match the returned payload's facts.
            on_disk = json.loads(out.read_text())
            self.assertEqual(on_disk["deterministic_facts"],
                             payload["deterministic_facts"])


if __name__ == "__main__":
    unittest.main()
