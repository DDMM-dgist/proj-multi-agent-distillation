"""Production-readiness integration audit (Priority #1/#2/#3, generic-framework check).

SiO2-x / Allegro -> SIMPLE-NN is only the CURRENT reference campaign; nothing in this module may
depend on that chemistry, model family, dataset size, or domain label. This file drives the real
production entry points -- ``workflow.controller.RunController``,
``runtimes.pydantic_ai.dispatch``/``controller_bridge``/``orchestrator_bridge``/``recovery_bridge``/
``root_cause``/``bounded_evidence`` -- through the complete bounded-autonomy recovery lifecycle
with a tiny, deterministic, semantically-generic fixture campaign:

    initialize -> freeze scientific contract -> produce/consume evidence -> gate -> diagnose
    -> propose recovery -> human approval -> authorization envelope -> bounded corrective action
    -> fresh evidence -> revalidation -> same-stage gate -> recovery resolved

No parallel/bypass test path is constructed: every step below calls the same
``RunController``/dispatch methods a human operator, the CLI, or a PydanticAI runtime would call.
No production scientific compute (Teacher inference, MD, DFT, Student training) runs anywhere in
this file.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from runtimes.pydantic_ai import bounded_evidence
from runtimes.pydantic_ai.controller_bridge import dispatch_via_controller
from runtimes.pydantic_ai.dispatch import ActionDescriptor, default_registry
from runtimes.pydantic_ai.orchestrator_bridge import (
    OrchestratorActionProposal, dispatch_orchestrator_action,
)
from runtimes.pydantic_ai.recovery_bridge import build_recovery_plan_draft
from runtimes.pydantic_ai.models import EvidenceReference
from runtimes.pydantic_ai.root_cause import RootCauseClassification, validate_root_cause_classification
from workflow.controller import RunController
from workflow.integrity import sha256_file

GATE_CRITERION = "artifact is complete and internally consistent"

# Deliberately abstract, non-chemistry, non-SiO2/Allegro/SIMPLE-NN population/domain labels: this
# fixture must read the same whether the underlying campaign is a materials-science distillation
# run or something else entirely.
DEPLOYMENT_DOMAIN = {"slice_classes": ["group_a", "group_b"], "parameter_range": [0, 1]}
SPLIT_POLICY = {"seed": 1, "validation_fraction": 0.2, "test_fraction": 0.2,
                "grouping_key": "group_id"}


class FixtureHelpers(unittest.TestCase):
    """Shared builders. No test_* methods here: zero tests are collected from this class."""

    # --- contract sources (Priority #1) --------------------------------------------------
    def _write_contract_sources(self, root: Path, *, deployment_domain=None):
        domain = deployment_domain or DEPLOYMENT_DOMAIN
        scope = root / "distillation_scope.yaml"
        scope.write_text(yaml.safe_dump({"deployment_domain": domain}))
        profile = root / "validation_profile.yaml"
        profile.write_text(yaml.safe_dump({
            "deployment_domain": domain, "shared_md_protocol": "protocol-v1",
            "checks": ["check_a", "check_b"],
        }))
        policy = root / "dataset_policy.yaml"
        policy.write_text(yaml.safe_dump({"split_policy": SPLIT_POLICY}))
        return {"distillation_scope": str(scope), "validation_profile": str(profile),
               "dataset_policy": str(policy)}

    def _workflow_cfg(self, root: Path, *, run_id="fixture-run", stages=None,
                      recovery_capability_roster=None, recovery_policy=None,
                      protected_reference_roles=None, contract_sources=True):
        cfg = {
            "run_id": run_id,
            "stages": stages or [
                {"name": "prepare", "command": None, "outputs": ["artifacts/baseline.json"],
                 "produces_student_results": True, "gate": {"criteria": [GATE_CRITERION]}},
                {"name": "produce_evidence", "command": None,
                 "outputs": ["artifacts/evidence.json"],
                 "produces_student_results": True, "gate": {"criteria": [GATE_CRITERION]}},
            ],
        }
        if contract_sources:
            cfg["validation_contract_sources"] = self._write_contract_sources(root)
        if recovery_capability_roster is not None:
            cfg["recovery_capability_roster"] = recovery_capability_roster
        if recovery_policy is not None:
            cfg["recovery_policy"] = recovery_policy
        if protected_reference_roles is not None:
            cfg["protected_reference_roles"] = protected_reference_roles
        path = root / "workflow.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return path

    def _init_controller(self, root: Path, **cfg_kwargs) -> RunController:
        cfg_path = self._workflow_cfg(root, **cfg_kwargs)
        return RunController.initialize(cfg_path, root / "run")

    # --- gate voting (mirrors tests/test_controller.py's established helper shape) ----------
    def _vote_bundle(self, controller: RunController, stage: str, verdict: str):
        artifacts = {a["path"]: a["sha256"] for a in controller.stage_artifacts(stage)}
        criteria = controller.stage(stage).get("gate_criteria")
        lenses = controller.stage(stage).get("gate_review_lenses")
        votes = [{"judge_id": f"judge-{i}", "review_lens": lens["id"], "verdict": verdict,
                 "criteria_checked": [{"criterion": c, "value_read": "checked",
                                       "ok": verdict == "PASS"} for c in criteria],
                 "rationale": "ok" if verdict == "PASS" else "needs correction",
                 "required_fix": "" if verdict == "PASS" else "see recovery"}
                for i, lens in enumerate(lenses, 1)]
        vote_path = controller.run_dir / "gates" / f"{stage}.{verdict}.votes.json"
        vote_path.write_text(json.dumps({"stage": stage, "criteria": criteria,
                                         "review_lenses": lenses, "artifact_sha256": artifacts,
                                         "decision": verdict, "votes": votes}))
        return vote_path

    def _gate(self, controller, stage, verdict):
        controller.record_gate(stage, votes_path=self._vote_bundle(controller, stage, verdict))

    # --- generic Priority #2 structural-coverage evidence fixture ---------------------------
    def _coverage_evidence_payload(self, *, unsupported_count=20, total=80, seed=""):
        stat = lambda n_unmatched, n_matched: {
            "n": n_matched, "n_unmatched": n_unmatched,
            "unmatched_fraction": n_unmatched / (n_matched + n_unmatched),
            "mean": 0.5, "p50": 0.4, "p75": 0.6, "p90": 0.8, "p95": 0.9, "p99": 0.95, "max": 1.2,
        }
        n_matched = total - unsupported_count
        return {
            "direction": "query_to_reference",
            "query_population": f"candidate_pool_v1{seed}",
            "reference_population": "reference_training_partition_v1",
            "n_query_environments": total, "n_query_structures": 10,
            "overall_global_summary": stat(unsupported_count, n_matched),
            "query_slice_resolved_summaries": {
                "group_a": stat(unsupported_count, n_matched), "group_b": stat(0, 40),
            },
            "reference_slice_resolved_summaries": {
                "group_a": stat(0, 40), "group_b": stat(0, 40),
            },
            "provenance": {
                "representation_hash": "a" * 64, "reference_manifest_sha256": "b" * 64,
                "reference_slice_counts": {"group_a": 40, "group_b": 40},
                "reference_total_atoms": 1000, "reference_total_frames": 50,
                # Representation/search-backend internals: bounded_evidence's Analyst-facing
                # summary must never surface these (Priority #2 / genericity requirement).
                "representation_provenance": {"kind": "generic_descriptor",
                                              "soap_hyperparameters": {"n_max": 8, "l_max": 6}},
                "search_backend_provenance": {"backend": "generic_nn_search",
                                             "cktree_workers": 4},
            },
            "excluded_partitions": ["group_c_excluded_small_sample"],
        }

    def _write_json(self, path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return path


class FullBoundedAutonomyLifecycleTest(FixtureHelpers):
    """The 17-step happy-path lifecycle, driven end to end through real production entry points."""

    def test_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = self._init_controller(
                root, recovery_capability_roster={"data_repair": "data-curator",
                                                  "orchestration": "orchestrator"})

            # --- 1/2: fresh run, validation contract exists immediately (Priority #1) -------
            self.assertIsNotNone(controller.state["validation_contract"])
            contract_sha_at_init = controller.state["validation_contract"]["contract_sha256"]

            # External source mutation after init never alters the run (Priority #1).
            (root / "distillation_scope.yaml").write_text(
                yaml.safe_dump({"deployment_domain": {"slice_classes": ["mutated"],
                                                       "parameter_range": [9, 9]}}))
            reloaded = RunController(controller.run_dir)
            self.assertEqual(reloaded.state["validation_contract"]["contract_sha256"],
                             contract_sha_at_init)

            # --- 3: an earlier stage produces evidence and passes cleanly -------------------
            baseline = controller.run_dir / "artifacts/baseline.json"
            self._write_json(baseline, {"role": "baseline", "revision": 1})
            controller.complete_external_stage("prepare", [baseline])
            self._gate(controller, "prepare", "PASS")

            # --- 3: the target stage produces (generic, Priority #2-shaped) evidence --------
            evidence_path = controller.run_dir / "artifacts/evidence.json"
            self._write_json(evidence_path, self._coverage_evidence_payload())
            controller.complete_external_stage("produce_evidence", [evidence_path])

            # --- 4: Gate REVISE -------------------------------------------------------------
            self._gate(controller, "produce_evidence", "REVISE")
            pending = controller.state["pending_recovery"]
            self.assertEqual(pending["status"], "required")
            self.assertEqual(pending["failed_stage"], "produce_evidence")
            frozen_hash = pending["artifact_sha256"][str(evidence_path)]
            self.assertEqual(frozen_hash, sha256_file(evidence_path))

            # --- 6: Analyst root-cause classification, evidence-bound, consuming ONLY the
            # generic bounded-evidence summary (Priority #2) -----------------------------------
            summary = bounded_evidence.summarize_artifact(evidence_path)
            coverage_summary = summary["structural_coverage_evidence"]
            self.assertEqual(set(coverage_summary), {
                "direction", "query_population", "reference_population", "n_query_environments",
                "n_query_structures", "reference_population_counts",
                "overall_distance_distribution", "query_slice_memberships",
                "query_slice_distance_distributions", "reference_slice_memberships",
                "reference_slice_distance_distributions", "provenance_hashes", "limitations",
            })
            leaked = json.dumps(coverage_summary).lower()
            self.assertNotIn("soap", leaked)
            self.assertNotIn("cktree", leaked)
            self.assertNotIn("descriptor", leaked)

            classification = RootCauseClassification(
                run_id="fixture-run", stage="produce_evidence", failure_category="dataset_coverage",
                evidence_refs=[EvidenceReference(role="coverage_evidence", path=str(evidence_path),
                                                 integrity={"sha256": frozen_hash})],
                evidence_summary=(
                    f"{coverage_summary['overall_distance_distribution']['unsupported_count']} of "
                    f"{coverage_summary['n_query_environments']} query environments "
                    f"({coverage_summary['overall_distance_distribution']['unsupported_fraction']:.0%}) "
                    "are unsupported by the reference population in slice group_a."
                ),
                confidence=0.8, recommended_recovery_target="prepare",
                recommended_next_action="augment the candidate pool with deployment-representative frames",
            )
            validate_root_cause_classification(
                classification, available_artifacts=[str(evidence_path)],
                valid_recovery_targets=[s["name"] for s in controller.state["stages"]])
            diagnosis_path = self._write_json(controller.run_dir / "diagnosis.json",
                                              json.loads(classification.model_dump_json()))

            # --- 7: Orchestrator builds a provenance-bound RecoveryPlan draft (Priority #3) --
            draft = build_recovery_plan_draft(
                classification, proposed_by={"actor_kind": "system", "canonical_id": "orchestrator"},
                failed_stage="produce_evidence", capability="data_repair",
                return_stage="prepare", proposed_changes=[{"type": "add_deployment_frames"}],
                labeling={"teacher_relabel": True, "new_dft": False},
                student_training={"retrain": False, "mode": "none"},
                revalidation={"reuse_profile": True, "targets": ["prepare", "produce_evidence"]},
                diagnosis_artifact_path=str(diagnosis_path),
                diagnosis_artifact_sha256=sha256_file(diagnosis_path),
            )
            plan_path = self._write_json(controller.run_dir / "plan.json", draft.to_plan_json())

            # --- 8: proposal via the real Orchestrator bridge; never self-executes/approves -
            proposal = OrchestratorActionProposal(
                run_id="fixture-run", stage="produce_evidence", requested_at="t",
                rationale="frozen evidence indicates a coverage gap",
                idempotency_key="propose-1", action_type="propose_recovery",
                parameters={"run_dir": str(controller.run_dir), "plan_path": str(plan_path)})
            outcome = dispatch_orchestrator_action(proposal, controller=controller, mode="primary")
            self.assertEqual(outcome.status, "EXECUTED")
            self.assertEqual(controller.state["pending_recovery"]["status"], "proposed")
            recovery_id = controller.state["pending_recovery"]["recovery_id"]
            recovery = next(r for r in controller.state["recoveries"] if r["id"] == recovery_id)
            self.assertEqual(recovery["proposed_by"],
                             {"actor_kind": "system", "canonical_id": "orchestrator",
                              "display_name": None})
            self.assertEqual(recovery["resolved_responsible_agent"], "data-curator")

            # An automated actor (even the trusted proposer identity) can never approve.
            with self.assertRaises(ValueError):
                controller.approve_recovery({"actor_kind": "system", "canonical_id": "orchestrator"})

            # --- 9: trusted human approves ----------------------------------------------------
            controller.approve_recovery({"actor_kind": "human", "canonical_id": "dr-lee"},
                                        note="approved for fixture recovery")

            # --- 10: start the recovery iteration; earlier stage(s) legitimately rerun --------
            controller.start_iteration()
            self.assertEqual(controller.stage("prepare")["status"], "pending")
            self.assertEqual(controller.stage("produce_evidence")["status"], "pending")

            # --- 10: bounded RecoveryAuthorizationEnvelope, human-issued, distinct from
            # approve_recovery itself ----------------------------------------------------------
            envelope = controller.authorize_recovery_capabilities(
                {"actor_kind": "human", "canonical_id": "dr-lee"},
                action_types=["label_with_teacher"], resource_limits={"cpu_hours": 10})

            # --- 11: responsible capability executes a fixture corrective action inside the
            # envelope, through the real dispatch/controller-bridge path ----------------------
            registry = default_registry()

            def _fixture_corrective_executor(proposal):
                new_baseline = controller.run_dir / "artifacts/baseline.json"
                self._write_json(new_baseline, {"role": "baseline", "revision": 2,
                                                "recovery_id": recovery_id})
                return {"path": str(new_baseline), "sha256": sha256_file(new_baseline)}

            registry["label_with_teacher"] = ActionDescriptor(
                action_type="label_with_teacher", role="data-curator",
                approval_boundary="costly_teacher_labeling", executor=_fixture_corrective_executor)
            child_proposal = {
                "requested_by_role": "data-curator", "action_type": "label_with_teacher",
                "idempotency_key": "corrective-1", "run_id": "fixture-run", "stage": "prepare",
                "requested_at": "t", "rationale": "apply orchestrator-authorized recovery",
                "parameters": {"resource_usage": {"cpu_hours": 3}},
            }
            child_outcome = dispatch_via_controller(child_proposal, controller=controller,
                                                    registry=registry, mode="primary")
            self.assertEqual(child_outcome.status, "EXECUTED")
            self.assertEqual(child_outcome.recovery_authorization_envelope_sha256,
                             envelope["envelope_sha256"])

            # --- 12: changed artifact registered under the SAME contract ----------------------
            controller.complete_external_stage("prepare", [Path(child_outcome.artifact["path"])])
            self._gate(controller, "prepare", "PASS")

            new_evidence_path = controller.run_dir / "artifacts/evidence.json"
            self._write_json(new_evidence_path,
                             self._coverage_evidence_payload(unsupported_count=5, seed="-augmented"))
            controller.complete_external_stage("produce_evidence", [new_evidence_path])
            self.assertIsNotNone(controller.state["validation_contract"])
            self.assertEqual(controller.state["validation_contract"]["contract_sha256"],
                             contract_sha_at_init)

            # --- recovery PASS must not be reachable before execution verification -----------
            with self.assertRaisesRegex(RuntimeError, "cannot PASS until recovery execution"):
                self._gate(controller, "produce_evidence", "PASS")

            # --- 13: verify the recovery genuinely changed evidence ---------------------------
            report = {
                "schema_version": 1, "recovery_id": recovery_id, "previous_iteration": 1,
                "current_iteration": 2,
                "changes": [{"type": "add_deployment_frames", "status": "APPLIED",
                            "evidence_artifacts": [str(new_evidence_path)]}],
                "labeling": {"teacher_relabel": True, "teacher_relabel_stage": "prepare",
                            "new_dft": False, "new_dft_stage": None},
                "student_training": {"retrain": False, "mode": "none", "stage": None},
                "revalidation": {"targets": ["prepare", "produce_evidence"],
                                 "stages": ["prepare", "produce_evidence"]},
            }
            verification = controller.verify_recovery_execution(
                self._write_json(controller.run_dir / "execution_report.json", report))
            self.assertEqual(verification["status"], "verified")

            # --- 15/16: same failed stage gated again; PASS resolves the exact recovery -------
            self._gate(controller, "produce_evidence", "PASS")
            self.assertEqual(controller.stage("produce_evidence")["gate"], "PASS")
            resolved = next(r for r in controller.state["recoveries"] if r["id"] == recovery_id)
            self.assertEqual(resolved["status"], "resolved")
            self.assertIsNone(controller.state["pending_recovery"])

            # --- 17: final manifest carries the complete provenance chain ---------------------
            reloaded = RunController(controller.run_dir)
            final_recovery = next(r for r in reloaded.state["recoveries"] if r["id"] == recovery_id)
            self.assertEqual(final_recovery["status"], "resolved")
            self.assertIn("diagnosis_binding", final_recovery["plan"])
            self.assertEqual(final_recovery["plan"]["diagnosis_binding"]["diagnosis_artifact_sha256"],
                             sha256_file(diagnosis_path))
            # The controller re-labels the (aliased) recovery_execution record's status from
            # "verified" to "resolved" the moment the same stage's gate PASSes (see
            # RunController.record_gate) -- "verified" is the pre-PASS transient state, not the
            # terminal one.
            self.assertEqual(final_recovery["execution"]["status"], "resolved")
            self.assertEqual(final_recovery["execution"]["change_types"], ["add_deployment_frames"])
            self.assertEqual(reloaded.state["validation_contract"]["contract_sha256"],
                             contract_sha_at_init)


if __name__ == "__main__":
    unittest.main()
