"""R16 forensic-defect regression suite (post-analysis corrective work; see recovery_taxonomy.py
and cli.py's ``_run_campaign_loop`` docstrings for the underlying fixes).

Covers, end to end and in isolation:
  1. RootCauseClassification.failure_category's generated JSON Schema is schema-visibly bound to
     the authoritative workflow.recovery_taxonomy registry (not a hidden validator on a plain str).
  2. An unregistered failure code is rejected before a recovery proposal is even attempted; every
     currently-registered code is accepted.
  3. A SINGLE run-campaign invocation drives a Judge REVISE gate straight through Analyst diagnosis
     and Orchestrator recovery-plan proposal to WAITING_FOR_HUMAN_APPROVAL, without ever returning
     RECOVERY_REQUIRED to the shell in between.
  4. An approval event recorded before a run's first campaign execution does not itself cause
     campaign_resumed on that first real invocation.
  5. No raw provider text or chain-of-thought is persisted for an accepted reasoning output --
     only the structured JSON the typed output model itself defines.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


class FailureTaxonomySchemaVisibilityTests(unittest.TestCase):
    def test_json_schema_enumerates_authoritative_registered_codes(self):
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from workflow import recovery_taxonomy

        schema = RootCauseClassification.model_json_schema()
        enum_def = schema["$defs"]["FailureCategory"]
        self.assertEqual(set(enum_def["enum"]), set(recovery_taxonomy.registered_codes()))
        # The schema's enum is the SOLE source of the allowed values seen by a provider enforcing
        # strict structured output -- not a second, independently-maintained list.
        prop = schema["properties"]["failure_category"]
        self.assertEqual(prop["$ref"], "#/$defs/FailureCategory")

    def test_unregistered_code_rejected_registered_codes_accepted(self):
        from pydantic import ValidationError
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from workflow import recovery_taxonomy

        base = dict(run_id="r", stage="s", evidence_summary="e", confidence=0.5,
                   recommended_recovery_target="s", recommended_next_action="a",
                   evidence_refs=[{"role": "x", "path": "p"}])
        with self.assertRaises(ValidationError):
            RootCauseClassification(failure_category="not_a_registered_code", **base)
        for code in recovery_taxonomy.registered_codes():
            inst = RootCauseClassification(failure_category=code, **base)
            self.assertEqual(inst.failure_category, code)
            self.assertEqual(inst.failure_domain, recovery_taxonomy.domain_of(code))


class NoRawProviderTextPersistedTests(unittest.TestCase):
    def test_reasoning_invocation_record_has_no_chain_of_thought_field(self):
        # Schema-level guarantee: RuntimeInvocationRecord has exactly raw_response (the structured
        # output's own JSON) and parsed_result -- no separate "reasoning"/"thoughts" field a
        # provider's chain-of-thought could ever be smuggled into.
        from runtimes.pydantic_ai.models import RuntimeInvocationRecord
        fields = set(RuntimeInvocationRecord.model_fields)
        self.assertIn("raw_response", fields)
        self.assertFalse(fields & {"reasoning", "thoughts", "chain_of_thought", "thinking"})

    def test_accepted_reasoning_artifact_is_exactly_the_typed_output_json(self):
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        classification = RootCauseClassification(
            run_id="r", stage="s", failure_category="dataset_coverage",
            evidence_refs=[{"role": "x", "path": "p"}], evidence_summary="e", confidence=0.5,
            recommended_recovery_target="s", recommended_next_action="a")
        persisted = json.loads(classification.model_dump_json())
        # Exactly the declared model fields -- nothing extra could have been smuggled in.
        self.assertEqual(set(persisted), set(RootCauseClassification.model_fields))


def _dataset(path: Path, n_frames: int, offset: int) -> Path:
    frames = []
    for i in range(n_frames):
        atoms = Atoms("Cu", positions=[[i + offset, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.info["structure_id"] = f"s{offset}-{i}"
        atoms.info["parent_structure_id"] = f"seed-pool:{900 + offset + i}"
        frames.append(atoms)
    write(str(path), frames)
    return path


def _single_stage_workflow(root: Path, dataset_path: Path) -> Path:
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "single-invocation-recovery-test",
        "inputs": [str(dataset_path)],
        "stages": [{
            "name": "stage_a", "command": None,
            "outputs": ["artifacts/stage_a_manifest.json"],
            "gate": {"criteria": ["dataset manifest is complete"]},
            "pydantic_ai": {
                "role": "data-curator", "action": "build_dataset_manifest",
                "idempotency_key": "single-invocation-recovery-test:stage_a:001",
                "parameters": {"dataset": str(dataset_path),
                              "manifest_path": "{artifacts_dir}/stage_a_manifest.json"},
            },
        }],
    }))
    return workflow


def _revise_vote(path: Path, lens: str, criteria: list) -> Path:
    path.write_text(json.dumps({
        "review_lens": lens, "verdict": "REVISE",
        "criteria_checked": [{"criterion": c, "value_read": "coverage gap", "ok": False}
                             for c in criteria],
        "rationale": "dataset does not cover the required composition",
        "required_fix": "rebuild the manifest from a corrected dataset",
    }))
    return path


class SingleInvocationGateToApprovalTests(unittest.TestCase):
    """The synthetic production-path proof: first launch -> campaign_started -> Stage A -> Judges
    REVISE -> (WITHOUT returning to the shell) -> Analyst RootCauseClassification -> Orchestrator
    RecoveryPlanProposal -> WAITING_FOR_HUMAN_APPROVAL, all from ONE run_campaign() call."""

    def test_gate_revise_flows_through_to_waiting_for_human_approval_in_one_call(self):
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.events import CampaignEventEmitter
        from workflow.controller import RunController
        from workflow.integrity import sha256_file
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root / "dataset_a.extxyz", 1, 0)
            workflow = _single_stage_workflow(root, dataset)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            lenses = [lens["id"] for lens in c.stage("stage_a")["gate_review_lenses"]]
            criteria = c.stage("stage_a")["gate_criteria"]
            vote_paths = [_revise_vote(root / f"revise-{i}.json", lens, criteria)
                         for i, lens in enumerate(lenses, 1)]

            # The Analyst's diagnosis cites stage_a's manifest artifact, which does not exist on
            # disk until run_campaign's own internal dispatch produces it -- but build_dataset_
            # manifest is a pure function of the dataset file's bytes + manifest_path, so its
            # content (and hence sha256) can be precomputed to a scratch path without going
            # through the controller/idempotency system at all, then the REAL in-run_campaign
            # dispatch reproduces byte-identical content at the real path.
            from runtimes.pydantic_ai.deterministic_executors import build_dataset_manifest
            from runtimes.pydantic_ai.root_cause import RootCauseClassification
            manifest_path = str((run_dir / "artifacts" / "stage_a_manifest.json").resolve())
            scratch_manifest = root / "scratch_stage_a_manifest.json"
            build_dataset_manifest({"parameters": {"dataset": str(dataset),
                                                   "manifest_path": str(scratch_manifest)}})
            manifest_sha256 = sha256_file(scratch_manifest)

            classification_payload = {
                "run_id": "single-invocation-recovery-test", "stage": "stage_a",
                "failure_category": "dataset_coverage",
                "evidence_refs": [{"role": "data-curator", "path": manifest_path,
                                   "integrity": {"sha256": manifest_sha256}}],
                "evidence_summary": "stage_a's dataset manifest lacks required coverage",
                "confidence": 0.75, "recommended_recovery_target": "stage_a",
                "recommended_next_action": "rebuild the stage_a manifest from a corrected dataset",
            }
            analyst_response_path = root / "mock_analyst_response.json"
            analyst_response_path.write_text(json.dumps(classification_payload))

            # The diagnosis artifact's sha256 is deterministic: sha256(model_dump_json(indent=2)
            # + "\n") of the EXACT accepted RootCauseClassification instance -- precompute it from
            # an equivalent instance the same way production_router._persist_reasoning_artifact
            # will, so the orchestrator response can bind to it up front.
            classification = RootCauseClassification(**classification_payload)
            diagnosis_sha256 = hashlib.sha256(
                (classification.model_dump_json(indent=2) + "\n").encode()).hexdigest()

            orchestrator_payload = {
                "run_id": "single-invocation-recovery-test", "failed_stage": "stage_a",
                "diagnosis_artifact_sha256": diagnosis_sha256,
                "capability": "data_repair", "return_stage": "stage_a",
                "proposed_changes": [{"type": "rebuild_manifest_from_corrected_dataset"}],
                "labeling": {"teacher_relabel": False, "new_dft": False},
                "student_training": {"retrain": False, "mode": "none"},
                "revalidation": {"reuse_profile": True, "targets": ["stage_a"]},
                "rationale": "rebuild stage_a's manifest from a dataset that fixes the coverage gap",
            }
            orchestrator_response_path = root / "mock_orchestrator_response.json"
            orchestrator_response_path.write_text(json.dumps(orchestrator_payload))

            emitter = CampaignEventEmitter(run_dir, run_id=c.state.get("run_id"), quiet=True)
            result = cli.run_campaign(
                c, runtime="mock", repo_root=str(ROOT),
                mock_judge_response=[str(p) for p in vote_paths],
                mock_analyst_response=str(analyst_response_path),
                mock_orchestrator_response=str(orchestrator_response_path),
                max_iterations=20, emitter=emitter)

            self.assertEqual(result.outcome, cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL, result.message)
            self.assertEqual(result.exit_code, cli.EXIT_APPROVAL_REQUIRED)

            events = [json.loads(line) for line in
                     (run_dir / "campaign_events.jsonl").read_text().splitlines()]
            names = [e["event"] for e in events]
            self.assertEqual(names[0], "campaign_started")
            self.assertIn("gate_recorded", names)
            gate_event = next(e for e in events if e["event"] == "gate_recorded")
            self.assertEqual(gate_event["detail"]["decision"], "REVISE")
            self.assertIn("recovery_started", names)
            self.assertIn("recovery_proposed", names)
            # The old defect: an intermediate RECOVERY_REQUIRED pause/return to the shell between
            # the gate REVISE and the Analyst dispatch. There must be exactly ONE terminal
            # campaign_* event (this whole flow happened inside a single run_campaign call), and
            # it must be the pause for human approval, never a recovery-required stop.
            terminal_events = [e for e in events if e["event"].startswith("campaign_")
                              and e["event"] not in ("campaign_started", "campaign_resumed")]
            self.assertEqual(len(terminal_events), 1, terminal_events)
            self.assertEqual(terminal_events[0]["detail"]["outcome"],
                             cli.CAMPAIGN_WAITING_FOR_HUMAN_APPROVAL)

            c = RunController(run_dir)
            self.assertEqual(c.state["pending_recovery"]["status"], "proposed")


class ApprovalBeforeFirstLaunchTests(unittest.TestCase):
    def test_approval_event_before_first_launch_is_not_campaign_resumed(self):
        # An approve/approve-recovery command writes its OWN event (approval_granted) to the same
        # durable campaign_events.jsonl via its own CampaignEventEmitter -- this must not itself
        # make the run's actual first run-campaign invocation see campaign_resumed.
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.events import CampaignEventEmitter
        from workflow.controller import RunController

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root / "dataset_a.extxyz", 1, 0)
            workflow = _single_stage_workflow(root, dataset)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            c = RunController(run_dir)

            # Simulate exactly what _cmd_approve/_cmd_approve_recovery do: emit approval_granted
            # via a fresh CampaignEventEmitter, before run-campaign has ever executed once.
            CampaignEventEmitter(run_dir, quiet=True).emit(
                "approval_granted", detail={"approval_boundary": "some_boundary"})
            self.assertTrue((run_dir / "campaign_events.jsonl").exists())

            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20)
            self.assertEqual(result.outcome, cli.CAMPAIGN_COMPLETED, result.message)

            events = [json.loads(line) for line in
                     (run_dir / "campaign_events.jsonl").read_text().splitlines()]
            lifecycle = [e["event"] for e in events if e["event"] in
                        ("campaign_started", "campaign_resumed")]
            self.assertEqual(lifecycle, ["campaign_started"])

    def test_campaign_previously_executed_helper_directly(self):
        from runtimes.pydantic_ai.events import CampaignEventEmitter, campaign_previously_executed

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self.assertFalse(campaign_previously_executed(run_dir))
            emitter = CampaignEventEmitter(run_dir, quiet=True)
            emitter.emit("approval_granted", detail={"approval_boundary": "x"})
            self.assertTrue((run_dir / "campaign_events.jsonl").exists())
            self.assertFalse(campaign_previously_executed(run_dir))
            emitter.emit("campaign_started")
            self.assertTrue(campaign_previously_executed(run_dir))


if __name__ == "__main__":
    unittest.main()
