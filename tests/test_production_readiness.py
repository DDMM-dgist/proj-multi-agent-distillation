from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


def _judge_vote(path: Path, lens: str, verdict="PASS") -> Path:
    path.write_text(json.dumps({
        "review_lens": lens,
        "verdict": verdict,
        "criteria_checked": [{"criterion": "committee manifest is complete", "value_read": "ok", "ok": verdict == "PASS"}],
        "rationale": "checked independently",
        "required_fix": "" if verdict == "PASS" else "fix the artifact",
    }))
    return path


def _three_judge_votes(root: Path, criteria, lenses):
    paths = []
    for i, lens in enumerate(lenses, 1):
        path = root / f"judge-{i}.json"
        path.write_text(json.dumps({
            "review_lens": lens["id"],
            "verdict": "PASS",
            "criteria_checked": [{"criterion": c, "value_read": "ok", "ok": True} for c in criteria],
            "rationale": f"judge {i} checked frozen evidence only",
            "required_fix": "",
        }))
        paths.append(path)
    return paths




def _protected_reference_package(root: Path, protected_atoms) -> Path:
    from workflow.integrity import sha256_file
    ref = root / "protected.xyz"
    write(str(ref), [protected_atoms])
    indices = root / "protected_indices.txt"
    indices.write_text("760\n761\n")
    manifest = root / "protected_manifest.json"
    manifest.write_text(json.dumps({"mapping": {
        "logical_test_frames": 1,
        "matched_logical_frames": 1,
        "unmatched_logical_frames": 0,
        "protected_source_rows": 2,
        "conflicting_label_duplicates": 0,
    }}))
    reference = root / "reference.yaml"
    reference.write_text("\n".join([
        "kind: protected-existing-dft",
        "reference_id: test-reference",
        "reference_class: ORIGINAL_TEACHER_TEST",
        "status: AVAILABLE_AND_PROTECTED",
        "logical_test_frames: 1",
        "protected_source_rows: 2",
        f"protection_manifest: {manifest}",
        f"protected_source_rows_file: {indices}",
        "duplicate_equivalent:",
        "  source_global_indices: [760, 761]",
        "  label_conflict: false",
        "prohibited_uses: [student_training, student_validation_tuning, acquisition_seed, augmentation_parent, recovery_training]",
        "structures:",
        f"  path: {ref}",
        "  logical_frames: 1",
        f"  sha256: {sha256_file(ref)}",
        "",
    ]))
    return reference

def _mock_training_workflow(root: Path, *, missing_extra=False, no_outputs=False, protected_reference=None, protected_dataset=False):
    dataset = root / "train.extxyz"
    frames = []
    for i in range(3):
        pos = [0, 0, 0] if protected_dataset else [i, 0, 0]
        a = Atoms("Cu", positions=[pos], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"s{i}"
        a.info["parent_structure_id"] = f"seed-pool:{760 if protected_dataset else 900 + i}"
        frames.append(a)
    write(str(dataset), frames)
    student_cfg = root / "student.yaml"
    student_cfg.write_text("kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n")
    outputs = [] if no_outputs else ["artifacts/student_committee.manifest.json", "artifacts/committee"]
    if missing_extra:
        outputs.append("artifacts/required-but-missing.json")
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "synthetic-prod-ready",
        "inputs": [str(student_cfg), str(dataset), *([str(protected_reference)] if protected_reference else [])],
        "stages": [{
            "name": "training",
            "command": None,
            "outputs": outputs,
            "pydantic_ai": {
                "role": "ml-trainer",
                "action": "train_committee",
                "approval_boundary": "costly_training",
                "idempotency_key": "synthetic-training-001",
                "parameters": {
                    "student_config": str(student_cfg),
                    "dataset": str(dataset),
                    "output_dir": "{artifacts_dir}/committee",
                    "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
                },
            },
            "gate": {"criteria": ["committee manifest is complete"]},
        }],
    }))
    return workflow


def _authoritative_response(run_dir, stage="training"):
    from workflow.controller import RunController
    from runtimes.pydantic_ai.cli import _proposal_from_stage, _stage_config
    c = RunController(run_dir)
    proposal, _ = _proposal_from_stage(c, stage, _stage_config(c, stage))
    return proposal


class ProductionReadinessTests(unittest.TestCase):
    def test_bounded_evidence_summary_is_hash_linked_and_small(self):
        from runtimes.pydantic_ai.bounded_evidence import MAX_EVIDENCE_BYTES, build_bounded_evidence
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_path = root / "frames.extxyz"
            frames = []
            for i in range(5):
                a = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
                          cell=[8, 8, 8], pbc=True)
                a.info["structure_id"] = f"s{i}"
                a.info["parent_structure_id"] = f"p{i}"
                a.info["config_type"] = "synthetic"
                frames.append(a)
            write(str(frames_path), frames)
            out = root / "evidence.json"
            summary = build_bounded_evidence([frames_path], out)
            self.assertTrue(out.stat().st_size <= MAX_EVIDENCE_BYTES)
            self.assertEqual(summary["artifacts"][0]["n_frames"], 5)
            self.assertEqual(summary["artifacts"][0]["integrity"]["kind"], "file")
            self.assertEqual(summary["summary_sha256"], summary["summary_sha256"].lower())

    def test_synthetic_stage_runner_approval_executor_controller_and_three_judges(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.extxyz"
            frames = []
            for i in range(3):
                a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
                a.info["structure_id"] = f"s{i}"
                a.info["parent_structure_id"] = f"p{i}"
                frames.append(a)
            write(str(dataset), frames)
            student_cfg = root / "student.yaml"
            student_cfg.write_text("kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n")
            workflow = _mock_training_workflow(root)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            with self.assertRaises(SystemExit):
                cli.main(["run-stage", "--run-dir", str(run_dir), "--stage", "training"])
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir), "--stage", "training"])
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training",
                                       "--note", "synthetic test approval"]), cli.EXIT_SUCCESS)
            lenses = RunController(run_dir).stage("training")["gate_review_lenses"]
            vote_paths = _three_judge_votes(root, ["committee manifest is complete"], lenses)
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir), "--stage", "training",
                             "--mock-judge-response", str(vote_paths[0]),
                             "--mock-judge-response", str(vote_paths[1]),
                             "--mock-judge-response", str(vote_paths[2])])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            c = RunController(run_dir)
            self.assertEqual(c.stage("training")["status"], "completed")
            self.assertEqual(c.stage("training")["gate"], "PASS")
            self.assertTrue((run_dir / "artifacts" / "student_committee.manifest.json").is_file())
            self.assertEqual(len(c.state.get("idempotency", {})), 1)
            gate_events = [e for e in c.state["events"] if e.get("type") == "gate"]
            self.assertEqual(gate_events[-1]["vote_bundle"]["decision"], "PASS")
            self.assertEqual(len(gate_events[-1]["vote_bundle"]["votes"]), 3)
            judge_prov = list((run_dir / "exchange" / "provenance").glob("training-judge-*.json"))
            self.assertEqual(len(judge_prov), 3)


    def test_run_stage_fails_on_missing_declared_output(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _mock_training_workflow(root, missing_extra=True)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training", "--note", "test"]), cli.EXIT_SUCCESS)
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir), "--stage", "training",
                             "--auto-mock-judges"])
            self.assertNotEqual(code, cli.EXIT_SUCCESS)
            self.assertNotEqual(RunController(run_dir).stage("training")["status"], "completed")

    def test_run_stage_fails_when_stage_cannot_complete(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _mock_training_workflow(root, no_outputs=True)
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_training", "--note", "test"]), cli.EXIT_SUCCESS)
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir), "--stage", "training"])
            self.assertNotEqual(code, cli.EXIT_SUCCESS)
            self.assertEqual(RunController(run_dir).stage("training")["status"], "pending")

    def test_teacher_baseline_is_routable_end_to_end(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({
                "run_id": "teacher-baseline-route",
                "inputs": [
                    str(ROOT / "examples/mock/teacher.yaml"),
                    str(ROOT / "examples/mock/validation.yaml"),
                    str(ROOT / "examples/mock/distillation_scope.yaml"),
                    str(ROOT / "examples/mock/seed.example.xyz"),
                ],
                "stages": [{
                    "name": "teacher_baseline",
                    "command": None,
                    "outputs": ["artifacts/teacher_baseline.json"],
                    "contract": {"kind": "validation_manifest",
                                  "manifest": "artifacts/teacher_baseline.json",
                                  "validator": "validation.teacher_baseline.validate_teacher_baseline_report",
                                  "options": {"accepted_applicability": ["SUPPORTED", "CONDITIONAL", "NOT_ESTABLISHED"]}},
                    "pydantic_ai": {
                        "role": "simulation",
                        "action": "build_teacher_baseline",
                        "approval_boundary": "costly_teacher_labeling",
                        "parameters": {
                            "teacher_config": str(ROOT / "examples/mock/teacher.yaml"),
                            "validation_profile": str(ROOT / "examples/mock/validation.yaml"),
                            "distillation_scope": str(ROOT / "examples/mock/distillation_scope.yaml"),
                            "structures_path": str(ROOT / "examples/mock/seed.example.xyz"),
                            "report_path": "{artifacts_dir}/teacher_baseline.json",
                            "deployment_domain": {"dilute_oxygen_vacancy": "NOT_ESTABLISHED"},
                            "applicability_status": "NOT_ESTABLISHED",
                            "applicability_limitations": ["dilute_oxygen_vacancy = NOT_ESTABLISHED"],
                            "require_lineage": False,
                        },
                    },
                    "gate": {"criteria": ["teacher baseline report is valid"]},
                }],
            }))
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            self.assertEqual(cli.main(["approve", "--run-dir", str(run_dir),
                                       "--boundary", "costly_teacher_labeling", "--note", "test"]), cli.EXIT_SUCCESS)
            lenses = RunController(run_dir).stage("teacher_baseline")["gate_review_lenses"]
            votes = _three_judge_votes(root, ["teacher baseline report is valid"], lenses)
            for path in votes:
                payload = json.loads(path.read_text())
                payload["criteria_checked"] = [{"criterion": "teacher baseline report is valid", "value_read": "ok", "ok": True}]
                path.write_text(json.dumps(payload))
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--stage", "teacher_baseline",
                             "--mock-judge-response", str(votes[0]),
                             "--mock-judge-response", str(votes[1]),
                             "--mock-judge-response", str(votes[2])])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            c = RunController(run_dir)
            self.assertEqual(c.stage("teacher_baseline")["gate"], "PASS")
            report = json.loads((run_dir / "artifacts/teacher_baseline.json").read_text())
            self.assertEqual(report["applicability"]["status"], "NOT_ESTABLISHED")
            self.assertEqual(report["checks"][0]["purpose"], "deployment_stability")
            self.assertEqual(report["deployment_domain"]["dilute_oxygen_vacancy"], "NOT_ESTABLISHED")
            self.assertIn("dilute_oxygen_vacancy = NOT_ESTABLISHED", report["applicability"]["limitations"])


    def test_authoritative_proposal_tampering_is_rejected_before_executor(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            RunController.initialize(_mock_training_workflow(root), run_dir)
            cli.main(["approve", "--run-dir", str(run_dir), "--boundary", "costly_training", "--note", "test"])
            base = _authoritative_response(run_dir)
            cases = [
                ("structures_path", lambda p: p["parameters"].__setitem__("dataset", str(root / "other.extxyz"))),
                ("output path", lambda p: p["parameters"].__setitem__("manifest_path", str(root / "other.json"))),
                ("action_type", lambda p: p.__setitem__("action_type", "collect_checkpoints")),
            ]
            for label, mutate in cases:
                proposal = json.loads(json.dumps(base))
                mutate(proposal)
                resp = root / f"tamper-{label.replace(' ', '-')}.json"
                resp.write_text(json.dumps(proposal))
                code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir),
                                 "--stage", "training", "--mock-response", str(resp),
                                 "--auto-mock-judges"])
                self.assertNotEqual(code, cli.EXIT_SUCCESS, label)
                self.assertEqual(RunController(run_dir).stage("training")["status"], "pending")
                self.assertFalse((run_dir / "artifacts" / "student_committee.manifest.json").exists())

    def test_protected_run_reference_binding_is_authoritative(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
            reference = _protected_reference_package(root, protected)
            run_dir = root / "run"
            RunController.initialize(_mock_training_workflow(root, protected_reference=reference), run_dir)
            cli.main(["approve", "--run-dir", str(run_dir), "--boundary", "costly_training", "--note", "test"])
            base = _authoritative_response(run_dir)
            bound_reference = Path(RunController(run_dir).state["inputs"][2]["snapshot"]).resolve()
            self.assertEqual(Path(base["parameters"]["reference_yaml"]).resolve(), bound_reference)
            for label, mutate in [
                ("omit", lambda p: p["parameters"].pop("reference_yaml")),
                ("replace", lambda p: p["parameters"].__setitem__("reference_yaml", str(root / "other-reference.yaml"))),
            ]:
                proposal = json.loads(json.dumps(base))
                mutate(proposal)
                resp = root / f"reference-{label}.json"
                resp.write_text(json.dumps(proposal))
                code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir),
                                 "--stage", "training", "--mock-response", str(resp),
                                 "--auto-mock-judges"])
                self.assertNotEqual(code, cli.EXIT_SUCCESS)
                self.assertEqual(RunController(run_dir).stage("training")["status"], "pending")

    def test_protected_run_blocks_protected_geometry_at_stage_runner(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
            reference = _protected_reference_package(root, protected)
            run_dir = root / "run"
            RunController.initialize(_mock_training_workflow(root, protected_reference=reference,
                                                             protected_dataset=True), run_dir)
            cli.main(["approve", "--run-dir", str(run_dir), "--boundary", "costly_training", "--note", "test"])
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--stage", "training", "--auto-mock-judges"])
            self.assertNotEqual(code, cli.EXIT_SUCCESS)
            self.assertFalse((run_dir / "artifacts" / "student_committee.manifest.json").exists())

    def test_reference_validation_stage_is_not_forced_to_training_protection(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
            reference = _protected_reference_package(root, protected)
            artifact = root / "reference_validation.json"
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({
                "run_id": "reference-validation",
                "inputs": [str(reference)],
                "stages": [{
                    "name": "reference_validation",
                    "command": None,
                    "outputs": ["artifacts/reference_validation.json"],
                    "pydantic_ai": {
                        "role": "data-curator",
                        "action": "build_dataset_manifest",
                        "parameters": {
                            "dataset": str(reference),
                            "manifest_path": "{artifacts_dir}/reference_validation.json",
                        },
                    },
                }],
            }))
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            proposal = _authoritative_response(run_dir, "reference_validation")
            self.assertNotIn("reference_yaml", proposal["parameters"])

    def test_judge_read_allowlists_are_mutually_blind(self):
        from runtimes.pydantic_ai.cli import judge_read_allowlist
        gate_context = {"artifact_sha256": {"/tmp/run/artifacts/a.json": "abc"}}
        evidence = Path("/tmp/run/exchange/bounded_evidence/stage.json")
        allow = judge_read_allowlist(gate_context, evidence)
        self.assertEqual(allow, [str(evidence.resolve()), "/tmp/run/artifacts/a.json"])
        self.assertFalse(any(path == "/tmp/run" or path.endswith("/gates") or "/provenance" in path for path in allow))


    def test_three_judge_gate_uses_three_blind_runtime_contexts(self):
        from orchestration.specs import load_agent_specs
        from runtimes.pydantic_ai.cli import run_three_judge_gate
        from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
        from runtimes.pydantic_ai.models import RuntimeContext
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({"run_id": "judge-blind", "stages": [{
                "name": "s", "command": None, "outputs": ["artifacts/a.json"],
                "gate": {"criteria": ["c"]}}]}))
            run_dir = root / "run"
            c = RunController.initialize(workflow, run_dir)
            artifact = run_dir / "artifacts" / "a.json"
            artifact.write_text('{"ok": true}\n')
            c.complete_external_stage("s", [artifact])
            evidence = run_dir / "exchange" / "bounded_evidence" / "s.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text('{"schema_version": 1}\n')
            specs = load_agent_specs(str(ROOT / "agent_specs"))
            contexts = []
            def runtime_factory(index):
                def responder(task, spec, toolset):
                    return json.dumps({"review_lens": task["context"]["review_lens"],
                                       "verdict": "PASS",
                                       "criteria_checked": [{"criterion": "c", "value_read": "ok", "ok": True}],
                                       "rationale": f"judge {index}", "required_fix": ""}), (0, 0)
                return MockAgentRuntime(responder)
            def ctx_factory(index):
                ctx = RuntimeContext(exchange_dir=str(run_dir / "exchange"), repo_root=str(ROOT),
                                     provider="mock", model_id="mock",
                                     read_allow_prefixes=[str(evidence.resolve()), str(artifact.resolve())])
                contexts.append(ctx)
                return ctx
            decision, _ = run_three_judge_gate(c, "s", specs, runtime_factory, ctx_factory, evidence)
            self.assertEqual(decision, "PASS")
            self.assertEqual(len(contexts), 4)  # one exchange setup context + three Judge contexts
            judge_contexts = contexts[1:]
            self.assertEqual(len(judge_contexts), 3)
            expected = [str(evidence.resolve()), str(artifact.resolve())]
            for ctx in judge_contexts:
                self.assertEqual(ctx.read_allow_prefixes, expected)
                self.assertFalse(any(path == str(run_dir) or path.endswith("/gates") or
                                     "/provenance" in path or "/results" in path
                                     for path in ctx.read_allow_prefixes))

    def test_teacher_baseline_requires_explicit_structures_and_does_not_pick_xyz(self):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({"run_id": "tb-no-guess", "inputs": [
                str(ROOT / "examples/mock/teacher.yaml"), str(ROOT / "examples/mock/validation.yaml"),
                str(ROOT / "examples/mock/distillation_scope.yaml"), str(ROOT / "examples/mock/seed.example.xyz")],
                "stages": [{"name": "teacher_baseline", "command": None,
                            "outputs": ["artifacts/teacher_baseline.json"],
                            "gate": {"criteria": ["teacher baseline report is valid"]}}]}))
            run_dir = root / "run"
            RunController.initialize(workflow, run_dir)
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir),
                             "--stage", "teacher_baseline", "--auto-mock-judges"])
            self.assertNotEqual(code, cli.EXIT_SUCCESS)
            self.assertFalse((run_dir / "artifacts" / "teacher_baseline.json").exists())


if __name__ == "__main__":
    unittest.main()
